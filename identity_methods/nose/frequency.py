"""Fixed 11-channel NoseID-v1 frequency decomposition."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from identity_methods.nose.photometric import TexturePhotometricNormalizer


def _descriptor_inputs(
    luminance: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(luminance, dtype=np.float32)
    mask = np.asarray(valid_mask) > 0.5
    if image.ndim != 2 or mask.shape != image.shape or not np.isfinite(image).all():
        raise ValueError("descriptor luminance and mask must be finite same-shaped 2D arrays")
    return image, mask


def gabor_descriptor(
    luminance: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Return fixed, masked Gabor response summaries without learned parameters."""
    image, mask = _descriptor_inputs(luminance, valid_mask)
    supported = cv2.erode(
        mask.astype(np.uint8), np.ones((9, 9), dtype=np.uint8)
    ).astype(bool)
    values: list[float] = []
    for wavelength in (4.0, 8.0):
        for orientation in (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0):
            kernel = cv2.getGaborKernel(
                (9, 9), 2.0, orientation, wavelength, 0.5, 0.0, ktype=cv2.CV_32F
            )
            kernel -= kernel.mean()
            response = cv2.filter2D(image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
            selected = np.abs(response[supported])
            values.append(float(np.median(selected)) if selected.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def lbp_descriptor(luminance: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Return a normalized 8-neighbour LBP histogram over observed interior pixels."""
    image, mask = _descriptor_inputs(luminance, valid_mask)
    if min(image.shape) < 3:
        return np.zeros(256, dtype=np.float32)
    center = image[1:-1, 1:-1]
    code = np.zeros(center.shape, dtype=np.uint8)
    neighbors = (
        image[:-2, :-2], image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:],
        image[2:, 2:], image[2:, 1:-1], image[2:, :-2], image[1:-1, :-2],
    )
    for bit, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    supported = cv2.erode(
        mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    interior_mask = supported[1:-1, 1:-1]
    histogram = np.bincount(code[interior_mask], minlength=256).astype(np.float32)
    total = float(histogram.sum())
    if total > 0.0:
        histogram /= total
    return histogram


def radial_frequency_descriptor(
    luminance: np.ndarray, valid_mask: np.ndarray, *, bins: int = 8
) -> np.ndarray:
    """Summarize windowed radial Fourier power without interpreting missing pixels."""
    image, mask = _descriptor_inputs(luminance, valid_mask)
    if bins < 1:
        raise ValueError("frequency bins must be positive")
    selected = image[mask]
    if selected.size < 2:
        return np.zeros(bins, dtype=np.float32)
    centered = np.where(mask, image - float(np.median(selected)), 0.0)
    window_y = np.hanning(image.shape[0]).astype(np.float32)
    window_x = np.hanning(image.shape[1]).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered * np.outer(window_y, window_x)))) ** 2
    y, x = np.indices(image.shape)
    radius = np.hypot(y - (image.shape[0] - 1) / 2.0, x - (image.shape[1] - 1) / 2.0)
    edges = np.linspace(0.0, float(radius.max()) + 1e-6, bins + 1)
    output = np.zeros(bins, dtype=np.float64)
    for index in range(bins):
        annulus = (radius >= edges[index]) & (radius < edges[index + 1])
        if np.any(annulus):
            output[index] = np.log1p(np.median(power[annulus]))
    norm = float(np.linalg.norm(output))
    if norm > 0.0:
        output /= norm
    return output.astype(np.float32)


def classical_texture_descriptors(
    luminance: np.ndarray, valid_mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "gabor": gabor_descriptor(luminance, valid_mask),
        "lbp": lbp_descriptor(luminance, valid_mask),
        "radial_frequency": radial_frequency_descriptor(luminance, valid_mask),
    }


def _gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3.0 * sigma))
    coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(coordinate**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _log_kernel(sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3.0 * sigma))
    y, x = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        indexing="ij",
    )
    radius2 = x.square() + y.square()
    kernel = ((radius2 - 2.0 * sigma**2) / sigma**4) * torch.exp(
        -radius2 / (2.0 * sigma**2)
    )
    kernel -= kernel.mean()
    kernel *= sigma**2
    return kernel


def _filter(value: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    return F.conv2d(F.pad(value, (padding,) * 4, mode="reflect"), kernel)


def _filter_separable(value: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    result = F.conv2d(
        F.pad(value, (padding, padding, 0, 0), mode="reflect"), kernel
    )
    return F.conv2d(
        F.pad(result, (0, 0, padding, padding), mode="reflect"),
        kernel.transpose(-1, -2),
    )


def _robust_standardize(
    channels: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    output = torch.zeros_like(channels, dtype=torch.float32)
    values = channels.float()
    mask = valid_mask.float() > 0.5
    for batch_index in range(values.shape[0]):
        selected_mask = mask[batch_index, 0]
        if int(selected_mask.sum()) < 2:
            continue
        for channel_index in range(values.shape[1]):
            selected = values[batch_index, channel_index][selected_mask]
            median = selected.median()
            mad = (selected - median).abs().median()
            output[batch_index, channel_index] = (
                (values[batch_index, channel_index] - median)
                / (1.4826 * mad + 1e-6)
            ).clamp(-5.0, 5.0)
    return output.to(dtype=channels.dtype)


class FixedFrequencyBank(nn.Module):
    output_channels = 11

    def __init__(self) -> None:
        super().__init__()
        for name, sigma in (
            ("g08", 0.8),
            ("g16", 1.6),
            ("g32", 3.2),
            ("g64", 6.4),
        ):
            kernel = _gaussian_kernel(sigma)
            self.register_buffer(name, kernel.view(1, 1, 1, -1))
        self.register_buffer("log10", _log_kernel(1.0).view(1, 1, 7, 7))
        self.register_buffer("log20", _log_kernel(2.0).view(1, 1, 13, 13))
        self.register_buffer(
            "sobel_x",
            torch.tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)), dtype=torch.float32).view(1, 1, 3, 3) / 8.0,
        )
        self.register_buffer("sobel_y", self.sobel_x.transpose(-1, -2).contiguous())
        self.photometric = TexturePhotometricNormalizer()

    def forward(self, rgb: torch.Tensor, valid_texture_mask: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.shape[-2:] != (448, 448):
            raise ValueError("frequency RGB must have shape [B,3,448,448]")
        if valid_texture_mask.shape != (rgb.shape[0], 1, 448, 448):
            raise ValueError("frequency mask must have shape [B,1,448,448]")
        if not torch.isfinite(rgb).all() or not torch.isfinite(valid_texture_mask).all():
            raise ValueError("frequency inputs must be finite")
        normalized, homomorphic = self.photometric(rgb, valid_texture_mask)
        gaussian08 = _filter_separable(normalized, self.g08)
        gaussian16 = _filter_separable(normalized, self.g16)
        gaussian32 = _filter_separable(normalized, self.g32)
        gaussian64 = _filter_separable(normalized, self.g64)
        dog1 = gaussian08 - gaussian16
        dog2 = gaussian16 - gaussian32
        dog3 = gaussian32 - gaussian64
        log1 = _filter(normalized, self.log10)
        log2 = _filter(normalized, self.log20)
        sobel_x = _filter(normalized, self.sobel_x)
        sobel_y = _filter(normalized, self.sobel_y)
        magnitude = torch.sqrt(sobel_x.square() + sobel_y.square() + 1e-12)
        filtered = torch.cat(
            [normalized, homomorphic, dog1, dog2, dog3, log1, log2, sobel_x, sobel_y, magnitude],
            dim=1,
        )
        standardized = _robust_standardize(filtered, valid_texture_mask)
        standardized = standardized * valid_texture_mask.to(dtype=standardized.dtype)
        result = torch.cat([standardized, valid_texture_mask.to(dtype=rgb.dtype)], dim=1)
        if result.shape != (rgb.shape[0], 11, 448, 448) or not torch.isfinite(result).all():
            raise RuntimeError("fixed frequency bank produced invalid output")
        return result


__all__ = [
    "FixedFrequencyBank",
    "classical_texture_descriptors",
    "gabor_descriptor",
    "lbp_descriptor",
    "radial_frequency_descriptor",
]
