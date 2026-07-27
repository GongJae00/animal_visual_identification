"""Fixed 11-channel NoseID-v1 frequency decomposition."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from cvi.nose_id.photometric import TexturePhotometricNormalizer


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


__all__ = ["FixedFrequencyBank"]
