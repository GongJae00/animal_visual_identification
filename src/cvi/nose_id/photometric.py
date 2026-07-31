"""Deterministic photometric preparation for NoseID-v1."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("RGB input must have shape [B,3,H,W]")
    if not torch.isfinite(rgb).all():
        raise ValueError("RGB input must be finite")
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("linear RGB input must have shape [B,3,H,W]")
    if not torch.isfinite(rgb).all() or torch.any((rgb < 0.0) | (rgb > 1.0)):
        raise ValueError("linear RGB input must be finite and in [0,1]")
    return torch.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * rgb.pow(1.0 / 2.4) - 0.055,
    )


def linear_rgb_luminance(linear_rgb: torch.Tensor) -> torch.Tensor:
    if linear_rgb.ndim != 4 or linear_rgb.shape[1] != 3:
        raise ValueError("linear RGB input must have shape [B,3,H,W]")
    if not torch.isfinite(linear_rgb).all():
        raise ValueError("linear RGB input must be finite")
    weights = linear_rgb.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    return torch.sum(linear_rgb * weights, dim=1, keepdim=True)


def linear_luminance(rgb: torch.Tensor) -> torch.Tensor:
    return linear_rgb_luminance(srgb_to_linear(rgb))


def glare_saturation_invalid_mask(
    rgb: torch.Tensor,
    source_valid_mask: torch.Tensor | None = None,
    *,
    glare_luminance: float = 0.90,
    clipped_channel: float = 0.995,
    dark_clip: float = 0.002,
) -> torch.Tensor:
    """Mark clipped or bright, low-chroma pixels whose texture is not observed."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("RGB input must have shape [B,3,H,W]")
    if not torch.isfinite(rgb).all() or torch.any((rgb < 0.0) | (rgb > 1.0)):
        raise ValueError("RGB input must be finite and in [0,1]")
    if not 0.0 <= dark_clip < glare_luminance < clipped_channel <= 1.0:
        raise ValueError("invalid glare or clipping thresholds")
    if source_valid_mask is not None and source_valid_mask.shape != rgb[:, :1].shape:
        raise ValueError("source valid mask must have shape [B,1,H,W]")
    channel_max = rgb.amax(dim=1, keepdim=True)
    channel_min = rgb.amin(dim=1, keepdim=True)
    luminance = linear_luminance(rgb)
    bright_neutral_glare = (luminance >= glare_luminance) & (
        (channel_max - channel_min) <= 0.12
    )
    clipped = (channel_max >= clipped_channel) | (channel_max <= dark_clip)
    invalid = bright_neutral_glare | clipped
    if source_valid_mask is not None:
        if not torch.isfinite(source_valid_mask).all():
            raise ValueError("source valid mask must be finite")
        invalid |= source_valid_mask <= 0.5
    return invalid


def masked_percentile_normalize(
    luminance: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    low: float = 0.02,
    high: float = 0.98,
) -> torch.Tensor:
    if luminance.ndim != 4 or luminance.shape[1] != 1:
        raise ValueError("luminance must have shape [B,1,H,W]")
    if valid_mask.shape != luminance.shape:
        raise ValueError("valid mask must match luminance")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 1")
    if not torch.isfinite(luminance).all() or not torch.isfinite(valid_mask).all():
        raise ValueError("normalization inputs must be finite")
    outputs: list[torch.Tensor] = []
    with torch.autocast(device_type=luminance.device.type, enabled=False):
        values = luminance.float()
        mask = valid_mask.float() > 0.5
        for index in range(values.shape[0]):
            selected = values[index][mask[index]]
            if selected.numel() < 2:
                outputs.append(torch.zeros_like(values[index]))
                continue
            p2 = torch.quantile(selected, low)
            p98 = torch.quantile(selected, high)
            outputs.append(((values[index] - p2) / (p98 - p2 + 1e-6)).clamp(0, 1))
    return torch.stack(outputs).to(dtype=luminance.dtype)


def _masked_separable_blur(
    values: torch.Tensor, valid_mask: torch.Tensor, kernel: torch.Tensor
) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    padding_mode = (
        "reflect"
        if values.shape[-2] > padding and values.shape[-1] > padding
        else "replicate"
    )

    def blur(item: torch.Tensor) -> torch.Tensor:
        horizontal = F.conv2d(
            F.pad(item, (padding, padding, 0, 0), mode=padding_mode), kernel
        )
        return F.conv2d(
            F.pad(horizontal, (0, 0, padding, padding), mode=padding_mode),
            kernel.transpose(-1, -2),
        )

    numerator = blur(values * valid_mask)
    denominator = blur(valid_mask)
    return numerator / denominator.clamp_min(1e-6)


def masked_illumination_normalize(
    rgb: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    low: float = 0.02,
    high: float = 0.98,
    kernel_size: int = 31,
    sigma: float = 9.0,
    max_gain: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return linear RGB normalized only from observed pixels and its illumination map."""
    if valid_mask.shape != rgb[:, :1].shape:
        raise ValueError("valid mask must have shape [B,1,H,W]")
    if kernel_size < 3 or kernel_size % 2 != 1 or sigma <= 0.0:
        raise ValueError("illumination kernel must be odd and sigma positive")
    if max_gain < 1.0:
        raise ValueError("max_gain must be at least one")
    linear = srgb_to_linear(rgb).float()
    mask = (valid_mask > 0.5).to(dtype=linear.dtype)
    luminance = linear_rgb_luminance(linear)
    coordinate = torch.arange(kernel_size, device=rgb.device, dtype=torch.float32)
    coordinate -= kernel_size // 2
    kernel = torch.exp(-(coordinate**2) / (2.0 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, 1, kernel_size)
    illumination = _masked_separable_blur(luminance, mask, kernel)
    corrected: list[torch.Tensor] = []
    for index in range(linear.shape[0]):
        selected = illumination[index][mask[index] > 0.5]
        anchor = selected.median() if selected.numel() else illumination.new_tensor(1.0)
        gain = (anchor / illumination[index : index + 1].clamp_min(1e-6)).clamp(
            1.0 / max_gain, max_gain
        )
        corrected.append(linear[index : index + 1] * gain)
    corrected_rgb = torch.cat(corrected, dim=0).clamp(0.0, 1.0)
    corrected_luminance = linear_rgb_luminance(corrected_rgb)
    target_luminance = masked_percentile_normalize(
        corrected_luminance, mask, low=low, high=high
    )
    global_gain = (
        target_luminance / corrected_luminance.clamp_min(1e-6)
    ).clamp(1.0 / max_gain, max_gain)
    normalized = (corrected_rgb * global_gain).clamp(0.0, 1.0) * mask
    return normalized.to(dtype=rgb.dtype), illumination.to(dtype=rgb.dtype)


class TexturePhotometricNormalizer(nn.Module):
    def __init__(self, kernel_size: int = 73, sigma: float = 35.84) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("homomorphic kernel size must be odd")
        coordinate = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel = torch.exp(-(coordinate**2) / (2.0 * sigma**2))
        kernel /= kernel.sum()
        self.register_buffer("homomorphic_kernel", kernel.view(1, 1, 1, kernel_size))

    def forward(
        self, rgb: torch.Tensor, valid_texture_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        luminance = linear_luminance(rgb)
        normalized = masked_percentile_normalize(luminance, valid_texture_mask)
        log_luma = torch.log(normalized.float() + 1e-6)
        mask = (valid_texture_mask > 0.5).to(dtype=log_luma.dtype)
        illumination = _masked_separable_blur(
            log_luma, mask, self.homomorphic_kernel
        )
        high_pass = (log_luma - illumination) * mask
        return normalized * mask, high_pass.to(dtype=rgb.dtype)


__all__ = [
    "TexturePhotometricNormalizer",
    "glare_saturation_invalid_mask",
    "linear_rgb_luminance",
    "linear_luminance",
    "linear_to_srgb",
    "masked_illumination_normalize",
    "masked_percentile_normalize",
    "srgb_to_linear",
]
