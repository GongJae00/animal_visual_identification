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


def linear_luminance(rgb: torch.Tensor) -> torch.Tensor:
    linear = srgb_to_linear(rgb)
    weights = linear.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    return torch.sum(linear * weights, dim=1, keepdim=True)


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
        padding = self.homomorphic_kernel.shape[-1] // 2
        illumination = F.conv2d(
            F.pad(log_luma, (padding, padding, 0, 0), mode="reflect"),
            self.homomorphic_kernel,
        )
        illumination = F.conv2d(
            F.pad(illumination, (0, 0, padding, padding), mode="reflect"),
            self.homomorphic_kernel.transpose(-1, -2),
        )
        return normalized, (log_luma - illumination).to(dtype=rgb.dtype)


__all__ = [
    "TexturePhotometricNormalizer",
    "linear_luminance",
    "masked_percentile_normalize",
    "srgb_to_linear",
]
