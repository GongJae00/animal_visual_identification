"""Mask-aware anatomical pooling for NoseID-v1 streams."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


_REGION_CENTERS = (
    (0.50, 0.50, 0.45, 0.38),
    (0.32, 0.51, 0.20, 0.28),
    (0.68, 0.51, 0.20, 0.28),
    (0.50, 0.31, 0.24, 0.20),
    (0.50, 0.69, 0.24, 0.20),
)


def anatomical_region_masks(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    regions = []
    for center_x, center_y, sigma_x, sigma_y in _REGION_CENTERS:
        regions.append(
            torch.exp(
                -0.5
                * (
                    ((x - center_x) / sigma_x).square()
                    + ((y - center_y) / sigma_y).square()
                )
            )
        )
    return torch.stack(regions, dim=0)


class AnatomicalPartPool(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.attention = nn.Conv2d(input_dim, 1, kernel_size=1)
        self.gem_power = nn.Parameter(torch.tensor(3.0))
        self.projection = nn.Sequential(
            nn.Linear(input_dim * 10, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, output_dim),
        )

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or valid_mask.shape != (
            features.shape[0], 1, features.shape[2], features.shape[3]
        ):
            raise ValueError("pooling features/mask shapes differ")
        mask = valid_mask.clamp(0.0, 1.0)
        regions = anatomical_region_masks(
            features.shape[2], features.shape[3], device=features.device, dtype=features.dtype
        )[None]
        attention_logits = self.attention(features)
        pooled: list[torch.Tensor] = []
        power = self.gem_power.clamp(1.0, 6.0)
        for region_index in range(5):
            weights = mask * regions[:, region_index : region_index + 1]
            coverage = weights.mean(dim=(2, 3), keepdim=True)
            usable = coverage >= 0.03
            logits = attention_logits.masked_fill(weights <= 1e-6, -1e4)
            attention = torch.softmax(logits.flatten(2), dim=-1).view_as(logits) * weights
            attention = attention / attention.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            attention_pool = torch.sum(features * attention, dim=(2, 3))
            gem = (
                torch.sum(features.clamp_min(1e-6).pow(power) * weights, dim=(2, 3))
                / weights.sum(dim=(2, 3)).clamp_min(1e-6)
            ).pow(1.0 / power)
            pooled.extend(
                [
                    torch.where(usable.flatten(1), attention_pool, torch.zeros_like(attention_pool)),
                    torch.where(usable.flatten(1), gem, torch.zeros_like(gem)),
                ]
            )
        return F.normalize(self.projection(torch.cat(pooled, dim=1)), dim=1)


__all__ = ["AnatomicalPartPool", "anatomical_region_masks"]
