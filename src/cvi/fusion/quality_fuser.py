"""Low-capacity calibrated late score fusion for NoseID-v1 experiments."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PositiveAffineCalibrator(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        initial = math.log(math.expm1(1.0))
        self.raw_scale = nn.Parameter(torch.full((channels,), initial))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim < 1 or scores.shape[-1] != len(self.bias):
            raise ValueError("calibration score channel count differs")
        if not torch.isfinite(scores).all():
            raise ValueError("calibration scores must be finite")
        return F.softplus(self.raw_scale) * scores + self.bias


class QualityFusionMLP(nn.Module):
    """Map the fixed 27D query descriptor to appearance/face/nose weights."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(27, 32)
        self.output = nn.Linear(32, 3)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(torch.log(torch.tensor((0.55, 0.15, 0.30))))

    def forward(self, query_features: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        if query_features.ndim != 2 or query_features.shape[1] != 27:
            raise ValueError("fusion query features must have shape [B,27]")
        if availability.shape != (query_features.shape[0], 3) or availability.dtype != torch.bool:
            raise ValueError("fusion availability must be bool [B,3]")
        if not torch.isfinite(query_features).all() or not availability.any(dim=1).all():
            raise ValueError("fusion features must be finite with one available channel")
        logits = self.output(F.gelu(self.hidden(query_features)))
        return torch.softmax(logits.masked_fill(~availability, float("-inf")), dim=1)


def fuse_channel_scores(
    calibrated_scores: torch.Tensor,
    channel_weights: torch.Tensor,
    availability: torch.Tensor,
) -> torch.Tensor:
    if calibrated_scores.shape != channel_weights.shape or availability.shape != channel_weights.shape:
        raise ValueError("fusion score, weight, and availability shapes must match")
    masked_weights = channel_weights * availability.to(channel_weights.dtype)
    masked_weights = masked_weights / masked_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.sum(calibrated_scores * masked_weights, dim=-1)


__all__ = ["PositiveAffineCalibrator", "QualityFusionMLP", "fuse_channel_scores"]
