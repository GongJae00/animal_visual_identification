"""Factorized anatomy and invalid-pixel segmentation for NoseID-v1."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class _ConvNormAct(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(input_dim, output_dim, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, output_dim),
            nn.GELU(),
        )


class FactorizedNoseSegmenter(nn.Module):
    """Fuse DINO blocks {2,5,8,11} with a 448-pixel RGB stem."""

    hidden_indices = (2, 5, 8, 11)

    def __init__(self) -> None:
        super().__init__()
        self.token_projections = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 96)) for _ in range(4)]
        )
        self.rgb_stem = nn.Sequential(
            _ConvNormAct(3, 32, stride=2),
            _ConvNormAct(32, 64, stride=2),
        )
        self.fusion = nn.Sequential(
            _ConvNormAct(96 + 64, 128),
            _ConvNormAct(128, 96),
            _ConvNormAct(96, 64),
        )
        self.semantic_head = nn.Conv2d(64, 3, kernel_size=1)
        self.invalid_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(
        self,
        aligned_rgb: torch.Tensor,
        hidden_states: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if aligned_rgb.ndim != 4 or aligned_rgb.shape[1:] != (3, 448, 448):
            raise ValueError("segmenter RGB must have shape [B,3,448,448]")
        projected = []
        for layer, projection in zip(self.hidden_indices, self.token_projections, strict=True):
            state = hidden_states[layer + 1]
            if state.shape[1:] != (577, 384):
                raise ValueError("DINO hidden state must have shape [B,577,384]")
            tokens = projection(state[:, 1:]).transpose(1, 2).reshape(-1, 96, 24, 24)
            projected.append(tokens)
        dino = torch.stack(projected).mean(dim=0)
        dino = F.interpolate(dino, size=(112, 112), mode="bilinear", align_corners=False)
        rgb = self.rgb_stem(aligned_rgb)
        features = self.fusion(torch.cat([dino, rgb], dim=1))
        semantic = F.interpolate(
            self.semantic_head(features), size=(448, 448), mode="bilinear", align_corners=False
        )
        invalid = F.interpolate(
            self.invalid_head(features), size=(448, 448), mode="bilinear", align_corners=False
        )
        return semantic, invalid


def factorized_segmentation_loss(
    semantic_logits: torch.Tensor,
    invalid_logit: torch.Tensor,
    semantic_target: torch.Tensor,
    invalid_target: torch.Tensor,
) -> torch.Tensor:
    if semantic_logits.ndim != 4 or semantic_logits.shape[1] != 3:
        raise ValueError("semantic logits must have shape [B,3,H,W]")
    if invalid_logit.shape != semantic_logits[:, :1].shape:
        raise ValueError("invalid logit shape differs")
    focal_ce = F.cross_entropy(semantic_logits, semantic_target, reduction="none")
    probability = torch.softmax(semantic_logits, dim=1)
    pt = probability.gather(1, semantic_target[:, None]).squeeze(1)
    focal_ce = ((1.0 - pt).square() * focal_ce).mean()
    semantic_one_hot = F.one_hot(semantic_target, num_classes=3).permute(0, 3, 1, 2).float()
    semantic_dice = 0.0
    for channel in (1, 2):
        intersection = torch.sum(probability[:, channel] * semantic_one_hot[:, channel])
        denominator = torch.sum(probability[:, channel] + semantic_one_hot[:, channel])
        semantic_dice = semantic_dice + (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))
    invalid_target = invalid_target.float()
    invalid_probability = torch.sigmoid(invalid_logit)
    bce = F.binary_cross_entropy_with_logits(invalid_logit, invalid_target, reduction="none")
    pt_invalid = torch.where(invalid_target > 0.5, invalid_probability, 1.0 - invalid_probability)
    focal_bce = ((1.0 - pt_invalid).square() * bce).mean()
    intersection = torch.sum(invalid_probability * invalid_target)
    invalid_dice = 1.0 - (2.0 * intersection + 1.0) / (
        torch.sum(invalid_probability + invalid_target) + 1.0
    )
    boundary = semantic_logits.sum() * 0.0
    for channel in (1, 2):
        predicted = probability[:, channel : channel + 1]
        target = semantic_one_hot[:, channel : channel + 1]
        predicted_edge = F.max_pool2d(predicted, 3, stride=1, padding=1) - (
            -F.max_pool2d(-predicted, 3, stride=1, padding=1)
        )
        target_edge = F.max_pool2d(target, 3, stride=1, padding=1) - (
            -F.max_pool2d(-target, 3, stride=1, padding=1)
        )
        boundary = boundary + F.l1_loss(predicted_edge, target_edge)
    return (
        focal_ce
        + semantic_dice
        + 0.75 * focal_bce
        + 0.5 * invalid_dice
        + 0.25 * boundary
    )


__all__ = ["FactorizedNoseSegmenter", "factorized_segmentation_loss"]
