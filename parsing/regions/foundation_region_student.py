"""Task-specific decoders for frozen high-resolution foundation features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from parsing.regions.region_teacher_consensus import REGION_CLASS_MAPS, ConsensusState


@dataclass(frozen=True, slots=True)
class RegionDecoderConfig:
    region: str
    input_dimension: int
    hidden_dimension: int = 256
    depth: int = 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.region not in REGION_CLASS_MAPS:
            raise ValueError("region decoder target must be A, F, or N")
        for name in ("input_dimension", "hidden_dimension", "depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"region decoder {name} must be positive")
        if not isinstance(self.dropout, float) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("region decoder dropout must be a float in [0,1)")

    @property
    def class_count(self) -> int:
        return len(REGION_CLASS_MAPS[self.region])

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "input_dimension": self.input_dimension,
            "hidden_dimension": self.hidden_dimension,
            "depth": self.depth,
            "dropout": self.dropout,
            "class_map": dict(REGION_CLASS_MAPS[self.region]),
            "input_contract": "FROZEN_FOUNDATION_DENSE_FEATURES_BHWC",
            "output_contract": "SOURCE_OR_CROP_ALIGNED_CLASS_LOGITS",
        }


def build_region_decoder(config: RegionDecoderConfig):
    """Build one decoder; A, F, and N never share classification heads."""

    import torch

    class ResidualBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.depthwise = torch.nn.Conv2d(
                config.hidden_dimension,
                config.hidden_dimension,
                kernel_size=3,
                padding=1,
                groups=config.hidden_dimension,
            )
            self.norm = torch.nn.GroupNorm(1, config.hidden_dimension)
            self.expand = torch.nn.Conv2d(
                config.hidden_dimension, config.hidden_dimension * 4, kernel_size=1
            )
            self.contract = torch.nn.Conv2d(
                config.hidden_dimension * 4, config.hidden_dimension, kernel_size=1
            )
            self.dropout = torch.nn.Dropout2d(config.dropout)

        def forward(self, values):
            residual = values
            values = self.depthwise(values)
            values = self.norm(values)
            values = self.expand(values)
            values = torch.nn.functional.gelu(values)
            values = self.dropout(values)
            return residual + self.contract(values)

    class RegionDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.projection = torch.nn.Conv2d(
                config.input_dimension, config.hidden_dimension, kernel_size=1
            )
            self.blocks = torch.nn.Sequential(
                *(ResidualBlock() for _ in range(config.depth))
            )
            self.classifier = torch.nn.Conv2d(
                config.hidden_dimension, config.class_count, kernel_size=1
            )

        def forward(self, features, *, output_size: tuple[int, int]):
            if features.ndim != 4 or features.shape[-1] != config.input_dimension:
                raise ValueError("region decoder features must match [B,H,W,C]")
            if (
                len(output_size) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in output_size
                )
            ):
                raise ValueError("region decoder output size must be positive")
            values = features.permute(0, 3, 1, 2)
            values = self.classifier(self.blocks(self.projection(values)))
            return torch.nn.functional.interpolate(
                values, size=output_size, mode="bilinear", align_corners=False
            )

    return RegionDecoder()


def region_distillation_loss(
    logits,
    *,
    state: ConsensusState,
    soft_probabilities,
    hard_mask,
    uncertainty,
    validity,
):
    """Use hard targets only after a hard consensus decision."""

    import torch

    if state is ConsensusState.ABSTAIN:
        raise ValueError("abstained region consensus cannot supervise a student")
    if logits.ndim != 4:
        raise ValueError("region student logits must be [B,C,H,W]")
    expected = logits.shape
    if soft_probabilities.shape != expected:
        raise ValueError("region soft targets must match logits")
    if uncertainty.shape != expected[:1] + expected[2:] or validity.shape != uncertainty.shape:
        raise ValueError("region uncertainty and validity shapes differ")
    if not torch.isfinite(logits).all() or not torch.isfinite(soft_probabilities).all():
        raise ValueError("region student tensors must be finite")
    if not validity.any():
        raise ValueError("region student supervision contains no valid pixels")
    valid = validity.to(dtype=logits.dtype)
    confidence = (1.0 - uncertainty.to(dtype=logits.dtype)).clamp(0.0, 1.0) * valid
    denominator = confidence.sum().clamp_min(1.0)
    log_probabilities = torch.nn.functional.log_softmax(logits, dim=1)
    soft = -(soft_probabilities.to(dtype=logits.dtype) * log_probabilities).sum(dim=1)
    soft_loss = (soft * confidence).sum() / denominator
    if state is ConsensusState.SOFT_CANDIDATE:
        return soft_loss
    if hard_mask.shape != expected[:1] + expected[2:]:
        raise ValueError("region hard target shape differs")
    hard = torch.nn.functional.cross_entropy(logits, hard_mask.long(), reduction="none")
    hard_loss = (hard * valid).sum() / valid.sum().clamp_min(1.0)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(
        hard_mask.long(), num_classes=expected[1]
    ).permute(0, 3, 1, 2)
    valid_channels = valid[:, None]
    intersection = (probabilities * one_hot * valid_channels).sum(dim=(0, 2, 3))
    union = ((probabilities + one_hot) * valid_channels).sum(dim=(0, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
    return 0.50 * soft_loss + 0.35 * hard_loss + 0.15 * dice_loss


__all__ = [
    "RegionDecoderConfig",
    "build_region_decoder",
    "region_distillation_loss",
]
