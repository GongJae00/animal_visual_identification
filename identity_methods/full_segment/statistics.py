"""Index-only feature-channel and output-dimension statistics hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import torch
from torch import nn

from identity_methods.full_segment.model import MaskedGAP128


@dataclass(slots=True)
class _RunningDimensions:
    count: int = 0
    total: torch.Tensor | None = None
    total_square: torch.Tensor | None = None
    minimum: torch.Tensor | None = None
    maximum: torch.Tensor | None = None

    def update(self, value: torch.Tensor, *, dimension_axis: int) -> None:
        detached = value.detach().float().cpu()
        moved = detached.movedim(dimension_axis, -1).reshape(-1, detached.shape[dimension_axis])
        if moved.numel() == 0 or not torch.isfinite(moved).all():
            raise RuntimeError("statistics hook observed empty or non-finite values")
        batch_total = moved.sum(dim=0, dtype=torch.float64)
        batch_square = moved.square().sum(dim=0, dtype=torch.float64)
        batch_minimum = moved.min(dim=0).values
        batch_maximum = moved.max(dim=0).values
        if self.total is None:
            self.total = batch_total
            self.total_square = batch_square
            self.minimum = batch_minimum
            self.maximum = batch_maximum
        else:
            self.total += batch_total
            self.total_square += batch_square
            self.minimum = torch.minimum(self.minimum, batch_minimum)
            self.maximum = torch.maximum(self.maximum, batch_maximum)
        self.count += moved.shape[0]

    def snapshot(self, *, axis_kind: str) -> dict[str, Any]:
        if self.count == 0 or self.total is None:
            raise RuntimeError("statistics hooks have not observed a successful forward pass")
        mean = self.total / self.count
        variance = (self.total_square / self.count - mean.square()).clamp_min(0.0)
        return {
            "axis_kind": axis_kind,
            "axis_interpretation": "INDEX_ONLY_NO_SEMANTIC_DIMENSION_CLAIM",
            "observation_count_per_index": self.count,
            "mean": mean.tolist(),
            "standard_deviation": variance.sqrt().tolist(),
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
        }


class FeatureOutputStatisticsHooks:
    """Collect descriptive channel/index statistics through removable hooks."""

    def __init__(self, model: MaskedGAP128) -> None:
        if not isinstance(model, MaskedGAP128):
            raise TypeError("statistics hooks require MaskedGAP128")
        self._feature = _RunningDimensions()
        self._output = _RunningDimensions()
        self._handles = (
            model.features.register_forward_hook(self._capture_features),
            model.register_forward_hook(self._capture_output),
        )

    def _capture_features(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        del module, inputs
        self._feature.update(output, dimension_axis=1)

    def _capture_output(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        del module, inputs
        self._output.update(output, dimension_axis=1)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "cvi.full_segment_dimension_statistics.v1",
            "feature_channels": self._feature.snapshot(axis_kind="FEATURE_CHANNEL_INDEX"),
            "output_dimensions": self._output.snapshot(axis_kind="OUTPUT_DIMENSION_INDEX"),
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = ()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = ["FeatureOutputStatisticsHooks"]
