"""Deterministic optimization admission rules and first-order cost models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class PromotionDecision(StrEnum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class ProtectedMetric:
    """Signed degradation where positive values are worse."""

    name: str
    degradation_estimate: float
    degradation_lcb: float
    degradation_ucb: float
    noninferiority_margin: float

    def __post_init__(self) -> None:
        _validate_metric_name(self.name)
        _validate_finite(
            self.degradation_estimate,
            self.degradation_lcb,
            self.degradation_ucb,
            self.noninferiority_margin,
        )
        if not (
            self.degradation_lcb
            <= self.degradation_estimate
            <= self.degradation_ucb
        ):
            raise ValueError("protected metric estimate must lie inside its interval")
        if self.noninferiority_margin < 0:
            raise ValueError("noninferiority_margin must be non-negative")


@dataclass(frozen=True, slots=True)
class ImprovementMetric:
    """Signed degradation for a resource/utility metric.

    A negative upper confidence bound proves strict improvement.
    """

    name: str
    degradation_estimate: float
    degradation_ucb: float

    def __post_init__(self) -> None:
        _validate_metric_name(self.name)
        _validate_finite(self.degradation_estimate, self.degradation_ucb)
        if self.degradation_estimate > self.degradation_ucb:
            raise ValueError("improvement estimate cannot exceed its upper bound")


def _validate_metric_name(name: str) -> None:
    if not name or not name.strip():
        raise ValueError("metric name must be non-empty")


def _validate_finite(*values: float) -> None:
    if not all(isfinite(value) for value in values):
        raise ValueError("metric values must be finite")


def evaluate_promotion(
    protected: tuple[ProtectedMetric, ...],
    improvements: tuple[ImprovementMetric, ...],
) -> PromotionDecision:
    """Apply the predeclared non-inferiority and Pareto promotion rule."""

    if not protected or not improvements:
        return PromotionDecision.INCONCLUSIVE
    if any(
        metric.degradation_lcb > metric.noninferiority_margin
        for metric in protected
    ):
        return PromotionDecision.REJECT
    if any(
        metric.degradation_ucb > metric.noninferiority_margin
        for metric in protected
    ):
        return PromotionDecision.INCONCLUSIVE
    if any(metric.degradation_ucb < 0.0 for metric in improvements):
        return PromotionDecision.PROMOTE
    return PromotionDecision.INCONCLUSIVE


def compute_cost(
    *,
    decode_cost: float,
    detection_calls: int,
    detection_cost: float,
    tracking_calls: int,
    tracking_cost: float,
    quality_calls: int,
    quality_cost: float,
    embedding_calls: int,
    embedding_cost: float,
    search_calls: int,
    search_cost: float,
    aggregation_cost: float,
) -> float:
    """Evaluate the first-order end-to-end compute model."""

    costs = (
        decode_cost,
        detection_cost,
        tracking_cost,
        quality_cost,
        embedding_cost,
        search_cost,
        aggregation_cost,
    )
    calls = (
        detection_calls,
        tracking_calls,
        quality_calls,
        embedding_calls,
        search_calls,
    )
    _validate_nonnegative_finite(*costs)
    _validate_nonnegative_int(*calls)
    return (
        decode_cost
        + detection_calls * detection_cost
        + tracking_calls * tracking_cost
        + quality_calls * quality_cost
        + embedding_calls * embedding_cost
        + search_calls * search_cost
        + aggregation_cost
    )


def gallery_bytes(
    identities: int,
    prototypes_per_identity: int,
    embedding_dimension: int,
    bytes_per_value: int,
) -> int:
    """Return raw gallery bytes, excluding index and allocator overhead."""

    _validate_nonnegative_int(
        identities,
        prototypes_per_identity,
        embedding_dimension,
        bytes_per_value,
    )
    return (
        identities
        * prototypes_per_identity
        * embedding_dimension
        * bytes_per_value
    )


def encoded_video_bytes(bitrate_mbps: float, duration_seconds: float) -> float:
    """Return nominal encoded bytes for decimal Mbit/s."""

    _validate_nonnegative_finite(bitrate_mbps, duration_seconds)
    return 1_000_000 * bitrate_mbps * duration_seconds / 8


def _validate_nonnegative_finite(*values: float) -> None:
    _validate_finite(*values)
    if any(value < 0 for value in values):
        raise ValueError("cost values must be non-negative")


def _validate_nonnegative_int(*values: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("counts and dimensions must be integers")
    if any(value < 0 for value in values):
        raise ValueError("counts and dimensions must be non-negative")
