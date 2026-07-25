"""Portable timing summaries and immutable benchmark receipt structure."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import ceil, isfinite
from statistics import fmean
from time import perf_counter_ns
from typing import Any, TypeVar

from cvi.optimization import PromotionDecision
from cvi.provenance import content_sha256

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TimingSummary:
    samples: int
    minimum_ns: int
    p50_ns: int
    p95_ns: int
    maximum_ns: int
    mean_ns: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.samples, bool)
            or not isinstance(self.samples, int)
            or self.samples <= 0
        ):
            raise ValueError("timing sample count must be positive")
        for name in ("minimum_ns", "p50_ns", "p95_ns", "maximum_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not (
            self.minimum_ns
            <= self.p50_ns
            <= self.p95_ns
            <= self.maximum_ns
        ):
            raise ValueError("timing quantiles must be monotonic")
        if (
            isinstance(self.mean_ns, bool)
            or not isinstance(self.mean_ns, (int, float))
            or not isfinite(self.mean_ns)
            or not self.minimum_ns <= self.mean_ns <= self.maximum_ns
        ):
            raise ValueError("timing mean must lie inside the sample range")

    @classmethod
    def from_samples(cls, samples_ns: tuple[int, ...]) -> TimingSummary:
        if not samples_ns:
            raise ValueError("at least one timing sample is required")
        if any(
            isinstance(sample, bool) or not isinstance(sample, int)
            for sample in samples_ns
        ):
            raise TypeError("timing samples must be integer nanoseconds")
        if any(sample < 0 for sample in samples_ns):
            raise ValueError("timing samples must be non-negative")
        ordered = tuple(sorted(samples_ns))
        return cls(
            samples=len(ordered),
            minimum_ns=ordered[0],
            p50_ns=_nearest_rank(ordered, 0.50),
            p95_ns=_nearest_rank(ordered, 0.95),
            maximum_ns=ordered[-1],
            mean_ns=fmean(ordered),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "samples": self.samples,
            "minimum_ns": self.minimum_ns,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "maximum_ns": self.maximum_ns,
            "mean_ns": self.mean_ns,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TimingSummary:
        expected = {
            "samples",
            "minimum_ns",
            "p50_ns",
            "p95_ns",
            "maximum_ns",
            "mean_ns",
        }
        if set(payload) != expected:
            raise ValueError("timing summary keys mismatch")
        return cls(**payload)


def _nearest_rank(ordered: tuple[int, ...], quantile: float) -> int:
    index = max(0, ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def measure_operation(
    operation: Callable[[], T],
    *,
    warmup: int,
    repeats: int,
    synchronize: Callable[[], None] | None = None,
) -> tuple[TimingSummary, T]:
    """Measure wall time with an optional device synchronization boundary."""

    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    sync = synchronize or (lambda: None)
    result: T
    for _ in range(warmup):
        result = operation()
    sync()

    samples: list[int] = []
    for _ in range(repeats):
        sync()
        start = perf_counter_ns()
        result = operation()
        sync()
        samples.append(perf_counter_ns() - start)
    return TimingSummary.from_samples(tuple(samples)), result


@dataclass(frozen=True, slots=True)
class MetricInterval:
    name: str
    estimate: float
    lower: float
    upper: float
    unit: str
    method: str

    def __post_init__(self) -> None:
        for field_name in ("name", "unit", "method"):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if not all(isfinite(value) for value in (self.estimate, self.lower, self.upper)):
            raise ValueError("metric interval values must be finite")
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("metric estimate must lie inside [lower, upper]")

    def to_dict(self) -> dict[str, str | float]:
        return {
            "name": self.name,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "unit": self.unit,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReceipt:
    """Minimum reproducibility receipt for an optimization comparison."""

    reference_config_sha256: str
    candidate_config_sha256: str
    code_revision: str
    dependency_lock_sha256: str
    dataset_manifest_sha256: str
    split_sha256: str
    calibration_role: str
    test_role: str
    hardware: Mapping[str, str | int | float]
    workload: Mapping[str, str | int | float]
    warmup_iterations: int
    repeat_iterations: int
    timing: TimingSummary
    safety_metrics: tuple[MetricInterval, ...]
    resource_metrics: tuple[MetricInterval, ...]
    decision: PromotionDecision
    schema_version: str = "cvi.benchmark.v1"

    def __post_init__(self) -> None:
        for field_name in (
            "reference_config_sha256",
            "candidate_config_sha256",
            "code_revision",
            "dependency_lock_sha256",
            "dataset_manifest_sha256",
            "split_sha256",
            "calibration_role",
            "test_role",
        ):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.warmup_iterations < 0 or self.repeat_iterations <= 0:
            raise ValueError("invalid benchmark iteration counts")
        if not self.hardware or not self.workload:
            raise ValueError("hardware and workload contexts are required")
        if not self.safety_metrics or not self.resource_metrics:
            raise ValueError("safety and resource metric intervals are required")
        _require_unique_metric_names(self.safety_metrics, "safety")
        _require_unique_metric_names(self.resource_metrics, "resource")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_config_sha256": self.reference_config_sha256,
            "candidate_config_sha256": self.candidate_config_sha256,
            "code_revision": self.code_revision,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "split_sha256": self.split_sha256,
            "calibration_role": self.calibration_role,
            "test_role": self.test_role,
            "hardware": dict(sorted(self.hardware.items())),
            "workload": dict(sorted(self.workload.items())),
            "warmup_iterations": self.warmup_iterations,
            "repeat_iterations": self.repeat_iterations,
            "timing": self.timing.to_dict(),
            "safety_metrics": [metric.to_dict() for metric in self.safety_metrics],
            "resource_metrics": [metric.to_dict() for metric in self.resource_metrics],
            "decision": self.decision.value,
        }

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())


def _require_unique_metric_names(
    metrics: tuple[MetricInterval, ...],
    group: str,
) -> None:
    names = tuple(metric.name for metric in metrics)
    if len(names) != len(set(names)):
        raise ValueError(f"{group} metric names must be unique")
