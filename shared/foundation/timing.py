"""Portable, deterministic timing summary values."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from statistics import fmean
from typing import Any


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


__all__ = ["TimingSummary"]
