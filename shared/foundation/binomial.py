"""Exact binomial helpers shared by governance and evaluation."""

from __future__ import annotations

from math import ceil, expm1, isfinite, log1p


def zero_event_exact_upper_bound(
    trials: int,
    *,
    confidence_level: float,
) -> float:
    """One-sided exact binomial upper bound after observing zero events."""

    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    _validate_open_fraction(confidence_level, "confidence_level")
    return -expm1(log1p(-confidence_level) / trials)


def required_zero_event_trials(
    target_upper_rate: float,
    *,
    confidence_level: float,
) -> int:
    """Trials needed for a zero-event one-sided upper bound at target."""

    _validate_open_fraction(target_upper_rate, "target_upper_rate")
    _validate_open_fraction(confidence_level, "confidence_level")
    return ceil(log1p(-confidence_level) / log1p(-target_upper_rate))


def _validate_open_fraction(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 < value < 1
    ):
        raise ValueError(f"{name} must be finite and in (0, 1)")


__all__ = ["required_zero_event_trials", "zero_event_exact_upper_bound"]
