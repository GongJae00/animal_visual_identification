"""Deterministic OOF-only simplex fusion for appearance/face/nose research."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

OOF_SIMPLEX_SCHEMA_VERSION = "cvi.oof_simplex_fusion.v1"


class OOFSimplexError(ValueError):
    """Raised when the OOF simplex fusion contract is violated."""


def _validate_scalar(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise OOFSimplexError(f"{name} must be a finite number")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < minimum or (
        maximum is not None and numeric > maximum
    ):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise OOFSimplexError(f"{name} must be finite and {bounds}")


@dataclass(frozen=True, slots=True)
class OOFSimplexConfig:
    """Low-capacity grid-search settings.

    ``quality_floor`` is applied only to available channels, preventing an
    available zero-quality channel from making a row unusable when positive.
    """

    resolution: int = 20
    l2_strength: float = 0.0
    prior_weights: tuple[float, ...] | None = None
    quality_floor: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.resolution, bool) or not isinstance(self.resolution, int):
            raise OOFSimplexError("resolution must be a positive integer")
        if self.resolution < 1:
            raise OOFSimplexError("resolution must be a positive integer")
        _validate_scalar(self.l2_strength, "l2_strength", minimum=0.0)
        _validate_scalar(self.quality_floor, "quality_floor", minimum=0.0, maximum=1.0)
        if self.prior_weights is not None:
            prior = _validate_prior(self.prior_weights, expected_channels=None)
            object.__setattr__(self, "prior_weights", tuple(prior.tolist()))
        object.__setattr__(self, "l2_strength", float(self.l2_strength))
        object.__setattr__(self, "quality_floor", float(self.quality_floor))


@dataclass(frozen=True, slots=True)
class OOFSimplexModel:
    channel_names: tuple[str, ...]
    weights: np.ndarray
    config: OOFSimplexConfig
    objective: float
    report: dict[str, Any]

    def __post_init__(self) -> None:
        names = _validate_channel_names(self.channel_names)
        weights = _validate_simplex(self.weights, expected_channels=len(names))
        if not isinstance(self.config, OOFSimplexConfig):
            raise TypeError("config must be OOFSimplexConfig")
        _validate_scalar(self.objective, "objective", minimum=0.0)
        try:
            report = json.loads(json.dumps(self.report, allow_nan=False, sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise OOFSimplexError("report must be JSON-safe") from exc
        weights.setflags(write=False)
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "report", report)

    def predict_proba(
        self,
        calibrated_scores: Any,
        *,
        availability: Any | None = None,
        quality: Any | None = None,
    ) -> np.ndarray:
        """Fuse calibrated positive-class probabilities without producing labels."""

        scores = _numeric_matrix(calibrated_scores, "calibrated_scores", len(self.weights))
        available = _availability_matrix(availability, scores.shape)
        qualities = _quality_matrix(quality, scores.shape)
        effective = available.astype(np.float64) * np.maximum(
            qualities, float(self.config.quality_floor)
        )
        effective *= self.weights[None, :]
        totals = np.sum(effective, axis=1)
        if np.any(totals <= 0.0):
            raise OOFSimplexError("each row must have at least one usable channel")
        fused = np.sum(scores * effective, axis=1) / totals
        fused = np.asarray(fused, dtype=np.float64)
        fused.setflags(write=False)
        return fused


def fit_oof_simplex(
    channel_names: Any,
    calibrated_scores: Any,
    labels: Any,
    fold_ids: Any,
    *,
    availability: Any | None = None,
    quality: Any | None = None,
    sample_weights: Any | None = None,
    config: OOFSimplexConfig = OOFSimplexConfig(),
) -> OOFSimplexModel:
    """Fit simplex weights from caller-attested OOF calibrated probabilities."""

    if not isinstance(config, OOFSimplexConfig):
        raise TypeError("config must be OOFSimplexConfig")
    names = _validate_channel_names(channel_names)
    scores = _numeric_matrix(calibrated_scores, "calibrated_scores", len(names))
    n_samples = scores.shape[0]
    binary_labels = _binary_labels(labels, n_samples)
    _, fold_counts = _fold_ids(fold_ids, binary_labels, n_samples)
    available = _availability_matrix(availability, scores.shape)
    if np.any(np.sum(available, axis=0) == 0):
        raise OOFSimplexError("every channel must be available at least once")
    qualities = _quality_matrix(quality, scores.shape)
    weights_per_sample = _sample_weights(sample_weights, n_samples)
    prior = (
        np.full(len(names), 1.0 / len(names), dtype=np.float64)
        if config.prior_weights is None
        else _validate_prior(config.prior_weights, expected_channels=len(names))
    )
    _fuse(scores, prior, available, qualities, config.quality_floor)

    best_weights: np.ndarray | None = None
    best_objective = float("inf")
    best_brier = float("inf")
    best_distance = float("inf")
    for candidate in _simplex_grid(len(names), config.resolution):
        try:
            predictions = _fuse(
                scores, candidate, available, qualities, config.quality_floor
            )
        except OOFSimplexError:
            continue
        brier = float(
            np.sum(weights_per_sample * np.square(predictions - binary_labels))
            / np.sum(weights_per_sample)
        )
        distance = float(np.sum(np.square(candidate - prior)))
        objective = brier + float(config.l2_strength) * distance
        tie = abs(objective - best_objective) <= 1e-15
        if objective < best_objective - 1e-15 or (
            tie
            and (distance, tuple(candidate.tolist()))
            < (best_distance, tuple(best_weights.tolist()) if best_weights is not None else ())
        ):
            best_weights = candidate
            best_objective = objective
            best_brier = brier
            best_distance = distance
    if best_weights is None:  # pragma: no cover - grid construction guarantees candidates
        raise RuntimeError("simplex grid produced no candidates")

    report = {
        "schema_version": OOF_SIMPLEX_SCHEMA_VERSION,
        "channel_names": list(names),
        "sample_count": n_samples,
        "fold_count": len(fold_counts),
        "fold_sample_counts": fold_counts,
        "channel_available_counts": np.sum(available, axis=0).astype(int).tolist(),
        "weights": best_weights.tolist(),
        "objective": {
            "name": "weighted_brier_plus_l2_to_prior",
            "total": best_objective,
            "weighted_brier": best_brier,
            "l2_penalty": float(config.l2_strength) * best_distance,
        },
        "config": {
            **asdict(config),
            "prior_weights": prior.tolist(),
        },
        "limitations": {
            "oof_provenance_is_caller_attested": True,
            "does_not_verify_oof_provenance": True,
            "does_not_admit_final_test_data": True,
            "caller_must_exclude_final_test_data": True,
        },
    }
    return OOFSimplexModel(names, best_weights, config, best_objective, report)


def _fuse(
    scores: np.ndarray,
    weights: np.ndarray,
    availability: np.ndarray,
    quality: np.ndarray,
    quality_floor: float,
) -> np.ndarray:
    effective = weights[None, :] * availability * np.maximum(quality, quality_floor)
    totals = np.sum(effective, axis=1)
    if np.any(totals <= 0.0):
        raise OOFSimplexError("each row must have at least one usable channel")
    return np.sum(scores * effective, axis=1) / totals


def _simplex_grid(channels: int, resolution: int) -> Iterator[np.ndarray]:
    if channels == 2:
        for first in range(resolution + 1):
            yield np.asarray((first, resolution - first), dtype=np.float64) / resolution
        return
    for first in range(resolution + 1):
        for second in range(resolution - first + 1):
            yield np.asarray(
                (first, second, resolution - first - second), dtype=np.float64
            ) / resolution


def _validate_channel_names(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise OOFSimplexError("channel_names must be a sequence of 2 or 3 strings")
    names = tuple(values)
    if len(names) not in (2, 3):
        raise OOFSimplexError("exactly 2 or 3 channel names are required")
    if any(not isinstance(name, str) or not name or name != name.strip() for name in names):
        raise OOFSimplexError("channel names must be non-empty trimmed strings")
    if len(set(names)) != len(names):
        raise OOFSimplexError("channel names must be unique")
    return names


def _numeric_matrix(values: Any, name: str, channels: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise OOFSimplexError(f"{name} must be a numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != channels or array.shape[0] == 0:
        raise OOFSimplexError(f"{name} must have non-empty shape [N,{channels}]")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise OOFSimplexError(f"{name} must be finite and in [0, 1]")
    return array


def _binary_labels(values: Any, length: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != length or raw.dtype.kind not in "biu":
        raise OOFSimplexError("labels must be a length-N binary integer array")
    labels = np.asarray(raw, dtype=np.int64)
    if not np.array_equal(np.unique(labels), np.asarray((0, 1))):
        raise OOFSimplexError("labels must contain both binary classes")
    return labels.astype(np.float64)


def _fold_ids(values: Any, labels: np.ndarray, length: int) -> tuple[np.ndarray, list[int]]:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != length or raw.dtype.kind not in "iu":
        raise OOFSimplexError("fold_ids must be a length-N integer array")
    folds = np.asarray(raw, dtype=np.int64)
    unique = np.unique(folds)
    if len(unique) < 2 or not np.array_equal(unique, np.arange(len(unique))):
        raise OOFSimplexError("fold_ids must contain at least 2 contiguous non-negative folds")
    counts: list[int] = []
    for fold in unique:
        mask = folds == fold
        counts.append(int(np.sum(mask)))
        if not np.array_equal(np.unique(labels[mask]), np.asarray((0.0, 1.0))):
            raise OOFSimplexError("each fold must contain both label classes")
    return folds, counts


def _availability_matrix(values: Any | None, shape: tuple[int, int]) -> np.ndarray:
    if values is None:
        return np.ones(shape, dtype=bool)
    array = np.asarray(values)
    if array.shape != shape or array.dtype != np.dtype(bool):
        raise OOFSimplexError("availability must be a bool array matching scores")
    return array


def _quality_matrix(values: Any | None, shape: tuple[int, int]) -> np.ndarray:
    if values is None:
        return np.ones(shape, dtype=np.float64)
    raw = np.asarray(values)
    if raw.shape != shape or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise OOFSimplexError("quality must be a numeric array matching scores")
    quality = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(quality)) or np.any(quality < 0.0) or np.any(quality > 1.0):
        raise OOFSimplexError("quality must be finite and in [0, 1]")
    return quality


def _sample_weights(values: Any | None, length: int) -> np.ndarray:
    if values is None:
        return np.ones(length, dtype=np.float64)
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != length or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise OOFSimplexError("sample_weights must be a length-N numeric array")
    weights = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise OOFSimplexError("sample_weights must be finite and positive")
    return weights


def _validate_prior(values: Any, expected_channels: int | None) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise OOFSimplexError("prior_weights must be a numeric simplex")
    prior = np.array(raw, dtype=np.float64, copy=True)
    if expected_channels is not None and len(prior) != expected_channels:
        raise OOFSimplexError("prior_weights length must match channel_names")
    if len(prior) not in (2, 3) or not np.all(np.isfinite(prior)) or np.any(prior <= 0.0):
        raise OOFSimplexError("prior_weights must be a strict positive simplex of length 2 or 3")
    if not np.isclose(float(np.sum(prior)), 1.0, rtol=1e-12, atol=1e-12):
        raise OOFSimplexError("prior_weights must sum to one")
    return prior


def _validate_simplex(values: Any, expected_channels: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != expected_channels or raw.dtype.kind not in "iuf":
        raise OOFSimplexError("weights must be a numeric simplex matching channel_names")
    weights = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise OOFSimplexError("weights must be finite and non-negative")
    if not np.isclose(float(np.sum(weights)), 1.0, rtol=1e-12, atol=1e-12):
        raise OOFSimplexError("weights must sum to one")
    return weights


__all__ = [
    "OOF_SIMPLEX_SCHEMA_VERSION",
    "OOFSimplexConfig",
    "OOFSimplexError",
    "OOFSimplexModel",
    "fit_oof_simplex",
]
