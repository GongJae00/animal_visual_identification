"""Dataset-balanced, identity-grouped OOF assignment within an admitted role."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from math import isclose
from typing import Any

import numpy as np

from evaluation.splits.role_exposure import ExposureStage


ROBUSTNESS_PROTOCOL_SCHEMA_VERSION = "cvi.robustness_protocol.v1"
_ALLOWED_TARGET_STAGES = {
    ExposureStage.MODEL_SELECTION_SCORED,
    ExposureStage.CALIBRATION_SCORED,
}


class RobustnessProtocolError(ValueError):
    """Raised when an in-role OOF protocol contract is violated."""


@dataclass(frozen=True, slots=True)
class RobustnessProtocolConfig:
    n_splits: int
    seed: int
    target_exposure_stage: ExposureStage

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_splits, bool)
            or not isinstance(self.n_splits, int)
            or self.n_splits < 2
        ):
            raise RobustnessProtocolError("n_splits must be an integer >= 2")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise RobustnessProtocolError("seed must be a non-negative integer")
        if not isinstance(self.target_exposure_stage, ExposureStage):
            raise TypeError("target_exposure_stage must be ExposureStage")
        if self.target_exposure_stage not in _ALLOWED_TARGET_STAGES:
            raise RobustnessProtocolError(
                "target_exposure_stage must be MODEL_SELECTION_SCORED or "
                "CALIBRATION_SCORED"
            )


@dataclass(frozen=True, slots=True)
class RobustnessProtocolResult:
    fold_ids: np.ndarray
    sample_weights: np.ndarray
    report: dict[str, Any]

    def __post_init__(self) -> None:
        folds = np.asarray(self.fold_ids)
        weights = np.asarray(self.sample_weights)
        if folds.ndim != 1 or folds.dtype != np.dtype(np.int64):
            raise RobustnessProtocolError("fold_ids must be a 1-D int64 array")
        if weights.ndim != 1 or weights.dtype != np.dtype(np.float64):
            raise RobustnessProtocolError("sample_weights must be a 1-D float64 array")
        if len(folds) == 0 or len(folds) != len(weights):
            raise RobustnessProtocolError("result arrays must be non-empty and equal length")
        if not np.all(np.isfinite(weights)) or not np.all(weights > 0.0):
            raise RobustnessProtocolError("sample_weights must be finite and positive")
        try:
            report = json.loads(json.dumps(self.report, allow_nan=False, sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise RobustnessProtocolError("report must be JSON-safe") from exc

        folds = np.array(folds, dtype=np.int64, copy=True)
        weights = np.array(weights, dtype=np.float64, copy=True)
        folds.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "fold_ids", folds)
        object.__setattr__(self, "sample_weights", weights)
        object.__setattr__(self, "report", report)


def build_dataset_balanced_oof_protocol(
    dataset_ids: Any,
    identity_ids: Any,
    *,
    config: RobustnessProtocolConfig,
    historical_exposure_stages: Any | None = None,
) -> RobustnessProtocolResult:
    """Build deterministic OOF folds and hierarchical weights inside one role.

    This function operates only inside an already admitted role. It does not
    create a train/calibration/test split, restore untouched status, or admit
    exposed data. ``target_exposure_stage`` identifies that existing role and
    exposure history is used only to reject a historical stage above it.
    """

    if not isinstance(config, RobustnessProtocolConfig):
        raise TypeError("config must be RobustnessProtocolConfig")
    datasets = _as_stable_ids(dataset_ids, "dataset_ids")
    identities = _as_stable_ids(identity_ids, "identity_ids")
    if len(datasets) != len(identities):
        raise RobustnessProtocolError("dataset_ids and identity_ids lengths differ")
    historical_stages = _as_historical_stages(
        historical_exposure_stages, len(datasets)
    )
    _validate_historical_exposure(historical_stages, config.target_exposure_stage)

    samples_by_dataset_identity: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    dataset_by_identity: dict[str, str] = {}
    for index, (dataset, identity) in enumerate(zip(datasets, identities, strict=True)):
        prior_dataset = dataset_by_identity.setdefault(identity, dataset)
        if prior_dataset != dataset:
            raise RobustnessProtocolError(
                "an identity_id must not be shared across datasets"
            )
        samples_by_dataset_identity[dataset][identity].append(index)

    for dataset, identity_samples in samples_by_dataset_identity.items():
        if len(identity_samples) < config.n_splits:
            raise RobustnessProtocolError(
                f"dataset {dataset!r} has {len(identity_samples)} identities; "
                f"at least n_splits={config.n_splits} are required"
            )

    fold_ids = np.empty(len(datasets), dtype=np.int64)
    for dataset in sorted(samples_by_dataset_identity):
        identity_samples = samples_by_dataset_identity[dataset]
        ordered_identities = sorted(
            identity_samples,
            key=lambda identity: (
                _identity_order_digest(config.seed, dataset, identity),
                identity,
            ),
        )
        for rank, identity in enumerate(ordered_identities):
            fold = rank % config.n_splits
            fold_ids[identity_samples[identity]] = fold

    sample_weights = np.empty(len(datasets), dtype=np.float64)
    dataset_weight = len(datasets) / len(samples_by_dataset_identity)
    for identity_samples in samples_by_dataset_identity.values():
        identity_weight = dataset_weight / len(identity_samples)
        for sample_indexes in identity_samples.values():
            sample_weights[sample_indexes] = identity_weight / len(sample_indexes)
    sample_weights /= float(np.mean(sample_weights))

    _validate_constructed_protocol(
        fold_ids,
        sample_weights,
        samples_by_dataset_identity,
        config.n_splits,
    )
    report = _aggregate_report(
        fold_ids,
        samples_by_dataset_identity,
        config,
        len(datasets),
        historical_stages is not None,
    )
    return RobustnessProtocolResult(fold_ids, sample_weights, report)


def _as_stable_ids(values: Any, name: str) -> tuple[str, ...]:
    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise RobustnessProtocolError(f"{name} must be a 1-D sequence") from exc
    if array.ndim != 1:
        raise RobustnessProtocolError(f"{name} must be 1-D, got shape {array.shape}")
    if len(array) == 0:
        raise RobustnessProtocolError(f"{name} must not be empty")
    result: list[str] = []
    for index, value in enumerate(array.tolist()):
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or "\x00" in value
        ):
            raise RobustnessProtocolError(
                f"{name}[{index}] must be a stable non-empty string"
            )
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise RobustnessProtocolError(
                f"{name}[{index}] must be valid UTF-8 text"
            ) from exc
        result.append(value)
    return tuple(result)


def _as_historical_stages(
    values: Any | None, expected_length: int
) -> tuple[ExposureStage, ...] | None:
    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise RobustnessProtocolError(
            "historical_exposure_stages must be a 1-D sequence"
        ) from exc
    if array.ndim != 1:
        raise RobustnessProtocolError(
            "historical_exposure_stages must be 1-D"
        )
    if len(array) != expected_length:
        raise RobustnessProtocolError(
            "historical_exposure_stages length differs from IDs"
        )
    stages = tuple(array.tolist())
    if any(not isinstance(stage, ExposureStage) for stage in stages):
        raise TypeError("historical_exposure_stages must contain ExposureStage values")
    return stages


def _validate_historical_exposure(
    stages: tuple[ExposureStage, ...] | None,
    target: ExposureStage,
) -> None:
    if stages is None:
        return
    ordered_stages = tuple(ExposureStage)
    target_rank = ordered_stages.index(target)
    for stage in stages:
        if ordered_stages.index(stage) > target_rank:
            raise RobustnessProtocolError(
                f"historical exposure {stage.value} exceeds target {target.value}; "
                "exposed data cannot be admitted or restored to an earlier role"
            )


def _identity_order_digest(seed: int, dataset: str, identity: str) -> bytes:
    payload = json.dumps(
        ["CVI_ROBUSTNESS_OOF_ORDER_V1", seed, dataset, identity],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _validate_constructed_protocol(
    fold_ids: np.ndarray,
    sample_weights: np.ndarray,
    samples_by_dataset_identity: dict[str, dict[str, list[int]]],
    n_splits: int,
) -> None:
    if fold_ids.dtype != np.int64 or np.any(fold_ids < 0) or np.any(
        fold_ids >= n_splits
    ):
        raise RuntimeError("constructed fold IDs violate the configured range")
    if not np.all(np.isfinite(sample_weights)) or not np.all(sample_weights > 0.0):
        raise RuntimeError("constructed sample weights are not finite and positive")
    if not isclose(float(np.mean(sample_weights)), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("constructed sample weights do not have mean one")

    dataset_totals: list[float] = []
    for identity_samples in samples_by_dataset_identity.values():
        fold_identity_counts = [0] * n_splits
        identity_totals: list[float] = []
        for sample_indexes in identity_samples.values():
            identity_folds = np.unique(fold_ids[sample_indexes])
            if len(identity_folds) != 1:
                raise RuntimeError("constructed identity crosses OOF folds")
            fold_identity_counts[int(identity_folds[0])] += 1
            identity_weights = sample_weights[sample_indexes]
            if not np.allclose(
                identity_weights,
                identity_weights[0],
                rtol=1e-12,
                atol=1e-12,
            ):
                raise RuntimeError("samples within an identity have unequal weight")
            identity_totals.append(float(np.sum(identity_weights)))
        if max(fold_identity_counts) - min(fold_identity_counts) > 1:
            raise RuntimeError("constructed dataset identity folds are imbalanced")
        if not np.allclose(
            identity_totals,
            identity_totals[0],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError("identities within a dataset have unequal weight")
        dataset_totals.append(float(sum(identity_totals)))
    if not np.allclose(
        dataset_totals,
        dataset_totals[0],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("datasets have unequal total weight")


def _aggregate_report(
    fold_ids: np.ndarray,
    samples_by_dataset_identity: dict[str, dict[str, list[int]]],
    config: RobustnessProtocolConfig,
    sample_count: int,
    historical_exposure_stages_provided: bool,
) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    for dataset in sorted(samples_by_dataset_identity):
        identity_samples = samples_by_dataset_identity[dataset]
        fold_identity_counts = [0] * config.n_splits
        fold_sample_counts = [0] * config.n_splits
        for sample_indexes in identity_samples.values():
            fold = int(fold_ids[sample_indexes[0]])
            fold_identity_counts[fold] += 1
            fold_sample_counts[fold] += len(sample_indexes)
        datasets.append(
            {
                "dataset_name": dataset,
                "sample_count": sum(len(indexes) for indexes in identity_samples.values()),
                "identity_count": len(identity_samples),
                "fold_identity_counts": fold_identity_counts,
                "fold_sample_counts": fold_sample_counts,
            }
        )
    return {
        "schema_version": ROBUSTNESS_PROTOCOL_SCHEMA_VERSION,
        "target_exposure_stage": config.target_exposure_stage.value,
        "n_splits": config.n_splits,
        "seed": config.seed,
        "sample_count": sample_count,
        "dataset_count": len(datasets),
        "historical_exposure_stages_provided": (
            historical_exposure_stages_provided
        ),
        "datasets": datasets,
        "limitations": {
            "operates_only_inside_already_admitted_role": True,
            "does_not_create_train_calibration_test_split": True,
            "does_not_restore_untouched_status": True,
            "does_not_admit_exposed_data": True,
            "caller_must_supply_authenticated_exposure_history": True,
        },
    }


__all__ = [
    "ROBUSTNESS_PROTOCOL_SCHEMA_VERSION",
    "RobustnessProtocolConfig",
    "RobustnessProtocolError",
    "RobustnessProtocolResult",
    "build_dataset_balanced_oof_protocol",
]
