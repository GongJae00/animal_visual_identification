"""Research-only fixed-population evaluation of Nose sequence architectures."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from artifact_contracts.artifact_manifest import (
    ArtifactContractError,
    ExactOnnxRuntime,
    NoseEmbeddingManifest,
    NoseMaskManifest,
    UsageLane,
    preprocess_image,
)
from evaluation.retrieval import (
    compute_cosine_score_matrix,
    identity_clustered_bootstrap_ci,
)
from identity_methods.nose.restoration import RestorationConfig, restore_nose_frames
from identity_methods.nose.temporal import aggregate_nose_embeddings
from localization.nose_region.embedding_views import student_masked_rgb
from localization.nose_region.native_yt import validate_manifest_bundle
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.provenance import content_sha256


REPORT_SCHEMA = "cvi.yt_nose_architecture_evaluation.v2"
REPORT_BUNDLE_SCHEMA = "cvi.yt_nose_architecture_evaluation_bundle.v2"
INTERPRETATION = (
    "WITHIN_VIDEO_TRACK_RESEARCH_ARCHITECTURE_DIAGNOSTIC_"
    "NOT_BIOLOGICAL_IDENTITY_VALIDATION_OR_FINAL_EVALUATION"
)
METHODS = {
    "A_raw_K5": "raw per-frame embeddings followed by an unweighted K=5 L2 mean",
    "B_student_masked_K5": (
        "student-mask conservative RGB views embedded per frame followed by an "
        "unweighted K=5 L2 mean"
    ),
    "C_raw_plus_masked_late_fusion": "L2 mean of A_raw_K5 and B_student_masked_K5",
    "D_restored_early_fusion": (
        "one embedding of the conservative five-frame restored RGB image"
    ),
    "E_raw_plus_restored_late_fusion": "L2 mean of A_raw_K5 and D_restored_early_fusion",
    "F_all_branches_late_fusion": (
        "L2 mean of A_raw_K5, B_student_masked_K5, and D_restored_early_fusion"
    ),
}
_BASELINE = "A_raw_K5"
_METRICS = ("Rank-1", "Rank-5", "MRR", "mAP")
_SCORE_FUSION_BRANCHES = (
    "A_raw_K5",
    "B_student_masked_K5",
    "D_restored_early_fusion",
)
_SCORE_FUSION_INTERPRETATION = (
    "CALIBRATION_IDENTITY_SELECTED_SCORE_FUSION_RESEARCH_DIAGNOSTIC_"
    "EVALUATION_IDENTITIES_NOT_USED_FOR_WEIGHT_SELECTION_"
    "NOT_BIOLOGICAL_IDENTITY_VALIDATION_OR_FINAL_EVALUATION"
)
_MINIMUM_FUSION_PARTITION_IDENTITIES = 10
_ZSCORE_STD_EPSILON = 1e-8
_CODE_PATHS = (
    "experiments/nose_architecture.py",
    "identity_methods/nose/restoration.py",
    "identity_methods/nose/temporal.py",
    "artifact_contracts/artifact_manifest.py",
    "evaluation/retrieval.py",
    "localization/nose_region/embedding_views.py",
    "localization/nose_region/native_yt.py",
    "foundation/protected_io.py",
    "foundation/provenance.py",
    "workflows/evaluate_yt_nose_architecture.py",
)
_IMAGE_SIZE = 224
_OUTSIDE_SUPPORT_ORIGINAL_WEIGHT = 0.25
_POPULATION_ROLES = {"all", "consistency_eval"}


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input must be a regular non-symlink file: {path}")
    before = path.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"input changed while hashing: {path}")
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": os.fspath(path),
        "sha256": _file_sha256(path),
        "byte_size": path.stat(follow_symlinks=False).st_size,
    }


def _validate_population_role_inputs(
    population_role: str,
    embedding_lineage_path: Path | None,
    embedding_lineage_sha256: str | None,
) -> str | None:
    if population_role not in _POPULATION_ROLES:
        raise ValueError("population_role must be 'all' or 'consistency_eval'")
    supplied = (embedding_lineage_path is not None, embedding_lineage_sha256 is not None)
    if population_role == "all":
        if any(supplied):
            raise ValueError("population_role='all' rejects embedding lineage inputs")
        return None
    if not all(supplied):
        raise ValueError(
            "population_role='consistency_eval' requires both embedding lineage inputs"
        )
    return _require_sha256(embedding_lineage_sha256, "embedding_lineage_sha256")


def _canonical_uuid5(value: object, context: str) -> str:
    try:
        parsed = uuid.UUID(value)  # type: ignore[arg-type]
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{context} must be a canonical UUIDv5") from exc
    if parsed.version != 5 or str(parsed) != value:
        raise ValueError(f"{context} must be a canonical UUIDv5")
    return str(parsed)


def _filter_consistency_eval_population(
    population: Sequence[dict[str, Any]], eval_identities: object
) -> list[dict[str, Any]]:
    if (
        not isinstance(eval_identities, list)
        or len(eval_identities) < 20
        or any(not isinstance(identity, str) for identity in eval_identities)
        or eval_identities != sorted(set(eval_identities))
    ):
        raise ValueError(
            "consistency lineage EVAL identities must be a sorted unique list with at least 20 IDs"
        )
    for index, identity in enumerate(eval_identities):
        _canonical_uuid5(identity, f"consistency lineage EVAL identity {index}")

    expected = set(eval_identities)
    selected = [item for item in population if item["registered_dog_id"] in expected]
    selected_identities = [item["registered_dog_id"] for item in selected]
    missing = expected - set(selected_identities)
    extras = set(selected_identities) - expected
    if missing or extras or selected_identities != eval_identities:
        raise ValueError(
            "eligible K5 population does not exactly match consistency lineage EVAL "
            f"identities (missing={len(missing)}, extras={len(extras)})"
        )
    return selected


def _load_consistency_lineage(
    *,
    lineage_path: Path,
    expected_content_sha256: str,
    embedding_manifest_path: Path,
    embedding_document: Any,
    embedding_onnx_path: Path,
    embedding_onnx_binding: Mapping[str, Any],
) -> tuple[Any, list[str]]:
    lineage_document = read_strict_json_document(lineage_path)
    if lineage_document.canonical_payload_sha256 != expected_content_sha256:
        raise ValueError("embedding lineage content SHA-256 differs from the external pin")
    lineage_root = lineage_path.parent.resolve(strict=True)
    from localization.nose_region.embedding_consistency_training import validate_lineage_manifest

    validate_lineage_manifest(lineage_document.payload, lineage_root)
    artifacts = lineage_document.payload["artifacts"]
    runtime_artifact = artifacts["runtime_manifest"]
    onnx_artifact = artifacts["onnx"]
    expected_runtime_path = lineage_root.joinpath(
        *PurePosixPath(runtime_artifact["path"]).parts
    ).resolve(strict=True)
    expected_onnx_path = lineage_root.joinpath(
        *PurePosixPath(onnx_artifact["path"]).parts
    ).resolve(strict=True)
    if embedding_manifest_path.resolve(strict=True) != expected_runtime_path:
        raise ValueError("embedding runtime manifest is not the lineage artifact path")
    if embedding_onnx_path.resolve(strict=True) != expected_onnx_path:
        raise ValueError("embedding ONNX is not the lineage artifact path")
    if (
        embedding_document.raw_sha256 != runtime_artifact["sha256"]
        or embedding_document.byte_size != runtime_artifact["bytes"]
    ):
        raise ValueError("embedding runtime manifest bytes differ from the lineage artifact")
    if (
        embedding_onnx_binding["sha256"] != onnx_artifact["sha256"]
        or embedding_onnx_binding["byte_size"] != onnx_artifact["bytes"]
    ):
        raise ValueError("embedding ONNX bytes differ from the lineage artifact")
    return lineage_document, lineage_document.payload["bindings"]["splits"][
        "identity_lists"
    ]["eval"]


def _normalize(vector: np.ndarray, context: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if (
        value.ndim != 1
        or not np.isfinite(value).all()
        or not math.isfinite(norm)
        or norm <= 1e-8
    ):
        raise ArtifactContractError(f"{context} produced a non-finite or zero-norm vector")
    return np.asarray(value / norm, dtype=np.float32)


def _l2_mean(vectors: Sequence[np.ndarray], context: str) -> np.ndarray:
    if len(vectors) < 2:
        raise ValueError(f"{context} requires at least two vectors")
    normalized = [_normalize(vector, context) for vector in vectors]
    dimensions = {vector.shape for vector in normalized}
    if len(dimensions) != 1:
        raise ValueError(f"{context} vector dimensions differ")
    return _normalize(np.mean(np.stack(normalized), axis=0), context)


def _fuse_k5(vectors: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    if len(vectors) != 5:
        raise RuntimeError("architecture evaluation requires exactly five frame embeddings")
    result = aggregate_nose_embeddings(vectors)
    if result.aggregation != "UNWEIGHTED_L2_MEAN":
        raise RuntimeError("K=5 temporal fusion was not the strict unweighted L2 mean")
    return result.embedding, result.diagnostics()


def _group_population(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    localized = [row for row in records if row["record_state"] != "NO_ROI"]
    by_identity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    owners: dict[str, dict[str, str]] = {
        field: {} for field in ("identity_token", "track_token", "sequence_token")
    }
    for row in localized:
        identity = row["registered_dog_id"]
        by_identity[identity].append(row)
        for field, owner_by_value in owners.items():
            value = row[field]
            owner = owner_by_value.setdefault(value, identity)
            if owner != identity:
                raise ValueError(f"one YT {field} maps to multiple registered identities")

    population: list[dict[str, Any]] = []
    excluded = {"fewer_than_ten_localized_frames": 0}
    for identity in sorted(by_identity):
        rows = by_identity[identity]
        identity_tokens = {row["identity_token"] for row in rows}
        tracks = {row["track_token"] for row in rows}
        sequences = {row["sequence_token"] for row in rows}
        if len(identity_tokens) != 1 or len(tracks) != 1 or len(sequences) != 1:
            raise ValueError(
                "registered YT identity must map to exactly one identity/track/sequence"
            )
        ordered = sorted(rows, key=lambda row: (row["frame_index"], row["sample_token"]))
        frame_indices = [row["frame_index"] for row in ordered]
        if len(frame_indices) != len(set(frame_indices)):
            raise ValueError("localized YT track repeats a frame index")
        if len(ordered) < 10:
            excluded["fewer_than_ten_localized_frames"] += 1
            continue
        gallery = ordered[:5]
        query = ordered[-5:]
        if {row["sample_token"] for row in gallery} & {
            row["sample_token"] for row in query
        } or {row["frame_index"] for row in gallery} & {
            row["frame_index"] for row in query
        }:
            raise RuntimeError("earliest/latest five-frame windows overlap")
        population.append(
            {
                "registered_dog_id": identity,
                "identity_token": next(iter(identity_tokens)),
                "track_token": next(iter(tracks)),
                "sequence_token": next(iter(sequences)),
                "localized_frame_count": len(ordered),
                "gallery": gallery,
                "query": query,
            }
        )
    if len(population) < 2:
        raise ValueError("evaluation requires at least two YT identities with ten localized frames")
    return population, excluded


def _load_crop_rgb(root: Path, row: Mapping[str, Any]) -> np.ndarray:
    with Image.open(root / row["crop_path"]) as opened:
        values = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    if values.ndim != 3 or values.shape[2] != 3 or min(values.shape[:2]) <= 0:
        raise RuntimeError("Nose crop must be nonempty HxWx3 RGB")
    return values


def _resize_mask_branch_rgb(image: np.ndarray) -> np.ndarray:
    resized = Image.fromarray(image, mode="RGB").resize(
        (_IMAGE_SIZE, _IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.uint8)


def _embed(
    runtime: ExactOnnxRuntime,
    manifest: NoseEmbeddingManifest,
    image: np.ndarray,
) -> np.ndarray:
    output = runtime.run(preprocess_image(Image.fromarray(image, mode="RGB"), manifest))[0]
    return _normalize(output, "Nose embedding ONNX")


def _student_mask(
    runtime: ExactOnnxRuntime,
    manifest: NoseMaskManifest,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    probability = runtime.run(
        preprocess_image(Image.fromarray(image, mode="RGB"), manifest)
    )[0, 0]
    if probability.shape != (_IMAGE_SIZE, _IMAGE_SIZE):
        raise ArtifactContractError("student mask output must be exactly 224x224")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ArtifactContractError("student mask probabilities must be in [0,1]")
    support = probability >= manifest.threshold
    median = np.median(image, axis=(0, 1)).astype(np.float32)
    masked = student_masked_rgb(
        image,
        support,
        outside_support_original_weight=_OUTSIDE_SUPPORT_ORIGINAL_WEIGHT,
    )
    uncertainty = 1.0 - np.abs(2.0 * probability.astype(np.float64) - 1.0)
    diagnostics = {
        "support_fraction": float(support.mean()),
        "mean_probability": float(probability.mean()),
        "mean_binary_uncertainty": float(uncertainty.mean()),
        "threshold_margin_le_0_05_fraction": float(
            (np.abs(probability - manifest.threshold) <= 0.05).mean()
        ),
        "channel_median_rgb": [float(value) for value in median],
    }
    return masked, support, diagnostics


def _rank_rows(scores: np.ndarray, identities: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, identity in enumerate(identities):
        order = np.argsort(-scores[index], kind="stable")
        rank = int(np.flatnonzero(order == index)[0]) + 1
        impostors = np.delete(scores[index], index)
        genuine = float(scores[index, index])
        rows.append(
            {
                "registered_dog_id": identity,
                "rank": rank,
                "Rank-1": float(rank <= 1),
                "Rank-5": float(rank <= 5),
                "MRR": 1.0 / rank,
                "mAP": 1.0 / rank,
                "genuine_score": genuine,
                "best_impostor_score": float(impostors.max()),
                "genuine_margin": genuine - float(impostors.max()),
            }
        )
    return rows


def _metrics(rows: Sequence[Mapping[str, Any]], gallery_count: int) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "gallery_count": gallery_count,
        **{
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in _METRICS
        },
    }


def _delta(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        metric: float(current[metric] - baseline[metric]) for metric in _METRICS
    } | {
        "rank_improvement": float(baseline["rank"] - current["rank"]),
        "genuine_score": float(current["genuine_score"] - baseline["genuine_score"]),
        "genuine_margin": float(current["genuine_margin"] - baseline["genuine_margin"]),
    }


def _score_fusion_parameters(
    calibration_fraction: float,
    calibration_seed: int,
    fusion_grid_step: float,
) -> tuple[float, int, float, int]:
    if (
        isinstance(calibration_fraction, bool)
        or not isinstance(calibration_fraction, (int, float))
        or not math.isfinite(float(calibration_fraction))
        or not 0.0 < float(calibration_fraction) < 1.0
    ):
        raise ValueError("calibration_fraction must be finite and in (0,1)")
    if (
        isinstance(calibration_seed, bool)
        or not isinstance(calibration_seed, int)
        or calibration_seed < 0
    ):
        raise ValueError("calibration_seed must be a non-negative integer")
    if (
        isinstance(fusion_grid_step, bool)
        or not isinstance(fusion_grid_step, (int, float))
        or not math.isfinite(float(fusion_grid_step))
        or not 0.0 < float(fusion_grid_step) <= 1.0
    ):
        raise ValueError("fusion_grid_step must be finite and in (0,1]")
    grid_units = int(round(1.0 / float(fusion_grid_step)))
    if grid_units <= 0 or not math.isclose(
        grid_units * float(fusion_grid_step), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("fusion_grid_step must divide one exactly")
    return (
        float(calibration_fraction),
        calibration_seed,
        float(fusion_grid_step),
        grid_units,
    )


def _split_score_fusion_identities(
    identities: Sequence[str], *, calibration_fraction: float, calibration_seed: int
) -> tuple[list[int], list[int]]:
    threshold = int(calibration_fraction * (1 << 256))
    calibration_indices: list[int] = []
    evaluation_indices: list[int] = []
    for index, identity in enumerate(identities):
        digest = hashlib.sha256(
            f"{calibration_seed}:{identity}".encode("utf-8")
        ).digest()
        target = (
            calibration_indices
            if int.from_bytes(digest, byteorder="big") < threshold
            else evaluation_indices
        )
        target.append(index)
    if min(len(calibration_indices), len(evaluation_indices)) < (
        _MINIMUM_FUSION_PARTITION_IDENTITIES
    ):
        raise ValueError(
            "calibrated score fusion requires at least 10 calibration and 10 "
            "evaluation identities"
        )
    return calibration_indices, evaluation_indices


def _partition_row_zscores(scores: np.ndarray, context: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError(f"{context} score matrix must be square with at least two rows")
    if not np.isfinite(values).all():
        raise ValueError(f"{context} score matrix must be finite")
    means = values.mean(axis=1, keepdims=True)
    standard_deviations = values.std(axis=1, ddof=0, keepdims=True)
    if (
        not np.isfinite(means).all()
        or not np.isfinite(standard_deviations).all()
        or np.any(standard_deviations <= _ZSCORE_STD_EPSILON)
    ):
        raise ValueError(
            f"{context} has a non-finite or near-zero full-gallery row standard deviation"
        )
    normalized = (values - means) / standard_deviations
    if not np.isfinite(normalized).all():
        raise ValueError(f"{context} row z-scores must be finite")
    return normalized


def _identity_list_sha256(identities: Sequence[str]) -> str:
    return content_sha256({"registered_dog_ids": list(identities)})


def evaluate_calibrated_score_fusion(
    identities: Sequence[str],
    score_matrices: Mapping[str, np.ndarray],
    *,
    calibration_fraction: float = 0.3,
    calibration_seed: int = 73,
    fusion_grid_step: float = 0.05,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Select A/B/D score-fusion weights on CAL identities and evaluate on EVAL."""

    fraction, split_seed, grid_step, grid_units = _score_fusion_parameters(
        calibration_fraction, calibration_seed, fusion_grid_step
    )
    identity_list = list(identities)
    if (
        not identity_list
        or any(not isinstance(identity, str) or not identity for identity in identity_list)
        or identity_list != sorted(set(identity_list))
    ):
        raise ValueError("score-fusion identities must be unique and lexically sorted")
    if set(score_matrices) != set(_SCORE_FUSION_BRANCHES):
        raise ValueError("score fusion requires exactly the A raw, B masked, and D restored matrices")
    full_scores: dict[str, np.ndarray] = {}
    expected_shape = (len(identity_list), len(identity_list))
    for branch in _SCORE_FUSION_BRANCHES:
        matrix = np.asarray(score_matrices[branch], dtype=np.float64)
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(f"{branch} score matrix must be finite with shape {expected_shape}")
        full_scores[branch] = matrix

    calibration_indices, evaluation_indices = _split_score_fusion_identities(
        identity_list,
        calibration_fraction=fraction,
        calibration_seed=split_seed,
    )
    partition_indices = {
        "CAL": calibration_indices,
        "EVAL": evaluation_indices,
    }
    partition_identities = {
        name: [identity_list[index] for index in indices]
        for name, indices in partition_indices.items()
    }
    partition_zscores: dict[str, dict[str, np.ndarray]] = {}
    for partition, indices in partition_indices.items():
        selector = np.ix_(indices, indices)
        partition_zscores[partition] = {
            branch: _partition_row_zscores(
                full_scores[branch][selector], f"{partition} {branch}"
            )
            for branch in _SCORE_FUSION_BRANCHES
        }

    calibration_outcomes = {
        branch: _rank_rows(
            partition_zscores["CAL"][branch], partition_identities["CAL"]
        )
        for branch in _SCORE_FUSION_BRANCHES
    }
    calibration_metrics = {
        branch: _metrics(rows, len(partition_identities["CAL"]))
        for branch, rows in calibration_outcomes.items()
    }
    best_key: tuple[float, ...] | None = None
    selected_weights: dict[str, float] | None = None
    selected_metrics: dict[str, Any] | None = None
    candidate_count = 0
    for raw_units in range(grid_units + 1):
        for masked_units in range(grid_units - raw_units + 1):
            raw_weight = raw_units / grid_units
            masked_weight = masked_units / grid_units
            restored_weight = 1.0 - (raw_weight + masked_weight)
            weights = {
                "A_raw_K5": raw_weight,
                "B_student_masked_K5": masked_weight,
                "D_restored_early_fusion": restored_weight,
            }
            if sum(weights.values()) != 1.0 or min(weights.values()) < 0.0:
                raise RuntimeError("simplex search did not produce exact nonnegative weights")
            fused = sum(
                weights[branch] * partition_zscores["CAL"][branch]
                for branch in _SCORE_FUSION_BRANCHES
            )
            rows = _rank_rows(fused, partition_identities["CAL"])
            metrics = _metrics(rows, len(partition_identities["CAL"]))
            key = (
                metrics["Rank-1"],
                metrics["MRR"],
                metrics["Rank-5"],
                raw_weight,
                masked_weight,
            )
            candidate_count += 1
            if best_key is None or key > best_key:
                best_key = key
                selected_weights = weights
                selected_metrics = metrics
    if selected_weights is None or selected_metrics is None:
        raise RuntimeError("simplex score-fusion search produced no candidate")
    calibration_metrics["selected_score_fusion"] = selected_metrics

    evaluation_baseline_scores = partition_zscores["EVAL"][_BASELINE]
    evaluation_fused_scores = sum(
        selected_weights[branch] * partition_zscores["EVAL"][branch]
        for branch in _SCORE_FUSION_BRANCHES
    )
    evaluation_baseline_rows = _rank_rows(
        evaluation_baseline_scores, partition_identities["EVAL"]
    )
    evaluation_fused_rows = _rank_rows(
        evaluation_fused_scores, partition_identities["EVAL"]
    )
    paired_rows: list[dict[str, Any]] = []
    per_identity: list[dict[str, Any]] = []
    for identity, baseline, fused in zip(
        partition_identities["EVAL"],
        evaluation_baseline_rows,
        evaluation_fused_rows,
        strict=True,
    ):
        delta = _delta(fused, baseline)
        per_identity.append(
            {
                "registered_dog_id": identity,
                "baseline_A_outcome": baseline,
                "fused_outcome": fused,
                "fused_minus_A_delta": delta,
            }
        )
        paired_rows.append({"bootstrap_cluster_id": identity, **delta})
    paired_bootstrap = {
        metric: identity_clustered_bootstrap_ci(
            paired_rows,
            metric=metric,
            confidence_level=float(bootstrap_confidence_level),
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        for metric in _METRICS
    }

    split_config = {
        "algorithm": "SHA256_INTEGER_THRESHOLD_V1",
        "hash_input": "UTF8(str(calibration_seed) + ':' + registered_dog_id)",
        "calibration_fraction": fraction,
        "calibration_seed": split_seed,
        "minimum_identities_per_partition": _MINIMUM_FUSION_PARTITION_IDENTITIES,
    }
    zscore_config = {
        "scope": "EACH_QUERY_ROW_FULL_WITHIN_PARTITION_GALLERY",
        "mean": "ARITHMETIC_MEAN",
        "standard_deviation": "POPULATION_DDOF_0",
        "near_zero_standard_deviation_threshold": _ZSCORE_STD_EPSILON,
        "finite_values_required": True,
    }
    search_config = {
        "branches": list(_SCORE_FUSION_BRANCHES),
        "constraint": "NONNEGATIVE_SIMPLEX_WEIGHTS_SUM_EXACTLY_ONE",
        "grid_step": grid_step,
        "objective_lexicographic": ["Rank-1", "MRR", "Rank-5"],
        "deterministic_tie_break": [
            "HIGHER_A_RAW_WEIGHT",
            "HIGHER_B_MASKED_WEIGHT",
        ],
        "labels_used_for_search": "CAL_ONLY",
    }
    paired_bootstrap_config = {
        "cluster_unit": "registered_dog_id",
        "interval_method": "whole_identity_percentile_bootstrap",
        "confidence_level": float(bootstrap_confidence_level),
        "resamples": bootstrap_resamples,
        "seed": bootstrap_seed,
    }
    config = {
        "identity_split": split_config,
        "row_zscore": zscore_config,
        "weight_search": search_config,
        "paired_bootstrap": paired_bootstrap_config,
    }
    calibration_index_set = set(calibration_indices)
    assignment = [
        {
            "registered_dog_id": identity,
            "partition": "CAL" if index in calibration_index_set else "EVAL",
        }
        for index, identity in enumerate(identity_list)
    ]
    population = {
        "full_identity_count": len(identity_list),
        "full_registered_dog_ids": identity_list,
        "full_registered_dog_ids_sha256": _identity_list_sha256(identity_list),
        "calibration_identity_count": len(partition_identities["CAL"]),
        "calibration_registered_dog_ids": partition_identities["CAL"],
        "calibration_registered_dog_ids_sha256": _identity_list_sha256(
            partition_identities["CAL"]
        ),
        "evaluation_identity_count": len(partition_identities["EVAL"]),
        "evaluation_registered_dog_ids": partition_identities["EVAL"],
        "evaluation_registered_dog_ids_sha256": _identity_list_sha256(
            partition_identities["EVAL"]
        ),
        "partition_assignment_sha256": content_sha256(assignment),
    }
    return {
        "interpretation": _SCORE_FUSION_INTERPRETATION,
        "config": config,
        "config_sha256": content_sha256(config),
        "config_component_sha256s": {
            "identity_split": content_sha256(split_config),
            "row_zscore": content_sha256(zscore_config),
            "weight_search": content_sha256(search_config),
            "paired_bootstrap": content_sha256(paired_bootstrap_config),
        },
        "population": population,
        "calibration": {
            "search_candidate_count": candidate_count,
            "objective": {
                "optimization": "LEXICOGRAPHIC_MAXIMUM",
                "metrics": ["Rank-1", "MRR", "Rank-5"],
                "selected_values": {
                    metric: selected_metrics[metric]
                    for metric in ("Rank-1", "MRR", "Rank-5")
                },
            },
            "selected_weights": selected_weights,
            "metrics": calibration_metrics,
        },
        "evaluation": {
            "baseline_A_metrics": _metrics(
                evaluation_baseline_rows, len(partition_identities["EVAL"])
            ),
            "fused_metrics": _metrics(
                evaluation_fused_rows, len(partition_identities["EVAL"])
            ),
            "per_identity": per_identity,
            "paired_delta_bootstrap_cis": paired_bootstrap,
        },
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("diagnostic distribution requires finite values")
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "q05": float(np.quantile(array, 0.05)),
        "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _mask_summary(identity_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frames = [
        diagnostic
        for identity in identity_reports
        for role in ("gallery", "query")
        for diagnostic in identity[role]["mask_diagnostics"]
    ]
    return {
        "frame_count": len(frames),
        **{
            field: _distribution([frame[field] for frame in frames])
            for field in (
                "support_fraction",
                "mean_probability",
                "mean_binary_uncertainty",
                "threshold_margin_le_0_05_fraction",
            )
        },
    }


def _code_hashes(repository_root: Path) -> dict[str, str]:
    return {relative: _file_sha256(repository_root / relative) for relative in _CODE_PATHS}


def _require_exact_mapping(
    value: object, expected_keys: set[str], context: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"{context} schema differs")
    return value


def _validate_metric_summary(
    value: object, *, identity_count: int, context: str
) -> dict[str, Any]:
    summary = _require_exact_mapping(
        value, {"query_count", "gallery_count", *_METRICS}, context
    )
    if summary["query_count"] != identity_count or summary["gallery_count"] != identity_count:
        raise ValueError(f"{context} population count differs")
    for metric in _METRICS:
        metric_value = summary[metric]
        if (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or not math.isfinite(float(metric_value))
            or not 0.0 <= float(metric_value) <= 1.0
        ):
            raise ValueError(f"{context}.{metric} must be finite and in [0,1]")
    return summary


def _validate_rank_outcome(
    value: object, *, identity: str, gallery_count: int, context: str
) -> dict[str, Any]:
    outcome = _require_exact_mapping(
        value,
        {
            "registered_dog_id",
            "rank",
            "Rank-1",
            "Rank-5",
            "MRR",
            "mAP",
            "genuine_score",
            "best_impostor_score",
            "genuine_margin",
        },
        context,
    )
    rank = outcome["rank"]
    if (
        outcome["registered_dog_id"] != identity
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 1 <= rank <= gallery_count
    ):
        raise ValueError(f"{context} identity or rank differs")
    expected = {
        "Rank-1": float(rank <= 1),
        "Rank-5": float(rank <= 5),
        "MRR": 1.0 / rank,
        "mAP": 1.0 / rank,
    }
    if any(outcome[metric] != expected[metric] for metric in _METRICS):
        raise ValueError(f"{context} rank metrics differ")
    for name in ("genuine_score", "best_impostor_score", "genuine_margin"):
        metric_value = outcome[name]
        if (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or not math.isfinite(float(metric_value))
        ):
            raise ValueError(f"{context}.{name} must be finite")
    if outcome["genuine_margin"] != (
        outcome["genuine_score"] - outcome["best_impostor_score"]
    ):
        raise ValueError(f"{context} genuine margin differs")
    return outcome


def _validate_score_fusion_stage(
    value: object, *, full_report_identities: Sequence[str]
) -> None:
    stage = _require_exact_mapping(
        value,
        {
            "interpretation",
            "config",
            "config_sha256",
            "config_component_sha256s",
            "population",
            "calibration",
            "evaluation",
        },
        "calibrated score fusion",
    )
    if stage["interpretation"] != _SCORE_FUSION_INTERPRETATION:
        raise ValueError("calibrated score fusion interpretation differs")
    config = _require_exact_mapping(
        stage["config"],
        {"identity_split", "row_zscore", "weight_search", "paired_bootstrap"},
        "calibrated score fusion config",
    )
    _require_sha256(stage["config_sha256"], "calibrated_score_fusion.config_sha256")
    if content_sha256(config) != stage["config_sha256"]:
        raise ValueError("calibrated score fusion config digest differs")
    component_hashes = _require_exact_mapping(
        stage["config_component_sha256s"],
        set(config),
        "calibrated score fusion component hashes",
    )
    for name, component in config.items():
        _require_sha256(component_hashes[name], f"score fusion {name} config SHA-256")
        if content_sha256(component) != component_hashes[name]:
            raise ValueError(f"calibrated score fusion {name} config digest differs")

    split_config = _require_exact_mapping(
        config["identity_split"],
        {
            "algorithm",
            "hash_input",
            "calibration_fraction",
            "calibration_seed",
            "minimum_identities_per_partition",
        },
        "calibrated score fusion identity split config",
    )
    fraction, split_seed, grid_step, grid_units = _score_fusion_parameters(
        split_config["calibration_fraction"],
        split_config["calibration_seed"],
        _require_exact_mapping(
            config["weight_search"],
            {
                "branches",
                "constraint",
                "grid_step",
                "objective_lexicographic",
                "deterministic_tie_break",
                "labels_used_for_search",
            },
            "calibrated score fusion weight search config",
        )["grid_step"],
    )
    if split_config != {
        "algorithm": "SHA256_INTEGER_THRESHOLD_V1",
        "hash_input": "UTF8(str(calibration_seed) + ':' + registered_dog_id)",
        "calibration_fraction": fraction,
        "calibration_seed": split_seed,
        "minimum_identities_per_partition": _MINIMUM_FUSION_PARTITION_IDENTITIES,
    }:
        raise ValueError("calibrated score fusion identity split config differs")
    if config["row_zscore"] != {
        "scope": "EACH_QUERY_ROW_FULL_WITHIN_PARTITION_GALLERY",
        "mean": "ARITHMETIC_MEAN",
        "standard_deviation": "POPULATION_DDOF_0",
        "near_zero_standard_deviation_threshold": _ZSCORE_STD_EPSILON,
        "finite_values_required": True,
    }:
        raise ValueError("calibrated score fusion row z-score config differs")
    if config["weight_search"] != {
        "branches": list(_SCORE_FUSION_BRANCHES),
        "constraint": "NONNEGATIVE_SIMPLEX_WEIGHTS_SUM_EXACTLY_ONE",
        "grid_step": grid_step,
        "objective_lexicographic": ["Rank-1", "MRR", "Rank-5"],
        "deterministic_tie_break": [
            "HIGHER_A_RAW_WEIGHT",
            "HIGHER_B_MASKED_WEIGHT",
        ],
        "labels_used_for_search": "CAL_ONLY",
    }:
        raise ValueError("calibrated score fusion weight search config differs")
    bootstrap_config = _require_exact_mapping(
        config["paired_bootstrap"],
        {"cluster_unit", "interval_method", "confidence_level", "resamples", "seed"},
        "calibrated score fusion bootstrap config",
    )
    if (
        bootstrap_config["cluster_unit"] != "registered_dog_id"
        or bootstrap_config["interval_method"]
        != "whole_identity_percentile_bootstrap"
        or isinstance(bootstrap_config["confidence_level"], bool)
        or not isinstance(bootstrap_config["confidence_level"], (int, float))
        or not 0.0 < float(bootstrap_config["confidence_level"]) < 1.0
        or isinstance(bootstrap_config["resamples"], bool)
        or not isinstance(bootstrap_config["resamples"], int)
        or bootstrap_config["resamples"] <= 0
        or isinstance(bootstrap_config["seed"], bool)
        or not isinstance(bootstrap_config["seed"], int)
        or bootstrap_config["seed"] < 0
    ):
        raise ValueError("calibrated score fusion bootstrap config differs")

    population = _require_exact_mapping(
        stage["population"],
        {
            "full_identity_count",
            "full_registered_dog_ids",
            "full_registered_dog_ids_sha256",
            "calibration_identity_count",
            "calibration_registered_dog_ids",
            "calibration_registered_dog_ids_sha256",
            "evaluation_identity_count",
            "evaluation_registered_dog_ids",
            "evaluation_registered_dog_ids_sha256",
            "partition_assignment_sha256",
        },
        "calibrated score fusion population",
    )
    full_identities = population["full_registered_dog_ids"]
    calibration_identities = population["calibration_registered_dog_ids"]
    evaluation_identities = population["evaluation_registered_dog_ids"]
    if (
        not isinstance(full_identities, list)
        or not isinstance(calibration_identities, list)
        or not isinstance(evaluation_identities, list)
        or full_identities != list(full_report_identities)
        or full_identities != sorted(set(full_identities))
    ):
        raise ValueError("calibrated score fusion full population differs")
    calibration_indices, evaluation_indices = _split_score_fusion_identities(
        full_identities,
        calibration_fraction=fraction,
        calibration_seed=split_seed,
    )
    expected_calibration = [full_identities[index] for index in calibration_indices]
    expected_evaluation = [full_identities[index] for index in evaluation_indices]
    if calibration_identities != expected_calibration or evaluation_identities != expected_evaluation:
        raise ValueError("calibrated score fusion partition membership differs")
    expected_counts = (
        len(full_identities),
        len(calibration_identities),
        len(evaluation_identities),
    )
    if (
        population["full_identity_count"],
        population["calibration_identity_count"],
        population["evaluation_identity_count"],
    ) != expected_counts:
        raise ValueError("calibrated score fusion population counts differ")
    for name, identities in (
        ("full", full_identities),
        ("calibration", calibration_identities),
        ("evaluation", evaluation_identities),
    ):
        digest_name = f"{name}_registered_dog_ids_sha256"
        _require_sha256(population[digest_name], digest_name)
        if _identity_list_sha256(identities) != population[digest_name]:
            raise ValueError(f"calibrated score fusion {name} population digest differs")
    calibration_set = set(calibration_indices)
    assignment = [
        {
            "registered_dog_id": identity,
            "partition": "CAL" if index in calibration_set else "EVAL",
        }
        for index, identity in enumerate(full_identities)
    ]
    _require_sha256(
        population["partition_assignment_sha256"], "partition_assignment_sha256"
    )
    if content_sha256(assignment) != population["partition_assignment_sha256"]:
        raise ValueError("calibrated score fusion partition assignment digest differs")

    calibration = _require_exact_mapping(
        stage["calibration"],
        {"search_candidate_count", "objective", "selected_weights", "metrics"},
        "calibrated score fusion calibration result",
    )
    if calibration["search_candidate_count"] != (grid_units + 1) * (grid_units + 2) // 2:
        raise ValueError("calibrated score fusion search candidate count differs")
    weights = _require_exact_mapping(
        calibration["selected_weights"],
        set(_SCORE_FUSION_BRANCHES),
        "calibrated score fusion selected weights",
    )
    for branch, weight in weights.items():
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
            or not math.isclose(
                float(weight) / grid_step,
                round(float(weight) / grid_step),
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        ):
            raise ValueError(f"calibrated score fusion {branch} weight differs from grid")
    if sum(weights.values()) != 1.0:
        raise ValueError("calibrated score fusion selected weights do not sum exactly one")
    calibration_metrics = _require_exact_mapping(
        calibration["metrics"],
        {*_SCORE_FUSION_BRANCHES, "selected_score_fusion"},
        "calibrated score fusion calibration metrics",
    )
    for name, summary in calibration_metrics.items():
        _validate_metric_summary(
            summary,
            identity_count=len(calibration_identities),
            context=f"calibrated score fusion calibration metrics {name}",
        )
    objective = _require_exact_mapping(
        calibration["objective"],
        {"optimization", "metrics", "selected_values"},
        "calibrated score fusion objective",
    )
    if (
        objective["optimization"] != "LEXICOGRAPHIC_MAXIMUM"
        or objective["metrics"] != ["Rank-1", "MRR", "Rank-5"]
    ):
        raise ValueError("calibrated score fusion objective differs")
    selected_values = _require_exact_mapping(
        objective["selected_values"],
        {"Rank-1", "MRR", "Rank-5"},
        "calibrated score fusion selected objective",
    )
    if selected_values != {
        metric: calibration_metrics["selected_score_fusion"][metric]
        for metric in ("Rank-1", "MRR", "Rank-5")
    }:
        raise ValueError("calibrated score fusion selected objective values differ")

    evaluation = _require_exact_mapping(
        stage["evaluation"],
        {
            "baseline_A_metrics",
            "fused_metrics",
            "per_identity",
            "paired_delta_bootstrap_cis",
        },
        "calibrated score fusion evaluation result",
    )
    for name in ("baseline_A_metrics", "fused_metrics"):
        _validate_metric_summary(
            evaluation[name],
            identity_count=len(evaluation_identities),
            context=f"calibrated score fusion evaluation {name}",
        )
    per_identity = evaluation["per_identity"]
    if (
        not isinstance(per_identity, list)
        or [row.get("registered_dog_id") for row in per_identity]
        != evaluation_identities
    ):
        raise ValueError("calibrated score fusion evaluation identity rows differ")
    for row in per_identity:
        detail = _require_exact_mapping(
            row,
            {
                "registered_dog_id",
                "baseline_A_outcome",
                "fused_outcome",
                "fused_minus_A_delta",
            },
            "calibrated score fusion per-identity row",
        )
        baseline_outcome = _validate_rank_outcome(
            detail["baseline_A_outcome"],
            identity=detail["registered_dog_id"],
            gallery_count=len(evaluation_identities),
            context="calibrated score fusion baseline A outcome",
        )
        fused_outcome = _validate_rank_outcome(
            detail["fused_outcome"],
            identity=detail["registered_dog_id"],
            gallery_count=len(evaluation_identities),
            context="calibrated score fusion fused outcome",
        )
        if detail["fused_minus_A_delta"] != _delta(
            fused_outcome, baseline_outcome
        ):
            raise ValueError("calibrated score fusion per-identity delta differs")
    expected_baseline_metrics = _metrics(
        [row["baseline_A_outcome"] for row in per_identity],
        len(evaluation_identities),
    )
    expected_fused_metrics = _metrics(
        [row["fused_outcome"] for row in per_identity],
        len(evaluation_identities),
    )
    if (
        evaluation["baseline_A_metrics"] != expected_baseline_metrics
        or evaluation["fused_metrics"] != expected_fused_metrics
    ):
        raise ValueError("calibrated score fusion evaluation metric summaries differ")
    intervals = _require_exact_mapping(
        evaluation["paired_delta_bootstrap_cis"],
        set(_METRICS),
        "calibrated score fusion paired bootstrap intervals",
    )
    for metric, interval_value in intervals.items():
        interval = _require_exact_mapping(
            interval_value,
            {
                "metric",
                "estimate",
                "lower_bound",
                "upper_bound",
                "confidence_level",
                "cluster_unit",
                "cluster_count",
                "query_row_count",
                "resamples",
                "seed",
                "interval_method",
            },
            f"calibrated score fusion {metric} bootstrap interval",
        )
        expected_estimate = float(
            np.mean([row["fused_minus_A_delta"][metric] for row in per_identity])
        )
        if (
            interval["metric"] != metric
            or interval["estimate"] != expected_estimate
            or interval["cluster_count"] != len(evaluation_identities)
            or interval["query_row_count"] != len(evaluation_identities)
            or interval["confidence_level"] != bootstrap_config["confidence_level"]
            or interval["resamples"] != bootstrap_config["resamples"]
            or interval["seed"] != bootstrap_config["seed"]
            or interval["interval_method"] != bootstrap_config["interval_method"]
            or interval["cluster_unit"] != "query_identity"
            or any(
                isinstance(interval[name], bool)
                or not isinstance(interval[name], (int, float))
                or not math.isfinite(float(interval[name]))
                for name in ("lower_bound", "upper_bound")
            )
            or interval["lower_bound"] > interval["upper_bound"]
        ):
            raise ValueError(f"calibrated score fusion {metric} bootstrap interval differs")


def validate_report_bundle(bundle: object) -> dict[str, Any]:
    if not isinstance(bundle, dict) or set(bundle) != {
        "schema_version",
        "report_sha256",
        "report",
    }:
        raise ValueError("architecture evaluation report bundle schema differs")
    if bundle["schema_version"] != REPORT_BUNDLE_SCHEMA:
        raise ValueError("architecture evaluation report bundle version differs")
    _require_sha256(bundle["report_sha256"], "report_sha256")
    report = bundle["report"]
    expected = {
        "schema_version",
        "status",
        "interpretation",
        "input_bindings",
        "input_sha256s",
        "code_sha256s",
        "protocol",
        "population",
        "methods",
        "metrics",
        "mask_diagnostics",
        "paired_per_identity",
        "paired_delta_bootstrap_cis",
        "calibrated_score_fusion",
    }
    if not isinstance(report, dict) or set(report) != expected:
        raise ValueError("architecture evaluation report schema differs")
    if report["schema_version"] != REPORT_SCHEMA or report["interpretation"] != INTERPRETATION:
        raise ValueError("architecture evaluation interpretation differs")
    if report["methods"] != METHODS:
        raise ValueError("architecture evaluation methods differ")
    protocol = report["protocol"]
    population = report["population"]
    if not isinstance(protocol, dict) or not isinstance(population, dict):
        raise ValueError("architecture evaluation protocol or population differs")
    population_role = protocol.get("population_role")
    if population_role not in _POPULATION_ROLES:
        raise ValueError("architecture evaluation population role differs")

    binding_keys = {
        "native_manifest_bundle",
        "native_artifact_root",
        "embedding_runtime_manifest",
        "embedding_onnx",
        "mask_runtime_manifest",
        "mask_onnx",
        "population_selection",
    }
    input_sha256_keys = {
        "native_manifest_bundle_content",
        "native_manifest",
        "embedding_manifest_content",
        "embedding_onnx",
        "mask_manifest_content",
        "mask_onnx",
        "selected_registered_dog_ids",
    }
    if population_role == "consistency_eval":
        binding_keys.add("embedding_lineage")
        input_sha256_keys.update(
            {
                "embedding_lineage_raw",
                "embedding_lineage_content",
                "embedding_lineage",
            }
        )
    input_bindings = _require_exact_mapping(
        report["input_bindings"], binding_keys, "architecture evaluation input bindings"
    )
    input_sha256s = _require_exact_mapping(
        report["input_sha256s"],
        input_sha256_keys,
        "architecture evaluation input SHA-256s",
    )
    for name, digest in input_sha256s.items():
        _require_sha256(digest, f"input_sha256s.{name}")
    for name, digest in report["code_sha256s"].items():
        _require_sha256(digest, f"code_sha256s.{name}")
    for name in (
        "pairing_sha256",
        "mask_background_transform_sha256",
        "restoration_config_sha256",
    ):
        _require_sha256(protocol[name], name)
    paired = report["paired_per_identity"]
    if not isinstance(paired, list):
        raise ValueError("architecture evaluation paired identity rows differ")
    identities = [row.get("registered_dog_id") for row in paired if isinstance(row, dict)]
    if (
        len(identities) != len(paired)
        or any(not isinstance(identity, str) for identity in identities)
        or identities != sorted(set(identities))
    ):
        raise ValueError("architecture evaluation selected identities differ")
    for index, identity in enumerate(identities):
        _canonical_uuid5(identity, f"architecture evaluation identity {index}")
    identity_list_sha256 = _identity_list_sha256(identities)
    expected_population_keys = {
        "selected_population_role",
        "selected_identity_count",
        "selected_registered_dog_ids_sha256",
        "localized_identity_count",
        "eligible_identity_count",
        "excluded_identity_counts",
        "gallery_vector_count_per_method",
        "query_vector_count_per_method",
    }
    _require_exact_mapping(
        population, expected_population_keys, "architecture evaluation population"
    )
    if (
        population["selected_population_role"] != population_role
        or protocol.get("selected_identity_count") != len(identities)
        or population["selected_identity_count"] != len(identities)
        or population["eligible_identity_count"] != len(identities)
        or population["gallery_vector_count_per_method"] != len(identities)
        or population["query_vector_count_per_method"] != len(identities)
        or protocol.get("selected_registered_dog_ids_sha256")
        != identity_list_sha256
        or population["selected_registered_dog_ids_sha256"]
        != identity_list_sha256
        or input_sha256s["selected_registered_dog_ids"] != identity_list_sha256
    ):
        raise ValueError("architecture evaluation selected population binding differs")
    selection_binding = _require_exact_mapping(
        input_bindings["population_selection"],
        {"role", "identity_count", "registered_dog_ids_sha256"},
        "architecture evaluation population selection binding",
    )
    if selection_binding != {
        "role": population_role,
        "identity_count": len(identities),
        "registered_dog_ids_sha256": identity_list_sha256,
    }:
        raise ValueError("architecture evaluation population selection binding differs")

    expected_input_digests = {
        "native_manifest_bundle_content": input_bindings["native_manifest_bundle"][
            "content_sha256"
        ],
        "native_manifest": input_bindings["native_manifest_bundle"]["manifest_sha256"],
        "embedding_manifest_content": input_bindings["embedding_runtime_manifest"][
            "content_sha256"
        ],
        "embedding_onnx": input_bindings["embedding_onnx"]["sha256"],
        "mask_manifest_content": input_bindings["mask_runtime_manifest"][
            "content_sha256"
        ],
        "mask_onnx": input_bindings["mask_onnx"]["sha256"],
        "selected_registered_dog_ids": identity_list_sha256,
    }
    if population_role == "consistency_eval":
        if len(identities) < 20:
            raise ValueError("consistency EVAL population must contain at least 20 identities")
        lineage_binding = _require_exact_mapping(
            input_bindings["embedding_lineage"],
            {
                "path",
                "parent_root",
                "raw_sha256",
                "content_sha256",
                "lineage_sha256",
                "byte_size",
                "eval_identity_count",
                "eval_registered_dog_ids_sha256",
            },
            "architecture evaluation embedding lineage binding",
        )
        for name in ("raw_sha256", "content_sha256", "lineage_sha256"):
            _require_sha256(lineage_binding[name], f"embedding_lineage.{name}")
        if (
            not isinstance(lineage_binding["path"], str)
            or not lineage_binding["path"]
            or not isinstance(lineage_binding["parent_root"], str)
            or not lineage_binding["parent_root"]
            or isinstance(lineage_binding["byte_size"], bool)
            or not isinstance(lineage_binding["byte_size"], int)
            or lineage_binding["byte_size"] <= 0
            or lineage_binding["eval_identity_count"] != len(identities)
            or lineage_binding["eval_registered_dog_ids_sha256"]
            != identity_list_sha256
        ):
            raise ValueError("architecture evaluation embedding lineage binding differs")
        expected_input_digests.update(
            {
                "embedding_lineage_raw": lineage_binding["raw_sha256"],
                "embedding_lineage_content": lineage_binding["content_sha256"],
                "embedding_lineage": lineage_binding["lineage_sha256"],
            }
        )
    if input_sha256s != expected_input_digests:
        raise ValueError("architecture evaluation input SHA-256 bindings differ")
    _validate_score_fusion_stage(
        report["calibrated_score_fusion"],
        full_report_identities=identities,
    )
    if content_sha256(report) != bundle["report_sha256"]:
        raise ValueError("architecture evaluation report digest differs")
    return report


def evaluate_nose_architectures(
    *,
    native_bundle_path: Path,
    native_bundle_sha256: str,
    native_root: Path,
    embedding_manifest_path: Path,
    embedding_manifest_sha256: str,
    embedding_onnx_path: Path,
    mask_manifest_path: Path,
    mask_manifest_sha256: str,
    mask_onnx_path: Path,
    output_path: Path,
    use_cuda: bool = False,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_confidence_level: float = 0.95,
    calibration_fraction: float = 0.3,
    calibration_seed: int = 73,
    fusion_grid_step: float = 0.05,
    repository_root: Path | None = None,
    embedding_lineage_path: Path | None = None,
    embedding_lineage_sha256: str | None = None,
    population_role: str = "all",
) -> dict[str, Any]:
    """Evaluate six architectures plus leakage-safe calibrated score fusion."""

    expected_native = _require_sha256(native_bundle_sha256, "native_bundle_sha256")
    expected_embedding = _require_sha256(
        embedding_manifest_sha256, "embedding_manifest_sha256"
    )
    expected_mask = _require_sha256(mask_manifest_sha256, "mask_manifest_sha256")
    expected_lineage = _validate_population_role_inputs(
        population_role, embedding_lineage_path, embedding_lineage_sha256
    )
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples <= 0
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if (
        isinstance(bootstrap_confidence_level, bool)
        or not isinstance(bootstrap_confidence_level, (int, float))
        or not 0.0 < float(bootstrap_confidence_level) < 1.0
    ):
        raise ValueError("bootstrap_confidence_level must be in (0,1)")
    _score_fusion_parameters(
        calibration_fraction, calibration_seed, fusion_grid_step
    )

    repository = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    output_absolute = Path(os.path.abspath(os.fspath(output_path)))
    resolved_output = output_absolute.parent.resolve(strict=True) / output_absolute.name
    if resolved_output.is_relative_to(repository):
        raise ValueError("evaluation report must be written outside the Git repository")
    if resolved_output.exists() or resolved_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evaluation report: {resolved_output}")

    native_document = read_strict_json_document(
        native_bundle_path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    if native_document.canonical_payload_sha256 != expected_native:
        raise ValueError("native manifest bundle content SHA-256 differs from the external pin")
    root = native_root.resolve(strict=True)
    native_manifest = validate_manifest_bundle(native_document.payload, root=root)

    embedding_document = read_strict_json_document(embedding_manifest_path)
    if embedding_document.canonical_payload_sha256 != expected_embedding:
        raise ValueError("embedding manifest content SHA-256 differs from the external pin")
    embedding_manifest = NoseEmbeddingManifest.from_dict(embedding_document.payload)
    mask_document = read_strict_json_document(mask_manifest_path)
    if mask_document.canonical_payload_sha256 != expected_mask:
        raise ValueError("mask manifest content SHA-256 differs from the external pin")
    mask_manifest = NoseMaskManifest.from_dict(mask_document.payload)
    for name, manifest in (
        ("embedding", embedding_manifest),
        ("mask", mask_manifest),
    ):
        if manifest.license.usage_lane not in {UsageLane.RESEARCH_ONLY, UsageLane.TEST_FIXTURE}:
            raise ArtifactContractError(f"{name} artifact is not admitted to the research-only lane")
    if embedding_manifest.input_shape[2:] != (_IMAGE_SIZE, _IMAGE_SIZE):
        raise ArtifactContractError("embedding manifest input must be exactly 224x224")
    if mask_manifest.input_shape[2:] != (_IMAGE_SIZE, _IMAGE_SIZE) or mask_manifest.output_shape[
        2:
    ] != (_IMAGE_SIZE, _IMAGE_SIZE):
        raise ArtifactContractError("mask manifest input and output must be exactly 224x224")
    if mask_manifest.preprocessing.resize != "bilinear":
        raise ArtifactContractError(
            "mask-aware RGB branch requires the declared bilinear mask contract"
        )

    embedding_onnx_binding = _file_binding(embedding_onnx_path)
    mask_onnx_binding = _file_binding(mask_onnx_path)
    if embedding_onnx_binding["sha256"] != embedding_manifest.artifact_sha256:
        raise ArtifactContractError("embedding artifact SHA256 does not match its manifest")
    if mask_onnx_binding["sha256"] != mask_manifest.artifact_sha256:
        raise ArtifactContractError("mask artifact SHA256 does not match its manifest")

    lineage_document = None
    lineage_eval_identities = None
    if population_role == "consistency_eval":
        if embedding_lineage_path is None or expected_lineage is None:
            raise RuntimeError("validated consistency lineage inputs disappeared")
        lineage_document, lineage_eval_identities = _load_consistency_lineage(
            lineage_path=embedding_lineage_path,
            expected_content_sha256=expected_lineage,
            embedding_manifest_path=embedding_manifest_path,
            embedding_document=embedding_document,
            embedding_onnx_path=embedding_onnx_path,
            embedding_onnx_binding=embedding_onnx_binding,
        )

    embedding_runtime = ExactOnnxRuntime(
        embedding_onnx_path, embedding_manifest, use_cuda=use_cuda
    )
    mask_runtime = ExactOnnxRuntime(mask_onnx_path, mask_manifest, use_cuda=use_cuda)

    restoration_config = RestorationConfig(
        registration_mode="canonical_crop_identity",
        illumination_normalization=False,
    )
    mask_transform = {
        "input": "mask_contract_bilinear_resized_224x224_RGB_frame",
        "support": "student_probability_greater_than_or_equal_to_manifest_threshold",
        "inside_support": "retain_original_RGB",
        "outside_support": (
            "0.25_times_original_RGB_plus_0.75_times_per_image_per_channel_RGB_median_"
            "then_round_to_uint8"
        ),
        "outside_support_original_weight": _OUTSIDE_SUPPORT_ORIGINAL_WEIGHT,
        "median_axes": ["height", "width"],
    }

    population, excluded = _group_population(native_manifest["records"])
    if population_role == "consistency_eval":
        population = _filter_consistency_eval_population(
            population, lineage_eval_identities
        )
    identities = [item["registered_dog_id"] for item in population]
    _split_score_fusion_identities(
        identities,
        calibration_fraction=float(calibration_fraction),
        calibration_seed=calibration_seed,
    )
    vectors: dict[str, dict[str, list[np.ndarray]]] = {
        method: {"gallery": [], "query": []} for method in METHODS
    }
    identity_reports: list[dict[str, Any]] = []
    for item in population:
        identity_report: dict[str, Any] = {
            key: item[key]
            for key in (
                "registered_dog_id",
                "identity_token",
                "track_token",
                "sequence_token",
                "localized_frame_count",
            )
        }
        for role in ("gallery", "query"):
            rows = item[role]
            source_images = [_load_crop_rgb(root, row) for row in rows]
            images = np.stack(
                [_resize_mask_branch_rgb(image) for image in source_images]
            )
            raw_embeddings = [
                _embed(embedding_runtime, embedding_manifest, image)
                for image in source_images
            ]
            masked_images: list[np.ndarray] = []
            source_masks: list[np.ndarray] = []
            mask_diagnostics: list[dict[str, Any]] = []
            for image in images:
                masked, support, diagnostic = _student_mask(
                    mask_runtime, mask_manifest, image
                )
                masked_images.append(masked)
                source_masks.append(support)
                mask_diagnostics.append(diagnostic)
            masked_embeddings = [
                _embed(embedding_runtime, embedding_manifest, image)
                for image in masked_images
            ]
            raw_k5, raw_temporal = _fuse_k5(raw_embeddings)
            masked_k5, masked_temporal = _fuse_k5(masked_embeddings)
            restoration = restore_nose_frames(
                images,
                np.stack(source_masks),
                config=restoration_config,
            )
            restored_uint8 = np.rint(
                np.clip(restoration.restored_rgb, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            restored = _embed(embedding_runtime, embedding_manifest, restored_uint8)
            role_vectors = {
                "A_raw_K5": raw_k5,
                "B_student_masked_K5": masked_k5,
                "C_raw_plus_masked_late_fusion": _l2_mean(
                    (raw_k5, masked_k5), "raw/masked late fusion"
                ),
                "D_restored_early_fusion": restored,
                "E_raw_plus_restored_late_fusion": _l2_mean(
                    (raw_k5, restored), "raw/restored late fusion"
                ),
                "F_all_branches_late_fusion": _l2_mean(
                    (raw_k5, masked_k5, restored), "all-branch late fusion"
                ),
            }
            for method, vector in role_vectors.items():
                vectors[method][role].append(vector)
            identity_report[role] = {
                "sample_tokens": [row["sample_token"] for row in rows],
                "frame_indices": [row["frame_index"] for row in rows],
                "mask_diagnostics": mask_diagnostics,
                "raw_K5_diagnostics": raw_temporal,
                "student_masked_K5_diagnostics": masked_temporal,
                "restoration_diagnostics": restoration.diagnostics.to_dict(),
            }
        identity_reports.append(identity_report)

    metrics: dict[str, Any] = {}
    outcomes: dict[str, list[dict[str, Any]]] = {}
    score_matrices: dict[str, np.ndarray] = {}
    for method in METHODS:
        gallery = np.stack(vectors[method]["gallery"])
        query = np.stack(vectors[method]["query"])
        if gallery.shape != query.shape or gallery.shape[0] != len(identities):
            raise RuntimeError("evaluation did not produce one paired vector per identity and role")
        scores = compute_cosine_score_matrix(query, gallery)
        if method in _SCORE_FUSION_BRANCHES:
            score_matrices[method] = scores
        outcomes[method] = _rank_rows(scores, identities)
        metrics[method] = _metrics(outcomes[method], len(identities))

    calibrated_score_fusion = evaluate_calibrated_score_fusion(
        identities,
        score_matrices,
        calibration_fraction=calibration_fraction,
        calibration_seed=calibration_seed,
        fusion_grid_step=fusion_grid_step,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence_level=bootstrap_confidence_level,
    )

    paired_rows: dict[str, list[dict[str, Any]]] = {
        f"{method}_minus_A": [] for method in METHODS if method != _BASELINE
    }
    for index, identity_report in enumerate(identity_reports):
        identity_report["method_outcomes"] = {
            method: outcomes[method][index] for method in METHODS
        }
        identity_report["paired_deltas_against_A"] = {}
        for method in METHODS:
            if method == _BASELINE:
                continue
            contrast = f"{method}_minus_A"
            delta = _delta(outcomes[method][index], outcomes[_BASELINE][index])
            identity_report["paired_deltas_against_A"][contrast] = delta
            paired_rows[contrast].append(
                {"bootstrap_cluster_id": identities[index], **delta}
            )

    bootstrap_cis = {
        contrast: {
            metric: identity_clustered_bootstrap_ci(
                rows,
                metric=metric,
                confidence_level=float(bootstrap_confidence_level),
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
            for metric in _METRICS
        }
        for contrast, rows in paired_rows.items()
    }
    pairing_contract = [
        {
            "registered_dog_id": report["registered_dog_id"],
            "identity_token": report["identity_token"],
            "track_token": report["track_token"],
            "sequence_token": report["sequence_token"],
            "gallery_sample_tokens": report["gallery"]["sample_tokens"],
            "query_sample_tokens": report["query"]["sample_tokens"],
        }
        for report in identity_reports
    ]
    input_sha256s = {
        "native_manifest_bundle_content": native_document.canonical_payload_sha256,
        "native_manifest": native_document.payload["manifest_sha256"],
        "embedding_manifest_content": embedding_document.canonical_payload_sha256,
        "embedding_onnx": embedding_onnx_binding["sha256"],
        "mask_manifest_content": mask_document.canonical_payload_sha256,
        "mask_onnx": mask_onnx_binding["sha256"],
        "selected_registered_dog_ids": _identity_list_sha256(identities),
    }
    input_bindings = {
        "native_manifest_bundle": {
            "path": os.fspath(native_bundle_path),
            "raw_sha256": native_document.raw_sha256,
            "content_sha256": native_document.canonical_payload_sha256,
            "manifest_sha256": native_document.payload["manifest_sha256"],
            "byte_size": native_document.byte_size,
        },
        "native_artifact_root": os.fspath(root),
        "embedding_runtime_manifest": {
            "path": os.fspath(embedding_manifest_path),
            "raw_sha256": embedding_document.raw_sha256,
            "content_sha256": embedding_document.canonical_payload_sha256,
            "byte_size": embedding_document.byte_size,
        },
        "embedding_onnx": embedding_onnx_binding,
        "mask_runtime_manifest": {
            "path": os.fspath(mask_manifest_path),
            "raw_sha256": mask_document.raw_sha256,
            "content_sha256": mask_document.canonical_payload_sha256,
            "byte_size": mask_document.byte_size,
        },
        "mask_onnx": mask_onnx_binding,
        "population_selection": {
            "role": population_role,
            "identity_count": len(identities),
            "registered_dog_ids_sha256": _identity_list_sha256(identities),
        },
    }
    if lineage_document is not None:
        input_bindings["embedding_lineage"] = {
            "path": os.fspath(embedding_lineage_path),
            "parent_root": os.fspath(embedding_lineage_path.parent.resolve(strict=True)),
            "raw_sha256": lineage_document.raw_sha256,
            "content_sha256": lineage_document.canonical_payload_sha256,
            "lineage_sha256": lineage_document.payload["lineage_sha256"],
            "byte_size": lineage_document.byte_size,
            "eval_identity_count": len(lineage_eval_identities),
            "eval_registered_dog_ids_sha256": _identity_list_sha256(
                lineage_eval_identities
            ),
        }
        input_sha256s.update(
            {
                "embedding_lineage_raw": lineage_document.raw_sha256,
                "embedding_lineage_content": lineage_document.canonical_payload_sha256,
                "embedding_lineage": lineage_document.payload["lineage_sha256"],
            }
        )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_WITHIN_VIDEO_TRACK_RESEARCH_DIAGNOSTIC",
        "interpretation": INTERPRETATION,
        "input_bindings": input_bindings,
        "input_sha256s": input_sha256s,
        "code_sha256s": _code_hashes(repository),
        "protocol": {
            "population_role": population_role,
            "selected_identity_count": len(identities),
            "selected_registered_dog_ids_sha256": _identity_list_sha256(identities),
            "grouping_unit": "registered_dog_id_with_one_to_one_YT_identity_track_sequence",
            "localized_states": ["AVAILABLE", "LOW_QUALITY"],
            "minimum_localized_frames": 10,
            "fixed_common_population_across_methods": True,
            "temporal_order": ["frame_index", "sample_token"],
            "gallery_selection": "earliest_five",
            "query_selection": "latest_five",
            "frame_overlap_allowed": False,
            "frames_per_role": 5,
            "image_size": [_IMAGE_SIZE, _IMAGE_SIZE],
            "raw_crop_preprocessing": (
                "original_crop_directly_through_embedding_manifest_bicubic_contract"
            ),
            "mask_and_restoration_crop_resize": (
                "PIL_bilinear_RGB_to_224_matching_mask_training_contract"
            ),
            "raw_and_masked_temporal_fusion": (
                "L2_normalize_each_of_five_embeddings_then_unweighted_arithmetic_mean_"
                "then_L2_normalize"
            ),
            "branch_late_fusion": (
                "L2_normalize_each_branch_vector_then_unweighted_arithmetic_mean_then_"
                "L2_normalize"
            ),
            "mask_threshold": mask_manifest.threshold,
            "mask_output_semantics": "SIGMOID_PROBABILITY_IN_[0,1]",
            "mask_background_transform": mask_transform,
            "mask_background_transform_sha256": content_sha256(mask_transform),
            "restoration_source_masks": "student_probability_greater_than_or_equal_to_threshold",
            "restoration_config": restoration_config.to_dict(),
            "restoration_config_sha256": content_sha256(restoration_config.to_dict()),
            "retrieval": "exhaustive_cosine_one_gallery_vector_per_identity",
            "tie_policy": "stable_registered_dog_id_lexical_order",
            "pairing_sha256": content_sha256(pairing_contract),
            "bootstrap": {
                "cluster_unit": "registered_dog_id",
                "interval_method": "whole_identity_percentile_bootstrap",
                "confidence_level": float(bootstrap_confidence_level),
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
                "config_sha256": content_sha256(
                    {
                        "cluster_unit": "registered_dog_id",
                        "confidence_level": float(bootstrap_confidence_level),
                        "resamples": bootstrap_resamples,
                        "seed": bootstrap_seed,
                    }
                ),
            },
        },
        "population": {
            "selected_population_role": population_role,
            "selected_identity_count": len(identities),
            "selected_registered_dog_ids_sha256": _identity_list_sha256(identities),
            "localized_identity_count": len(
                {
                    row["registered_dog_id"]
                    for row in native_manifest["records"]
                    if row["record_state"] != "NO_ROI"
                }
            ),
            "eligible_identity_count": len(identities),
            "excluded_identity_counts": excluded,
            "gallery_vector_count_per_method": len(identities),
            "query_vector_count_per_method": len(identities),
        },
        "methods": METHODS,
        "metrics": metrics,
        "mask_diagnostics": _mask_summary(identity_reports),
        "paired_per_identity": identity_reports,
        "paired_delta_bootstrap_cis": bootstrap_cis,
        "calibrated_score_fusion": calibrated_score_fusion,
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    bundle = {
        "schema_version": REPORT_BUNDLE_SCHEMA,
        "report_sha256": content_sha256(report),
        "report": report,
    }
    validate_report_bundle(bundle)
    write_private_json_bundle(((resolved_output, bundle),))
    return bundle


__all__ = [
    "INTERPRETATION",
    "METHODS",
    "REPORT_BUNDLE_SCHEMA",
    "REPORT_SCHEMA",
    "evaluate_calibrated_score_fusion",
    "evaluate_nose_architectures",
    "validate_report_bundle",
]
