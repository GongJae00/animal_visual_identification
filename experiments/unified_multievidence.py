"""Research-only same-track Appearance/Face/Nose evaluation."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from contracts.artifact_manifest import (
    ArtifactContractError,
    ExactOnnxRuntime,
    NoseEmbeddingManifest,
    UsageLane,
    preprocess_image,
)
from evaluation.retrieval import (
    compute_cosine_score_matrix,
    identity_clustered_bootstrap_ci,
)
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.provenance import content_sha256
from embedding.methods.appearance import ReceiptBoundDinov2Small
from embedding.methods.nose.signal.temporal import aggregate_nose_embeddings
from parsing.nose_region.native_yt import validate_manifest_bundle
from parsing.roi_manifest import read_roi_manifest

REPORT_SCHEMA = "cvi.yt_unified_multievidence_evaluation.v2"
REPORT_BUNDLE_SCHEMA = "cvi.yt_unified_multievidence_evaluation_bundle.v2"
INTERPRETATION = (
    "WITHIN_VIDEO_TRACK_FROZEN_APPEARANCE_FACE_NOSE_RESEARCH_DIAGNOSTIC_"
    "NOT_CROSS_SESSION_BIOMETRIC_VALIDATION_OR_FINAL_EVALUATION"
)
METHODS = {
    "A0_frozen_dinov2_K5": "frozen DINOv2-small on original publisher dog crops",
    "F0_frozen_dinov2_K5": "frozen DINOv2-small on identity-bound face ROI crops",
    "N3_consistency_raw_K5": "consistency-v3 Nose embedding on native Nose crops",
    "A0_plus_F0": "DEV-selected row-z-score simplex fusion of A0 and F0",
    "A0_plus_N3": "DEV-selected row-z-score simplex fusion of A0 and N3",
    "F0_plus_N3": "DEV-selected row-z-score simplex fusion of F0 and N3",
    "A0_plus_F0_plus_N3": "DEV-selected row-z-score simplex fusion of A0, F0, and N3",
}
_BRANCHES = tuple(list(METHODS)[:3])
_FUSIONS = {
    "A0_plus_F0": _BRANCHES[:2],
    "A0_plus_N3": (_BRANCHES[0], _BRANCHES[2]),
    "F0_plus_N3": _BRANCHES[1:],
    "A0_plus_F0_plus_N3": _BRANCHES,
}
_METRICS = ("Rank-1", "Rank-5", "MRR", "mAP")
_CODE_PATHS = (
    "experiments/unified_multievidence.py",
    "evaluation/retrieval.py",
    "embedding/methods/appearance/__init__.py",
    "contracts/artifact_manifest.py",
    "parsing/roi_manifest.py",
    "parsing/nose_region/native_yt.py",
    "embedding/methods/nose/training/embedding_consistency_training.py",
    "embedding/methods/nose/signal/temporal.py",
    "workflows/evaluate_yt_unified_multievidence.py",
)
_PRE_TRAINING_OWNERSHIP_CODE_PATHS = tuple(
    "parsing/nose_region/embedding_consistency_training.py"
    if path == "embedding/methods/nose/training/embedding_consistency_training.py"
    else path
    for path in _CODE_PATHS
)
_PRE_NESTED_EMBEDDING_CODE_PATHS = tuple(
    path.replace("embedding/methods/nose/signal/", "embedding/methods/nose/", 1)
    if path.startswith("embedding/methods/nose/signal/")
    else path
    for path in _PRE_TRAINING_OWNERSHIP_CODE_PATHS
)
_PRE_EMBEDDING_CODE_PATHS = tuple(
    path.replace("embedding/methods/", "identity_methods/", 1)
    if path.startswith("embedding/methods/")
    else path
    for path in _PRE_NESTED_EMBEDDING_CODE_PATHS
)
_LEGACY_CODE_PATHS = tuple(
    path.replace("parsing/", "localization/", 1)
    if path.startswith("parsing/")
    else path
    for path in _PRE_EMBEDDING_CODE_PATHS
)
_ZSCORE_EPSILON = 1e-8


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


def _identity_list_sha256(identities: Sequence[str]) -> str:
    return content_sha256({"registered_dog_ids": list(identities)})


def _normalize(vector: np.ndarray, context: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if value.ndim != 1 or not np.isfinite(value).all() or not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"{context} produced a non-finite or zero-norm vector")
    return np.asarray(value / norm, dtype=np.float32)


def _k5(vectors: Sequence[np.ndarray], context: str) -> np.ndarray:
    if len(vectors) != 5:
        raise ValueError(f"{context} requires exactly five embeddings")
    result = aggregate_nose_embeddings([_normalize(vector, context) for vector in vectors])
    if result.aggregation != "UNWEIGHTED_L2_MEAN":
        raise RuntimeError("K=5 aggregation contract differs")
    return result.embedding


def _rank_rows(scores: np.ndarray, identities: Sequence[str]) -> list[dict[str, Any]]:
    matrix = np.asarray(scores, dtype=np.float64)
    expected = (len(identities), len(identities))
    if matrix.shape != expected or not np.isfinite(matrix).all():
        raise ValueError(f"score matrix must be finite with shape {expected}")
    rows: list[dict[str, Any]] = []
    for index, identity in enumerate(identities):
        order = np.argsort(-matrix[index], kind="stable")
        rank = int(np.flatnonzero(order == index)[0]) + 1
        impostors = np.delete(matrix[index], index)
        genuine = float(matrix[index, index])
        best_impostor = float(impostors.max())
        rows.append(
            {
                "registered_dog_id": identity,
                "rank": rank,
                "Rank-1": float(rank == 1),
                "Rank-5": float(rank <= 5),
                "MRR": 1.0 / rank,
                "mAP": 1.0 / rank,
                "genuine_score": genuine,
                "best_impostor_score": best_impostor,
                "genuine_margin": genuine - best_impostor,
            }
        )
    return rows


def _metrics(rows: Sequence[Mapping[str, Any]], identity_count: int) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "gallery_count": identity_count,
        **{metric: float(np.mean([row[metric] for row in rows])) for metric in _METRICS},
    }


def _row_zscores(scores: np.ndarray, context: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError(f"{context} score matrix must be square")
    if not np.isfinite(values).all():
        raise ValueError(f"{context} score matrix must be finite")
    means = values.mean(axis=1, keepdims=True)
    standard_deviations = values.std(axis=1, ddof=0, keepdims=True)
    if np.any(standard_deviations <= _ZSCORE_EPSILON):
        raise ValueError(f"{context} has a near-zero row standard deviation")
    return (values - means) / standard_deviations


def _simplex_grid(channels: int, resolution: int):
    if channels == 2:
        for first in range(resolution + 1):
            yield np.asarray((first, resolution - first), dtype=np.float64) / resolution
        return
    for first in range(resolution + 1):
        for second in range(resolution - first + 1):
            yield np.asarray(
                (first, second, resolution - first - second), dtype=np.float64
            ) / resolution


def _fit_weights(
    identities: Sequence[str],
    score_matrices: Mapping[str, np.ndarray],
    branches: Sequence[str],
    *,
    resolution: int,
) -> dict[str, Any]:
    normalized = {branch: _row_zscores(score_matrices[branch], f"DEV {branch}") for branch in branches}
    best_key: tuple[float, ...] | None = None
    best_weights: np.ndarray | None = None
    best_rows: list[dict[str, Any]] | None = None
    candidate_count = 0
    for weights in _simplex_grid(len(branches), resolution):
        fused = sum(weight * normalized[branch] for branch, weight in zip(branches, weights, strict=True))
        rows = _rank_rows(fused, identities)
        summary = _metrics(rows, len(identities))
        key = (
            summary["Rank-1"],
            summary["MRR"],
            summary["Rank-5"],
            *weights.tolist(),
        )
        candidate_count += 1
        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights
            best_rows = rows
    if best_weights is None or best_rows is None:
        raise RuntimeError("simplex search produced no candidate")
    return {
        "branches": list(branches),
        "resolution": resolution,
        "candidate_count": candidate_count,
        "objective_lexicographic": ["Rank-1", "MRR", "Rank-5"],
        "tie_break": [f"HIGHER_{branch}_WEIGHT" for branch in branches],
        "selected_weights": {
            branch: float(weight)
            for branch, weight in zip(branches, best_weights, strict=True)
        },
        "selected_metrics": _metrics(best_rows, len(identities)),
    }


def calibrate_and_evaluate_score_fusion(
    dev_identities: Sequence[str],
    eval_identities: Sequence[str],
    dev_score_matrices: Mapping[str, np.ndarray],
    eval_score_matrices: Mapping[str, np.ndarray],
    *,
    resolution: int = 20,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Fit fusion weights on DEV labels and apply them once to EVAL."""

    dev_ids = list(dev_identities)
    eval_ids = list(eval_identities)
    if dev_ids != sorted(set(dev_ids)) or eval_ids != sorted(set(eval_ids)):
        raise ValueError("DEV and EVAL identities must be sorted unique lists")
    if set(dev_ids) & set(eval_ids):
        raise ValueError("DEV and EVAL identities must be disjoint")
    if min(len(dev_ids), len(eval_ids)) < 2:
        raise ValueError("DEV and EVAL each require at least two identities")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 1:
        raise ValueError("resolution must be a positive integer")
    if set(dev_score_matrices) != set(_BRANCHES) or set(eval_score_matrices) != set(_BRANCHES):
        raise ValueError("score matrices must contain exactly A0, F0, and N3")

    dev_rows = {branch: _rank_rows(dev_score_matrices[branch], dev_ids) for branch in _BRANCHES}
    eval_rows = {branch: _rank_rows(eval_score_matrices[branch], eval_ids) for branch in _BRANCHES}
    calibration: dict[str, Any] = {
        "labels_used": "DEV_ONLY",
        "row_zscore": {
            "scope": "EACH_QUERY_ROW_FULL_WITHIN_PARTITION_GALLERY",
            "standard_deviation": "POPULATION_DDOF_0",
            "near_zero_threshold": _ZSCORE_EPSILON,
        },
        "branch_metrics": {
            branch: _metrics(rows, len(dev_ids)) for branch, rows in dev_rows.items()
        },
        "fusions": {},
    }
    eval_normalized = {
        branch: _row_zscores(eval_score_matrices[branch], f"EVAL {branch}")
        for branch in _BRANCHES
    }
    for method, branches in _FUSIONS.items():
        fitted = _fit_weights(
            dev_ids,
            dev_score_matrices,
            branches,
            resolution=resolution,
        )
        calibration["fusions"][method] = fitted
        fused = sum(
            fitted["selected_weights"][branch] * eval_normalized[branch]
            for branch in branches
        )
        eval_rows[method] = _rank_rows(fused, eval_ids)
    return calibration, eval_rows


def _group_k5_population(
    records: Sequence[Mapping[str, Any]], identities: Sequence[str]
) -> list[dict[str, Any]]:
    by_identity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    owners: dict[str, dict[str, str]] = {
        field: {} for field in ("identity_token", "track_token", "sequence_token")
    }
    for row in records:
        if row["record_state"] != "NO_ROI" and row["registered_dog_id"] in identities:
            identity = row["registered_dog_id"]
            by_identity[identity].append(row)
            for field, owner_by_token in owners.items():
                owner = owner_by_token.setdefault(row[field], identity)
                if owner != identity:
                    raise ValueError(f"one {field} maps to multiple registered identities")
    population: list[dict[str, Any]] = []
    for identity in identities:
        rows = by_identity.get(identity, [])
        if len(rows) < 10:
            raise ValueError(f"split identity lacks ten localized frames: {identity}")
        if any(
            len({row[field] for row in rows}) != 1
            for field in ("identity_token", "track_token", "sequence_token")
        ):
            raise ValueError("split identity must map to one identity, track, and sequence")
        ordered = sorted(rows, key=lambda row: (row["frame_index"], row["sample_token"]))
        frame_indices = [row["frame_index"] for row in ordered]
        if len(frame_indices) != len(set(frame_indices)):
            raise ValueError("one track repeats a frame index")
        gallery = ordered[:5]
        query = ordered[-5:]
        if (
            {row["sample_token"] for row in gallery}
            & {row["sample_token"] for row in query}
            or {row["frame_index"] for row in gallery}
            & {row["frame_index"] for row in query}
        ):
            raise ValueError("earliest and latest K=5 windows overlap")
        population.append(
            {
                "registered_dog_id": identity,
                "gallery": gallery,
                "query": query,
            }
        )
    return population


def _native_source_key(row: Mapping[str, Any]) -> str:
    value = row["source_archive_member"]
    path = PurePosixPath(value)
    if not isinstance(value, str) or value != path.as_posix() or path.is_absolute() or ".." in path.parts:
        raise ValueError("native source archive member is unsafe")
    return value


def _roi_source_key(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("ROI image path must be text")
    path = PurePosixPath(value)
    if value != path.as_posix() or path.is_absolute() or ".." in path.parts:
        raise ValueError("ROI image path is unsafe")
    parts = path.parts[1:] if path.parts and path.parts[0] == "YT-BB-dog" else path.parts
    return PurePosixPath(*parts).as_posix()


def _identity_bound_face_records(
    population: Sequence[Mapping[str, Any]], roi_records: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    permissive_by_source: dict[str, int] = defaultdict(int)
    for row in roi_records:
        if row["face_crop_path"] is None:
            continue
        source = _roi_source_key(row["image_path"])
        permissive_by_source[source] += 1
        identity = row["registered_identity_id"]
        if identity is not None:
            by_key[(identity, source)].append(row)

    selected: list[dict[str, Any]] = []
    exclusions = {
        "missing_any_face_crop": 0,
        "missing_identity_bound_face_crop": 0,
        "repeated_identity_bound_face_crop": 0,
    }
    permissive_complete_count = 0
    for item in population:
        identity = item["registered_dog_id"]
        face_by_token: dict[str, Mapping[str, Any]] = {}
        missing_any = False
        missing_identity_bound = False
        repeated_identity_bound = False
        for role in ("gallery", "query"):
            for native in item[role]:
                source = _native_source_key(native)
                candidates = by_key.get((identity, source), [])
                if permissive_by_source.get(source, 0) == 0:
                    missing_any = True
                if not candidates:
                    missing_identity_bound = True
                    continue
                if len(candidates) != 1:
                    repeated_identity_bound = True
                    continue
                face = candidates[0]
                if face["image_sha256"] != native["source_sha256"]:
                    raise ValueError("native and ROI source image SHA-256 differ")
                face_by_token[native["sample_token"]] = face
        if not missing_any:
            permissive_complete_count += 1
        if missing_any:
            exclusions["missing_any_face_crop"] += 1
            continue
        if missing_identity_bound:
            exclusions["missing_identity_bound_face_crop"] += 1
            continue
        if repeated_identity_bound:
            exclusions["repeated_identity_bound_face_crop"] += 1
            continue
        selected.append({**item, "face_by_sample_token": face_by_token})
    return selected, exclusions, permissive_complete_count


def _read_bound_rgb(root: Path, relative: object, expected_sha256: object) -> Image.Image:
    if not isinstance(relative, str):
        raise ValueError("artifact path must be text")
    path = PurePosixPath(relative)
    if relative != path.as_posix() or path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path is unsafe")
    resolved = root.joinpath(*path.parts).resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved.is_symlink() or not resolved.is_file():
        raise ValueError("artifact path is unsafe")
    before = resolved.stat(follow_symlinks=False)
    payload = resolved.read_bytes()
    after = resolved.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or hashlib.sha256(payload).hexdigest()
        != _require_sha256(expected_sha256, "artifact SHA-256")
    ):
        raise ValueError("artifact SHA-256 differs")
    with Image.open(io.BytesIO(payload)) as opened:
        image = opened.convert("RGB")
        image.load()
    return image


def _extract_dino_embeddings(
    images: Sequence[Image.Image], model: ReceiptBoundDinov2Small, *, batch_size: int
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for offset in range(0, len(images), batch_size):
        values = model.extract_batch(list(images[offset : offset + batch_size]))
        vectors.extend(_normalize(value, "frozen DINOv2") for value in values)
    return vectors


def _score_matrices(
    population: Sequence[Mapping[str, Any]],
    *,
    source_root: Path,
    native_root: Path,
    roi_root: Path,
    dino: ReceiptBoundDinov2Small,
    nose_runtime: ExactOnnxRuntime,
    nose_manifest: NoseEmbeddingManifest,
    batch_size: int,
) -> tuple[list[str], dict[str, np.ndarray], list[dict[str, Any]]]:
    identities = [item["registered_dog_id"] for item in population]
    vectors = {
        branch: {"gallery": [], "query": []}
        for branch in _BRANCHES
    }
    pairings: list[dict[str, Any]] = []
    for item in population:
        identity_pairing = {"registered_dog_id": item["registered_dog_id"]}
        for role in ("gallery", "query"):
            rows = item[role]
            appearance_images = [
                _read_bound_rgb(source_root, _native_source_key(row), row["source_sha256"])
                for row in rows
            ]
            face_rows = [item["face_by_sample_token"][row["sample_token"]] for row in rows]
            face_images = [
                _read_bound_rgb(roi_root, row["face_crop_path"], row["face_crop_sha256"])
                for row in face_rows
            ]
            nose_images = [
                _read_bound_rgb(native_root, row["crop_path"], row["crop_sha256"])
                for row in rows
            ]
            appearance = _extract_dino_embeddings(appearance_images, dino, batch_size=batch_size)
            face = _extract_dino_embeddings(face_images, dino, batch_size=batch_size)
            nose = [
                _normalize(
                    nose_runtime.run(preprocess_image(image, nose_manifest))[0],
                    "Nose consistency-v3",
                )
                for image in nose_images
            ]
            vectors[_BRANCHES[0]][role].append(_k5(appearance, "Appearance K5"))
            vectors[_BRANCHES[1]][role].append(_k5(face, "Face K5"))
            vectors[_BRANCHES[2]][role].append(_k5(nose, "Nose K5"))
            identity_pairing[role] = {
                "frame_indices": [row["frame_index"] for row in rows],
                "sample_tokens": [row["sample_token"] for row in rows],
                "source_sha256s": [row["source_sha256"] for row in rows],
                "face_crop_sha256s": [row["face_crop_sha256"] for row in face_rows],
                "nose_crop_sha256s": [row["crop_sha256"] for row in rows],
            }
        pairings.append(identity_pairing)
    scores = {
        branch: compute_cosine_score_matrix(
            np.stack(vectors[branch]["query"]), np.stack(vectors[branch]["gallery"])
        )
        for branch in _BRANCHES
    }
    return identities, scores, pairings


def _load_bound_lineage(
    path: Path,
    expected_sha256: str,
    runtime_manifest_path: Path,
    onnx_path: Path,
) -> tuple[Any, Mapping[str, Any]]:
    document = read_strict_json_document(path)
    if document.canonical_payload_sha256 != expected_sha256:
        raise ValueError("Nose lineage content SHA-256 differs from the external pin")
    from embedding.methods.nose.training.embedding_consistency_training import (
        validate_lineage_manifest,
    )

    root = path.parent.resolve(strict=True)
    validate_lineage_manifest(document.payload, root)
    artifacts = document.payload["artifacts"]
    if runtime_manifest_path.resolve(strict=True) != (root / artifacts["runtime_manifest"]["path"]).resolve(strict=True):
        raise ValueError("Nose runtime manifest is not the lineage artifact")
    if onnx_path.resolve(strict=True) != (root / artifacts["onnx"]["path"]).resolve(strict=True):
        raise ValueError("Nose ONNX is not the lineage artifact")
    return document, document.payload["bindings"]


def _paired_bootstrap(
    per_identity: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        if method == _BRANCHES[0]:
            continue
        rows = [
            {
                "bootstrap_cluster_id": row["registered_dog_id"],
                **{
                    metric: row["method_outcomes"][method][metric]
                    - row["method_outcomes"][_BRANCHES[0]][metric]
                    for metric in _METRICS
                },
            }
            for row in per_identity
        ]
        result[f"{method}_minus_{_BRANCHES[0]}"] = {
            metric: identity_clustered_bootstrap_ci(
                rows,
                metric=metric,
                resamples=resamples,
                seed=seed,
                confidence_level=confidence_level,
            )
            for metric in _METRICS
        }
    return result


def _validate_outcome(value: object, identity: str, gallery_count: int) -> dict[str, Any]:
    expected = {
        "registered_dog_id", "rank", "Rank-1", "Rank-5", "MRR", "mAP",
        "genuine_score", "best_impostor_score", "genuine_margin",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("unified evaluation method outcome schema differs")
    rank = value["rank"]
    if (
        value["registered_dog_id"] != identity
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 1 <= rank <= gallery_count
        or value["Rank-1"] != float(rank == 1)
        or value["Rank-5"] != float(rank <= 5)
        or value["MRR"] != 1.0 / rank
        or value["mAP"] != 1.0 / rank
    ):
        raise ValueError("unified evaluation method outcome identity or rank differs")
    for name in ("genuine_score", "best_impostor_score", "genuine_margin"):
        metric = value[name]
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric):
            raise ValueError("unified evaluation method outcome score differs")
    if value["genuine_margin"] != value["genuine_score"] - value["best_impostor_score"]:
        raise ValueError("unified evaluation method outcome margin differs")
    return value


def validate_report_bundle(bundle: object) -> dict[str, Any]:
    if not isinstance(bundle, dict) or set(bundle) != {"schema_version", "report_sha256", "report"}:
        raise ValueError("unified evaluation report bundle schema differs")
    if bundle["schema_version"] != REPORT_BUNDLE_SCHEMA:
        raise ValueError("unified evaluation report bundle version differs")
    _require_sha256(bundle["report_sha256"], "report_sha256")
    report = bundle["report"]
    expected = {
        "schema_version", "status", "interpretation", "methods", "input_bindings",
        "input_sha256s", "code_sha256s", "protocol", "population", "calibration",
        "evaluation", "paired_delta_bootstrap_cis",
    }
    if not isinstance(report, dict) or set(report) != expected:
        raise ValueError("unified evaluation report schema differs")
    if content_sha256(report) != bundle["report_sha256"]:
        raise ValueError("unified evaluation report digest differs")
    if report["schema_version"] != REPORT_SCHEMA or report["interpretation"] != INTERPRETATION:
        raise ValueError("unified evaluation interpretation differs")
    if report["methods"] != METHODS or report["status"] != "PASS_SAME_TRACK_UNIFIED_RESEARCH_DIAGNOSTIC":
        raise ValueError("unified evaluation methods or status differ")
    protocol = report["protocol"]
    protocol_keys = {
        "population_source", "face_admission", "localized_states", "temporal_order",
        "gallery_selection", "query_selection", "frames_per_role", "frame_overlap_allowed",
        "fixed_common_population_across_methods", "temporal_aggregation", "retrieval",
        "fusion_labels", "evaluation_labels_used_for_weight_selection", "bootstrap",
        "limitations",
    }
    if not isinstance(protocol, dict) or set(protocol) != protocol_keys:
        raise ValueError("unified evaluation protocol schema differs")
    if (
        protocol["face_admission"]
        != "EXACTLY_ONE_FACE_CROP_BOUND_TO_REGISTERED_DOG_ID_AND_SOURCE_SHA256"
        or protocol["localized_states"] != ["AVAILABLE", "LOW_QUALITY"]
        or protocol["temporal_order"] != ["frame_index", "sample_token"]
        or protocol["gallery_selection"] != "earliest_five"
        or protocol["query_selection"] != "latest_five"
        or protocol["frames_per_role"] != 5
        or protocol["frame_overlap_allowed"] is not False
        or protocol["fixed_common_population_across_methods"] is not True
        or protocol["fusion_labels"] != "DEV_ONLY"
        or protocol["evaluation_labels_used_for_weight_selection"] is not False
        or not isinstance(protocol["limitations"], list)
        or not protocol["limitations"]
    ):
        raise ValueError("unified evaluation protocol contract differs")
    bootstrap_config = protocol["bootstrap"]
    if not isinstance(bootstrap_config, dict) or set(bootstrap_config) != {
        "cluster_unit", "resamples", "seed", "confidence_level"
    }:
        raise ValueError("unified evaluation bootstrap config differs")
    if (
        bootstrap_config["cluster_unit"] != "registered_dog_id"
        or isinstance(bootstrap_config["resamples"], bool)
        or not isinstance(bootstrap_config["resamples"], int)
        or bootstrap_config["resamples"] < 1
        or isinstance(bootstrap_config["seed"], bool)
        or not isinstance(bootstrap_config["seed"], int)
        or bootstrap_config["seed"] < 0
        or isinstance(bootstrap_config["confidence_level"], bool)
        or not isinstance(bootstrap_config["confidence_level"], (int, float))
        or not 0.0 < bootstrap_config["confidence_level"] < 1.0
    ):
        raise ValueError("unified evaluation bootstrap config contract differs")
    population = report["population"]
    if not isinstance(population, dict) or set(population) != {
        "parent_dev_identity_count", "parent_eval_identity_count",
        "permissive_dev_identity_count", "permissive_eval_identity_count",
        "selected_dev_identity_count", "selected_eval_identity_count",
        "selected_dev_registered_dog_ids", "selected_eval_registered_dog_ids",
        "selected_dev_registered_dog_ids_sha256", "selected_eval_registered_dog_ids_sha256",
        "dev_excluded_identity_counts", "eval_excluded_identity_counts",
    }:
        raise ValueError("unified evaluation population schema differs")
    dev_ids = population["selected_dev_registered_dog_ids"]
    eval_ids = population["selected_eval_registered_dog_ids"]
    exclusion_keys = {
        "missing_any_face_crop", "missing_identity_bound_face_crop",
        "repeated_identity_bound_face_crop",
    }
    dev_exclusions = population["dev_excluded_identity_counts"]
    eval_exclusions = population["eval_excluded_identity_counts"]
    if (
        not isinstance(dev_ids, list) or not isinstance(eval_ids, list)
        or dev_ids != sorted(set(dev_ids)) or eval_ids != sorted(set(eval_ids))
        or set(dev_ids) & set(eval_ids)
        or population["selected_dev_identity_count"] != len(dev_ids)
        or population["selected_eval_identity_count"] != len(eval_ids)
        or population["selected_dev_registered_dog_ids_sha256"] != _identity_list_sha256(dev_ids)
        or population["selected_eval_registered_dog_ids_sha256"] != _identity_list_sha256(eval_ids)
        or not isinstance(dev_exclusions, dict) or set(dev_exclusions) != exclusion_keys
        or not isinstance(eval_exclusions, dict) or set(eval_exclusions) != exclusion_keys
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in (*dev_exclusions.values(), *eval_exclusions.values()))
        or population["parent_dev_identity_count"]
        != population["permissive_dev_identity_count"] + dev_exclusions["missing_any_face_crop"]
        or population["parent_eval_identity_count"]
        != population["permissive_eval_identity_count"] + eval_exclusions["missing_any_face_crop"]
        or population["permissive_dev_identity_count"]
        != len(dev_ids) + dev_exclusions["missing_identity_bound_face_crop"] + dev_exclusions["repeated_identity_bound_face_crop"]
        or population["permissive_eval_identity_count"]
        != len(eval_ids) + eval_exclusions["missing_identity_bound_face_crop"] + eval_exclusions["repeated_identity_bound_face_crop"]
    ):
        raise ValueError("unified evaluation population identity binding differs")

    calibration = report["calibration"]
    if not isinstance(calibration, dict) or set(calibration) != {
        "labels_used", "row_zscore", "branch_metrics", "fusions"
    } or calibration["labels_used"] != "DEV_ONLY":
        raise ValueError("unified evaluation calibration schema differs")
    if set(calibration["branch_metrics"]) != set(_BRANCHES) or set(calibration["fusions"]) != set(_FUSIONS):
        raise ValueError("unified evaluation calibration methods differ")
    for summary in calibration["branch_metrics"].values():
        if summary["query_count"] != len(dev_ids) or summary["gallery_count"] != len(dev_ids):
            raise ValueError("unified evaluation DEV metric population differs")
    for method, branches in _FUSIONS.items():
        fitted = calibration["fusions"][method]
        expected_fit_keys = {
            "branches", "resolution", "candidate_count", "objective_lexicographic",
            "tie_break", "selected_weights", "selected_metrics",
        }
        if not isinstance(fitted, dict) or set(fitted) != expected_fit_keys or fitted["branches"] != list(branches):
            raise ValueError("unified evaluation fusion calibration differs")
        resolution = fitted["resolution"]
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 1:
            raise ValueError("unified evaluation fusion resolution differs")
        expected_candidates = resolution + 1 if len(branches) == 2 else (resolution + 1) * (resolution + 2) // 2
        weights = fitted["selected_weights"]
        if (
            fitted["candidate_count"] != expected_candidates
            or not isinstance(weights, dict)
            or set(weights) != set(branches)
            or any(isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0.0 for weight in weights.values())
            or not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
            or fitted["selected_metrics"]["query_count"] != len(dev_ids)
            or fitted["selected_metrics"]["gallery_count"] != len(dev_ids)
        ):
            raise ValueError("unified evaluation selected fusion differs")
    evaluation = report["evaluation"]
    if not isinstance(evaluation, dict) or set(evaluation) != {"metrics", "per_identity"}:
        raise ValueError("unified evaluation result schema differs")
    rows = evaluation["per_identity"]
    if not isinstance(rows, list) or [row.get("registered_dog_id") for row in rows] != eval_ids:
        raise ValueError("unified evaluation per-identity rows differ")
    if set(evaluation["metrics"]) != set(METHODS):
        raise ValueError("unified evaluation metric methods differ")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"registered_dog_id", "method_outcomes"} or set(row["method_outcomes"]) != set(METHODS):
            raise ValueError("unified evaluation per-identity outcome schema differs")
        for method in METHODS:
            _validate_outcome(row["method_outcomes"][method], row["registered_dog_id"], len(eval_ids))
    for method in METHODS:
        outcomes = [row["method_outcomes"][method] for row in rows]
        expected_metrics = _metrics(outcomes, len(eval_ids))
        if evaluation["metrics"][method] != expected_metrics:
            raise ValueError(f"unified evaluation {method} metrics differ")
    input_hashes = report["input_sha256s"]
    expected_input_hashes = {
        "native_bundle_content", "roi_manifest_content", "nose_lineage_content",
        "nose_runtime_manifest_content", "nose_onnx", "frozen_dinov2",
        "dev_registered_dog_ids", "eval_registered_dog_ids", "dev_pairing", "eval_pairing",
    }
    if not isinstance(input_hashes, dict) or set(input_hashes) != expected_input_hashes:
        raise ValueError("unified evaluation input hash schema differs")
    input_bindings = report["input_bindings"]
    if not isinstance(input_bindings, dict) or set(input_bindings) != {
        "native_bundle", "native_root", "source_image_root", "roi_manifest", "nose_lineage",
        "nose_runtime_manifest", "nose_onnx", "frozen_dinov2", "dev_pairing_sha256",
        "eval_pairing_sha256",
    }:
        raise ValueError("unified evaluation input binding schema differs")
    if (
        input_hashes["native_bundle_content"] != input_bindings["native_bundle"]["content_sha256"]
        or input_hashes["roi_manifest_content"] != input_bindings["roi_manifest"]["content_sha256"]
        or input_hashes["nose_lineage_content"] != input_bindings["nose_lineage"]["content_sha256"]
        or input_hashes["nose_runtime_manifest_content"] != input_bindings["nose_runtime_manifest"]["content_sha256"]
        or input_hashes["nose_onnx"] != input_bindings["nose_onnx"]["sha256"]
        or input_hashes["frozen_dinov2"] != input_bindings["frozen_dinov2"]["model_sha256"]
        or input_hashes["dev_registered_dog_ids"] != population["selected_dev_registered_dog_ids_sha256"]
        or input_hashes["eval_registered_dog_ids"] != population["selected_eval_registered_dog_ids_sha256"]
        or input_hashes["dev_pairing"] != input_bindings["dev_pairing_sha256"]
        or input_hashes["eval_pairing"] != input_bindings["eval_pairing_sha256"]
    ):
        raise ValueError("unified evaluation input binding digest differs")
    for digest in input_hashes.values():
        _require_sha256(digest, "input SHA-256")
    code_paths = (
        frozenset(report["code_sha256s"])
        if isinstance(report["code_sha256s"], dict)
        else frozenset()
    )
    if code_paths not in {
        frozenset(_CODE_PATHS),
        frozenset(_PRE_TRAINING_OWNERSHIP_CODE_PATHS),
        frozenset(_PRE_NESTED_EMBEDDING_CODE_PATHS),
        frozenset(_PRE_EMBEDDING_CODE_PATHS),
        frozenset(_LEGACY_CODE_PATHS),
    }:
        raise ValueError("unified evaluation code hash schema differs")
    for digest in report["code_sha256s"].values():
        _require_sha256(digest, "code SHA-256")

    intervals = report["paired_delta_bootstrap_cis"]
    expected_contrasts = {f"{method}_minus_{_BRANCHES[0]}" for method in METHODS if method != _BRANCHES[0]}
    if not isinstance(intervals, dict) or set(intervals) != expected_contrasts:
        raise ValueError("unified evaluation bootstrap contrasts differ")
    for method in METHODS:
        if method == _BRANCHES[0]:
            continue
        contrast = f"{method}_minus_{_BRANCHES[0]}"
        if set(intervals[contrast]) != set(_METRICS):
            raise ValueError("unified evaluation bootstrap contrast binding differs")
        for metric, interval in intervals[contrast].items():
            estimate = float(np.mean([
                row["method_outcomes"][method][metric] - row["method_outcomes"][_BRANCHES[0]][metric]
                for row in rows
            ]))
            if (
                not isinstance(interval, dict)
                or interval.get("metric") != metric
                or interval.get("estimate") != estimate
                or interval.get("cluster_count") != len(eval_ids)
                or interval.get("query_row_count") != len(eval_ids)
                or interval.get("resamples") != bootstrap_config["resamples"]
                or interval.get("seed") != bootstrap_config["seed"]
                or interval.get("confidence_level") != bootstrap_config["confidence_level"]
            ):
                raise ValueError("unified evaluation bootstrap interval differs")
    return report


def evaluate_unified_multievidence(
    *,
    native_bundle_path: Path,
    native_bundle_sha256: str,
    native_root: Path,
    source_image_root: Path,
    roi_manifest_path: Path,
    roi_manifest_sha256: str,
    nose_lineage_path: Path,
    nose_lineage_sha256: str,
    nose_manifest_path: Path,
    nose_manifest_sha256: str,
    nose_onnx_path: Path,
    model_directory: Path,
    weight_intake_bundle: Path,
    preprocessor_intake_bundle: Path,
    frozen_model_sha256: str,
    output_path: Path,
    device: str = "cpu",
    nose_use_cuda: bool = False,
    batch_size: int = 32,
    fusion_resolution: int = 20,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_confidence_level: float = 0.95,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Run a fixed common-population A0/F0/N3 same-track diagnostic."""

    for value, name in (
        (native_bundle_sha256, "native_bundle_sha256"),
        (roi_manifest_sha256, "roi_manifest_sha256"),
        (nose_lineage_sha256, "nose_lineage_sha256"),
        (nose_manifest_sha256, "nose_manifest_sha256"),
        (frozen_model_sha256, "frozen_model_sha256"),
    ):
        _require_sha256(value, name)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")
    if not 0.0 < float(bootstrap_confidence_level) < 1.0:
        raise ValueError("bootstrap_confidence_level must be in (0,1)")

    repository = Path(__file__).resolve().parents[1] if repository_root is None else repository_root.resolve(strict=True)
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
    if native_document.canonical_payload_sha256 != native_bundle_sha256:
        raise ValueError("native bundle content SHA-256 differs from the external pin")
    native_resolved_root = native_root.resolve(strict=True)
    native_manifest = validate_manifest_bundle(native_document.payload, root=native_resolved_root)

    roi_document = read_strict_json_document(
        roi_manifest_path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    if roi_document.canonical_payload_sha256 != roi_manifest_sha256:
        raise ValueError("ROI manifest content SHA-256 differs from the external pin")
    roi_manifest = read_roi_manifest(roi_manifest_path)
    if roi_document.payload.get("manifest") != roi_manifest:
        raise RuntimeError("ROI manifest changed between pinned and validated reads")
    roi_root = roi_manifest_path.parent.resolve(strict=True)

    lineage_document, lineage_bindings = _load_bound_lineage(
        nose_lineage_path,
        nose_lineage_sha256,
        nose_manifest_path,
        nose_onnx_path,
    )
    native_binding = lineage_bindings["native_v4_bundle"]
    if native_binding != {
        "file_sha256": native_document.raw_sha256,
        "payload_sha256": native_document.canonical_payload_sha256,
        "manifest_sha256": native_document.payload["manifest_sha256"],
        "input_sha256s": native_manifest["input_sha256s"],
    }:
        raise ValueError("native bundle differs from the Nose consistency lineage binding")
    splits = lineage_bindings["splits"]
    parent_dev_ids = splits["identity_lists"]["dev"]
    parent_eval_ids = splits["identity_lists"]["eval"]
    for role, identities in (("dev", parent_dev_ids), ("eval", parent_eval_ids)):
        identity_set = set(identities)
        observed_tokens = sorted(
            row["sample_token"]
            for row in native_manifest["records"]
            if row["record_state"] != "NO_ROI"
            and row["registered_dog_id"] in identity_set
        )
        if observed_tokens != splits["sample_token_lists"][role]:
            raise ValueError(f"native {role.upper()} samples differ from the lineage split")
    nose_document = read_strict_json_document(nose_manifest_path)
    if nose_document.canonical_payload_sha256 != nose_manifest_sha256:
        raise ValueError("Nose runtime manifest content SHA-256 differs from the external pin")
    nose_manifest = NoseEmbeddingManifest.from_dict(nose_document.payload)
    if nose_manifest.license.usage_lane != UsageLane.RESEARCH_ONLY:
        raise ArtifactContractError("Nose embedding must be research-only")
    nose_onnx_binding = _file_binding(nose_onnx_path)
    if nose_onnx_binding["sha256"] != nose_manifest.artifact_sha256:
        raise ArtifactContractError("Nose ONNX differs from its runtime manifest")

    dino = ReceiptBoundDinov2Small(
        model_directory=model_directory,
        weight_intake_bundle=weight_intake_bundle,
        preprocessor_intake_bundle=preprocessor_intake_bundle,
        device=device,
        max_batch_size=batch_size,
    )
    if dino.model_sha256 != frozen_model_sha256:
        raise ValueError("frozen DINOv2 model SHA-256 differs from the external pin")
    nose_runtime = ExactOnnxRuntime(nose_onnx_path, nose_manifest, use_cuda=nose_use_cuda)

    dev_parent = _group_k5_population(native_manifest["records"], parent_dev_ids)
    eval_parent = _group_k5_population(native_manifest["records"], parent_eval_ids)
    dev_population, dev_exclusions, permissive_dev = _identity_bound_face_records(
        dev_parent, roi_manifest["records"]
    )
    eval_population, eval_exclusions, permissive_eval = _identity_bound_face_records(
        eval_parent, roi_manifest["records"]
    )
    if min(len(dev_population), len(eval_population)) < 10:
        raise ValueError("identity-bound unified population is too small")

    source_root = source_image_root.resolve(strict=True)
    dev_ids, dev_scores, dev_pairings = _score_matrices(
        dev_population,
        source_root=source_root,
        native_root=native_resolved_root,
        roi_root=roi_root,
        dino=dino,
        nose_runtime=nose_runtime,
        nose_manifest=nose_manifest,
        batch_size=batch_size,
    )
    eval_ids, eval_scores, eval_pairings = _score_matrices(
        eval_population,
        source_root=source_root,
        native_root=native_resolved_root,
        roi_root=roi_root,
        dino=dino,
        nose_runtime=nose_runtime,
        nose_manifest=nose_manifest,
        batch_size=batch_size,
    )
    calibration, eval_rows = calibrate_and_evaluate_score_fusion(
        dev_ids,
        eval_ids,
        dev_scores,
        eval_scores,
        resolution=fusion_resolution,
    )
    per_identity = [
        {
            "registered_dog_id": identity,
            "method_outcomes": {
                method: eval_rows[method][index] for method in METHODS
            },
        }
        for index, identity in enumerate(eval_ids)
    ]
    paired_bootstrap = _paired_bootstrap(
        per_identity,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        confidence_level=float(bootstrap_confidence_level),
    )
    protocol = {
        "population_source": "Nose consistency-v3 DEV and EVAL identity lists",
        "face_admission": "EXACTLY_ONE_FACE_CROP_BOUND_TO_REGISTERED_DOG_ID_AND_SOURCE_SHA256",
        "localized_states": ["AVAILABLE", "LOW_QUALITY"],
        "temporal_order": ["frame_index", "sample_token"],
        "gallery_selection": "earliest_five",
        "query_selection": "latest_five",
        "frames_per_role": 5,
        "frame_overlap_allowed": False,
        "fixed_common_population_across_methods": True,
        "temporal_aggregation": "L2_NORMALIZED_UNWEIGHTED_K5_MEAN",
        "retrieval": "EXHAUSTIVE_COSINE_ONE_GALLERY_VECTOR_PER_IDENTITY",
        "fusion_labels": "DEV_ONLY",
        "evaluation_labels_used_for_weight_selection": False,
        "bootstrap": {
            "cluster_unit": "registered_dog_id",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence_level": float(bootstrap_confidence_level),
        },
        "limitations": [
            "SAME_VIDEO_TRACK_GALLERY_AND_QUERY",
            "TRACK_IDENTITIES_NOT_LIFELONG_DOG_IDENTITIES",
            "FACE_ROI_AVAILABILITY_SELECTION_BIAS",
            "NO_UPSTREAM_DINOV2_YT_BB_DOG_NONOVERLAP_ASSERTION",
            "NO_OPEN_SET_UNKNOWN_REJECTION",
        ],
    }
    input_bindings = {
        "native_bundle": {
            "path": os.fspath(native_bundle_path),
            "raw_sha256": native_document.raw_sha256,
            "content_sha256": native_document.canonical_payload_sha256,
            "byte_size": native_document.byte_size,
        },
        "native_root": os.fspath(native_resolved_root),
        "source_image_root": os.fspath(source_root),
        "roi_manifest": {
            "path": os.fspath(roi_manifest_path),
            "raw_sha256": roi_document.raw_sha256,
            "content_sha256": roi_document.canonical_payload_sha256,
            "byte_size": roi_document.byte_size,
        },
        "nose_lineage": {
            "path": os.fspath(nose_lineage_path),
            "raw_sha256": lineage_document.raw_sha256,
            "content_sha256": lineage_document.canonical_payload_sha256,
            "lineage_sha256": lineage_document.payload["lineage_sha256"],
            "byte_size": lineage_document.byte_size,
        },
        "nose_runtime_manifest": {
            "path": os.fspath(nose_manifest_path),
            "raw_sha256": nose_document.raw_sha256,
            "content_sha256": nose_document.canonical_payload_sha256,
            "byte_size": nose_document.byte_size,
        },
        "nose_onnx": nose_onnx_binding,
        "frozen_dinov2": {
            "model_directory": os.fspath(model_directory.resolve(strict=True)),
            "model_sha256": dino.model_sha256,
            "preprocessor_sha256": dino.preprocessor_sha256,
            "weight_intake_receipt_sha256": dino.weight_receipt_sha256,
            "preprocessor_intake_receipt_sha256": dino.preprocessor_receipt_sha256,
        },
        "dev_pairing_sha256": content_sha256(dev_pairings),
        "eval_pairing_sha256": content_sha256(eval_pairings),
    }
    population = {
        "parent_dev_identity_count": len(parent_dev_ids),
        "parent_eval_identity_count": len(parent_eval_ids),
        "permissive_dev_identity_count": permissive_dev,
        "permissive_eval_identity_count": permissive_eval,
        "selected_dev_identity_count": len(dev_ids),
        "selected_eval_identity_count": len(eval_ids),
        "selected_dev_registered_dog_ids": dev_ids,
        "selected_eval_registered_dog_ids": eval_ids,
        "selected_dev_registered_dog_ids_sha256": _identity_list_sha256(dev_ids),
        "selected_eval_registered_dog_ids_sha256": _identity_list_sha256(eval_ids),
        "dev_excluded_identity_counts": dev_exclusions,
        "eval_excluded_identity_counts": eval_exclusions,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_SAME_TRACK_UNIFIED_RESEARCH_DIAGNOSTIC",
        "interpretation": INTERPRETATION,
        "methods": METHODS,
        "input_bindings": input_bindings,
        "input_sha256s": {
            "native_bundle_content": native_document.canonical_payload_sha256,
            "roi_manifest_content": roi_document.canonical_payload_sha256,
            "nose_lineage_content": lineage_document.canonical_payload_sha256,
            "nose_runtime_manifest_content": nose_document.canonical_payload_sha256,
            "nose_onnx": nose_onnx_binding["sha256"],
            "frozen_dinov2": dino.model_sha256,
            "dev_registered_dog_ids": population["selected_dev_registered_dog_ids_sha256"],
            "eval_registered_dog_ids": population["selected_eval_registered_dog_ids_sha256"],
            "dev_pairing": input_bindings["dev_pairing_sha256"],
            "eval_pairing": input_bindings["eval_pairing_sha256"],
        },
        "code_sha256s": {
            relative: _file_sha256(repository / relative) for relative in _CODE_PATHS
        },
        "protocol": protocol,
        "population": population,
        "calibration": calibration,
        "evaluation": {
            "metrics": {
                method: _metrics(eval_rows[method], len(eval_ids)) for method in METHODS
            },
            "per_identity": per_identity,
        },
        "paired_delta_bootstrap_cis": paired_bootstrap,
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
    "calibrate_and_evaluate_score_fusion",
    "evaluate_unified_multievidence",
    "validate_report_bundle",
]
