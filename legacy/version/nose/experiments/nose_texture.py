"""Identity-disjoint evaluation of a fixed classical Nose texture branch."""

from __future__ import annotations

from legacy.version.root import repository_root as find_repo_root
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from contracts.artifact_manifest import ExactOnnxRuntime, NoseEmbeddingManifest
from evaluation.embedding_diagnostics import compute_embedding_diagnostics
from evaluation.retrieval import (
    compute_cosine_score_matrix,
    identity_clustered_bootstrap_ci,
)
from legacy.version.nose.experiments.nose_fusion_scaling import (
    _embed,
    _file_binding,
    _file_sha256,
    _group_population,
    _metrics,
    _rank_rows,
    _require_sha256,
)
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.provenance import content_sha256
from embedding.methods.nose.signal.frequency import classical_texture_descriptors
from embedding.methods.nose.signal.temporal import aggregate_nose_embeddings
from parsing.nose_region.native_yt import validate_manifest_bundle

REPORT_SCHEMA = "cvi.yt_nose_texture_evaluation.v2"
REPORT_BUNDLE_SCHEMA = "cvi.yt_nose_texture_evaluation_bundle.v2"
INTERPRETATION = (
    "YT_FIT_TRACK_PROXY_CLASSICAL_TEXTURE_DEVELOPMENT_DIAGNOSTIC_"
    "NOT_CROSS_SESSION_BIOMETRIC_VALIDATION"
)
FUSION_WEIGHTS = tuple(index / 20.0 for index in range(7))
_METRICS = ("Rank-1", "Rank-5", "MRR", "mAP")
_CODE_PATHS = (
    "legacy/version/nose/experiments/nose_texture.py",
    "embedding/methods/nose/signal/frequency.py",
    "embedding/methods/nose/signal/temporal.py",
    "legacy/version/nose/experiments/nose_fusion_scaling.py",
    "evaluation/embedding_diagnostics.py",
    "evaluation/retrieval.py",
    "legacy/version/nose/workflows/evaluate_yt_nose_texture.py",
)


def _l2(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if value.ndim != 1 or not np.isfinite(value).all() or not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("classical Nose descriptor is non-finite or degenerate")
    return np.asarray(value / norm, dtype=np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    support = np.asarray(mask) > 0.5
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise ValueError("Nose texture mask has no foreground support")
    index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    largest = (labels == index).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(largest, cv2.MORPH_CLOSE, kernel).astype(bool)


def _linear_luminance(rgb: np.ndarray) -> np.ndarray:
    srgb = np.asarray(rgb, dtype=np.float32) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    return np.sum(linear * np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32), axis=2)


def _photometric_views(luminance: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = luminance[mask]
    if selected.size < 16:
        raise ValueError("Nose texture mask has insufficient observed pixels")
    low, high = np.quantile(selected, (0.02, 0.98))
    normalized = np.clip((luminance - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    log_luminance = np.log(normalized + 1e-4)
    illumination = cv2.GaussianBlur(
        log_luminance, (0, 0), sigmaX=9.0, sigmaY=9.0, borderType=cv2.BORDER_REFLECT_101
    )
    high_pass = log_luminance - illumination
    values = high_pass[mask]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    high_pass = np.clip((high_pass - median) / (1.4826 * mad + 1e-6), -5.0, 5.0)
    return normalized.astype(np.float32), high_pass.astype(np.float32)


def _descriptor_block(luminance: np.ndarray, mask: np.ndarray) -> np.ndarray:
    descriptors = classical_texture_descriptors(luminance, mask)
    blocks: list[np.ndarray] = []
    for name in ("gabor", "lbp", "radial_frequency"):
        value = descriptors[name]
        if name == "lbp":
            value = np.sqrt(np.maximum(value, 0.0))
        norm = float(np.linalg.norm(value))
        blocks.append(value if norm <= 1e-12 else value / norm)
    return _l2(np.concatenate(blocks))


def texture_descriptor(root: Path, row: Mapping[str, Any]) -> np.ndarray:
    """Create a fixed raw/normalized/homomorphic descriptor on a canonical grid."""

    with Image.open(root / row["crop_path"]) as opened:
        rgb = np.asarray(opened.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC))
    with Image.open(root / row["binary_mask_path"]) as opened:
        mask = np.asarray(opened.convert("L").resize((128, 128), Image.Resampling.NEAREST)) >= 128
    cleaned = _largest_component(mask)
    luminance = _linear_luminance(rgb)
    normalized, high_pass = _photometric_views(luminance, cleaned)
    return _l2(
        np.concatenate(
            (
                _descriptor_block(luminance, mask),
                _descriptor_block(normalized, cleaned),
                _descriptor_block(high_pass, cleaned),
            )
        )
    )


def _fuse(vectors: Sequence[np.ndarray]) -> np.ndarray:
    return aggregate_nose_embeddings(vectors).embedding


def _identity_partition(identity: str) -> str:
    digest = hashlib.sha256(f"cvi.nose_texture.v1:{identity}".encode("ascii")).digest()
    return "DEVELOPMENT" if int.from_bytes(digest[:4], "big") % 4 == 0 else "EVALUATION"


def _fit_whitener(matrix: np.ndarray) -> dict[str, np.ndarray | float | int]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("whitener fit matrix must be finite and two-dimensional")
    mean = values.mean(axis=0)
    centered = values - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    variances = singular_values * singular_values / (len(values) - 1)
    if not len(variances) or variances[0] <= 0.0:
        raise ValueError("classical Nose descriptor covariance is degenerate")
    retained = min(128, int(np.sum(variances >= variances[0] * 1e-4)))
    retained = max(1, retained)
    regularization = max(float(variances[0]) * 1e-3, 1e-8)
    return {
        "mean": mean,
        "components": components[:retained],
        "scales": np.sqrt(variances[:retained] + regularization),
        "retained_dimension": retained,
        "regularization": regularization,
    }


def _apply_whitener(
    matrix: np.ndarray, whitener: Mapping[str, np.ndarray | float | int]
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    transformed = (
        (values - np.asarray(whitener["mean"]))
        @ np.asarray(whitener["components"]).T
    ) / np.asarray(whitener["scales"])
    norms = np.linalg.norm(transformed, axis=1)
    if not np.isfinite(transformed).all() or np.any(norms <= 1e-12):
        raise ValueError("whitened classical Nose descriptor is invalid")
    return np.asarray(transformed / norms[:, None], dtype=np.float32)


def _subset_scores(scores: np.ndarray, indices: list[int]) -> np.ndarray:
    return scores[np.ix_(indices, indices)]


def _fit_weight(
    raw_scores: np.ndarray,
    texture_scores: np.ndarray,
    identities: Sequence[str],
) -> tuple[float, list[dict[str, Any]]]:
    grid: list[dict[str, Any]] = []
    for weight in FUSION_WEIGHTS:
        scores = (1.0 - weight) * raw_scores + weight * texture_scores
        metrics = _metrics(_rank_rows(scores, identities), len(identities))
        grid.append({"texture_weight": weight, "metrics": metrics})
    selected = max(
        grid,
        key=lambda item: (
            item["metrics"]["MRR"],
            item["metrics"]["Rank-1"],
            -item["texture_weight"],
        ),
    )
    return float(selected["texture_weight"]), grid


def _paired_cis(
    method_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    identities: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    return {
        metric: identity_clustered_bootstrap_ci(
            [
                {
                    "bootstrap_cluster_id": identity,
                    metric: float(method[metric] - baseline[metric]),
                }
                for identity, method, baseline in zip(
                    identities, method_rows, baseline_rows, strict=True
                )
            ],
            metric=metric,
            resamples=resamples,
            seed=seed + index,
        )
        for index, metric in enumerate(_METRICS)
    }


def evaluate_nose_texture(
    *,
    native_bundle_path: Path,
    native_bundle_sha256: str,
    native_root: Path,
    nose_runtime_manifest_path: Path,
    nose_runtime_manifest_sha256: str,
    nose_onnx_path: Path,
    output_path: Path,
    use_cuda: bool = False,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    expected_native = _require_sha256(native_bundle_sha256, "native_bundle_sha256")
    expected_runtime = _require_sha256(
        nose_runtime_manifest_sha256, "nose_runtime_manifest_sha256"
    )
    if bootstrap_resamples <= 0 or bootstrap_seed < 0:
        raise ValueError("bootstrap policy differs")
    repository = find_repo_root(__file__)
    output = Path(os.path.abspath(os.fspath(output_path)))
    output = output.parent.resolve(strict=True) / output.name
    if output.is_relative_to(repository):
        raise ValueError("texture evaluation report must be outside Git")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    native_document = read_strict_json_document(
        native_bundle_path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    if native_document.canonical_payload_sha256 != expected_native:
        raise ValueError("native manifest bundle differs from external pin")
    root = native_root.resolve(strict=True)
    native_manifest = validate_manifest_bundle(native_document.payload, root=root)
    runtime_document = read_strict_json_document(nose_runtime_manifest_path)
    if runtime_document.canonical_payload_sha256 != expected_runtime:
        raise ValueError("Nose runtime manifest differs from external pin")
    runtime_manifest = NoseEmbeddingManifest.from_dict(runtime_document.payload)
    runtime = ExactOnnxRuntime(nose_onnx_path, runtime_manifest, use_cuda=use_cuda)
    if _file_sha256(nose_onnx_path) != runtime_manifest.artifact_sha256:
        raise ValueError("Nose ONNX differs from runtime manifest")

    population, excluded = _group_population(native_manifest["records"])
    identities = [item["registered_dog_id"] for item in population]
    raw_gallery: list[np.ndarray] = []
    raw_query: list[np.ndarray] = []
    texture_gallery: list[np.ndarray] = []
    texture_query: list[np.ndarray] = []
    for item in population:
        for role, raw_output, texture_output in (
            ("gallery", raw_gallery, texture_gallery),
            ("query", raw_query, texture_query),
        ):
            rows = item[role]
            raw_output.append(
                _fuse([_embed(runtime, runtime_manifest, root, row) for row in rows])
            )
            texture_output.append(_fuse([texture_descriptor(root, row) for row in rows]))

    raw_gallery_array = np.stack(raw_gallery)
    raw_query_array = np.stack(raw_query)
    texture_gallery_array = np.stack(texture_gallery)
    texture_query_array = np.stack(texture_query)
    raw_scores = compute_cosine_score_matrix(raw_query_array, raw_gallery_array)
    unwhitened_texture_scores = compute_cosine_score_matrix(
        texture_query_array, texture_gallery_array
    )
    development = [index for index, identity in enumerate(identities) if _identity_partition(identity) == "DEVELOPMENT"]
    evaluation = [index for index, identity in enumerate(identities) if _identity_partition(identity) == "EVALUATION"]
    if len(development) < 2 or len(evaluation) < 2:
        raise RuntimeError("deterministic Nose texture partition is too small")
    development_ids = [identities[index] for index in development]
    evaluation_ids = [identities[index] for index in evaluation]
    whitener = _fit_whitener(
        np.concatenate(
            (texture_gallery_array[development], texture_query_array[development]),
            axis=0,
        )
    )
    whitened_texture_gallery = _apply_whitener(texture_gallery_array, whitener)
    whitened_texture_query = _apply_whitener(texture_query_array, whitener)
    texture_scores = compute_cosine_score_matrix(
        whitened_texture_query, whitened_texture_gallery
    )
    selected_weight, grid = _fit_weight(
        _subset_scores(raw_scores, development),
        _subset_scores(texture_scores, development),
        development_ids,
    )
    evaluation_raw_scores = _subset_scores(raw_scores, evaluation)
    evaluation_unwhitened_texture_scores = _subset_scores(
        unwhitened_texture_scores, evaluation
    )
    evaluation_texture_scores = _subset_scores(texture_scores, evaluation)
    evaluation_fused_scores = (
        (1.0 - selected_weight) * evaluation_raw_scores
        + selected_weight * evaluation_texture_scores
    )
    raw_rows = _rank_rows(evaluation_raw_scores, evaluation_ids)
    unwhitened_texture_rows = _rank_rows(
        evaluation_unwhitened_texture_scores, evaluation_ids
    )
    texture_rows = _rank_rows(evaluation_texture_scores, evaluation_ids)
    fused_rows = _rank_rows(evaluation_fused_scores, evaluation_ids)
    repeated_ids = [identity for identity in evaluation_ids for _ in range(2)]
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_YT_FIT_NOSE_TEXTURE_DIAGNOSTIC",
        "interpretation": INTERPRETATION,
        "input_bindings": {
            "native_manifest_bundle": {
                **_file_binding(native_bundle_path),
                "content_sha256": native_document.canonical_payload_sha256,
                "manifest_sha256": native_document.payload["manifest_sha256"],
            },
            "native_artifact_root": os.fspath(root),
            "nose_runtime_manifest": {
                **_file_binding(nose_runtime_manifest_path),
                "content_sha256": runtime_document.canonical_payload_sha256,
            },
            "nose_onnx": _file_binding(nose_onnx_path),
        },
        "code_sha256s": {
            relative: _file_sha256(repository / relative) for relative in _CODE_PATHS
        },
        "protocol": {
            "population": "YT_FIT identities with at least ten localized frames",
            "gallery_selection": "earliest_five",
            "query_selection": "latest_five",
            "partition": "SHA256_IDENTITY_1_OF_4_DEVELOPMENT_3_OF_4_EVALUATION",
            "raw_method": "consistency-v3 Nose embedding unweighted K5 mean",
            "texture_method": (
                "128px linear-light raw-mask plus percentile-normalized and homomorphic "
                "largest-component-mask Gabor/LBP/radial-FFT descriptor, unweighted K5 mean, "
                "then label-blind DEVELOPMENT-fitted regularized PCA whitening"
            ),
            "fusion": "score-level convex combination selected on DEVELOPMENT MRR",
            "candidate_texture_weights": list(FUSION_WEIGHTS),
            "bootstrap": {
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
                "cluster_unit": "registered_dog_id",
            },
        },
        "population": {
            "eligible_identity_count": len(identities),
            "development_identity_count": len(development),
            "evaluation_identity_count": len(evaluation),
            "excluded": excluded,
        },
        "development_selection": {
            "selected_texture_weight": selected_weight,
            "whitening": {
                "fit_sample_count": 2 * len(development),
                "input_dimension": texture_gallery_array.shape[1],
                "retained_dimension": whitener["retained_dimension"],
                "relative_eigenvalue_floor": 1e-4,
                "regularization": whitener["regularization"],
                "identity_labels_used": False,
            },
            "grid": grid,
        },
        "evaluation": {
            "raw_embedding": _metrics(raw_rows, len(evaluation_ids)),
            "classical_texture_unwhitened": _metrics(
                unwhitened_texture_rows, len(evaluation_ids)
            ),
            "classical_texture_whitened": _metrics(
                texture_rows, len(evaluation_ids)
            ),
            "fused": _metrics(fused_rows, len(evaluation_ids)),
            "fused_minus_raw_bootstrap_cis": _paired_cis(
                fused_rows,
                raw_rows,
                evaluation_ids,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            ),
        },
        "embedding_geometry": {
            "raw_embedding": compute_embedding_diagnostics(
                np.stack(
                    [vector for pair in zip(
                        np.asarray(raw_gallery_array)[evaluation],
                        np.asarray(raw_query_array)[evaluation],
                        strict=True,
                    ) for vector in pair]
                ),
                identity_ids=repeated_ids,
            ),
            "classical_texture_unwhitened": compute_embedding_diagnostics(
                np.stack(
                    [vector for pair in zip(
                        np.asarray(texture_gallery_array)[evaluation],
                        np.asarray(texture_query_array)[evaluation],
                        strict=True,
                    ) for vector in pair]
                ),
                identity_ids=repeated_ids,
            ),
            "classical_texture_whitened": compute_embedding_diagnostics(
                np.stack(
                    [vector for pair in zip(
                        np.asarray(whitened_texture_gallery)[evaluation],
                        np.asarray(whitened_texture_query)[evaluation],
                        strict=True,
                    ) for vector in pair]
                ),
                identity_ids=repeated_ids,
            ),
        },
        "limitations": [
            "YT publisher track IDs are proxy identities, not lifelong dog identities.",
            "Earliest/latest windows come from the same source video track.",
            "The classical branch uses a predicted mask and has no human-GT anatomical admission.",
            "No result from this report is eligible for final or deployment claims.",
        ],
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    bundle = {
        "schema_version": REPORT_BUNDLE_SCHEMA,
        "report_sha256": content_sha256(report),
        "report": report,
    }
    write_private_json_bundle(((output, bundle),))
    return bundle


__all__ = [
    "FUSION_WEIGHTS",
    "evaluate_nose_texture",
    "texture_descriptor",
]
