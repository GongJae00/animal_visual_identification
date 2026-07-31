"""Paired within-YT-track evaluation of raw and conservatively restored noses."""

from __future__ import annotations

import hashlib
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from cvi.evidence.artifact_manifest import (
    ArtifactContractError,
    ExactOnnxRuntime,
    NoseEmbeddingManifest,
    preprocess_image,
)
from cvi.evaluation.retrieval import compute_cosine_score_matrix
from cvi.nose_id.restoration import RestorationConfig, restore_nose_frames
from cvi.nose_id.temporal import aggregate_nose_embeddings
from cvi.nose_region.native_yt import validate_manifest_bundle
from cvi.protected_io import read_strict_json_document, write_private_json_bundle
from cvi.provenance import content_sha256


REPORT_SCHEMA = "cvi.yt_nose_raw_restored_evaluation.v1"
REPORT_BUNDLE_SCHEMA = "cvi.yt_nose_raw_restored_evaluation_bundle.v1"
INTERPRETATION = (
    "WITHIN_VIDEO_TRACK_DIAGNOSTIC_NOT_BIOLOGICAL_IDENTITY_VALIDATION_OR_FINAL_EVALUATION"
)
METHODS = {
    "A_single_best_raw": "single highest deterministic-quality raw frame from each exact triple",
    "B_three_raw_late_fusion": "L2-normalized mean of embeddings from each exact raw triple",
    "C_three_frame_restored_early_fusion": (
        "embedding of conservative restored pixel fusion from each exact raw triple"
    ),
    "D_quality_consensus_late_fusion": (
        "quality-weighted L2 mean after consensus outlier rejection from each exact raw triple"
    ),
    "E_unweighted_consensus_late_fusion": (
        "equal-weight L2 mean after consensus outlier rejection from each exact raw triple"
    ),
}
_CODE_PATHS = (
    "src/cvi/nose_id/restoration_evaluation.py",
    "src/cvi/nose_id/restoration.py",
    "src/cvi/nose_id/temporal.py",
    "src/cvi/evidence/artifact_manifest.py",
    "src/cvi/nose_region/native_yt.py",
    "tools/evaluate_yt_nose_restoration.py",
)


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


def _l2_normalize(vector: np.ndarray, context: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if value.ndim != 1 or not np.isfinite(value).all() or not math.isfinite(norm) or norm <= 0:
        raise ArtifactContractError(f"{context} produced a non-finite or zero-norm vector")
    return np.asarray(value / norm, dtype=np.float32)


def _quality_score(record: Mapping[str, Any]) -> float:
    quality = record["quality"]
    positive = (
        quality["blur_score"],
        quality["contrast_score"],
        quality["detector_confidence"],
        quality["frontality"],
        1.0 - quality["clipped_pixel_fraction"],
        1.0 - quality["specular_fraction"],
        1.0 - quality["jpeg_blocking_score"],
        1.0 - quality["noise_score"],
        1.0 - quality["mask_uncertainty"],
    )
    clipped = np.clip(np.asarray(positive, dtype=np.float64), 1e-6, 1.0)
    return float(np.exp(np.mean(np.log(clipped))))


def _group_temporal_splits(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    localized = [row for row in records if row["record_state"] != "NO_ROI"]
    by_identity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    track_owner: dict[str, str] = {}
    for row in localized:
        identity = row["registered_dog_id"]
        owner = track_owner.setdefault(row["track_token"], identity)
        if owner != identity:
            raise ValueError("one YT track maps to multiple registered identities")
        by_identity[identity].append(row)

    splits: list[dict[str, Any]] = []
    excluded = {"fewer_than_six_localized_frames": 0}
    for identity in sorted(by_identity):
        rows = by_identity[identity]
        identity_tokens = {row["identity_token"] for row in rows}
        tracks = {row["track_token"] for row in rows}
        sequences = {row["sequence_token"] for row in rows}
        if len(identity_tokens) != 1 or len(tracks) != 1 or len(sequences) != 1:
            raise ValueError("registered YT identity must map to exactly one identity/track/sequence")
        ordered = sorted(rows, key=lambda row: (row["frame_index"], row["sample_token"]))
        frame_indices = [row["frame_index"] for row in ordered]
        if len(frame_indices) != len(set(frame_indices)):
            raise ValueError("localized YT track repeats a frame index")
        if len(ordered) < 6:
            excluded["fewer_than_six_localized_frames"] += 1
            continue
        gallery = ordered[:3]
        query = ordered[-3:]
        gallery_tokens = {row["sample_token"] for row in gallery}
        query_tokens = {row["sample_token"] for row in query}
        if gallery_tokens & query_tokens or set(row["frame_index"] for row in gallery) & set(
            row["frame_index"] for row in query
        ):
            raise RuntimeError("deterministic temporal split contains frame overlap")
        splits.append(
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
    if len(splits) < 2:
        raise ValueError("evaluation requires at least two YT identities with six localized frames")
    return splits, excluded


def _load_resized_triple(
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    evaluation_size: int,
    mask_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for row in rows:
        crop_path = root / row["crop_path"]
        mask_path = root / row["binary_mask_path"]
        with Image.open(crop_path) as opened:
            resized = opened.convert("RGB").resize(
                (evaluation_size, evaluation_size), Image.Resampling.BILINEAR
            )
            images.append(np.asarray(resized, dtype=np.uint8))
        with Image.open(mask_path) as opened:
            resized_mask = opened.convert("L").resize(
                (evaluation_size, evaluation_size), Image.Resampling.NEAREST
            )
            mask = np.asarray(resized_mask, dtype=np.uint8) >= 128
            if not np.any(mask):
                raise ValueError("evaluation Nose segmentation mask is empty")
            masks.append(
                mask
                if mask_mode == "MANIFEST_BINARY"
                else np.ones(mask.shape, dtype=bool)
            )
    return np.stack(images), np.stack(masks)


def _embed(runtime: ExactOnnxRuntime, manifest: NoseEmbeddingManifest, image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("evaluation image must be uint8 HxWx3 RGB")
    output = runtime.run(preprocess_image(Image.fromarray(image, mode="RGB"), manifest))[0]
    return _l2_normalize(output, "nose embedding artifact")


def _distribution(
    values: np.ndarray, *, histogram_range: tuple[float, float] | None = None
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("score distribution must contain finite values")
    counts, edges = np.histogram(array, bins=20, range=histogram_range)
    quantiles = np.quantile(array, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "quantiles": {
            "q05": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q95": float(quantiles[4]),
        },
        "histogram": {"bin_edges": edges.tolist(), "counts": counts.tolist()},
    }


def _rank_rows(
    scores: np.ndarray, query_ids: Sequence[str], gallery_ids: Sequence[str]
) -> list[dict[str, Any]]:
    gallery_lookup = {identity: index for index, identity in enumerate(gallery_ids)}
    rows: list[dict[str, Any]] = []
    for query_index, identity in enumerate(query_ids):
        if identity not in gallery_lookup:
            raise ValueError("closed-set query identity is absent from gallery")
        order = np.argsort(-scores[query_index], kind="stable")
        positive = gallery_lookup[identity]
        rank = int(np.flatnonzero(order == positive)[0]) + 1
        impostors = np.delete(scores[query_index], positive)
        genuine = float(scores[query_index, positive])
        rows.append(
            {
                "rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "average_precision": 1.0 / rank,
                "rank_1": rank <= 1,
                "rank_5": rank <= 5,
                "genuine_score": genuine,
                "best_impostor_score": float(impostors.max()),
                "genuine_margin": genuine - float(impostors.max()),
            }
        )
    return rows


def _metrics(
    scores: np.ndarray, query_ids: Sequence[str], gallery_ids: Sequence[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _rank_rows(scores, query_ids, gallery_ids)
    values = lambda key: np.asarray([row[key] for row in rows], dtype=np.float64)
    gallery_lookup = {identity: index for index, identity in enumerate(gallery_ids)}
    genuine = np.asarray(
        [scores[index, gallery_lookup[identity]] for index, identity in enumerate(query_ids)]
    )
    impostor_mask = np.ones(scores.shape, dtype=bool)
    for index, identity in enumerate(query_ids):
        impostor_mask[index, gallery_lookup[identity]] = False
    return (
        {
            "query_count": len(query_ids),
            "gallery_count": len(gallery_ids),
            "Rank-1": float(values("rank_1").mean()),
            "Rank-5": float(values("rank_5").mean()),
            "mAP": float(values("average_precision").mean()),
            "MRR": float(values("reciprocal_rank").mean()),
            "genuine_scores": _distribution(genuine, histogram_range=(-1.0, 1.0)),
            "impostor_scores": _distribution(
                scores[impostor_mask], histogram_range=(-1.0, 1.0)
            ),
        },
        rows,
    )


def _subset_metrics(
    scores_by_method: Mapping[str, np.ndarray],
    identities: Sequence[str],
    indices: Sequence[int],
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.size == 0:
        return {"identity_count": 0, "status": "NO_ELIGIBLE_IDENTITIES"}
    query_ids = [identities[index] for index in selected]
    return {
        "identity_count": int(selected.size),
        "metrics": {
            method: _metrics(scores[selected], query_ids, identities)[0]
            for method, scores in scores_by_method.items()
        },
    }


def _strata(
    scores_by_method: Mapping[str, np.ndarray],
    identities: Sequence[str],
    native_sides: Sequence[float],
    quality_scores: Sequence[float],
) -> dict[str, Any]:
    native = np.asarray(native_sides, dtype=np.float64)
    quality = np.asarray(quality_scores, dtype=np.float64)
    native_masks = {
        "native_lt_96": native < 96,
        "native_96_159": (native >= 96) & (native < 160),
        "native_160_223": (native >= 160) & (native < 224),
        "native_ge_224": native >= 224,
    }
    quality_masks = {
        "quality_lt_0_50": quality < 0.50,
        "quality_0_50_0_74": (quality >= 0.50) & (quality < 0.75),
        "quality_ge_0_75": quality >= 0.75,
    }
    return {
        "identity_assignment": (
            "minimum native short side and mean deterministic quality across the paired six frames"
        ),
        "native_resolution": {
            name: _subset_metrics(scores_by_method, identities, np.flatnonzero(mask))
            for name, mask in native_masks.items()
        },
        "quality": {
            name: _subset_metrics(scores_by_method, identities, np.flatnonzero(mask))
            for name, mask in quality_masks.items()
        },
    }


def _delta(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        "genuine_score": float(current["genuine_score"] - baseline["genuine_score"]),
        "genuine_margin": float(current["genuine_margin"] - baseline["genuine_margin"]),
        "reciprocal_rank": float(current["reciprocal_rank"] - baseline["reciprocal_rank"]),
        "rank_improvement": float(baseline["rank"] - current["rank"]),
        "rank_1": float(current["rank_1"] - baseline["rank_1"]),
        "rank_5": float(current["rank_5"] - baseline["rank_5"]),
    }


def _delta_summary(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    return {
        field: _distribution(np.asarray([row[field] for row in rows], dtype=np.float64))
        for field in (
            "genuine_score",
            "genuine_margin",
            "reciprocal_rank",
            "rank_improvement",
            "rank_1",
            "rank_5",
        )
    }


def _restoration_summary(restorations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = np.asarray([len(item["accepted_indices"]) for item in restorations], dtype=np.float64)
    stability_mean = np.asarray([item["leave_one_out_mean"] for item in restorations])
    stability_max = np.asarray([item["leave_one_out_max"] for item in restorations])
    return {
        "restoration_count": len(restorations),
        "input_frame_count": 3 * len(restorations),
        "accepted_frame_count": int(accepted.sum()),
        "accepted_frames_per_restoration": _distribution(accepted),
        "leave_one_out_mean": _distribution(stability_mean),
        "leave_one_out_max": _distribution(stability_max),
    }


def _code_hashes(repository_root: Path) -> dict[str, str]:
    return {relative: _file_sha256(repository_root / relative) for relative in _CODE_PATHS}


def validate_report_bundle(bundle: object) -> dict[str, Any]:
    if not isinstance(bundle, dict) or set(bundle) != {
        "schema_version",
        "report_sha256",
        "report",
    }:
        raise ValueError("raw/restored evaluation report bundle schema differs")
    if bundle["schema_version"] != REPORT_BUNDLE_SCHEMA:
        raise ValueError("raw/restored evaluation report bundle version differs")
    _require_sha256(bundle["report_sha256"], "report_sha256")
    report = bundle["report"]
    expected = {
        "schema_version",
        "status",
        "interpretation",
        "input_bindings",
        "code_sha256s",
        "protocol",
        "population",
        "methods",
        "metrics",
        "restoration",
        "strata",
        "paired_per_identity",
        "paired_delta_summaries",
    }
    if not isinstance(report, dict) or set(report) != expected:
        raise ValueError("raw/restored evaluation report schema differs")
    if report["schema_version"] != REPORT_SCHEMA or report["interpretation"] != INTERPRETATION:
        raise ValueError("raw/restored evaluation interpretation differs")
    if content_sha256(report) != bundle["report_sha256"]:
        raise ValueError("raw/restored evaluation report digest differs")
    return report


def evaluate_raw_vs_restored(
    *,
    native_bundle_path: Path,
    native_bundle_sha256: str,
    native_root: Path,
    embedding_manifest_path: Path,
    embedding_manifest_sha256: str,
    embedding_onnx_path: Path,
    output_path: Path,
    evaluation_size: int = 224,
    use_cuda: bool = False,
    restoration_config: RestorationConfig | None = None,
    mask_mode: str = "FULL_CROP",
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate paired temporal triples and publish one no-overwrite report bundle."""

    if isinstance(evaluation_size, bool) or not isinstance(evaluation_size, int) or evaluation_size < 8:
        raise ValueError("evaluation_size must be an integer of at least 8")
    if mask_mode not in {"FULL_CROP", "MANIFEST_BINARY"}:
        raise ValueError("mask_mode must be FULL_CROP or MANIFEST_BINARY")
    expected_native = _require_sha256(native_bundle_sha256, "native_bundle_sha256")
    expected_embedding = _require_sha256(
        embedding_manifest_sha256, "embedding_manifest_sha256"
    )
    repository = (
        Path(__file__).resolve().parents[3]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    output_absolute = Path(os.path.abspath(os.fspath(output_path)))
    output_parent = output_absolute.parent.resolve(strict=True)
    resolved_output = output_parent / output_absolute.name
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
    manifest = validate_manifest_bundle(native_document.payload, root=root)

    embedding_document = read_strict_json_document(embedding_manifest_path)
    if embedding_document.canonical_payload_sha256 != expected_embedding:
        raise ValueError("embedding manifest content SHA-256 differs from the external pin")
    embedding_manifest = NoseEmbeddingManifest.from_dict(embedding_document.payload)
    runtime = ExactOnnxRuntime(embedding_onnx_path, embedding_manifest, use_cuda=use_cuda)
    onnx_binding = _file_binding(embedding_onnx_path)
    if onnx_binding["sha256"] != embedding_manifest.artifact_sha256:
        raise ArtifactContractError("artifact SHA256 does not match its manifest")

    splits, excluded = _group_temporal_splits(manifest["records"])
    settings = restoration_config or RestorationConfig()
    identities = [split["registered_dog_id"] for split in splits]
    vectors: dict[str, dict[str, list[np.ndarray]]] = {
        method: {"gallery": [], "query": []} for method in METHODS
    }
    split_reports: list[dict[str, Any]] = []
    all_restorations: list[dict[str, Any]] = []
    native_sides: list[float] = []
    quality_scores: list[float] = []

    for split in splits:
        split_report: dict[str, Any] = {
            key: split[key]
            for key in (
                "registered_dog_id",
                "identity_token",
                "track_token",
                "sequence_token",
                "localized_frame_count",
            )
        }
        six_rows = [*split["gallery"], *split["query"]]
        native_sides.append(float(min(row["quality"]["native_short_side"] for row in six_rows)))
        quality_scores.append(float(np.mean([_quality_score(row) for row in six_rows])))
        for role in ("gallery", "query"):
            rows = split[role]
            images, masks = _load_resized_triple(
                rows, root, evaluation_size, mask_mode
            )
            frame_embeddings = [_embed(runtime, embedding_manifest, image) for image in images]
            best_index = min(
                range(3), key=lambda index: (-_quality_score(rows[index]), index)
            )
            vectors["A_single_best_raw"][role].append(frame_embeddings[best_index])
            baseline_temporal = aggregate_nose_embeddings(frame_embeddings)
            vectors["B_three_raw_late_fusion"][role].append(
                baseline_temporal.embedding
            )
            temporal = aggregate_nose_embeddings(
                frame_embeddings,
                [_quality_score(row) for row in rows],
                reject_outliers=True,
            )
            vectors["D_quality_consensus_late_fusion"][role].append(
                temporal.embedding
            )
            unweighted_temporal = aggregate_nose_embeddings(
                frame_embeddings, reject_outliers=True
            )
            vectors["E_unweighted_consensus_late_fusion"][role].append(
                unweighted_temporal.embedding
            )
            restoration = restore_nose_frames(images, masks, config=settings)
            restored_uint8 = np.rint(np.clip(restoration.restored_rgb, 0.0, 1.0) * 255.0).astype(
                np.uint8
            )
            vectors["C_three_frame_restored_early_fusion"][role].append(
                _embed(runtime, embedding_manifest, restored_uint8)
            )
            diagnostics = restoration.diagnostics.to_dict()
            all_restorations.append(diagnostics)
            split_report[role] = {
                "sample_tokens": [row["sample_token"] for row in rows],
                "frame_indices": [row["frame_index"] for row in rows],
                "single_best_sample_token": rows[best_index]["sample_token"],
                "deterministic_quality_scores": [_quality_score(row) for row in rows],
                "restoration_diagnostics": diagnostics,
                "baseline_temporal_embedding_diagnostics": (
                    baseline_temporal.diagnostics()
                ),
                "temporal_embedding_diagnostics": temporal.diagnostics(),
                "unweighted_temporal_embedding_diagnostics": (
                    unweighted_temporal.diagnostics()
                ),
            }
        split_reports.append(split_report)

    scores_by_method: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    rank_rows: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        gallery = np.stack(vectors[method]["gallery"])
        query = np.stack(vectors[method]["query"])
        if gallery.shape[0] != len(identities) or query.shape[0] != len(identities):
            raise RuntimeError("evaluation did not build exactly one gallery/query vector per identity")
        scores = compute_cosine_score_matrix(query, gallery)
        scores_by_method[method] = scores
        metrics[method], rank_rows[method] = _metrics(scores, identities, identities)

    delta_names = {
        "B_minus_A": ("B_three_raw_late_fusion", "A_single_best_raw"),
        "C_minus_A": ("C_three_frame_restored_early_fusion", "A_single_best_raw"),
        "C_minus_B": (
            "C_three_frame_restored_early_fusion",
            "B_three_raw_late_fusion",
        ),
        "D_minus_A": ("D_quality_consensus_late_fusion", "A_single_best_raw"),
        "D_minus_B": (
            "D_quality_consensus_late_fusion",
            "B_three_raw_late_fusion",
        ),
        "E_minus_B": (
            "E_unweighted_consensus_late_fusion",
            "B_three_raw_late_fusion",
        ),
    }
    deltas: dict[str, list[dict[str, float]]] = {name: [] for name in delta_names}
    for index, split_report in enumerate(split_reports):
        outcomes = {method: rank_rows[method][index] for method in METHODS}
        identity_deltas = {
            name: _delta(outcomes[current], outcomes[baseline])
            for name, (current, baseline) in delta_names.items()
        }
        for name, value in identity_deltas.items():
            deltas[name].append(value)
        split_report["method_outcomes"] = outcomes
        split_report["paired_deltas"] = identity_deltas

    pairing_contract = [
        {
            "registered_dog_id": row["registered_dog_id"],
            "track_token": row["track_token"],
            "gallery_sample_tokens": row["gallery"]["sample_tokens"],
            "query_sample_tokens": row["query"]["sample_tokens"],
        }
        for row in split_reports
    ]
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_WITHIN_VIDEO_TRACK_DIAGNOSTIC",
        "interpretation": INTERPRETATION,
        "input_bindings": {
            "native_manifest_bundle": {
                "path": os.fspath(native_bundle_path),
                "raw_sha256": native_document.raw_sha256,
                "content_sha256": native_document.canonical_payload_sha256,
                "manifest_sha256": native_document.payload["manifest_sha256"],
                "byte_size": native_document.byte_size,
            },
            "native_artifact_root": os.fspath(root),
            "embedding_manifest": {
                "path": os.fspath(embedding_manifest_path),
                "raw_sha256": embedding_document.raw_sha256,
                "content_sha256": embedding_document.canonical_payload_sha256,
                "byte_size": embedding_document.byte_size,
            },
            "embedding_onnx": onnx_binding,
        },
        "code_sha256s": _code_hashes(repository),
        "protocol": {
            "grouping_unit": "registered_dog_id_with_one_to_one_YT_identity_track_sequence",
            "localized_states": ["AVAILABLE", "LOW_QUALITY"],
            "minimum_localized_frames": 6,
            "temporal_order": ["frame_index", "sample_token"],
            "gallery_selection": "earliest_three",
            "query_selection": "latest_three",
            "frame_overlap_allowed": False,
            "frames_per_vector": 3,
            "gallery_vectors_per_identity": 1,
            "query_vectors_per_identity": 1,
            "single_best_quality": (
                "geometric_mean_of_blur_score_contrast_score_detector_confidence_"
                "frontality_and_one_minus_clipped_specular_jpeg_blocking_noise_mask_uncertainty;_"
                "ties_choose_earliest_within_the_temporal_triple"
            ),
            "late_fusion": (
                "L2_normalize_each_of_three_embeddings_then_arithmetic_mean_then_L2_normalize"
            ),
            "early_fusion": (
                "restore_from_the_same_exact_three_fully_observed_resized_crops_with_internal_"
                "glare_masking_then_embed_and_L2_normalize"
            ),
            "evaluation_size": [evaluation_size, evaluation_size],
            "crop_resize": "PIL_bilinear_RGB_before_raw_embedding_and_restoration",
            "segmentation_mask_use": (
                "manifest_binary_mask_defines_observed_source_support"
                if mask_mode == "MANIFEST_BINARY"
                else "validated_nonempty_but_not_used_as_source_validity;_the_entire_crop_is_observed"
            ),
            "mask_mode": mask_mode,
            "retrieval": "exhaustive_cosine_one_gallery_vector_per_identity",
            "tie_policy": "stable_registered_dog_id_lexical_order",
            "pairing_sha256": content_sha256(pairing_contract),
            "restoration_config": settings.to_dict(),
        },
        "population": {
            "localized_identity_count": len({row["registered_dog_id"] for row in manifest["records"] if row["record_state"] != "NO_ROI"}),
            "eligible_identity_count": len(identities),
            "excluded_identity_counts": excluded,
            "gallery_vector_count_per_method": len(identities),
            "query_vector_count_per_method": len(identities),
        },
        "methods": METHODS,
        "metrics": metrics,
        "restoration": _restoration_summary(all_restorations),
        "strata": _strata(
            scores_by_method, identities, native_sides, quality_scores
        ),
        "paired_per_identity": split_reports,
        "paired_delta_summaries": {
            name: _delta_summary(values) for name, values in deltas.items()
        },
    }
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
    "evaluate_raw_vs_restored",
    "validate_report_bundle",
]
