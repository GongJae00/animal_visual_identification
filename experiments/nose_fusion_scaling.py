"""Fixed-population within-track evaluation of Nose temporal fusion scaling."""

from __future__ import annotations

import hashlib
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from contracts.artifact_manifest import (
    ArtifactContractError,
    ExactOnnxRuntime,
    NoseEmbeddingManifest,
    preprocess_image,
)
from evaluation.retrieval import (
    compute_cosine_score_matrix,
    identity_clustered_bootstrap_ci,
)
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.provenance import content_sha256
from embedding.methods.nose.signal.temporal import aggregate_nose_embeddings
from parsing.nose_region.native_yt import validate_manifest_bundle

REPORT_SCHEMA = "cvi.yt_nose_fusion_scaling_evaluation.v1"
REPORT_BUNDLE_SCHEMA = "cvi.yt_nose_fusion_scaling_evaluation_bundle.v1"
INTERPRETATION = (
    "WITHIN_VIDEO_TRACK_TEMPORAL_FUSION_SCALING_DIAGNOSTIC_"
    "NOT_BIOLOGICAL_IDENTITY_VALIDATION_OR_FINAL_EVALUATION"
)
SCALES = (1, 3, 5)
METHODS = {
    "K1": "earliest/latest one L2-normalized Nose embedding",
    "K3": "unweighted L2-normalized mean of earliest/latest three Nose embeddings",
    "K5": "unweighted L2-normalized mean of earliest/latest five Nose embeddings",
}
_CONTRASTS = {
    "K3_minus_K1": ("K3", "K1"),
    "K5_minus_K1": ("K5", "K1"),
    "K5_minus_K3": ("K5", "K3"),
}
_BOOTSTRAP_METRICS = ("Rank-1", "Rank-5", "MRR", "mAP")
_CODE_PATHS = (
    "experiments/nose_fusion_scaling.py",
    "embedding/methods/nose/signal/temporal.py",
    "contracts/artifact_manifest.py",
    "evaluation/retrieval.py",
    "parsing/nose_region/native_yt.py",
    "foundation/protected_io.py",
    "foundation/provenance.py",
    "workflows/evaluate_yt_nose_fusion_scaling.py",
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


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if value.ndim != 1 or not np.isfinite(value).all() or not math.isfinite(norm) or norm <= 0:
        raise ArtifactContractError("Nose embedding artifact produced an invalid vector")
    return np.asarray(value / norm, dtype=np.float32)


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


def _embed(
    runtime: ExactOnnxRuntime,
    manifest: NoseEmbeddingManifest,
    root: Path,
    row: Mapping[str, Any],
) -> np.ndarray:
    with Image.open(root / row["crop_path"]) as opened:
        tensor = preprocess_image(opened, manifest)
    return _l2_normalize(runtime.run(tensor)[0])


def _fuse(embeddings: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    if len(embeddings) == 1:
        return embeddings[0], {
            "aggregation": "SINGLE_L2_NORMALIZED_EMBEDDING",
            "temporal_api_invoked": False,
        }
    result = aggregate_nose_embeddings(embeddings)
    if result.aggregation != "UNWEIGHTED_L2_MEAN":
        raise RuntimeError("temporal aggregation did not use strict unweighted L2 mean")
    diagnostics = result.diagnostics()
    diagnostics["temporal_api_invoked"] = True
    return result.embedding, diagnostics


def _rank_rows(
    scores: np.ndarray, identities: Sequence[str]
) -> list[dict[str, Any]]:
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
            for metric in _BOOTSTRAP_METRICS
        },
    }


def _paired_delta(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    return {
        metric: float(current[metric] - baseline[metric])
        for metric in _BOOTSTRAP_METRICS
    } | {
        "rank_improvement": float(baseline["rank"] - current["rank"]),
        "genuine_score": float(current["genuine_score"] - baseline["genuine_score"]),
        "genuine_margin": float(current["genuine_margin"] - baseline["genuine_margin"]),
    }


def _code_hashes(repository_root: Path) -> dict[str, str]:
    return {relative: _file_sha256(repository_root / relative) for relative in _CODE_PATHS}


def validate_report_bundle(bundle: object) -> dict[str, Any]:
    if not isinstance(bundle, dict) or set(bundle) != {
        "schema_version",
        "report_sha256",
        "report",
    }:
        raise ValueError("fusion scaling evaluation report bundle schema differs")
    if bundle["schema_version"] != REPORT_BUNDLE_SCHEMA:
        raise ValueError("fusion scaling evaluation report bundle version differs")
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
        "paired_per_identity",
        "paired_delta_bootstrap_cis",
    }
    if not isinstance(report, dict) or set(report) != expected:
        raise ValueError("fusion scaling evaluation report schema differs")
    if report["schema_version"] != REPORT_SCHEMA or report["interpretation"] != INTERPRETATION:
        raise ValueError("fusion scaling evaluation interpretation differs")
    if report["methods"] != METHODS:
        raise ValueError("fusion scaling evaluation methods differ")
    for name, digest in report["input_sha256s"].items():
        _require_sha256(digest, f"input_sha256s.{name}")
    for name, digest in report["code_sha256s"].items():
        _require_sha256(digest, f"code_sha256s.{name}")
    _require_sha256(report["protocol"]["pairing_sha256"], "pairing_sha256")
    if content_sha256(report) != bundle["report_sha256"]:
        raise ValueError("fusion scaling evaluation report digest differs")
    return report


def evaluate_fusion_scaling(
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
    bootstrap_confidence_level: float = 0.95,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate K=1/3/5 on one fixed population and publish a private JSON bundle."""

    expected_native = _require_sha256(native_bundle_sha256, "native_bundle_sha256")
    expected_runtime = _require_sha256(
        nose_runtime_manifest_sha256, "nose_runtime_manifest_sha256"
    )
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if not isinstance(bootstrap_confidence_level, (int, float)) or isinstance(
        bootstrap_confidence_level, bool
    ) or not 0.0 < bootstrap_confidence_level < 1.0:
        raise ValueError("bootstrap_confidence_level must be in (0, 1)")

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

    runtime_document = read_strict_json_document(nose_runtime_manifest_path)
    if runtime_document.canonical_payload_sha256 != expected_runtime:
        raise ValueError("Nose runtime manifest content SHA-256 differs from the external pin")
    runtime_manifest = NoseEmbeddingManifest.from_dict(runtime_document.payload)
    runtime = ExactOnnxRuntime(nose_onnx_path, runtime_manifest, use_cuda=use_cuda)
    onnx_binding = _file_binding(nose_onnx_path)
    if onnx_binding["sha256"] != runtime_manifest.artifact_sha256:
        raise ArtifactContractError("artifact SHA256 does not match its manifest")

    population, excluded = _group_population(native_manifest["records"])
    identities = [item["registered_dog_id"] for item in population]
    vectors: dict[str, dict[str, list[np.ndarray]]] = {
        method: {"gallery": [], "query": []} for method in METHODS
    }
    identity_reports: list[dict[str, Any]] = []
    for item in population:
        identity_report = {
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
            five_rows = item[role]
            frame_embeddings = [
                _embed(runtime, runtime_manifest, root, row) for row in five_rows
            ]
            role_report: dict[str, Any] = {}
            for scale in SCALES:
                method = f"K{scale}"
                selected_rows = five_rows[:scale] if role == "gallery" else five_rows[-scale:]
                selected_embeddings = (
                    frame_embeddings[:scale]
                    if role == "gallery"
                    else frame_embeddings[-scale:]
                )
                fused, diagnostics = _fuse(selected_embeddings)
                vectors[method][role].append(fused)
                role_report[method] = {
                    "sample_tokens": [row["sample_token"] for row in selected_rows],
                    "frame_indices": [row["frame_index"] for row in selected_rows],
                    "fusion_diagnostics": diagnostics,
                }
            identity_report[role] = role_report
        identity_reports.append(identity_report)

    metrics: dict[str, Any] = {}
    outcomes: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        gallery = np.stack(vectors[method]["gallery"])
        query = np.stack(vectors[method]["query"])
        scores = compute_cosine_score_matrix(query, gallery)
        outcomes[method] = _rank_rows(scores, identities)
        metrics[method] = _metrics(outcomes[method], len(identities))

    paired_rows: dict[str, list[dict[str, Any]]] = {
        contrast: [] for contrast in _CONTRASTS
    }
    for index, identity_report in enumerate(identity_reports):
        identity_report["method_outcomes"] = {
            method: outcomes[method][index] for method in METHODS
        }
        identity_report["paired_deltas"] = {}
        for contrast, (current, baseline) in _CONTRASTS.items():
            delta = _paired_delta(outcomes[current][index], outcomes[baseline][index])
            identity_report["paired_deltas"][contrast] = delta
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
            for metric in _BOOTSTRAP_METRICS
        }
        for contrast, rows in paired_rows.items()
    }
    pairing_contract = [
        {
            "registered_dog_id": item["registered_dog_id"],
            "identity_token": item["identity_token"],
            "track_token": item["track_token"],
            "sequence_token": item["sequence_token"],
            "gallery": {
                method: identity_report["gallery"][method]["sample_tokens"]
                for method in METHODS
            },
            "query": {
                method: identity_report["query"][method]["sample_tokens"]
                for method in METHODS
            },
        }
        for item, identity_report in zip(population, identity_reports, strict=True)
    ]
    input_sha256s = {
        "native_manifest_bundle_content": native_document.canonical_payload_sha256,
        "native_manifest": native_document.payload["manifest_sha256"],
        "nose_runtime_manifest_content": runtime_document.canonical_payload_sha256,
        "nose_onnx": onnx_binding["sha256"],
    }
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
            "nose_runtime_manifest": {
                "path": os.fspath(nose_runtime_manifest_path),
                "raw_sha256": runtime_document.raw_sha256,
                "content_sha256": runtime_document.canonical_payload_sha256,
                "byte_size": runtime_document.byte_size,
            },
            "nose_onnx": onnx_binding,
        },
        "input_sha256s": input_sha256s,
        "code_sha256s": _code_hashes(repository),
        "protocol": {
            "grouping_unit": "registered_dog_id_with_one_to_one_YT_identity_track_sequence",
            "localized_states": ["AVAILABLE", "LOW_QUALITY"],
            "minimum_localized_frames": 10,
            "fixed_common_population_across_scales": True,
            "temporal_order": ["frame_index", "sample_token"],
            "scales": list(SCALES),
            "gallery_selection": "earliest_K",
            "query_selection": "latest_K",
            "frame_overlap_allowed": False,
            "vectors_per_identity_per_role": 1,
            "fusion": (
                "L2_normalize_each_embedding_then_unweighted_arithmetic_mean_then_"
                "L2_normalize;_K1_is_the_singleton_identity_operation"
            ),
            "temporal_implementation": "embedding.methods.nose.signal.temporal.aggregate_nose_embeddings",
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
        "paired_per_identity": identity_reports,
        "paired_delta_bootstrap_cis": bootstrap_cis,
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
    "SCALES",
    "evaluate_fusion_scaling",
    "validate_report_bundle",
]
