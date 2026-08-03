"""Bounded residual metric learning over frozen N3 embedding vectors.

This candidate changes only normalized embedding-space geometry. Track tokens
are deterministic proxy labels for same-track retrieval; they are not lifelong
dog identities, and this module makes no physical nose-ridge topology claim.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from evaluation.retrieval import identity_clustered_bootstrap_ci
from experiments.fixed_multievidence import (
    FRAMES_PER_WINDOW,
    METHODS,
    file_sha256,
    read_fixed_panel,
    validate_fixed_topology_bindings,
    validate_panel_bundle,
)
from experiments.identity_topology import (
    IDENTITY_TOPOLOGY_MANIFEST_SCHEMA_VERSION,
    audit_identity_topology,
    validate_identity_topology_manifest,
)
from foundation.protected_io import (
    json_document_bytes,
    read_strict_json_document,
    write_private_json_bundle,
)
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256

CACHE_SCHEMA_VERSION = "cvi.n4_metric_embedding_cache.v1"
CACHE_BUNDLE_SCHEMA_VERSION = "cvi.n4_metric_embedding_cache_bundle.v1"
CHECKPOINT_SCHEMA_VERSION = "cvi.n4_metric_adapter_checkpoint.v1"
REPORT_SCHEMA_VERSION = "cvi.n4_metric_adapter_evaluation.v1"
REPORT_BUNDLE_SCHEMA_VERSION = "cvi.n4_metric_adapter_evaluation_bundle.v1"
N3_BRANCH = METHODS[2]
METRICS = ("Rank-1", "MRR", "Rank-5")
MAXIMUM_SCALE = 0.1
MAXIMUM_BOTTLENECK = 128
NORMALIZATION_TOLERANCE = 1e-4
LIMITATIONS = (
    "SAME_VIDEO_TRACK_GALLERY_AND_QUERY",
    "TRACK_LABELS_ARE_NOT_LIFELONG_IDENTITIES",
    "PUBLISHER_TEST_EXPOSED_DIAGNOSTIC",
    "WEAK_NOSE_ROI_INPUT",
    "CLOSED_SET_ONLY_NO_UNKNOWN_REJECTION",
    "NO_PHYSICAL_NOSE_TOPOLOGY_CLAIM",
    "NO_BIOMETRIC_VALIDATION_CLAIM",
)
_CACHE_FILES = ("embeddings.npy", "manifest.json")
_CACHE_CODE_PATHS = (
    "experiments/n4_metric_adapter.py",
    "workflows/materialize_n4_embedding_cache.py",
)
_TRAIN_CODE_PATHS = (
    "experiments/n4_metric_adapter.py",
    "workflows/train_n4_metric_adapter.py",
)
_EVALUATION_CODE_PATHS = (
    "experiments/n4_metric_adapter.py",
    "workflows/evaluate_n4_metric_adapter.py",
)


class ResidualMetricAdapter(nn.Module):
    """Small readable adapter implementing normalize(x + scale * A(x))."""

    def __init__(self, input_dim: int, bottleneck_dim: int, *, scale: float) -> None:
        super().__init__()
        if (
            isinstance(input_dim, bool)
            or not isinstance(input_dim, int)
            or input_dim < 1
        ):
            raise ValueError("adapter input_dim must be a positive integer")
        if (
            isinstance(bottleneck_dim, bool)
            or not isinstance(bottleneck_dim, int)
            or not 1 <= bottleneck_dim <= min(input_dim, MAXIMUM_BOTTLENECK)
        ):
            raise ValueError("adapter bottleneck must be in [1, min(input_dim, 128)]")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or not 0.0 < float(scale) <= MAXIMUM_SCALE
        ):
            raise ValueError("adapter scale must be finite and in (0, 0.1]")
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.scale = float(scale)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.down.in_features:
            raise ValueError("adapter input shape differs from its fixed dimension")
        residual = self.up(F.relu(self.down(embeddings.float())))
        return F.normalize(embeddings.float() + self.scale * residual, p=2, dim=1)


class DeterministicPKBatchSampler:
    """Yield deterministic P-track/K-frame batches with replacement by cycling."""

    def __init__(
        self,
        labels: Sequence[str],
        *,
        tracks_per_batch: int,
        samples_per_track: int,
        seed: int,
    ) -> None:
        if not labels or any(
            not isinstance(label, str) or not label for label in labels
        ):
            raise ValueError("P/K labels must be non-empty track tokens")
        for name, value in (
            ("tracks_per_batch", tracks_per_batch),
            ("samples_per_track", samples_per_track),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("P/K seed must be a non-negative integer")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            grouped[label].append(index)
        if len(grouped) < tracks_per_batch:
            raise ValueError("P/K training has fewer tracks than tracks_per_batch")
        if any(len(indices) < 2 for indices in grouped.values()):
            raise ValueError("P/K training requires at least two frames per track")
        self._grouped = dict(sorted(grouped.items()))
        self._p = tracks_per_batch
        self._k = samples_per_track
        self._seed = seed

    def batches(self, epoch: int) -> tuple[tuple[int, ...], ...]:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("P/K epoch must be a non-negative integer")
        generator = torch.Generator().manual_seed(self._seed + epoch * 1_000_003)
        tracks = list(self._grouped)
        track_order = torch.randperm(len(tracks), generator=generator).tolist()
        batch_count = max(1, math.ceil(len(tracks) / self._p))
        result: list[tuple[int, ...]] = []
        for batch_index in range(batch_count):
            selected_tracks = [
                tracks[track_order[(batch_index * self._p + offset) % len(tracks)]]
                for offset in range(self._p)
            ]
            indices: list[int] = []
            for track_offset, track in enumerate(selected_tracks):
                candidates = self._grouped[track]
                local = torch.randperm(len(candidates), generator=generator).tolist()
                start = (epoch + batch_index + track_offset) % len(candidates)
                indices.extend(
                    candidates[local[(start + sample) % len(local)]]
                    for sample in range(self._k)
                )
            result.append(tuple(indices))
        return tuple(result)


def batch_hard_metric_loss(
    adapted: torch.Tensor,
    parent: torch.Tensor,
    labels: Sequence[str],
    quality_weights: torch.Tensor,
    *,
    margin: float,
    anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Batch-hard track-proxy loss plus a bounded frozen-parent cosine anchor."""

    if (
        adapted.ndim != 2
        or parent.shape != adapted.shape
        or len(labels) != len(adapted)
    ):
        raise ValueError("metric loss embeddings and labels are not aligned")
    if quality_weights.shape != (len(adapted),):
        raise ValueError("metric loss quality weights are not aligned")
    if not torch.isfinite(adapted).all() or not torch.isfinite(parent).all():
        raise ValueError("metric loss embeddings must be finite")
    if not torch.isfinite(quality_weights).all() or torch.any(
        (quality_weights < 0.5) | (quality_weights > 1.0)
    ):
        raise ValueError("metric quality weights must be bounded in [0.5, 1]")
    if not 0.0 < margin < 2.0 or not 0.0 <= anchor_weight <= 1.0:
        raise ValueError("metric margin or parent-anchor weight differs")
    label_values = list(labels)
    positive = torch.tensor(
        [[left == right for right in label_values] for left in label_values],
        dtype=torch.bool,
        device=adapted.device,
    )
    positive.fill_diagonal_(False)
    negative = ~positive
    negative.fill_diagonal_(False)
    if not positive.any(dim=1).all() or not negative.any(dim=1).all():
        raise ValueError(
            "batch-hard loss requires a positive and hard negative per row"
        )
    similarity = adapted @ adapted.T
    hardest_positive = similarity.masked_fill(~positive, torch.inf).min(dim=1).values
    hardest_negative = similarity.masked_fill(~negative, -torch.inf).max(dim=1).values
    triplet_rows = F.relu((1.0 - hardest_positive) - (1.0 - hardest_negative) + margin)
    denominator = quality_weights.sum().clamp_min(1e-12)
    metric = (triplet_rows * quality_weights).sum() / denominator
    anchor_rows = (1.0 - F.cosine_similarity(adapted, parent, dim=1)).clamp(0.0, 2.0)
    anchor = (anchor_rows * quality_weights).sum() / denominator
    total = metric + anchor_weight * anchor
    if total.ndim != 0 or not torch.isfinite(total):
        raise RuntimeError("metric adapter loss became non-finite")
    return total, {"batch_hard_metric": metric, "parent_anchor": anchor}


def evaluate_k5(
    embeddings: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate exact earliest/latest K5 means using track proxy labels."""

    matrix = _normalized_float32_matrix(embeddings, len(rows), "K5 embeddings")
    grouped: dict[str, list[int]] = defaultdict(list)
    registered_by_track: dict[str, str] = {}
    for index, row in enumerate(rows):
        track = _text(row.get("track_token"), "K5 track_token")
        registered = _text(
            row.get("registered_identity_id"), "K5 registered_identity_id"
        )
        if track in registered_by_track and registered_by_track[track] != registered:
            raise ValueError("one track token maps to multiple registered identities")
        registered_by_track[track] = registered
        grouped[track].append(index)
    tracks = sorted(grouped)
    if len(tracks) < 2:
        raise ValueError("K5 evaluation requires at least two tracks")
    gallery: list[np.ndarray] = []
    query: list[np.ndarray] = []
    windows: list[dict[str, Any]] = []
    for track in tracks:
        indices = sorted(
            grouped[track],
            key=lambda index: (
                _nonnegative_int(rows[index].get("frame_index"), "frame_index"),
                _text(rows[index].get("sample_token"), "sample_token"),
            ),
        )
        if len(indices) < 2 * FRAMES_PER_WINDOW:
            raise ValueError("K5 track has fewer than ten frames")
        gallery_indices = indices[:FRAMES_PER_WINDOW]
        query_indices = indices[-FRAMES_PER_WINDOW:]
        if set(gallery_indices) & set(query_indices):
            raise RuntimeError("earliest/latest K5 windows overlap")
        gallery.append(_normalized_mean(matrix[gallery_indices]))
        query.append(_normalized_mean(matrix[query_indices]))
        windows.append(
            {
                "registered_identity_id": registered_by_track[track],
                "track_token": track,
                "gallery_sample_tokens": [
                    rows[index]["sample_token"] for index in gallery_indices
                ],
                "query_sample_tokens": [
                    rows[index]["sample_token"] for index in query_indices
                ],
            }
        )
    scores = np.stack(query) @ np.stack(gallery).T
    outcomes = _rank_rows(scores, tracks, registered_by_track)
    return {
        "aggregation": "EARLIEST_K5_GALLERY_LATEST_K5_QUERY_NORMALIZED_MEAN",
        "proxy_label_semantics": "TRACK_TOKEN_NOT_LIFELONG_DOG_IDENTITY",
        "track_count": len(tracks),
        "metrics": _metric_summary(outcomes),
        "outcomes": outcomes,
        "windows": windows,
    }


def select_dev_epoch(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select DEV-only metrics with strict Rank-1/MRR epoch-zero floors."""

    if not history or history[0].get("epoch") != 0:
        raise ValueError("adapter history must begin with epoch zero")
    baseline = _history_metrics(history[0])
    decisions: list[dict[str, Any]] = []
    candidates: list[
        tuple[tuple[float, float, float, float, int], Mapping[str, Any]]
    ] = []
    seen: set[int] = set()
    for item in history:
        epoch = _nonnegative_int(item.get("epoch"), "selection epoch")
        if epoch in seen:
            raise ValueError("selection epochs must be unique")
        seen.add(epoch)
        metrics = _history_metrics(item)
        change = _finite_number(item.get("adapter_change"), "adapter_change", 0.0, 2.0)
        admissible = (
            metrics["Rank-1"] >= baseline["Rank-1"]
            and metrics["MRR"] >= baseline["MRR"]
        )
        key = (
            metrics["Rank-1"],
            metrics["MRR"],
            metrics["Rank-5"],
            -change,
            -epoch,
        )
        decisions.append(
            {
                "epoch": epoch,
                "admissible": admissible,
                "metrics": metrics,
                "adapter_change": change,
                "objective": list(key),
            }
        )
        if admissible:
            candidates.append((key, item))
    if not candidates:
        raise RuntimeError("epoch zero unexpectedly failed its own non-regression gate")
    objective, selected = max(candidates, key=lambda pair: pair[0])
    return {
        "selected_epoch": selected["epoch"],
        "epoch0_metrics": baseline,
        "selected_metrics": _history_metrics(selected),
        "non_regression_gate": ["Rank-1 >= EPOCH0", "MRR >= EPOCH0"],
        "objective_order": ["Rank-1", "MRR", "Rank-5"],
        "tie_break": ["SMALLER_ADAPTER_CHANGE", "EARLIER_EPOCH_INCLUDING_EPOCH0"],
        "selected_objective": list(objective),
        "labels_used": "LINEAGE_DEV_ONLY",
        "lineage_eval_labels_used": False,
        "publisher_test_labels_used": False,
        "decisions": decisions,
    }


def materialize_embedding_cache(
    *,
    native_bundle_path: Path,
    native_bundle_sha256: str,
    native_root: Path,
    n3_lineage_path: Path,
    n3_lineage_sha256: str,
    n3_runtime_manifest_path: Path,
    n3_runtime_manifest_sha256: str,
    n3_onnx_path: Path,
    n3_onnx_sha256: str,
    output_dir: Path,
    use_cuda: bool = False,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Run exact N3 ONNX on lineage TRAIN/DEV crops and publish a safe cache."""

    repository = _repository(repository_root)
    output = _new_external_path(output_dir, repository, "N4 embedding cache")
    if not isinstance(use_cuda, bool):
        raise TypeError("use_cuda must be boolean")
    native_root = _absolute_directory(native_root, "native root")
    native_document = _pinned_json(
        native_bundle_path, native_bundle_sha256, "native bundle", large=True
    )
    lineage_document = _pinned_json(n3_lineage_path, n3_lineage_sha256, "N3 lineage")
    runtime_document = _pinned_json(
        n3_runtime_manifest_path,
        n3_runtime_manifest_sha256,
        "N3 runtime manifest",
    )
    from artifact_contracts.artifact_manifest import (
        ExactOnnxRuntime,
        NoseEmbeddingManifest,
        UsageLane,
        preprocess_image,
    )
    from localization.nose_region.embedding_consistency_training import (
        validate_lineage_manifest,
    )
    from localization.nose_region.native_yt import validate_manifest_bundle

    manifest = validate_manifest_bundle(native_document.payload, root=native_root)
    lineage_root = n3_lineage_path.parent.resolve(strict=True)
    validate_lineage_manifest(lineage_document.payload, lineage_root)
    artifacts = lineage_document.payload["artifacts"]
    _require_same_file(
        n3_runtime_manifest_path,
        lineage_root / artifacts["runtime_manifest"]["path"],
        "N3 runtime manifest",
    )
    _require_same_file(
        n3_onnx_path, lineage_root / artifacts["onnx"]["path"], "N3 ONNX"
    )
    onnx_pin = _sha256(n3_onnx_sha256, "N3 ONNX external pin")
    if file_sha256(n3_onnx_path) != onnx_pin or onnx_pin != artifacts["onnx"]["sha256"]:
        raise ValueError("N3 ONNX SHA-256 differs from lineage or external pin")
    runtime_manifest = NoseEmbeddingManifest.from_dict(runtime_document.payload)
    if runtime_manifest.license.usage_lane != UsageLane.RESEARCH_ONLY:
        raise ValueError("N3 runtime must remain research-only")
    if runtime_manifest.artifact_sha256 != onnx_pin:
        raise ValueError("N3 runtime manifest and ONNX digest differ")
    native_binding = lineage_document.payload["bindings"]["native_v4_bundle"]
    if (
        native_binding["payload_sha256"] != native_document.canonical_payload_sha256
        or native_binding["manifest_sha256"]
        != native_document.payload["manifest_sha256"]
        or native_binding["input_sha256s"] != manifest["input_sha256s"]
    ):
        raise ValueError("native bundle differs from the N3 lineage binding")

    splits = lineage_document.payload["bindings"]["splits"]
    identity_lists = splits["identity_lists"]
    sample_lists = splits["sample_token_lists"]
    if set(identity_lists["ssl_train"]) & set(identity_lists["dev"]):
        raise ValueError("lineage TRAIN and DEV identities overlap")
    selected_by_token = {row["sample_token"]: row for row in manifest["records"]}
    partition_samples = {
        role: set(sample_lists[role])
        for role in ("ssl_train", "dev", "eval", "excluded_no_roi")
    }
    if any(
        left_samples & partition_samples[right]
        for left_index, (left, left_samples) in enumerate(partition_samples.items())
        for right in tuple(partition_samples)[left_index + 1 :]
    ):
        raise ValueError("N3 lineage sample roles overlap")
    if set().union(*partition_samples.values()) != set(selected_by_token):
        raise ValueError("N3 lineage sample roles do not cover the exact native bundle")
    localized = {"AVAILABLE", "LOW_QUALITY"}
    train_candidates = [
        selected_by_token[token]
        for token in sample_lists["ssl_train"]
        if token in selected_by_token
        and selected_by_token[token]["record_state"] in localized
    ]
    train_counts = Counter(row["registered_dog_id"] for row in train_candidates)
    excluded_train = sorted(
        identity
        for identity in identity_lists["ssl_train"]
        if train_counts[identity] < 2
    )
    excluded_set = set(excluded_train)
    train_rows = [
        row for row in train_candidates if row["registered_dog_id"] not in excluded_set
    ]
    train_track_counts = Counter(row["track_token"] for row in train_rows)
    excluded_train_tracks = sorted(
        track for track, count in train_track_counts.items() if count < 2
    )
    train_rows = [
        row
        for row in train_rows
        if row["track_token"] not in set(excluded_train_tracks)
    ]
    dev_rows = [
        selected_by_token[token]
        for token in sample_lists["dev"]
        if token in selected_by_token
        and selected_by_token[token]["record_state"] in localized
    ]
    if not train_rows:
        raise ValueError(
            "N4 cache has no TRAIN identities with at least two localized frames"
        )
    if not dev_rows:
        raise ValueError("N4 cache has no lineage DEV frames")
    if {row["registered_dog_id"] for row in train_rows} & {
        row["registered_dog_id"] for row in dev_rows
    }:
        raise ValueError("N4 cache TRAIN and DEV identities overlap")
    ordered = [("TRAIN", row) for row in train_rows] + [
        ("DEV", row) for row in dev_rows
    ]
    ordered.sort(key=lambda item: (item[0], item[1]["sample_token"]))

    runtime = ExactOnnxRuntime(n3_onnx_path, runtime_manifest, use_cuda=use_cuda)
    vectors: list[np.ndarray] = []
    cache_rows: list[dict[str, Any]] = []
    for row_index, (role, row) in enumerate(ordered):
        image = _read_bound_crop(native_root, row)
        vector = runtime.run(preprocess_image(image, runtime_manifest))[0]
        normalized = _normalized_vector(vector, f"N3 sample {row['sample_token']}")
        vectors.append(normalized)
        cache_rows.append(_cache_row(row_index, role, row))
    matrix = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
    code_hashes = _code_hashes(repository, _CACHE_CODE_PATHS)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    os.chmod(staging, 0o700)
    try:
        matrix_path = staging / _CACHE_FILES[0]
        with matrix_path.open("xb") as stream:
            np.save(stream, matrix, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(matrix_path, 0o600)
        matrix_binding = {
            "path": matrix_path.name,
            "sha256": file_sha256(matrix_path),
            "shape": list(matrix.shape),
            "dtype": "float32",
            "format": "NUMPY_NPY_ALLOW_PICKLE_FALSE",
        }
        input_bindings = {
            "native_bundle": _document_binding(native_bundle_path, native_document),
            "native_manifest_sha256": native_document.payload["manifest_sha256"],
            "native_input_sha256s": manifest["input_sha256s"],
            "n3_lineage": {
                **_document_binding(n3_lineage_path, lineage_document),
                "lineage_sha256": lineage_document.payload["lineage_sha256"],
            },
            "n3_runtime_manifest": _document_binding(
                n3_runtime_manifest_path, runtime_document
            ),
            "n3_onnx": {
                "path": os.fspath(n3_onnx_path),
                "sha256": onnx_pin,
                "bytes": n3_onnx_path.stat().st_size,
            },
            "lineage_identity_lists": identity_lists,
            "lineage_sample_token_list_sha256s": {
                name: content_sha256({"sample_tokens": values})
                for name, values in sorted(sample_lists.items())
            },
        }
        cache = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": "FROZEN_N3_EMBEDDINGS_FOR_N4_TRAIN_AND_DEV_ONLY",
            "interpretation": (
                "TRACK_PROXY_METRIC_LEARNING_NOT_LIFELONG_IDENTITY_OR_PHYSICAL_"
                "NOSE_TOPOLOGY"
            ),
            "protocol": {
                "source_roles": ["ssl_train", "dev"],
                "emitted_roles": ["TRAIN", "DEV"],
                "lineage_eval_included": False,
                "publisher_test_included": False,
                "train_record_states": ["AVAILABLE", "LOW_QUALITY"],
                "dev_record_states": ["AVAILABLE", "LOW_QUALITY"],
                "train_label": "track_token proxy, not lifelong identity",
                "quality_weight": (
                    "0.5 + 0.5 * mean(blur_score, contrast_score, "
                    "detector_confidence, frontality, 1-mask_uncertainty)"
                ),
            },
            "input_bindings": input_bindings,
            "input_bindings_sha256": content_sha256(input_bindings),
            "matrix": matrix_binding,
            "rows": cache_rows,
            "exclusions": {
                "train_identity_reason": "FEWER_THAN_TWO_LOCALIZED_FRAMES",
                "train_registered_identity_ids": excluded_train,
                "train_track_tokens_fewer_than_two_frames": excluded_train_tracks,
                "included_low_quality_train_sample_tokens": sorted(
                    token
                    for token in sample_lists["ssl_train"]
                    if token in selected_by_token
                    and selected_by_token[token]["record_state"] == "LOW_QUALITY"
                ),
                "no_roi_sample_tokens_not_cached": list(
                    sample_lists["excluded_no_roi"]
                ),
            },
            "code_sha256s": code_hashes,
        }
        bundle = {
            "schema_version": CACHE_BUNDLE_SCHEMA_VERSION,
            "cache_sha256": content_sha256(cache),
            "cache": cache,
        }
        manifest_path = staging / _CACHE_FILES[1]
        _write_exclusive(manifest_path, json_document_bytes(bundle))
        validate_cache_manifest(bundle, root=staging)
        fsync_directory(staging)
        rename_directory_noreplace(staging, output)
        fsync_directory(output.parent)
        return bundle
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_cache_manifest(
    bundle: object, *, root: Path | None = None
) -> dict[str, Any]:
    """Validate cache role separation, row bindings, and optional matrix bytes."""

    if not isinstance(bundle, dict) or set(bundle) != {
        "schema_version",
        "cache_sha256",
        "cache",
    }:
        raise ValueError("N4 cache bundle fields differ")
    if bundle["schema_version"] != CACHE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("N4 cache bundle schema differs")
    _sha256(bundle["cache_sha256"], "N4 cache digest")
    cache = bundle["cache"]
    expected = {
        "schema_version",
        "status",
        "interpretation",
        "protocol",
        "input_bindings",
        "input_bindings_sha256",
        "matrix",
        "rows",
        "exclusions",
        "code_sha256s",
    }
    if not isinstance(cache, dict) or set(cache) != expected:
        raise ValueError("N4 cache fields differ")
    if (
        cache["schema_version"] != CACHE_SCHEMA_VERSION
        or content_sha256(cache) != bundle["cache_sha256"]
    ):
        raise ValueError("N4 cache schema or digest differs")
    if (
        cache["status"] != "FROZEN_N3_EMBEDDINGS_FOR_N4_TRAIN_AND_DEV_ONLY"
        or cache["interpretation"]
        != "TRACK_PROXY_METRIC_LEARNING_NOT_LIFELONG_IDENTITY_OR_PHYSICAL_NOSE_TOPOLOGY"
        or cache["protocol"]
        != {
            "source_roles": ["ssl_train", "dev"],
            "emitted_roles": ["TRAIN", "DEV"],
            "lineage_eval_included": False,
            "publisher_test_included": False,
            "train_record_states": ["AVAILABLE", "LOW_QUALITY"],
            "dev_record_states": ["AVAILABLE", "LOW_QUALITY"],
            "train_label": "track_token proxy, not lifelong identity",
            "quality_weight": (
                "0.5 + 0.5 * mean(blur_score, contrast_score, "
                "detector_confidence, frontality, 1-mask_uncertainty)"
            ),
        }
    ):
        raise ValueError("N4 cache role and exposure protocol differs")
    if content_sha256(cache["input_bindings"]) != cache["input_bindings_sha256"]:
        raise ValueError("N4 cache input binding digest differs")
    rows = cache["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("N4 cache rows differ")
    expected_row_fields = {
        "row_index",
        "sample_token",
        "registered_identity_id",
        "identity_token",
        "track_token",
        "sequence_token",
        "frame_index",
        "record_state",
        "role",
        "quality",
        "quality_weight",
        "source_hashes",
    }
    samples: set[str] = set()
    train_ids: set[str] = set()
    dev_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_row_fields:
            raise ValueError("N4 cache row fields differ")
        if row["row_index"] != index:
            raise ValueError("N4 cache row indices are not contiguous")
        sample = _text(row["sample_token"], "cache sample_token")
        if sample in samples:
            raise ValueError("N4 cache repeats a sample token")
        samples.add(sample)
        role = row["role"]
        if role not in {"TRAIN", "DEV"}:
            raise ValueError("N4 cache role differs")
        identity = _text(row["registered_identity_id"], "cache identity")
        (train_ids if role == "TRAIN" else dev_ids).add(identity)
        _text(row["track_token"], "cache track")
        _text(row["sequence_token"], "cache sequence")
        _nonnegative_int(row["frame_index"], "cache frame index")
        if row["record_state"] not in {"AVAILABLE", "LOW_QUALITY"}:
            raise ValueError("N4 cache TRAIN/DEV state differs")
        _finite_number(row["quality_weight"], "cache quality weight", 0.5, 1.0)
        if not isinstance(row["quality"], dict) or not row["quality"]:
            raise ValueError("N4 cache quality fields differ")
        hashes = row["source_hashes"]
        if not isinstance(hashes, dict) or set(hashes) != {
            "source_sha256",
            "crop_sha256",
            "soft_mask_sha256",
            "binary_mask_sha256",
        }:
            raise ValueError("N4 cache source hashes differ")
        for digest in hashes.values():
            _sha256(digest, "cache row source hash")
    if train_ids & dev_ids:
        raise ValueError("N4 cache TRAIN and DEV identities overlap")
    if any(
        count < 2
        for count in Counter(
            row["track_token"] for row in rows if row["role"] == "TRAIN"
        ).values()
    ):
        raise ValueError("N4 cache TRAIN contains a singleton track")
    lineage_lists = cache["input_bindings"]["lineage_identity_lists"]
    if train_ids - set(lineage_lists["ssl_train"]) or dev_ids - set(
        lineage_lists["dev"]
    ):
        raise ValueError("N4 cache roles differ from lineage identities")
    if (train_ids | dev_ids) & set(lineage_lists["eval"]):
        raise ValueError("N4 cache includes lineage EVAL identities")
    matrix = cache["matrix"]
    if not isinstance(matrix, dict) or set(matrix) != {
        "path",
        "sha256",
        "shape",
        "dtype",
        "format",
    }:
        raise ValueError("N4 cache matrix binding differs")
    if matrix["path"] != "embeddings.npy" or matrix["shape"][0] != len(rows):
        raise ValueError("N4 cache matrix row binding differs")
    if (
        matrix["dtype"] != "float32"
        or matrix["format"] != "NUMPY_NPY_ALLOW_PICKLE_FALSE"
    ):
        raise ValueError("N4 cache matrix type or format differs")
    _sha256(matrix["sha256"], "N4 matrix digest")
    for digest in cache["code_sha256s"].values():
        _sha256(digest, "N4 cache code hash")
    if root is not None:
        _load_bound_cache_matrix(cache, root)
    return cache


def load_embedding_cache(
    manifest_path: Path, expected_content_sha256: str
) -> tuple[dict[str, Any], np.ndarray]:
    document = _pinned_json(
        manifest_path, expected_content_sha256, "N4 cache manifest", large=True
    )
    cache = validate_cache_manifest(document.payload)
    matrix = _load_bound_cache_matrix(cache, manifest_path.parent)
    return document.payload, matrix


def _load_bound_cache_matrix(cache: Mapping[str, Any], root: Path) -> np.ndarray:
    binding = cache["matrix"]
    resolved = root.resolve(strict=True)
    path = resolved / binding["path"]
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve(strict=True).is_relative_to(resolved)
    ):
        raise ValueError("N4 cache matrix path is unsafe")
    before = path.stat(follow_symlinks=False)
    payload = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        raise RuntimeError("N4 cache matrix changed while reading")
    if hashlib.sha256(payload).hexdigest() != binding["sha256"]:
        raise ValueError("N4 cache matrix file hash differs")
    values = np.load(io.BytesIO(payload), allow_pickle=False)
    if values.dtype != np.float32 or list(values.shape) != binding["shape"]:
        raise ValueError("N4 cache matrix dtype or shape differs")
    return _normalized_float32_matrix(values, len(cache["rows"]), "N4 cache matrix")


def train_metric_adapter(
    *,
    cache_manifest_path: Path,
    cache_manifest_sha256: str,
    output_checkpoint_path: Path,
    epochs: int = 20,
    bottleneck_dim: int = 64,
    scale: float = 0.1,
    tracks_per_batch: int = 8,
    samples_per_track: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    margin: float = 0.2,
    anchor_weight: float = 0.25,
    patience: int = 5,
    seed: int = 42,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Train only the residual adapter and publish one strict weights-only checkpoint."""

    repository = _repository(repository_root)
    output = _new_external_path(
        output_checkpoint_path, repository, "N4 adapter checkpoint"
    )
    for name, value in (("epochs", epochs), ("patience", patience)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("learning_rate", learning_rate),
        ("weight_decay", weight_decay),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            or (name == "learning_rate" and value == 0.0)
        ):
            raise ValueError(f"{name} differs")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    cache_bundle, matrix = load_embedding_cache(
        cache_manifest_path, cache_manifest_sha256
    )
    cache = cache_bundle["cache"]
    rows = cache["rows"]
    train_indices = [index for index, row in enumerate(rows) if row["role"] == "TRAIN"]
    dev_indices = [index for index, row in enumerate(rows) if row["role"] == "DEV"]
    train_rows = [rows[index] for index in train_indices]
    dev_rows = [rows[index] for index in dev_indices]
    if not train_rows or not dev_rows:
        raise ValueError("N4 training requires non-empty TRAIN and DEV cache roles")
    sampler = DeterministicPKBatchSampler(
        [row["track_token"] for row in train_rows],
        tracks_per_batch=tracks_per_batch,
        samples_per_track=samples_per_track,
        seed=seed,
    )
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ResidualMetricAdapter(matrix.shape[1], bottleneck_dim, scale=scale)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_tensor = torch.from_numpy(matrix[train_indices].copy())
    dev_tensor = torch.from_numpy(matrix[dev_indices].copy())
    quality = torch.tensor(
        [row["quality_weight"] for row in train_rows], dtype=torch.float32
    )
    history: list[dict[str, Any]] = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    no_improvement = 0
    best_epoch = 0
    for epoch in range(epochs + 1):
        train_report = None
        if epoch > 0:
            model.train()
            totals: list[float] = []
            metric_values: list[float] = []
            anchor_values: list[float] = []
            for local_indices in sampler.batches(epoch - 1):
                index_tensor = torch.tensor(local_indices, dtype=torch.long)
                parent = train_tensor[index_tensor]
                labels = [train_rows[index]["track_token"] for index in local_indices]
                adapted = model(parent)
                loss, components = batch_hard_metric_loss(
                    adapted,
                    parent,
                    labels,
                    quality[index_tensor],
                    margin=margin,
                    anchor_weight=anchor_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                totals.append(float(loss.detach()))
                metric_values.append(float(components["batch_hard_metric"].detach()))
                anchor_values.append(float(components["parent_anchor"].detach()))
            train_report = {
                "batch_count": len(totals),
                "loss_mean": float(np.mean(totals)),
                "batch_hard_metric_mean": float(np.mean(metric_values)),
                "parent_anchor_mean": float(np.mean(anchor_values)),
                "hard_negatives": "TRAIN_BATCH_ONLY",
            }
        model.eval()
        with torch.inference_mode():
            adapted_dev = model(dev_tensor).cpu().numpy().astype(np.float32)
        evaluation = evaluate_k5(adapted_dev, dev_rows)
        change = float(
            np.mean(
                1.0
                - np.sum(
                    adapted_dev.astype(np.float64)
                    * matrix[dev_indices].astype(np.float64),
                    axis=1,
                )
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_report,
                "dev": evaluation,
                "adapter_change": max(0.0, change),
            }
        )
        states[epoch] = _cpu_state_dict(model.state_dict())
        selection = select_dev_epoch(history)
        if selection["selected_epoch"] != best_epoch:
            best_epoch = selection["selected_epoch"]
            no_improvement = 0
        elif epoch > 0:
            no_improvement += 1
        if epoch > 0 and no_improvement >= patience:
            break
    selection = select_dev_epoch(history)
    selected_epoch = selection["selected_epoch"]
    selected_state = states[selected_epoch]
    config = {
        "architecture": "Linear-ReLU-Linear residual adapter",
        "input_dimension": matrix.shape[1],
        "bottleneck_dimension": bottleneck_dim,
        "scale": float(scale),
        "scale_maximum": MAXIMUM_SCALE,
        "epochs_requested": epochs,
        "epochs_completed": history[-1]["epoch"],
        "early_stop_patience": patience,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
        },
        "sampler": {
            "name": "DETERMINISTIC_P_K_TRACK_PROXY",
            "tracks_per_batch": tracks_per_batch,
            "samples_per_track": samples_per_track,
            "seed": seed,
        },
        "loss": {
            "metric": "BATCH_HARD_COSINE_TRIPLET_TRAIN_ONLY_HARD_NEGATIVES",
            "margin": float(margin),
            "parent_anchor": "BOUNDED_ONE_MINUS_COSINE_IN_[0,2]",
            "parent_anchor_weight": float(anchor_weight),
            "sample_quality_weight_range": [0.5, 1.0],
        },
        "labels": "track_token proxy labels are not lifelong identities",
        "selection": "LINEAGE_DEV_EXACT_EARLIEST_LATEST_K5_ONLY",
        "lineage_eval_used": False,
        "publisher_test_used": False,
        "device": "cpu",
    }
    bindings = {
        "cache_manifest": {
            "path": os.fspath(cache_manifest_path),
            "content_sha256": _sha256(
                cache_manifest_sha256, "cache manifest external pin"
            ),
            "cache_sha256": cache_bundle["cache_sha256"],
            "matrix_sha256": cache["matrix"]["sha256"],
        },
        "n3": {
            "lineage_payload_sha256": cache["input_bindings"]["n3_lineage"][
                "content_sha256"
            ],
            "lineage_sha256": cache["input_bindings"]["n3_lineage"]["lineage_sha256"],
            "runtime_manifest_payload_sha256": cache["input_bindings"][
                "n3_runtime_manifest"
            ]["content_sha256"],
            "onnx_sha256": cache["input_bindings"]["n3_onnx"]["sha256"],
            "identity_lists": cache["input_bindings"]["lineage_identity_lists"],
        },
        "train_registered_identity_ids": sorted(
            {row["registered_identity_id"] for row in train_rows}
        ),
        "dev_registered_identity_ids": sorted(
            {row["registered_identity_id"] for row in dev_rows}
        ),
        "train_track_tokens_sha256": content_sha256(
            {"track_tokens": sorted({row["track_token"] for row in train_rows})}
        ),
        "dev_track_tokens_sha256": content_sha256(
            {"track_tokens": sorted({row["track_token"] for row in dev_rows})}
        ),
    }
    checkpoint = build_adapter_checkpoint(
        state_dict=selected_state,
        config=config,
        bindings=bindings,
        selected_epoch=selected_epoch,
        history=history,
        selection=selection,
        worktree_provenance=_worktree_provenance(repository),
        code_sha256s=_code_hashes(repository, _TRAIN_CODE_PATHS),
    )
    _publish_torch_checkpoint(output, checkpoint)
    return checkpoint


def build_adapter_checkpoint(
    *,
    state_dict: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    bindings: Mapping[str, Any],
    selected_epoch: int,
    history: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    worktree_provenance: Mapping[str, Any],
    code_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    state = _cpu_state_dict(state_dict)
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "format": "PYTORCH_WEIGHTS_ONLY_STRICT_SCHEMA",
        "selected_epoch": selected_epoch,
        "config": dict(config),
        "config_sha256": content_sha256(dict(config)),
        "bindings": dict(bindings),
        "bindings_sha256": content_sha256(dict(bindings)),
        "selection": dict(selection),
        "history": [dict(item) for item in history],
        "state_dict": state,
        "state_sha256": _state_dict_sha256(state),
        "worktree_provenance": dict(worktree_provenance),
        "code_sha256s": dict(sorted(code_sha256s.items())),
    }
    payload["checkpoint_payload_sha256"] = _checkpoint_metadata_sha256(payload)
    validate_adapter_checkpoint(payload)
    return payload


def validate_adapter_checkpoint(payload: object) -> dict[str, Any]:
    expected = {
        "schema_version",
        "format",
        "selected_epoch",
        "config",
        "config_sha256",
        "bindings",
        "bindings_sha256",
        "selection",
        "history",
        "state_dict",
        "state_sha256",
        "worktree_provenance",
        "code_sha256s",
        "checkpoint_payload_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("N4 adapter checkpoint fields differ")
    if (
        payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or payload["format"] != "PYTORCH_WEIGHTS_ONLY_STRICT_SCHEMA"
    ):
        raise ValueError("N4 adapter checkpoint schema or format differs")
    selected = _nonnegative_int(payload["selected_epoch"], "selected epoch")
    if content_sha256(payload["config"]) != payload["config_sha256"]:
        raise ValueError("N4 adapter config digest differs")
    if content_sha256(payload["bindings"]) != payload["bindings_sha256"]:
        raise ValueError("N4 adapter bindings digest differs")
    config = payload["config"]
    expected_config_fields = {
        "architecture",
        "input_dimension",
        "bottleneck_dimension",
        "scale",
        "scale_maximum",
        "epochs_requested",
        "epochs_completed",
        "early_stop_patience",
        "optimizer",
        "sampler",
        "loss",
        "labels",
        "selection",
        "lineage_eval_used",
        "publisher_test_used",
        "device",
    }
    if (
        not isinstance(config, dict)
        or set(config) != expected_config_fields
        or config.get("architecture") != "Linear-ReLU-Linear residual adapter"
        or config.get("labels")
        != "track_token proxy labels are not lifelong identities"
        or config.get("selection") != "LINEAGE_DEV_EXACT_EARLIEST_LATEST_K5_ONLY"
        or config.get("lineage_eval_used") is not False
        or config.get("publisher_test_used") is not False
        or config.get("device") != "cpu"
    ):
        raise ValueError("N4 adapter configuration contract differs")
    input_dim = _positive_int(config.get("input_dimension"), "adapter input dimension")
    bottleneck = _positive_int(
        config.get("bottleneck_dimension"), "adapter bottleneck dimension"
    )
    if bottleneck > min(input_dim, MAXIMUM_BOTTLENECK):
        raise ValueError("N4 adapter bottleneck exceeds its bound")
    _finite_number(
        config.get("scale"), "adapter scale", 0.0, MAXIMUM_SCALE, lower_open=True
    )
    if config.get("scale_maximum") != MAXIMUM_SCALE:
        raise ValueError("N4 adapter scale bound differs")
    epochs_requested = _positive_int(config["epochs_requested"], "adapter epochs requested")
    epochs_completed = _nonnegative_int(
        config["epochs_completed"], "adapter epochs completed"
    )
    _positive_int(config["early_stop_patience"], "adapter early-stop patience")
    if not selected <= epochs_completed <= epochs_requested:
        raise ValueError("N4 adapter epoch bounds differ")
    optimizer = config["optimizer"]
    if not isinstance(optimizer, dict) or set(optimizer) != {
        "name",
        "learning_rate",
        "weight_decay",
    } or optimizer["name"] != "AdamW":
        raise ValueError("N4 adapter optimizer contract differs")
    _finite_number(
        optimizer["learning_rate"],
        "adapter learning rate",
        0.0,
        math.inf,
        lower_open=True,
    )
    _finite_number(optimizer["weight_decay"], "adapter weight decay", 0.0, math.inf)
    sampler = config["sampler"]
    if not isinstance(sampler, dict) or set(sampler) != {
        "name",
        "tracks_per_batch",
        "samples_per_track",
        "seed",
    } or sampler["name"] != "DETERMINISTIC_P_K_TRACK_PROXY":
        raise ValueError("N4 adapter sampler contract differs")
    if (
        _positive_int(sampler["tracks_per_batch"], "adapter tracks per batch") < 2
        or _positive_int(sampler["samples_per_track"], "adapter samples per track") < 2
    ):
        raise ValueError("N4 adapter P/K sampler bounds differ")
    _nonnegative_int(sampler["seed"], "adapter sampler seed")
    loss = config["loss"]
    if (
        not isinstance(loss, dict)
        or set(loss)
        != {
            "metric",
            "margin",
            "parent_anchor",
            "parent_anchor_weight",
            "sample_quality_weight_range",
        }
        or loss["metric"]
        != "BATCH_HARD_COSINE_TRIPLET_TRAIN_ONLY_HARD_NEGATIVES"
        or loss["parent_anchor"] != "BOUNDED_ONE_MINUS_COSINE_IN_[0,2]"
        or loss["sample_quality_weight_range"] != [0.5, 1.0]
    ):
        raise ValueError("N4 adapter loss contract differs")
    _finite_number(loss["margin"], "adapter loss margin", 0.0, 2.0, lower_open=True)
    if float(loss["margin"]) >= 2.0:
        raise ValueError("adapter loss margin must be less than 2")
    _finite_number(
        loss["parent_anchor_weight"], "adapter parent anchor weight", 0.0, 1.0
    )
    history = payload["history"]
    if (
        not isinstance(history, list)
        or not history
        or history[-1].get("epoch") != epochs_completed
    ):
        raise ValueError("N4 adapter completed epoch differs from history")
    selection = select_dev_epoch(history)
    if selection != payload["selection"] or selection["selected_epoch"] != selected:
        raise ValueError("N4 adapter selection differs from history")
    state = payload["state_dict"]
    expected_shapes = {
        "down.weight": (bottleneck, input_dim),
        "down.bias": (bottleneck,),
        "up.weight": (input_dim, bottleneck),
        "up.bias": (input_dim,),
    }
    if not isinstance(state, dict) or set(state) != set(expected_shapes):
        raise ValueError("N4 adapter state fields differ")
    for name, shape in expected_shapes.items():
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"N4 adapter tensor {name} differs")
    if _state_dict_sha256(state) != payload["state_sha256"]:
        raise ValueError("N4 adapter state digest differs")
    if _checkpoint_metadata_sha256(payload) != payload["checkpoint_payload_sha256"]:
        raise ValueError("N4 adapter checkpoint payload digest differs")
    bindings = payload["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {
        "cache_manifest",
        "n3",
        "train_registered_identity_ids",
        "dev_registered_identity_ids",
        "train_track_tokens_sha256",
        "dev_track_tokens_sha256",
    }:
        raise ValueError("N4 adapter bindings differ")
    if set(bindings["n3"]) != {
        "lineage_payload_sha256",
        "lineage_sha256",
        "runtime_manifest_payload_sha256",
        "onnx_sha256",
        "identity_lists",
    }:
        raise ValueError("N4 adapter N3 bindings differ")
    if set(bindings["n3"]["identity_lists"]) != {
        "parent_seen_yt",
        "parent_seen_native_ssl_train",
        "ssl_train",
        "dev",
        "eval",
    }:
        raise ValueError("N4 adapter lineage identity lists differ")
    cache_binding = bindings["cache_manifest"]
    if not isinstance(cache_binding, dict) or set(cache_binding) != {
        "path",
        "content_sha256",
        "cache_sha256",
        "matrix_sha256",
    }:
        raise ValueError("N4 adapter cache binding differs")
    _text(cache_binding["path"], "N4 adapter cache path")
    for name in ("content_sha256", "cache_sha256", "matrix_sha256"):
        _sha256(cache_binding[name], f"N4 adapter cache {name}")
    n3_binding = bindings["n3"]
    for name in (
        "lineage_payload_sha256",
        "lineage_sha256",
        "runtime_manifest_payload_sha256",
        "onnx_sha256",
    ):
        _sha256(n3_binding[name], f"N4 adapter N3 {name}")
    for name, values in n3_binding["identity_lists"].items():
        if not isinstance(values, list) or values != sorted(set(values)):
            raise ValueError(f"N4 adapter N3 identity list {name} differs")
        for value in values:
            _text(value, f"N4 adapter N3 identity list {name}")
    for name in ("train_registered_identity_ids", "dev_registered_identity_ids"):
        values = bindings[name]
        if not isinstance(values, list) or values != sorted(set(values)) or not values:
            raise ValueError(f"N4 adapter {name} differs")
        for value in values:
            _text(value, f"N4 adapter {name}")
    for name in ("train_track_tokens_sha256", "dev_track_tokens_sha256"):
        _sha256(bindings[name], f"N4 adapter {name}")
    train = set(bindings["train_registered_identity_ids"])
    dev = set(bindings["dev_registered_identity_ids"])
    lists = bindings["n3"]["identity_lists"]
    if train & dev or train - set(lists["ssl_train"]) or dev - set(lists["dev"]):
        raise ValueError("N4 adapter identity role binding differs")
    if (train | dev) & set(lists["eval"]):
        raise ValueError("N4 adapter binds lineage EVAL identities")
    for digest in payload["code_sha256s"].values():
        _sha256(digest, "adapter code hash")
    return payload


def load_adapter_checkpoint(
    path: Path, *, expected_file_sha256: str | None = None
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("N4 adapter checkpoint must be a regular non-symlink file")
    before = source.stat(follow_symlinks=False)
    payload = source.read_bytes()
    after = source.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        raise RuntimeError("N4 adapter checkpoint changed while reading")
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_file_sha256 is not None and observed_sha256 != _sha256(
        expected_file_sha256, "adapter checkpoint external pin"
    ):
        raise ValueError("N4 adapter checkpoint file SHA-256 differs from external pin")
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    return validate_adapter_checkpoint(checkpoint)


def adapter_from_checkpoint(checkpoint: Mapping[str, Any]) -> ResidualMetricAdapter:
    validated = validate_adapter_checkpoint(dict(checkpoint))
    config = validated["config"]
    model = ResidualMetricAdapter(
        config["input_dimension"], config["bottleneck_dimension"], scale=config["scale"]
    )
    model.load_state_dict(validated["state_dict"], strict=True)
    return model.eval()


def apply_adapter(checkpoint: Mapping[str, Any], embeddings: np.ndarray) -> np.ndarray:
    model = adapter_from_checkpoint(checkpoint)
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != model.down.in_features:
        raise ValueError("adapter evaluation matrix shape differs")
    if not np.isfinite(values).all():
        raise ValueError("adapter evaluation matrix contains non-finite values")
    with torch.inference_mode():
        output = model(torch.from_numpy(np.ascontiguousarray(values))).cpu().numpy()
    return _normalized_float32_matrix(
        output.astype(np.float32), len(values), "adapter output"
    )


def evaluate_metric_adapter(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    panel_path: Path,
    panel_sha256: str,
    topology_manifest_path: Path,
    topology_sha256: str,
    output_path: Path,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_confidence_level: float = 0.95,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Apply a DEV-selected adapter once to fixed-panel EVAL N3 frame vectors."""

    repository = _repository(repository_root)
    output = _new_external_path(output_path, repository, "N4 adapter evaluation")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be non-negative")
    if not 0.0 < bootstrap_confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must be in (0,1)")
    checkpoint = load_adapter_checkpoint(
        checkpoint_path, expected_file_sha256=checkpoint_sha256
    )
    panel_document, panel = read_fixed_panel(panel_path, panel_sha256)
    topology_document = _pinned_json(
        topology_manifest_path, topology_sha256, "topology manifest", large=True
    )
    validate_identity_topology_manifest(topology_document.payload)
    validate_fixed_topology_bindings(
        panel_document.payload,
        topology_document.payload,
        n3_runtime_manifest_content_sha256=checkpoint["bindings"]["n3"][
            "runtime_manifest_payload_sha256"
        ],
        n3_onnx_sha256=checkpoint["bindings"]["n3"]["onnx_sha256"],
    )
    panel_lineage = panel["input_bindings"].get("n3_lineage")
    if (
        not isinstance(panel_lineage, Mapping)
        or panel_lineage.get("content_sha256")
        != checkpoint["bindings"]["n3"]["lineage_payload_sha256"]
    ):
        raise ValueError("fixed panel N3 lineage differs from the adapter lineage")
    panel_ids = set(panel["population"]["eligible_identity_ids"])
    exposed = set().union(
        *(
            set(values)
            for values in checkpoint["bindings"]["n3"]["identity_lists"].values()
        )
    )
    if panel_ids & exposed:
        raise ValueError("fixed panel identity overlaps adapter/N3 exposure")
    vectors, eval_rows, topology_rows = _panel_eval_vectors(
        panel_document.payload, topology_document.payload
    )
    adapted = apply_adapter(checkpoint, vectors)
    baseline = evaluate_k5(vectors, eval_rows)
    candidate = evaluate_k5(adapted, eval_rows)
    paired_cis = _paired_bootstrap(
        baseline["outcomes"],
        candidate["outcomes"],
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        confidence_level=bootstrap_confidence_level,
    )
    before_manifest = {
        "schema_version": IDENTITY_TOPOLOGY_MANIFEST_SCHEMA_VERSION,
        "records": topology_rows,
    }
    after_manifest = {
        "schema_version": IDENTITY_TOPOLOGY_MANIFEST_SCHEMA_VERSION,
        "records": [
            {**row, "branch": "N4_metric_adapter", "embedding": adapted[index].tolist()}
            for index, row in enumerate(topology_rows)
        ],
    }
    before_topology = audit_identity_topology(before_manifest)["branches"][N3_BRANCH]
    after_topology = audit_identity_topology(after_manifest)["branches"][
        "N4_metric_adapter"
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS_BOUNDED_N4_EMBEDDING_SPACE_PUBLISHER_TEST_DIAGNOSTIC",
        "interpretation": (
            "DEV_SELECTED_RESIDUAL_ADAPTER_APPLIED_ONCE_TO_EXPOSED_SAME_TRACK_"
            "PUBLISHER_EVAL_NOT_PHYSICAL_TOPOLOGY_OR_BIOMETRIC_VALIDATION"
        ),
        "protocol": {
            "input_branch": N3_BRANCH,
            "adapter": "normalize(x + fixed_scale * Linear-ReLU-Linear(x))",
            "adapter_scale": checkpoint["config"]["scale"],
            "adapter_scale_maximum": MAXIMUM_SCALE,
            "checkpoint_selection": checkpoint["config"]["selection"],
            "publisher_dev_used_for_selection": False,
            "publisher_eval_used_for_selection": False,
            "lineage_eval_used": checkpoint["config"]["lineage_eval_used"],
            "publisher_partition_evaluated": "EVAL_ONLY",
            "gallery_query": "EXACT_FIXED_PANEL_EARLIEST_LATEST_K5_NORMALIZED_MEAN",
            "same_track_only": True,
            "publisher_test_exposed": True,
            "weak_nose_roi": True,
            "closed_set": True,
            "open_set": False,
            "embedding_space_topology_only": True,
            "physical_nose_topology_claim": False,
            "bootstrap": {
                "cluster_unit": "registered_identity_id",
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
                "confidence_level": bootstrap_confidence_level,
            },
            "limitations": list(LIMITATIONS),
        },
        "input_bindings": {
            "checkpoint_file_sha256": _sha256(
                checkpoint_sha256, "adapter checkpoint external pin"
            ),
            "checkpoint_payload_sha256": checkpoint["checkpoint_payload_sha256"],
            "cache_manifest_content_sha256": checkpoint["bindings"]["cache_manifest"][
                "content_sha256"
            ],
            "n3_lineage_payload_sha256": checkpoint["bindings"]["n3"][
                "lineage_payload_sha256"
            ],
            "panel": _document_binding(panel_path, panel_document),
            "topology_manifest": _document_binding(
                topology_manifest_path, topology_document
            ),
            "n3_eval_frame_matrix_float32_sha256": hashlib.sha256(
                np.ascontiguousarray(vectors, dtype=np.float32).tobytes()
            ).hexdigest(),
        },
        "population": {
            "eval_identity_ids": panel["population"]["eval_identity_ids"],
            "eval_identity_ids_sha256": panel["population"]["eval_identity_ids_sha256"],
            "train_dev_eval_identity_overlap": False,
        },
        "evaluation": {
            "baseline_N3": baseline,
            "candidate_N4": candidate,
            "paired_N4_minus_N3_identity_bootstrap_cis": paired_cis,
            "rescue_break": _rescue_break(baseline["outcomes"], candidate["outcomes"]),
            "embedding_topology_dispersion": {
                "before_N3": before_topology,
                "after_N4": after_topology,
            },
        },
        "code_sha256s": _code_hashes(repository, _EVALUATION_CODE_PATHS),
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    bundle = {
        "schema_version": REPORT_BUNDLE_SCHEMA_VERSION,
        "report_sha256": content_sha256(report),
        "report": report,
    }
    validate_evaluation_bundle(bundle)
    write_private_json_bundle(((output, bundle),))
    return bundle


def validate_evaluation_bundle(bundle: object) -> dict[str, Any]:
    if not isinstance(bundle, dict) or set(bundle) != {
        "schema_version",
        "report_sha256",
        "report",
    }:
        raise ValueError("N4 adapter report bundle fields differ")
    if bundle["schema_version"] != REPORT_BUNDLE_SCHEMA_VERSION:
        raise ValueError("N4 adapter report bundle schema differs")
    report = bundle["report"]
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or content_sha256(report) != bundle["report_sha256"]
    ):
        raise ValueError("N4 adapter report schema or digest differs")
    protocol = report.get("protocol", {})
    if (
        protocol.get("publisher_dev_used_for_selection") is not False
        or protocol.get("publisher_eval_used_for_selection") is not False
        or protocol.get("physical_nose_topology_claim") is not False
        or protocol.get("open_set") is not False
    ):
        raise ValueError("N4 adapter report limitations differ")
    return report


def _panel_eval_vectors(
    panel_bundle: Mapping[str, Any], topology_manifest: Mapping[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    panel = validate_panel_bundle(panel_bundle)
    topology_rows = topology_manifest["records"]
    n3 = {
        row["sample_token"]: row for row in topology_rows if row["branch"] == N3_BRANCH
    }
    panel_samples = {row["sample_id"] for row in panel["records"]}
    if set(n3) != panel_samples:
        raise ValueError("N3 topology sample coverage differs from the fixed panel")
    eval_ids = set(panel["population"]["eval_identity_ids"])
    selected = [
        row for row in panel["records"] if row["registered_identity_id"] in eval_ids
    ]
    selected.sort(
        key=lambda row: (
            row["registered_identity_id"],
            row["publisher_frame_index"],
            row["sample_id"],
        )
    )
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for record in selected:
        topology = n3[record["sample_id"]]
        if (
            topology["identity_token"] != record["registered_identity_id"]
            or topology["session_token"] != record["capture_group_id"]
            or topology["quality"] != record["source"]["quality"]["overall"]
            or topology["available"] is not True
        ):
            raise ValueError("N3 topology semantic binding differs from fixed panel")
        vector = _normalized_vector(topology["embedding"], "fixed-panel N3 vector")
        vectors.append(vector)
        rows.append(
            {
                "sample_token": record["sample_id"],
                "registered_identity_id": record["registered_identity_id"],
                "track_token": record["capture_group_id"],
                "frame_index": record["publisher_frame_index"],
            }
        )
        audit_rows.append(dict(topology))
    return np.stack(vectors), rows, audit_rows


def _paired_bootstrap(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    _paired_outcomes(baseline, candidate)
    delta = [
        {
            "bootstrap_cluster_id": left["registered_identity_id"],
            **{metric: right[metric] - left[metric] for metric in METRICS},
        }
        for left, right in zip(baseline, candidate, strict=True)
    ]
    return {
        metric: identity_clustered_bootstrap_ci(
            delta,
            metric=metric,
            resamples=resamples,
            seed=seed + index,
            confidence_level=confidence_level,
        )
        for index, metric in enumerate(METRICS)
    }


def _rescue_break(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    _paired_outcomes(baseline, candidate)
    count = len(baseline)
    rescued = sum(
        left["rank"] > 1 and right["rank"] == 1
        for left, right in zip(baseline, candidate, strict=True)
    )
    broken = sum(
        left["rank"] == 1 and right["rank"] > 1
        for left, right in zip(baseline, candidate, strict=True)
    )
    return {
        "paired_track_count": count,
        "rescue_count": rescued,
        "break_count": broken,
        "rescue_fraction": rescued / count,
        "break_fraction": broken / count,
    }


def _paired_outcomes(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> None:
    if not baseline or [row["track_token"] for row in baseline] != [
        row["track_token"] for row in candidate
    ]:
        raise ValueError("baseline and candidate outcomes are not exactly paired")


def _rank_rows(
    scores: np.ndarray,
    tracks: Sequence[str],
    registered_by_track: Mapping[str, str],
) -> list[dict[str, Any]]:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.shape != (len(tracks), len(tracks)) or not np.isfinite(matrix).all():
        raise ValueError("retrieval score matrix differs")
    rows: list[dict[str, Any]] = []
    for query_index, track in enumerate(tracks):
        order = sorted(
            range(len(tracks)),
            key=lambda index: (-float(matrix[query_index, index]), tracks[index]),
        )
        rank = order.index(query_index) + 1
        rows.append(
            {
                "registered_identity_id": registered_by_track[track],
                "track_token": track,
                "rank": rank,
                "Rank-1": float(rank == 1),
                "MRR": 1.0 / rank,
                "Rank-5": float(rank <= 5),
            }
        )
    return rows


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("retrieval metric rows must not be empty")
    return {
        "query_track_count": len(rows),
        "gallery_track_count": len(rows),
        **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS},
    }


def _history_metrics(item: Mapping[str, Any]) -> dict[str, float]:
    try:
        metrics = item["dev"]["metrics"]
    except (KeyError, TypeError) as exc:
        raise ValueError("selection history DEV metrics are missing") from exc
    return {
        metric: _finite_number(metrics.get(metric), f"DEV {metric}", 0.0, 1.0)
        for metric in METRICS
    }


def _cache_row(index: int, role: str, row: Mapping[str, Any]) -> dict[str, Any]:
    quality = dict(row["quality"])
    signals = (
        quality["blur_score"],
        quality["contrast_score"],
        quality["detector_confidence"],
        quality["frontality"],
        1.0 - quality["mask_uncertainty"],
    )
    if any(value is None for value in signals):
        raise ValueError("localized N4 cache row has incomplete quality evidence")
    bounded = [min(1.0, max(0.0, float(value))) for value in signals]
    return {
        "row_index": index,
        "sample_token": row["sample_token"],
        "registered_identity_id": row["registered_dog_id"],
        "identity_token": row["identity_token"],
        "track_token": row["track_token"],
        "sequence_token": row["sequence_token"],
        "frame_index": row["frame_index"],
        "record_state": row["record_state"],
        "role": role,
        "quality": quality,
        "quality_weight": 0.5 + 0.5 * float(np.mean(bounded)),
        "source_hashes": {
            "source_sha256": row["source_sha256"],
            "crop_sha256": row["crop_sha256"],
            "soft_mask_sha256": row["soft_mask_sha256"],
            "binary_mask_sha256": row["binary_mask_sha256"],
        },
    }


def _read_bound_crop(root: Path, row: Mapping[str, Any]) -> Image.Image:
    relative = PurePosixPath(row["crop_path"])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != row["crop_path"]
    ):
        raise ValueError("N4 crop path is unsafe")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("N4 crop path is unsafe")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["crop_sha256"]:
        raise ValueError("N4 crop SHA-256 differs")
    with Image.open(io.BytesIO(payload)) as opened:
        image = opened.convert("RGB")
        image.load()
    return image


def _normalized_vector(value: object, context: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{context} must be a finite non-empty vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"{context} has a degenerate norm")
    if abs(norm - 1.0) > NORMALIZATION_TOLERANCE:
        raise ValueError(f"{context} is not N3-normalized")
    return np.asarray(vector / norm, dtype=np.float32)


def _normalized_float32_matrix(value: object, rows: int, context: str) -> np.ndarray:
    matrix = np.asarray(value)
    if (
        matrix.dtype != np.float32
        or matrix.ndim != 2
        or matrix.shape[0] != rows
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(f"{context} must be finite float32 [rows, dimension]")
    if not np.allclose(
        np.linalg.norm(matrix, axis=1), 1.0, atol=NORMALIZATION_TOLERANCE, rtol=0.0
    ):
        raise ValueError(f"{context} rows must be L2-normalized")
    return np.ascontiguousarray(matrix)


def _normalized_mean(values: np.ndarray) -> np.ndarray:
    mean = np.asarray(values, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("K5 mean prototype is degenerate")
    return np.asarray(mean / norm, dtype=np.float32)


def _publish_torch_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    validate_adapter_checkpoint(dict(payload))
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        load_adapter_checkpoint(temporary)
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().to(dtype=torch.float32).contiguous().clone()
        for name, tensor in state.items()
    }


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    if not state:
        raise ValueError("adapter state must not be empty")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(tensor, torch.Tensor)
        ):
            raise ValueError("adapter state must contain named tensors only")
        value = tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("adapter state contains non-finite tensors")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(
            ",".join(str(item) for item in value.shape).encode("ascii") + b"\0"
        )
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_metadata_sha256(payload: Mapping[str, Any]) -> str:
    return content_sha256(
        {
            key: value
            for key, value in payload.items()
            if key not in {"state_dict", "checkpoint_payload_sha256"}
        }
    )


def _worktree_provenance(repository: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    head = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_head": head,
        "worktree_dirty": bool(status),
        "git_status_porcelain_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
    }


def _code_hashes(repository: Path, paths: Sequence[str]) -> dict[str, str]:
    from artifact_contracts.source_provenance import build_source_provenance

    provenance = build_source_provenance(repository / relative for relative in paths)
    return {
        row["relative_path"]: row["content_sha256"]
        for row in provenance["code_source_files"]
    }


def _document_binding(path: Path, document: Any) -> dict[str, Any]:
    return {
        "path": os.fspath(path),
        "raw_sha256": document.raw_sha256,
        "content_sha256": document.canonical_payload_sha256,
        "byte_size": document.byte_size,
    }


def _pinned_json(path: Path, expected: str, name: str, *, large: bool = False):
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} must be an absolute regular non-symlink file")
    limits = (
        {
            "maximum_bytes": 536_870_912,
            "maximum_nodes": 10_000_000,
            "maximum_keys": 5_000_000,
            "maximum_array_length": 1_000_000,
        }
        if large
        else {}
    )
    document = read_strict_json_document(source, **limits)
    if document.canonical_payload_sha256 != _sha256(expected, f"{name} canonical pin"):
        raise ValueError(f"{name} content SHA-256 differs from external pin")
    return document


def _require_same_file(candidate: Path, expected: Path, name: str) -> None:
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or candidate.resolve(strict=True) != expected.resolve(strict=True)
    ):
        raise ValueError(f"{name} is not the exact N3 lineage artifact")


def _repository(root: Path | None) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        if root is None
        else root.resolve(strict=True)
    )


def _new_external_path(path: Path, repository: Path, name: str) -> Path:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"{name} output must be absolute")
    absolute = Path(os.path.abspath(os.fspath(source)))
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"refusing to overwrite {name}: {absolute}")
    parent = absolute.parent.resolve(strict=True)
    result = parent / absolute.name
    if result.is_relative_to(repository):
        raise ValueError(f"{name} must be written outside the Git repository")
    return result


def _absolute_directory(path: Path, name: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink directory")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{name} must be a directory")
    return resolved


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be non-empty trimmed text")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(
    value: object,
    name: str,
    minimum: float,
    maximum: float,
    *,
    lower_open: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= minimum if lower_open else value < minimum)
        or value > maximum
    ):
        interval = (
            f"({minimum}, {maximum}]" if lower_open else f"[{minimum}, {maximum}]"
        )
        raise ValueError(f"{name} must be finite and in {interval}")
    return float(value)


__all__ = [
    "CACHE_BUNDLE_SCHEMA_VERSION",
    "CACHE_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "LIMITATIONS",
    "MAXIMUM_BOTTLENECK",
    "MAXIMUM_SCALE",
    "REPORT_BUNDLE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "DeterministicPKBatchSampler",
    "ResidualMetricAdapter",
    "adapter_from_checkpoint",
    "apply_adapter",
    "batch_hard_metric_loss",
    "build_adapter_checkpoint",
    "evaluate_k5",
    "evaluate_metric_adapter",
    "load_adapter_checkpoint",
    "load_embedding_cache",
    "materialize_embedding_cache",
    "select_dev_epoch",
    "train_metric_adapter",
    "validate_adapter_checkpoint",
    "validate_cache_manifest",
    "validate_evaluation_bundle",
]
