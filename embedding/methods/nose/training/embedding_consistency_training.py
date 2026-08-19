"""Receipt-bound raw/masked/degraded nose embedding consistency fine-tuning."""

from __future__ import annotations

import copy
import hashlib
import io
import math
import os
import random
import shutil
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from contracts.artifact_manifest import (
    ArtifactLicense,
    ImagePreprocessing,
    NoseEmbeddingManifest,
    UsageLane,
    preprocess_image,
)
from contracts.model_parity import (
    ModelParityReceipt,
    ModelUsageLane,
    ParityFixtureKind,
    ParityFixtureResult,
    ParityThresholds,
)
from foundation.protected_io import (
    json_document_bytes,
    read_strict_json_document,
    read_strict_json_object,
)
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from embedding.methods.nose.training.embedding_training import (
    ArcFaceClassificationHead,
    IdentityBalancedBatchSampler,
    NoseEmbeddingModel,
    NoseRegionCropDataset,
    build_dev_protocol,
    evaluate_dev_leave_one_out,
    export_static_onnx,
    load_receipt_bound_dinov2,
    validate_static_onnx,
)
from embedding.methods.nose.training.embedding_training import (
    load_embedding_checkpoint as load_parent_checkpoint,
)
from embedding.methods.nose.training.embedding_training import (
    validate_lineage_manifest as validate_parent_lineage,
)
from embedding.methods.nose.data.embedding_views import (
    load_embedding_views_manifest,
    reconstruct_student_masked_rgb,
)
from parsing.nose_region.manifest import read_nose_region_manifest
from parsing.nose_region.native_yt import validate_manifest_bundle

CHECKPOINT_SCHEMA = "cvi.nose_region_rgb_embedding_consistency_checkpoint.v3"
LINEAGE_SCHEMA = "cvi.nose_region_rgb_embedding_consistency_artifact_bundle.v3"
CONFIG_SCHEMA = "cvi.nose_region_rgb_embedding_consistency_training_config.v3"
SELECTION_SCHEMA = "cvi.nose_region_rgb_embedding_consistency_dev_selection.v3"
EVALUATION_SCHEMA = "cvi.nose_region_rgb_embedding_consistency_evaluation.v3"
MODEL_ID = "cvi.nose_region_rgb_embedding.dinov2-small-cls.v3"
LICENSE_ID = "CC-BY-NC-4.0-derived"
INTERPRETATION = (
    "RESEARCH_ONLY_RAW_MASKED_DEGRADED_CONSISTENCY_NOT_BIOMETRIC_VALIDATION"
)
IMAGE_SIZE = 224
EMBEDDING_DIM = 384
DEV_FRACTION = 0.30
PARTITION_SALT = "73:"
MIN_EVALUATION_FRAMES = 10
_CODE_PATHS = (
    "embedding/methods/nose/training/embedding_consistency_training.py",
    "embedding/methods/nose/training/embedding_training.py",
    "embedding/methods/nose/data/embedding_views.py",
    "parsing/nose_region/native_yt.py",
    "legacy/version/nose/workflows/train_nose_region_consistency.py",
)
_PRE_EMBEDDING_CODE_PATHS = tuple(
    path.replace(
        "embedding/methods/nose/training/", "parsing/nose_region/", 1
    ).replace(
        "embedding/methods/nose/data/embedding_views.py",
        "parsing/nose_region/embedding_views.py",
        1,
    )
    for path in _CODE_PATHS
)
_LEGACY_CODE_PATHS = tuple(
    path.replace("parsing/", "localization/", 1)
    if path.startswith("parsing/")
    else path
    for path in _PRE_EMBEDDING_CODE_PATHS
)
LOSS_WEIGHTS = {
    "raw_parent_anchor": 10.0,
    "masked_parent_consistency": 1.0,
    "degraded_parent_consistency": 0.25,
    "student_raw_temporal_consistency": 0.02,
}
OLD_MPDD_MAP_TOLERANCE = 0.01
NATIVE_RAW_MAP_TOLERANCE = 0.01
DEGRADATION_WEIGHTS = {
    "downsample": 1.0,
    "blur": 0.9,
    "JPEG": 0.8,
    "noise": 0.7,
}
_CHECKPOINT_KEYS = {
    "schema_version",
    "epoch",
    "global_step",
    "selection",
    "bindings",
    "bindings_sha256",
    "training_config",
    "training_config_sha256",
    "model_state_dict",
    "model_state_sha256",
    "arcface_state_dict",
    "arcface_state_sha256",
    "parent_model_state_sha256",
    "parent_arcface_state_sha256",
    "checkpoint_payload_sha256",
}


def build_identity_partitions(
    old_records: Sequence[Mapping[str, Any]],
    native_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the fixed identity-disjoint native SSL/DEV/EVAL partition."""

    parent_seen = sorted(
        {
            row["registered_dog_id"]
            for row in old_records
            if row.get("split_role") == "TRAIN"
            and row.get("dataset_name") == "yt-bb-dog"
        }
    )
    for identity in parent_seen:
        _canonical_uuid5(identity, "old TRAIN YT identity")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in native_records:
        identity = _canonical_uuid5(row.get("registered_dog_id"), "native identity")
        state = row.get("record_state")
        if state not in {"AVAILABLE", "LOW_QUALITY", "NO_ROI"}:
            raise ValueError("native record state differs")
        grouped[identity].append(row)

    parent_seen_set = set(parent_seen)
    ssl_identities: list[str] = []
    dev_identities: list[str] = []
    eval_identities: list[str] = []
    excluded_parent_seen = sorted(set(grouped) & parent_seen_set)
    ssl_identities.extend(excluded_parent_seen)
    for identity in sorted(set(grouped) - parent_seen_set):
        localized_count = sum(
            row["record_state"] in {"AVAILABLE", "LOW_QUALITY"}
            for row in grouped[identity]
        )
        if localized_count < MIN_EVALUATION_FRAMES:
            ssl_identities.append(identity)
            continue
        digest = hashlib.sha256(f"{PARTITION_SALT}{identity}".encode("ascii")).digest()
        if int.from_bytes(digest, "big") < int(DEV_FRACTION * (1 << 256)):
            dev_identities.append(identity)
        else:
            eval_identities.append(identity)

    ssl_identities.sort()

    ownership = {
        **{identity: "SSL_TRAIN" for identity in ssl_identities},
        **{identity: "DEV" for identity in dev_identities},
        **{identity: "EVAL" for identity in eval_identities},
    }
    sample_tokens: dict[str, list[str]] = {
        "ssl_train": [],
        "dev": [],
        "eval": [],
        "excluded_no_roi": [],
    }
    for identity, rows in sorted(grouped.items()):
        for row in sorted(rows, key=lambda item: item["sample_token"]):
            token = _sha256(row.get("sample_token"), "native sample token")
            state = row["record_state"]
            if state == "NO_ROI":
                sample_tokens["excluded_no_roi"].append(token)
            elif ownership[identity] == "SSL_TRAIN":
                sample_tokens["ssl_train"].append(token)
            elif ownership[identity] == "DEV":
                sample_tokens["dev"].append(token)
            else:
                sample_tokens["eval"].append(token)
    for tokens in sample_tokens.values():
        tokens.sort()

    identity_lists = {
        "parent_seen_yt": parent_seen,
        "parent_seen_native_ssl_train": excluded_parent_seen,
        "ssl_train": ssl_identities,
        "dev": dev_identities,
        "eval": eval_identities,
    }
    if set(ssl_identities) & set(dev_identities) or set(ssl_identities) & set(
        eval_identities
    ) or set(dev_identities) & set(eval_identities):
        raise RuntimeError("native identity partitions overlap")
    body = {
        "schema_version": "cvi.nose_region_embedding_consistency_splits.v1",
        "rule": {
            "parent_seen": "old TRAIN yt-bb-dog registered_dog_id",
            "minimum_localized_frames_for_dev_eval": MIN_EVALUATION_FRAMES,
            "hash": "SHA256('73:'+canonical_UUIDv5)",
            "dev_fraction": DEV_FRACTION,
            "remaining_fraction": 1.0 - DEV_FRACTION,
            "low_quality_usage": (
                "SSL_TRAIN_OR_FIXED_DIAGNOSTIC_ONLY_NEVER_IDENTITY_SUPERVISION"
            ),
        },
        "identity_lists": identity_lists,
        "sample_token_lists": sample_tokens,
    }
    return {**body, "splits_sha256": content_sha256(body)}


def records_for_partition(
    native_records: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, Any],
    role: str,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve a bound partition to its exact canonical native records."""

    if role not in {"ssl_train", "dev", "eval"}:
        raise ValueError("native partition role differs")
    expected = partitions["sample_token_lists"][role]
    by_token = {row["sample_token"]: row for row in native_records}
    if len(by_token) != len(native_records) or any(token not in by_token for token in expected):
        raise ValueError("native partition sample population differs")
    rows = tuple(by_token[token] for token in expected)
    allowed = {"AVAILABLE", "LOW_QUALITY"}
    if any(row["record_state"] not in allowed for row in rows):
        raise ValueError("native partition contains an inadmissible record state")
    return rows


def select_native_frame_pairs(
    records: Sequence[Mapping[str, Any]], *, seed: int, epoch: int
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    """Select exactly one deterministic temporal pair per SSL identity and epoch."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("pair seed must be an integer")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("pair epoch must be nonnegative")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("record_state") not in {"AVAILABLE", "LOW_QUALITY"}:
            raise ValueError("SSL pair records must be localized")
        grouped[row["registered_dog_id"]].append(row)
    pairs = []
    for identity, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: (row["frame_index"], row["sample_token"]))
        key = f"{seed}:{epoch}:{identity}".encode("ascii")
        first = int.from_bytes(hashlib.sha256(key + b":first").digest(), "big") % len(
            ordered
        )
        if len(ordered) == 1:
            second = first
        else:
            offset = 1 + int.from_bytes(
                hashlib.sha256(key + b":second").digest(), "big"
            ) % (len(ordered) - 1)
            second = (first + offset) % len(ordered)
        pairs.append((ordered[first], ordered[second]))
    if not pairs:
        raise ValueError("SSL TRAIN requires at least one localized native identity")
    return tuple(pairs)


def deterministic_mild_degradation(
    crop_rgb: np.ndarray, *, seed: int, epoch: int, sample_token: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply exactly one deterministic mild degradation and report its weight."""

    image = np.asarray(crop_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("degradation input must be HxWx3 uint8 RGB")
    _sha256(sample_token, "degradation sample token")
    material = f"{seed}:{epoch}:{sample_token}".encode("ascii")
    digest = hashlib.sha256(material).digest()
    kinds = ("downsample", "blur", "JPEG", "noise")
    kind = kinds[digest[0] % len(kinds)]
    pil = Image.fromarray(image, mode="RGB")
    parameter: dict[str, Any]
    if kind == "downsample":
        factor = (0.75, 0.85, 0.95)[digest[1] % 3]
        size = (
            max(1, round(pil.width * factor)),
            max(1, round(pil.height * factor)),
        )
        degraded = pil.resize(size, Image.Resampling.BICUBIC).resize(
            pil.size, Image.Resampling.BICUBIC
        )
        parameter = {"scale": factor}
    elif kind == "blur":
        radius = (0.2, 0.5, 0.8)[digest[1] % 3]
        degraded = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        parameter = {"radius": radius}
    elif kind == "JPEG":
        quality = (80, 85, 90, 95)[digest[1] % 4]
        stream = io.BytesIO()
        pil.save(stream, format="JPEG", quality=quality, subsampling=0)
        with Image.open(io.BytesIO(stream.getvalue())) as opened:
            degraded = opened.convert("RGB")
            degraded.load()
        parameter = {"quality": quality, "subsampling": 0}
    else:
        sigma = (0.5, 1.0, 2.0)[digest[1] % 3]
        rng = np.random.default_rng(int.from_bytes(digest[2:10], "big"))
        values = np.clip(
            image.astype(np.float32) + rng.normal(0.0, sigma, image.shape), 0, 255
        )
        result = np.rint(values).astype(np.uint8)
        parameter = {"sigma": sigma}
        return result, {
            "kind": kind,
            "parameter": parameter,
            "loss_weight": DEGRADATION_WEIGHTS[kind],
        }
    result = np.asarray(degraded, dtype=np.uint8).copy()
    return result, {
        "kind": kind,
        "parameter": parameter,
        "loss_weight": DEGRADATION_WEIGHTS[kind],
    }


def old_supervised_loss(
    student_raw: torch.Tensor,
    parent_raw: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ArcFace cross-entropy plus the frozen-parent raw cosine anchor."""

    classification = F.cross_entropy(logits, labels)
    anchor = _cosine_distance(student_raw, parent_raw).mean()
    total = classification + LOSS_WEIGHTS["raw_parent_anchor"] * anchor
    _finite_loss(total)
    return total, {"classification": classification, "raw_parent_anchor": anchor}


def native_consistency_loss(
    student_raw: torch.Tensor,
    student_masked: torch.Tensor,
    student_degraded: torch.Tensor,
    parent_raw: torch.Tensor,
    mask_confidence: torch.Tensor,
    degradation_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the exact two-frame raw/masked/degraded consistency objective."""

    if student_raw.ndim != 3 or student_raw.shape[1:] != (2, EMBEDDING_DIM):
        raise ValueError("native student raw embeddings must have shape [B,2,384]")
    if student_masked.shape != student_raw.shape or student_degraded.shape != student_raw.shape:
        raise ValueError("native embedding view shapes differ")
    if parent_raw.shape != student_raw.shape:
        raise ValueError("native parent target shape differs")
    if mask_confidence.shape != student_raw.shape[:2] or degradation_weight.shape != student_raw.shape[:2]:
        raise ValueError("native consistency weights must have shape [B,2]")
    if torch.any(mask_confidence < 0) or torch.any(mask_confidence > 1):
        raise ValueError("mask confidence must be in [0,1]")
    if torch.any(degradation_weight < 0) or torch.any(degradation_weight > 1):
        raise ValueError("degradation weight must be in [0,1]")
    raw_anchor = _cosine_distance(student_raw, parent_raw).mean()
    masked = (_cosine_distance(student_masked, parent_raw) * mask_confidence).mean()
    degraded = (
        _cosine_distance(student_degraded, parent_raw) * degradation_weight
    ).mean()
    temporal = _cosine_distance(student_raw[:, 0], student_raw[:, 1]).mean()
    total = (
        LOSS_WEIGHTS["raw_parent_anchor"] * raw_anchor
        + LOSS_WEIGHTS["masked_parent_consistency"] * masked
        + LOSS_WEIGHTS["degraded_parent_consistency"] * degraded
        + LOSS_WEIGHTS["student_raw_temporal_consistency"] * temporal
    )
    _finite_loss(total)
    return total, {
        "raw_parent_anchor": raw_anchor,
        "masked_parent_consistency": masked,
        "degraded_parent_consistency": degraded,
        "student_raw_temporal_consistency": temporal,
    }


def initialize_from_parent(
    student: torch.nn.Module,
    arcface: torch.nn.Module,
    parent_checkpoint: Mapping[str, Any],
) -> torch.nn.Module:
    """Initialize every student/head tensor exactly and return a frozen parent copy."""

    model_state = parent_checkpoint.get("model_state_dict")
    arcface_state = parent_checkpoint.get("arcface_state_dict")
    if not isinstance(model_state, Mapping) or not isinstance(arcface_state, Mapping):
        raise ValueError("parent checkpoint state dictionaries differ")
    student.load_state_dict(model_state, strict=True)
    arcface.load_state_dict(arcface_state, strict=True)
    if _state_dict_sha256(student.state_dict(), "student state") != parent_checkpoint.get(
        "model_state_sha256"
    ):
        raise RuntimeError("student model initialization differs from parent")
    if _state_dict_sha256(arcface.state_dict(), "ArcFace state") != parent_checkpoint.get(
        "arcface_state_sha256"
    ):
        raise RuntimeError("ArcFace initialization differs from parent")
    parent = copy.deepcopy(student).eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in parent.parameters()):
        raise RuntimeError("frozen parent contains trainable parameters")
    return parent


def evaluate_native_k5(
    embeddings: np.ndarray,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate earliest-five gallery against latest-five query per identity."""

    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.dtype != np.float32
        or embeddings.shape != (len(records), EMBEDDING_DIM)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("native embeddings must be finite float32 [records,384]")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4, rtol=0.0):
        raise ValueError("native embeddings must be L2-normalized")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[row["registered_dog_id"]].append(index)
    identities = sorted(grouped)
    if len(identities) < 2:
        raise ValueError("native K5 evaluation requires at least two identities")
    gallery: list[np.ndarray] = []
    query: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []
    for identity in identities:
        indices = sorted(
            grouped[identity],
            key=lambda index: (records[index]["frame_index"], records[index]["sample_token"]),
        )
        if len(indices) < MIN_EVALUATION_FRAMES:
            raise ValueError("native K5 identity has fewer than ten frames")
        gallery_indices, query_indices = indices[:5], indices[-5:]
        if set(gallery_indices) & set(query_indices):
            raise RuntimeError("native K5 gallery and query windows overlap")
        gallery.append(_normalized_mean(embeddings[gallery_indices]))
        query.append(_normalized_mean(embeddings[query_indices]))
        samples.append(
            {
                "registered_dog_id": identity,
                "gallery_sample_tokens": [records[index]["sample_token"] for index in gallery_indices],
                "query_sample_tokens": [records[index]["sample_token"] for index in query_indices],
            }
        )
    scores = np.stack(query) @ np.stack(gallery).T
    ranks: list[int] = []
    for index in range(len(identities)):
        order = sorted(range(len(identities)), key=lambda item: (-float(scores[index, item]), identities[item]))
        ranks.append(order.index(index) + 1)
    return {
        "aggregation": "EARLIEST_5_GALLERY_LATEST_5_QUERY_L2_NORMALIZED_MEAN",
        "identity_count": len(identities),
        "gallery_count": len(identities),
        "query_count": len(identities),
        "Rank-1": float(np.mean([rank == 1 for rank in ranks])),
        "mAP": float(np.mean([1.0 / rank for rank in ranks])),
        "identities": identities,
        "windows": samples,
    }


def select_epoch(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the best admissible epoch, retaining epoch zero as a fallback."""

    if not history or history[0].get("epoch") != 0:
        raise ValueError("selection history must begin at epoch zero")
    baseline = history[0]["dev"]
    old_baseline = _metric(baseline["old_mpdd_raw"], "mAP")
    raw_baseline = _metric(baseline["native_raw_k5"], "mAP")
    masked_baseline = _metric(baseline["native_masked_k5"], "mAP")
    candidates: list[tuple[tuple[float, float, float, int], Mapping[str, Any]]] = []
    decisions: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    for item in history:
        epoch = item.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0 or epoch in seen_epochs:
            raise ValueError("selection epochs must be unique nonnegative integers")
        seen_epochs.add(epoch)
        dev = item["dev"]
        old_map = _metric(dev["old_mpdd_raw"], "mAP")
        raw_map = _metric(dev["native_raw_k5"], "mAP")
        masked_map = _metric(dev["native_masked_k5"], "mAP")
        raw_improvement = raw_map - raw_baseline
        masked_improvement = masked_map - masked_baseline
        admissible = (
            old_map >= old_baseline - OLD_MPDD_MAP_TOLERANCE
            and raw_map >= raw_baseline - NATIVE_RAW_MAP_TOLERANCE
            and masked_map >= masked_baseline
        )
        objective = (
            masked_improvement,
            raw_improvement + masked_improvement,
            raw_map,
            -epoch,
        )
        decisions.append(
            {
                "epoch": epoch,
                "admissible": admissible,
                "old_mpdd_mAP_floor": old_baseline - OLD_MPDD_MAP_TOLERANCE,
                "native_raw_mAP_floor": raw_baseline
                - NATIVE_RAW_MAP_TOLERANCE,
                "raw_mAP_improvement": raw_improvement,
                "masked_mAP_improvement": masked_improvement,
                "objective": list(objective),
            }
        )
        if admissible:
            candidates.append((objective, item))
    if not candidates:  # epoch zero is admissible by construction unless metrics are malformed
        raise RuntimeError("no admissible consistency checkpoint, including epoch zero")
    objective, selected = max(candidates, key=lambda pair: pair[0])
    return {
        "selected_epoch": selected["epoch"],
        "selected_objective": list(objective),
        "epoch0_baselines": {
            "old_mpdd_raw_mAP": old_baseline,
            "native_raw_k5_mAP": raw_baseline,
            "native_masked_k5_mAP": masked_baseline,
        },
        "admissibility": {
            "old_mpdd_raw_mAP_tolerance": OLD_MPDD_MAP_TOLERANCE,
            "native_raw_k5_mAP_tolerance": NATIVE_RAW_MAP_TOLERANCE,
            "native_masked_k5_mAP_minimum": "EPOCH0",
        },
        "objective_order": [
            "masked_mAP_improvement",
            "sum(raw_mAP_improvement,masked_mAP_improvement)",
            "raw_K5_mAP",
            "earlier_epoch",
        ],
        "decisions": decisions,
    }


def build_consistency_checkpoint(
    *,
    model: torch.nn.Module,
    arcface: torch.nn.Module,
    epoch: int,
    global_step: int,
    selection: Mapping[str, Any],
    bindings: Mapping[str, Any],
    training_config: Mapping[str, Any],
    parent_model_state_sha256: str,
    parent_arcface_state_sha256: str,
) -> dict[str, Any]:
    """Build a weights-only-safe v2 checkpoint binding all state and metadata."""

    model_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    arcface_state = {
        key: value.detach().cpu().clone() for key, value in arcface.state_dict().items()
    }
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "global_step": global_step,
        "selection": dict(selection),
        "bindings": dict(bindings),
        "bindings_sha256": content_sha256(dict(bindings)),
        "training_config": dict(training_config),
        "training_config_sha256": content_sha256(dict(training_config)),
        "model_state_dict": model_state,
        "model_state_sha256": _state_dict_sha256(model_state, "model state"),
        "arcface_state_dict": arcface_state,
        "arcface_state_sha256": _state_dict_sha256(arcface_state, "ArcFace state"),
        "parent_model_state_sha256": _sha256(
            parent_model_state_sha256, "parent model state SHA-256"
        ),
        "parent_arcface_state_sha256": _sha256(
            parent_arcface_state_sha256, "parent ArcFace state SHA-256"
        ),
    }
    payload["checkpoint_payload_sha256"] = _checkpoint_metadata_sha256(payload)
    validate_consistency_checkpoint(payload)
    return payload


def validate_consistency_checkpoint(payload: object) -> None:
    """Strictly validate every consistency checkpoint tensor and metadata binding."""

    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        raise ValueError("consistency checkpoint keys differ")
    if payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported consistency checkpoint schema")
    for name in ("epoch", "global_step"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"consistency checkpoint {name} differs")
    if not isinstance(payload["selection"], dict) or not payload["selection"]:
        raise ValueError("consistency checkpoint selection differs")
    if not isinstance(payload["bindings"], dict) or not payload["bindings"]:
        raise ValueError("consistency checkpoint bindings differ")
    _validate_bindings(payload["bindings"])
    _validate_training_config(payload["training_config"])
    if content_sha256(payload["bindings"]) != payload["bindings_sha256"]:
        raise ValueError("consistency checkpoint bindings digest differs")
    if content_sha256(payload["training_config"]) != payload["training_config_sha256"]:
        raise ValueError("consistency checkpoint training config digest differs")
    for name in ("parent_model_state_sha256", "parent_arcface_state_sha256"):
        _sha256(payload[name], name)
    if payload["model_state_sha256"] != _state_dict_sha256(
        payload["model_state_dict"], "model state"
    ):
        raise ValueError("consistency checkpoint model state digest differs")
    if payload["arcface_state_sha256"] != _state_dict_sha256(
        payload["arcface_state_dict"], "ArcFace state"
    ):
        raise ValueError("consistency checkpoint ArcFace state digest differs")
    if payload["checkpoint_payload_sha256"] != _checkpoint_metadata_sha256(payload):
        raise ValueError("consistency checkpoint payload digest differs")


def replace_consistency_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    validate_consistency_checkpoint(dict(payload))
    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("consistency checkpoint target differs")
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        validate_consistency_checkpoint(
            torch.load(temporary, map_location="cpu", weights_only=True)
        )
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_consistency_checkpoint(
    path: Path,
    *,
    expected_bindings: Mapping[str, Any] | None = None,
    expected_training_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("consistency checkpoint must be a regular non-symlink file")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    validate_consistency_checkpoint(payload)
    if expected_bindings is not None and payload["bindings"] != dict(expected_bindings):
        raise ValueError("consistency checkpoint bindings differ from expected bindings")
    if expected_training_config is not None and payload["training_config"] != dict(
        expected_training_config
    ):
        raise ValueError("consistency checkpoint config differs from expected config")
    return payload


def build_runtime_manifest(onnx_path: Path) -> NoseEmbeddingManifest:
    """Build the exact static batch-one v2 runtime manifest."""

    return NoseEmbeddingManifest(
        artifact_id=MODEL_ID,
        artifact_sha256=_file_sha256(onnx_path),
        input_name="images",
        input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        output_name="embedding",
        output_shape=(1, EMBEDDING_DIM),
        license=ArtifactLicense(LICENSE_ID, UsageLane.RESEARCH_ONLY),
        preprocessing=ImagePreprocessing(
            color_mode="RGB",
            layout="NCHW",
            dtype="float32",
            resize="bicubic",
            scale=1.0 / 255.0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            clahe=None,
        ),
    )


def produce_parity_receipt(
    *,
    model: torch.nn.Module,
    onnx_path: Path,
    runtime_manifest: NoseEmbeddingManifest,
    crop_root: Path,
    crop_record: Mapping[str, Any],
    crop_manifest_file_sha256: str,
    source_weights_sha256: str,
    weight_intake_receipt_sha256: str,
    preprocessor_intake_receipt_sha256: str,
    thresholds: ParityThresholds,
) -> ModelParityReceipt:
    """Check CPU ORT parity against an actual bound crop for the v2 artifact."""

    import onnxruntime as ort

    path = _bound_crop_path(crop_root, crop_record)
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as opened:
        image = opened.convert("RGB")
    reference_model = model.to(torch.device("cpu")).eval()
    session = ort.InferenceSession(onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
    y, x = np.indices((61, 79), dtype=np.uint16)
    synthetic_values = np.stack(
        ((x + 17) % 256, (3 * y + 29) % 256, (x + 2 * y + 43) % 256),
        axis=2,
    ).astype(np.uint8)
    synthetic = Image.fromarray(synthetic_values, mode="RGB")
    synthetic_stream = io.BytesIO()
    synthetic.save(synthetic_stream, format="PNG")
    fixtures = (
        (
            f"crop-{crop_record['sample_token']}",
            ParityFixtureKind.RECEIPT_BOUND_CROP,
            crop_record["crop_sha256"],
            image,
        ),
        (
            "synthetic-gradient",
            ParityFixtureKind.SYNTHETIC,
            hashlib.sha256(synthetic_stream.getvalue()).hexdigest(),
            synthetic,
        ),
    )
    results = []
    for fixture_id, fixture_kind, input_sha256, fixture_image in fixtures:
        tensor = preprocess_image(fixture_image, runtime_manifest)
        with torch.inference_mode():
            reference = reference_model(torch.from_numpy(tensor)).numpy()
        candidate = session.run(["embedding"], {"images": tensor})[0]
        for name, values in (("PyTorch", reference), ("CPU ORT", candidate)):
            if values.dtype != np.float32 or values.shape != (1, EMBEDDING_DIM) or not np.isfinite(values).all():
                raise ValueError(f"{name} parity output differs")
        difference = np.abs(reference - candidate)
        absolute = float(difference.max())
        relative = float(
            np.max(difference / np.maximum(np.abs(reference), thresholds.relative_error_floor))
        )
        cosine = float(reference[0] @ candidate[0])
        if (
            absolute > thresholds.maximum_absolute_error
            or relative > thresholds.maximum_relative_error
            or cosine < thresholds.minimum_cosine_similarity
        ):
            raise RuntimeError(f"v2 nose embedding CPU ORT parity failed for {fixture_id}")
        results.append(
            ParityFixtureResult(
                fixture_id=fixture_id,
                fixture_kind=fixture_kind,
                input_sha256=input_sha256,
                reference_output_sha256=hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest(),
                candidate_output_sha256=hashlib.sha256(np.ascontiguousarray(candidate).tobytes()).hexdigest(),
                maximum_absolute_error=absolute,
                maximum_relative_error=relative,
                cosine_similarity=min(1.0, cosine),
                decision="PASS",
            )
        )
    return ModelParityReceipt(
        model_id=MODEL_ID,
        artifact_sha256=runtime_manifest.artifact_sha256,
        source_weights_sha256=_sha256(source_weights_sha256, "source weights SHA-256"),
        weight_intake_receipt_sha256=_sha256(
            weight_intake_receipt_sha256, "weight intake receipt SHA-256"
        ),
        preprocessing_sha256=content_sha256(runtime_manifest.preprocessing.to_dict()),
        preprocessor_intake_receipt_sha256=_sha256(
            preprocessor_intake_receipt_sha256, "preprocessor intake receipt SHA-256"
        ),
        usage_lane=ModelUsageLane.RESEARCH_ONLY,
        reference_backend=f"torch={torch.__version__};selected-consistency-checkpoint",
        candidate_backend=f"onnxruntime-cpu={ort.__version__}",
        thresholds=thresholds,
        fixture_panel_receipt_sha256=_sha256(
            crop_manifest_file_sha256, "crop manifest file SHA-256"
        ),
        fixtures=tuple(results),
        decision="PASS",
    )


class NativeConsistencyPairDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Materialize deterministic native raw/masked/degraded temporal pairs."""

    def __init__(
        self,
        *,
        native_root: Path,
        support_root: Path,
        pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
        view_records: Mapping[str, Mapping[str, Any]],
        seed: int,
        epoch: int,
    ) -> None:
        self._native_root = native_root
        self._support_root = support_root
        self._pairs = tuple(pairs)
        self._views = view_records
        self._seed = seed
        self._epoch = epoch
        self.degradation_reports: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        raw_tensors, masked_tensors, degraded_tensors = [], [], []
        confidences, degradation_weights = [], []
        for row in self._pairs[index]:
            token = row["sample_token"]
            view = self._views[token]
            path = _bound_crop_path(self._native_root, row)
            with Image.open(path) as opened:
                raw = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            masked = reconstruct_student_masked_rgb(
                native_root=self._native_root,
                view_root=self._support_root,
                native_record=row,
                view_record=view,
            )
            degraded, report = deterministic_mild_degradation(
                raw, seed=self._seed, epoch=self._epoch, sample_token=token
            )
            report = {"sample_token": token, **report}
            self.degradation_reports.append(report)
            raw_tensors.append(_image_tensor(raw))
            masked_tensors.append(_image_tensor(masked))
            degraded_tensors.append(_image_tensor(degraded))
            confidences.append(1.0 - float(view["mean_binary_uncertainty"]))
            degradation_weights.append(float(report["loss_weight"]))
        return (
            torch.stack(raw_tensors),
            torch.stack(masked_tensors),
            torch.stack(degraded_tensors),
            torch.tensor(confidences, dtype=torch.float32),
            torch.tensor(degradation_weights, dtype=torch.float32),
        )


class NativeViewDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        *,
        native_root: Path,
        support_root: Path,
        records: Sequence[Mapping[str, Any]],
        view_records: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self._native_root = native_root
        self._support_root = support_root
        self._records = tuple(records)
        self._views = view_records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._records[index]
        path = _bound_crop_path(self._native_root, row)
        with Image.open(path) as opened:
            raw = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        masked = reconstruct_student_masked_rgb(
            native_root=self._native_root,
            view_root=self._support_root,
            native_record=row,
            view_record=self._views[row["sample_token"]],
        )
        return _image_tensor(raw), _image_tensor(masked)


def train_and_export(
    *,
    parent_lineage_path: Path,
    parent_lineage_sha256: str,
    parent_root: Path,
    parent_checkpoint_path: Path,
    parent_checkpoint_sha256: str,
    model_directory: Path,
    model_directory_sha256: str,
    weight_intake_bundle: Path,
    weight_intake_bundle_sha256: str,
    preprocessor_intake_bundle: Path,
    preprocessor_intake_bundle_sha256: str,
    old_crop_manifest_path: Path,
    old_crop_manifest_sha256: str,
    old_crop_root: Path,
    native_bundle_path: Path,
    native_bundle_sha256: str,
    native_root: Path,
    support_manifest_path: Path,
    support_manifest_sha256: str,
    support_root: Path,
    output_dir: Path,
    device_name: str = "cuda",
    epochs: int = 3,
    old_batch_size: int = 32,
    native_pair_batch_size: int = 8,
    backbone_lr: float = 5e-7,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    seed: int = 42,
    mixed_precision: bool = True,
    parity_thresholds: ParityThresholds = ParityThresholds(1e-4, 2e-2, 1e-4, 0.99999),
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Fine-tune, select, evaluate once, and atomically publish the v2 artifact."""

    _validate_arguments(
        epochs=epochs,
        old_batch_size=old_batch_size,
        native_pair_batch_size=native_pair_batch_size,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        seed=seed,
        mixed_precision=mixed_precision,
        device_name=device_name,
    )
    repository = (
        Path(__file__).resolve().parents[4]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    output = _external_new_directory(output_dir, repository)
    parent_root = _absolute_directory(parent_root, "parent root")
    old_crop_root = _absolute_directory(old_crop_root, "old crop root")
    native_root = _absolute_directory(native_root, "native root")
    support_root = _absolute_directory(support_root, "support root")
    parent_document = _pinned_document(
        parent_lineage_path, parent_lineage_sha256, "parent lineage"
    )
    validate_parent_lineage(parent_document.payload, parent_root)
    parent_artifact = parent_document.payload["artifacts"]["selected_checkpoint"]
    _same_file(
        parent_checkpoint_path,
        parent_root.joinpath(*PurePosixPath(parent_artifact["path"]).parts),
        "parent selected checkpoint",
    )
    if _file_sha256(parent_checkpoint_path) != _sha256(
        parent_checkpoint_sha256, "parent checkpoint external pin"
    ) or parent_checkpoint_sha256 != parent_artifact["sha256"]:
        raise ValueError("parent selected checkpoint SHA-256 differs")
    parent_checkpoint = load_parent_checkpoint(
        parent_checkpoint_path,
        expected_bindings=parent_document.payload["bindings"],
        expected_training_config=parent_document.payload["training_config"],
    )

    _validate_model_directory_pin(model_directory, model_directory_sha256)
    _external_pin(weight_intake_bundle, weight_intake_bundle_sha256, "weight intake bundle")
    _external_pin(
        preprocessor_intake_bundle,
        preprocessor_intake_bundle_sha256,
        "preprocessor intake bundle",
    )
    old_document = _pinned_document(
        old_crop_manifest_path, old_crop_manifest_sha256, "old crop manifest"
    )
    if old_crop_manifest_path.parent.resolve(strict=True) != old_crop_root:
        raise ValueError("old crop manifest must be rooted at old_crop_root")
    old_manifest = read_nose_region_manifest(old_crop_manifest_path)
    native_document = _pinned_document(
        native_bundle_path, native_bundle_sha256, "native v4 bundle"
    )
    native_manifest = validate_manifest_bundle(native_document.payload, root=native_root)
    native_records = tuple(native_manifest["records"])
    old_records = tuple(old_manifest["records"])
    old_train = tuple(row for row in old_records if row["split_role"] == "TRAIN")
    old_dev = tuple(row for row in old_records if row["split_role"] == "DEV")
    old_identities = sorted({row["registered_dog_id"] for row in old_train})
    parent_identities = parent_checkpoint["bindings"]["identity_populations"][
        "train_registered_dog_ids"
    ]
    if old_identities != parent_identities:
        raise ValueError("old TRAIN identities differ from parent ArcFace class order")
    old_protocol = build_dev_protocol(old_dev)
    partitions = build_identity_partitions(old_records, native_records)
    ssl_records = records_for_partition(native_records, partitions, "ssl_train")
    native_dev = records_for_partition(native_records, partitions, "dev")
    native_eval = records_for_partition(native_records, partitions, "eval")
    _validate_eval_population(native_dev, "native DEV")
    _validate_eval_population(native_eval, "native EVAL")
    support_manifest = load_embedding_views_manifest(
        support_manifest_path,
        expected_payload_sha256=_sha256(
            support_manifest_sha256, "support manifest external pin"
        ),
        root=support_root,
        repository_root=repository,
    )
    if support_manifest["source_binding"]["native_bundle_payload_sha256"] != native_bundle_sha256:
        raise ValueError("support cache is not bound to the supplied native bundle")
    localized_tokens = {
        row["sample_token"] for row in native_records if row["record_state"] != "NO_ROI"
    }
    view_records = {row["sample_token"]: row for row in support_manifest["records"]}
    if set(view_records) != localized_tokens:
        raise ValueError("support cache and localized native population differ")

    model, contract = load_receipt_bound_dinov2(
        model_directory=model_directory,
        weight_intake_bundle=weight_intake_bundle,
        preprocessor_intake_bundle=preprocessor_intake_bundle,
    )
    if contract.model_sha256 != parent_checkpoint["bindings"]["dinov2"]["weight_sha256"]:
        raise ValueError("DINO source weights differ from parent checkpoint")
    arcface = ArcFaceClassificationHead(
        len(old_identities),
        scale=float(parent_checkpoint["training_config"]["arcface"]["scale"]),
        margin=float(parent_checkpoint["training_config"]["arcface"]["margin"]),
    )
    frozen_parent = initialize_from_parent(model, arcface, parent_checkpoint)

    training_config = _training_config(
        epochs=epochs,
        old_batch_size=old_batch_size,
        native_pair_batch_size=native_pair_batch_size,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        seed=seed,
        mixed_precision=mixed_precision,
        device_name=device_name,
        parity_thresholds=parity_thresholds,
    )
    bindings = _bindings(
        repository=repository,
        parent_document=parent_document,
        parent_checkpoint_path=parent_checkpoint_path,
        parent_checkpoint=parent_checkpoint,
        model_directory=model_directory,
        model_directory_sha256=model_directory_sha256,
        weight_intake_bundle=weight_intake_bundle,
        weight_intake_bundle_sha256=weight_intake_bundle_sha256,
        preprocessor_intake_bundle=preprocessor_intake_bundle,
        preprocessor_intake_bundle_sha256=preprocessor_intake_bundle_sha256,
        old_document=old_document,
        old_manifest=old_manifest,
        native_document=native_document,
        native_manifest=native_manifest,
        support_manifest_path=support_manifest_path,
        support_manifest=support_manifest,
        partitions=partitions,
        old_protocol=old_protocol,
        training_config=training_config,
    )
    _validate_bindings(bindings)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device_name == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(device_name)
    model.to(device)
    arcface.to(device)
    frozen_parent.to(device).eval()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": backbone_lr},
            {"params": arcface.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and mixed_precision
        else None
    )
    labels = [old_identities.index(row["registered_dog_id"]) for row in old_train]
    old_dataset = NoseRegionCropDataset(
        old_crop_root,
        old_train,
        labels,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        training=False,
        seed=seed,
    )
    old_sampler = IdentityBalancedBatchSampler(
        labels, batch_size=old_batch_size, samples_per_identity=1, seed=seed
    )
    old_loader = DataLoader(
        old_dataset,
        batch_sampler=old_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    old_dev_dataset = NoseRegionCropDataset(
        old_crop_root,
        old_dev,
        [0] * len(old_dev),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        training=False,
        seed=seed,
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        checkpoint_dir = staging / "checkpoints"
        checkpoint_dir.mkdir(mode=0o700)
        history: list[dict[str, Any]] = []
        global_step = 0
        for epoch in range(epochs + 1):
            train_report = None
            if epoch > 0:
                old_sampler.set_epoch(epoch - 1)
                pairs = select_native_frame_pairs(ssl_records, seed=seed, epoch=epoch - 1)
                native_dataset = NativeConsistencyPairDataset(
                    native_root=native_root,
                    support_root=support_root,
                    pairs=pairs,
                    view_records=view_records,
                    seed=seed,
                    epoch=epoch - 1,
                )
                native_loader = DataLoader(
                    native_dataset,
                    batch_size=native_pair_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                )
                train_report, steps = _train_epoch(
                    model,
                    frozen_parent,
                    arcface,
                    old_loader,
                    native_loader,
                    optimizer,
                    device,
                    scaler,
                )
                global_step += steps
            dev = _evaluate_dev(
                model=model,
                old_dev_dataset=old_dev_dataset,
                old_dev_records=old_dev,
                old_protocol=old_protocol,
                native_dev_records=native_dev,
                native_root=native_root,
                support_root=support_root,
                view_records=view_records,
                device=device,
                batch_size=max(old_batch_size, native_pair_batch_size),
                num_workers=num_workers,
            )
            history.append({"epoch": epoch, "train": train_report, "dev": dev})
            provisional = select_epoch(history)
            checkpoint = build_consistency_checkpoint(
                model=model,
                arcface=arcface,
                epoch=epoch,
                global_step=global_step,
                selection={"dev": dev, "selection_at_epoch": provisional},
                bindings=bindings,
                training_config=training_config,
                parent_model_state_sha256=parent_checkpoint["model_state_sha256"],
                parent_arcface_state_sha256=parent_checkpoint["arcface_state_sha256"],
            )
            replace_consistency_checkpoint(checkpoint_dir / "last.pt", checkpoint)
            if provisional["selected_epoch"] == epoch:
                replace_consistency_checkpoint(
                    checkpoint_dir / "selected.pt", checkpoint
                )

        selection = select_epoch(history)
        selected_epoch = selection["selected_epoch"]
        selected = load_consistency_checkpoint(
            checkpoint_dir / "selected.pt",
            expected_bindings=bindings,
            expected_training_config=training_config,
        )
        model.load_state_dict(selected["model_state_dict"], strict=True)
        model.to(device).eval()

        evaluation = {
            "schema_version": EVALUATION_SCHEMA,
            "selected_epoch": selected_epoch,
            "native_eval": _evaluate_native_views(
                model=model,
                records=native_eval,
                native_root=native_root,
                support_root=support_root,
                view_records=view_records,
                device=device,
                batch_size=max(old_batch_size, native_pair_batch_size),
                num_workers=num_workers,
            ),
            "evaluation_count": 1,
            "interpretation": INTERPRETATION,
        }
        selection_report = {
            "schema_version": SELECTION_SCHEMA,
            "selected_epoch": selected_epoch,
            "epoch0_selected": selected_epoch == 0,
            "selection": selection,
            "history": history,
            "interpretation": INTERPRETATION,
        }
        model.to(torch.device("cpu")).eval()
        onnx_path = staging / "nose_embedding.onnx"
        onnx_sha256, onnx_bytes = export_static_onnx(model, onnx_path)
        validate_static_onnx(onnx_path)
        runtime_manifest = build_runtime_manifest(onnx_path)
        if runtime_manifest.artifact_sha256 != onnx_sha256:
            raise RuntimeError("v2 runtime manifest ONNX digest differs")
        runtime_path = staging / "nose_embedding.runtime.json"
        _write_exclusive(runtime_path, json_document_bytes(runtime_manifest.to_dict()))
        parity_record = native_eval[0] if native_eval else old_dev[0]
        parity_root = native_root if native_eval else old_crop_root
        parity = produce_parity_receipt(
            model=model,
            onnx_path=onnx_path,
            runtime_manifest=runtime_manifest,
            crop_root=parity_root,
            crop_record=parity_record,
            crop_manifest_file_sha256=(
                native_document.raw_sha256 if native_eval else old_document.raw_sha256
            ),
            source_weights_sha256=contract.model_sha256,
            weight_intake_receipt_sha256=contract.weight_receipt_sha256,
            preprocessor_intake_receipt_sha256=contract.preprocessor_receipt_sha256,
            thresholds=parity_thresholds,
        )
        parity_path = staging / "nose_embedding.parity.json"
        selection_path = staging / "dev_selection.json"
        evaluation_path = staging / "evaluation.json"
        _write_exclusive(parity_path, json_document_bytes(parity.to_dict()))
        _write_exclusive(selection_path, json_document_bytes(selection_report))
        _write_exclusive(evaluation_path, json_document_bytes(evaluation))
        lineage = _build_lineage(
            root=staging,
            onnx_bytes=onnx_bytes,
            bindings=bindings,
            training_config=training_config,
            selection_report=selection_report,
            evaluation=evaluation,
            parity=parity,
        )
        lineage_path = staging / "artifact_lineage.json"
        _write_exclusive(lineage_path, json_document_bytes(lineage))
        validate_lineage_manifest(lineage, staging)
        fsync_directory(checkpoint_dir)
        fsync_directory(staging)
        rename_directory_noreplace(staging, output)
        fsync_directory(output.parent)
        return lineage
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_lineage_manifest(payload: object, root: Path) -> None:
    """Validate the complete v2 consistency artifact publication."""

    expected = {
        "schema_version",
        "artifacts",
        "onnx_contract",
        "bindings",
        "bindings_sha256",
        "training_config",
        "training_config_sha256",
        "selection_payload_sha256",
        "evaluation_payload_sha256",
        "parity_payload_sha256",
        "license",
        "interpretation",
        "lineage_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("consistency lineage keys differ")
    body = {key: value for key, value in payload.items() if key != "lineage_sha256"}
    if payload["schema_version"] != LINEAGE_SCHEMA or payload["lineage_sha256"] != content_sha256(body):
        raise ValueError("consistency lineage schema or digest differs")
    _validate_bindings(payload["bindings"])
    _validate_training_config(payload["training_config"])
    if content_sha256(payload["bindings"]) != payload["bindings_sha256"]:
        raise ValueError("consistency lineage bindings digest differs")
    if content_sha256(payload["training_config"]) != payload["training_config_sha256"]:
        raise ValueError("consistency lineage config digest differs")
    if payload["license"] != {"license_id": LICENSE_ID, "usage_lane": "RESEARCH_ONLY"} or payload["interpretation"] != INTERPRETATION:
        raise ValueError("consistency lineage usage contract differs")
    artifacts = payload["artifacts"]
    paths = {
        "onnx": "nose_embedding.onnx",
        "runtime_manifest": "nose_embedding.runtime.json",
        "parity_receipt": "nose_embedding.parity.json",
        "selected_checkpoint": "checkpoints/selected.pt",
        "last_checkpoint": "checkpoints/last.pt",
        "dev_selection": "dev_selection.json",
        "evaluation": "evaluation.json",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(paths):
        raise ValueError("consistency lineage artifacts differ")
    resolved = root.resolve(strict=True)
    for name, relative in paths.items():
        binding = artifacts[name]
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256", "bytes"} or binding["path"] != relative:
            raise ValueError("consistency lineage artifact binding differs")
        path = resolved.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(resolved):
            raise ValueError("consistency lineage artifact path is unsafe")
        if _file_sha256(path) != binding["sha256"] or path.stat().st_size != binding["bytes"]:
            raise ValueError("consistency lineage artifact bytes differ")
    runtime = NoseEmbeddingManifest.from_dict(
        read_strict_json_object(resolved / paths["runtime_manifest"])
    )
    if runtime.artifact_id != MODEL_ID or runtime.artifact_sha256 != artifacts["onnx"]["sha256"] or runtime.license != ArtifactLicense(LICENSE_ID, UsageLane.RESEARCH_ONLY):
        raise ValueError("consistency runtime manifest differs")
    if payload["onnx_contract"] != {
        "input_name": "images",
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "output_name": "embedding",
        "output_shape": [1, EMBEDDING_DIM],
        "opset": 18,
        "external_data": False,
        "onnx_bytes": artifacts["onnx"]["bytes"],
    }:
        raise ValueError("consistency ONNX contract differs")
    selection = read_strict_json_object(resolved / paths["dev_selection"])
    evaluation = read_strict_json_object(resolved / paths["evaluation"])
    parity = ModelParityReceipt.from_dict(
        read_strict_json_object(resolved / paths["parity_receipt"])
    )
    if content_sha256(selection) != payload["selection_payload_sha256"] or selection.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("consistency selection report differs")
    if content_sha256(evaluation) != payload["evaluation_payload_sha256"] or evaluation.get("schema_version") != EVALUATION_SCHEMA or evaluation.get("evaluation_count") != 1:
        raise ValueError("consistency EVAL report differs")
    if parity.receipt_sha256 != payload["parity_payload_sha256"] or parity.model_id != MODEL_ID or parity.artifact_sha256 != artifacts["onnx"]["sha256"]:
        raise ValueError("consistency parity receipt differs")
    selected = load_consistency_checkpoint(
        resolved / paths["selected_checkpoint"],
        expected_bindings=payload["bindings"],
        expected_training_config=payload["training_config"],
    )
    last = load_consistency_checkpoint(
        resolved / paths["last_checkpoint"],
        expected_bindings=payload["bindings"],
        expected_training_config=payload["training_config"],
    )
    if selected["epoch"] != selection["selected_epoch"] or last["epoch"] != payload["training_config"]["epochs"] or evaluation["selected_epoch"] != selected["epoch"]:
        raise ValueError("consistency checkpoint selection differs")


def _train_epoch(
    model: NoseEmbeddingModel,
    parent: torch.nn.Module,
    arcface: ArcFaceClassificationHead,
    old_loader: DataLoader,
    native_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> tuple[dict[str, float], int]:
    model.train()
    arcface.train()
    totals: dict[str, float] = defaultdict(float)
    counts = {"old": 0, "native": 0}
    steps = 0
    for images, labels, _ in old_loader:
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            student = model(images)
            with torch.no_grad():
                target = parent(images)
            loss, parts = old_supervised_loss(student, target, arcface(student, labels), labels)
        _optimizer_step(loss, model, arcface, optimizer, scaler)
        count = int(labels.shape[0])
        totals["old_total"] += float(loss.detach()) * count
        for name, value in parts.items():
            totals[f"old_{name}"] += float(value.detach()) * count
        counts["old"] += count
        steps += 1
    for raw, masked, degraded, confidence, degradation_weight in native_loader:
        batch = raw.shape[0]
        raw = raw.to(device)
        masked = masked.to(device)
        degraded = degraded.to(device)
        confidence = confidence.to(device)
        degradation_weight = degradation_weight.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            student_raw = model(raw.flatten(0, 1)).reshape(batch, 2, EMBEDDING_DIM)
            student_masked = model(masked.flatten(0, 1)).reshape(batch, 2, EMBEDDING_DIM)
            student_degraded = model(degraded.flatten(0, 1)).reshape(batch, 2, EMBEDDING_DIM)
            with torch.no_grad():
                parent_raw = parent(raw.flatten(0, 1)).reshape(batch, 2, EMBEDDING_DIM)
            loss, parts = native_consistency_loss(
                student_raw,
                student_masked,
                student_degraded,
                parent_raw,
                confidence,
                degradation_weight,
            )
        _optimizer_step(loss, model, arcface, optimizer, scaler)
        totals["native_total"] += float(loss.detach()) * batch
        for name, value in parts.items():
            totals[f"native_{name}"] += float(value.detach()) * batch
        counts["native"] += batch
        steps += 1
    if min(counts.values()) <= 0:
        raise RuntimeError("consistency training pass was empty")
    return {
        **{name: value / counts[name.split("_", 1)[0]] for name, value in totals.items()},
        "old_sample_count": float(counts["old"]),
        "native_pair_count": float(counts["native"]),
    }, steps


def _optimizer_step(
    loss: torch.Tensor,
    model: torch.nn.Module,
    arcface: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
) -> None:
    _finite_loss(loss)
    parameters = [*model.parameters(), *arcface.parameters()]
    if scaler is None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        scaler.step(optimizer)
        scaler.update()


def _evaluate_dev(**kwargs: Any) -> dict[str, Any]:
    old_embeddings = _extract_old_embeddings(
        kwargs["model"],
        kwargs["old_dev_dataset"],
        device=kwargs["device"],
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
    )
    old = evaluate_dev_leave_one_out(
        old_embeddings, kwargs["old_dev_records"], kwargs["old_protocol"]
    )
    native = _evaluate_native_views(
        model=kwargs["model"],
        records=kwargs["native_dev_records"],
        native_root=kwargs["native_root"],
        support_root=kwargs["support_root"],
        view_records=kwargs["view_records"],
        device=kwargs["device"],
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
    )
    return {
        "old_mpdd_raw": old,
        "native_raw_k5": native["raw_k5"],
        "native_masked_k5": native["masked_k5"],
    }


def _evaluate_native_views(
    *,
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    native_root: Path,
    support_root: Path,
    view_records: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    dataset = NativeViewDataset(
        native_root=native_root,
        support_root=support_root,
        records=records,
        view_records=view_records,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.to(device).eval()
    raw_values, masked_values = [], []
    with torch.inference_mode():
        for raw, masked in loader:
            raw_values.append(model(raw.to(device)).cpu().numpy())
            masked_values.append(model(masked.to(device)).cpu().numpy())
    raw = np.ascontiguousarray(np.concatenate(raw_values), dtype=np.float32)
    masked = np.ascontiguousarray(np.concatenate(masked_values), dtype=np.float32)
    return {
        "raw_k5": evaluate_native_k5(raw, records),
        "masked_k5": evaluate_native_k5(masked, records),
    }


def _extract_old_embeddings(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    values = []
    with torch.inference_mode():
        for images, _, _ in loader:
            values.append(model(images.to(device)).cpu().numpy())
    return np.ascontiguousarray(np.concatenate(values), dtype=np.float32)


def _training_config(**values: Any) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "model": "DINOv2-small CLS L2-normalized 384D",
        "input": {
            "shape": [3, IMAGE_SIZE, IMAGE_SIZE],
            "resize": "DIRECT_BICUBIC_STRETCH",
            "scale": 1.0 / 255.0,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "epochs": values["epochs"],
        "old_batch_size": values["old_batch_size"],
        "native_pair_batch_size": values["native_pair_batch_size"],
        "optimizer": {
            "type": "AdamW",
            "backbone_lr": values["backbone_lr"],
            "arcface_head_lr": values["head_lr"],
            "weight_decay": values["weight_decay"],
        },
        "old_pass": {
            "samples_per_identity_per_epoch": 1,
            "losses": ["ArcFace_cross_entropy", "parent_raw_cosine_anchor"],
        },
        "native_pass": {
            "pairs_per_identity_per_epoch": 1,
            "views": ["raw", "student_masked", "mild_degraded"],
            "restoration_rgb_target": False,
            "loss_weights": dict(LOSS_WEIGHTS),
            "degradation_weights": dict(DEGRADATION_WEIGHTS),
        },
        "partition": {
            "salt": PARTITION_SALT,
            "dev_fraction": DEV_FRACTION,
            "minimum_evaluation_frames": MIN_EVALUATION_FRAMES,
        },
        "selection": {
            "old_mpdd_mAP_tolerance": OLD_MPDD_MAP_TOLERANCE,
            "native_raw_mAP_tolerance": NATIVE_RAW_MAP_TOLERANCE,
            "native_masked_mAP_floor": "EPOCH0",
            "epoch0_selectable": True,
        },
        "num_workers": values["num_workers"],
        "seed": values["seed"],
        "mixed_precision": values["mixed_precision"],
        "device": values["device_name"],
        "parity_thresholds": values["parity_thresholds"].to_dict(),
        "usage_lane": "RESEARCH_ONLY",
    }


def _bindings(**values: Any) -> dict[str, Any]:
    repository = values["repository"]
    parent_document = values["parent_document"]
    parent_checkpoint = values["parent_checkpoint"]
    old_document = values["old_document"]
    native_document = values["native_document"]
    support_manifest = values["support_manifest"]
    training_config = values["training_config"]
    return {
        "parent_embedding": {
            "lineage_file_sha256": parent_document.raw_sha256,
            "lineage_payload_sha256": parent_document.canonical_payload_sha256,
            "lineage_sha256": parent_document.payload["lineage_sha256"],
            "selected_checkpoint_file_sha256": _file_sha256(values["parent_checkpoint_path"]),
            "selected_checkpoint_payload_sha256": parent_checkpoint["checkpoint_payload_sha256"],
            "model_state_sha256": parent_checkpoint["model_state_sha256"],
            "arcface_state_sha256": parent_checkpoint["arcface_state_sha256"],
            "selected_epoch": parent_checkpoint["epoch"],
        },
        "dinov2": {
            "model_directory_sha256": values["model_directory_sha256"],
            "weight_intake_bundle_file_sha256": values["weight_intake_bundle_sha256"],
            "preprocessor_intake_bundle_file_sha256": values["preprocessor_intake_bundle_sha256"],
            "parent_dinov2_binding": parent_checkpoint["bindings"]["dinov2"],
        },
        "old_crop_manifest": {
            "file_sha256": old_document.raw_sha256,
            "payload_sha256": old_document.canonical_payload_sha256,
            "manifest_sha256": old_document.payload["manifest_sha256"],
            "summary_sha256": values["old_manifest"]["summary"]["summary_sha256"],
        },
        "native_v4_bundle": {
            "file_sha256": native_document.raw_sha256,
            "payload_sha256": native_document.canonical_payload_sha256,
            "manifest_sha256": native_document.payload["manifest_sha256"],
            "input_sha256s": values["native_manifest"]["input_sha256s"],
        },
        "support_cache": {
            "manifest_file_sha256": _file_sha256(values["support_manifest_path"]),
            "manifest_payload_sha256": content_sha256(support_manifest),
            "manifest_sha256": support_manifest["manifest_sha256"],
            "transform_sha256": support_manifest["transform_sha256"],
            "student_binding": support_manifest["student_binding"],
        },
        "splits": values["partitions"],
        "old_dev_protocol": values["old_protocol"],
        "code_sha256s": {
            relative: _file_sha256(repository.joinpath(*PurePosixPath(relative).parts))
            for relative in _CODE_PATHS
        },
        "config_sha256": content_sha256(training_config),
        "license": {"license_id": LICENSE_ID, "usage_lane": "RESEARCH_ONLY"},
    }


def _validate_bindings(bindings: object) -> None:
    expected = {
        "parent_embedding",
        "dinov2",
        "old_crop_manifest",
        "native_v4_bundle",
        "support_cache",
        "splits",
        "old_dev_protocol",
        "code_sha256s",
        "config_sha256",
        "license",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected:
        raise ValueError("consistency bindings keys differ")
    parent = bindings["parent_embedding"]
    if not isinstance(parent, dict) or set(parent) != {
        "lineage_file_sha256",
        "lineage_payload_sha256",
        "lineage_sha256",
        "selected_checkpoint_file_sha256",
        "selected_checkpoint_payload_sha256",
        "model_state_sha256",
        "arcface_state_sha256",
        "selected_epoch",
    }:
        raise ValueError("consistency parent binding differs")
    for name, value in parent.items():
        if name == "selected_epoch":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("consistency parent epoch differs")
        else:
            _sha256(value, f"parent_embedding.{name}")
    for key in ("old_crop_manifest", "native_v4_bundle", "support_cache"):
        value = bindings[key]
        if not isinstance(value, dict) or not value:
            raise ValueError(f"consistency {key} binding differs")
        for name, digest in value.items():
            if name.endswith("sha256"):
                _sha256(digest, f"{key}.{name}")
    if not isinstance(bindings["dinov2"], dict) or set(bindings["dinov2"]) != {
        "model_directory_sha256",
        "weight_intake_bundle_file_sha256",
        "preprocessor_intake_bundle_file_sha256",
        "parent_dinov2_binding",
    }:
        raise ValueError("consistency DINO binding differs")
    for name in (
        "model_directory_sha256",
        "weight_intake_bundle_file_sha256",
        "preprocessor_intake_bundle_file_sha256",
    ):
        _sha256(bindings["dinov2"][name], f"dinov2.{name}")
    splits = bindings["splits"]
    if not isinstance(splits, dict) or splits.get("splits_sha256") != content_sha256(
        {key: value for key, value in splits.items() if key != "splits_sha256"}
    ):
        raise ValueError("consistency split binding differs")
    identities = splits["identity_lists"]
    for name, items in identities.items():
        if not isinstance(items, list) or items != sorted(items) or len(items) != len(set(items)):
            raise ValueError(f"consistency split identity list {name} differs")
    if set(identities["ssl_train"]) & set(identities["dev"]) or set(identities["ssl_train"]) & set(identities["eval"]) or set(identities["dev"]) & set(identities["eval"]):
        raise ValueError("consistency split identities overlap")
    for name, tokens in splits["sample_token_lists"].items():
        if not isinstance(tokens, list) or tokens != sorted(tokens) or len(tokens) != len(set(tokens)):
            raise ValueError(f"consistency split sample list {name} differs")
        for token in tokens:
            _sha256(token, f"splits.{name}")
    code = bindings["code_sha256s"]
    code_paths = frozenset(code) if isinstance(code, dict) else frozenset()
    if code_paths not in {
        frozenset(_CODE_PATHS),
        frozenset(_PRE_EMBEDDING_CODE_PATHS),
        frozenset(_LEGACY_CODE_PATHS),
    }:
        raise ValueError("consistency code binding differs")
    for name, digest in code.items():
        if not isinstance(name, str) or not name:
            raise ValueError("consistency code path differs")
        _sha256(digest, f"code_sha256s.{name}")
    _sha256(bindings["config_sha256"], "config_sha256")
    if bindings["license"] != {"license_id": LICENSE_ID, "usage_lane": "RESEARCH_ONLY"}:
        raise ValueError("consistency license binding differs")


def _validate_training_config(config: object) -> None:
    expected = {
        "schema_version",
        "model",
        "input",
        "epochs",
        "old_batch_size",
        "native_pair_batch_size",
        "optimizer",
        "old_pass",
        "native_pass",
        "partition",
        "selection",
        "num_workers",
        "seed",
        "mixed_precision",
        "device",
        "parity_thresholds",
        "usage_lane",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("consistency training config keys differ")
    if config["schema_version"] != CONFIG_SCHEMA or config["model"] != "DINOv2-small CLS L2-normalized 384D" or config["usage_lane"] != "RESEARCH_ONLY":
        raise ValueError("consistency training config contract differs")
    if config["input"] != {
        "shape": [3, IMAGE_SIZE, IMAGE_SIZE],
        "resize": "DIRECT_BICUBIC_STRETCH",
        "scale": 1.0 / 255.0,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }:
        raise ValueError("consistency input config differs")
    if config["native_pass"].get("loss_weights") != LOSS_WEIGHTS or config["native_pass"].get("restoration_rgb_target") is not False:
        raise ValueError("consistency native loss config differs")
    if config["old_pass"].get("samples_per_identity_per_epoch") != 1:
        raise ValueError("consistency old sampling config differs")
    if config["partition"] != {
        "salt": PARTITION_SALT,
        "dev_fraction": DEV_FRACTION,
        "minimum_evaluation_frames": MIN_EVALUATION_FRAMES,
    }:
        raise ValueError("consistency partition config differs")
    for name in ("epochs", "old_batch_size", "native_pair_batch_size"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"consistency {name} differs")
    if config["device"] not in {"cpu", "cuda"} or not isinstance(config["mixed_precision"], bool):
        raise ValueError("consistency device config differs")
    ParityThresholds.from_dict(config["parity_thresholds"])


def _build_lineage(
    *,
    root: Path,
    onnx_bytes: int,
    bindings: Mapping[str, Any],
    training_config: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    parity: ModelParityReceipt,
) -> dict[str, Any]:
    def artifact(relative: str) -> dict[str, Any]:
        path = root.joinpath(*PurePosixPath(relative).parts)
        return {"path": relative, "sha256": _file_sha256(path), "bytes": path.stat().st_size}

    body = {
        "schema_version": LINEAGE_SCHEMA,
        "artifacts": {
            "onnx": artifact("nose_embedding.onnx"),
            "runtime_manifest": artifact("nose_embedding.runtime.json"),
            "parity_receipt": artifact("nose_embedding.parity.json"),
            "selected_checkpoint": artifact("checkpoints/selected.pt"),
            "last_checkpoint": artifact("checkpoints/last.pt"),
            "dev_selection": artifact("dev_selection.json"),
            "evaluation": artifact("evaluation.json"),
        },
        "onnx_contract": {
            "input_name": "images",
            "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
            "output_name": "embedding",
            "output_shape": [1, EMBEDDING_DIM],
            "opset": 18,
            "external_data": False,
            "onnx_bytes": onnx_bytes,
        },
        "bindings": dict(bindings),
        "bindings_sha256": content_sha256(dict(bindings)),
        "training_config": dict(training_config),
        "training_config_sha256": content_sha256(dict(training_config)),
        "selection_payload_sha256": content_sha256(dict(selection_report)),
        "evaluation_payload_sha256": content_sha256(dict(evaluation)),
        "parity_payload_sha256": parity.receipt_sha256,
        "license": {"license_id": LICENSE_ID, "usage_lane": "RESEARCH_ONLY"},
        "interpretation": INTERPRETATION,
    }
    return {**body, "lineage_sha256": content_sha256(body)}


def _validate_arguments(**values: Any) -> None:
    for name in ("epochs", "old_batch_size", "native_pair_batch_size"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("backbone_lr", "head_lr", "weight_decay"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if name != "weight_decay" else value < 0):
            raise ValueError(f"{name} differs")
    if isinstance(values["num_workers"], bool) or not isinstance(values["num_workers"], int) or values["num_workers"] < 0:
        raise ValueError("num_workers differs")
    if isinstance(values["seed"], bool) or not isinstance(values["seed"], int):
        raise ValueError("seed differs")
    if not isinstance(values["mixed_precision"], bool) or values["device_name"] not in {"cpu", "cuda"}:
        raise ValueError("mixed precision or device differs")


def _validate_eval_population(records: Sequence[Mapping[str, Any]], name: str) -> None:
    grouped: dict[str, int] = defaultdict(int)
    for row in records:
        grouped[row["registered_dog_id"]] += 1
    if len(grouped) < 2 or any(count < MIN_EVALUATION_FRAMES for count in grouped.values()):
        raise ValueError(f"{name} requires at least two identities with ten frames")


def _image_tensor(image: np.ndarray) -> torch.Tensor:
    resized = Image.fromarray(image, mode="RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC
    )
    values = np.asarray(resized, dtype=np.uint8).copy()
    tensor = torch.from_numpy(values).permute(2, 0, 1).float().mul_(1.0 / 255.0)
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    return (tensor - mean) / std


def _cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(first.float(), second.float(), dim=-1)


def _finite_loss(loss: torch.Tensor) -> None:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("consistency loss became non-finite")


def _metric(payload: Mapping[str, Any], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"selection metric {name} differs")
    return float(value)


def _normalized_mean(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64).mean(axis=0)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("K5 embedding mean is not normalizable")
    return (result / norm).astype(np.float32)


def _checkpoint_metadata_sha256(payload: Mapping[str, Any]) -> str:
    return content_sha256(
        {
            key: value
            for key, value in payload.items()
            if key not in {"model_state_dict", "arcface_state_dict", "checkpoint_payload_sha256"}
        }
    )


def _state_dict_sha256(state: Mapping[str, Any], name: str) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{name} must be a non-empty state dictionary")
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must contain named tensors only")
        value = tensor.detach().cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite tensors")
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(",".join(str(item) for item in value.shape).encode("ascii") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _external_pin(path: Path, expected: str, name: str) -> None:
    expected = _sha256(expected, f"{name} external pin")
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an absolute regular non-symlink file")
    if _file_sha256(path) != expected:
        raise ValueError(f"{name} file SHA-256 differs from external pin")


def _pinned_document(path: Path, expected: str, name: str):
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an absolute regular non-symlink file")
    document = read_strict_json_document(
        path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    if document.canonical_payload_sha256 != _sha256(expected, f"{name} external pin"):
        raise ValueError(f"{name} content SHA-256 differs from external pin")
    return document


def _validate_model_directory_pin(path: Path, expected: str) -> None:
    root = _absolute_directory(path, "DINO model directory")
    files = sorted(item for item in root.iterdir() if item.is_file() and not item.is_symlink())
    if not files or any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError("DINO model directory entries differ")
    binding = [
        {"name": item.name, "sha256": _file_sha256(item), "bytes": item.stat().st_size}
        for item in files
    ]
    if content_sha256(binding) != _sha256(expected, "DINO model directory external pin"):
        raise ValueError("DINO model directory content SHA-256 differs from external pin")


def model_directory_content_sha256(path: Path) -> str:
    """Return the canonical external pin expected for a flat DINO directory."""

    root = _absolute_directory(path, "DINO model directory")
    files = sorted(item for item in root.iterdir() if item.is_file() and not item.is_symlink())
    if not files or any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError("DINO model directory entries differ")
    return content_sha256(
        [{"name": item.name, "sha256": _file_sha256(item), "bytes": item.stat().st_size} for item in files]
    )


def _bound_crop_path(root: Path, row: Mapping[str, Any]) -> Path:
    relative = PurePosixPath(row["crop_path"])
    if relative.is_absolute() or ".." in relative.parts or row["crop_path"] != relative.as_posix():
        raise ValueError("bound crop path is unsafe")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError("bound crop path is unsafe")
    if _file_sha256(path) != row["crop_sha256"]:
        raise ValueError("bound crop SHA-256 differs")
    return path


def _same_file(candidate: Path, expected: Path, name: str) -> None:
    if not candidate.is_absolute() or candidate.is_symlink() or candidate.resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError(f"{name} path differs from lineage")


def _absolute_directory(path: Path, name: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink directory")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{name} must be a directory")
    return resolved


def _external_new_directory(path: Path, repository: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("output directory must be absolute")
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {absolute}")
    parent = absolute.parent.resolve(strict=True)
    if absolute.is_relative_to(repository):
        raise ValueError("consistency output must be outside the Git repository")
    return parent / absolute.name


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"hash input must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _canonical_uuid5(value: object, name: str) -> str:
    try:
        parsed = uuid.UUID(value)  # type: ignore[arg-type]
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{name} must be a canonical UUIDv5") from exc
    if parsed.version != 5 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUIDv5")
    return str(parsed)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "DEGRADATION_WEIGHTS",
    "DEV_FRACTION",
    "EMBEDDING_DIM",
    "INTERPRETATION",
    "LINEAGE_SCHEMA",
    "LOSS_WEIGHTS",
    "MODEL_ID",
    "NativeConsistencyPairDataset",
    "build_consistency_checkpoint",
    "build_identity_partitions",
    "build_runtime_manifest",
    "deterministic_mild_degradation",
    "evaluate_native_k5",
    "initialize_from_parent",
    "load_consistency_checkpoint",
    "model_directory_content_sha256",
    "native_consistency_loss",
    "old_supervised_loss",
    "produce_parity_receipt",
    "records_for_partition",
    "replace_consistency_checkpoint",
    "select_epoch",
    "select_native_frame_pairs",
    "train_and_export",
    "validate_consistency_checkpoint",
    "validate_lineage_manifest",
]
