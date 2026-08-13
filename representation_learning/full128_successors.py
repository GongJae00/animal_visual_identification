"""Deterministic training and artifacts for the Full128 successor family."""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from foundation.protected_io import json_document_bytes, read_strict_json_document
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from identity_methods.full_segment.artifacts import file_binding
from identity_methods.full_segment.losses import batch_hard_triplet_loss
from identity_methods.full_segment.successor_models import (
    DINOV2_PATCH_DIMENSION,
    Dinov2OccupancyProbe128,
    IdentityBlindResidualTokenAdapter128,
    SpatialScorer128,
    parameter_partition,
    reject_identity_metadata,
)

SUCCESSOR_CONFIG_SCHEMA = "cvi.full128_successor_training_config.v1"
SUCCESSOR_FAMILY_SCHEMA = "cvi.full128_successor_family.v1"
SUCCESSOR_CHECKPOINT_SCHEMA = "cvi.full128_successor_checkpoint.v1"
SUCCESSOR_TOKEN_CACHE_SCHEMA = "cvi.full128_dinov2_token_cache.v1"
SUCCESSOR_SSL_OBJECTIVE = "TWO_VIEW_COSINE_DISTANCE"

SUCCESSOR_CANDIDATES = (
    ("B0-FV", "CLASSICAL128_REUSE", "FIT_SCALER_PCA"),
    ("B1-FV", "MASKED_GAP128_RESNET18", "RANDOM_SCRATCH"),
    ("B2-FV", "MASKED_GAP128_RESNET18", "RECEIPT_BOUND_IMAGENET_FRESH"),
    ("B3-FV", "DINOV2_PATCH_OCCUPANCY_PROBE128", "RECEIPT_BOUND_FROZEN_DINOV2"),
    ("B4-U0-FV", "IDENTITY_BLIND_RESIDUAL_TOKEN_ADAPTER", "ZERO_INIT_NO_UPDATE"),
    ("B4-U1-FV", "IDENTITY_BLIND_RESIDUAL_TOKEN_ADAPTER", "ZERO_INIT_SSL_UPDATE"),
    ("B5-FV", "ZERO_INIT_SPATIAL_SCORER", "OCCUPANCY_AND_LEARNED_LOGITS"),
    ("B5-UNIFORM-FV", "ZERO_INIT_SPATIAL_SCORER", "UNIFORM_OCCUPANCY_CONTROL"),
    ("B5-CHANNEL-GATE-FV", "ZERO_INIT_SPATIAL_SCORER", "CHANNEL_GATE_ONLY_CONTROL"),
)

_CONFIG_FIELDS = {
    "schema_version",
    "seed",
    "supervised_steps",
    "ssl_steps",
    "optimizer",
    "precision",
    "workers",
    "triplet_margin",
    "ssl_objective",
}
_SUPERVISED_RAW_FIELDS = {"rgb", "mask", "label"}
_SUPERVISED_TOKEN_FIELDS = {"tokens", "occupancy", "label"}
_SSL_FIELDS = {
    "view_a_tokens",
    "view_a_occupancy",
    "view_b_tokens",
    "view_b_occupancy",
}


def build_successor_family_manifest() -> dict[str, Any]:
    """Return the fixed candidate family, including required controls."""

    return {
        "schema_version": SUCCESSOR_FAMILY_SCHEMA,
        "family_id": "FULL128_SUCCESSORS_B0_FV_TO_B5_FV",
        "interpretation": "OFFLINE_RESEARCH_CANDIDATES_NOT_BIOMETRIC_VALIDATION",
        "output": {"dimension": 128, "dtype": "float32", "normalization": "L2"},
        "candidates": [
            {
                "candidate_id": candidate_id,
                "method": method,
                "initialization": initialization,
            }
            for candidate_id, method, initialization in SUCCESSOR_CANDIDATES
        ],
        "training_contracts": {
            "supervised": "BATCH_HARD_EUCLIDEAN_TRIPLET_MARGIN_0.2_FIXED_STEPS",
            "identity_blind": (
                "TWO_VIEW_COSINE_DISTANCE_FIXED_STEPS_EQUAL_U0_U1_SCHEDULE"
            ),
            "b2_reuse": "FORBIDDEN_LEARNED_STATE_FRESH_RECEIPT_INITIALIZATION_ONLY",
        },
    }


def default_successor_training_config(*, smoke: bool = False) -> dict[str, Any]:
    """Return deterministic fixed-step defaults or a bounded smoke configuration."""

    return {
        "schema_version": SUCCESSOR_CONFIG_SCHEMA,
        "seed": 20260811,
        "supervised_steps": 1 if smoke else 2_000,
        "ssl_steps": 1 if smoke else 2_000,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
        "precision": {
            "device": "cpu" if smoke else "cuda",
            "amp": not smoke,
            "amp_dtype": "float32" if smoke else "float16",
        },
        "workers": 0 if smoke else 8,
        "triplet_margin": 0.2,
        "ssl_objective": SUCCESSOR_SSL_OBJECTIVE,
    }


def validate_successor_training_config(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CONFIG_FIELDS:
        raise ValueError("Full128 successor config fields differ")
    config = dict(value)
    if config["schema_version"] != SUCCESSOR_CONFIG_SCHEMA:
        raise ValueError("Full128 successor config schema differs")
    for name in ("seed", "supervised_steps", "ssl_steps", "workers"):
        item = config[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"Full128 successor {name} must be an integer")
    if (
        config["seed"] < 0
        or config["supervised_steps"] <= 0
        or config["ssl_steps"] <= 0
        or config["workers"] < 0
    ):
        raise ValueError("Full128 successor config integer range differs")
    optimizer = config["optimizer"]
    if not isinstance(optimizer, Mapping) or set(optimizer) != {
        "name",
        "learning_rate",
        "weight_decay",
    }:
        raise ValueError("Full128 successor optimizer fields differ")
    if optimizer["name"] != "AdamW":
        raise ValueError("Full128 successor optimizer must be AdamW")
    for name in ("learning_rate", "weight_decay"):
        item = optimizer[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not np.isfinite(item)
        ):
            raise ValueError(f"Full128 successor optimizer {name} must be finite")
    if optimizer["learning_rate"] <= 0 or optimizer["weight_decay"] < 0:
        raise ValueError("Full128 successor optimizer value range differs")
    precision = config["precision"]
    if not isinstance(precision, Mapping) or set(precision) != {
        "device",
        "amp",
        "amp_dtype",
    }:
        raise ValueError("Full128 successor precision fields differ")
    if precision["device"] not in {"cpu", "cuda"} or not isinstance(
        precision["amp"], bool
    ):
        raise ValueError("Full128 successor precision values differ")
    expected_dtype = "float16" if precision["amp"] else "float32"
    if precision["amp_dtype"] != expected_dtype or (
        precision["device"] == "cpu" and precision["amp"]
    ):
        raise ValueError("Full128 successor AMP contract differs")
    if config["triplet_margin"] != 0.2:
        raise ValueError("Full128 successor triplet margin is fixed at 0.2")
    if config["ssl_objective"] != SUCCESSOR_SSL_OBJECTIVE:
        raise ValueError("Full128 successor SSL objective differs")
    return config


def reset_successor_seed(seed: int, *, use_cuda: bool = False) -> None:
    """Reset all admitted RNGs and deterministic backend controls."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_supervised_fixed_steps(
    model: nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train B1/B2/B3/B5 for an exact number of batch-hard triplet steps."""

    validated = validate_successor_training_config(config)
    if not batches:
        raise ValueError("supervised successor training requires batches")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("supervised successor stage has no trainable parameters")
    device = torch.device(validated["precision"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Full128 successor CUDA was requested but is unavailable")
    model.to(device).train()
    optimizer = torch.optim.AdamW(parameters, **_optimizer_kwargs(validated))
    amp_enabled = bool(validated["precision"]["amp"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    losses: list[float] = []
    for step in range(validated["supervised_steps"]):
        batch = batches[step % len(batches)]
        fields = set(batch)
        token_batch = False
        if fields == _SUPERVISED_RAW_FIELDS:
            args = (
                batch["rgb"].to(device=device, dtype=torch.float32),
                batch["mask"].to(device=device, dtype=torch.float32),
            )
        elif fields == _SUPERVISED_TOKEN_FIELDS and hasattr(model, "forward_from_tokens"):
            token_batch = True
            args = (
                batch["tokens"].to(device=device, dtype=torch.float32),
                batch["occupancy"].to(device=device, dtype=torch.float32),
            )
        else:
            raise ValueError("supervised successor batch fields differ")
        labels = batch["label"].to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            embeddings = (
                model.forward_from_tokens(*args) if token_batch else model(*args)
            )
            loss = batch_hard_triplet_loss(embeddings, labels, margin=0.2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().float().cpu()))
    return {
        "kind": "BATCH_HARD_EUCLIDEAN_TRIPLET_FIXED_STEPS",
        "margin": 0.2,
        "attempted_steps": validated["supervised_steps"],
        "update_steps": validated["supervised_steps"],
        "mean_loss": sum(losses) / len(losses),
        "batch_cycle_length": len(batches),
    }


def train_identity_blind_fixed_steps(
    model: IdentityBlindResidualTokenAdapter128,
    batches: Sequence[Mapping[str, torch.Tensor]],
    config: Mapping[str, Any],
    *,
    update_enabled: bool,
) -> dict[str, Any]:
    """Execute U0/U1 with equal fixed-step schedules and no identity metadata."""

    if not isinstance(update_enabled, bool):
        raise TypeError("B4 update_enabled must be boolean")
    validated = validate_successor_training_config(config)
    if not batches:
        raise ValueError("B4 identity-blind training requires batches")
    for batch in batches:
        reject_identity_metadata(batch)
        if set(batch) != _SSL_FIELDS:
            raise ValueError("B4 identity-blind batch fields differ")
    del batch
    device = torch.device(validated["precision"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Full128 successor CUDA was requested but is unavailable")
    model.to(device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if {name for name, parameter in model.named_parameters() if parameter.requires_grad} != {
        name for name, _ in model.adapter.named_parameters(prefix="adapter")
    }:
        raise RuntimeError("B4 trainable boundary must contain only adapter parameters")
    optimizer = torch.optim.AdamW(parameters, **_optimizer_kwargs(validated))
    losses: list[float] = []
    for step in range(validated["ssl_steps"]):
        batch = batches[step % len(batches)]
        values = {
            key: tensor.to(device=device, dtype=torch.float32)
            for key, tensor in batch.items()
        }
        first = model.forward_from_tokens(
            values["view_a_tokens"], values["view_a_occupancy"]
        )
        second = model.forward_from_tokens(
            values["view_b_tokens"], values["view_b_occupancy"]
        )
        loss = (1.0 - F.cosine_similarity(first, second, dim=1)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("B4 SSL objective produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        if update_enabled:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "kind": "IDENTITY_BLIND_FIXED_STEP_SSL",
        "objective": SUCCESSOR_SSL_OBJECTIVE,
        "attempted_steps": validated["ssl_steps"],
        "update_steps": validated["ssl_steps"] if update_enabled else 0,
        "mean_loss": sum(losses) / len(losses),
        "batch_cycle_length": len(batches),
        "identity_metadata": "REJECTED",
    }


def run_dinov2_successor_stages(
    *,
    backbone: nn.Module,
    supervised_batches: Sequence[Mapping[str, torch.Tensor]],
    identity_blind_batches: Sequence[Mapping[str, torch.Tensor]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Orchestrate B3, equal-schedule B4 U0/U1, and all B5 controls.

    Batches contain cached DINOv2 patch tokens, keeping the frozen foundation
    pass separate and reusable by every successor stage.
    """

    validated = validate_successor_training_config(config)
    reset_successor_seed(
        validated["seed"], use_cuda=validated["precision"]["device"] == "cuda"
    )
    b3 = Dinov2OccupancyProbe128(backbone)
    receipts: dict[str, dict[str, Any]] = {
        "B3-FV": train_supervised_fixed_steps(b3, supervised_batches, validated)
    }

    frozen_projection = deepcopy(b3.projection).cpu()
    b4_u0 = IdentityBlindResidualTokenAdapter128(
        backbone, deepcopy(frozen_projection)
    )
    b4_u1 = IdentityBlindResidualTokenAdapter128(
        backbone, deepcopy(frozen_projection)
    )
    b4_u1.adapter.load_state_dict(b4_u0.adapter.state_dict(), strict=True)
    receipts["B4-U0-FV"] = train_identity_blind_fixed_steps(
        b4_u0, identity_blind_batches, validated, update_enabled=False
    )
    receipts["B4-U1-FV"] = train_identity_blind_fixed_steps(
        b4_u1, identity_blind_batches, validated, update_enabled=True
    )

    b5 = SpatialScorer128(backbone, deepcopy(frozen_projection))
    b5_uniform = SpatialScorer128(
        backbone, deepcopy(frozen_projection), uniform_spatial=True
    )
    b5_channel = SpatialScorer128(
        backbone,
        deepcopy(frozen_projection),
        uniform_spatial=True,
        channel_gate=True,
    )
    receipts["B5-FV"] = train_supervised_fixed_steps(
        b5, supervised_batches, validated
    )
    receipts["B5-UNIFORM-FV"] = train_supervised_no_update_fixed_steps(
        b5_uniform, supervised_batches, validated
    )
    receipts["B5-CHANNEL-GATE-FV"] = train_supervised_fixed_steps(
        b5_channel, supervised_batches, validated
    )
    return {
        "models": {
            "B3-FV": b3,
            "B4-U0-FV": b4_u0,
            "B4-U1-FV": b4_u1,
            "B5-FV": b5,
            "B5-UNIFORM-FV": b5_uniform,
            "B5-CHANNEL-GATE-FV": b5_channel,
        },
        "training_receipts": receipts,
    }


def train_supervised_no_update_fixed_steps(
    model: SpatialScorer128,
    batches: Sequence[Mapping[str, torch.Tensor]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not batches:
        raise ValueError("supervised no-update control requires batches")
    device = torch.device(config["precision"]["device"])
    model.to(device).eval()
    losses: list[float] = []
    with torch.no_grad():
        for step in range(config["supervised_steps"]):
            batch = batches[step % len(batches)]
            if set(batch) != _SUPERVISED_TOKEN_FIELDS:
                raise ValueError("supervised no-update control batch fields differ")
            embeddings = model.forward_from_tokens(
                batch["tokens"].to(device=device, dtype=torch.float32),
                batch["occupancy"].to(device=device, dtype=torch.float32),
            )
            assert isinstance(embeddings, torch.Tensor)
            loss = batch_hard_triplet_loss(
                embeddings,
                batch["label"].to(device=device, dtype=torch.long),
                margin=0.2,
            )
            losses.append(float(loss.cpu()))
    return {
        "kind": "BATCH_HARD_EUCLIDEAN_TRIPLET_FIXED_STEPS_NO_UPDATE_CONTROL",
        "margin": 0.2,
        "attempted_steps": config["supervised_steps"],
        "update_steps": 0,
        "mean_loss": sum(losses) / len(losses),
        "batch_cycle_length": len(batches),
    }


def write_successor_checkpoint(
    directory: Path,
    *,
    candidate_id: str,
    model: nn.Module,
    config: Mapping[str, Any],
    bindings: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist only trainable successor state with exact frozen bindings."""

    _require_candidate(candidate_id)
    validated = validate_successor_training_config(config)
    target = _new_external_directory(directory, subject="successor checkpoint")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    os.chmod(staging, 0o700)
    try:
        partition = parameter_partition(model)
        trainable_names = set(partition["trainable"])
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
            if name in trainable_names
        }
        if set(state) != trainable_names or any(
            not torch.isfinite(value).all() for value in state.values()
        ):
            raise RuntimeError("successor trainable checkpoint state differs")
        state_path = staging / "trainable-state.safetensors"
        save_file(
            state,
            str(state_path),
            metadata={
                "schema_version": SUCCESSOR_CHECKPOINT_SCHEMA,
                "candidate_id": candidate_id,
                "config_sha256": content_sha256(validated),
            },
        )
        payload = {
            "schema_version": SUCCESSOR_CHECKPOINT_SCHEMA,
            "candidate_id": candidate_id,
            "state": {"relative_path": state_path.name, **file_binding(state_path)},
            "parameter_partition": {
                "trainable": list(partition["trainable"]),
                "frozen": list(partition["frozen"]),
            },
            "config_sha256": content_sha256(validated),
            "bindings": dict(bindings),
            "bindings_sha256": content_sha256(dict(bindings)),
            "training_receipt": dict(training_receipt),
            "training_receipt_sha256": content_sha256(dict(training_receipt)),
        }
        manifest = {**payload, "checkpoint_manifest_sha256": content_sha256(payload)}
        (staging / "checkpoint-manifest.json").write_bytes(json_document_bytes(manifest))
        fsync_directory(staging)
        rename_directory_noreplace(staging, target)
        fsync_directory(target.parent)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_successor_checkpoint(
    directory: Path,
    *,
    candidate_id: str,
    model: nn.Module,
    config: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a checkpoint and restore its exact trainable-only state."""

    root = directory.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("successor checkpoint must be a regular directory")
    manifest = read_strict_json_document(
        root / "checkpoint-manifest.json", maximum_bytes=16_777_216
    ).payload
    expected_fields = {
        "schema_version",
        "candidate_id",
        "state",
        "parameter_partition",
        "config_sha256",
        "bindings",
        "bindings_sha256",
        "training_receipt",
        "training_receipt_sha256",
        "checkpoint_manifest_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise ValueError("successor checkpoint manifest fields differ")
    payload = {
        key: value for key, value in manifest.items() if key != "checkpoint_manifest_sha256"
    }
    partition = parameter_partition(model)
    if (
        manifest["schema_version"] != SUCCESSOR_CHECKPOINT_SCHEMA
        or manifest["candidate_id"] != candidate_id
        or manifest["checkpoint_manifest_sha256"] != content_sha256(payload)
        or manifest["config_sha256"]
        != content_sha256(validate_successor_training_config(config))
        or manifest["bindings"] != dict(bindings)
        or manifest["bindings_sha256"] != content_sha256(dict(bindings))
        or manifest["training_receipt_sha256"]
        != content_sha256(manifest["training_receipt"])
        or manifest["parameter_partition"]
        != {
            "trainable": list(partition["trainable"]),
            "frozen": list(partition["frozen"]),
        }
    ):
        raise ValueError("successor checkpoint immutable binding differs")
    state_binding = manifest["state"]
    if not isinstance(state_binding, Mapping) or set(state_binding) != {
        "relative_path",
        "sha256",
        "byte_size",
    }:
        raise ValueError("successor checkpoint state binding differs")
    state_path = root / state_binding["relative_path"]
    if file_binding(state_path) != {
        "sha256": state_binding["sha256"],
        "byte_size": state_binding["byte_size"],
    }:
        raise ValueError("successor checkpoint state digest differs")
    state = load_file(str(state_path), device="cpu")
    if set(state) != set(partition["trainable"]) or any(
        not torch.isfinite(value).all() for value in state.values()
    ):
        raise ValueError("successor checkpoint trainable state differs")
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)
    return dict(manifest)


def write_dinov2_token_cache(
    directory: Path,
    *,
    sample_ids: Sequence[str],
    tokens: np.ndarray,
    occupancy: np.ndarray,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Write deterministic packed float32 DINO patch tokens and occupancies."""

    ids = tuple(sample_ids)
    token_values = np.asarray(tokens, dtype="<f4")
    occupancy_values = np.asarray(occupancy, dtype="<f4")
    if (
        not ids
        or ids != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or any(not _is_sha256(value) for value in ids)
    ):
        raise ValueError("successor token cache sample IDs must be sorted unique SHA-256")
    if (
        token_values.ndim != 3
        or token_values.shape[0] != len(ids)
        or token_values.shape[2] != DINOV2_PATCH_DIMENSION
        or occupancy_values.shape != token_values.shape[:2]
        or not np.isfinite(token_values).all()
        or not np.isfinite(occupancy_values).all()
        or np.any((occupancy_values < 0) | (occupancy_values > 1))
        or np.any(occupancy_values.sum(axis=1) <= 0)
    ):
        raise ValueError("successor token cache arrays differ")
    target = _new_external_directory(directory, subject="successor token cache")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    os.chmod(staging, 0o700)
    try:
        token_path = staging / "tokens.f32le"
        occupancy_path = staging / "occupancy.f32le"
        token_path.write_bytes(np.ascontiguousarray(token_values).tobytes())
        occupancy_path.write_bytes(np.ascontiguousarray(occupancy_values).tobytes())
        payload = {
            "schema_version": SUCCESSOR_TOKEN_CACHE_SCHEMA,
            "dtype": "float32_little_endian",
            "sample_ids": list(ids),
            "sample_count": len(ids),
            "token_count": token_values.shape[1],
            "token_dimension": DINOV2_PATCH_DIMENSION,
            "tokens": {"relative_path": token_path.name, **file_binding(token_path)},
            "occupancy": {
                "relative_path": occupancy_path.name,
                **file_binding(occupancy_path),
            },
            "bindings": dict(bindings),
            "bindings_sha256": content_sha256(dict(bindings)),
        }
        manifest = {**payload, "cache_manifest_sha256": content_sha256(payload)}
        (staging / "cache-manifest.json").write_bytes(json_document_bytes(manifest))
        fsync_directory(staging)
        rename_directory_noreplace(staging, target)
        fsync_directory(target.parent)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_dinov2_token_cache(
    directory: Path, *, bindings: Mapping[str, Any]
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, Any]]:
    root = directory.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("successor token cache must be a regular directory")
    manifest = read_strict_json_document(
        root / "cache-manifest.json", maximum_bytes=268_435_456
    ).payload
    fields = {
        "schema_version",
        "dtype",
        "sample_ids",
        "sample_count",
        "token_count",
        "token_dimension",
        "tokens",
        "occupancy",
        "bindings",
        "bindings_sha256",
        "cache_manifest_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != fields:
        raise ValueError("successor token cache manifest fields differ")
    payload = {
        key: value for key, value in manifest.items() if key != "cache_manifest_sha256"
    }
    ids = tuple(manifest["sample_ids"])
    if (
        manifest["schema_version"] != SUCCESSOR_TOKEN_CACHE_SCHEMA
        or manifest["dtype"] != "float32_little_endian"
        or manifest["token_dimension"] != DINOV2_PATCH_DIMENSION
        or manifest["sample_count"] != len(ids)
        or ids != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or any(not _is_sha256(value) for value in ids)
        or manifest["bindings"] != dict(bindings)
        or manifest["bindings_sha256"] != content_sha256(dict(bindings))
        or manifest["cache_manifest_sha256"] != content_sha256(payload)
    ):
        raise ValueError("successor token cache immutable binding differs")
    arrays: list[np.ndarray] = []
    shapes = (
        (len(ids), manifest["token_count"], DINOV2_PATCH_DIMENSION),
        (len(ids), manifest["token_count"]),
    )
    for name, shape in zip(("tokens", "occupancy"), shapes, strict=True):
        binding = manifest[name]
        if not isinstance(binding, Mapping) or set(binding) != {
            "relative_path",
            "sha256",
            "byte_size",
        }:
            raise ValueError("successor token cache file binding differs")
        path = root / binding["relative_path"]
        if file_binding(path) != {
            "sha256": binding["sha256"],
            "byte_size": binding["byte_size"],
        }:
            raise ValueError("successor token cache file digest differs")
        expected_bytes = int(np.prod(shape)) * 4
        if binding["byte_size"] != expected_bytes:
            raise ValueError("successor token cache byte size differs")
        values = np.fromfile(path, dtype="<f4").reshape(shape).copy()
        arrays.append(values)
    tokens, occupancy = arrays
    if (
        not np.isfinite(tokens).all()
        or not np.isfinite(occupancy).all()
        or np.any((occupancy < 0) | (occupancy > 1))
        or np.any(occupancy.sum(axis=1) <= 0)
    ):
        raise ValueError("successor token cache values differ")
    return ids, tokens, occupancy, dict(manifest)


def make_identity_blind_views(
    tokens: torch.Tensor, occupancy: torch.Tensor, *, phase: int
) -> dict[str, torch.Tensor]:
    """Create deterministic complementary patch views without identity fields."""

    if isinstance(phase, bool) or not isinstance(phase, int) or phase < 0:
        raise ValueError("identity-blind view phase must be a non-negative integer")
    if tokens.ndim != 3 or occupancy.shape != tokens.shape[:2]:
        raise ValueError("identity-blind source token shapes differ")
    indices = torch.arange(tokens.shape[1], device=occupancy.device)
    first_keep = ((indices + phase) % 2 == 0).to(occupancy.dtype)
    second_keep = 1.0 - first_keep
    first_occupancy = occupancy * first_keep
    second_occupancy = occupancy * second_keep
    if torch.any(first_occupancy.sum(dim=1) <= 0) or torch.any(
        second_occupancy.sum(dim=1) <= 0
    ):
        raise ValueError("identity-blind complementary views require occupied patches")
    return {
        "view_a_tokens": tokens,
        "view_a_occupancy": first_occupancy,
        "view_b_tokens": tokens,
        "view_b_occupancy": second_occupancy,
    }


def smoke_successor_execution(output_dir: Path) -> dict[str, Any]:
    """Run bounded synthetic model/training/cache/checkpoint contract execution."""

    config = default_successor_training_config(smoke=True)
    reset_successor_seed(config["seed"])

    class _SmokeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(
            self, *, pixel_values: torch.Tensor, interpolate_pos_encoding: bool
        ) -> Any:
            if not interpolate_pos_encoding:
                raise AssertionError("smoke DINO must interpolate position encoding")
            pooled = F.avg_pool2d(pixel_values, kernel_size=14, stride=14)
            base = pooled.flatten(2).transpose(1, 2).mean(dim=2, keepdim=True)
            patches = base.expand(-1, -1, DINOV2_PATCH_DIMENSION) * self.scale
            cls = torch.zeros(
                len(pixel_values), 1, DINOV2_PATCH_DIMENSION, device=pixel_values.device
            )
            return type("SmokeOutput", (), {"last_hidden_state": torch.cat((cls, patches), dim=1)})()

    root = _new_external_directory(output_dir, subject="successor smoke run")
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
    os.chmod(staging, 0o700)
    try:
        rgb = torch.rand(4, 3, 28, 28)
        mask = torch.ones(4, 1, 28, 28)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        b3 = Dinov2OccupancyProbe128(_SmokeBackbone())
        tokens, occupancy = b3.extract_tokens(rgb, mask)
        supervised = train_supervised_fixed_steps(
            b3,
            ({"tokens": tokens, "occupancy": occupancy, "label": labels},),
            config,
        )
        projection = nn.Linear(384, 128)
        projection.load_state_dict(b3.projection.state_dict())
        b4 = IdentityBlindResidualTokenAdapter128(_SmokeBackbone(), projection)
        ssl_batch = make_identity_blind_views(tokens, occupancy, phase=0)
        ssl = train_identity_blind_fixed_steps(
            b4, (ssl_batch,), config, update_enabled=False
        )
        indexed_ids = sorted(
            (hashlib.sha256(f"smoke-{index}".encode()).hexdigest(), index)
            for index in range(4)
        )
        sample_ids = tuple(sample_id for sample_id, _ in indexed_ids)
        order = [index for _, index in indexed_ids]
        smoke_tokens = tokens.detach().cpu().numpy()[order]
        smoke_occupancy = occupancy.detach().cpu().numpy()[order]
        bindings = {"family_manifest_sha256": content_sha256(build_successor_family_manifest())}
        cache = write_dinov2_token_cache(
            staging / "token-cache",
            sample_ids=sample_ids,
            tokens=smoke_tokens,
            occupancy=smoke_occupancy,
            bindings=bindings,
        )
        checkpoint = write_successor_checkpoint(
            staging / "b3-checkpoint",
            candidate_id="B3-FV",
            model=b3,
            config=config,
            bindings=bindings,
            training_receipt=supervised,
        )
        family = build_successor_family_manifest()
        receipt_payload = {
            "schema_version": "cvi.full128_successor_smoke.v1",
            "config": config,
            "config_sha256": content_sha256(config),
            "family_manifest": family,
            "family_manifest_sha256": content_sha256(family),
            "supervised": supervised,
            "identity_blind_u0": ssl,
            "token_cache_manifest_sha256": cache["cache_manifest_sha256"],
            "checkpoint_manifest_sha256": checkpoint["checkpoint_manifest_sha256"],
            "interpretation": "SYNTHETIC_EXECUTION_ONLY_NOT_MODEL_VALIDATION",
        }
        receipt = {**receipt_payload, "smoke_receipt_sha256": content_sha256(receipt_payload)}
        (staging / "smoke-receipt.json").write_bytes(json_document_bytes(receipt))
        fsync_directory(staging)
        rename_directory_noreplace(staging, root)
        fsync_directory(root.parent)
        return receipt
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _optimizer_kwargs(config: Mapping[str, Any]) -> dict[str, float]:
    return {
        "lr": float(config["optimizer"]["learning_rate"]),
        "weight_decay": float(config["optimizer"]["weight_decay"]),
    }


def _new_external_directory(path: Path, *, subject: str) -> Path:
    requested = path.absolute()
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    repository = Path(__file__).resolve().parents[1]
    if target == repository or target.is_relative_to(repository):
        raise ValueError(f"{subject} must remain outside the repository")
    if requested.is_symlink() or target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite {subject}: {target}")
    return target


def _require_candidate(candidate_id: str) -> None:
    if candidate_id not in {item[0] for item in SUCCESSOR_CANDIDATES}:
        raise ValueError("unsupported Full128 successor candidate")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "SUCCESSOR_CANDIDATES",
    "SUCCESSOR_CHECKPOINT_SCHEMA",
    "SUCCESSOR_CONFIG_SCHEMA",
    "SUCCESSOR_FAMILY_SCHEMA",
    "SUCCESSOR_SSL_OBJECTIVE",
    "SUCCESSOR_TOKEN_CACHE_SCHEMA",
    "build_successor_family_manifest",
    "default_successor_training_config",
    "load_dinov2_token_cache",
    "load_successor_checkpoint",
    "make_identity_blind_views",
    "reset_successor_seed",
    "run_dinov2_successor_stages",
    "smoke_successor_execution",
    "train_identity_blind_fixed_steps",
    "train_supervised_fixed_steps",
    "train_supervised_no_update_fixed_steps",
    "validate_successor_training_config",
    "write_dinov2_token_cache",
    "write_successor_checkpoint",
]
