"""Strict, receipt-bound NoseID-v1 training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any
import uuid

import numpy as np
import torch


SCHEMA_VERSION = "identification.nose.training_checkpoint.v1"
_CHECKPOINT_KEYS = {
    "schema_version",
    "epoch",
    "global_step",
    "best_metric",
    "identity_to_index",
    "noseid_config",
    "train_config",
    "dino_contract",
    "model_state_dict",
    "objective_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "rng_state",
}
_DINO_RECEIPT_KEYS = {
    "model_sha256",
    "preprocessor_sha256",
    "weight_receipt_sha256",
    "preprocessor_receipt_sha256",
}
_RNG_KEYS = {"python", "numpy", "torch_cpu", "torch_cuda"}
_NUMPY_RNG_KEYS = {
    "bit_generator",
    "keys",
    "position",
    "has_gauss",
    "cached_gaussian",
}


def _require_nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _require_identity_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError("identity_to_index must be a non-empty object")
    result: dict[str, int] = {}
    for identity, index in value.items():
        if not isinstance(identity, str):
            raise ValueError("identity_to_index keys must be canonical UUIDv5 strings")
        try:
            parsed = uuid.UUID(identity)
        except ValueError as exc:
            raise ValueError(
                "identity_to_index keys must be canonical UUIDv5 strings"
            ) from exc
        if parsed.version != 5 or str(parsed) != identity:
            raise ValueError("identity_to_index keys must be canonical UUIDv5 strings")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("identity indices must be integers")
        result[identity] = index
    if sorted(result.values()) != list(range(len(result))):
        raise ValueError("identity indices must be unique and contiguous from zero")
    return dict(sorted(result.items()))


def _config_value(value: object, name: str) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [_config_value(item, name) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError(f"{name} keys must be non-empty strings")
        return {
            key: _config_value(item, name)
            for key, item in sorted(value.items())
        }
    raise TypeError(f"{name} must contain only weights-only-safe config values")


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and torch RNGs without NumPy-only objects."""

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": [int(value) for value in numpy_state[1]],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _validate_rng_state(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _RNG_KEYS:
        raise ValueError("checkpoint RNG keys differ")
    python_state = value["python"]
    try:
        random.Random().setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint Python RNG state is invalid") from exc

    numpy_state = value["numpy"]
    if not isinstance(numpy_state, dict) or set(numpy_state) != _NUMPY_RNG_KEYS:
        raise ValueError("checkpoint NumPy RNG keys differ")
    if numpy_state["bit_generator"] != "MT19937":
        raise ValueError("checkpoint NumPy RNG bit generator differs")
    keys = numpy_state["keys"]
    if (
        not isinstance(keys, list)
        or len(keys) != 624
        or any(isinstance(item, bool) or not isinstance(item, int) for item in keys)
    ):
        raise ValueError("checkpoint NumPy RNG keys are invalid")
    try:
        candidate_numpy_state = (
            numpy_state["bit_generator"],
            np.asarray(keys, dtype=np.uint32),
            _require_nonnegative_integer(numpy_state["position"], "NumPy RNG position"),
            _require_nonnegative_integer(numpy_state["has_gauss"], "NumPy RNG has_gauss"),
            float(numpy_state["cached_gaussian"]),
        )
        np.random.RandomState().set_state(candidate_numpy_state)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("checkpoint NumPy RNG state is invalid") from exc
    if numpy_state["has_gauss"] not in (0, 1) or not math.isfinite(
        numpy_state["cached_gaussian"]
    ):
        raise ValueError("checkpoint NumPy Gaussian RNG state is invalid")

    torch_cpu = value["torch_cpu"]
    if (
        not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.device.type != "cpu"
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
    ):
        raise ValueError("checkpoint torch CPU RNG state is invalid")
    try:
        torch.Generator(device="cpu").set_state(torch_cpu)
    except RuntimeError as exc:
        raise ValueError("checkpoint torch CPU RNG state is invalid") from exc
    torch_cuda = value["torch_cuda"]
    if not isinstance(torch_cuda, list) or any(
        not isinstance(state, torch.Tensor)
        or state.device.type != "cpu"
        or state.dtype != torch.uint8
        or state.ndim != 1
        for state in torch_cuda
    ):
        raise ValueError("checkpoint torch CUDA RNG state is invalid")
    return value


def restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    """Restore all captured RNGs, failing if CUDA state cannot be reproduced."""

    state = _validate_rng_state(dict(rng_state))
    numpy_state = state["numpy"]
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint has CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG device count differs")
    random.setstate(state["python"])
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def _state_dict(component: object, name: str) -> dict[str, Any]:
    method = getattr(component, "state_dict", None)
    if not callable(method):
        raise TypeError(f"{name} must provide state_dict()")
    state = method()
    if not isinstance(state, Mapping):
        raise TypeError(f"{name}.state_dict() must return a mapping")
    return dict(state)


def build_training_checkpoint(
    *,
    model: object,
    objective: object,
    optimizer: object,
    scheduler: object | None,
    scaler: object | None,
    identity_to_index: Mapping[str, int],
    noseid_config: object,
    train_config: object,
    best_dev_n3_map: float,
    dino_contract: Mapping[str, str],
    epoch: int,
    global_step: int,
) -> dict[str, Any]:
    """Build the exact weights-only-compatible NoseID checkpoint payload."""

    if isinstance(best_dev_n3_map, bool) or not isinstance(
        best_dev_n3_map, (int, float)
    ):
        raise ValueError("best DEV_N3_mAP must be finite and in [0, 1]")
    metric = float(best_dev_n3_map)
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise ValueError("best DEV_N3_mAP must be finite and in [0, 1]")
    normalized_noseid_config = _config_value(noseid_config, "noseid_config")
    normalized_train_config = _config_value(train_config, "train_config")
    if not isinstance(normalized_noseid_config, dict) or not normalized_noseid_config:
        raise ValueError("noseid_config must be a non-empty object")
    if not isinstance(normalized_train_config, dict) or not normalized_train_config:
        raise ValueError("train_config must be a non-empty object")
    hashes = dict(dino_contract)
    if set(hashes) != _DINO_RECEIPT_KEYS:
        raise ValueError("DINO receipt hash keys differ")
    hashes = {
        key: _require_sha256(value, key) for key, value in sorted(hashes.items())
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "epoch": _require_nonnegative_integer(epoch, "epoch"),
        "global_step": _require_nonnegative_integer(global_step, "global_step"),
        "best_metric": {"name": "DEV_N3_mAP", "value": metric},
        "identity_to_index": _require_identity_mapping(dict(identity_to_index)),
        "noseid_config": normalized_noseid_config,
        "train_config": normalized_train_config,
        "dino_contract": hashes,
        "model_state_dict": _state_dict(model, "model"),
        "objective_state_dict": _state_dict(objective, "objective"),
        "optimizer_state_dict": _state_dict(optimizer, "optimizer"),
        "scheduler_state_dict": (
            None if scheduler is None else _state_dict(scheduler, "scheduler")
        ),
        "scaler_state_dict": None if scaler is None else _state_dict(scaler, "scaler"),
        "rng_state": capture_rng_state(),
    }
    validate_training_checkpoint(payload)
    return payload


def validate_training_checkpoint(payload: object) -> None:
    """Validate exact checkpoint structure without mutating runtime state."""

    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        raise ValueError("NoseID training checkpoint keys differ")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported NoseID training checkpoint schema")
    _require_nonnegative_integer(payload["epoch"], "epoch")
    _require_nonnegative_integer(payload["global_step"], "global_step")
    best_metric = payload["best_metric"]
    if not isinstance(best_metric, dict) or set(best_metric) != {"name", "value"}:
        raise ValueError("best metric keys differ")
    if best_metric["name"] != "DEV_N3_mAP":
        raise ValueError("best metric name differs")
    metric = best_metric["value"]
    if (
        isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(metric)
        or not 0.0 <= metric <= 1.0
    ):
        raise ValueError("best DEV_N3_mAP must be finite and in [0, 1]")
    _require_identity_mapping(payload["identity_to_index"])
    for name in ("noseid_config", "train_config"):
        config = _config_value(payload[name], name)
        if not isinstance(config, dict) or not config or config != payload[name]:
            raise ValueError(f"checkpoint {name} is not canonical")
    hashes = payload["dino_contract"]
    if not isinstance(hashes, dict) or set(hashes) != _DINO_RECEIPT_KEYS:
        raise ValueError("DINO receipt hash keys differ")
    for key, value in hashes.items():
        _require_sha256(value, key)
    for name in (
        "model_state_dict",
        "objective_state_dict",
        "optimizer_state_dict",
    ):
        if not isinstance(payload[name], dict):
            raise ValueError(f"{name} must be an object")
    for name in ("scheduler_state_dict", "scaler_state_dict"):
        if payload[name] is not None and not isinstance(payload[name], dict):
            raise ValueError(f"{name} must be an object or null")
    _validate_rng_state(payload["rng_state"])


def save_training_checkpoint(
    path: Path,
    *,
    model: object,
    objective: object,
    optimizer: object,
    scheduler: object | None,
    scaler: object | None,
    identity_to_index: Mapping[str, int],
    noseid_config: object,
    train_config: object,
    best_dev_n3_map: float,
    dino_contract: Mapping[str, str],
    epoch: int,
    global_step: int,
) -> None:
    """Atomically publish one checkpoint, refusing to replace any path."""

    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"checkpoint parent directory does not exist: {parent}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    payload = build_training_checkpoint(
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        identity_to_index=identity_to_index,
        noseid_config=noseid_config,
        train_config=train_config,
        best_dev_n3_map=best_dev_n3_map,
        dino_contract=dino_contract,
        epoch=epoch,
        global_step=global_step,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        validate_training_checkpoint(
            torch.load(temporary, map_location="cpu", weights_only=True)
        )
        os.link(temporary, target)
        temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_training_checkpoint(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load and strictly validate one weights-only NoseID checkpoint."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    payload = torch.load(source, map_location=map_location, weights_only=True)
    validate_training_checkpoint(payload)
    return payload


def replace_training_checkpoint(
    path: Path,
    *,
    model: object,
    objective: object,
    optimizer: object,
    scheduler: object | None,
    scaler: object | None,
    identity_to_index: Mapping[str, int],
    noseid_config: object,
    train_config: object,
    best_dev_n3_map: float,
    dino_contract: Mapping[str, str],
    epoch: int,
    global_step: int,
) -> None:
    """Atomically replace the mutable best/last checkpoint aliases."""

    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"checkpoint parent directory does not exist: {parent}")
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("checkpoint target must be a regular file or absent")
    payload = build_training_checkpoint(
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        identity_to_index=identity_to_index,
        noseid_config=noseid_config,
        train_config=train_config,
        best_dev_n3_map=best_dev_n3_map,
        dino_contract=dino_contract,
        epoch=epoch,
        global_step=global_step,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        validate_training_checkpoint(
            torch.load(temporary, map_location="cpu", weights_only=True)
        )
        os.replace(temporary, target)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def restore_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    model: object,
    objective: object,
    optimizer: object,
    scheduler: object | None,
    scaler: object | None,
    restore_rng: bool = True,
) -> None:
    """Restore component state and, by default, the exact captured RNG state."""

    checkpoint = dict(payload)
    validate_training_checkpoint(checkpoint)
    for component, state_name, name in (
        (scheduler, "scheduler_state_dict", "scheduler"),
        (scaler, "scaler_state_dict", "scaler"),
    ):
        if (component is None) != (checkpoint[state_name] is None):
            raise ValueError(f"checkpoint {name} presence differs from runtime")
    components = (
        (model, "model_state_dict", "model"),
        (objective, "objective_state_dict", "objective"),
        (optimizer, "optimizer_state_dict", "optimizer"),
    )
    for component, state_name, name in components:
        loader = getattr(component, "load_state_dict", None)
        if not callable(loader):
            raise TypeError(f"{name} must provide load_state_dict()")
        loader(checkpoint[state_name])
    for component, state_name, name in (
        (scheduler, "scheduler_state_dict", "scheduler"),
        (scaler, "scaler_state_dict", "scaler"),
    ):
        state = checkpoint[state_name]
        if component is not None:
            loader = getattr(component, "load_state_dict", None)
            if not callable(loader):
                raise TypeError(f"{name} must provide load_state_dict()")
            loader(state)
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])


__all__ = [
    "SCHEMA_VERSION",
    "build_training_checkpoint",
    "capture_rng_state",
    "load_training_checkpoint",
    "replace_training_checkpoint",
    "restore_rng_state",
    "restore_training_checkpoint",
    "save_training_checkpoint",
    "validate_training_checkpoint",
]
