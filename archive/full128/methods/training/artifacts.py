"""Versioned Full128 run configuration and array artifact contracts."""

from __future__ import annotations

import hashlib
import os
import platform
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from shared.foundation.protected_io import read_strict_json_document
from shared.foundation.provenance import content_sha256
from archive.full128.methods.preparation.data import Full128Sample

RUN_CONFIG_SCHEMA = "archive.full128.training_run_config.v2"
EMBEDDING_CACHE_SCHEMA = "archive.full128.embedding_cache.v1"
VARIANT_RUN_SCHEMA = "archive.full128.variant_run.v1"
FAMILY_RUN_SCHEMA = "archive.full128.family_run.v1"
CURRENT_GROUP_QUOTAS = (
    {"dataset_name": "dogfacenet224", "view": "body", "identities": 9},
    {"dataset_name": "yt-bb-dog", "view": "body", "identities": 18},
)
_RUN_CONFIG_FIELDS = {
    "schema_version",
    "seed",
    "epochs",
    "optimizer",
    "sampler",
    "precision",
    "workers",
    "model_selection",
    "augmentation",
}
_VARIANT_FIELDS = {
    "schema_version",
    "variant_id",
    "method",
    "initialization",
    "bindings",
    "fit_population",
    "training",
    "artifacts",
    "variant_run_sha256",
}
_CACHE_VECTOR_FIELDS = {
    "sample_id",
    "identity_id",
    "dataset_name",
    "view",
    "role",
    "crop_record_sha256",
    "offset_bytes",
    "byte_size",
    "sha256",
}
_CACHE_ROLES = {"FIT", "DEV", "CAL", "EVAL"}
_MAX_EMBEDDING_PACK_BYTES = 8 * 1024 * 1024 * 1024


def default_full128_run_config() -> dict[str, Any]:
    """Return factual protocol defaults without making optimality claims."""

    return {
        "schema_version": RUN_CONFIG_SCHEMA,
        "seed": 20260811,
        "epochs": 10,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
        "sampler": {
            "kind": "DATASET_VIEW_BALANCED_PK",
            "samples_per_identity": 2,
            "group_quotas": [dict(item) for item in CURRENT_GROUP_QUOTAS],
            "logical_batch_size": 54,
        },
        "precision": {
            "device": "cuda",
            "amp": True,
            "amp_dtype": "float16",
        },
        "workers": 8,
        "model_selection": "FIXED_LAST_EPOCH",
        "augmentation": "NONE",
    }


def validate_full128_run_config(value: object) -> dict[str, Any]:
    """Validate the exact current Full128 training protocol configuration."""

    if not isinstance(value, Mapping) or set(value) != _RUN_CONFIG_FIELDS:
        raise ValueError("Full128 run config fields differ")
    config = dict(value)
    if config["schema_version"] != RUN_CONFIG_SCHEMA:
        raise ValueError("Full128 run config schema differs")
    for name in ("seed", "epochs", "workers"):
        item = config[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"Full128 run config {name} must be an integer")
    if config["seed"] < 0 or config["epochs"] <= 0 or config["workers"] < 0:
        raise ValueError("Full128 run config integer range differs")
    optimizer = config["optimizer"]
    if not isinstance(optimizer, Mapping) or set(optimizer) != {
        "name",
        "learning_rate",
        "weight_decay",
    }:
        raise ValueError("Full128 optimizer config fields differ")
    if optimizer["name"] != "AdamW":
        raise ValueError("Full128 optimizer must be AdamW")
    for name in ("learning_rate", "weight_decay"):
        item = optimizer[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not np.isfinite(item)
        ):
            raise ValueError(f"Full128 optimizer {name} must be finite")
    if optimizer["learning_rate"] <= 0 or optimizer["weight_decay"] < 0:
        raise ValueError("Full128 optimizer value range differs")
    sampler = config["sampler"]
    if not isinstance(sampler, Mapping) or set(sampler) != {
        "kind",
        "samples_per_identity",
        "group_quotas",
        "logical_batch_size",
    }:
        raise ValueError("Full128 sampler config fields differ")
    if (
        sampler["kind"] != "DATASET_VIEW_BALANCED_PK"
        or sampler["samples_per_identity"] != 2
        or sampler["logical_batch_size"] != 54
        or sampler["group_quotas"] != [dict(item) for item in CURRENT_GROUP_QUOTAS]
    ):
        raise ValueError(
            "Full128 sampler must use admitted DogFaceNet=9 and YT-BB=18 quotas with K=2"
        )
    precision = config["precision"]
    if not isinstance(precision, Mapping) or set(precision) != {
        "device",
        "amp",
        "amp_dtype",
    }:
        raise ValueError("Full128 precision config fields differ")
    if precision["device"] not in {"cpu", "cuda"} or not isinstance(
        precision["amp"], bool
    ):
        raise ValueError("Full128 precision device or AMP flag differs")
    expected_dtype = "float16" if precision["amp"] else "float32"
    if precision["amp_dtype"] != expected_dtype:
        raise ValueError("Full128 AMP dtype differs")
    if precision["device"] == "cpu" and precision["amp"]:
        raise ValueError("Full128 CPU runs cannot claim CUDA AMP")
    if config["model_selection"] != "FIXED_LAST_EPOCH":
        raise ValueError("Full128 model selection must use the fixed last epoch")
    if config["augmentation"] != "NONE":
        raise ValueError("Full128 training does not admit augmentation")
    return config


def group_quotas_from_config(config: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    validated = validate_full128_run_config(config)
    return {
        (item["dataset_name"], item["view"]): item["identities"]
        for item in validated["sampler"]["group_quotas"]
    }


def runtime_versions() -> dict[str, str]:
    return {
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "numpy": _distribution_version("numpy"),
        "pillow": _distribution_version("pillow"),
        "opencv_python_headless": _distribution_version("opencv-python-headless"),
        "scikit_learn": _distribution_version("scikit-learn"),
        "torch": _distribution_version("torch"),
        "torchvision": _distribution_version("torchvision"),
        "safetensors": _distribution_version("safetensors"),
    }


def file_binding(path: Path) -> dict[str, Any]:
    digest, size = _hash_regular_file(path)
    return {"sha256": digest, "byte_size": size}


def write_embedding_cache(
    path: Path,
    samples: Sequence[Full128Sample],
    embeddings: np.ndarray,
) -> dict[str, Any]:
    """Write 128D vectors as one packed little-endian float32 file."""

    rows = tuple(samples)
    _validate_cache_samples(rows)
    matrix = _validated_embedding_matrix(embeddings, expected_count=len(rows))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite Full128 embedding pack: {path}")
    packed_array = np.ascontiguousarray(matrix, dtype="<f4")
    packed = memoryview(packed_array).cast("B")
    with path.open("xb") as stream:
        stream.write(packed)
        stream.flush()
        os.fsync(stream.fileno())
    vectors = []
    for index, sample in enumerate(rows):
        start = index * 512
        vector_bytes = packed[start : start + 512]
        vectors.append(
            {
                "sample_id": sample.sample_id,
                "identity_id": sample.identity_id,
                "dataset_name": sample.dataset_name,
                "view": sample.view,
                "role": sample.role,
                "crop_record_sha256": sample.crop_record_sha256,
                "offset_bytes": start,
                "byte_size": 512,
                "sha256": hashlib.sha256(vector_bytes).hexdigest(),
            }
        )
    payload = {
        "schema_version": EMBEDDING_CACHE_SCHEMA,
        "relative_path": path.name,
        "dtype": "float32_little_endian",
        "dimension": 128,
        "bytes_per_vector": 512,
        "vector_count": len(rows),
        "pack_byte_size": packed.nbytes,
        "pack_sha256": hashlib.sha256(packed).hexdigest(),
        "vectors": vectors,
    }
    return {**payload, "cache_manifest_sha256": content_sha256(payload)}


def validate_embedding_cache(root: Path, value: object) -> np.ndarray:
    """Validate every vector and the whole pack before returning a copied matrix."""

    matrix = _stream_validate_embedding_cache(root, value, retain_embeddings=True)
    assert matrix is not None
    return matrix


def _stream_validate_embedding_cache(
    root: Path,
    value: object,
    *,
    retain_embeddings: bool,
) -> np.ndarray | None:
    """Validate one pack in bounded memory, optionally retaining its vectors."""

    expected = {
        "schema_version",
        "relative_path",
        "dtype",
        "dimension",
        "bytes_per_vector",
        "vector_count",
        "pack_byte_size",
        "pack_sha256",
        "vectors",
        "cache_manifest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Full128 embedding cache manifest fields differ")
    manifest = dict(value)
    payload = {
        key: item for key, item in manifest.items() if key != "cache_manifest_sha256"
    }
    for name in ("dimension", "bytes_per_vector", "vector_count", "pack_byte_size"):
        _require_plain_int(manifest[name], f"Full128 embedding cache {name}")
    _require_sha256(manifest["cache_manifest_sha256"], "cache manifest SHA-256")
    _require_sha256(manifest["pack_sha256"], "embedding pack SHA-256")
    if (
        manifest["schema_version"] != EMBEDDING_CACHE_SCHEMA
        or manifest["cache_manifest_sha256"] != content_sha256(payload)
        or manifest["dtype"] != "float32_little_endian"
        or manifest["dimension"] != 128
        or manifest["bytes_per_vector"] != 512
    ):
        raise ValueError("Full128 embedding cache contract differs")
    if (
        manifest["vector_count"] <= 0
        or manifest["pack_byte_size"] != manifest["vector_count"] * 512
        or manifest["pack_byte_size"] > _MAX_EMBEDDING_PACK_BYTES
    ):
        raise ValueError("Full128 embedding cache count or byte range differs")
    relative = manifest["relative_path"]
    if not isinstance(relative, str) or not relative or Path(relative).name != relative:
        raise ValueError("Full128 embedding pack path differs")
    vectors = manifest["vectors"]
    if not isinstance(vectors, list) or len(vectors) != manifest["vector_count"]:
        raise ValueError("Full128 embedding vector manifest count differs")
    seen: set[str] = set()
    for index, row in enumerate(vectors):
        if not isinstance(row, Mapping) or set(row) != _CACHE_VECTOR_FIELDS:
            raise ValueError("Full128 embedding vector manifest fields differ")
        _require_sha256(row["sample_id"], "Full128 embedding vector sample_id")
        _require_identity_id(row["identity_id"])
        dataset_name = _require_canonical_text(
            row["dataset_name"], "Full128 embedding vector dataset_name"
        )
        view = _require_canonical_text(row["view"], "Full128 embedding vector view")
        role = _require_canonical_text(row["role"], "Full128 embedding vector role")
        if (
            dataset_name.lower() != dataset_name
            or view not in {"face", "body"}
            or role not in _CACHE_ROLES
        ):
            raise ValueError("Full128 embedding vector view or role differs")
        _require_sha256(row["crop_record_sha256"], "crop record SHA-256")
        _require_sha256(row["sha256"], "embedding vector SHA-256")
        _require_plain_int(row["offset_bytes"], "embedding vector offset")
        _require_plain_int(row["byte_size"], "embedding vector byte size")
        if (
            row["sample_id"] in seen
            or (index > 0 and vectors[index - 1]["sample_id"] >= row["sample_id"])
            or row["offset_bytes"] != index * 512
            or row["byte_size"] != 512
        ):
            raise ValueError("Full128 embedding vector order or identity differs")
        seen.add(row["sample_id"])

    path = root / relative
    if path.is_symlink():
        raise ValueError("Full128 artifact must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    pack_digest = hashlib.sha256()
    observed = 0
    first_vector_digest_error: int | None = None
    first_vector_value_error: int | None = None
    matrix = (
        np.empty((manifest["vector_count"], 128), dtype=np.float32)
        if retain_embeddings
        else None
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= _MAX_EMBEDDING_PACK_BYTES
        ):
            raise ValueError("Full128 artifact size or file type differs")
        for index, row in enumerate(vectors):
            chunks: list[bytes] = []
            remaining = 512
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                pack_digest.update(chunk)
                observed += len(chunk)
                remaining -= len(chunk)
            vector_bytes = b"".join(chunks)
            if len(vector_bytes) != 512:
                break
            if (
                first_vector_digest_error is None
                and hashlib.sha256(vector_bytes).hexdigest() != row["sha256"]
            ):
                first_vector_digest_error = index
            vector = np.frombuffer(vector_bytes, dtype="<f4")
            if first_vector_value_error is None and (
                not np.isfinite(vector).all()
                or not np.isclose(
                    np.linalg.norm(vector.astype(np.float64)),
                    1.0,
                    atol=1e-5,
                    rtol=1e-5,
                )
            ):
                first_vector_value_error = index
            if matrix is not None:
                matrix[index] = vector
        while chunk := os.read(descriptor, 1_048_576):
            observed += len(chunk)
            if observed > _MAX_EMBEDDING_PACK_BYTES:
                raise ValueError("Full128 artifact exceeds byte limit")
            pack_digest.update(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        raise RuntimeError("Full128 artifact changed while being read")
    if (
        observed != before.st_size
        or observed != manifest["pack_byte_size"]
        or observed != manifest["vector_count"] * 512
        or pack_digest.hexdigest() != manifest["pack_sha256"]
    ):
        raise ValueError("Full128 embedding pack digest or length differs")
    if first_vector_digest_error is not None:
        raise ValueError("Full128 embedding vector digest differs")
    if first_vector_value_error is not None:
        raise ValueError("Full128 embeddings must be finite [N,128] and L2 normalized")
    return matrix


def validate_variant_run(root: Path, value: object) -> dict[str, Any]:
    """Validate a completed variant's manifest and every bound local artifact."""

    if not isinstance(value, Mapping) or set(value) != _VARIANT_FIELDS:
        raise ValueError("Full128 variant run manifest fields differ")
    manifest = dict(value)
    payload = {
        key: item for key, item in manifest.items() if key != "variant_run_sha256"
    }
    if manifest["schema_version"] != VARIANT_RUN_SCHEMA or manifest[
        "variant_run_sha256"
    ] != content_sha256(payload):
        raise ValueError("Full128 variant run manifest digest differs")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "state",
        "model_manifest",
        "preprocessing_manifest",
        "embedding_manifest",
        "checkpoint_manifest",
        "embedding_cache_manifest",
    }:
        raise ValueError("Full128 variant artifact bindings differ")
    for name in (
        "state",
        "model_manifest",
        "preprocessing_manifest",
        "embedding_manifest",
        "checkpoint_manifest",
    ):
        binding = artifacts[name]
        if not isinstance(binding, Mapping) or set(binding) != {
            "relative_path",
            "sha256",
            "byte_size",
        }:
            raise ValueError(f"Full128 {name} binding fields differ")
        if (
            not isinstance(binding["relative_path"], str)
            or Path(binding["relative_path"]).name != binding["relative_path"]
        ):
            raise ValueError(f"Full128 {name} relative path differs")
        path = root / binding["relative_path"]
        if file_binding(path) != {
            "sha256": binding["sha256"],
            "byte_size": binding["byte_size"],
        }:
            raise ValueError(f"Full128 {name} binding differs")
    cache_binding = artifacts["embedding_cache_manifest"]
    if not isinstance(cache_binding, Mapping) or set(cache_binding) != {
        "relative_path",
        "sha256",
        "byte_size",
        "manifest",
    }:
        raise ValueError("Full128 embedding cache manifest binding fields differ")
    if (
        not isinstance(cache_binding["relative_path"], str)
        or Path(cache_binding["relative_path"]).name != cache_binding["relative_path"]
    ):
        raise ValueError("Full128 embedding cache manifest relative path differs")
    cache_path = root / cache_binding["relative_path"]
    cache_document = read_strict_json_document(
        cache_path,
        maximum_bytes=1_073_741_824,
        maximum_nodes=10_000_000,
        maximum_keys=10_000_000,
        maximum_array_length=1_000_000,
    )
    if (
        cache_document.raw_sha256 != cache_binding["sha256"]
        or cache_document.byte_size != cache_binding["byte_size"]
        or cache_document.payload != cache_binding["manifest"]
    ):
        raise ValueError("Full128 embedding cache manifest binding differs")
    _stream_validate_embedding_cache(
        root,
        cache_binding["manifest"],
        retain_embeddings=False,
    )
    return manifest


def _validated_embedding_matrix(
    values: np.ndarray, *, expected_count: int
) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.shape != (expected_count, 128) or not np.isfinite(matrix).all():
        raise ValueError("Full128 embeddings must be finite [N,128]")
    matrix = np.asarray(matrix, dtype=np.float32)
    for start in range(0, len(matrix), 16_384):
        block = matrix[start : start + 16_384]
        norms = np.linalg.norm(block.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("Full128 embeddings must be L2 normalized")
    return matrix


def _hash_regular_file(path: Path) -> tuple[str, int]:
    if path.is_symlink():
        raise ValueError("Full128 artifact must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    observed = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= _MAX_EMBEDDING_PACK_BYTES
        ):
            raise ValueError("Full128 artifact size or file type differs")
        while chunk := os.read(descriptor, 1_048_576):
            observed += len(chunk)
            if observed > _MAX_EMBEDDING_PACK_BYTES:
                raise ValueError("Full128 artifact exceeds byte limit")
            digest.update(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        raise RuntimeError("Full128 artifact changed while being hashed")
    if observed != before.st_size:
        raise RuntimeError("Full128 artifact changed while being hashed")
    return digest.hexdigest(), observed


def _validate_cache_samples(rows: Sequence[Full128Sample]) -> None:
    if not rows:
        raise ValueError("Full128 embedding cache must be non-empty")
    previous_sample_id: str | None = None
    for sample in rows:
        _require_sha256(sample.sample_id, "Full128 cache sample sample_id")
        _require_identity_id(sample.identity_id)
        dataset_name = _require_canonical_text(
            sample.dataset_name, "Full128 cache sample dataset_name"
        )
        if previous_sample_id is not None and previous_sample_id >= sample.sample_id:
            raise ValueError(
                "Full128 embedding cache samples must be canonically ordered"
            )
        if (
            dataset_name.lower() != dataset_name
            or sample.view not in {"face", "body"}
            or sample.role not in _CACHE_ROLES
        ):
            raise ValueError("Full128 cache sample view or role differs")
        _require_sha256(sample.crop_record_sha256, "crop record SHA-256")
        previous_sample_id = sample.sample_id


def _require_plain_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _require_canonical_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def _require_identity_id(value: object) -> str:
    text = _require_canonical_text(value, "Full128 embedding vector identity_id")
    try:
        parsed = uuid.UUID(text)
    except ValueError as exc:
        raise ValueError("Full128 embedding vector identity_id must be a UUID") from exc
    if parsed.version != 5 or str(parsed) != text:
        raise ValueError(
            "Full128 embedding vector identity_id must be canonical UUIDv5"
        )
    return text


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Full128 runtime distribution {name!r} is not installed"
        ) from exc


__all__ = [
    "CURRENT_GROUP_QUOTAS",
    "EMBEDDING_CACHE_SCHEMA",
    "FAMILY_RUN_SCHEMA",
    "RUN_CONFIG_SCHEMA",
    "VARIANT_RUN_SCHEMA",
    "default_full128_run_config",
    "file_binding",
    "group_quotas_from_config",
    "runtime_versions",
    "validate_embedding_cache",
    "validate_full128_run_config",
    "validate_variant_run",
    "write_embedding_cache",
]
