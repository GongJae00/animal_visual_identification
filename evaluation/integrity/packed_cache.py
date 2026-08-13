"""Canonical single-file embedding-cache candidate and format verifier."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.controls.control_scoring import (
    ArtifactCacheBinding,
    ControlScoringInventory,
    EmbeddingCachePolicy,
    embedding_cache_key,
)
from foundation.provenance import content_sha256


PACKED_VECTOR_FILE_NAME = "vectors.f32le.pack"
PACKED_STORAGE_LAYOUT = "PACKED_CONTIGUOUS_FIXED_WIDTH"
PACKED_ENTRY_ORDERING = "cache_key_lexicographic"
_MAX_SIGNED_OFFSET = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class PackedEmbeddingCacheEntry:
    cache_key: str
    content_sha256: str
    byte_offset: int
    byte_size: int

    def __post_init__(self) -> None:
        _validate_sha256(self.cache_key, "cache_key")
        _validate_sha256(self.content_sha256, "content_sha256")
        _require_nonnegative_int(self.byte_offset, "byte_offset")
        _require_positive_int(self.byte_size, "byte_size")
        if self.byte_offset > _MAX_SIGNED_OFFSET - self.byte_size:
            raise ValueError("packed embedding entry exceeds signed offset range")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "cache_key": self.cache_key,
            "content_sha256": self.content_sha256,
            "byte_offset": self.byte_offset,
            "byte_size": self.byte_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PackedEmbeddingCacheEntry:
        _require_exact_keys(
            payload,
            {"cache_key", "content_sha256", "byte_offset", "byte_size"},
            "packed embedding cache entry",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PackedEmbeddingCacheStorage:
    content_sha256: str
    byte_size: int
    vector_count: int
    vector_stride_bytes: int
    relative_path: str = PACKED_VECTOR_FILE_NAME
    layout: str = PACKED_STORAGE_LAYOUT
    ordering: str = PACKED_ENTRY_ORDERING

    def __post_init__(self) -> None:
        if self.relative_path != PACKED_VECTOR_FILE_NAME:
            raise ValueError("packed embedding storage path is fixed")
        if self.layout != PACKED_STORAGE_LAYOUT:
            raise ValueError("unsupported packed embedding storage layout")
        if self.ordering != PACKED_ENTRY_ORDERING:
            raise ValueError("unsupported packed embedding entry ordering")
        _validate_sha256(self.content_sha256, "content_sha256")
        _require_positive_int(self.byte_size, "byte_size")
        _require_positive_int(self.vector_count, "vector_count")
        _require_positive_int(
            self.vector_stride_bytes,
            "vector_stride_bytes",
        )
        if self.byte_size > _MAX_SIGNED_OFFSET:
            raise ValueError("packed embedding storage exceeds signed offset range")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "relative_path": self.relative_path,
            "layout": self.layout,
            "ordering": self.ordering,
            "content_sha256": self.content_sha256,
            "byte_size": self.byte_size,
            "vector_count": self.vector_count,
            "vector_stride_bytes": self.vector_stride_bytes,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> PackedEmbeddingCacheStorage:
        _require_exact_keys(
            payload,
            {
                "relative_path",
                "layout",
                "ordering",
                "content_sha256",
                "byte_size",
                "vector_count",
                "vector_stride_bytes",
            },
            "packed embedding cache storage",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PackedEmbeddingCacheManifest:
    scoring_inventory_sha256: str
    model_sha256: str
    inference_config_sha256: str
    dependency_lock_sha256: str
    code_revision: str
    precision: str
    vector_dimension: int
    normalization_tolerance: float
    storage: PackedEmbeddingCacheStorage
    bindings: tuple[ArtifactCacheBinding, ...]
    entries: tuple[PackedEmbeddingCacheEntry, ...]
    vector_format: str = "float32_le"
    schema_version: str = "cvi.embedding_cache_manifest.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_cache_manifest.v2":
            raise ValueError("unsupported packed embedding cache manifest schema")
        for name in (
            "scoring_inventory_sha256",
            "model_sha256",
            "inference_config_sha256",
            "dependency_lock_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        _require_nonempty(self.code_revision, "code_revision")
        _require_nonempty(self.precision, "precision")
        _require_positive_int(self.vector_dimension, "vector_dimension")
        _require_finite_positive(
            self.normalization_tolerance,
            "normalization_tolerance",
        )
        if self.vector_format != "float32_le":
            raise ValueError("embedding vector format is fixed to float32_le")
        if not self.bindings or not self.entries:
            raise ValueError("packed embedding cache manifest must not be empty")

        binding_tokens = tuple(item.artifact_token for item in self.bindings)
        if binding_tokens != tuple(sorted(binding_tokens)):
            raise ValueError("packed embedding bindings must be token-sorted")
        if len(binding_tokens) != len(set(binding_tokens)):
            raise ValueError("packed embedding artifact bindings must be unique")

        entry_keys = tuple(item.cache_key for item in self.entries)
        if entry_keys != tuple(sorted(entry_keys)):
            raise ValueError("packed embedding entries must be cache-key-sorted")
        if len(entry_keys) != len(set(entry_keys)):
            raise ValueError("packed embedding cache keys must be unique")
        if {item.cache_key for item in self.bindings} != set(entry_keys):
            raise ValueError("packed entries and bindings must match exactly")

        stride = self.vector_dimension * 4
        if stride > _MAX_SIGNED_OFFSET:
            raise ValueError("packed embedding vector stride exceeds offset range")
        if self.storage.vector_stride_bytes != stride:
            raise ValueError("packed embedding stride differs from dimension")
        if self.storage.vector_count != len(self.entries):
            raise ValueError("packed embedding vector count differs")
        expected_storage_bytes = stride * len(self.entries)
        if expected_storage_bytes > _MAX_SIGNED_OFFSET:
            raise ValueError("packed embedding storage exceeds offset range")
        if self.storage.byte_size != expected_storage_bytes:
            raise ValueError("packed embedding storage byte size differs")
        for ordinal, entry in enumerate(self.entries):
            if entry.byte_size != stride:
                raise ValueError("packed embedding entry byte size differs")
            if entry.byte_offset != ordinal * stride:
                raise ValueError("packed embedding entry offset is not canonical")

        for binding in self.bindings:
            expected_key = embedding_cache_key(
                artifact_content_sha256=binding.artifact_content_sha256,
                model_sha256=self.model_sha256,
                inference_config_sha256=self.inference_config_sha256,
                dependency_lock_sha256=self.dependency_lock_sha256,
                code_revision=self.code_revision,
                precision=self.precision,
                vector_dimension=self.vector_dimension,
                vector_format=self.vector_format,
            )
            if binding.cache_key != expected_key:
                raise ValueError("packed embedding cache key provenance mismatch")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def logical_cache_sha256(self) -> str:
        """Hash storage-independent vector semantics and content bindings."""

        return content_sha256(
            {
                "schema_version": "cvi.logical_embedding_cache.v1",
                "scoring_inventory_sha256": self.scoring_inventory_sha256,
                "model_sha256": self.model_sha256,
                "inference_config_sha256": self.inference_config_sha256,
                "dependency_lock_sha256": self.dependency_lock_sha256,
                "code_revision": self.code_revision,
                "precision": self.precision,
                "vector_dimension": self.vector_dimension,
                "normalization_tolerance": self.normalization_tolerance,
                "vector_format": self.vector_format,
                "bindings": [item.to_dict() for item in self.bindings],
                "entries": [
                    {
                        "cache_key": item.cache_key,
                        "content_sha256": item.content_sha256,
                        "byte_size": item.byte_size,
                    }
                    for item in self.entries
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scoring_inventory_sha256": self.scoring_inventory_sha256,
            "model_sha256": self.model_sha256,
            "inference_config_sha256": self.inference_config_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "code_revision": self.code_revision,
            "precision": self.precision,
            "vector_dimension": self.vector_dimension,
            "normalization_tolerance": self.normalization_tolerance,
            "vector_format": self.vector_format,
            "storage": self.storage.to_dict(),
            "bindings": [item.to_dict() for item in self.bindings],
            "entries": [item.to_dict() for item in self.entries],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> PackedEmbeddingCacheManifest:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "scoring_inventory_sha256",
                "model_sha256",
                "inference_config_sha256",
                "dependency_lock_sha256",
                "code_revision",
                "precision",
                "vector_dimension",
                "normalization_tolerance",
                "vector_format",
                "storage",
                "bindings",
                "entries",
            },
            "packed embedding cache manifest",
        )
        storage = payload["storage"]
        bindings = payload["bindings"]
        entries = payload["entries"]
        if not isinstance(storage, dict):
            raise TypeError("packed embedding storage must be an object")
        if not isinstance(bindings, list) or not isinstance(entries, list):
            raise TypeError("packed embedding bindings and entries must be lists")
        return cls(
            schema_version=payload["schema_version"],
            scoring_inventory_sha256=payload["scoring_inventory_sha256"],
            model_sha256=payload["model_sha256"],
            inference_config_sha256=payload["inference_config_sha256"],
            dependency_lock_sha256=payload["dependency_lock_sha256"],
            code_revision=payload["code_revision"],
            precision=payload["precision"],
            vector_dimension=payload["vector_dimension"],
            normalization_tolerance=payload["normalization_tolerance"],
            vector_format=payload["vector_format"],
            storage=PackedEmbeddingCacheStorage.from_dict(storage),
            bindings=tuple(ArtifactCacheBinding.from_dict(item) for item in bindings),
            entries=tuple(PackedEmbeddingCacheEntry.from_dict(item) for item in entries),
        )


@dataclass(frozen=True, slots=True)
class PackedEmbeddingCacheVerification:
    cache_manifest_sha256: str
    logical_cache_sha256: str
    cache_policy_sha256: str
    observed_pack_sha256: str
    verified_files: int
    verified_bytes: int
    verified_vectors: int
    maximum_observed_norm_error: float
    schema_version: str = "cvi.embedding_cache_verification.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_cache_verification.v2":
            raise ValueError("unsupported packed cache verification schema")
        for name in (
            "cache_manifest_sha256",
            "logical_cache_sha256",
            "cache_policy_sha256",
            "observed_pack_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in (
            "verified_files",
            "verified_bytes",
            "verified_vectors",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        _require_finite_nonnegative(
            self.maximum_observed_norm_error,
            "maximum_observed_norm_error",
        )

    @property
    def verification_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "schema_version": self.schema_version,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "logical_cache_sha256": self.logical_cache_sha256,
            "cache_policy_sha256": self.cache_policy_sha256,
            "observed_pack_sha256": self.observed_pack_sha256,
            "verified_files": self.verified_files,
            "verified_bytes": self.verified_bytes,
            "verified_vectors": self.verified_vectors,
            "maximum_observed_norm_error": self.maximum_observed_norm_error,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> PackedEmbeddingCacheVerification:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "cache_manifest_sha256",
                "logical_cache_sha256",
                "cache_policy_sha256",
                "observed_pack_sha256",
                "verified_files",
                "verified_bytes",
                "verified_vectors",
                "maximum_observed_norm_error",
            },
            "packed embedding cache verification",
        )
        return cls(**payload)


def verify_packed_embedding_cache_files(
    *,
    root: Path,
    inventory: ControlScoringInventory,
    manifest: PackedEmbeddingCacheManifest,
    policy: EmbeddingCachePolicy,
    verification_phase_callback: Callable[[str], None] | None = None,
) -> PackedEmbeddingCacheVerification:
    """Verify one closed pack with bounded memory and one retained file FD."""

    _verify_inventory_and_policy(inventory, manifest, policy)
    resolved = _resolve_real_directory(root)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    file_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
        file_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW

    directory_fd = os.open(resolved, directory_flags)
    try:
        initial_directory = os.fstat(directory_fd)
        _require_closed_directory(directory_fd)
        pack_fd = os.open(
            PACKED_VECTOR_FILE_NAME,
            file_flags,
            dir_fd=directory_fd,
        )
        try:
            initial_pack = os.fstat(pack_fd)
            if not stat.S_ISREG(initial_pack.st_mode):
                raise ValueError("packed embedding storage must be a regular file")
            if initial_pack.st_size != manifest.storage.byte_size:
                raise ValueError("packed embedding storage byte size mismatch")
            _require_path_matches_fd(directory_fd, pack_fd, initial_pack)
            if verification_phase_callback is not None:
                verification_phase_callback("PACK_OPENED")

            pack_hasher = hashlib.sha256()
            maximum_error = 0.0
            for entry in manifest.entries:
                slice_sha256, norm_error = _scan_vector_slice(
                    pack_fd,
                    entry=entry,
                    dimension=manifest.vector_dimension,
                    chunk_floats=policy.scan_chunk_floats,
                    pack_hasher=pack_hasher,
                )
                if slice_sha256 != entry.content_sha256:
                    raise ValueError("packed embedding slice hash mismatch")
                if norm_error > manifest.normalization_tolerance:
                    raise ValueError("packed embedding vector is not L2-normalized")
                maximum_error = max(maximum_error, norm_error)

            observed_pack_sha256 = pack_hasher.hexdigest()
            if observed_pack_sha256 != manifest.storage.content_sha256:
                raise ValueError("packed embedding whole-file hash mismatch")
            if verification_phase_callback is not None:
                verification_phase_callback("PACK_SCANNED")

            final_pack = os.fstat(pack_fd)
            if _stat_identity(initial_pack) != _stat_identity(final_pack):
                raise RuntimeError("packed embedding storage changed during verification")
            _require_path_matches_fd(directory_fd, pack_fd, final_pack)
        finally:
            os.close(pack_fd)

        _require_closed_directory(directory_fd)
        final_directory = os.fstat(directory_fd)
        if _directory_identity(initial_directory) != _directory_identity(final_directory):
            raise RuntimeError("packed embedding directory changed during verification")
    finally:
        os.close(directory_fd)

    return PackedEmbeddingCacheVerification(
        cache_manifest_sha256=manifest.manifest_sha256,
        logical_cache_sha256=manifest.logical_cache_sha256,
        cache_policy_sha256=policy.policy_sha256,
        observed_pack_sha256=observed_pack_sha256,
        verified_files=1,
        verified_bytes=manifest.storage.byte_size,
        verified_vectors=len(manifest.entries),
        maximum_observed_norm_error=maximum_error,
    )


def _verify_inventory_and_policy(
    inventory: ControlScoringInventory,
    manifest: PackedEmbeddingCacheManifest,
    policy: EmbeddingCachePolicy,
) -> None:
    if manifest.scoring_inventory_sha256 != inventory.inventory_sha256:
        raise ValueError("packed embedding cache belongs to another inventory")
    inventory_by_token = {
        item.artifact_token: item for item in inventory.entries
    }
    binding_by_token = {
        item.artifact_token: item for item in manifest.bindings
    }
    if set(binding_by_token) != set(inventory_by_token):
        raise ValueError("packed embedding bindings do not cover inventory")
    for token, inventory_entry in inventory_by_token.items():
        if (
            binding_by_token[token].artifact_content_sha256
            != inventory_entry.content_sha256
        ):
            raise ValueError("packed embedding artifact content mismatch")
    if len(manifest.bindings) > policy.maximum_artifacts:
        raise ValueError("packed embedding bindings exceed policy")
    if len(manifest.entries) > policy.maximum_unique_vectors:
        raise ValueError("packed embedding vectors exceed policy")
    if manifest.vector_dimension > policy.maximum_vector_dimension:
        raise ValueError("packed embedding dimension exceeds policy")
    if manifest.storage.vector_stride_bytes > policy.maximum_vector_bytes:
        raise ValueError("packed embedding vector bytes exceed policy")
    if manifest.storage.byte_size > policy.maximum_total_cache_bytes:
        raise ValueError("packed embedding cache bytes exceed policy")
    if manifest.normalization_tolerance > policy.maximum_normalization_tolerance:
        raise ValueError("packed embedding normalization tolerance exceeds policy")


def _resolve_real_directory(root: Path) -> Path:
    if root.is_symlink():
        raise ValueError("packed embedding cache root must not be a symlink")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("packed embedding cache root must be a directory")
    return resolved


def _require_closed_directory(directory_fd: int) -> None:
    names = os.listdir(directory_fd)
    if names != [PACKED_VECTOR_FILE_NAME] and set(names) != {PACKED_VECTOR_FILE_NAME}:
        raise ValueError("packed embedding cache directory is not a closed set")


def _require_path_matches_fd(
    directory_fd: int,
    pack_fd: int,
    expected: os.stat_result,
) -> None:
    del pack_fd
    observed = os.stat(
        PACKED_VECTOR_FILE_NAME,
        dir_fd=directory_fd,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError("packed embedding path is not a regular file")
    if (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError("packed embedding path no longer names opened storage")


def _pread_exact(descriptor: int, byte_size: int, offset: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_size
    position = offset
    while remaining:
        payload = os.pread(descriptor, remaining, position)
        if not payload:
            raise ValueError("packed embedding slice is truncated")
        chunks.append(payload)
        position += len(payload)
        remaining -= len(payload)
    return b"".join(chunks)


def _scan_vector_slice(
    descriptor: int,
    *,
    entry: PackedEmbeddingCacheEntry,
    dimension: int,
    chunk_floats: int,
    pack_hasher: Any,
) -> tuple[str, float]:
    remaining = dimension
    position = entry.byte_offset
    total = 0.0
    compensation = 0.0
    slice_hasher = hashlib.sha256()
    while remaining:
        count = min(remaining, chunk_floats)
        payload = _pread_exact(descriptor, count * 4, position)
        slice_hasher.update(payload)
        pack_hasher.update(payload)
        subtotal = math.fsum(_finite_squares(payload))
        total, compensation = _neumaier_add(
            total,
            compensation,
            subtotal,
        )
        position += len(payload)
        remaining -= count
    norm = math.sqrt(total + compensation)
    return slice_hasher.hexdigest(), abs(norm - 1.0)


def _finite_squares(payload: bytes) -> Iterator[float]:
    for (value,) in struct.iter_unpack("<f", payload):
        if not math.isfinite(value):
            raise ValueError("packed embedding vector is malformed")
        yield value * value


def _neumaier_add(
    total: float,
    compensation: float,
    value: float,
) -> tuple[float, float]:
    updated = total + value
    if abs(total) >= abs(value):
        compensation += (total - updated) + value
    else:
        compensation += (value - updated) + total
    return updated, compensation


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    name: str,
) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{name} fields differ")


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite_positive(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _require_finite_nonnegative(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
