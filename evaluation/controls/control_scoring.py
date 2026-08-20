"""Label-blind artifact inventory, embedding-cache, and control scoring."""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator

from evaluation.controls.policy import ControlScoringRequest
from evaluation.controls.control_transform import (
    ControlTransformReceipt,
    verify_control_artifact_files,
)
from shared.foundation.provenance import content_sha256
from evaluation.controls.scoring import (
    PairArtifactManifest,
    PairArtifactVerification,
    verify_pair_artifact_files,
)

_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ArtifactSourceKind(StrEnum):
    BASE = "BASE"
    CONTROL = "CONTROL"


@dataclass(frozen=True, slots=True)
class ScoringArtifactEntry:
    artifact_token: str
    content_sha256: str
    byte_size: int
    source_kind: ArtifactSourceKind

    def __post_init__(self) -> None:
        _require_safe_token(self.artifact_token, "artifact_token")
        _validate_sha256(self.content_sha256, "content_sha256")
        _require_positive_int(self.byte_size, "byte_size")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "artifact_token": self.artifact_token,
            "content_sha256": self.content_sha256,
            "byte_size": self.byte_size,
            "source_kind": self.source_kind.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoringArtifactEntry:
        _require_exact_keys(
            payload,
            {
                "artifact_token",
                "content_sha256",
                "byte_size",
                "source_kind",
            },
            "scoring artifact entry",
        )
        return cls(
            artifact_token=payload["artifact_token"],
            content_sha256=payload["content_sha256"],
            byte_size=payload["byte_size"],
            source_kind=ArtifactSourceKind(payload["source_kind"]),
        )


@dataclass(frozen=True, slots=True)
class ControlScoringInventory:
    plan_sha256: str
    scoring_requests_sha256: str
    base_artifact_manifest_sha256: str
    base_artifact_verification_sha256: str
    control_transform_receipt_sha256: str
    entries: tuple[ScoringArtifactEntry, ...]
    schema_version: str = "cvi.control_scoring_inventory.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.control_scoring_inventory.v1":
            raise ValueError("unsupported control scoring inventory schema")
        for name in (
            "plan_sha256",
            "scoring_requests_sha256",
            "base_artifact_manifest_sha256",
            "base_artifact_verification_sha256",
            "control_transform_receipt_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if not self.entries:
            raise ValueError("control scoring inventory must not be empty")
        tokens = tuple(entry.artifact_token for entry in self.entries)
        if len(tokens) != len(set(tokens)):
            raise ValueError("scoring artifact tokens must be unique")

    @property
    def inventory_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "scoring_requests_sha256": self.scoring_requests_sha256,
            "base_artifact_manifest_sha256": (
                self.base_artifact_manifest_sha256
            ),
            "base_artifact_verification_sha256": (
                self.base_artifact_verification_sha256
            ),
            "control_transform_receipt_sha256": (
                self.control_transform_receipt_sha256
            ),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlScoringInventory:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "plan_sha256",
                "scoring_requests_sha256",
                "base_artifact_manifest_sha256",
                "base_artifact_verification_sha256",
                "control_transform_receipt_sha256",
                "entries",
            },
            "control scoring inventory",
        )
        entries = payload["entries"]
        if not isinstance(entries, list):
            raise TypeError("control scoring inventory entries must be a list")
        return cls(
            schema_version=payload["schema_version"],
            plan_sha256=payload["plan_sha256"],
            scoring_requests_sha256=payload[
                "scoring_requests_sha256"
            ],
            base_artifact_manifest_sha256=payload[
                "base_artifact_manifest_sha256"
            ],
            base_artifact_verification_sha256=payload[
                "base_artifact_verification_sha256"
            ],
            control_transform_receipt_sha256=payload[
                "control_transform_receipt_sha256"
            ],
            entries=tuple(
                ScoringArtifactEntry.from_dict(item) for item in entries
            ),
        )


def control_scoring_requests_from_payload(
    payload: dict[str, Any],
) -> tuple[str, tuple[ControlScoringRequest, ...]]:
    _require_exact_keys(
        payload,
        {"schema_version", "plan_sha256", "requests"},
        "control scoring requests",
    )
    if payload["schema_version"] != "cvi.visual_control_scoring_requests.v1":
        raise ValueError("unsupported control scoring request schema")
    _validate_sha256(payload["plan_sha256"], "plan_sha256")
    requests = payload["requests"]
    if not isinstance(requests, list):
        raise TypeError("control scoring requests must be a list")
    parsed = tuple(ControlScoringRequest.from_dict(item) for item in requests)
    _validate_request_set(parsed)
    return payload["plan_sha256"], parsed


def build_control_scoring_inventory(
    *,
    plan_sha256: str,
    requests: tuple[ControlScoringRequest, ...],
    base_root: Path,
    base_manifest: PairArtifactManifest,
    base_verification: PairArtifactVerification,
    control_root: Path,
    transform_receipt: ControlTransformReceipt,
) -> ControlScoringInventory:
    """Rehash both source directories and select the exact scoring token set."""

    _validate_sha256(plan_sha256, "plan_sha256")
    _validate_request_set(requests)
    if transform_receipt.plan_sha256 != plan_sha256:
        raise ValueError("transform receipt belongs to another control plan")
    scoring_requests_sha256 = content_sha256(
        {
            "schema_version": "cvi.visual_control_scoring_requests.v1",
            "plan_sha256": plan_sha256,
            "requests": [request.to_dict() for request in requests],
        }
    )
    if (
        transform_receipt.scoring_requests_sha256
        != scoring_requests_sha256
    ):
        raise ValueError(
            "scoring requests differ from transform receipt binding"
        )
    if (
        transform_receipt.base_artifact_manifest_sha256
        != base_manifest.manifest_sha256
        or transform_receipt.base_artifact_verification_sha256
        != content_sha256(base_verification.to_dict())
    ):
        raise ValueError("transform receipt and base artifacts differ")
    current_base = verify_pair_artifact_files(base_root, base_manifest)
    if current_base != base_verification:
        raise ValueError("base artifacts changed before scoring inventory")
    current_control = verify_control_artifact_files(
        control_root,
        transform_receipt.artifact_manifest,
    )
    if current_control != transform_receipt.verification:
        raise ValueError("control artifacts changed before scoring inventory")

    base_entries = {
        entry.artifact_token: ScoringArtifactEntry(
            entry.artifact_token,
            entry.content_sha256,
            entry.byte_size,
            ArtifactSourceKind.BASE,
        )
        for entry in base_manifest.entries
    }
    control_entries = {
        entry.artifact_token: ScoringArtifactEntry(
            entry.artifact_token,
            entry.content_sha256,
            entry.byte_size,
            ArtifactSourceKind.CONTROL,
        )
        for entry in transform_receipt.artifact_manifest.entries
    }
    overlap = set(base_entries) & set(control_entries)
    if overlap:
        raise ValueError("base and control artifact tokens collide")
    available = base_entries | control_entries
    required = {
        token
        for request in requests
        for token in (
            request.query_artifact_token,
            request.reference_artifact_token,
        )
    }
    missing = required - set(available)
    if missing:
        raise ValueError(
            f"scoring requests reference missing artifacts: {sorted(missing)}"
        )
    unused_control = set(control_entries) - required
    if unused_control:
        raise ValueError(
            "transform receipt contains unrequested control artifacts"
        )
    return ControlScoringInventory(
        plan_sha256=plan_sha256,
        scoring_requests_sha256=scoring_requests_sha256,
        base_artifact_manifest_sha256=base_manifest.manifest_sha256,
        base_artifact_verification_sha256=content_sha256(
            base_verification.to_dict()
        ),
        control_transform_receipt_sha256=(
            transform_receipt.receipt_sha256
        ),
        entries=tuple(available[token] for token in sorted(required)),
    )


@dataclass(frozen=True, slots=True)
class ArtifactCacheBinding:
    artifact_token: str
    artifact_content_sha256: str
    cache_key: str

    def __post_init__(self) -> None:
        _require_safe_token(self.artifact_token, "artifact_token")
        _validate_sha256(
            self.artifact_content_sha256,
            "artifact_content_sha256",
        )
        _validate_sha256(self.cache_key, "cache_key")

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_token": self.artifact_token,
            "artifact_content_sha256": self.artifact_content_sha256,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactCacheBinding:
        _require_exact_keys(
            payload,
            {
                "artifact_token",
                "artifact_content_sha256",
                "cache_key",
            },
            "artifact cache binding",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EmbeddingCacheEntry:
    cache_key: str
    relative_path: str
    content_sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        _validate_sha256(self.cache_key, "cache_key")
        expected = f"{self.cache_key}.f32le"
        if (
            not isinstance(self.relative_path, str)
            or PurePosixPath(self.relative_path).name != self.relative_path
            or self.relative_path != expected
        ):
            raise ValueError(
                "embedding cache path must be <cache_key>.f32le"
            )
        _validate_sha256(self.content_sha256, "content_sha256")
        _require_positive_int(self.byte_size, "byte_size")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "cache_key": self.cache_key,
            "relative_path": self.relative_path,
            "content_sha256": self.content_sha256,
            "byte_size": self.byte_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbeddingCacheEntry:
        _require_exact_keys(
            payload,
            {
                "cache_key",
                "relative_path",
                "content_sha256",
                "byte_size",
            },
            "embedding cache entry",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EmbeddingCacheManifest:
    scoring_inventory_sha256: str
    model_sha256: str
    inference_config_sha256: str
    dependency_lock_sha256: str
    code_revision: str
    precision: str
    vector_dimension: int
    normalization_tolerance: float
    bindings: tuple[ArtifactCacheBinding, ...]
    entries: tuple[EmbeddingCacheEntry, ...]
    vector_format: str = "float32_le"
    schema_version: str = "cvi.embedding_cache_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_cache_manifest.v1":
            raise ValueError("unsupported embedding cache manifest schema")
        for name in (
            "scoring_inventory_sha256",
            "model_sha256",
            "inference_config_sha256",
            "dependency_lock_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if not isinstance(self.precision, str) or not self.precision.strip():
            raise ValueError("precision must be non-empty")
        if (
            not isinstance(self.code_revision, str)
            or not self.code_revision.strip()
        ):
            raise ValueError("code_revision must be non-empty")
        _require_positive_int(self.vector_dimension, "vector_dimension")
        _require_finite_positive(
            self.normalization_tolerance,
            "normalization_tolerance",
        )
        if self.vector_format != "float32_le":
            raise ValueError("embedding vector format is fixed to float32_le")
        if not self.bindings or not self.entries:
            raise ValueError("embedding cache manifest must not be empty")
        tokens = tuple(binding.artifact_token for binding in self.bindings)
        if len(tokens) != len(set(tokens)):
            raise ValueError("embedding artifact bindings must be unique")
        entry_keys = tuple(entry.cache_key for entry in self.entries)
        if len(entry_keys) != len(set(entry_keys)):
            raise ValueError("embedding cache keys must be unique")
        if {binding.cache_key for binding in self.bindings} != set(entry_keys):
            raise ValueError(
                "embedding cache entries and bindings must match exactly"
            )
        expected_bytes = self.vector_dimension * 4
        if any(entry.byte_size != expected_bytes for entry in self.entries):
            raise ValueError("embedding cache byte size differs from dimension")
        for binding in self.bindings:
            expected_key = embedding_cache_key(
                artifact_content_sha256=(
                    binding.artifact_content_sha256
                ),
                model_sha256=self.model_sha256,
                inference_config_sha256=self.inference_config_sha256,
                dependency_lock_sha256=self.dependency_lock_sha256,
                code_revision=self.code_revision,
                precision=self.precision,
                vector_dimension=self.vector_dimension,
                vector_format=self.vector_format,
            )
            if binding.cache_key != expected_key:
                raise ValueError("embedding cache key provenance mismatch")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

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
            "bindings": [binding.to_dict() for binding in self.bindings],
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbeddingCacheManifest:
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
                "bindings",
                "entries",
            },
            "embedding cache manifest",
        )
        bindings = payload["bindings"]
        entries = payload["entries"]
        if not isinstance(bindings, list) or not isinstance(entries, list):
            raise TypeError(
                "embedding cache bindings and entries must be lists"
            )
        return cls(
            schema_version=payload["schema_version"],
            scoring_inventory_sha256=payload[
                "scoring_inventory_sha256"
            ],
            model_sha256=payload["model_sha256"],
            inference_config_sha256=payload[
                "inference_config_sha256"
            ],
            dependency_lock_sha256=payload["dependency_lock_sha256"],
            code_revision=payload["code_revision"],
            precision=payload["precision"],
            vector_dimension=payload["vector_dimension"],
            normalization_tolerance=payload[
                "normalization_tolerance"
            ],
            vector_format=payload["vector_format"],
            bindings=tuple(
                ArtifactCacheBinding.from_dict(item) for item in bindings
            ),
            entries=tuple(
                EmbeddingCacheEntry.from_dict(item) for item in entries
            ),
        )


def embedding_cache_key(
    *,
    artifact_content_sha256: str,
    model_sha256: str,
    inference_config_sha256: str,
    dependency_lock_sha256: str,
    code_revision: str,
    precision: str,
    vector_dimension: int,
    vector_format: str = "float32_le",
) -> str:
    for name, value in (
        ("artifact_content_sha256", artifact_content_sha256),
        ("model_sha256", model_sha256),
        ("inference_config_sha256", inference_config_sha256),
        ("dependency_lock_sha256", dependency_lock_sha256),
    ):
        _validate_sha256(value, name)
    if not isinstance(precision, str) or not precision.strip():
        raise ValueError("precision must be non-empty")
    if not isinstance(code_revision, str) or not code_revision.strip():
        raise ValueError("code_revision must be non-empty")
    _require_positive_int(vector_dimension, "vector_dimension")
    if vector_format != "float32_le":
        raise ValueError("embedding vector format is fixed to float32_le")
    return content_sha256(
        {
            "artifact_content_sha256": artifact_content_sha256,
            "model_sha256": model_sha256,
            "inference_config_sha256": inference_config_sha256,
            "dependency_lock_sha256": dependency_lock_sha256,
            "code_revision": code_revision,
            "precision": precision,
            "vector_dimension": vector_dimension,
            "vector_format": vector_format,
            "l2_normalized": True,
        }
    )


@dataclass(frozen=True, slots=True)
class EmbeddingCachePolicy:
    maximum_artifacts: int = 100_000
    maximum_unique_vectors: int = 100_000
    maximum_vector_dimension: int = 65_536
    maximum_vector_bytes: int = 262_144
    maximum_total_cache_bytes: int = 8_589_934_592
    scan_chunk_floats: int = 4_096
    maximum_normalization_tolerance: float = 0.001
    schema_version: str = "cvi.embedding_cache_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_cache_policy.v1":
            raise ValueError("unsupported embedding cache policy schema")
        for name in (
            "maximum_artifacts",
            "maximum_unique_vectors",
            "maximum_vector_dimension",
            "maximum_vector_bytes",
            "maximum_total_cache_bytes",
            "scan_chunk_floats",
        ):
            _require_positive_int(getattr(self, name), name)
        _require_finite_positive(
            self.maximum_normalization_tolerance,
            "maximum_normalization_tolerance",
        )

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "schema_version": self.schema_version,
            "maximum_artifacts": self.maximum_artifacts,
            "maximum_unique_vectors": self.maximum_unique_vectors,
            "maximum_vector_dimension": self.maximum_vector_dimension,
            "maximum_vector_bytes": self.maximum_vector_bytes,
            "maximum_total_cache_bytes": self.maximum_total_cache_bytes,
            "scan_chunk_floats": self.scan_chunk_floats,
            "maximum_normalization_tolerance": (
                self.maximum_normalization_tolerance
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbeddingCachePolicy:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "maximum_artifacts",
                "maximum_unique_vectors",
                "maximum_vector_dimension",
                "maximum_vector_bytes",
                "maximum_total_cache_bytes",
                "scan_chunk_floats",
                "maximum_normalization_tolerance",
            },
            "embedding cache policy",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EmbeddingCacheVerification:
    cache_manifest_sha256: str
    cache_policy_sha256: str
    verified_files: int
    verified_bytes: int
    verified_vectors: int
    maximum_observed_norm_error: float
    schema_version: str = "cvi.embedding_cache_verification.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_cache_verification.v1":
            raise ValueError(
                "unsupported embedding cache verification schema"
            )
        _validate_sha256(
            self.cache_manifest_sha256,
            "cache_manifest_sha256",
        )
        _validate_sha256(self.cache_policy_sha256, "cache_policy_sha256")
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
            "cache_policy_sha256": self.cache_policy_sha256,
            "verified_files": self.verified_files,
            "verified_bytes": self.verified_bytes,
            "verified_vectors": self.verified_vectors,
            "maximum_observed_norm_error": (
                self.maximum_observed_norm_error
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> EmbeddingCacheVerification:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "cache_manifest_sha256",
                "cache_policy_sha256",
                "verified_files",
                "verified_bytes",
                "verified_vectors",
                "maximum_observed_norm_error",
            },
            "embedding cache verification",
        )
        return cls(**payload)


def verify_embedding_cache_files(
    *,
    root: Path,
    inventory: ControlScoringInventory,
    manifest: EmbeddingCacheManifest,
    policy: EmbeddingCachePolicy,
) -> EmbeddingCacheVerification:
    if manifest.scoring_inventory_sha256 != inventory.inventory_sha256:
        raise ValueError("embedding cache belongs to another inventory")
    inventory_by_token = {
        entry.artifact_token: entry for entry in inventory.entries
    }
    binding_by_token = {
        binding.artifact_token: binding for binding in manifest.bindings
    }
    if set(binding_by_token) != set(inventory_by_token):
        raise ValueError("embedding bindings do not cover inventory exactly")
    for token, inventory_entry in inventory_by_token.items():
        binding = binding_by_token[token]
        if (
            binding.artifact_content_sha256
            != inventory_entry.content_sha256
        ):
            raise ValueError("embedding binding artifact content mismatch")
    if len(manifest.bindings) > policy.maximum_artifacts:
        raise ValueError("embedding bindings exceed maximum_artifacts")
    if len(manifest.entries) > policy.maximum_unique_vectors:
        raise ValueError("embedding cache exceeds maximum_unique_vectors")
    if manifest.vector_dimension > policy.maximum_vector_dimension:
        raise ValueError("embedding dimension exceeds policy")
    if manifest.normalization_tolerance > (
        policy.maximum_normalization_tolerance
    ):
        raise ValueError("embedding normalization tolerance exceeds policy")
    vector_bytes = manifest.vector_dimension * 4
    if vector_bytes > policy.maximum_vector_bytes:
        raise ValueError("embedding vector exceeds byte cap")
    total_bytes = vector_bytes * len(manifest.entries)
    if total_bytes > policy.maximum_total_cache_bytes:
        raise ValueError("embedding cache exceeds total byte cap")

    resolved = _real_directory(root, "embedding cache")
    directory_entries = tuple(resolved.iterdir())
    if any(entry.is_symlink() for entry in directory_entries):
        raise ValueError("embedding cache directory contains a symlink")
    if any(not entry.is_file() for entry in directory_entries):
        raise ValueError("embedding cache directory must contain files only")
    expected_names = {entry.relative_path for entry in manifest.entries}
    actual_names = {entry.name for entry in directory_entries}
    if expected_names != actual_names:
        raise ValueError("embedding cache directory is not a closed set")

    maximum_error = 0.0
    verified_bytes = 0
    for entry in manifest.entries:
        path = resolved / entry.relative_path
        initial = path.stat()
        if initial.st_size != entry.byte_size:
            raise ValueError("embedding cache byte-size mismatch")
        digest, norm_error = _scan_l2_vector_and_hash(
            path,
            dimension=manifest.vector_dimension,
            chunk_floats=policy.scan_chunk_floats,
        )
        final = path.stat()
        if (
            initial.st_size != final.st_size
            or initial.st_mtime_ns != final.st_mtime_ns
        ):
            raise RuntimeError(
                "embedding cache changed during verification"
            )
        if digest != entry.content_sha256:
            raise ValueError("embedding cache content hash mismatch")
        if norm_error > manifest.normalization_tolerance:
            raise ValueError("embedding vector is not L2-normalized")
        maximum_error = max(maximum_error, norm_error)
        verified_bytes += entry.byte_size
    return EmbeddingCacheVerification(
        cache_manifest_sha256=manifest.manifest_sha256,
        cache_policy_sha256=policy.policy_sha256,
        verified_files=len(manifest.entries),
        verified_bytes=verified_bytes,
        verified_vectors=len(manifest.entries),
        maximum_observed_norm_error=maximum_error,
    )


@dataclass(frozen=True, slots=True)
class ControlScorePolicy:
    maximum_requests: int = 1_000_000
    maximum_scalar_products: int = 1_000_000_000
    maximum_embedding_bytes_read: int = 8_589_934_592
    dot_chunk_floats: int = 4_096
    metric: str = "cosine_l2_dot"
    accumulation: str = "float64_chunk_fsum_neumaier"
    schema_version: str = "cvi.control_score_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.control_score_policy.v1":
            raise ValueError("unsupported control score policy schema")
        for name in (
            "maximum_requests",
            "maximum_scalar_products",
            "maximum_embedding_bytes_read",
            "dot_chunk_floats",
        ):
            _require_positive_int(getattr(self, name), name)
        if self.metric != "cosine_l2_dot":
            raise ValueError("control score metric is fixed to cosine_l2_dot")
        if self.accumulation != "float64_chunk_fsum_neumaier":
            raise ValueError("unsupported control score accumulation")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int]:
        return {
            "schema_version": self.schema_version,
            "maximum_requests": self.maximum_requests,
            "maximum_scalar_products": self.maximum_scalar_products,
            "maximum_embedding_bytes_read": (
                self.maximum_embedding_bytes_read
            ),
            "dot_chunk_floats": self.dot_chunk_floats,
            "metric": self.metric,
            "accumulation": self.accumulation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlScorePolicy:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "maximum_requests",
                "maximum_scalar_products",
                "maximum_embedding_bytes_read",
                "dot_chunk_floats",
                "metric",
                "accumulation",
            },
            "control score policy",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ControlBlindScore:
    request_id: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
        ):
            raise ValueError("control score must be finite")

    def to_dict(self) -> dict[str, str | float]:
        return {"request_id": self.request_id, "score": self.score}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlBlindScore:
        _require_exact_keys(
            payload,
            {"request_id", "score"},
            "control blind score",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ControlScoreCost:
    scoring_requests: int
    dot_product_scalar_products: int
    cache_verification_square_terms: int
    dot_product_bytes_read: int
    cache_verification_bytes_read: int
    total_file_bytes_read: int
    unique_artifacts: int
    unique_embedding_vectors: int
    neural_embedding_calls_saved: int
    peak_raw_chunk_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "scoring_requests",
            "dot_product_scalar_products",
            "cache_verification_square_terms",
            "dot_product_bytes_read",
            "cache_verification_bytes_read",
            "total_file_bytes_read",
            "unique_artifacts",
            "unique_embedding_vectors",
            "neural_embedding_calls_saved",
            "peak_raw_chunk_bytes",
        ):
            _require_nonnegative_int(getattr(self, name), name)

    def to_dict(self) -> dict[str, int]:
        return {
            "scoring_requests": self.scoring_requests,
            "dot_product_scalar_products": (
                self.dot_product_scalar_products
            ),
            "cache_verification_square_terms": (
                self.cache_verification_square_terms
            ),
            "dot_product_bytes_read": self.dot_product_bytes_read,
            "cache_verification_bytes_read": (
                self.cache_verification_bytes_read
            ),
            "total_file_bytes_read": self.total_file_bytes_read,
            "unique_artifacts": self.unique_artifacts,
            "unique_embedding_vectors": self.unique_embedding_vectors,
            "neural_embedding_calls_saved": (
                self.neural_embedding_calls_saved
            ),
            "peak_raw_chunk_bytes": self.peak_raw_chunk_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlScoreCost:
        _require_exact_keys(
            payload,
            {
                "scoring_requests",
                "dot_product_scalar_products",
                "cache_verification_square_terms",
                "dot_product_bytes_read",
                "cache_verification_bytes_read",
                "total_file_bytes_read",
                "unique_artifacts",
                "unique_embedding_vectors",
                "neural_embedding_calls_saved",
                "peak_raw_chunk_bytes",
            },
            "control score cost",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ControlBlindScoreReceipt:
    plan_sha256: str
    scoring_requests_sha256: str
    scoring_inventory_sha256: str
    embedding_cache_manifest_sha256: str
    embedding_cache_verification_sha256: str
    gallery_sha256: str
    score_policy_sha256: str
    scores: tuple[ControlBlindScore, ...]
    cost: ControlScoreCost
    scorer_version: str = "cvi.cpu_cosine_reference.v1"
    device: str = "cpu"
    schema_version: str = "cvi.control_blind_score_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.control_blind_score_receipt.v1":
            raise ValueError("unsupported control score receipt schema")
        for name in (
            "plan_sha256",
            "scoring_requests_sha256",
            "scoring_inventory_sha256",
            "embedding_cache_manifest_sha256",
            "embedding_cache_verification_sha256",
            "gallery_sha256",
            "score_policy_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if self.scorer_version != "cvi.cpu_cosine_reference.v1":
            raise ValueError("unsupported control reference scorer")
        if self.device != "cpu":
            raise ValueError("reference control scorer device is fixed to cpu")
        if not self.scores:
            raise ValueError("control score receipt must contain scores")
        ids = tuple(score.request_id for score in self.scores)
        if len(ids) != len(set(ids)):
            raise ValueError("control score request IDs must be unique")
        if self.cost.scoring_requests != len(self.scores):
            raise ValueError("control score cost/request count mismatch")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "scoring_requests_sha256": self.scoring_requests_sha256,
            "scoring_inventory_sha256": self.scoring_inventory_sha256,
            "embedding_cache_manifest_sha256": (
                self.embedding_cache_manifest_sha256
            ),
            "embedding_cache_verification_sha256": (
                self.embedding_cache_verification_sha256
            ),
            "gallery_sha256": self.gallery_sha256,
            "score_policy_sha256": self.score_policy_sha256,
            "scorer_version": self.scorer_version,
            "device": self.device,
            "scores": [score.to_dict() for score in self.scores],
            "cost": self.cost.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> ControlBlindScoreReceipt:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "plan_sha256",
                "scoring_requests_sha256",
                "scoring_inventory_sha256",
                "embedding_cache_manifest_sha256",
                "embedding_cache_verification_sha256",
                "gallery_sha256",
                "score_policy_sha256",
                "scorer_version",
                "device",
                "scores",
                "cost",
            },
            "control blind score receipt",
        )
        scores = payload["scores"]
        if not isinstance(scores, list):
            raise TypeError("control blind scores must be a list")
        return cls(
            schema_version=payload["schema_version"],
            plan_sha256=payload["plan_sha256"],
            scoring_requests_sha256=payload[
                "scoring_requests_sha256"
            ],
            scoring_inventory_sha256=payload[
                "scoring_inventory_sha256"
            ],
            embedding_cache_manifest_sha256=payload[
                "embedding_cache_manifest_sha256"
            ],
            embedding_cache_verification_sha256=payload[
                "embedding_cache_verification_sha256"
            ],
            gallery_sha256=payload["gallery_sha256"],
            score_policy_sha256=payload["score_policy_sha256"],
            scorer_version=payload["scorer_version"],
            device=payload["device"],
            scores=tuple(ControlBlindScore.from_dict(item) for item in scores),
            cost=ControlScoreCost.from_dict(payload["cost"]),
        )


def score_control_requests_from_cache(
    *,
    requests: tuple[ControlScoringRequest, ...],
    inventory: ControlScoringInventory,
    cache_root: Path,
    cache_manifest: EmbeddingCacheManifest,
    cache_verification: EmbeddingCacheVerification,
    cache_policy: EmbeddingCachePolicy,
    score_policy: ControlScorePolicy,
    gallery_sha256: str,
) -> ControlBlindScoreReceipt:
    """Score opaque requests from a verified content-addressed cache."""

    _validate_request_set(requests)
    _validate_sha256(gallery_sha256, "gallery_sha256")
    expected_request_hash = content_sha256(
        {
            "schema_version": "cvi.visual_control_scoring_requests.v1",
            "plan_sha256": inventory.plan_sha256,
            "requests": [request.to_dict() for request in requests],
        }
    )
    if expected_request_hash != inventory.scoring_requests_sha256:
        raise ValueError("scoring requests differ from inventory")
    current_cache = verify_embedding_cache_files(
        root=cache_root,
        inventory=inventory,
        manifest=cache_manifest,
        policy=cache_policy,
    )
    if current_cache != cache_verification:
        raise ValueError("embedding cache changed before scoring")
    request_tokens = {
        token
        for request in requests
        for token in (
            request.query_artifact_token,
            request.reference_artifact_token,
        )
    }
    inventory_tokens = {entry.artifact_token for entry in inventory.entries}
    if request_tokens != inventory_tokens:
        raise ValueError("scoring request tokens differ from inventory")
    if len(requests) > score_policy.maximum_requests:
        raise ValueError("control score requests exceed policy")
    scalar_products = len(requests) * cache_manifest.vector_dimension
    if scalar_products > score_policy.maximum_scalar_products:
        raise ValueError("control scalar-product work exceeds policy")
    bytes_read = scalar_products * 2 * 4
    if bytes_read > score_policy.maximum_embedding_bytes_read:
        raise ValueError("control embedding reads exceed policy")

    binding_by_token = {
        binding.artifact_token: binding
        for binding in cache_manifest.bindings
    }
    entry_by_key = {
        entry.cache_key: entry for entry in cache_manifest.entries
    }
    resolved_root = _real_directory(cache_root, "embedding cache")
    scores: list[ControlBlindScore] = []
    for request in requests:
        query_entry = entry_by_key[
            binding_by_token[request.query_artifact_token].cache_key
        ]
        reference_entry = entry_by_key[
            binding_by_token[request.reference_artifact_token].cache_key
        ]
        score = _stream_dot_product(
            resolved_root / query_entry.relative_path,
            resolved_root / reference_entry.relative_path,
            dimension=cache_manifest.vector_dimension,
            chunk_floats=score_policy.dot_chunk_floats,
        )
        scores.append(ControlBlindScore(request.request_id, score))
    if (
        verify_embedding_cache_files(
            root=cache_root,
            inventory=inventory,
            manifest=cache_manifest,
            policy=cache_policy,
        )
        != cache_verification
    ):
        raise RuntimeError("embedding cache changed during scoring")
    unique_artifacts = len(inventory.entries)
    unique_vectors = len(cache_manifest.entries)
    cache_verification_bytes = 2 * cache_verification.verified_bytes
    cache_verification_terms = (
        2 * unique_vectors * cache_manifest.vector_dimension
    )
    return ControlBlindScoreReceipt(
        plan_sha256=inventory.plan_sha256,
        scoring_requests_sha256=inventory.scoring_requests_sha256,
        scoring_inventory_sha256=inventory.inventory_sha256,
        embedding_cache_manifest_sha256=cache_manifest.manifest_sha256,
        embedding_cache_verification_sha256=(
            cache_verification.verification_sha256
        ),
        gallery_sha256=gallery_sha256,
        score_policy_sha256=score_policy.policy_sha256,
        scores=tuple(scores),
        cost=ControlScoreCost(
            scoring_requests=len(requests),
            dot_product_scalar_products=scalar_products,
            cache_verification_square_terms=cache_verification_terms,
            dot_product_bytes_read=bytes_read,
            cache_verification_bytes_read=cache_verification_bytes,
            total_file_bytes_read=bytes_read + cache_verification_bytes,
            unique_artifacts=unique_artifacts,
            unique_embedding_vectors=unique_vectors,
            neural_embedding_calls_saved=max(
                0,
                2 * len(requests) - unique_vectors,
            ),
            peak_raw_chunk_bytes=(
                2 * min(
                    score_policy.dot_chunk_floats,
                    cache_manifest.vector_dimension,
                )
                * 4
            ),
        ),
    )


def _scan_l2_vector_and_hash(
    path: Path,
    *,
    dimension: int,
    chunk_floats: int,
) -> tuple[str, float]:
    remaining = dimension
    total = 0.0
    compensation = 0.0
    digest = sha256()
    with path.open("rb") as handle:
        while remaining:
            count = min(remaining, chunk_floats)
            data = _read_exact(handle, count * 4)
            digest.update(data)
            subtotal = math.fsum(_finite_squares(data))
            total, compensation = _neumaier_add(
                total,
                compensation,
                subtotal,
            )
            remaining -= count
        if handle.read(1):
            raise ValueError("embedding vector has trailing bytes")
    norm = math.sqrt(total + compensation)
    return digest.hexdigest(), abs(norm - 1.0)


def _finite_squares(data: bytes) -> Iterator[float]:
    for (value,) in struct.iter_unpack("<f", data):
        if not math.isfinite(value):
            raise ValueError("embedding vector contains non-finite value")
        yield value * value


def _stream_dot_product(
    first: Path,
    second: Path,
    *,
    dimension: int,
    chunk_floats: int,
) -> float:
    remaining = dimension
    total = 0.0
    compensation = 0.0
    with first.open("rb") as first_handle, second.open("rb") as second_handle:
        while remaining:
            count = min(remaining, chunk_floats)
            first_data = _read_exact(first_handle, count * 4)
            second_data = _read_exact(second_handle, count * 4)
            first_values = struct.iter_unpack("<f", first_data)
            second_values = struct.iter_unpack("<f", second_data)
            subtotal = math.fsum(
                left[0] * right[0]
                for left, right in zip(
                    first_values,
                    second_values,
                    strict=True,
                )
            )
            total, compensation = _neumaier_add(
                total,
                compensation,
                subtotal,
            )
            remaining -= count
        if first_handle.read(1) or second_handle.read(1):
            raise ValueError("embedding vector has trailing bytes")
    result = total + compensation
    if not math.isfinite(result):
        raise ValueError("control dot product is non-finite")
    return result


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


def _validate_request_set(
    requests: tuple[ControlScoringRequest, ...],
) -> None:
    if not requests:
        raise ValueError("control scoring requests must not be empty")
    ids = tuple(request.request_id for request in requests)
    if len(ids) != len(set(ids)):
        raise ValueError("control scoring request IDs must be unique")


def _real_directory(path: Path, context: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{context} root must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("embedding vector byte count is incomplete")
    return data


def _require_safe_token(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value):
        raise ValueError(f"{name} is not a safe token")


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


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


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
