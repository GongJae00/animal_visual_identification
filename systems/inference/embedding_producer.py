"""Frozen, label-blind neural embedding production contract."""

from __future__ import annotations

import math
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Any, Callable, Protocol

from data.acquisition import sha256_file
from evaluation.benchmark import TimingSummary
from evaluation.controls.control_scoring import (
    ArtifactCacheBinding,
    ControlScoringInventory,
    EmbeddingCacheEntry,
    EmbeddingCacheManifest,
    EmbeddingCachePolicy,
    EmbeddingCacheVerification,
    embedding_cache_key,
    verify_embedding_cache_files,
)
from foundation.provenance import content_sha256


@dataclass(frozen=True, slots=True)
class EmbeddingBackendIdentity:
    backend_name: str
    backend_version: str
    runtime_version: str
    execution_provider: str
    device: str
    precision: str
    determinism_mode: str
    backend_config_sha256: str
    schema_version: str = "cvi.embedding_backend_identity.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_backend_identity.v1":
            raise ValueError("unsupported embedding backend identity schema")
        for name in (
            "backend_name",
            "backend_version",
            "runtime_version",
            "execution_provider",
            "device",
            "precision",
            "determinism_mode",
        ):
            _require_nonempty(getattr(self, name), name)
        _validate_sha256(
            self.backend_config_sha256,
            "backend_config_sha256",
        )

    @property
    def identity_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "runtime_version": self.runtime_version,
            "execution_provider": self.execution_provider,
            "device": self.device,
            "precision": self.precision,
            "determinism_mode": self.determinism_mode,
            "backend_config_sha256": self.backend_config_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> EmbeddingBackendIdentity:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "backend_name",
                "backend_version",
                "runtime_version",
                "execution_provider",
                "device",
                "precision",
                "determinism_mode",
                "backend_config_sha256",
            },
            "embedding backend identity",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EmbeddingProducerConfig:
    model_sha256: str
    model_lineage_sha256: str
    preprocessing_sha256: str
    preprocessing_semantics_sha256: str
    dependency_lock_sha256: str
    code_revision: str
    backend: EmbeddingBackendIdentity
    vector_dimension: int
    batch_size: int
    input_width: int
    input_height: int
    input_channels: int
    input_value_bytes: int
    l2_epsilon: float
    normalization_tolerance: float
    warmup_batches: int = 0
    output_vector_format: str = "float32_le"
    schema_version: str = "cvi.embedding_producer_config.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_producer_config.v1":
            raise ValueError("unsupported embedding producer config schema")
        for name in (
            "model_sha256",
            "model_lineage_sha256",
            "preprocessing_sha256",
            "preprocessing_semantics_sha256",
            "dependency_lock_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        _require_nonempty(self.code_revision, "code_revision")
        _require_positive_int(self.vector_dimension, "vector_dimension")
        _require_positive_int(self.batch_size, "batch_size")
        _require_positive_int(self.input_width, "input_width")
        _require_positive_int(self.input_height, "input_height")
        _require_positive_int(self.input_channels, "input_channels")
        _require_positive_int(self.input_value_bytes, "input_value_bytes")
        _require_finite_positive(self.l2_epsilon, "l2_epsilon")
        _require_finite_positive(
            self.normalization_tolerance,
            "normalization_tolerance",
        )
        _require_nonnegative_int(self.warmup_batches, "warmup_batches")
        if self.output_vector_format != "float32_le":
            raise ValueError("embedding output format is fixed to float32_le")

    @property
    def config_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_sha256": self.model_sha256,
            "model_lineage_sha256": self.model_lineage_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "preprocessing_semantics_sha256": (
                self.preprocessing_semantics_sha256
            ),
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "code_revision": self.code_revision,
            "backend": self.backend.to_dict(),
            "vector_dimension": self.vector_dimension,
            "batch_size": self.batch_size,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "input_channels": self.input_channels,
            "input_value_bytes": self.input_value_bytes,
            "l2_epsilon": self.l2_epsilon,
            "normalization_tolerance": self.normalization_tolerance,
            "warmup_batches": self.warmup_batches,
            "output_vector_format": self.output_vector_format,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbeddingProducerConfig:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "model_sha256",
                "model_lineage_sha256",
                "preprocessing_sha256",
                "preprocessing_semantics_sha256",
                "dependency_lock_sha256",
                "code_revision",
                "backend",
                "vector_dimension",
                "batch_size",
                "input_width",
                "input_height",
                "input_channels",
                "input_value_bytes",
                "l2_epsilon",
                "normalization_tolerance",
                "warmup_batches",
                "output_vector_format",
            },
            "embedding producer config",
        )
        backend = payload["backend"]
        if not isinstance(backend, dict):
            raise TypeError("embedding producer backend must be an object")
        return cls(
            schema_version=payload["schema_version"],
            model_sha256=payload["model_sha256"],
            model_lineage_sha256=payload["model_lineage_sha256"],
            preprocessing_sha256=payload["preprocessing_sha256"],
            preprocessing_semantics_sha256=payload[
                "preprocessing_semantics_sha256"
            ],
            dependency_lock_sha256=payload["dependency_lock_sha256"],
            code_revision=payload["code_revision"],
            backend=EmbeddingBackendIdentity.from_dict(backend),
            vector_dimension=payload["vector_dimension"],
            batch_size=payload["batch_size"],
            input_width=payload["input_width"],
            input_height=payload["input_height"],
            input_channels=payload["input_channels"],
            input_value_bytes=payload["input_value_bytes"],
            l2_epsilon=payload["l2_epsilon"],
            normalization_tolerance=payload["normalization_tolerance"],
            warmup_batches=payload["warmup_batches"],
            output_vector_format=payload["output_vector_format"],
        )


@dataclass(frozen=True, slots=True)
class EmbeddingProductionPolicy:
    maximum_artifacts: int = 100_000
    maximum_unique_inputs: int = 100_000
    maximum_batches: int = 100_000
    maximum_batch_size: int = 1_024
    maximum_vector_dimension: int = 65_536
    maximum_input_bytes_per_batch: int = 2_147_483_648
    maximum_nominal_tensor_bytes_per_batch: int = 4_294_967_296
    maximum_output_bytes_per_batch: int = 268_435_456
    maximum_total_input_bytes: int = 1_099_511_627_776
    maximum_total_output_bytes: int = 8_589_934_592
    maximum_model_bytes: int = 17_179_869_184
    maximum_provenance_file_bytes: int = 1_073_741_824
    maximum_batch_wall_time_ns: int = 3_600_000_000_000
    maximum_total_wall_time_ns: int = 86_400_000_000_000
    schema_version: str = "cvi.embedding_production_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_production_policy.v1":
            raise ValueError("unsupported embedding production policy schema")
        for name in (
            "maximum_artifacts",
            "maximum_unique_inputs",
            "maximum_batches",
            "maximum_batch_size",
            "maximum_vector_dimension",
            "maximum_input_bytes_per_batch",
            "maximum_nominal_tensor_bytes_per_batch",
            "maximum_output_bytes_per_batch",
            "maximum_total_input_bytes",
            "maximum_total_output_bytes",
            "maximum_model_bytes",
            "maximum_provenance_file_bytes",
            "maximum_batch_wall_time_ns",
            "maximum_total_wall_time_ns",
        ):
            _require_positive_int(getattr(self, name), name)

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int]:
        return {
            "schema_version": self.schema_version,
            "maximum_artifacts": self.maximum_artifacts,
            "maximum_unique_inputs": self.maximum_unique_inputs,
            "maximum_batches": self.maximum_batches,
            "maximum_batch_size": self.maximum_batch_size,
            "maximum_vector_dimension": self.maximum_vector_dimension,
            "maximum_input_bytes_per_batch": (
                self.maximum_input_bytes_per_batch
            ),
            "maximum_nominal_tensor_bytes_per_batch": (
                self.maximum_nominal_tensor_bytes_per_batch
            ),
            "maximum_output_bytes_per_batch": (
                self.maximum_output_bytes_per_batch
            ),
            "maximum_total_input_bytes": self.maximum_total_input_bytes,
            "maximum_total_output_bytes": self.maximum_total_output_bytes,
            "maximum_model_bytes": self.maximum_model_bytes,
            "maximum_provenance_file_bytes": (
                self.maximum_provenance_file_bytes
            ),
            "maximum_batch_wall_time_ns": (
                self.maximum_batch_wall_time_ns
            ),
            "maximum_total_wall_time_ns": (
                self.maximum_total_wall_time_ns
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> EmbeddingProductionPolicy:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "maximum_artifacts",
                "maximum_unique_inputs",
                "maximum_batches",
                "maximum_batch_size",
                "maximum_vector_dimension",
                "maximum_input_bytes_per_batch",
                "maximum_nominal_tensor_bytes_per_batch",
                "maximum_output_bytes_per_batch",
                "maximum_total_input_bytes",
                "maximum_total_output_bytes",
                "maximum_model_bytes",
                "maximum_provenance_file_bytes",
                "maximum_batch_wall_time_ns",
                "maximum_total_wall_time_ns",
            },
            "embedding production policy",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EmbeddingRuntimeResources:
    cpu_peak_rss_bytes: int | None
    device_peak_memory_bytes: int | None
    energy_millijoules: float | None
    measurement_scope: str
    measurement_method: str
    schema_version: str = "cvi.embedding_runtime_resources.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_runtime_resources.v1":
            raise ValueError("unsupported embedding resource schema")
        for name in ("cpu_peak_rss_bytes", "device_peak_memory_bytes"):
            value = getattr(self, name)
            if value is not None:
                _require_nonnegative_int(value, name)
        if self.energy_millijoules is not None:
            _require_finite_nonnegative(
                self.energy_millijoules,
                "energy_millijoules",
            )
        _require_nonempty(self.measurement_scope, "measurement_scope")
        _require_nonempty(self.measurement_method, "measurement_method")
        all_unavailable = (
            self.cpu_peak_rss_bytes is None
            and self.device_peak_memory_bytes is None
            and self.energy_millijoules is None
        )
        if all_unavailable != (
            self.measurement_scope == "UNAVAILABLE"
            and self.measurement_method == "UNAVAILABLE"
        ):
            raise ValueError(
                "resource availability and measurement declaration differ"
            )

    @classmethod
    def unavailable(cls) -> EmbeddingRuntimeResources:
        return cls(
            cpu_peak_rss_bytes=None,
            device_peak_memory_bytes=None,
            energy_millijoules=None,
            measurement_scope="UNAVAILABLE",
            measurement_method="UNAVAILABLE",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cpu_peak_rss_bytes": self.cpu_peak_rss_bytes,
            "device_peak_memory_bytes": self.device_peak_memory_bytes,
            "energy_millijoules": self.energy_millijoules,
            "measurement_scope": self.measurement_scope,
            "measurement_method": self.measurement_method,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> EmbeddingRuntimeResources:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "cpu_peak_rss_bytes",
                "device_peak_memory_bytes",
                "energy_millijoules",
                "measurement_scope",
                "measurement_method",
            },
            "embedding runtime resources",
        )
        return cls(**payload)


class EmbeddingBatchBackend(Protocol):
    @property
    def identity(self) -> EmbeddingBackendIdentity:
        """Return the immutable backend identity used for this run."""

    @property
    def preprocessing_semantics_sha256(self) -> str:
        """Return the exact structured preprocessing-semantics hash."""

    @property
    def model_sha256(self) -> str:
        """Return the digest of the exact model bytes loaded by the backend."""

    def infer_batch(
        self,
        artifact_paths: tuple[Path, ...],
    ) -> Sequence[Sequence[float]]:
        """Return one raw embedding vector per input path in input order."""

    def synchronize(self) -> None:
        """Block until all backend work submitted so far is complete."""

    def runtime_resources(self) -> EmbeddingRuntimeResources:
        """Return measured resources or explicit UNAVAILABLE values."""


@dataclass(frozen=True, slots=True)
class EmbeddingProductionCost:
    artifact_bindings: int
    unique_content_inputs: int
    content_deduplication_calls_saved: int
    warmup_batches: int
    warmup_artifact_evaluations: int
    production_batches: int
    production_artifact_evaluations: int
    total_backend_artifact_evaluations: int
    warmup_wall_time_ns: int
    production_wall_time_ns: int
    total_backend_wall_time_ns: int
    input_integrity_hash_passes: int
    input_integrity_bytes_read: int
    provenance_integrity_hash_passes: int
    provenance_integrity_bytes_read: int
    output_float_values: int
    output_bytes_written: int
    peak_batch_artifacts: int
    peak_batch_input_bytes: int
    peak_nominal_input_tensor_bytes: int
    peak_batch_output_bytes: int
    schema_version: str = "cvi.embedding_production_cost.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_production_cost.v1":
            raise ValueError("unsupported embedding production cost schema")
        for name in (
            "artifact_bindings",
            "unique_content_inputs",
            "content_deduplication_calls_saved",
            "warmup_batches",
            "warmup_artifact_evaluations",
            "production_batches",
            "production_artifact_evaluations",
            "total_backend_artifact_evaluations",
            "warmup_wall_time_ns",
            "production_wall_time_ns",
            "total_backend_wall_time_ns",
            "input_integrity_hash_passes",
            "input_integrity_bytes_read",
            "provenance_integrity_hash_passes",
            "provenance_integrity_bytes_read",
            "output_float_values",
            "output_bytes_written",
            "peak_batch_artifacts",
            "peak_batch_input_bytes",
            "peak_nominal_input_tensor_bytes",
            "peak_batch_output_bytes",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        if (
            self.unique_content_inputs
            + self.content_deduplication_calls_saved
            != self.artifact_bindings
        ):
            raise ValueError("embedding production deduplication count mismatch")
        if self.total_backend_artifact_evaluations != (
            self.production_artifact_evaluations
            + self.warmup_artifact_evaluations
        ):
            raise ValueError("embedding production backend work mismatch")
        if self.total_backend_wall_time_ns != (
            self.warmup_wall_time_ns + self.production_wall_time_ns
        ):
            raise ValueError("embedding production wall-time mismatch")
        if self.output_bytes_written != self.output_float_values * 4:
            raise ValueError("embedding production output byte mismatch")
        if self.input_integrity_hash_passes != 2:
            raise ValueError("embedding inputs require exactly two hash passes")
        if self.provenance_integrity_hash_passes != 2:
            raise ValueError(
                "embedding provenance requires exactly two hash passes"
            )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "schema_version": self.schema_version,
            "artifact_bindings": self.artifact_bindings,
            "unique_content_inputs": self.unique_content_inputs,
            "content_deduplication_calls_saved": (
                self.content_deduplication_calls_saved
            ),
            "warmup_batches": self.warmup_batches,
            "warmup_artifact_evaluations": (
                self.warmup_artifact_evaluations
            ),
            "production_batches": self.production_batches,
            "production_artifact_evaluations": (
                self.production_artifact_evaluations
            ),
            "total_backend_artifact_evaluations": (
                self.total_backend_artifact_evaluations
            ),
            "warmup_wall_time_ns": self.warmup_wall_time_ns,
            "production_wall_time_ns": self.production_wall_time_ns,
            "total_backend_wall_time_ns": self.total_backend_wall_time_ns,
            "input_integrity_hash_passes": self.input_integrity_hash_passes,
            "input_integrity_bytes_read": self.input_integrity_bytes_read,
            "provenance_integrity_hash_passes": (
                self.provenance_integrity_hash_passes
            ),
            "provenance_integrity_bytes_read": (
                self.provenance_integrity_bytes_read
            ),
            "output_float_values": self.output_float_values,
            "output_bytes_written": self.output_bytes_written,
            "peak_batch_artifacts": self.peak_batch_artifacts,
            "peak_batch_input_bytes": self.peak_batch_input_bytes,
            "peak_nominal_input_tensor_bytes": (
                self.peak_nominal_input_tensor_bytes
            ),
            "peak_batch_output_bytes": self.peak_batch_output_bytes,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> EmbeddingProductionCost:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "artifact_bindings",
                "unique_content_inputs",
                "content_deduplication_calls_saved",
                "warmup_batches",
                "warmup_artifact_evaluations",
                "production_batches",
                "production_artifact_evaluations",
                "total_backend_artifact_evaluations",
                "warmup_wall_time_ns",
                "production_wall_time_ns",
                "total_backend_wall_time_ns",
                "input_integrity_hash_passes",
                "input_integrity_bytes_read",
                "provenance_integrity_hash_passes",
                "provenance_integrity_bytes_read",
                "output_float_values",
                "output_bytes_written",
                "peak_batch_artifacts",
                "peak_batch_input_bytes",
                "peak_nominal_input_tensor_bytes",
                "peak_batch_output_bytes",
            },
            "embedding production cost",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EmbeddingProductionReceipt:
    scoring_inventory_sha256: str
    producer_config_sha256: str
    production_policy_sha256: str
    cache_policy_sha256: str
    model_lineage_sha256: str
    cache_manifest: EmbeddingCacheManifest
    cache_verification: EmbeddingCacheVerification
    batch_timing: TimingSummary
    runtime_resources: EmbeddingRuntimeResources
    cost: EmbeddingProductionCost
    timing_interpretation: str = (
        "OBSERVATIONAL_CACHE_PRODUCTION_ONLY_NOT_PROMOTION_EVIDENCE"
    )
    schema_version: str = "cvi.embedding_production_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_production_receipt.v1":
            raise ValueError("unsupported embedding production receipt schema")
        for name in (
            "scoring_inventory_sha256",
            "producer_config_sha256",
            "production_policy_sha256",
            "cache_policy_sha256",
            "model_lineage_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if (
            self.timing_interpretation
            != "OBSERVATIONAL_CACHE_PRODUCTION_ONLY_NOT_PROMOTION_EVIDENCE"
        ):
            raise ValueError("embedding timing interpretation is fixed")
        if (
            self.cache_manifest.scoring_inventory_sha256
            != self.scoring_inventory_sha256
        ):
            raise ValueError("embedding receipt inventory binding mismatch")
        if (
            self.cache_verification.cache_manifest_sha256
            != self.cache_manifest.manifest_sha256
        ):
            raise ValueError("embedding receipt cache verification mismatch")
        if (
            self.cache_verification.cache_policy_sha256
            != self.cache_policy_sha256
        ):
            raise ValueError("embedding receipt cache policy mismatch")
        if self.cost.artifact_bindings != len(
            self.cache_manifest.bindings
        ):
            raise ValueError("embedding receipt binding count mismatch")
        if self.cost.unique_content_inputs != len(
            self.cache_manifest.entries
        ):
            raise ValueError("embedding receipt unique-input count mismatch")
        if self.cost.production_artifact_evaluations != (
            self.cost.unique_content_inputs
        ):
            raise ValueError("embedding receipt production work mismatch")
        if self.cost.total_backend_artifact_evaluations != (
            self.cost.production_artifact_evaluations
            + self.cost.warmup_artifact_evaluations
        ):
            raise ValueError("embedding receipt backend work mismatch")
        if self.cost.total_backend_wall_time_ns != (
            self.cost.warmup_wall_time_ns
            + self.cost.production_wall_time_ns
        ):
            raise ValueError("embedding receipt wall-time accounting mismatch")
        if self.batch_timing.samples != self.cost.production_batches:
            raise ValueError("embedding receipt timing sample count mismatch")
        if not math.isclose(
            self.batch_timing.mean_ns * self.batch_timing.samples,
            self.cost.production_wall_time_ns,
            rel_tol=0.0,
            abs_tol=0.5,
        ):
            raise ValueError("embedding receipt production timing mismatch")
        expected_output_bytes = sum(
            entry.byte_size for entry in self.cache_manifest.entries
        )
        if self.cost.output_bytes_written != expected_output_bytes:
            raise ValueError("embedding receipt output byte count mismatch")
        if self.cost.output_float_values != (
            self.cost.unique_content_inputs
            * self.cache_manifest.vector_dimension
        ):
            raise ValueError("embedding receipt output scalar count mismatch")
        if (
            self.cache_verification.verified_files
            != len(self.cache_manifest.entries)
            or self.cache_verification.verified_bytes
            != expected_output_bytes
        ):
            raise ValueError("embedding receipt verified cache count mismatch")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scoring_inventory_sha256": self.scoring_inventory_sha256,
            "producer_config_sha256": self.producer_config_sha256,
            "production_policy_sha256": self.production_policy_sha256,
            "cache_policy_sha256": self.cache_policy_sha256,
            "model_lineage_sha256": self.model_lineage_sha256,
            "cache_manifest": self.cache_manifest.to_dict(),
            "cache_verification": self.cache_verification.to_dict(),
            "batch_timing": self.batch_timing.to_dict(),
            "runtime_resources": self.runtime_resources.to_dict(),
            "cost": self.cost.to_dict(),
            "timing_interpretation": self.timing_interpretation,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> EmbeddingProductionReceipt:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "scoring_inventory_sha256",
                "producer_config_sha256",
                "production_policy_sha256",
                "cache_policy_sha256",
                "model_lineage_sha256",
                "cache_manifest",
                "cache_verification",
                "batch_timing",
                "runtime_resources",
                "cost",
                "timing_interpretation",
            },
            "embedding production receipt",
        )
        for name in (
            "cache_manifest",
            "cache_verification",
            "batch_timing",
            "runtime_resources",
            "cost",
        ):
            if not isinstance(payload[name], dict):
                raise TypeError(f"{name} must be an object")
        return cls(
            schema_version=payload["schema_version"],
            scoring_inventory_sha256=payload[
                "scoring_inventory_sha256"
            ],
            producer_config_sha256=payload["producer_config_sha256"],
            production_policy_sha256=payload[
                "production_policy_sha256"
            ],
            cache_policy_sha256=payload["cache_policy_sha256"],
            model_lineage_sha256=payload["model_lineage_sha256"],
            cache_manifest=EmbeddingCacheManifest.from_dict(
                payload["cache_manifest"]
            ),
            cache_verification=EmbeddingCacheVerification.from_dict(
                payload["cache_verification"]
            ),
            batch_timing=_timing_summary_from_dict(payload["batch_timing"]),
            runtime_resources=EmbeddingRuntimeResources.from_dict(
                payload["runtime_resources"]
            ),
            cost=EmbeddingProductionCost.from_dict(payload["cost"]),
            timing_interpretation=payload["timing_interpretation"],
        )


def produce_embedding_cache(
    *,
    inventory: ControlScoringInventory,
    artifact_paths: Mapping[str, Path],
    model_path: Path,
    model_lineage_path: Path,
    preprocessing_path: Path,
    dependency_lock_path: Path,
    config: EmbeddingProducerConfig,
    production_policy: EmbeddingProductionPolicy,
    cache_policy: EmbeddingCachePolicy,
    backend: EmbeddingBatchBackend,
    output_directory: Path,
    runtime_phase_callback: Callable[[str], None] | None = None,
) -> EmbeddingProductionReceipt:
    """Produce a content-deduplicated, verified embedding cache atomically."""

    if backend.identity != config.backend:
        raise ValueError("runtime backend identity differs from frozen config")
    if (
        backend.preprocessing_semantics_sha256
        != config.preprocessing_semantics_sha256
    ):
        raise ValueError(
            "runtime preprocessing semantics differ from frozen config"
        )
    if backend.model_sha256 != config.model_sha256:
        raise ValueError(
            "runtime loaded model differs from frozen config"
        )
    _validate_policy_compatibility(config, production_policy, cache_policy)
    output_root = _empty_real_directory(output_directory)
    resolved_artifacts = _validate_artifact_paths(inventory, artifact_paths)
    provenance_paths = {
        "model": _regular_file(model_path, "model"),
        "model_lineage": _regular_file(
            model_lineage_path,
            "model lineage",
        ),
        "preprocessing": _regular_file(
            preprocessing_path,
            "preprocessing",
        ),
        "dependency_lock": _regular_file(
            dependency_lock_path,
            "dependency lock",
        ),
    }
    expected_provenance = {
        "model": config.model_sha256,
        "model_lineage": config.model_lineage_sha256,
        "preprocessing": config.preprocessing_sha256,
        "dependency_lock": config.dependency_lock_sha256,
    }
    provenance_sizes = _verify_provenance_files(
        provenance_paths,
        expected_provenance,
        production_policy,
    )
    input_sizes = _verify_inventory_files(
        inventory,
        resolved_artifacts,
    )
    _preflight_work(
        inventory,
        input_sizes,
        config,
        production_policy,
        cache_policy,
    )

    representative_by_content: dict[str, str] = {}
    inventory_by_token = {
        entry.artifact_token: entry for entry in inventory.entries
    }
    for entry in sorted(inventory.entries, key=lambda item: item.artifact_token):
        representative_by_content.setdefault(
            entry.content_sha256,
            entry.artifact_token,
        )
    representative_tokens = tuple(representative_by_content.values())
    batches = tuple(
        representative_tokens[index : index + config.batch_size]
        for index in range(0, len(representative_tokens), config.batch_size)
    )
    if not batches:
        raise RuntimeError("embedding inventory unexpectedly produced no work")

    temporary_entries: dict[str, EmbeddingCacheEntry] = {}
    timing_samples: list[int] = []
    warmup_artifact_evaluations = 0
    warmup_wall_time_ns = 0
    total_backend_wall_time_ns = 0
    peak_batch_input_bytes = 0
    linked_paths: list[Path] = []
    first_output_observed = False
    with TemporaryDirectory(
        prefix=".cvi-embedding-cache-",
        dir=output_root.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        first_paths = tuple(resolved_artifacts[token] for token in batches[0])
        for _ in range(config.warmup_batches):
            backend.synchronize()
            started = perf_counter_ns()
            backend_vectors = backend.infer_batch(first_paths)
            backend.synchronize()
            _validate_backend_vectors(
                backend_vectors,
                expected_rows=len(first_paths),
                dimension=config.vector_dimension,
                l2_epsilon=config.l2_epsilon,
                encode=False,
            )
            elapsed = perf_counter_ns() - started
            if not first_output_observed:
                if runtime_phase_callback is not None:
                    runtime_phase_callback("FIRST_OUTPUT_READY")
                first_output_observed = True
            if elapsed > production_policy.maximum_batch_wall_time_ns:
                raise ValueError("embedding warmup exceeded wall-time policy")
            warmup_wall_time_ns += elapsed
            total_backend_wall_time_ns += elapsed
            if total_backend_wall_time_ns > (
                production_policy.maximum_total_wall_time_ns
            ):
                raise ValueError("embedding run exceeded total wall-time policy")
            warmup_artifact_evaluations += len(first_paths)

        for batch in batches:
            paths = tuple(resolved_artifacts[token] for token in batch)
            batch_input_bytes = sum(input_sizes[token] for token in batch)
            peak_batch_input_bytes = max(
                peak_batch_input_bytes,
                batch_input_bytes,
            )
            backend.synchronize()
            started = perf_counter_ns()
            raw_vectors = backend.infer_batch(paths)
            backend.synchronize()
            encoded = _validate_backend_vectors(
                raw_vectors,
                expected_rows=len(batch),
                dimension=config.vector_dimension,
                l2_epsilon=config.l2_epsilon,
                encode=True,
            )
            elapsed = perf_counter_ns() - started
            if not first_output_observed:
                if runtime_phase_callback is not None:
                    runtime_phase_callback("FIRST_OUTPUT_READY")
                first_output_observed = True
            if elapsed > production_policy.maximum_batch_wall_time_ns:
                raise ValueError("embedding batch exceeded wall-time policy")
            timing_samples.append(elapsed)
            total_backend_wall_time_ns += elapsed
            if total_backend_wall_time_ns > (
                production_policy.maximum_total_wall_time_ns
            ):
                raise ValueError("embedding run exceeded total wall-time policy")
            for token, vector_bytes in zip(batch, encoded, strict=True):
                inventory_entry = inventory_by_token[token]
                cache_key = embedding_cache_key(
                    artifact_content_sha256=(
                        inventory_entry.content_sha256
                    ),
                    model_sha256=config.model_sha256,
                    inference_config_sha256=config.config_sha256,
                    dependency_lock_sha256=config.dependency_lock_sha256,
                    code_revision=config.code_revision,
                    precision=config.backend.precision,
                    vector_dimension=config.vector_dimension,
                    vector_format=config.output_vector_format,
                )
                path = temporary_root / f"{cache_key}.f32le"
                with path.open("xb") as stream:
                    os.chmod(path, 0o600)
                    stream.write(vector_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary_entries[inventory_entry.content_sha256] = (
                    EmbeddingCacheEntry(
                        cache_key=cache_key,
                        relative_path=path.name,
                        content_sha256=sha256_file(path),
                        byte_size=len(vector_bytes),
                    )
                )

        runtime_resources = backend.runtime_resources()
        if not isinstance(runtime_resources, EmbeddingRuntimeResources):
            raise TypeError(
                "backend runtime_resources must return "
                "EmbeddingRuntimeResources"
            )
        _verify_inventory_files(inventory, resolved_artifacts)
        _verify_provenance_files(
            provenance_paths,
            expected_provenance,
            production_policy,
        )
        bindings = tuple(
            ArtifactCacheBinding(
                artifact_token=entry.artifact_token,
                artifact_content_sha256=entry.content_sha256,
                cache_key=temporary_entries[
                    entry.content_sha256
                ].cache_key,
            )
            for entry in inventory.entries
        )
        manifest = EmbeddingCacheManifest(
            scoring_inventory_sha256=inventory.inventory_sha256,
            model_sha256=config.model_sha256,
            inference_config_sha256=config.config_sha256,
            dependency_lock_sha256=config.dependency_lock_sha256,
            code_revision=config.code_revision,
            precision=config.backend.precision,
            vector_dimension=config.vector_dimension,
            normalization_tolerance=config.normalization_tolerance,
            bindings=bindings,
            entries=tuple(
                temporary_entries[content_sha256]
                for content_sha256 in sorted(temporary_entries)
            ),
        )
        try:
            for entry in manifest.entries:
                destination = output_root / entry.relative_path
                os.link(
                    temporary_root / entry.relative_path,
                    destination,
                )
                linked_paths.append(destination)
            verification = verify_embedding_cache_files(
                root=output_root,
                inventory=inventory,
                manifest=manifest,
                policy=cache_policy,
            )
        except BaseException:
            for path in linked_paths:
                path.unlink(missing_ok=True)
            raise

    unique_inputs = len(representative_tokens)
    output_bytes = unique_inputs * config.vector_dimension * 4
    try:
        cost = EmbeddingProductionCost(
            artifact_bindings=len(inventory.entries),
            unique_content_inputs=unique_inputs,
            content_deduplication_calls_saved=(
                len(inventory.entries) - unique_inputs
            ),
            warmup_batches=config.warmup_batches,
            warmup_artifact_evaluations=warmup_artifact_evaluations,
            production_batches=len(batches),
            production_artifact_evaluations=unique_inputs,
            total_backend_artifact_evaluations=(
                unique_inputs + warmup_artifact_evaluations
            ),
            warmup_wall_time_ns=warmup_wall_time_ns,
            production_wall_time_ns=sum(timing_samples),
            total_backend_wall_time_ns=total_backend_wall_time_ns,
            input_integrity_hash_passes=2,
            input_integrity_bytes_read=2 * sum(input_sizes.values()),
            provenance_integrity_hash_passes=2,
            provenance_integrity_bytes_read=(
                2 * sum(provenance_sizes.values())
            ),
            output_float_values=unique_inputs * config.vector_dimension,
            output_bytes_written=output_bytes,
            peak_batch_artifacts=max(len(batch) for batch in batches),
            peak_batch_input_bytes=peak_batch_input_bytes,
            peak_nominal_input_tensor_bytes=(
                max(len(batch) for batch in batches)
                * config.input_width
                * config.input_height
                * config.input_channels
                * config.input_value_bytes
            ),
            peak_batch_output_bytes=(
                max(len(batch) for batch in batches)
                * config.vector_dimension
                * 4
            ),
        )
        return EmbeddingProductionReceipt(
            scoring_inventory_sha256=inventory.inventory_sha256,
            producer_config_sha256=config.config_sha256,
            production_policy_sha256=production_policy.policy_sha256,
            cache_policy_sha256=cache_policy.policy_sha256,
            model_lineage_sha256=config.model_lineage_sha256,
            cache_manifest=manifest,
            cache_verification=verification,
            batch_timing=TimingSummary.from_samples(tuple(timing_samples)),
            runtime_resources=runtime_resources,
            cost=cost,
        )
    except BaseException:
        for path in linked_paths:
            path.unlink(missing_ok=True)
        raise


def validate_embedding_production_preflight(
    *,
    inventory: ControlScoringInventory,
    config: EmbeddingProducerConfig,
    production_policy: EmbeddingProductionPolicy,
    cache_policy: EmbeddingCachePolicy,
) -> None:
    """Reject structurally oversized work before backend construction."""

    _validate_policy_compatibility(config, production_policy, cache_policy)
    _preflight_work(
        inventory,
        {
            entry.artifact_token: entry.byte_size
            for entry in inventory.entries
        },
        config,
        production_policy,
        cache_policy,
    )


def _validate_policy_compatibility(
    config: EmbeddingProducerConfig,
    production_policy: EmbeddingProductionPolicy,
    cache_policy: EmbeddingCachePolicy,
) -> None:
    if config.batch_size > production_policy.maximum_batch_size:
        raise ValueError("embedding batch size exceeds production policy")
    if config.vector_dimension > (
        production_policy.maximum_vector_dimension
    ):
        raise ValueError("embedding dimension exceeds production policy")
    if config.vector_dimension > cache_policy.maximum_vector_dimension:
        raise ValueError("embedding dimension exceeds cache policy")
    if config.normalization_tolerance > (
        cache_policy.maximum_normalization_tolerance
    ):
        raise ValueError("normalization tolerance exceeds cache policy")


def _preflight_work(
    inventory: ControlScoringInventory,
    input_sizes: Mapping[str, int],
    config: EmbeddingProducerConfig,
    policy: EmbeddingProductionPolicy,
    cache_policy: EmbeddingCachePolicy,
) -> None:
    artifact_count = len(inventory.entries)
    representative_by_content: dict[str, str] = {}
    for entry in sorted(inventory.entries, key=lambda item: item.artifact_token):
        representative_by_content.setdefault(
            entry.content_sha256,
            entry.artifact_token,
        )
    unique_contents = len(representative_by_content)
    if artifact_count > policy.maximum_artifacts:
        raise ValueError("embedding artifacts exceed production policy")
    if unique_contents > policy.maximum_unique_inputs:
        raise ValueError("unique embedding inputs exceed production policy")
    if artifact_count > cache_policy.maximum_artifacts:
        raise ValueError("embedding artifacts exceed cache policy")
    if unique_contents > cache_policy.maximum_unique_vectors:
        raise ValueError("unique embedding inputs exceed cache policy")
    batch_count = math.ceil(unique_contents / config.batch_size)
    if batch_count > policy.maximum_batches:
        raise ValueError("embedding batches exceed production policy")
    representative_sizes = tuple(
        input_sizes[token] for token in representative_by_content.values()
    )
    integrity_input_bytes = sum(input_sizes.values())
    if integrity_input_bytes > policy.maximum_total_input_bytes:
        raise ValueError("embedding input bytes exceed production policy")
    largest_representative_sizes = sorted(
        representative_sizes,
        reverse=True,
    )
    maximum_batch_bytes = sum(
        largest_representative_sizes[: config.batch_size]
    )
    if maximum_batch_bytes > policy.maximum_input_bytes_per_batch:
        raise ValueError("embedding batch input bytes exceed production policy")
    nominal_tensor_bytes = (
        min(config.batch_size, unique_contents)
        * config.input_width
        * config.input_height
        * config.input_channels
        * config.input_value_bytes
    )
    if nominal_tensor_bytes > (
        policy.maximum_nominal_tensor_bytes_per_batch
    ):
        raise ValueError("nominal input tensor exceeds production policy")
    batch_output_bytes = (
        min(config.batch_size, unique_contents)
        * config.vector_dimension
        * 4
    )
    if batch_output_bytes > policy.maximum_output_bytes_per_batch:
        raise ValueError("embedding batch output exceeds production policy")
    output_bytes = unique_contents * config.vector_dimension * 4
    if output_bytes > policy.maximum_total_output_bytes:
        raise ValueError("embedding output bytes exceed production policy")
    if output_bytes > cache_policy.maximum_total_cache_bytes:
        raise ValueError("embedding output bytes exceed cache policy")


def _validate_artifact_paths(
    inventory: ControlScoringInventory,
    artifact_paths: Mapping[str, Path],
) -> dict[str, Path]:
    expected = {entry.artifact_token for entry in inventory.entries}
    if set(artifact_paths) != expected:
        raise ValueError("artifact paths do not cover scoring inventory exactly")
    return {
        token: _regular_file(path, f"artifact {token}")
        for token, path in artifact_paths.items()
    }


def _verify_inventory_files(
    inventory: ControlScoringInventory,
    artifact_paths: Mapping[str, Path],
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for entry in inventory.entries:
        path = artifact_paths[entry.artifact_token]
        before = path.stat()
        if before.st_size != entry.byte_size:
            raise ValueError("embedding input byte-size mismatch")
        digest = sha256_file(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError("embedding input changed during verification")
        if digest != entry.content_sha256:
            raise ValueError("embedding input content hash mismatch")
        sizes[entry.artifact_token] = entry.byte_size
    return sizes


def _verify_provenance_files(
    paths: Mapping[str, Path],
    expected: Mapping[str, str],
    policy: EmbeddingProductionPolicy,
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name, path in paths.items():
        before = path.stat()
        cap = (
            policy.maximum_model_bytes
            if name == "model"
            else policy.maximum_provenance_file_bytes
        )
        if before.st_size <= 0 or before.st_size > cap:
            raise ValueError(f"{name} file exceeds size policy")
        digest = sha256_file(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(f"{name} changed during verification")
        if digest != expected[name]:
            raise ValueError(f"{name} content hash mismatch")
        sizes[name] = before.st_size
    return sizes


def _validate_backend_vectors(
    raw_vectors: Sequence[Sequence[float]],
    *,
    expected_rows: int,
    dimension: int,
    l2_epsilon: float,
    encode: bool,
) -> tuple[bytes, ...]:
    if not isinstance(raw_vectors, Sequence):
        raise TypeError("backend output must be a sequence")
    if len(raw_vectors) != expected_rows:
        raise ValueError("backend output row count differs from batch")
    encoded: list[bytes] = []
    for vector in raw_vectors:
        if not isinstance(vector, Sequence) or len(vector) != dimension:
            raise ValueError("backend embedding dimension mismatch")
        values: list[float] = []
        for value in vector:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("backend embedding contains a non-finite value")
            values.append(float(value))
        norm = _stable_l2_norm(values)
        if norm <= l2_epsilon:
            raise ValueError("backend embedding norm is too small")
        if encode:
            normalized = tuple(value / norm for value in values)
            first = struct.pack(f"<{dimension}f", *normalized)
            rounded = struct.unpack(f"<{dimension}f", first)
            rounded_norm = _stable_l2_norm(rounded)
            final = struct.pack(
                f"<{dimension}f",
                *(value / rounded_norm for value in rounded),
            )
            if not all(
                math.isfinite(value)
                for value in struct.unpack(f"<{dimension}f", final)
            ):
                raise ValueError("normalized embedding is not finite")
            encoded.append(final)
    return tuple(encoded)


def _stable_l2_norm(values: Sequence[float]) -> float:
    scale = 0.0
    sum_squares = 1.0
    nonzero = False
    for value in values:
        absolute = abs(value)
        if absolute == 0.0:
            continue
        nonzero = True
        if scale < absolute:
            ratio = scale / absolute
            sum_squares = 1.0 + sum_squares * ratio * ratio
            scale = absolute
        else:
            ratio = absolute / scale
            sum_squares += ratio * ratio
    if not nonzero:
        return 0.0
    return scale * math.sqrt(sum_squares)


def _timing_summary_from_dict(payload: dict[str, Any]) -> TimingSummary:
    _require_exact_keys(
        payload,
        {
            "samples",
            "minimum_ns",
            "p50_ns",
            "p95_ns",
            "maximum_ns",
            "mean_ns",
        },
        "embedding batch timing",
    )
    for name in (
        "samples",
        "minimum_ns",
        "p50_ns",
        "p95_ns",
        "maximum_ns",
    ):
        if name == "samples":
            _require_positive_int(payload[name], name)
        else:
            _require_nonnegative_int(payload[name], name)
    if not (
        payload["minimum_ns"]
        <= payload["p50_ns"]
        <= payload["p95_ns"]
        <= payload["maximum_ns"]
    ):
        raise ValueError("embedding timing quantiles are inconsistent")
    mean = payload["mean_ns"]
    if (
        isinstance(mean, bool)
        or not isinstance(mean, (int, float))
        or not math.isfinite(mean)
        or not payload["minimum_ns"] <= mean <= payload["maximum_ns"]
    ):
        raise ValueError("embedding timing mean is inconsistent")
    return TimingSummary(
        samples=payload["samples"],
        minimum_ns=payload["minimum_ns"],
        p50_ns=payload["p50_ns"],
        p95_ns=payload["p95_ns"],
        maximum_ns=payload["maximum_ns"],
        mean_ns=float(mean),
    )


def _empty_real_directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("embedding output directory must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    if any(resolved.iterdir()):
        raise ValueError("embedding output directory must be empty")
    return resolved


def _regular_file(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
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
