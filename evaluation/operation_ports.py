"""Structural ports for operation-produced inputs consumed by evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from evaluation.control_scoring import (
    EmbeddingCacheManifest,
    EmbeddingCacheVerification,
)
from foundation.timing import TimingSummary


class EmbeddingBackendIdentity(Protocol):
    precision: str
    backend_config_sha256: str

    def to_dict(self) -> dict[str, str]: ...


class EmbeddingProducerConfig(Protocol):
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
    output_vector_format: str

    @property
    def config_sha256(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


class EmbeddingBatchBackend(Protocol):
    @property
    def identity(self) -> EmbeddingBackendIdentity: ...

    @property
    def preprocessing_semantics_sha256(self) -> str: ...

    @property
    def model_sha256(self) -> str: ...

    def infer_batch(
        self,
        artifact_paths: tuple[Path, ...],
    ) -> Sequence[Sequence[float]]: ...

    def synchronize(self) -> None: ...


class EmbeddingProductionReceipt(Protocol):
    scoring_inventory_sha256: str
    producer_config_sha256: str
    cache_policy_sha256: str
    model_lineage_sha256: str
    cache_manifest: EmbeddingCacheManifest
    cache_verification: EmbeddingCacheVerification

    @property
    def receipt_sha256(self) -> str: ...


class OnnxBenchmarkBackend(Protocol):
    value: str


class OnnxInferenceBenchmarkPolicy(Protocol):
    unrelated_system_work_excluded_by_operator: bool

    def to_dict(self) -> dict[str, Any]: ...


class OnnxInferenceBenchmarkSummary(Protocol):
    backend: OnnxBenchmarkBackend
    policy: OnnxInferenceBenchmarkPolicy
    unrelated_gpu_work_excluded_by_operator: bool | None
    runtime_library_decision: str
    host_identity: dict[str, str | int]
    host_fingerprint_sha256: str
    model_sha256: str
    preprocessing_config_sha256: str
    preprocessing_file_sha256: str
    dependency_lock_sha256: str
    code_revision: str
    artifact_content_sha256: tuple[str, ...]
    tensor_sha256: str
    tensor_bytes: int
    tensor_shape: tuple[int, ...]
    vector_dimension: int
    backend_config_sha256: str
    worker_results: tuple[dict[str, Any], ...]
    summary_sha256: str
    dependency_import_time: TimingSummary
    end_to_end_inference_time: TimingSummary
    first_preprocessed_inference_time: TimingSummary
    preprocessing_time: TimingSummary
    process_wall_time: TimingSummary
    session_construction_time: TimingSummary
    warm_preprocessed_inference_time: TimingSummary
    worker_rss_scope: str
    maximum_worker_ru_maxrss_bytes: int


__all__ = [
    "EmbeddingBatchBackend",
    "EmbeddingProducerConfig",
    "EmbeddingProductionReceipt",
    "OnnxInferenceBenchmarkPolicy",
    "OnnxInferenceBenchmarkSummary",
]
