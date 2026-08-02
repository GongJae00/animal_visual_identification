"""Paired ONNX measurement admission without optimization promotion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

from evaluation.control_scoring import EmbeddingCacheManifest
from operations.embedding_producer import EmbeddingBackendIdentity, EmbeddingProducerConfig
from evaluation.numerical_admission import (
    NumericalAdmissionDecision,
    NumericalAdmissionReceipt,
)
from operations.onnx_inference_benchmark import (
    OnnxBenchmarkBackend,
    OnnxInferenceBenchmarkPolicy,
    OnnxInferenceBenchmarkSummary,
)
from representation_learning.optimization import PromotionDecision
from foundation.provenance import content_sha256


class MeasurementAdmissionDecision(StrEnum):
    COMPARABLE_NOT_PROMOTED = "MEASUREMENT_COMPARABLE_NOT_PROMOTED"


@dataclass(frozen=True, slots=True)
class DescriptivePointComparison:
    name: str
    scope: str
    unit: str
    reference: float
    candidate: float
    candidate_minus_reference: float
    candidate_over_reference: float
    interpretation: str = "DESCRIPTIVE_POINT_ESTIMATE_NO_UNCERTAINTY"
    schema_version: str = "cvi.descriptive_point_comparison.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.descriptive_point_comparison.v1":
            raise ValueError("unsupported point-comparison schema")
        for name in ("name", "scope", "unit"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "reference",
            "candidate",
            "candidate_minus_reference",
            "candidate_over_reference",
        ):
            if not isfinite(getattr(self, name)):
                raise ValueError("point-comparison values must be finite")
        if self.reference <= 0 or self.candidate < 0:
            raise ValueError("point-comparison values must be non-negative")
        if self.candidate_over_reference <= 0:
            raise ValueError("point-comparison ratio must be positive")
        if self.candidate_minus_reference != self.candidate - self.reference:
            raise ValueError("point-comparison delta is inconsistent")
        if self.candidate_over_reference != self.candidate / self.reference:
            raise ValueError("point-comparison ratio is inconsistent")
        if self.interpretation != "DESCRIPTIVE_POINT_ESTIMATE_NO_UNCERTAINTY":
            raise ValueError("point-comparison interpretation is fixed")

    def to_dict(self) -> dict[str, str | float]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "scope": self.scope,
            "unit": self.unit,
            "reference": self.reference,
            "candidate": self.candidate,
            "candidate_minus_reference": self.candidate_minus_reference,
            "candidate_over_reference": self.candidate_over_reference,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> DescriptivePointComparison:
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("point-comparison keys mismatch")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PairedInferenceMeasurementReceipt:
    reference_summary_sha256: str
    candidate_summary_sha256: str
    reference_producer_config_sha256: str
    candidate_producer_config_sha256: str
    reference_manifest_sha256: str
    candidate_manifest_sha256: str
    numerical_admission_receipt_sha256: str
    workload_sha256: str
    comparable_policy_sha256: str
    host_fingerprint_sha256: str
    point_comparisons: tuple[DescriptivePointComparison, ...]
    excluded_metric_scopes: tuple[str, ...]
    decision: MeasurementAdmissionDecision
    promotion_decision: PromotionDecision
    interpretation: str = (
        "PAIRED_MEASUREMENT_AND_NUMERICAL_ADMISSION_ONLY_"
        "NOT_BIOMETRIC_NONINFERIORITY_OR_OPTIMIZATION_PROMOTION"
    )
    schema_version: str = "cvi.paired_inference_measurement_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.paired_inference_measurement_receipt.v1":
            raise ValueError("unsupported paired-measurement receipt schema")
        for name in (
            "reference_summary_sha256",
            "candidate_summary_sha256",
            "reference_producer_config_sha256",
            "candidate_producer_config_sha256",
            "reference_manifest_sha256",
            "candidate_manifest_sha256",
            "numerical_admission_receipt_sha256",
            "workload_sha256",
            "comparable_policy_sha256",
            "host_fingerprint_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if not self.point_comparisons:
            raise ValueError("paired measurement must contain point comparisons")
        names = tuple(item.name for item in self.point_comparisons)
        if len(names) != len(set(names)) or names != tuple(sorted(names)):
            raise ValueError("point-comparison names must be unique and sorted")
        if (
            not self.excluded_metric_scopes
            or self.excluded_metric_scopes != tuple(
                sorted(set(self.excluded_metric_scopes))
            )
        ):
            raise ValueError("excluded metric scopes must be unique and sorted")
        if any(not item.strip() for item in self.excluded_metric_scopes):
            raise ValueError("excluded metric scope must be non-empty")
        if self.decision is not MeasurementAdmissionDecision.COMPARABLE_NOT_PROMOTED:
            raise ValueError("paired measurement decision is fixed")
        if self.promotion_decision is not PromotionDecision.INCONCLUSIVE:
            raise ValueError("paired measurement cannot promote an optimization")
        if self.interpretation != (
            "PAIRED_MEASUREMENT_AND_NUMERICAL_ADMISSION_ONLY_"
            "NOT_BIOMETRIC_NONINFERIORITY_OR_OPTIMIZATION_PROMOTION"
        ):
            raise ValueError("paired measurement interpretation is fixed")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_summary_sha256": self.reference_summary_sha256,
            "candidate_summary_sha256": self.candidate_summary_sha256,
            "reference_producer_config_sha256": (
                self.reference_producer_config_sha256
            ),
            "candidate_producer_config_sha256": (
                self.candidate_producer_config_sha256
            ),
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "numerical_admission_receipt_sha256": (
                self.numerical_admission_receipt_sha256
            ),
            "workload_sha256": self.workload_sha256,
            "comparable_policy_sha256": self.comparable_policy_sha256,
            "host_fingerprint_sha256": self.host_fingerprint_sha256,
            "point_comparisons": [
                item.to_dict() for item in self.point_comparisons
            ],
            "excluded_metric_scopes": list(self.excluded_metric_scopes),
            "decision": self.decision.value,
            "promotion_decision": self.promotion_decision.value,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> PairedInferenceMeasurementReceipt:
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("paired-measurement receipt keys mismatch")
        points = payload["point_comparisons"]
        excluded = payload["excluded_metric_scopes"]
        if not isinstance(points, list) or not isinstance(excluded, list):
            raise TypeError("paired-measurement collections must be lists")
        values = dict(payload)
        values["point_comparisons"] = tuple(
            DescriptivePointComparison.from_dict(item) for item in points
        )
        values["excluded_metric_scopes"] = tuple(excluded)
        values["decision"] = MeasurementAdmissionDecision(payload["decision"])
        values["promotion_decision"] = PromotionDecision(
            payload["promotion_decision"]
        )
        return cls(**values)


def compare_paired_inference_measurements(
    *,
    reference: OnnxInferenceBenchmarkSummary,
    candidate: OnnxInferenceBenchmarkSummary,
    reference_config: EmbeddingProducerConfig,
    candidate_config: EmbeddingProducerConfig,
    reference_manifest: EmbeddingCacheManifest,
    candidate_manifest: EmbeddingCacheManifest,
    numerical_admission: NumericalAdmissionReceipt,
) -> PairedInferenceMeasurementReceipt:
    """Admit matched measurements while forcing promotion to remain unresolved."""

    if reference.backend is not OnnxBenchmarkBackend.CPU:
        raise ValueError("paired reference must be the strict CPU backend")
    if candidate.backend is not OnnxBenchmarkBackend.CUDA:
        raise ValueError("paired candidate must be the full-graph CUDA backend")
    if (
        not reference.policy.unrelated_system_work_excluded_by_operator
        or not candidate.policy.unrelated_system_work_excluded_by_operator
    ):
        raise ValueError("paired measurement requires clean system-work attestations")
    if candidate.unrelated_gpu_work_excluded_by_operator is not True:
        raise ValueError("paired CUDA measurement is device-wide contaminated")
    if (
        reference.runtime_library_decision != "PASS"
        or candidate.runtime_library_decision != "PASS"
    ):
        raise ValueError(
            "paired measurement requires strict runtime library PASS"
        )
    if reference.host_identity != candidate.host_identity or (
        reference.host_fingerprint_sha256
        != candidate.host_fingerprint_sha256
    ):
        raise ValueError("paired measurements were not made on the same host boot")

    workload = _paired_workload(reference)
    if workload != _paired_workload(candidate):
        raise ValueError("paired inference workload lineage differs")
    reference_policy = _comparable_policy(reference.policy)
    candidate_policy = _comparable_policy(candidate.policy)
    if reference_policy != candidate_policy:
        raise ValueError("paired inference measurement policies differ")

    _validate_producer_and_manifest_binding(
        summary=reference,
        config=reference_config,
        manifest=reference_manifest,
        expected_manifest_sha256=(
            numerical_admission.reference_manifest_sha256
        ),
        expected_config_sha256=numerical_admission.reference_config_sha256,
        label="reference",
    )
    _validate_producer_and_manifest_binding(
        summary=candidate,
        config=candidate_config,
        manifest=candidate_manifest,
        expected_manifest_sha256=(
            numerical_admission.candidate_manifest_sha256
        ),
        expected_config_sha256=numerical_admission.candidate_config_sha256,
        label="candidate",
    )
    if numerical_admission.decision is not NumericalAdmissionDecision.PASS:
        raise ValueError("paired measurement requires numerical admission PASS")
    if reference_manifest.scoring_inventory_sha256 != (
        candidate_manifest.scoring_inventory_sha256
    ) or reference_manifest.scoring_inventory_sha256 != (
        numerical_admission.scoring_inventory_sha256
    ):
        raise ValueError("paired numerical scoring inventory differs")
    reference_bindings = tuple(
        (item.artifact_token, item.artifact_content_sha256)
        for item in reference_manifest.bindings
    )
    candidate_bindings = tuple(
        (item.artifact_token, item.artifact_content_sha256)
        for item in candidate_manifest.bindings
    )
    if reference_bindings != candidate_bindings:
        raise ValueError("paired numerical artifact bindings differ")
    reference_semantics = _producer_comparable_semantics(reference_config)
    if reference_semantics != _producer_comparable_semantics(candidate_config):
        raise ValueError("paired numerical producer semantics differ")
    if content_sha256(reference_semantics) != (
        numerical_admission.comparable_semantics_sha256
    ):
        raise ValueError("paired numerical semantics receipt binding differs")
    unique_contents = {
        item.artifact_content_sha256 for item in reference_manifest.bindings
    }
    expected_values = len(unique_contents) * reference_manifest.vector_dimension
    if (
        numerical_admission.summary.vectors != len(unique_contents)
        or numerical_admission.summary.values != expected_values
        or numerical_admission.summary.bytes_read != expected_values * 4 * 2
    ):
        raise ValueError("paired numerical summary work accounting differs")

    points = _point_comparisons(reference, candidate)
    return PairedInferenceMeasurementReceipt(
        reference_summary_sha256=reference.summary_sha256,
        candidate_summary_sha256=candidate.summary_sha256,
        reference_producer_config_sha256=reference_config.config_sha256,
        candidate_producer_config_sha256=candidate_config.config_sha256,
        reference_manifest_sha256=reference_manifest.manifest_sha256,
        candidate_manifest_sha256=candidate_manifest.manifest_sha256,
        numerical_admission_receipt_sha256=(
            numerical_admission.receipt_sha256
        ),
        workload_sha256=content_sha256(workload),
        comparable_policy_sha256=content_sha256(reference_policy),
        host_fingerprint_sha256=reference.host_fingerprint_sha256,
        point_comparisons=points,
        excluded_metric_scopes=tuple(
            sorted(
                (
                    "CUDA_DEVICE_WIDE_BOARD_ENERGY_NOT_CPU_COMPARABLE",
                    "CUDA_DEVICE_WIDE_MEMORY_NOT_PROCESS_ATTRIBUTED",
                    "CUDA_DEVICE_WIDE_UTILIZATION_NOT_CPU_COMPARABLE",
                    "NO_BIOMETRIC_SAFETY_INTERVALS",
                    "NO_RESOURCE_UNCERTAINTY_INTERVALS",
                )
            )
        ),
        decision=MeasurementAdmissionDecision.COMPARABLE_NOT_PROMOTED,
        promotion_decision=PromotionDecision.INCONCLUSIVE,
    )


def _validate_producer_and_manifest_binding(
    *,
    summary: OnnxInferenceBenchmarkSummary,
    config: EmbeddingProducerConfig,
    manifest: EmbeddingCacheManifest,
    expected_manifest_sha256: str,
    expected_config_sha256: str,
    label: str,
) -> None:
    if config.config_sha256 != expected_config_sha256:
        raise ValueError(f"{label} numerical producer config binding differs")
    if manifest.manifest_sha256 != expected_manifest_sha256:
        raise ValueError(f"{label} numerical manifest binding differs")
    if manifest.inference_config_sha256 != config.config_sha256:
        raise ValueError(f"{label} manifest/config binding differs")
    expected = {
        "model_sha256": summary.model_sha256,
        "preprocessing_sha256": summary.preprocessing_file_sha256,
        "preprocessing_semantics_sha256": (
            summary.preprocessing_config_sha256
        ),
        "dependency_lock_sha256": summary.dependency_lock_sha256,
        "code_revision": summary.code_revision,
        "vector_dimension": summary.vector_dimension,
        "batch_size": len(summary.artifact_content_sha256),
        "input_value_bytes": 4,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(f"{label} producer {name} binding differs")
    if config.backend.backend_config_sha256 != summary.backend_config_sha256:
        raise ValueError(f"{label} producer backend config binding differs")
    worker_identity = EmbeddingBackendIdentity.from_dict(
        summary.worker_results[0]["measurement"]["backend_identity"]
    )
    if config.backend != worker_identity:
        raise ValueError(f"{label} producer backend identity differs")
    if (
        config.input_width * config.input_height * config.input_channels
        * config.batch_size * config.input_value_bytes
        != summary.tensor_bytes
    ):
        raise ValueError(f"{label} producer tensor byte binding differs")
    manifest_contents = {
        binding.artifact_content_sha256 for binding in manifest.bindings
    }
    if not set(summary.artifact_content_sha256).issubset(manifest_contents):
        raise ValueError(f"{label} measured artifacts lack numerical vectors")


def _paired_workload(summary: OnnxInferenceBenchmarkSummary) -> dict[str, Any]:
    return {
        "model_sha256": summary.model_sha256,
        "preprocessing_config_sha256": summary.preprocessing_config_sha256,
        "preprocessing_file_sha256": summary.preprocessing_file_sha256,
        "dependency_lock_sha256": summary.dependency_lock_sha256,
        "code_revision": summary.code_revision,
        "artifact_content_sha256": list(summary.artifact_content_sha256),
        "tensor_sha256": summary.tensor_sha256,
        "tensor_bytes": summary.tensor_bytes,
        "tensor_shape": list(summary.tensor_shape),
        "vector_dimension": summary.vector_dimension,
    }


def _comparable_policy(policy: OnnxInferenceBenchmarkPolicy) -> dict[str, Any]:
    payload = policy.to_dict()
    payload.pop("gpu_device_index")
    payload.pop("gpu_telemetry_interval_seconds")
    payload.pop("unrelated_gpu_work_excluded_by_operator")
    return payload


def _producer_comparable_semantics(
    config: EmbeddingProducerConfig,
) -> dict[str, Any]:
    payload = config.to_dict()
    payload.pop("backend")
    payload.pop("dependency_lock_sha256")
    return payload


def _point_comparisons(
    reference: OnnxInferenceBenchmarkSummary,
    candidate: OnnxInferenceBenchmarkSummary,
) -> tuple[DescriptivePointComparison, ...]:
    fields = (
        (
            "dependency_import_p50_ns",
            "fresh-worker-phase",
            reference.dependency_import_time.p50_ns,
            candidate.dependency_import_time.p50_ns,
        ),
        (
            "end_to_end_p50_ns",
            "image-to-cpu-output-api",
            reference.end_to_end_inference_time.p50_ns,
            candidate.end_to_end_inference_time.p50_ns,
        ),
        (
            "end_to_end_p95_ns",
            "image-to-cpu-output-api",
            reference.end_to_end_inference_time.p95_ns,
            candidate.end_to_end_inference_time.p95_ns,
        ),
        (
            "first_preprocessed_p50_ns",
            "preprocessed-tensor-to-cpu-output-api",
            reference.first_preprocessed_inference_time.p50_ns,
            candidate.first_preprocessed_inference_time.p50_ns,
        ),
        (
            "preprocessing_p50_ns",
            "image-to-host-float32-tensor",
            reference.preprocessing_time.p50_ns,
            candidate.preprocessing_time.p50_ns,
        ),
        (
            "process_wall_p50_ns",
            "fresh-worker-process",
            reference.process_wall_time.p50_ns,
            candidate.process_wall_time.p50_ns,
        ),
        (
            "session_construction_p50_ns",
            "backend-session-construction",
            reference.session_construction_time.p50_ns,
            candidate.session_construction_time.p50_ns,
        ),
        (
            "warm_preprocessed_p50_ns",
            "preprocessed-tensor-to-cpu-output-api",
            reference.warm_preprocessed_inference_time.p50_ns,
            candidate.warm_preprocessed_inference_time.p50_ns,
        ),
        (
            "warm_preprocessed_p95_ns",
            "preprocessed-tensor-to-cpu-output-api",
            reference.warm_preprocessed_inference_time.p95_ns,
            candidate.warm_preprocessed_inference_time.p95_ns,
        ),
        (
            "worker_ru_maxrss_max_bytes",
            reference.worker_rss_scope,
            reference.maximum_worker_ru_maxrss_bytes,
            candidate.maximum_worker_ru_maxrss_bytes,
        ),
    )
    if reference.worker_rss_scope != candidate.worker_rss_scope:
        raise ValueError("paired worker RSS scopes differ")
    points = tuple(
        _point(name, scope, "ns" if name.endswith("_ns") else "bytes", ref, cand)
        for name, scope, ref, cand in fields
    )
    return tuple(sorted(points, key=lambda item: item.name))


def _point(
    name: str,
    scope: str,
    unit: str,
    reference: int,
    candidate: int,
) -> DescriptivePointComparison:
    return DescriptivePointComparison(
        name=name,
        scope=scope,
        unit=unit,
        reference=float(reference),
        candidate=float(candidate),
        candidate_minus_reference=float(candidate - reference),
        candidate_over_reference=candidate / reference,
    )


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
