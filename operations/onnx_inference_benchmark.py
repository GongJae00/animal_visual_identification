"""Fresh-worker ONNX inference measurement without optimization promotion."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Any

from data_pipeline.acquisition import sha256_file
from evaluation.benchmark import TimingSummary
from operations.process_supervisor import (
    ProcessSupervisorPolicy,
    SupervisedProcessResult,
    SupervisedProcessStatus,
    run_supervised_process,
)
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from foundation.provenance import content_sha256
from operations.telemetry import GpuTelemetrySummary, monitor_operation
from artifact_contracts.runtime_library_provenance import (
    RuntimeLibraryManifest,
    RuntimeLibraryPhase,
    RuntimeLibraryPolicy,
    RuntimeLibraryTracker,
)
from operations.worker_environment import (
    ISOLATED_WORKER_BOOTSTRAP,
    WorkerEnvironmentIdentity,
    build_sanitized_worker_environment,
    validate_current_worker_environment,
)

_MAXIMUM_WORKER_REQUEST_BYTES = 4 * 1024 * 1024
_MAXIMUM_ARTIFACTS_HARD_CAP = 4096
_MAXIMUM_TOTAL_BYTES_HARD_CAP = 16 * 1024**3


class OnnxBenchmarkBackend(StrEnum):
    CPU = "CPU"
    CUDA = "CUDA"


@dataclass(frozen=True, slots=True)
class OnnxInferenceBenchmarkPolicy:
    fresh_processes: int
    model_warmup_iterations: int
    model_repeat_iterations: int
    end_to_end_repeat_iterations: int
    maximum_artifacts: int
    maximum_total_artifact_bytes: int
    maximum_tensor_bytes: int
    maximum_worker_result_bytes: int
    supervisor: ProcessSupervisorPolicy
    unrelated_system_work_excluded_by_operator: bool
    gpu_device_index: int | None = None
    gpu_telemetry_interval_seconds: float | None = None
    unrelated_gpu_work_excluded_by_operator: bool | None = None
    schema_version: str = "cvi.onnx_inference_benchmark_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.onnx_inference_benchmark_policy.v1":
            raise ValueError("unsupported ONNX benchmark policy schema")
        _positive_int(self.fresh_processes, "fresh_processes")
        if self.fresh_processes > 32:
            raise ValueError("fresh_processes exceeds policy cap")
        _nonnegative_int(
            self.model_warmup_iterations,
            "model_warmup_iterations",
        )
        _positive_int(
            self.model_repeat_iterations,
            "model_repeat_iterations",
        )
        _positive_int(
            self.end_to_end_repeat_iterations,
            "end_to_end_repeat_iterations",
        )
        if (
            self.model_warmup_iterations > 10_000
            or self.model_repeat_iterations > 10_000
            or self.end_to_end_repeat_iterations > 10_000
        ):
            raise ValueError("ONNX benchmark iterations exceed policy cap")
        _positive_int(self.maximum_artifacts, "maximum_artifacts")
        if self.maximum_artifacts > _MAXIMUM_ARTIFACTS_HARD_CAP:
            raise ValueError("maximum_artifacts exceeds hard cap")
        _positive_int(
            self.maximum_total_artifact_bytes,
            "maximum_total_artifact_bytes",
        )
        _positive_int(self.maximum_tensor_bytes, "maximum_tensor_bytes")
        if (
            self.maximum_total_artifact_bytes > _MAXIMUM_TOTAL_BYTES_HARD_CAP
            or self.maximum_tensor_bytes > _MAXIMUM_TOTAL_BYTES_HARD_CAP
        ):
            raise ValueError("ONNX benchmark byte cap is too large")
        _positive_int(
            self.maximum_worker_result_bytes,
            "maximum_worker_result_bytes",
        )
        if self.maximum_worker_result_bytes > 16 * 1024 * 1024:
            raise ValueError("worker result byte cap is too large")
        if not isinstance(
            self.unrelated_system_work_excluded_by_operator,
            bool,
        ):
            raise TypeError("system-work declaration must be boolean")
        telemetry = (
            self.gpu_device_index,
            self.gpu_telemetry_interval_seconds,
            self.unrelated_gpu_work_excluded_by_operator,
        )
        if any(value is None for value in telemetry) and not all(
            value is None for value in telemetry
        ):
            raise ValueError("GPU telemetry fields must be set together")
        if self.gpu_device_index is not None:
            _nonnegative_int(self.gpu_device_index, "gpu_device_index")
            interval = self.gpu_telemetry_interval_seconds
            if (
                isinstance(interval, bool)
                or not isinstance(interval, (int, float))
                or not isfinite(interval)
                or interval < 0.1
            ):
                raise ValueError(
                    "GPU telemetry interval must be at least 0.1 seconds"
                )
            if not isinstance(
                self.unrelated_gpu_work_excluded_by_operator,
                bool,
            ):
                raise TypeError("GPU-work declaration must be boolean")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fresh_processes": self.fresh_processes,
            "model_warmup_iterations": self.model_warmup_iterations,
            "model_repeat_iterations": self.model_repeat_iterations,
            "end_to_end_repeat_iterations": (
                self.end_to_end_repeat_iterations
            ),
            "maximum_artifacts": self.maximum_artifacts,
            "maximum_total_artifact_bytes": (
                self.maximum_total_artifact_bytes
            ),
            "maximum_tensor_bytes": self.maximum_tensor_bytes,
            "maximum_worker_result_bytes": (
                self.maximum_worker_result_bytes
            ),
            "supervisor": self.supervisor.to_dict(),
            "unrelated_system_work_excluded_by_operator": (
                self.unrelated_system_work_excluded_by_operator
            ),
            "gpu_device_index": self.gpu_device_index,
            "gpu_telemetry_interval_seconds": (
                self.gpu_telemetry_interval_seconds
            ),
            "unrelated_gpu_work_excluded_by_operator": (
                self.unrelated_gpu_work_excluded_by_operator
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> OnnxInferenceBenchmarkPolicy:
        expected = {
            "schema_version",
            "fresh_processes",
            "model_warmup_iterations",
            "model_repeat_iterations",
            "end_to_end_repeat_iterations",
            "maximum_artifacts",
            "maximum_total_artifact_bytes",
            "maximum_tensor_bytes",
            "maximum_worker_result_bytes",
            "supervisor",
            "unrelated_system_work_excluded_by_operator",
            "gpu_device_index",
            "gpu_telemetry_interval_seconds",
            "unrelated_gpu_work_excluded_by_operator",
        }
        if set(payload) != expected:
            raise ValueError("ONNX benchmark policy keys mismatch")
        supervisor = payload["supervisor"]
        if not isinstance(supervisor, dict):
            raise TypeError("supervisor policy must be an object")
        values = dict(payload)
        values["supervisor"] = ProcessSupervisorPolicy.from_dict(supervisor)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class OnnxInferenceBenchmarkSummary:
    backend: OnnxBenchmarkBackend
    policy: OnnxInferenceBenchmarkPolicy
    policy_sha256: str
    request_sha256: str
    model_sha256: str
    backend_config_sha256: str
    backend_config_file_sha256: str
    preprocessing_config_sha256: str
    preprocessing_file_sha256: str
    dependency_lock_sha256: str
    runtime_library_policy: RuntimeLibraryPolicy
    runtime_library_policy_sha256: str
    runtime_library_policy_file_sha256: str
    code_revision: str
    artifact_content_sha256: tuple[str, ...]
    tensor_sha256: str
    tensor_bytes: int
    tensor_shape: tuple[int, ...]
    vector_dimension: int
    output_sha256: str
    output_evaluations: int
    fresh_processes: int
    process_wall_time: TimingSummary
    dependency_import_time: TimingSummary
    session_construction_time: TimingSummary
    preprocessing_time: TimingSummary
    first_preprocessed_inference_time: TimingSummary
    warm_preprocessed_inference_time: TimingSummary
    end_to_end_inference_time: TimingSummary
    maximum_worker_ru_maxrss_bytes: int
    maximum_supervisor_sampled_rss_bytes: int | None
    worker_rss_scope: str
    supervisor_rss_scope: str
    host_identity: dict[str, str | int]
    host_fingerprint_sha256: str
    worker_environment_identity: WorkerEnvironmentIdentity
    worker_environment_identity_sha256: str
    onnxruntime_distribution_name: str
    onnxruntime_distribution_version: str
    runtime_library_binary_set_sha256: str
    runtime_library_decision: str
    runtime_library_provenance_time: TimingSummary
    maximum_runtime_library_bytes_hashed: int
    gpu_telemetry: GpuTelemetrySummary | None
    unrelated_gpu_work_excluded_by_operator: bool | None
    worker_results: tuple[dict[str, Any], ...]
    interpretation: str = (
        "MEASUREMENT_ONLY_NOT_OPTIMIZATION_PROMOTION_OR_BIOMETRIC_EVIDENCE"
    )
    schema_version: str = "cvi.onnx_inference_benchmark_summary.v3"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.onnx_inference_benchmark_summary.v3":
            raise ValueError("unsupported ONNX benchmark summary schema")
        if self.interpretation != (
            "MEASUREMENT_ONLY_NOT_OPTIMIZATION_PROMOTION_OR_BIOMETRIC_EVIDENCE"
        ):
            raise ValueError("invalid ONNX benchmark interpretation")
        for digest in (
            self.policy_sha256,
            self.request_sha256,
            self.model_sha256,
            self.backend_config_sha256,
            self.backend_config_file_sha256,
            self.preprocessing_config_sha256,
            self.preprocessing_file_sha256,
            self.dependency_lock_sha256,
            self.runtime_library_policy_sha256,
            self.runtime_library_policy_file_sha256,
            self.tensor_sha256,
            self.output_sha256,
            self.host_fingerprint_sha256,
            self.worker_environment_identity_sha256,
            self.runtime_library_binary_set_sha256,
            *self.artifact_content_sha256,
        ):
            _validate_sha256(digest)
        if not self.code_revision.strip():
            raise ValueError("summary code revision must be non-empty")
        if self.policy.policy_sha256 != self.policy_sha256:
            raise ValueError("summary policy hash differs from policy")
        if self.runtime_library_policy.policy_sha256 != (
            self.runtime_library_policy_sha256
        ):
            raise ValueError("summary runtime library policy binding differs")
        _positive_int(self.tensor_bytes, "summary tensor_bytes")
        _positive_int(self.output_evaluations, "summary output_evaluations")
        _positive_int(self.vector_dimension, "summary vector_dimension")
        _positive_int(self.fresh_processes, "summary fresh_processes")
        if len(self.worker_results) != self.fresh_processes:
            raise ValueError("worker result count differs from fresh processes")
        if not self.tensor_shape:
            raise ValueError("summary tensor shape must be non-empty")
        for dimension in self.tensor_shape:
            _positive_int(dimension, "summary tensor dimension")
        _positive_int(
            self.maximum_worker_ru_maxrss_bytes,
            "maximum_worker_ru_maxrss_bytes",
        )
        if self.maximum_supervisor_sampled_rss_bytes is not None:
            _positive_int(
                self.maximum_supervisor_sampled_rss_bytes,
                "maximum_supervisor_sampled_rss_bytes",
            )
        if not self.worker_rss_scope.strip() or not self.supervisor_rss_scope.strip():
            raise ValueError("summary RSS scopes must be explicit")
        if not self.host_identity or content_sha256(self.host_identity) != (
            self.host_fingerprint_sha256
        ):
            raise ValueError("summary host identity binding differs")
        if self.worker_environment_identity.identity_sha256 != (
            self.worker_environment_identity_sha256
        ):
            raise ValueError("summary worker environment binding differs")
        expected_distribution = (
            "onnxruntime"
            if self.backend is OnnxBenchmarkBackend.CPU
            else "onnxruntime-gpu"
        )
        if self.onnxruntime_distribution_name != expected_distribution:
            raise ValueError("summary ONNX Runtime distribution differs")
        if not self.onnxruntime_distribution_version.strip():
            raise ValueError("summary ONNX Runtime version is empty")
        expected_runtime_decision = (
            "DISCOVERY_ONLY"
            if self.runtime_library_policy.allow_discovery_only
            and not self.runtime_library_policy.expected_binaries
            else "PASS"
        )
        if self.runtime_library_decision != expected_runtime_decision:
            raise ValueError("summary runtime library decision differs")
        _positive_int(
            self.maximum_runtime_library_bytes_hashed,
            "maximum_runtime_library_bytes_hashed",
        )
        if self.backend is OnnxBenchmarkBackend.CPU:
            if (
                self.gpu_telemetry is not None
                or self.unrelated_gpu_work_excluded_by_operator is not None
            ):
                raise ValueError("CPU summary cannot contain GPU telemetry")
        elif (
            self.gpu_telemetry is None
            or self.unrelated_gpu_work_excluded_by_operator is None
        ):
            raise ValueError("CUDA summary requires GPU telemetry and declaration")
        _validate_summary_aggregation(self)

    @property
    def summary_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend.value,
            "policy": self.policy.to_dict(),
            "policy_sha256": self.policy_sha256,
            "request_sha256": self.request_sha256,
            "model_sha256": self.model_sha256,
            "backend_config_sha256": self.backend_config_sha256,
            "backend_config_file_sha256": (
                self.backend_config_file_sha256
            ),
            "preprocessing_config_sha256": (
                self.preprocessing_config_sha256
            ),
            "preprocessing_file_sha256": self.preprocessing_file_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "runtime_library_policy": self.runtime_library_policy.to_dict(),
            "runtime_library_policy_sha256": (
                self.runtime_library_policy_sha256
            ),
            "runtime_library_policy_file_sha256": (
                self.runtime_library_policy_file_sha256
            ),
            "code_revision": self.code_revision,
            "artifact_content_sha256": list(self.artifact_content_sha256),
            "tensor_sha256": self.tensor_sha256,
            "tensor_bytes": self.tensor_bytes,
            "tensor_shape": list(self.tensor_shape),
            "vector_dimension": self.vector_dimension,
            "output_sha256": self.output_sha256,
            "output_evaluations": self.output_evaluations,
            "fresh_processes": self.fresh_processes,
            "process_wall_time": self.process_wall_time.to_dict(),
            "dependency_import_time": self.dependency_import_time.to_dict(),
            "session_construction_time": (
                self.session_construction_time.to_dict()
            ),
            "preprocessing_time": self.preprocessing_time.to_dict(),
            "first_preprocessed_inference_time": (
                self.first_preprocessed_inference_time.to_dict()
            ),
            "warm_preprocessed_inference_time": (
                self.warm_preprocessed_inference_time.to_dict()
            ),
            "end_to_end_inference_time": (
                self.end_to_end_inference_time.to_dict()
            ),
            "maximum_worker_ru_maxrss_bytes": (
                self.maximum_worker_ru_maxrss_bytes
            ),
            "maximum_supervisor_sampled_rss_bytes": (
                self.maximum_supervisor_sampled_rss_bytes
            ),
            "worker_rss_scope": self.worker_rss_scope,
            "supervisor_rss_scope": self.supervisor_rss_scope,
            "host_identity": dict(sorted(self.host_identity.items())),
            "host_fingerprint_sha256": self.host_fingerprint_sha256,
            "worker_environment_identity": (
                self.worker_environment_identity.to_dict()
            ),
            "worker_environment_identity_sha256": (
                self.worker_environment_identity_sha256
            ),
            "onnxruntime_distribution_name": (
                self.onnxruntime_distribution_name
            ),
            "onnxruntime_distribution_version": (
                self.onnxruntime_distribution_version
            ),
            "runtime_library_binary_set_sha256": (
                self.runtime_library_binary_set_sha256
            ),
            "runtime_library_decision": self.runtime_library_decision,
            "runtime_library_provenance_time": (
                self.runtime_library_provenance_time.to_dict()
            ),
            "maximum_runtime_library_bytes_hashed": (
                self.maximum_runtime_library_bytes_hashed
            ),
            "gpu_telemetry": (
                None
                if self.gpu_telemetry is None
                else self.gpu_telemetry.to_dict()
            ),
            "unrelated_gpu_work_excluded_by_operator": (
                self.unrelated_gpu_work_excluded_by_operator
            ),
            "worker_results": list(self.worker_results),
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> OnnxInferenceBenchmarkSummary:
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("ONNX benchmark summary keys mismatch")
        policy = payload["policy"]
        if not isinstance(policy, dict):
            raise TypeError("ONNX benchmark policy must be an object")
        timing_fields = (
            "process_wall_time",
            "dependency_import_time",
            "session_construction_time",
            "preprocessing_time",
            "first_preprocessed_inference_time",
            "warm_preprocessed_inference_time",
            "end_to_end_inference_time",
            "runtime_library_provenance_time",
        )
        values = dict(payload)
        values["backend"] = OnnxBenchmarkBackend(payload["backend"])
        values["policy"] = OnnxInferenceBenchmarkPolicy.from_dict(policy)
        for name in timing_fields:
            item = payload[name]
            if not isinstance(item, dict):
                raise TypeError(f"{name} must be an object")
            values[name] = TimingSummary.from_dict(item)
        artifacts = payload["artifact_content_sha256"]
        tensor_shape = payload["tensor_shape"]
        worker_results = payload["worker_results"]
        host_identity = payload["host_identity"]
        worker_environment = payload["worker_environment_identity"]
        runtime_library_policy = payload["runtime_library_policy"]
        if (
            not isinstance(artifacts, list)
            or not isinstance(tensor_shape, list)
            or not isinstance(worker_results, list)
            or not isinstance(host_identity, dict)
            or not isinstance(worker_environment, dict)
            or not isinstance(runtime_library_policy, dict)
        ):
            raise TypeError("ONNX benchmark summary collection type mismatch")
        values["artifact_content_sha256"] = tuple(artifacts)
        values["tensor_shape"] = tuple(tensor_shape)
        values["worker_results"] = tuple(worker_results)
        values["host_identity"] = dict(host_identity)
        values["worker_environment_identity"] = (
            WorkerEnvironmentIdentity.from_dict(worker_environment)
        )
        values["runtime_library_policy"] = RuntimeLibraryPolicy.from_dict(
            runtime_library_policy
        )
        gpu = payload["gpu_telemetry"]
        if gpu is not None:
            if not isinstance(gpu, dict):
                raise TypeError("GPU telemetry must be an object or null")
            values["gpu_telemetry"] = GpuTelemetrySummary.from_dict(gpu)
        return cls(**values)


def benchmark_onnx_inference(
    *,
    backend: OnnxBenchmarkBackend,
    model_path: Path,
    backend_config_path: Path,
    preprocessing_path: Path,
    artifact_paths: tuple[Path, ...],
    dependency_lock_path: Path,
    runtime_library_policy_path: Path,
    code_revision: str,
    policy: OnnxInferenceBenchmarkPolicy,
) -> OnnxInferenceBenchmarkSummary:
    """Run identical inference work in independently supervised workers."""

    if not code_revision.strip():
        raise ValueError("code_revision must be non-empty")
    if not artifact_paths:
        raise ValueError("at least one benchmark artifact is required")
    if len(artifact_paths) > policy.maximum_artifacts:
        raise ValueError("benchmark artifact count exceeds policy cap")
    if backend is OnnxBenchmarkBackend.CPU and policy.gpu_device_index is not None:
        raise ValueError("CPU benchmark cannot attach GPU telemetry")
    if backend is OnnxBenchmarkBackend.CUDA and policy.gpu_device_index is None:
        raise ValueError("CUDA benchmark requires scoped GPU telemetry")

    from operations.onnx_backend import (
        ImagePreprocessingConfig,
        OnnxRuntimeBackendConfig,
    )

    backend_payload = read_strict_json_object(backend_config_path)
    preprocessing_payload = read_strict_json_object(preprocessing_path)
    backend_config_file = _regular_file(
        backend_config_path,
        "backend config",
    )
    preprocessing_file = _regular_file(
        preprocessing_path,
        "preprocessing config",
    )
    backend_config_file_sha256 = sha256_file(backend_config_file)
    preprocessing_file_sha256 = sha256_file(preprocessing_file)
    backend_config = OnnxRuntimeBackendConfig.from_dict(backend_payload)
    preprocessing = ImagePreprocessingConfig.from_dict(preprocessing_payload)
    if backend_config.preprocessing_config_sha256 != (
        preprocessing.config_sha256
    ):
        raise ValueError("backend and preprocessing semantics differ")
    model = _regular_file(model_path, "model")
    lock = _regular_file(dependency_lock_path, "dependency lock")
    runtime_library_policy_file = _regular_file(
        runtime_library_policy_path,
        "runtime library policy",
    )
    runtime_library_policy_payload = read_strict_json_object(
        runtime_library_policy_file
    )
    runtime_library_policy = RuntimeLibraryPolicy.from_dict(
        runtime_library_policy_payload
    )
    runtime_library_policy_file_sha256 = sha256_file(
        runtime_library_policy_file
    )
    artifacts = tuple(
        _artifact_request(index, path)
        for index, path in enumerate(artifact_paths)
    )
    if sum(item["byte_size"] for item in artifacts) > (
        policy.maximum_total_artifact_bytes
    ):
        raise ValueError("benchmark artifacts exceed total byte cap")
    if len(artifacts) > backend_config.maximum_batch_size:
        raise ValueError("benchmark batch exceeds backend maximum")
    nominal_tensor_bytes = (
        len(artifacts)
        * preprocessing.channels
        * preprocessing.height
        * preprocessing.width
        * 4
    )
    if nominal_tensor_bytes > policy.maximum_tensor_bytes:
        raise ValueError("benchmark tensor exceeds byte cap")
    model_sha256 = sha256_file(model)
    lock_sha256 = sha256_file(lock)
    worker_environment, worker_environment_identity = (
        build_sanitized_worker_environment(os.environ)
    )
    request = {
        "schema_version": "cvi.onnx_inference_worker_request.v3",
        "backend": backend.value,
        "model_path": str(model),
        "model_sha256": model_sha256,
        "backend_config": backend_payload,
        "backend_config_sha256": backend_config.config_sha256,
        "backend_config_file_sha256": backend_config_file_sha256,
        "preprocessing": preprocessing_payload,
        "preprocessing_config_sha256": preprocessing.config_sha256,
        "preprocessing_file_sha256": preprocessing_file_sha256,
        "artifacts": list(artifacts),
        "dependency_lock_path": str(lock),
        "dependency_lock_sha256": lock_sha256,
        "runtime_library_policy": runtime_library_policy_payload,
        "runtime_library_policy_sha256": (
            runtime_library_policy.policy_sha256
        ),
        "runtime_library_policy_file_sha256": (
            runtime_library_policy_file_sha256
        ),
        "code_revision": code_revision,
        "benchmark_policy_sha256": policy.policy_sha256,
        "worker_environment_identity": (
            worker_environment_identity.to_dict()
        ),
        "worker_environment_identity_sha256": (
            worker_environment_identity.identity_sha256
        ),
        "unrelated_system_work_excluded_by_operator": (
            policy.unrelated_system_work_excluded_by_operator
        ),
        "maximum_artifacts": policy.maximum_artifacts,
        "maximum_total_artifact_bytes": (
            policy.maximum_total_artifact_bytes
        ),
        "maximum_tensor_bytes": policy.maximum_tensor_bytes,
        "model_warmup_iterations": policy.model_warmup_iterations,
        "model_repeat_iterations": policy.model_repeat_iterations,
        "end_to_end_repeat_iterations": (
            policy.end_to_end_repeat_iterations
        ),
    }
    request_sha256 = content_sha256(request)

    with TemporaryDirectory(prefix="cvi-onnx-benchmark-") as temporary:
        root = Path(temporary)
        request_path = root / "request.json"
        request_path.write_text(
            _canonical_pretty_json(request),
            encoding="utf-8",
        )
        os.chmod(request_path, 0o600)

        def run_workers() -> tuple[
            tuple[SupervisedProcessResult, dict[str, Any]], ...
        ]:
            completed: list[tuple[SupervisedProcessResult, dict[str, Any]]] = []
            for index in range(policy.fresh_processes):
                result_path = root / f"worker-{index:03d}.json"
                command = (
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    ISOLATED_WORKER_BOOTSTRAP,
                    "operations.onnx_inference_benchmark",
                    str(request_path),
                    "--worker-request",
                    str(request_path),
                    "--worker-result",
                    str(result_path),
                )
                supervised = run_supervised_process(
                    command,
                    policy=policy.supervisor,
                    environment=worker_environment,
                )
                if supervised.status is not SupervisedProcessStatus.COMPLETED:
                    raise RuntimeError(
                        "ONNX benchmark worker failed: "
                        f"{supervised.status.value} rc={supervised.return_code}"
                    )
                measurement = _read_worker_result(
                    result_path,
                    maximum_bytes=policy.maximum_worker_result_bytes,
                )
                _validate_worker_result(
                    measurement,
                    request_sha256=request_sha256,
                    backend=backend,
                    model_sha256=model_sha256,
                    backend_config_sha256=backend_config.config_sha256,
                    backend_config_file_sha256=(
                        backend_config_file_sha256
                    ),
                    preprocessing_config_sha256=preprocessing.config_sha256,
                    preprocessing_file_sha256=preprocessing_file_sha256,
                    dependency_lock_sha256=lock_sha256,
                    runtime_library_policy=runtime_library_policy,
                    runtime_library_policy_sha256=(
                        runtime_library_policy.policy_sha256
                    ),
                    runtime_library_policy_file_sha256=(
                        runtime_library_policy_file_sha256
                    ),
                    code_revision=code_revision,
                    vector_dimension=backend_config.vector_dimension,
                    artifact_content_sha256=tuple(
                        item["content_sha256"] for item in artifacts
                    ),
                    expected_evaluations=(
                        1
                        + policy.model_warmup_iterations
                        + policy.model_repeat_iterations
                        + policy.end_to_end_repeat_iterations
                    ),
                    benchmark_policy_sha256=policy.policy_sha256,
                    unrelated_system_work_excluded_by_operator=(
                        policy.unrelated_system_work_excluded_by_operator
                    ),
                    worker_environment_identity=(
                        worker_environment_identity
                    ),
                    worker_environment_identity_sha256=(
                        worker_environment_identity.identity_sha256
                    ),
                )
                completed.append((supervised, measurement))
            return tuple(completed)

        gpu_telemetry: GpuTelemetrySummary | None = None
        if backend is OnnxBenchmarkBackend.CUDA:
            worker_pairs, gpu_telemetry = monitor_operation(
                run_workers,
                device_index=policy.gpu_device_index,
                interval_seconds=policy.gpu_telemetry_interval_seconds,
            )
        else:
            worker_pairs = run_workers()

    _verify_file_identity(model, model_sha256)
    _verify_file_identity(lock, lock_sha256)
    _verify_file_identity(
        runtime_library_policy_file,
        runtime_library_policy_file_sha256,
    )
    _verify_file_identity(
        Path(worker_environment_identity.python_executable_resolved_path),
        worker_environment_identity.python_executable_sha256,
        expected_size=worker_environment_identity.python_executable_bytes,
    )
    _verify_file_identity(
        backend_config_file,
        backend_config_file_sha256,
    )
    _verify_file_identity(
        preprocessing_file,
        preprocessing_file_sha256,
    )
    for artifact, path in zip(artifacts, artifact_paths, strict=True):
        _verify_file_identity(
            _regular_file(path, "benchmark artifact"),
            artifact["content_sha256"],
            expected_size=artifact["byte_size"],
        )
    measurements = tuple(pair[1] for pair in worker_pairs)
    _require_identical_field(measurements, "tensor_sha256")
    _require_identical_field(measurements, "tensor_bytes")
    _require_identical_field(measurements, "tensor_shape")
    _require_identical_field(measurements, "output_sha256")
    _require_identical_field(measurements, "host_fingerprint_sha256")
    _require_identical_field(measurements, "host_identity")
    _require_identical_field(
        measurements,
        "worker_environment_identity_sha256",
    )
    _require_identical_field(measurements, "worker_environment_identity")
    _require_identical_field(
        measurements,
        "onnxruntime_distribution_name",
    )
    _require_identical_field(
        measurements,
        "onnxruntime_distribution_version",
    )
    _require_identical_field(
        measurements,
        "runtime_library_binary_set_sha256",
    )
    _require_identical_field(measurements, "runtime_library_decision")
    output_evaluations = sum(item["output_evaluations"] for item in measurements)
    supervised_results = tuple(pair[0] for pair in worker_pairs)
    sampled_rss = tuple(
        item.sampled_peak_rss_bytes
        for item in supervised_results
        if item.sampled_peak_rss_bytes is not None
    )
    worker_payloads = tuple(
        {
            "supervisor": supervised.to_dict(),
            "measurement": measurement,
        }
        for supervised, measurement in worker_pairs
    )
    return OnnxInferenceBenchmarkSummary(
        backend=backend,
        policy=policy,
        policy_sha256=policy.policy_sha256,
        request_sha256=request_sha256,
        model_sha256=model_sha256,
        backend_config_sha256=backend_config.config_sha256,
        backend_config_file_sha256=backend_config_file_sha256,
        preprocessing_config_sha256=preprocessing.config_sha256,
        preprocessing_file_sha256=preprocessing_file_sha256,
        dependency_lock_sha256=lock_sha256,
        runtime_library_policy=runtime_library_policy,
        runtime_library_policy_sha256=(
            runtime_library_policy.policy_sha256
        ),
        runtime_library_policy_file_sha256=(
            runtime_library_policy_file_sha256
        ),
        code_revision=code_revision,
        artifact_content_sha256=tuple(
            item["content_sha256"] for item in artifacts
        ),
        tensor_sha256=measurements[0]["tensor_sha256"],
        tensor_bytes=measurements[0]["tensor_bytes"],
        tensor_shape=tuple(measurements[0]["tensor_shape"]),
        vector_dimension=backend_config.vector_dimension,
        output_sha256=measurements[0]["output_sha256"],
        output_evaluations=output_evaluations,
        fresh_processes=policy.fresh_processes,
        process_wall_time=TimingSummary.from_samples(
            tuple(item.wall_time_ns for item in supervised_results)
        ),
        dependency_import_time=_timing(measurements, "dependency_import_ns"),
        session_construction_time=_timing(
            measurements,
            "session_construction_ns",
        ),
        preprocessing_time=_timing(measurements, "preprocessing_ns"),
        first_preprocessed_inference_time=_timing(
            measurements,
            "first_preprocessed_inference_ns",
        ),
        warm_preprocessed_inference_time=TimingSummary.from_samples(
            tuple(
                sample
                for item in measurements
                for sample in item["warm_preprocessed_inference_ns"]
            )
        ),
        end_to_end_inference_time=TimingSummary.from_samples(
            tuple(
                sample
                for item in measurements
                for sample in item["end_to_end_inference_ns"]
            )
        ),
        maximum_worker_ru_maxrss_bytes=max(
            item["worker_ru_maxrss_bytes"] for item in measurements
        ),
        maximum_supervisor_sampled_rss_bytes=(
            max(sampled_rss) if sampled_rss else None
        ),
        worker_rss_scope="linux-worker-process-ru_maxrss-high-water-mark",
        supervisor_rss_scope=(
            supervised_results[0].rss_scope
            if all(
                item.rss_scope == supervised_results[0].rss_scope
                for item in supervised_results
            )
            else "MIXED_UNAVAILABLE"
        ),
        host_identity=dict(measurements[0]["host_identity"]),
        host_fingerprint_sha256=measurements[0][
            "host_fingerprint_sha256"
        ],
        worker_environment_identity=worker_environment_identity,
        worker_environment_identity_sha256=(
            worker_environment_identity.identity_sha256
        ),
        onnxruntime_distribution_name=measurements[0][
            "onnxruntime_distribution_name"
        ],
        onnxruntime_distribution_version=measurements[0][
            "onnxruntime_distribution_version"
        ],
        runtime_library_binary_set_sha256=measurements[0][
            "runtime_library_binary_set_sha256"
        ],
        runtime_library_decision=measurements[0][
            "runtime_library_decision"
        ],
        runtime_library_provenance_time=_timing(
            measurements,
            "runtime_library_provenance_wall_time_ns",
        ),
        maximum_runtime_library_bytes_hashed=max(
            item["runtime_library_bytes_hashed"] for item in measurements
        ),
        gpu_telemetry=gpu_telemetry,
        unrelated_gpu_work_excluded_by_operator=(
            policy.unrelated_gpu_work_excluded_by_operator
        ),
        worker_results=worker_payloads,
    )


def run_worker(request_path: Path, result_path: Path) -> None:
    """Execute one authenticated worker request and publish one private result."""

    worker_started_ns = perf_counter_ns()
    request_file = _regular_file(request_path, "worker request")
    if request_file.stat().st_size > _MAXIMUM_WORKER_REQUEST_BYTES:
        raise ValueError("worker request exceeds byte cap")
    request = read_strict_json_object(request_file)
    request_sha256 = content_sha256(request)
    _validate_worker_request(request)
    expected_worker_environment = WorkerEnvironmentIdentity.from_dict(
        request["worker_environment_identity"]
    )
    if expected_worker_environment.identity_sha256 != request[
        "worker_environment_identity_sha256"
    ]:
        raise ValueError("worker environment request binding differs")
    observed_worker_environment = validate_current_worker_environment(
        expected_worker_environment
    )
    runtime_library_policy = RuntimeLibraryPolicy.from_dict(
        request["runtime_library_policy"]
    )
    if runtime_library_policy.policy_sha256 != request[
        "runtime_library_policy_sha256"
    ]:
        raise ValueError("runtime library policy request binding differs")
    runtime_library_tracker = RuntimeLibraryTracker(runtime_library_policy)
    import_started_ns = perf_counter_ns()
    from operations.onnx_backend import (
        ImagePreprocessingConfig,
        OnnxRuntimeBackendConfig,
        OnnxRuntimeCpuBackend,
        OnnxRuntimeCudaBackend,
        onnxruntime_distribution_identity,
        preprocess_image_batch,
    )

    backend_config = OnnxRuntimeBackendConfig.from_dict(
        request["backend_config"]
    )
    preprocessing = ImagePreprocessingConfig.from_dict(
        request["preprocessing"]
    )
    dependency_import_ns = perf_counter_ns() - import_started_ns
    onnxruntime_distribution = onnxruntime_distribution_identity(
        require_gpu=(request["backend"] == OnnxBenchmarkBackend.CUDA.value)
    )
    runtime_library_tracker.capture(
        RuntimeLibraryPhase.DEPENDENCIES_IMPORTED
    )
    if backend_config.config_sha256 != request["backend_config_sha256"]:
        raise ValueError("worker backend config hash mismatch")
    if preprocessing.config_sha256 != request["preprocessing_config_sha256"]:
        raise ValueError("worker preprocessing config hash mismatch")
    model = _regular_file(Path(request["model_path"]), "model")
    lock = _regular_file(
        Path(request["dependency_lock_path"]),
        "dependency lock",
    )
    _verify_file_identity(model, request["model_sha256"])
    _verify_file_identity(lock, request["dependency_lock_sha256"])
    paths = tuple(
        _verified_worker_artifact(item) for item in request["artifacts"]
    )
    if len(paths) > backend_config.maximum_batch_size:
        raise ValueError("worker batch exceeds backend maximum")
    nominal_tensor_bytes = (
        len(paths)
        * preprocessing.channels
        * preprocessing.height
        * preprocessing.width
        * 4
    )
    if nominal_tensor_bytes > request["maximum_tensor_bytes"]:
        raise ValueError("worker tensor exceeds request byte cap")
    backend_class = (
        OnnxRuntimeCpuBackend
        if request["backend"] == OnnxBenchmarkBackend.CPU.value
        else OnnxRuntimeCudaBackend
    )
    session_started_ns = perf_counter_ns()
    backend = backend_class(
        model_path=model,
        config=backend_config,
        preprocessing=preprocessing,
    )
    session_construction_ns = perf_counter_ns() - session_started_ns
    runtime_library_tracker.capture(RuntimeLibraryPhase.SESSION_READY)
    preprocessing_started_ns = perf_counter_ns()
    tensor = preprocess_image_batch(paths, preprocessing)
    preprocessing_ns = perf_counter_ns() - preprocessing_started_ns
    tensor_bytes_view = memoryview(tensor).cast("B")
    tensor_sha256 = hashlib.sha256(tensor_bytes_view).hexdigest()

    output_sha256: str | None = None
    evaluations = 0

    def evaluate_preprocessed() -> int:
        nonlocal output_sha256, evaluations
        backend.synchronize()
        started = perf_counter_ns()
        rows = backend.infer_preprocessed_batch(tensor)
        backend.synchronize()
        elapsed = perf_counter_ns() - started
        output_sha256 = _bind_output_digest(output_sha256, rows)
        evaluations += 1
        return elapsed

    def evaluate_end_to_end() -> int:
        nonlocal output_sha256, evaluations
        backend.synchronize()
        started = perf_counter_ns()
        rows = backend.infer_batch(paths)
        backend.synchronize()
        elapsed = perf_counter_ns() - started
        output_sha256 = _bind_output_digest(output_sha256, rows)
        evaluations += 1
        return elapsed

    first_ns = evaluate_preprocessed()
    runtime_library_tracker.capture(RuntimeLibraryPhase.FIRST_OUTPUT_READY)
    for _ in range(request["model_warmup_iterations"]):
        evaluate_preprocessed()
    warm_samples = tuple(
        evaluate_preprocessed()
        for _ in range(request["model_repeat_iterations"])
    )
    end_to_end_samples = tuple(
        evaluate_end_to_end()
        for _ in range(request["end_to_end_repeat_iterations"])
    )
    runtime_library_tracker.capture(RuntimeLibraryPhase.FINAL_OUTPUT_READY)
    runtime_library_manifest = runtime_library_tracker.finalize()
    if runtime_library_manifest.decision == "FAIL":
        raise RuntimeError("runtime library provenance policy failed")
    _verify_file_identity(model, request["model_sha256"])
    _verify_file_identity(lock, request["dependency_lock_sha256"])
    for item, path in zip(request["artifacts"], paths, strict=True):
        _verify_file_identity(
            path,
            item["content_sha256"],
            expected_size=item["byte_size"],
        )
    if output_sha256 is None:
        raise RuntimeError("worker produced no inference output")
    host_identity = _host_identity()
    result = {
        "schema_version": "cvi.onnx_inference_worker_result.v3",
        "request_sha256": request_sha256,
        "backend": request["backend"],
        "model_sha256": request["model_sha256"],
        "backend_config_sha256": request["backend_config_sha256"],
        "backend_config_file_sha256": request[
            "backend_config_file_sha256"
        ],
        "preprocessing_config_sha256": request[
            "preprocessing_config_sha256"
        ],
        "preprocessing_file_sha256": request[
            "preprocessing_file_sha256"
        ],
        "dependency_lock_sha256": request["dependency_lock_sha256"],
        "runtime_library_policy_sha256": request[
            "runtime_library_policy_sha256"
        ],
        "runtime_library_policy_file_sha256": request[
            "runtime_library_policy_file_sha256"
        ],
        "code_revision": request["code_revision"],
        "benchmark_policy_sha256": request["benchmark_policy_sha256"],
        "worker_environment_identity": (
            observed_worker_environment.to_dict()
        ),
        "worker_environment_identity_sha256": (
            observed_worker_environment.identity_sha256
        ),
        "onnxruntime_distribution_name": onnxruntime_distribution[0],
        "onnxruntime_distribution_version": onnxruntime_distribution[1],
        "runtime_library_manifest": runtime_library_manifest.to_dict(),
        "runtime_library_manifest_sha256": (
            runtime_library_manifest.manifest_sha256
        ),
        "runtime_library_binary_set_sha256": (
            runtime_library_manifest.binary_set_sha256
        ),
        "runtime_library_decision": runtime_library_manifest.decision,
        "runtime_library_provenance_wall_time_ns": (
            runtime_library_manifest.provenance_wall_time_ns
        ),
        "runtime_library_bytes_hashed": (
            runtime_library_manifest.binary_bytes_hashed
        ),
        "unrelated_system_work_excluded_by_operator": request[
            "unrelated_system_work_excluded_by_operator"
        ],
        "backend_identity": backend.identity.to_dict(),
        "actual_providers": list(backend.actual_providers),
        "actual_provider_options_sha256": content_sha256(
            backend.actual_provider_options
        ),
        "artifact_content_sha256": [
            item["content_sha256"] for item in request["artifacts"]
        ],
        "tensor_sha256": tensor_sha256,
        "tensor_bytes": tensor.nbytes,
        "tensor_shape": list(tensor.shape),
        "vector_dimension": backend_config.vector_dimension,
        "output_sha256": output_sha256,
        "output_evaluations": evaluations,
        "dependency_import_ns": dependency_import_ns,
        "session_construction_ns": session_construction_ns,
        "preprocessing_ns": preprocessing_ns,
        "first_preprocessed_inference_ns": first_ns,
        "warm_preprocessed_inference_ns": list(warm_samples),
        "end_to_end_inference_ns": list(end_to_end_samples),
        "worker_ru_maxrss_bytes": _worker_ru_maxrss_bytes(),
        "worker_wall_time_ns": perf_counter_ns() - worker_started_ns,
        "pid": os.getpid(),
        "host_identity": host_identity,
        "host_fingerprint_sha256": content_sha256(host_identity),
        "interpretation": (
            "WORKER_MEASUREMENT_ONLY_NOT_OPTIMIZATION_PROMOTION"
        ),
    }
    write_private_json_bundle(((result_path, result),))


def _artifact_request(index: int, path: Path) -> dict[str, Any]:
    resolved = _regular_file(path, "benchmark artifact")
    stat = resolved.stat()
    return {
        "ordinal": index,
        "path": str(resolved),
        "content_sha256": sha256_file(resolved),
        "byte_size": stat.st_size,
    }


def _verified_worker_artifact(payload: dict[str, Any]) -> Path:
    expected = {"ordinal", "path", "content_sha256", "byte_size"}
    if set(payload) != expected:
        raise ValueError("worker artifact keys mismatch")
    _nonnegative_int(payload["ordinal"], "artifact ordinal")
    _positive_int(payload["byte_size"], "artifact byte_size")
    _validate_sha256(payload["content_sha256"])
    path = _regular_file(Path(payload["path"]), "benchmark artifact")
    _verify_file_identity(
        path,
        payload["content_sha256"],
        expected_size=payload["byte_size"],
    )
    return path


def _validate_worker_request(payload: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "backend",
        "model_path",
        "model_sha256",
        "backend_config",
        "backend_config_sha256",
        "backend_config_file_sha256",
        "preprocessing",
        "preprocessing_config_sha256",
        "preprocessing_file_sha256",
        "artifacts",
        "dependency_lock_path",
        "dependency_lock_sha256",
        "runtime_library_policy",
        "runtime_library_policy_sha256",
        "runtime_library_policy_file_sha256",
        "code_revision",
        "benchmark_policy_sha256",
        "worker_environment_identity",
        "worker_environment_identity_sha256",
        "unrelated_system_work_excluded_by_operator",
        "maximum_artifacts",
        "maximum_total_artifact_bytes",
        "maximum_tensor_bytes",
        "model_warmup_iterations",
        "model_repeat_iterations",
        "end_to_end_repeat_iterations",
    }
    if set(payload) != expected:
        raise ValueError("worker request keys mismatch")
    if payload["schema_version"] != "cvi.onnx_inference_worker_request.v3":
        raise ValueError("unsupported ONNX worker request schema")
    OnnxBenchmarkBackend(payload["backend"])
    for name in (
        "model_sha256",
        "backend_config_sha256",
        "backend_config_file_sha256",
        "preprocessing_config_sha256",
        "preprocessing_file_sha256",
        "dependency_lock_sha256",
        "runtime_library_policy_sha256",
        "runtime_library_policy_file_sha256",
        "benchmark_policy_sha256",
        "worker_environment_identity_sha256",
    ):
        _validate_sha256(payload[name])
    worker_environment = payload["worker_environment_identity"]
    if not isinstance(worker_environment, dict):
        raise TypeError("worker environment identity must be an object")
    parsed_worker_environment = WorkerEnvironmentIdentity.from_dict(
        worker_environment
    )
    if parsed_worker_environment.identity_sha256 != payload[
        "worker_environment_identity_sha256"
    ]:
        raise ValueError("worker environment identity hash mismatch")
    runtime_policy_payload = payload["runtime_library_policy"]
    if not isinstance(runtime_policy_payload, dict):
        raise TypeError("runtime library policy must be an object")
    parsed_runtime_policy = RuntimeLibraryPolicy.from_dict(
        runtime_policy_payload
    )
    if parsed_runtime_policy.policy_sha256 != payload[
        "runtime_library_policy_sha256"
    ]:
        raise ValueError("runtime library policy hash mismatch")
    if not isinstance(payload["backend_config"], dict) or not isinstance(
        payload["preprocessing"], dict
    ):
        raise TypeError("worker configurations must be objects")
    if not isinstance(
        payload["unrelated_system_work_excluded_by_operator"],
        bool,
    ):
        raise TypeError("worker system-work declaration must be boolean")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("worker artifacts must be a non-empty list")
    for item in artifacts:
        if not isinstance(item, dict):
            raise TypeError("worker artifact entries must be objects")
        if set(item) != {"ordinal", "path", "content_sha256", "byte_size"}:
            raise ValueError("worker artifact keys mismatch")
        _nonnegative_int(item["ordinal"], "artifact ordinal")
        _positive_int(item["byte_size"], "artifact byte_size")
        _validate_sha256(item["content_sha256"])
        if not isinstance(item["path"], str) or not item["path"]:
            raise ValueError("worker artifact path must be non-empty")
    ordinals = tuple(item.get("ordinal") for item in artifacts)
    if ordinals != tuple(range(len(artifacts))):
        raise ValueError("worker artifact ordinals must be contiguous")
    _positive_int(payload["maximum_artifacts"], "maximum_artifacts")
    _positive_int(
        payload["maximum_total_artifact_bytes"],
        "maximum_total_artifact_bytes",
    )
    _positive_int(payload["maximum_tensor_bytes"], "maximum_tensor_bytes")
    if payload["maximum_artifacts"] > _MAXIMUM_ARTIFACTS_HARD_CAP:
        raise ValueError("worker artifact cap exceeds hard cap")
    if (
        payload["maximum_total_artifact_bytes"]
        > _MAXIMUM_TOTAL_BYTES_HARD_CAP
        or payload["maximum_tensor_bytes"] > _MAXIMUM_TOTAL_BYTES_HARD_CAP
    ):
        raise ValueError("worker byte cap exceeds hard cap")
    if len(artifacts) > payload["maximum_artifacts"]:
        raise ValueError("worker artifact count exceeds request cap")
    if sum(item.get("byte_size", 0) for item in artifacts) > payload[
        "maximum_total_artifact_bytes"
    ]:
        raise ValueError("worker artifacts exceed request byte cap")
    if not isinstance(payload["code_revision"], str) or not payload[
        "code_revision"
    ].strip():
        raise ValueError("worker code revision must be non-empty")
    _nonnegative_int(
        payload["model_warmup_iterations"],
        "model_warmup_iterations",
    )
    _positive_int(
        payload["model_repeat_iterations"],
        "model_repeat_iterations",
    )
    _positive_int(
        payload["end_to_end_repeat_iterations"],
        "end_to_end_repeat_iterations",
    )
    if (
        payload["model_warmup_iterations"] > 10_000
        or payload["model_repeat_iterations"] > 10_000
        or payload["end_to_end_repeat_iterations"] > 10_000
    ):
        raise ValueError("worker iterations exceed hard cap")


def _read_worker_result(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("worker result must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.stat().st_size > maximum_bytes:
        raise ValueError("worker result exceeds byte cap")
    return read_strict_json_object(resolved)


def _validate_worker_result(
    payload: dict[str, Any],
    *,
    request_sha256: str,
    backend: OnnxBenchmarkBackend,
    model_sha256: str,
    backend_config_sha256: str,
    backend_config_file_sha256: str,
    preprocessing_config_sha256: str,
    preprocessing_file_sha256: str,
    dependency_lock_sha256: str,
    runtime_library_policy: RuntimeLibraryPolicy,
    runtime_library_policy_sha256: str,
    runtime_library_policy_file_sha256: str,
    code_revision: str,
    vector_dimension: int,
    artifact_content_sha256: tuple[str, ...],
    expected_evaluations: int,
    benchmark_policy_sha256: str,
    unrelated_system_work_excluded_by_operator: bool,
    worker_environment_identity: WorkerEnvironmentIdentity,
    worker_environment_identity_sha256: str,
) -> None:
    expected = {
        "schema_version",
        "request_sha256",
        "backend",
        "model_sha256",
        "backend_config_sha256",
        "backend_config_file_sha256",
        "preprocessing_config_sha256",
        "preprocessing_file_sha256",
        "dependency_lock_sha256",
        "runtime_library_policy_sha256",
        "runtime_library_policy_file_sha256",
        "code_revision",
        "benchmark_policy_sha256",
        "worker_environment_identity",
        "worker_environment_identity_sha256",
        "onnxruntime_distribution_name",
        "onnxruntime_distribution_version",
        "runtime_library_manifest",
        "runtime_library_manifest_sha256",
        "runtime_library_binary_set_sha256",
        "runtime_library_decision",
        "runtime_library_provenance_wall_time_ns",
        "runtime_library_bytes_hashed",
        "unrelated_system_work_excluded_by_operator",
        "backend_identity",
        "actual_providers",
        "actual_provider_options_sha256",
        "artifact_content_sha256",
        "tensor_sha256",
        "tensor_bytes",
        "tensor_shape",
        "vector_dimension",
        "output_sha256",
        "output_evaluations",
        "dependency_import_ns",
        "session_construction_ns",
        "preprocessing_ns",
        "first_preprocessed_inference_ns",
        "warm_preprocessed_inference_ns",
        "end_to_end_inference_ns",
        "worker_ru_maxrss_bytes",
        "worker_wall_time_ns",
        "pid",
        "host_identity",
        "host_fingerprint_sha256",
        "interpretation",
    }
    if set(payload) != expected:
        raise ValueError("worker result keys mismatch")
    bindings = {
        "request_sha256": request_sha256,
        "backend": backend.value,
        "model_sha256": model_sha256,
        "backend_config_sha256": backend_config_sha256,
        "backend_config_file_sha256": backend_config_file_sha256,
        "preprocessing_config_sha256": preprocessing_config_sha256,
        "preprocessing_file_sha256": preprocessing_file_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
        "runtime_library_policy_sha256": runtime_library_policy_sha256,
        "runtime_library_policy_file_sha256": (
            runtime_library_policy_file_sha256
        ),
        "code_revision": code_revision,
        "vector_dimension": vector_dimension,
        "benchmark_policy_sha256": benchmark_policy_sha256,
        "worker_environment_identity_sha256": (
            worker_environment_identity_sha256
        ),
        "unrelated_system_work_excluded_by_operator": (
            unrelated_system_work_excluded_by_operator
        ),
        "output_evaluations": expected_evaluations,
        "interpretation": (
            "WORKER_MEASUREMENT_ONLY_NOT_OPTIMIZATION_PROMOTION"
        ),
    }
    for name, expected_value in bindings.items():
        if payload[name] != expected_value:
            raise ValueError(f"worker result {name} mismatch")
    if payload["artifact_content_sha256"] != list(artifact_content_sha256):
        raise ValueError("worker result artifact contents mismatch")
    if payload["schema_version"] != "cvi.onnx_inference_worker_result.v3":
        raise ValueError("unsupported ONNX worker result schema")
    for name in (
        "actual_provider_options_sha256",
        "tensor_sha256",
        "output_sha256",
        "host_fingerprint_sha256",
        "worker_environment_identity_sha256",
        "runtime_library_policy_sha256",
        "runtime_library_policy_file_sha256",
        "runtime_library_manifest_sha256",
        "runtime_library_binary_set_sha256",
    ):
        _validate_sha256(payload[name])
    environment_payload = payload["worker_environment_identity"]
    if not isinstance(environment_payload, dict):
        raise TypeError("worker environment identity must be an object")
    observed_environment = WorkerEnvironmentIdentity.from_dict(
        environment_payload
    )
    if observed_environment != worker_environment_identity or (
        observed_environment.identity_sha256
        != worker_environment_identity_sha256
    ):
        raise ValueError("worker result environment identity differs")
    expected_distribution = (
        "onnxruntime"
        if backend is OnnxBenchmarkBackend.CPU
        else "onnxruntime-gpu"
    )
    if payload["onnxruntime_distribution_name"] != expected_distribution:
        raise ValueError("worker ONNX Runtime distribution differs")
    if not isinstance(payload["onnxruntime_distribution_version"], str) or (
        not payload["onnxruntime_distribution_version"].strip()
    ):
        raise ValueError("worker ONNX Runtime version is empty")
    runtime_manifest_payload = payload["runtime_library_manifest"]
    if not isinstance(runtime_manifest_payload, dict):
        raise TypeError("runtime library manifest must be an object")
    runtime_manifest = RuntimeLibraryManifest.from_dict(
        runtime_manifest_payload
    )
    if runtime_manifest.manifest_sha256 != payload[
        "runtime_library_manifest_sha256"
    ]:
        raise ValueError("runtime library manifest hash mismatch")
    if runtime_manifest.policy_sha256 != runtime_library_policy_sha256 or (
        runtime_library_policy.policy_sha256
        != runtime_library_policy_sha256
    ):
        raise ValueError("runtime library manifest policy binding differs")
    expected_runtime_decision = (
        "DISCOVERY_ONLY"
        if runtime_library_policy.allow_discovery_only
        and not runtime_library_policy.expected_binaries
        else "PASS"
    )
    if runtime_manifest.decision != expected_runtime_decision or payload[
        "runtime_library_decision"
    ] != expected_runtime_decision:
        raise ValueError("runtime library decision differs")
    if runtime_manifest.binary_set_sha256 != payload[
        "runtime_library_binary_set_sha256"
    ]:
        raise ValueError("runtime library binary set binding differs")
    if runtime_manifest.provenance_wall_time_ns != payload[
        "runtime_library_provenance_wall_time_ns"
    ] or runtime_manifest.binary_bytes_hashed != payload[
        "runtime_library_bytes_hashed"
    ]:
        raise ValueError("runtime library work accounting differs")
    for name in (
        "tensor_bytes",
        "vector_dimension",
        "dependency_import_ns",
        "session_construction_ns",
        "preprocessing_ns",
        "first_preprocessed_inference_ns",
        "worker_ru_maxrss_bytes",
        "worker_wall_time_ns",
        "pid",
        "runtime_library_provenance_wall_time_ns",
        "runtime_library_bytes_hashed",
    ):
        _positive_int(payload[name], name)
    for name in (
        "warm_preprocessed_inference_ns",
        "end_to_end_inference_ns",
    ):
        values = payload[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"worker result {name} must be non-empty")
        for value in values:
            _positive_int(value, name)
    if not isinstance(payload["tensor_shape"], list) or not payload[
        "tensor_shape"
    ]:
        raise ValueError("worker tensor shape must be non-empty")
    for dimension in payload["tensor_shape"]:
        _positive_int(dimension, "tensor dimension")
    if not isinstance(payload["actual_providers"], list) or not payload[
        "actual_providers"
    ]:
        raise ValueError("worker actual providers must be non-empty")
    if not isinstance(payload["backend_identity"], dict):
        raise TypeError("worker backend identity must be an object")
    from operations.embedding_producer import EmbeddingBackendIdentity

    identity = EmbeddingBackendIdentity.from_dict(payload["backend_identity"])
    if identity.backend_config_sha256 != backend_config_sha256:
        raise ValueError("worker backend identity config binding differs")
    expected_providers = (
        ["CPUExecutionProvider"]
        if backend is OnnxBenchmarkBackend.CPU
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    if payload["actual_providers"] != expected_providers:
        raise ValueError("worker actual provider order differs")
    if not isinstance(payload["host_identity"], dict) or not payload[
        "host_identity"
    ]:
        raise ValueError("worker host identity must be a non-empty object")
    if content_sha256(payload["host_identity"]) != payload[
        "host_fingerprint_sha256"
    ]:
        raise ValueError("worker host identity binding differs")


def _bind_output_digest(
    previous: str | None,
    rows: tuple[memoryview, ...],
) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.cast("B"))
    current = digest.hexdigest()
    if previous is not None and current != previous:
        raise RuntimeError("ONNX output changed across identical evaluations")
    return current


def _worker_ru_maxrss_bytes() -> int:
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith("linux"):
        return int(value) * 1024
    if sys.platform == "darwin":
        return int(value)
    raise RuntimeError("worker ru_maxrss unit is unsupported on this platform")


def _timing(
    measurements: tuple[dict[str, Any], ...],
    field: str,
) -> TimingSummary:
    return TimingSummary.from_samples(tuple(item[field] for item in measurements))


def _validate_summary_aggregation(
    summary: OnnxInferenceBenchmarkSummary,
) -> None:
    policy = summary.policy
    if summary.fresh_processes != policy.fresh_processes:
        raise ValueError("summary fresh-process count differs from policy")
    if len(summary.artifact_content_sha256) != summary.tensor_shape[0]:
        raise ValueError("summary artifact count differs from tensor batch")
    nominal_tensor_bytes = 4
    for dimension in summary.tensor_shape:
        nominal_tensor_bytes *= dimension
    if nominal_tensor_bytes != summary.tensor_bytes:
        raise ValueError("summary tensor byte count differs from shape")
    expected_per_worker = (
        1
        + policy.model_warmup_iterations
        + policy.model_repeat_iterations
        + policy.end_to_end_repeat_iterations
    )
    if summary.output_evaluations != (
        expected_per_worker * policy.fresh_processes
    ):
        raise ValueError("summary output evaluation count differs from policy")
    if policy.unrelated_gpu_work_excluded_by_operator != (
        summary.unrelated_gpu_work_excluded_by_operator
    ):
        raise ValueError("summary GPU-work declaration differs from policy")
    if summary.backend is OnnxBenchmarkBackend.CPU:
        if policy.gpu_device_index is not None:
            raise ValueError("CPU summary policy contains GPU telemetry")
    elif policy.gpu_device_index is None:
        raise ValueError("CUDA summary policy lacks GPU telemetry")

    supervisors: list[SupervisedProcessResult] = []
    measurements: list[dict[str, Any]] = []
    for item in summary.worker_results:
        if not isinstance(item, dict) or set(item) != {
            "supervisor",
            "measurement",
        }:
            raise ValueError("summary worker result keys mismatch")
        supervisor_payload = item["supervisor"]
        measurement = item["measurement"]
        if not isinstance(supervisor_payload, dict) or not isinstance(
            measurement,
            dict,
        ):
            raise TypeError("summary worker result payload must be objects")
        supervisor = SupervisedProcessResult.from_dict(supervisor_payload)
        if supervisor.status is not SupervisedProcessStatus.COMPLETED:
            raise ValueError("summary contains a non-completed worker")
        if supervisor.policy_sha256 != policy.supervisor.policy_sha256:
            raise ValueError("worker supervisor policy binding differs")
        command = supervisor.command
        if (
            len(command) != 11
            or command[0] != (
                summary.worker_environment_identity
                .python_executable_invocation_path
            )
            or command[1:6]
            != (
                "-I", "-B", "-c", ISOLATED_WORKER_BOOTSTRAP,
                "operations.onnx_inference_benchmark",
            )
            or command[6] != command[8]
            or command[7] != "--worker-request"
            or command[9] != "--worker-result"
        ):
            raise ValueError("worker command differs from benchmark module")
        _validate_worker_result(
            measurement,
            request_sha256=summary.request_sha256,
            backend=summary.backend,
            model_sha256=summary.model_sha256,
            backend_config_sha256=summary.backend_config_sha256,
            backend_config_file_sha256=summary.backend_config_file_sha256,
            preprocessing_config_sha256=(
                summary.preprocessing_config_sha256
            ),
            preprocessing_file_sha256=summary.preprocessing_file_sha256,
            dependency_lock_sha256=summary.dependency_lock_sha256,
            runtime_library_policy=summary.runtime_library_policy,
            runtime_library_policy_sha256=(
                summary.runtime_library_policy_sha256
            ),
            runtime_library_policy_file_sha256=(
                summary.runtime_library_policy_file_sha256
            ),
            code_revision=summary.code_revision,
            vector_dimension=summary.vector_dimension,
            artifact_content_sha256=summary.artifact_content_sha256,
            expected_evaluations=expected_per_worker,
            benchmark_policy_sha256=summary.policy_sha256,
            unrelated_system_work_excluded_by_operator=(
                policy.unrelated_system_work_excluded_by_operator
            ),
            worker_environment_identity=(
                summary.worker_environment_identity
            ),
            worker_environment_identity_sha256=(
                summary.worker_environment_identity_sha256
            ),
        )
        if measurement["tensor_sha256"] != summary.tensor_sha256:
            raise ValueError("worker tensor hash differs from summary")
        if measurement["tensor_bytes"] != summary.tensor_bytes:
            raise ValueError("worker tensor bytes differ from summary")
        if measurement["tensor_shape"] != list(summary.tensor_shape):
            raise ValueError("worker tensor shape differs from summary")
        if measurement["output_sha256"] != summary.output_sha256:
            raise ValueError("worker output hash differs from summary")
        if measurement["host_identity"] != summary.host_identity or (
            measurement["host_fingerprint_sha256"]
            != summary.host_fingerprint_sha256
        ):
            raise ValueError("worker host identity differs from summary")
        if measurement["worker_environment_identity"] != (
            summary.worker_environment_identity.to_dict()
        ) or measurement["worker_environment_identity_sha256"] != (
            summary.worker_environment_identity_sha256
        ):
            raise ValueError("worker environment differs from summary")
        if measurement["onnxruntime_distribution_name"] != (
            summary.onnxruntime_distribution_name
        ) or measurement["onnxruntime_distribution_version"] != (
            summary.onnxruntime_distribution_version
        ):
            raise ValueError("worker ONNX Runtime differs from summary")
        if measurement["runtime_library_binary_set_sha256"] != (
            summary.runtime_library_binary_set_sha256
        ) or measurement["runtime_library_decision"] != (
            summary.runtime_library_decision
        ):
            raise ValueError("worker runtime libraries differ from summary")
        if measurement["worker_wall_time_ns"] > supervisor.wall_time_ns:
            raise ValueError("worker wall time exceeds supervisor wall time")
        supervisors.append(supervisor)
        measurements.append(measurement)

    frozen_measurements = tuple(measurements)
    if summary.process_wall_time != TimingSummary.from_samples(
        tuple(item.wall_time_ns for item in supervisors)
    ):
        raise ValueError("summary process timing aggregation differs")
    for field, observed in (
        ("dependency_import_ns", summary.dependency_import_time),
        ("session_construction_ns", summary.session_construction_time),
        ("preprocessing_ns", summary.preprocessing_time),
        (
            "first_preprocessed_inference_ns",
            summary.first_preprocessed_inference_time,
        ),
    ):
        if observed != _timing(frozen_measurements, field):
            raise ValueError(f"summary {field} aggregation differs")
    warm_samples = tuple(
        sample
        for item in frozen_measurements
        for sample in item["warm_preprocessed_inference_ns"]
    )
    end_to_end_samples = tuple(
        sample
        for item in frozen_measurements
        for sample in item["end_to_end_inference_ns"]
    )
    if len(warm_samples) != (
        policy.fresh_processes * policy.model_repeat_iterations
    ) or summary.warm_preprocessed_inference_time != TimingSummary.from_samples(
        warm_samples
    ):
        raise ValueError("summary warm timing aggregation differs")
    if len(end_to_end_samples) != (
        policy.fresh_processes * policy.end_to_end_repeat_iterations
    ) or summary.end_to_end_inference_time != TimingSummary.from_samples(
        end_to_end_samples
    ):
        raise ValueError("summary end-to-end timing aggregation differs")
    maximum_worker_rss = max(
        item["worker_ru_maxrss_bytes"] for item in frozen_measurements
    )
    if maximum_worker_rss != summary.maximum_worker_ru_maxrss_bytes:
        raise ValueError("summary worker RSS aggregation differs")
    if summary.runtime_library_provenance_time != _timing(
        frozen_measurements,
        "runtime_library_provenance_wall_time_ns",
    ):
        raise ValueError("summary runtime provenance timing differs")
    if summary.maximum_runtime_library_bytes_hashed != max(
        item["runtime_library_bytes_hashed"]
        for item in frozen_measurements
    ):
        raise ValueError("summary runtime library bytes differ")
    sampled_rss = tuple(
        item.sampled_peak_rss_bytes
        for item in supervisors
        if item.sampled_peak_rss_bytes is not None
    )
    expected_sampled_rss = max(sampled_rss) if sampled_rss else None
    if expected_sampled_rss != summary.maximum_supervisor_sampled_rss_bytes:
        raise ValueError("summary sampled RSS aggregation differs")


def _require_identical_field(
    measurements: tuple[dict[str, Any], ...],
    field: str,
) -> None:
    reference = measurements[0][field]
    if any(item[field] != reference for item in measurements[1:]):
        raise RuntimeError(f"worker {field} changed across fresh processes")


def _host_identity() -> dict[str, str | int]:
    uname = platform.uname()
    cpu_model = "UNKNOWN"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    boot_id = "UNAVAILABLE"
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    if boot_id_path.is_file():
        boot_id = boot_id_path.read_text(encoding="utf-8").strip()
    return {
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count() or 0,
        "python_version": platform.python_version(),
        "boot_id": boot_id,
    }


def _regular_file(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _verify_file_identity(
    path: Path,
    expected_sha256: str,
    *,
    expected_size: int | None = None,
) -> None:
    _validate_sha256(expected_sha256)
    before = path.stat()
    if expected_size is not None and before.st_size != expected_size:
        raise ValueError("file size differs from benchmark request")
    actual = sha256_file(path)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise RuntimeError("benchmark input changed while hashing")
    if actual != expected_sha256:
        raise ValueError("benchmark input SHA-256 mismatch")


def _canonical_pretty_json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_sha256(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("expected a lowercase SHA-256 digest")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-request", type=Path, required=True)
    parser.add_argument("--worker-result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_worker(args.worker_request, args.worker_result)


if __name__ == "__main__":
    main()
