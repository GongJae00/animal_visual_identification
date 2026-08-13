"""Sanitized supervised fresh-worker boundary for batch invariance."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from contracts.runtime_library_provenance import RuntimeLibraryManifest
from data.acquisition import sha256_file
from evaluation.integrity.batch_invariance import (
    BatchInvariancePrecommitment,
    BatchInvarianceReceipt,
)
from foundation.provenance import content_sha256
from systems.workers.process_supervisor import (
    ProcessSupervisorPolicy,
    SupervisedProcessResult,
    SupervisedProcessStatus,
    run_supervised_process,
)
from systems.workers.worker_environment import (
    ISOLATED_WORKER_BOOTSTRAP,
    WorkerEnvironmentIdentity,
    build_sanitized_worker_environment,
)

_BATCH_WORKER_MODULES = {
    "operations.batch_invariance_worker",
    "systems.workers.batch_invariance_worker",
}
_CURRENT_BATCH_WORKER_MODULE = "systems.workers.batch_invariance_worker"


@dataclass(frozen=True, slots=True)
class BatchWorkerExecutionPolicy:
    supervisor: ProcessSupervisorPolicy
    maximum_worker_result_bytes: int = 67_108_864
    schema_version: str = "cvi.batch_worker_execution_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.batch_worker_execution_policy.v1":
            raise ValueError("unsupported batch worker execution policy schema")
        if (
            isinstance(self.maximum_worker_result_bytes, bool)
            or not isinstance(self.maximum_worker_result_bytes, int)
            or self.maximum_worker_result_bytes <= 0
        ):
            raise ValueError("batch worker result cap must be positive")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "supervisor": self.supervisor.to_dict(),
            "maximum_worker_result_bytes": self.maximum_worker_result_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BatchWorkerExecutionPolicy:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("batch worker execution policy keys mismatch")
        if not isinstance(payload["supervisor"], dict):
            raise TypeError("batch worker supervisor policy must be an object")
        values = dict(payload)
        values["supervisor"] = ProcessSupervisorPolicy.from_dict(
            values["supervisor"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class BatchFreshWorkerReceipt:
    worker_request_sha256: str
    batch_receipt_sha256: str
    batch_receipt: BatchInvarianceReceipt
    worker_environment_identity_sha256: str
    worker_environment_identity: WorkerEnvironmentIdentity
    onnxruntime_distribution_name: str
    onnxruntime_distribution_version: str
    execution_policy_sha256: str
    execution_policy: BatchWorkerExecutionPolicy
    supervised_process_result_sha256: str
    supervised_process_result: SupervisedProcessResult
    interpretation: str = (
        "FRESH_WORKER_BATCH_INVARIANCE_ONLY_NOT_OPTIMIZATION_PROMOTION"
    )
    schema_version: str = "cvi.batch_fresh_worker_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.batch_fresh_worker_receipt.v1":
            raise ValueError("unsupported batch fresh-worker receipt schema")
        for name in (
            "worker_request_sha256",
            "batch_receipt_sha256",
            "worker_environment_identity_sha256",
            "execution_policy_sha256",
            "supervised_process_result_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.batch_receipt.receipt_sha256 != self.batch_receipt_sha256:
            raise ValueError("embedded batch receipt hash differs")
        if self.worker_environment_identity.identity_sha256 != (
            self.worker_environment_identity_sha256
        ):
            raise ValueError("embedded worker environment hash differs")
        if self.execution_policy.policy_sha256 != self.execution_policy_sha256:
            raise ValueError("embedded batch worker policy hash differs")
        if self.supervised_process_result.result_sha256 != (
            self.supervised_process_result_sha256
        ):
            raise ValueError("embedded supervised process hash differs")
        if self.supervised_process_result.policy_sha256 != (
            self.execution_policy.supervisor.policy_sha256
        ):
            raise ValueError("supervised process policy differs")
        if self.supervised_process_result.status is not (
            SupervisedProcessStatus.COMPLETED
        ):
            raise ValueError("batch worker process did not complete")
        command = self.supervised_process_result.command
        if len(command) != 11 or command[:5] != (
            self.worker_environment_identity.python_executable_invocation_path,
            "-I", "-B", "-c", ISOLATED_WORKER_BOOTSTRAP,
        ) or command[5] not in _BATCH_WORKER_MODULES or command[6] != command[8] or command[7] != (
            "--request"
        ) or command[9] != "--result":
            raise ValueError("batch worker command differs")
        if (
            self.supervised_process_result.stdout_bytes != 0
            or self.supervised_process_result.stderr_bytes != 0
        ):
            raise ValueError("batch worker emitted unexpected output")
        precommitment = self.batch_receipt.precommitment
        if precommitment.worker_execution_policy_sha256 != (
            self.execution_policy_sha256
        ):
            raise ValueError("batch worker policy differs from precommitment")
        if precommitment.worker_environment_identity_sha256 != (
            self.worker_environment_identity_sha256
        ):
            raise ValueError("worker environment differs from precommitment")
        if not self.onnxruntime_distribution_name or not (
            self.onnxruntime_distribution_version
        ):
            raise ValueError("ONNX Runtime distribution identity is empty")
        expected_distribution = (
            "onnxruntime-gpu"
            if self.batch_receipt.actual_providers[0] == "CUDAExecutionProvider"
            else "onnxruntime"
        )
        if self.onnxruntime_distribution_name != expected_distribution:
            raise ValueError("ONNX Runtime distribution lane differs")
        if self.interpretation != (
            "FRESH_WORKER_BATCH_INVARIANCE_ONLY_NOT_OPTIMIZATION_PROMOTION"
        ):
            raise ValueError("batch fresh-worker interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "worker_request_sha256": self.worker_request_sha256,
            "batch_receipt_sha256": self.batch_receipt_sha256,
            "batch_receipt": self.batch_receipt.to_dict(),
            "worker_environment_identity_sha256": (
                self.worker_environment_identity_sha256
            ),
            "worker_environment_identity": (
                self.worker_environment_identity.to_dict()
            ),
            "onnxruntime_distribution_name": self.onnxruntime_distribution_name,
            "onnxruntime_distribution_version": (
                self.onnxruntime_distribution_version
            ),
            "execution_policy_sha256": self.execution_policy_sha256,
            "execution_policy": self.execution_policy.to_dict(),
            "supervised_process_result_sha256": (
                self.supervised_process_result_sha256
            ),
            "supervised_process_result": self.supervised_process_result.to_dict(),
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BatchFreshWorkerReceipt:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("batch fresh-worker receipt keys mismatch")
        values = dict(payload)
        values["batch_receipt"] = BatchInvarianceReceipt.from_dict(
            values["batch_receipt"]
        )
        values["worker_environment_identity"] = (
            WorkerEnvironmentIdentity.from_dict(
                values["worker_environment_identity"]
            )
        )
        values["execution_policy"] = BatchWorkerExecutionPolicy.from_dict(
            values["execution_policy"]
        )
        values["supervised_process_result"] = SupervisedProcessResult.from_dict(
            values["supervised_process_result"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class BatchFreshWorkerDiscovery:
    worker_request_sha256: str
    precommitment_sha256: str
    runtime_library_manifest_sha256: str
    runtime_library_manifest: RuntimeLibraryManifest
    worker_environment_identity_sha256: str
    worker_environment_identity: WorkerEnvironmentIdentity
    onnxruntime_distribution_name: str
    onnxruntime_distribution_version: str
    execution_policy_sha256: str
    execution_policy: BatchWorkerExecutionPolicy
    supervised_process_result_sha256: str
    supervised_process_result: SupervisedProcessResult
    interpretation: str = (
        "DISCOVERY_ONLY_REQUIRES_REVIEW_FREEZE_AND_STRICT_RERUN"
    )
    schema_version: str = "cvi.batch_fresh_worker_discovery.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.batch_fresh_worker_discovery.v1":
            raise ValueError("unsupported batch worker discovery schema")
        for name in (
            "worker_request_sha256", "precommitment_sha256",
            "runtime_library_manifest_sha256",
            "worker_environment_identity_sha256", "execution_policy_sha256",
            "supervised_process_result_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.runtime_library_manifest.manifest_sha256 != (
            self.runtime_library_manifest_sha256
        ) or self.runtime_library_manifest.decision != "DISCOVERY_ONLY":
            raise ValueError("batch discovery manifest differs")
        if self.worker_environment_identity.identity_sha256 != (
            self.worker_environment_identity_sha256
        ):
            raise ValueError("batch discovery worker environment differs")
        if self.execution_policy.policy_sha256 != self.execution_policy_sha256:
            raise ValueError("batch discovery execution policy differs")
        if self.supervised_process_result.result_sha256 != (
            self.supervised_process_result_sha256
        ) or self.supervised_process_result.status is not (
            SupervisedProcessStatus.COMPLETED
        ):
            raise ValueError("batch discovery supervised process differs")
        if self.supervised_process_result.policy_sha256 != (
            self.execution_policy.supervisor.policy_sha256
        ):
            raise ValueError("batch discovery supervisor policy differs")
        command = self.supervised_process_result.command
        if len(command) != 11 or command[:5] != (
            self.worker_environment_identity.python_executable_invocation_path,
            "-I", "-B", "-c", ISOLATED_WORKER_BOOTSTRAP,
        ) or command[5] not in _BATCH_WORKER_MODULES or command[6] != command[8] or command[7] != (
            "--request"
        ) or command[9] != "--result":
            raise ValueError("batch discovery worker command differs")
        if (
            self.supervised_process_result.stdout_bytes != 0
            or self.supervised_process_result.stderr_bytes != 0
        ):
            raise ValueError("batch discovery worker emitted output")
        if not self.onnxruntime_distribution_name or not (
            self.onnxruntime_distribution_version
        ):
            raise ValueError("batch discovery ORT identity is empty")

    @property
    def discovery_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "worker_request_sha256": self.worker_request_sha256,
            "precommitment_sha256": self.precommitment_sha256,
            "runtime_library_manifest_sha256": (
                self.runtime_library_manifest_sha256
            ),
            "runtime_library_manifest": self.runtime_library_manifest.to_dict(),
            "worker_environment_identity_sha256": (
                self.worker_environment_identity_sha256
            ),
            "worker_environment_identity": (
                self.worker_environment_identity.to_dict()
            ),
            "onnxruntime_distribution_name": self.onnxruntime_distribution_name,
            "onnxruntime_distribution_version": (
                self.onnxruntime_distribution_version
            ),
            "execution_policy_sha256": self.execution_policy_sha256,
            "execution_policy": self.execution_policy.to_dict(),
            "supervised_process_result_sha256": (
                self.supervised_process_result_sha256
            ),
            "supervised_process_result": self.supervised_process_result.to_dict(),
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> BatchFreshWorkerDiscovery:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("batch fresh-worker discovery keys mismatch")
        values = dict(payload)
        values["runtime_library_manifest"] = RuntimeLibraryManifest.from_dict(
            values["runtime_library_manifest"]
        )
        values["worker_environment_identity"] = (
            WorkerEnvironmentIdentity.from_dict(
                values["worker_environment_identity"]
            )
        )
        values["execution_policy"] = BatchWorkerExecutionPolicy.from_dict(
            values["execution_policy"]
        )
        values["supervised_process_result"] = SupervisedProcessResult.from_dict(
            values["supervised_process_result"]
        )
        return cls(**values)


def run_batch_invariance_fresh_worker(
    *,
    backend: str,
    files: Mapping[str, Path],
    precommitment: BatchInvariancePrecommitment,
    expected_precommitment_sha256: str,
    python_executable: Path,
    execution_policy: BatchWorkerExecutionPolicy,
    discovery: bool,
) -> BatchFreshWorkerReceipt | BatchFreshWorkerDiscovery:
    if backend not in {"cpu", "cuda"}:
        raise ValueError("batch worker backend differs")
    _sha256(expected_precommitment_sha256, "expected precommitment")
    if precommitment.precommitment_sha256 != expected_precommitment_sha256:
        raise ValueError("batch precommitment differs from external anchor")
    required_files = {
        "inventory", "artifact_paths", "producer_config", "onnx_config",
        "preprocessing", "model", "model_lineage", "dependency_lock",
        "batch_policy", "precommitment", "runtime_library_policy",
    }
    if set(files) != required_files:
        raise ValueError("batch worker input file names differ")
    bindings = {
        name: _file_binding(path, name) for name, path in sorted(files.items())
    }
    child_environment, environment_identity = build_sanitized_worker_environment(
        os.environ,
        python_executable=python_executable,
    )
    if precommitment.worker_execution_policy_sha256 != (
        execution_policy.policy_sha256
    ):
        raise ValueError("batch execution policy differs from precommitment")
    if precommitment.worker_environment_identity_sha256 != (
        environment_identity.identity_sha256
    ):
        raise ValueError("batch worker environment differs from precommitment")
    request = {
        "schema_version": "cvi.batch_fresh_worker_request.v1",
        "backend": backend,
        "files": bindings,
        "expected_precommitment_sha256": expected_precommitment_sha256,
        "worker_environment_identity": environment_identity.to_dict(),
        "worker_environment_identity_sha256": environment_identity.identity_sha256,
        "execution_policy_sha256": execution_policy.policy_sha256,
        "discovery": discovery,
        "scratch_path": "PENDING_PRIVATE_SCRATCH",
    }
    with TemporaryDirectory(prefix="cvi-batch-worker-") as temporary:
        root = Path(temporary)
        scratch = root / "scratch"
        scratch.mkdir(mode=0o700)
        request["scratch_path"] = str(scratch)
        request_sha256 = content_sha256(request)
        request_path = root / "request.json"
        result_path = root / "result.json"
        request_path.write_text(
            json.dumps(request, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.chmod(request_path, 0o600)
        command = (
            environment_identity.python_executable_invocation_path,
            "-I",
            "-B",
            "-c",
            ISOLATED_WORKER_BOOTSTRAP,
            _CURRENT_BATCH_WORKER_MODULE,
            str(request_path),
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        )
        supervised = run_supervised_process(
            command,
            policy=execution_policy.supervisor,
            environment=child_environment,
        )
        if supervised.status is not SupervisedProcessStatus.COMPLETED:
            raise RuntimeError(
                "batch fresh worker failed: "
                f"{supervised.status.value} rc={supervised.return_code}"
            )
        result = _read_worker_result(
            result_path,
            maximum_bytes=execution_policy.maximum_worker_result_bytes,
        )
    _validate_common_worker_result(
        result,
        request_sha256=request_sha256,
        expected_environment=environment_identity,
        expected_backend=backend,
        discovery=discovery,
    )
    for name, binding in bindings.items():
        _verify_file_binding(Path(binding["path"]), binding, name)
    common = {
        "worker_request_sha256": request_sha256,
        "worker_environment_identity_sha256": environment_identity.identity_sha256,
        "worker_environment_identity": environment_identity,
        "onnxruntime_distribution_name": result["onnxruntime_distribution_name"],
        "onnxruntime_distribution_version": result[
            "onnxruntime_distribution_version"
        ],
        "execution_policy_sha256": execution_policy.policy_sha256,
        "execution_policy": execution_policy,
        "supervised_process_result_sha256": supervised.result_sha256,
        "supervised_process_result": supervised,
    }
    if discovery:
        manifest = RuntimeLibraryManifest.from_dict(
            result["runtime_library_manifest"]
        )
        return BatchFreshWorkerDiscovery(
            precommitment_sha256=precommitment.precommitment_sha256,
            runtime_library_manifest_sha256=manifest.manifest_sha256,
            runtime_library_manifest=manifest,
            **common,
        )
    batch_receipt = BatchInvarianceReceipt.from_dict(result["batch_receipt"])
    return BatchFreshWorkerReceipt(
        batch_receipt_sha256=batch_receipt.receipt_sha256,
        batch_receipt=batch_receipt,
        **common,
    )


def _file_binding(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"batch worker {name} must not be a symlink")
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not resolved.is_file() or before.st_size <= 0:
        raise ValueError(f"batch worker {name} must be a nonempty file")
    digest = sha256_file(resolved)
    after = resolved.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"batch worker {name} changed while hashing")
    return {
        "path": str(resolved),
        "byte_size": before.st_size,
        "content_sha256": digest,
    }


def _verify_file_binding(
    path: Path,
    binding: Mapping[str, Any],
    name: str,
) -> None:
    if set(binding) != {"path", "byte_size", "content_sha256"}:
        raise ValueError(f"batch worker {name} binding keys differ")
    observed = _file_binding(path, name)
    if observed != dict(binding):
        raise RuntimeError(f"batch worker {name} changed across execution")


def _read_worker_result(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("batch worker result must not be a symlink")
    resolved = path.resolve(strict=True)
    payload = resolved.read_bytes()
    if not payload or len(payload) > maximum_bytes:
        raise ValueError("batch worker result byte size differs")
    return json.loads(payload, object_pairs_hook=_reject_duplicate_keys)


def _validate_common_worker_result(
    payload: dict[str, Any],
    *,
    request_sha256: str,
    expected_environment: WorkerEnvironmentIdentity,
    expected_backend: str,
    discovery: bool,
) -> None:
    common = {
        "schema_version", "request_sha256", "backend",
        "worker_environment_identity", "worker_environment_identity_sha256",
        "onnxruntime_distribution_name", "onnxruntime_distribution_version",
        "kind",
    }
    expected = common | (
        {"runtime_library_manifest", "runtime_library_manifest_sha256"}
        if discovery else {"batch_receipt", "batch_receipt_sha256"}
    )
    if set(payload) != expected or payload["schema_version"] != (
        "cvi.batch_fresh_worker_result.v1"
    ):
        raise ValueError("batch worker result schema differs")
    if payload["request_sha256"] != request_sha256:
        raise ValueError("batch worker result request differs")
    if payload["backend"] != expected_backend:
        raise ValueError("batch worker result backend differs")
    environment = WorkerEnvironmentIdentity.from_dict(
        payload["worker_environment_identity"]
    )
    if environment != expected_environment or environment.identity_sha256 != (
        payload["worker_environment_identity_sha256"]
    ):
        raise ValueError("batch worker result environment differs")
    expected_kind = "DISCOVERY" if discovery else "RECEIPT"
    if payload["kind"] != expected_kind:
        raise ValueError("batch worker result kind differs")
    if discovery:
        manifest = RuntimeLibraryManifest.from_dict(
            payload["runtime_library_manifest"]
        )
        if manifest.manifest_sha256 != payload[
            "runtime_library_manifest_sha256"
        ]:
            raise ValueError("batch worker discovery manifest hash differs")
    else:
        receipt = BatchInvarianceReceipt.from_dict(payload["batch_receipt"])
        if receipt.receipt_sha256 != payload["batch_receipt_sha256"]:
            raise ValueError("batch worker receipt hash differs")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in batch worker result")
        result[key] = value
    return result


def _sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
