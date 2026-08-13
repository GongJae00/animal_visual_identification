"""Internal worker for one sanitized batch-invariance execution."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from contracts.runtime_library_provenance import (
    RuntimeLibraryPhase,
    RuntimeLibraryPolicy,
    RuntimeLibraryTracker,
)
from evaluation.batch_invariance import (
    BatchInvariancePolicy,
    BatchInvariancePrecommitment,
    BatchRuntimeDiscoveryComplete,
    batch_artifact_paths_from_dict,
    evaluate_batch_composition_invariance,
)
from evaluation.control_scoring import ControlScoringInventory
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from foundation.provenance import content_sha256
from operations.batch_invariance_runner import _verify_file_binding
from operations.embedding_producer import EmbeddingProducerConfig
from operations.worker_environment import (
    WorkerEnvironmentIdentity,
    validate_current_worker_environment,
)

_FILE_NAMES = {
    "inventory", "artifact_paths", "producer_config", "onnx_config",
    "preprocessing", "model", "model_lineage", "dependency_lock",
    "batch_policy", "precommitment", "runtime_library_policy",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    request = read_strict_json_object(args.request)
    _validate_request(request)
    request_sha256 = content_sha256(request)
    expected_environment = WorkerEnvironmentIdentity.from_dict(
        request["worker_environment_identity"]
    )
    observed_environment = validate_current_worker_environment(
        expected_environment
    )
    scratch_path = Path(request["scratch_path"])
    if scratch_path.is_symlink():
        raise ValueError("batch worker scratch must not be a symlink")
    scratch_path = scratch_path.resolve(strict=True)
    if not scratch_path.is_dir():
        raise ValueError("batch worker scratch must be a directory")
    files = request["files"]
    for name, binding in files.items():
        _verify_file_binding(Path(binding["path"]), binding, name)

    precommitment_payload = read_strict_json_object(
        Path(files["precommitment"]["path"])
    )
    if set(precommitment_payload) != {
        "schema_version", "precommitment_sha256", "precommitment"
    } or precommitment_payload["schema_version"] != (
        "cvi.batch_invariance_precommitment_bundle.v1"
    ):
        raise ValueError("batch worker precommitment bundle differs")
    precommitment = BatchInvariancePrecommitment.from_dict(
        precommitment_payload["precommitment"]
    )
    if precommitment.precommitment_sha256 != precommitment_payload[
        "precommitment_sha256"
    ] or precommitment.precommitment_sha256 != request[
        "expected_precommitment_sha256"
    ]:
        raise ValueError("batch worker precommitment hash differs")
    if precommitment.worker_execution_policy_sha256 != request[
        "execution_policy_sha256"
    ]:
        raise ValueError("batch worker execution policy differs")
    if precommitment.worker_environment_identity_sha256 != (
        observed_environment.identity_sha256
    ):
        raise ValueError("batch worker environment differs from precommitment")

    runtime_policy = RuntimeLibraryPolicy.from_dict(
        read_strict_json_object(Path(files["runtime_library_policy"]["path"]))
    )
    if runtime_policy.policy_sha256 != (
        precommitment.runtime_library_policy_sha256
    ):
        raise ValueError("batch worker runtime policy differs")
    if runtime_policy.allow_discovery_only != request["discovery"]:
        raise ValueError("batch worker discovery mode differs from policy")
    runtime_tracker = RuntimeLibraryTracker(runtime_policy)

    from operations.onnx_backend import (
        ImagePreprocessingConfig,
        OnnxRuntimeBackendConfig,
        OnnxRuntimeCpuBackend,
        OnnxRuntimeCudaBackend,
        onnxruntime_distribution_identity,
    )

    onnxruntime_distribution = onnxruntime_distribution_identity(
        require_gpu=(request["backend"] == "cuda")
    )
    runtime_tracker.capture(RuntimeLibraryPhase.DEPENDENCIES_IMPORTED)
    producer_config = EmbeddingProducerConfig.from_dict(
        read_strict_json_object(Path(files["producer_config"]["path"]))
    )
    backend_config = OnnxRuntimeBackendConfig.from_dict(
        read_strict_json_object(Path(files["onnx_config"]["path"]))
    )
    preprocessing = ImagePreprocessingConfig.from_dict(
        read_strict_json_object(Path(files["preprocessing"]["path"]))
    )
    backend_class = (
        OnnxRuntimeCpuBackend
        if request["backend"] == "cpu"
        else OnnxRuntimeCudaBackend
    )
    backend = backend_class(
        model_path=Path(files["model"]["path"]),
        config=backend_config,
        preprocessing=preprocessing,
    )
    common = {
        "schema_version": "cvi.batch_fresh_worker_result.v1",
        "request_sha256": request_sha256,
        "backend": request["backend"],
        "worker_environment_identity": observed_environment.to_dict(),
        "worker_environment_identity_sha256": (
            observed_environment.identity_sha256
        ),
        "onnxruntime_distribution_name": onnxruntime_distribution[0],
        "onnxruntime_distribution_version": onnxruntime_distribution[1],
    }
    try:
        receipt = evaluate_batch_composition_invariance(
            backend=backend,
            inventory=ControlScoringInventory.from_dict(
                read_strict_json_object(Path(files["inventory"]["path"]))
            ),
            artifact_paths=batch_artifact_paths_from_dict(
                read_strict_json_object(Path(files["artifact_paths"]["path"]))
            ),
            producer_config=producer_config,
            provenance_paths={
                "model": Path(files["model"]["path"]),
                "model_lineage": Path(files["model_lineage"]["path"]),
                "preprocessing": Path(files["preprocessing"]["path"]),
                "dependency_lock": Path(files["dependency_lock"]["path"]),
            },
            policy=BatchInvariancePolicy.from_dict(
                read_strict_json_object(Path(files["batch_policy"]["path"]))
            ),
            precommitment=precommitment,
            expected_precommitment_sha256=request[
                "expected_precommitment_sha256"
            ],
            runtime_library_tracker=runtime_tracker,
            temporary_directory_parent=scratch_path,
        )
    except BatchRuntimeDiscoveryComplete as completed:
        if not request["discovery"]:
            raise
        manifest = completed.manifest
        result = {
            **common,
            "kind": "DISCOVERY",
            "runtime_library_manifest_sha256": manifest.manifest_sha256,
            "runtime_library_manifest": manifest.to_dict(),
        }
    else:
        if request["discovery"]:
            raise RuntimeError("batch discovery unexpectedly admitted")
        result = {
            **common,
            "kind": "RECEIPT",
            "batch_receipt_sha256": receipt.receipt_sha256,
            "batch_receipt": receipt.to_dict(),
        }
    write_private_json_bundle(((args.result, result),))


def _validate_request(payload: dict[str, Any]) -> None:
    expected = {
        "schema_version", "backend", "files",
        "expected_precommitment_sha256", "worker_environment_identity",
        "worker_environment_identity_sha256", "execution_policy_sha256",
        "discovery", "scratch_path",
    }
    if set(payload) != expected or payload["schema_version"] != (
        "cvi.batch_fresh_worker_request.v1"
    ):
        raise ValueError("batch fresh-worker request schema differs")
    if payload["backend"] not in {"cpu", "cuda"}:
        raise ValueError("batch worker backend differs")
    if not isinstance(payload["discovery"], bool):
        raise TypeError("batch worker discovery flag must be boolean")
    if not isinstance(payload["scratch_path"], str) or not (
        Path(payload["scratch_path"]).is_absolute()
    ):
        raise ValueError("batch worker scratch path differs")
    files = payload["files"]
    if not isinstance(files, dict) or set(files) != _FILE_NAMES:
        raise ValueError("batch worker request files differ")
    for name, binding in files.items():
        if not isinstance(binding, dict) or set(binding) != {
            "path", "byte_size", "content_sha256"
        }:
            raise ValueError(f"batch worker {name} binding differs")
    environment_payload = payload["worker_environment_identity"]
    if not isinstance(environment_payload, dict):
        raise TypeError("batch worker environment must be an object")
    environment = WorkerEnvironmentIdentity.from_dict(environment_payload)
    if environment.identity_sha256 != payload[
        "worker_environment_identity_sha256"
    ]:
        raise ValueError("batch worker environment hash differs")
    for name in (
        "expected_precommitment_sha256",
        "worker_environment_identity_sha256",
        "execution_policy_sha256",
    ):
        value = payload[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"batch worker {name} differs")


if __name__ == "__main__":
    main()
