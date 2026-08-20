"""Internal sanitized worker for protected embedding production."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

from shared.contracts.runtime_library_provenance import (
    RuntimeLibraryPhase,
    RuntimeLibraryPolicy,
    RuntimeLibraryTracker,
)
from evaluation.controls.control_scoring import ControlScoringInventory, EmbeddingCachePolicy
from shared.foundation.protected_io import read_strict_json_object, write_private_json_bundle
from shared.foundation.provenance import content_sha256
from shared.foundation.retained_file import verify_retained_regular_file_binding
from prototype.export.embedding_producer import (
    EmbeddingProducerConfig,
    EmbeddingProductionPolicy,
    produce_embedding_cache,
)
from operations.workers.embedding_production_runner import (
    EmbeddingProductionPrecommitment,
    EmbeddingWorkerExecutionPolicy,
    _verify_code_source_bindings,
    build_embedding_production_precommitment,
    embedding_artifact_paths_from_dict,
)
from operations.workers.worker_environment import (
    WorkerEnvironmentIdentity,
    validate_current_worker_environment,
)

_FILE_NAMES = {
    "inventory", "artifact_paths", "producer_config", "onnx_config",
    "preprocessing", "model", "model_lineage", "dependency_lock",
    "production_policy", "cache_policy", "precommitment",
    "runtime_library_policy",
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
    scratch_path = _private_directory(Path(request["scratch_path"]), "scratch")
    cache_path = _private_directory(Path(request["cache_path"]), "cache")
    code_root = _private_directory(
        Path(request["code_snapshot_root"]),
        "code snapshot",
    )
    if cache_path.parent != scratch_path or any(cache_path.iterdir()):
        raise ValueError("embedding worker cache staging differs")
    if Path(__file__).resolve().parent != code_root / "operations" / "workers":
        raise ValueError("embedding worker did not execute from code snapshot")

    files = request["files"]
    for name, binding in files.items():
        verify_retained_regular_file_binding(
            Path(binding["path"]),
            binding,
            subject=f"embedding worker {name}",
        )
    precommitment_payload = read_strict_json_object(
        Path(files["precommitment"]["path"])
    )
    if set(precommitment_payload) != {
        "schema_version", "precommitment_sha256", "precommitment"
    } or precommitment_payload["schema_version"] != (
        "cvi.embedding_production_precommitment_bundle.v1"
    ):
        raise ValueError("embedding worker precommitment bundle differs")
    precommitment = EmbeddingProductionPrecommitment.from_dict(
        precommitment_payload["precommitment"]
    )
    if precommitment.precommitment_sha256 != precommitment_payload[
        "precommitment_sha256"
    ] or precommitment.precommitment_sha256 != request[
        "expected_precommitment_sha256"
    ]:
        raise ValueError("embedding worker precommitment hash differs")
    if precommitment.code_source_manifest_sha256 != request[
        "code_snapshot_manifest_sha256"
    ]:
        raise ValueError("embedding worker code snapshot authority differs")
    _verify_code_source_bindings(precommitment.code_source_sha256)
    if precommitment.worker_execution_policy_sha256 != request[
        "execution_policy_sha256"
    ] or precommitment.worker_environment_identity_sha256 != (
        observed_environment.identity_sha256
    ):
        raise ValueError("embedding worker execution authority differs")
    execution_policy = EmbeddingWorkerExecutionPolicy.from_dict(
        request["execution_policy"]
    )
    if execution_policy.policy_sha256 != request["execution_policy_sha256"]:
        raise ValueError("embedding worker execution policy hash differs")

    inventory = ControlScoringInventory.from_dict(
        read_strict_json_object(Path(files["inventory"]["path"]))
    )
    artifact_paths = embedding_artifact_paths_from_dict(
        read_strict_json_object(Path(files["artifact_paths"]["path"]))
    )
    producer_config = EmbeddingProducerConfig.from_dict(
        read_strict_json_object(Path(files["producer_config"]["path"]))
    )
    production_policy = EmbeddingProductionPolicy.from_dict(
        read_strict_json_object(Path(files["production_policy"]["path"]))
    )
    cache_policy = EmbeddingCachePolicy.from_dict(
        read_strict_json_object(Path(files["cache_policy"]["path"]))
    )
    runtime_policy = RuntimeLibraryPolicy.from_dict(
        read_strict_json_object(Path(files["runtime_library_policy"]["path"]))
    )
    if runtime_policy.policy_sha256 != (
        precommitment.runtime_library_policy_sha256
    ) or runtime_policy.allow_discovery_only != request["discovery"]:
        raise ValueError("embedding worker runtime policy differs")
    derived = build_embedding_production_precommitment(
        inventory=inventory,
        artifact_paths=artifact_paths,
        producer_config=producer_config,
        provenance_paths={
            "model": Path(files["model"]["path"]),
            "model_lineage": Path(files["model_lineage"]["path"]),
            "onnx_config": Path(files["onnx_config"]["path"]),
            "preprocessing": Path(files["preprocessing"]["path"]),
            "dependency_lock": Path(files["dependency_lock"]["path"]),
        },
        production_policy=production_policy,
        cache_policy=cache_policy,
        runtime_library_policy_sha256=runtime_policy.policy_sha256,
        worker_execution_policy_sha256=request["execution_policy_sha256"],
        worker_environment_identity_sha256=observed_environment.identity_sha256,
        prior_attempt_ledger_sha256=(
            precommitment.prior_attempt_ledger_sha256
        ),
        candidate_attempt_token=precommitment.candidate_attempt_token,
        precommitment_sequence=precommitment.precommitment_sequence,
    )
    if derived != precommitment:
        raise ValueError("embedding execution inputs differ from precommitment")
    snapshot_paths, snapshot_files, snapshot_bytes = _snapshot_artifacts(
        inventory=inventory,
        artifact_paths=artifact_paths,
        scratch_path=scratch_path,
        maximum_bytes=execution_policy.maximum_snapshot_bytes,
    )

    runtime_tracker = RuntimeLibraryTracker(runtime_policy)
    from prototype.export.onnx_backend import (
        ImagePreprocessingConfig,
        OnnxRuntimeBackendConfig,
        OnnxRuntimeCpuBackend,
        OnnxRuntimeCudaBackend,
        onnxruntime_distribution_identity,
    )

    distribution = onnxruntime_distribution_identity(
        require_gpu=(request["backend"] == "cuda")
    )
    runtime_tracker.capture(RuntimeLibraryPhase.DEPENDENCIES_IMPORTED)
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
    runtime_tracker.capture(RuntimeLibraryPhase.SESSION_READY)

    def observe(phase: str) -> None:
        if phase != "FIRST_OUTPUT_READY":
            raise ValueError("embedding runtime phase differs")
        runtime_tracker.capture(RuntimeLibraryPhase.FIRST_OUTPUT_READY)

    receipt = produce_embedding_cache(
        inventory=inventory,
        artifact_paths=snapshot_paths,
        model_path=Path(files["model"]["path"]),
        model_lineage_path=Path(files["model_lineage"]["path"]),
        preprocessing_path=Path(files["preprocessing"]["path"]),
        dependency_lock_path=Path(files["dependency_lock"]["path"]),
        config=producer_config,
        production_policy=production_policy,
        cache_policy=cache_policy,
        backend=backend,
        output_directory=cache_path,
        runtime_phase_callback=observe,
    )
    runtime_tracker.capture(RuntimeLibraryPhase.FINAL_OUTPUT_READY)
    manifest = runtime_tracker.finalize()
    expected_decision = "DISCOVERY_ONLY" if request["discovery"] else "PASS"
    if manifest.decision != expected_decision:
        raise ValueError("embedding runtime library decision differs")
    actual_providers = tuple(backend.actual_providers)
    result = {
        "schema_version": "cvi.embedding_fresh_worker_result.v2",
        "request_sha256": request_sha256,
        "backend": request["backend"],
        "worker_environment_identity": observed_environment.to_dict(),
        "worker_environment_identity_sha256": (
            observed_environment.identity_sha256
        ),
        "onnxruntime_distribution_name": distribution[0],
        "onnxruntime_distribution_version": distribution[1],
        "actual_providers": list(actual_providers),
        "actual_provider_options_sha256": content_sha256(
            backend.actual_provider_options
        ),
        "snapshot_unique_files": snapshot_files,
        "snapshot_input_bytes": snapshot_bytes,
        "production_receipt_sha256": receipt.receipt_sha256,
        "production_receipt": receipt.to_dict(),
        "runtime_library_manifest_sha256": manifest.manifest_sha256,
        "runtime_library_manifest": manifest.to_dict(),
    }
    write_private_json_bundle(((args.result, result),))


def _validate_request(payload: dict[str, Any]) -> None:
    expected = {
        "schema_version", "backend", "files",
        "expected_precommitment_sha256", "worker_environment_identity",
        "worker_environment_identity_sha256", "execution_policy_sha256",
        "execution_policy", "discovery", "scratch_path", "cache_path",
        "code_snapshot_root", "code_snapshot_manifest_sha256",
    }
    if set(payload) != expected or payload["schema_version"] != (
        "cvi.embedding_fresh_worker_request.v2"
    ):
        raise ValueError("embedding fresh-worker request schema differs")
    if payload["backend"] not in {"cpu", "cuda"} or not isinstance(
        payload["discovery"], bool
    ):
        raise ValueError("embedding fresh-worker mode differs")
    for name in ("scratch_path", "cache_path", "code_snapshot_root"):
        if not isinstance(payload[name], str) or not Path(payload[name]).is_absolute():
            raise ValueError(f"embedding worker {name} differs")
    if not isinstance(payload["files"], dict) or set(payload["files"]) != (
        _FILE_NAMES
    ):
        raise ValueError("embedding worker request files differ")
    for name, binding in payload["files"].items():
        if not isinstance(binding, dict) or set(binding) != {
            "path", "byte_size", "content_sha256"
        }:
            raise ValueError(f"embedding worker {name} binding differs")
    environment = WorkerEnvironmentIdentity.from_dict(
        payload["worker_environment_identity"]
    )
    if environment.identity_sha256 != payload[
        "worker_environment_identity_sha256"
    ]:
        raise ValueError("embedding worker environment hash differs")
    if not isinstance(payload["execution_policy"], dict):
        raise TypeError("embedding worker execution policy must be an object")
    execution_policy = EmbeddingWorkerExecutionPolicy.from_dict(
        payload["execution_policy"]
    )
    if execution_policy.policy_sha256 != payload["execution_policy_sha256"]:
        raise ValueError("embedding worker execution policy differs")
    for name in (
        "expected_precommitment_sha256",
        "worker_environment_identity_sha256",
        "execution_policy_sha256",
        "code_snapshot_manifest_sha256",
    ):
        value = payload[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"embedding worker {name} differs")


def _private_directory(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"embedding worker {name} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or (resolved.stat().st_mode & 0o077):
        raise ValueError(f"embedding worker {name} must be private")
    return resolved


def _snapshot_artifacts(
    *,
    inventory: ControlScoringInventory,
    artifact_paths: dict[str, Path],
    scratch_path: Path,
    maximum_bytes: int,
) -> tuple[dict[str, Path], int, int]:
    snapshot_root = scratch_path / "snapshots"
    snapshot_root.mkdir(mode=0o700)
    by_content: dict[str, Path] = {}
    total_bytes = 0
    result: dict[str, Path] = {}
    for entry in inventory.entries:
        snapshot = by_content.get(entry.content_sha256)
        if snapshot is None:
            total_bytes += entry.byte_size
            if total_bytes > maximum_bytes:
                raise ValueError("embedding snapshot exceeds execution policy")
            source = artifact_paths[entry.artifact_token]
            if source.is_symlink():
                raise ValueError("embedding snapshot source must not be symlink")
            source = source.resolve(strict=True)
            before = source.stat()
            snapshot = snapshot_root / f"{entry.content_sha256}.input"
            digest = hashlib.sha256()
            written = 0
            source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                target_fd = os.open(
                    snapshot,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                try:
                    while True:
                        chunk = os.read(source_fd, 1_048_576)
                        if not chunk:
                            break
                        digest.update(chunk)
                        view = memoryview(chunk)
                        while view:
                            count = os.write(target_fd, view)
                            view = view[count:]
                        written += len(chunk)
                    os.fsync(target_fd)
                finally:
                    os.close(target_fd)
            finally:
                os.close(source_fd)
            after = source.stat()
            if (
                before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns,
            ) != (
                after.st_dev, after.st_ino, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns,
            ):
                raise RuntimeError("embedding source changed during snapshot")
            if written != entry.byte_size or digest.hexdigest() != (
                entry.content_sha256
            ):
                raise ValueError("embedding snapshot content differs")
            os.chmod(snapshot, 0o400)
            by_content[entry.content_sha256] = snapshot
        result[entry.artifact_token] = snapshot
    return result, len(by_content), total_bytes


if __name__ == "__main__":
    main()
