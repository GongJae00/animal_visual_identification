from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import PIL
    from onnx import TensorProto, helper, numpy_helper
    from PIL import Image
except ModuleNotFoundError:
    OPTIONAL_ONNX_AVAILABLE = False
else:
    OPTIONAL_ONNX_AVAILABLE = True

from cvi.onnx_inference_benchmark import (
    OnnxBenchmarkBackend,
    OnnxInferenceBenchmarkPolicy,
    OnnxInferenceBenchmarkSummary,
    benchmark_onnx_inference,
)
from cvi.acquisition import sha256_file
from cvi.batch_invariance import (
    BatchInvariancePolicy,
    build_batch_invariance_precommitment,
)
from cvi.batch_invariance_runner import (
    BatchFreshWorkerDiscovery,
    BatchFreshWorkerReceipt,
    BatchWorkerExecutionPolicy,
    run_batch_invariance_fresh_worker,
)
from cvi.control_scoring import (
    ArtifactSourceKind,
    ControlScoringInventory,
    EmbeddingCachePolicy,
    ScoringArtifactEntry,
)
from cvi.embedding_producer import (
    EmbeddingProducerConfig,
    EmbeddingProductionPolicy,
)
from cvi.embedding_production_runner import (
    EmbeddingFreshWorkerDiscovery,
    EmbeddingFreshWorkerReceipt,
    EmbeddingWorkerExecutionPolicy,
    build_embedding_production_precommitment,
    run_embedding_production_fresh_worker,
)
from cvi.onnx_backend import (
    ImagePreprocessingConfig,
    OnnxRuntimeBackendConfig,
    OnnxRuntimeCpuBackend,
    OnnxRuntimeCudaBackend,
)
from cvi.process_supervisor import ProcessSupervisorPolicy
from cvi.runtime_library_provenance import (
    RuntimeLibraryManifest,
    RuntimeLibraryPolicy,
    freeze_runtime_library_policy,
)
from cvi.worker_environment import build_sanitized_worker_environment

try:
    version("onnxruntime-gpu")
except PackageNotFoundError:
    OPTIONAL_CUDA_ONNX_AVAILABLE = False
else:
    OPTIONAL_CUDA_ONNX_AVAILABLE = (
        OPTIONAL_ONNX_AVAILABLE
        and "CUDAExecutionProvider" in ort.get_available_providers()
    )

try:
    version("onnxruntime")
except PackageNotFoundError:
    OPTIONAL_CPU_ONNX_AVAILABLE = False
else:
    OPTIONAL_CPU_ONNX_AVAILABLE = OPTIONAL_ONNX_AVAILABLE


def supervisor_policy() -> ProcessSupervisorPolicy:
    return ProcessSupervisorPolicy(
        timeout_seconds=20.0,
        termination_grace_seconds=1.0,
        poll_interval_seconds=0.01,
        maximum_stdout_bytes=4096,
        maximum_stderr_bytes=4096,
    )


def benchmark_policy(
    *,
    cuda: bool = False,
    fresh_processes: int = 2,
) -> OnnxInferenceBenchmarkPolicy:
    return OnnxInferenceBenchmarkPolicy(
        fresh_processes=fresh_processes,
        model_warmup_iterations=1,
        model_repeat_iterations=2,
        end_to_end_repeat_iterations=1,
        maximum_artifacts=4,
        maximum_total_artifact_bytes=1024 * 1024,
        maximum_tensor_bytes=1024 * 1024,
        maximum_worker_result_bytes=1024 * 1024,
        supervisor=supervisor_policy(),
        unrelated_system_work_excluded_by_operator=True,
        gpu_device_index=0 if cuda else None,
        gpu_telemetry_interval_seconds=0.1 if cuda else None,
        unrelated_gpu_work_excluded_by_operator=True if cuda else None,
    )


def preprocessing_payload() -> dict[str, object]:
    from cvi.onnx_backend import (
        ImageChannelOrder,
        ImageInterpolation,
        ImagePreprocessingConfig,
        ImageResizePolicy,
        ImageTensorLayout,
    )

    return ImagePreprocessingConfig(
        width=2,
        height=2,
        color_mode="RGB",
        channel_order=ImageChannelOrder.RGB,
        layout=ImageTensorLayout.NCHW,
        resize_policy=ImageResizePolicy.EXACT,
        interpolation=ImageInterpolation.BILINEAR,
        value_scale=1.0 / 255.0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        maximum_source_width=8,
        maximum_source_height=8,
        maximum_source_pixels=64,
        allowed_source_modes=("L", "RGB"),
        decoder_version=PIL.__version__,
        allowed_formats=("PNG",),
    ).to_dict()


def backend_payload(
    preprocessing: dict[str, object],
    *,
    cuda: bool,
) -> dict[str, object]:
    from cvi.onnx_backend import (
        ImagePreprocessingConfig,
        ImageTensorLayout,
        OnnxGraphOptimization,
        OnnxProviderOption,
        OnnxProviderSpec,
        OnnxRuntimeBackendConfig,
    )

    preprocessing_config = ImagePreprocessingConfig.from_dict(preprocessing)
    providers = (OnnxProviderSpec("CPUExecutionProvider"),)
    if cuda:
        options = {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
            "device_id": "0",
            "do_copy_in_default_stream": "1",
            "enable_cuda_graph": "0",
            "gpu_mem_limit": str(4 * 1024 * 1024 * 1024),
            "prefer_nhwc": "0",
            "use_ep_level_unified_stream": "0",
            "use_tf32": "0",
        }
        providers = (
            OnnxProviderSpec(
                "CUDAExecutionProvider",
                tuple(
                    OnnxProviderOption(key, value)
                    for key, value in sorted(options.items())
                ),
            ),
        )
    return OnnxRuntimeBackendConfig(
        preprocessing_config_sha256=preprocessing_config.config_sha256,
        input_name="images",
        output_name="embedding",
        input_tensor_type="tensor(float)",
        output_tensor_type="tensor(float)",
        input_layout=ImageTensorLayout.NCHW,
        vector_dimension=2,
        maximum_batch_size=4,
        graph_optimization=OnnxGraphOptimization.DISABLE_ALL,
        execution_mode="ORT_SEQUENTIAL",
        intra_op_num_threads=1,
        inter_op_num_threads=1,
        allow_intra_op_spinning=False,
        allow_inter_op_spinning=False,
        enable_mem_pattern=False,
        enable_cpu_mem_arena=False,
        use_deterministic_compute=True,
        providers=providers,
        maximum_model_bytes=1_000_000,
    ).to_dict()


def write_model(path: Path) -> None:
    weights = np.arange(1, 25, dtype=np.float32).reshape(12, 2)
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["images"], ["flat"], axis=1),
            helper.make_node(
                "MatMul",
                ["flat", "weights"],
                ["embedding"],
            ),
        ],
        "cvi-benchmark-test",
        [
            helper.make_tensor_value_info(
                "images",
                TensorProto.FLOAT,
                ["batch", 3, 2, 2],
            )
        ],
        [
            helper.make_tensor_value_info(
                "embedding",
                TensorProto.FLOAT,
                ["batch", 2],
            )
        ],
        [numpy_helper.from_array(weights, name="weights")],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def write_inputs(root: Path, *, cuda: bool) -> dict[str, Path]:
    model = root / "model.onnx"
    backend = root / "backend.json"
    preprocessing = root / "preprocessing.json"
    image_a = root / "a.png"
    image_b = root / "b.png"
    lock = root / "uv.lock"
    runtime_policy = root / "runtime-library-policy.json"
    write_model(model)
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image_a)
    Image.new("RGB", (2, 2), (11, 22, 33)).save(image_b)
    preprocessing_data = preprocessing_payload()
    preprocessing.write_text(json.dumps(preprocessing_data) + "\n")
    backend.write_text(
        json.dumps(backend_payload(preprocessing_data, cuda=cuda)) + "\n"
    )
    lock.write_text("synthetic-lock\n")
    runtime_policy.write_text(
        json.dumps(
            RuntimeLibraryPolicy(
                expected_binaries=(),
                allow_wsl_driver_projection_device_mismatch=cuda,
                allow_discovery_only=True,
            ).to_dict()
        )
        + "\n"
    )
    return {
        "model": model,
        "backend": backend,
        "preprocessing": preprocessing,
        "image_a": image_a,
        "image_b": image_b,
        "lock": lock,
        "runtime_policy": runtime_policy,
    }


def write_strict_runtime_policy_from_summary(
    path: Path,
    summary: OnnxInferenceBenchmarkSummary,
) -> None:
    manifests = tuple(
        RuntimeLibraryManifest.from_dict(
            item["measurement"]["runtime_library_manifest"]
        )
        for item in summary.worker_results
    )
    strict = freeze_runtime_library_policy(
        summary.runtime_library_policy,
        manifests,
    )
    path.write_text(json.dumps(strict.to_dict()) + "\n")


class OnnxInferenceBenchmarkPolicyTests(unittest.TestCase):
    def test_policy_roundtrip_and_scope_validation(self) -> None:
        current = benchmark_policy()
        self.assertEqual(
            OnnxInferenceBenchmarkPolicy.from_dict(current.to_dict()),
            current,
        )
        with self.assertRaisesRegex(ValueError, "fields must be set together"):
            OnnxInferenceBenchmarkPolicy(
                fresh_processes=1,
                model_warmup_iterations=0,
                model_repeat_iterations=1,
                end_to_end_repeat_iterations=1,
                maximum_artifacts=1,
                maximum_total_artifact_bytes=1024,
                maximum_tensor_bytes=1024,
                maximum_worker_result_bytes=1024,
                supervisor=supervisor_policy(),
                unrelated_system_work_excluded_by_operator=True,
                gpu_device_index=0,
            )
        with self.assertRaisesRegex(ValueError, "hard cap"):
            replace(current, maximum_artifacts=4097)

    def test_direct_cli_help_imports_installed_package(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                "tools/benchmark_onnx_inference.py",
                "--help",
            ),
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        self.assertIn("--backend", completed.stdout)


@unittest.skipUnless(OPTIONAL_ONNX_AVAILABLE, "requires ONNX dependencies")
class OnnxInferenceBenchmarkIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        OPTIONAL_CPU_ONNX_AVAILABLE or OPTIONAL_CUDA_ONNX_AVAILABLE,
        "requires one exact ONNX Runtime dependency lane",
    )
    def test_lane_batch_fresh_worker_discovery_and_strict_rerun(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cuda = OPTIONAL_CUDA_ONNX_AVAILABLE
            paths = write_inputs(root, cuda=cuda)
            image_c = root / "c.png"
            Image.new("RGB", (2, 2), (17, 31, 43)).save(image_c)
            artifacts = (paths["image_a"], paths["image_b"], image_c)
            entries = tuple(
                ScoringArtifactEntry(
                    artifact_token=f"artifact-{index}",
                    content_sha256=sha256_file(path),
                    byte_size=path.stat().st_size,
                    source_kind=ArtifactSourceKind.BASE,
                )
                for index, path in enumerate(artifacts)
            )
            inventory = ControlScoringInventory(
                plan_sha256="a" * 64,
                scoring_requests_sha256="b" * 64,
                base_artifact_manifest_sha256="c" * 64,
                base_artifact_verification_sha256="d" * 64,
                control_transform_receipt_sha256="e" * 64,
                entries=entries,
            )
            artifact_paths = {
                entry.artifact_token: path
                for entry, path in zip(entries, artifacts, strict=True)
            }
            preprocessing = ImagePreprocessingConfig.from_dict(
                json.loads(paths["preprocessing"].read_text())
            )
            backend_config = OnnxRuntimeBackendConfig.from_dict(
                json.loads(paths["backend"].read_text())
            )
            backend_class = (
                OnnxRuntimeCudaBackend if cuda else OnnxRuntimeCpuBackend
            )
            backend = backend_class(
                model_path=paths["model"],
                config=backend_config,
                preprocessing=preprocessing,
            )
            lineage = root / "lineage.json"
            lineage.write_text('{"license":"synthetic-test"}\n')
            producer = EmbeddingProducerConfig(
                model_sha256=sha256_file(paths["model"]),
                model_lineage_sha256=sha256_file(lineage),
                preprocessing_sha256=sha256_file(paths["preprocessing"]),
                preprocessing_semantics_sha256=preprocessing.config_sha256,
                dependency_lock_sha256=sha256_file(paths["lock"]),
                code_revision="batch-fresh-worker-synthetic",
                backend=backend.identity,
                vector_dimension=2,
                batch_size=4,
                input_width=2,
                input_height=2,
                input_channels=3,
                input_value_bytes=4,
                l2_epsilon=1e-12,
                normalization_tolerance=1e-6,
            )
            batch_policy = BatchInvariancePolicy(
                absolute_tolerance=1e-6,
                relative_tolerance=1e-6,
                relative_floor=1e-12,
                maximum_raw_l2_drift=1e-5,
                maximum_raw_norm_drift=1e-5,
                maximum_normalized_l2_drift=1e-6,
                maximum_cosine_drift=1e-7,
                maximum_artifacts=8,
                maximum_vector_dimension=8,
                maximum_backend_calls=100,
                maximum_artifact_evaluations=100,
                maximum_comparison_values=1_000,
                maximum_input_bytes_hashed=1_000_000,
                maximum_provenance_bytes_hashed=2_000_000,
                maximum_anchor_temporary_bytes=1_000,
            )
            execution_policy = BatchWorkerExecutionPolicy(
                supervisor=supervisor_policy(),
                maximum_worker_result_bytes=16 * 1024 * 1024,
            )
            _, worker_environment = build_sanitized_worker_environment(
                os.environ,
                python_executable=sys.executable,
            )
            inventory_path = root / "inventory.json"
            artifact_paths_path = root / "artifact-paths.json"
            producer_path = root / "producer.json"
            batch_policy_path = root / "batch-policy.json"
            precommitment_path = root / "precommitment.json"
            inventory_path.write_text(json.dumps(inventory.to_dict()))
            artifact_paths_path.write_text(json.dumps({
                "schema_version": "cvi.batch_artifact_paths.v1",
                "entries": [
                    {"artifact_token": token, "path": str(path)}
                    for token, path in artifact_paths.items()
                ],
            }))
            producer_path.write_text(json.dumps(producer.to_dict()))
            batch_policy_path.write_text(json.dumps(batch_policy.to_dict()))
            discovery_policy = RuntimeLibraryPolicy.from_dict(
                json.loads(paths["runtime_policy"].read_text())
            )

            def write_precommitment(
                runtime_policy: RuntimeLibraryPolicy,
                sequence: int,
            ):
                precommitment = build_batch_invariance_precommitment(
                    inventory=inventory,
                    artifact_paths=artifact_paths,
                    producer_config=producer,
                    provenance_paths={
                        "model": paths["model"],
                        "model_lineage": lineage,
                        "preprocessing": paths["preprocessing"],
                        "dependency_lock": paths["lock"],
                    },
                    policy=batch_policy,
                    runtime_library_policy_sha256=(
                        runtime_policy.policy_sha256
                    ),
                    worker_execution_policy_sha256=(
                        execution_policy.policy_sha256
                    ),
                    worker_environment_identity_sha256=(
                        worker_environment.identity_sha256
                    ),
                    prior_attempt_ledger_sha256="1" * 64,
                    candidate_attempt_token=str(sequence + 2) * 64,
                    precommitment_sequence=sequence,
                )
                precommitment_path.write_text(json.dumps({
                    "schema_version": (
                        "cvi.batch_invariance_precommitment_bundle.v1"
                    ),
                    "precommitment_sha256": (
                        precommitment.precommitment_sha256
                    ),
                    "precommitment": precommitment.to_dict(),
                }))
                return precommitment

            worker_files = {
                "inventory": inventory_path,
                "artifact_paths": artifact_paths_path,
                "producer_config": producer_path,
                "onnx_config": paths["backend"],
                "preprocessing": paths["preprocessing"],
                "model": paths["model"],
                "model_lineage": lineage,
                "dependency_lock": paths["lock"],
                "batch_policy": batch_policy_path,
                "precommitment": precommitment_path,
                "runtime_library_policy": paths["runtime_policy"],
            }
            discoveries = []
            for sequence in (1, 2):
                precommitment = write_precommitment(
                    discovery_policy,
                    sequence,
                )
                result = run_batch_invariance_fresh_worker(
                    backend="cuda" if cuda else "cpu",
                    files=worker_files,
                    precommitment=precommitment,
                    expected_precommitment_sha256=(
                        precommitment.precommitment_sha256
                    ),
                    python_executable=Path(sys.executable),
                    execution_policy=execution_policy,
                    discovery=True,
                )
                self.assertIsInstance(result, BatchFreshWorkerDiscovery)
                discoveries.append(result)
            strict_policy = freeze_runtime_library_policy(
                discovery_policy,
                tuple(item.runtime_library_manifest for item in discoveries),
            )
            paths["runtime_policy"].write_text(
                json.dumps(strict_policy.to_dict())
            )
            strict_precommitment = write_precommitment(strict_policy, 3)
            strict = run_batch_invariance_fresh_worker(
                backend="cuda" if cuda else "cpu",
                files=worker_files,
                precommitment=strict_precommitment,
                expected_precommitment_sha256=(
                    strict_precommitment.precommitment_sha256
                ),
                python_executable=Path(sys.executable),
                execution_policy=execution_policy,
                discovery=False,
            )
            self.assertIsInstance(strict, BatchFreshWorkerReceipt)
            self.assertEqual(strict.batch_receipt.decision.value, (
                "BATCH_COMPOSITION_INVARIANCE_PASS"
            ))
            self.assertEqual(
                BatchFreshWorkerReceipt.from_dict(strict.to_dict()),
                strict,
            )

    @unittest.skipUnless(
        OPTIONAL_CPU_ONNX_AVAILABLE or OPTIONAL_CUDA_ONNX_AVAILABLE,
        "requires one exact ONNX Runtime dependency lane",
    )
    def test_lane_embedding_fresh_worker_discovery_and_strict_rerun(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cuda = OPTIONAL_CUDA_ONNX_AVAILABLE
            paths = write_inputs(root, cuda=cuda)
            image_c = root / "c.png"
            Image.new("RGB", (2, 2), (17, 31, 43)).save(image_c)
            artifacts = (paths["image_a"], paths["image_b"], image_c)
            entries = tuple(
                ScoringArtifactEntry(
                    artifact_token=f"artifact-{index}",
                    content_sha256=sha256_file(path),
                    byte_size=path.stat().st_size,
                    source_kind=ArtifactSourceKind.BASE,
                )
                for index, path in enumerate(artifacts)
            )
            inventory = ControlScoringInventory(
                plan_sha256="a" * 64,
                scoring_requests_sha256="b" * 64,
                base_artifact_manifest_sha256="c" * 64,
                base_artifact_verification_sha256="d" * 64,
                control_transform_receipt_sha256="e" * 64,
                entries=entries,
            )
            artifact_paths = {
                entry.artifact_token: path
                for entry, path in zip(entries, artifacts, strict=True)
            }
            preprocessing = ImagePreprocessingConfig.from_dict(
                json.loads(paths["preprocessing"].read_text())
            )
            backend_config = OnnxRuntimeBackendConfig.from_dict(
                json.loads(paths["backend"].read_text())
            )
            backend_class = (
                OnnxRuntimeCudaBackend if cuda else OnnxRuntimeCpuBackend
            )
            backend = backend_class(
                model_path=paths["model"],
                config=backend_config,
                preprocessing=preprocessing,
            )
            lineage = root / "lineage.json"
            lineage.write_text('{"license":"synthetic-test"}\n')
            producer = EmbeddingProducerConfig(
                model_sha256=sha256_file(paths["model"]),
                model_lineage_sha256=sha256_file(lineage),
                preprocessing_sha256=sha256_file(paths["preprocessing"]),
                preprocessing_semantics_sha256=preprocessing.config_sha256,
                dependency_lock_sha256=sha256_file(paths["lock"]),
                code_revision="embedding-fresh-worker-synthetic",
                backend=backend.identity,
                vector_dimension=2,
                batch_size=2,
                input_width=2,
                input_height=2,
                input_channels=3,
                input_value_bytes=4,
                l2_epsilon=1e-12,
                normalization_tolerance=1e-6,
                warmup_batches=1,
            )
            production_policy = EmbeddingProductionPolicy()
            cache_policy = EmbeddingCachePolicy()
            execution_policy = EmbeddingWorkerExecutionPolicy(
                supervisor=supervisor_policy(),
                maximum_worker_result_bytes=16 * 1024 * 1024,
            )
            _, worker_environment = build_sanitized_worker_environment(
                os.environ,
                python_executable=sys.executable,
            )
            inventory_path = root / "embedding-inventory.json"
            artifact_paths_path = root / "embedding-artifact-paths.json"
            producer_path = root / "embedding-producer.json"
            production_policy_path = root / "embedding-policy.json"
            cache_policy_path = root / "cache-policy.json"
            precommitment_path = root / "embedding-precommitment.json"
            inventory_path.write_text(json.dumps(inventory.to_dict()))
            artifact_paths_path.write_text(json.dumps({
                "schema_version": "cvi.embedding_artifact_paths.v1",
                "entries": [
                    {"artifact_token": token, "path": str(path)}
                    for token, path in artifact_paths.items()
                ],
            }))
            producer_path.write_text(json.dumps(producer.to_dict()))
            production_policy_path.write_text(
                json.dumps(production_policy.to_dict())
            )
            cache_policy_path.write_text(json.dumps(cache_policy.to_dict()))
            discovery_policy = RuntimeLibraryPolicy.from_dict(
                json.loads(paths["runtime_policy"].read_text())
            )

            def write_precommitment(
                runtime_policy: RuntimeLibraryPolicy,
                sequence: int,
                prior_attempt_ledger_sha256: str,
            ):
                precommitment = build_embedding_production_precommitment(
                    inventory=inventory,
                    artifact_paths=artifact_paths,
                    producer_config=producer,
                    provenance_paths={
                        "model": paths["model"],
                        "model_lineage": lineage,
                        "onnx_config": paths["backend"],
                        "preprocessing": paths["preprocessing"],
                        "dependency_lock": paths["lock"],
                    },
                    production_policy=production_policy,
                    cache_policy=cache_policy,
                    runtime_library_policy_sha256=runtime_policy.policy_sha256,
                    worker_execution_policy_sha256=execution_policy.policy_sha256,
                    worker_environment_identity_sha256=(
                        worker_environment.identity_sha256
                    ),
                    prior_attempt_ledger_sha256=(
                        prior_attempt_ledger_sha256
                    ),
                    candidate_attempt_token=str(sequence + 2) * 64,
                    precommitment_sequence=sequence,
                )
                precommitment_path.write_text(json.dumps({
                    "schema_version": (
                        "cvi.embedding_production_precommitment_bundle.v1"
                    ),
                    "precommitment_sha256": (
                        precommitment.precommitment_sha256
                    ),
                    "precommitment": precommitment.to_dict(),
                }))
                return precommitment

            worker_files = {
                "inventory": inventory_path,
                "artifact_paths": artifact_paths_path,
                "producer_config": producer_path,
                "onnx_config": paths["backend"],
                "preprocessing": paths["preprocessing"],
                "model": paths["model"],
                "model_lineage": lineage,
                "dependency_lock": paths["lock"],
                "production_policy": production_policy_path,
                "cache_policy": cache_policy_path,
                "precommitment": precommitment_path,
                "runtime_library_policy": paths["runtime_policy"],
            }
            discoveries = []
            attempt_head = "1" * 64
            for sequence in (1, 2):
                output = root / f"discovery-cache-{sequence}"
                precommitment = write_precommitment(
                    discovery_policy,
                    sequence,
                    attempt_head,
                )
                result = run_embedding_production_fresh_worker(
                    backend="cuda" if cuda else "cpu",
                    files=worker_files,
                    precommitment=precommitment,
                    expected_precommitment_sha256=(
                        precommitment.precommitment_sha256
                    ),
                    python_executable=Path(sys.executable),
                    execution_policy=execution_policy,
                    output_directory=output,
                    discovery=True,
                )
                self.assertIsInstance(result, EmbeddingFreshWorkerDiscovery)
                self.assertFalse(output.exists())
                discoveries.append(result)
                attempt_head = result.completed_attempt_ledger_head_sha256
            discovery_receipts = []
            for sequence, discovery in enumerate(discoveries, start=1):
                path = root / f"embedding-discovery-{sequence}.json"
                path.write_text(json.dumps({
                    "schema_version": (
                        "cvi.embedding_runtime_discovery_bundle.v1"
                    ),
                    "discovery_sha256": discovery.discovery_sha256,
                    "discovery": discovery.to_dict(),
                }))
                discovery_receipts.append(path)
            strict_policy_path = root / "embedding-strict-runtime-policy.json"
            freeze_receipt_path = root / "embedding-freeze-receipt.json"
            freeze_command = [
                sys.executable,
                "tools/freeze_embedding_runtime_library_policy.py",
                "--discovery-policy",
                str(paths["runtime_policy"]),
            ]
            for path, discovery in zip(
                discovery_receipts,
                discoveries,
                strict=True,
            ):
                freeze_command.extend(("--discovery-receipt", str(path)))
                freeze_command.extend((
                    "--expected-discovery-sha256",
                    discovery.discovery_sha256,
                ))
            freeze_command.extend((
                "--policy", str(strict_policy_path),
                "--freeze-receipt", str(freeze_receipt_path),
            ))
            subprocess.run(
                freeze_command,
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            strict_policy = RuntimeLibraryPolicy.from_dict(
                json.loads(strict_policy_path.read_text())
            )
            duplicate_policy = root / "duplicate-strict-policy.json"
            duplicate_receipt = root / "duplicate-freeze-receipt.json"
            duplicate_command = [
                sys.executable,
                "tools/freeze_embedding_runtime_library_policy.py",
                "--discovery-policy", str(paths["runtime_policy"]),
                "--discovery-receipt", str(discovery_receipts[0]),
                "--discovery-receipt", str(discovery_receipts[0]),
                "--expected-discovery-sha256",
                discoveries[0].discovery_sha256,
                "--expected-discovery-sha256",
                discoveries[0].discovery_sha256,
                "--policy", str(duplicate_policy),
                "--freeze-receipt", str(duplicate_receipt),
            ]
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    duplicate_command,
                    cwd=Path(__file__).resolve().parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertFalse(duplicate_policy.exists())
            self.assertFalse(duplicate_receipt.exists())
            paths["runtime_policy"].write_text(json.dumps(strict_policy.to_dict()))
            strict_precommitment = write_precommitment(
                strict_policy,
                3,
                attempt_head,
            )
            output = root / "strict-cache"
            strict = run_embedding_production_fresh_worker(
                backend="cuda" if cuda else "cpu",
                files=worker_files,
                precommitment=strict_precommitment,
                expected_precommitment_sha256=(
                    strict_precommitment.precommitment_sha256
                ),
                python_executable=Path(sys.executable),
                execution_policy=execution_policy,
                output_directory=output,
                discovery=False,
            )
            self.assertIsInstance(strict, EmbeddingFreshWorkerReceipt)
            self.assertEqual(
                len(tuple(output.iterdir())),
                len(strict.production_receipt.cache_manifest.entries),
            )
            self.assertEqual(
                EmbeddingFreshWorkerReceipt.from_dict(strict.to_dict()),
                strict,
            )

    def test_artifact_byte_cap_rejects_before_worker_launch(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = write_inputs(Path(temporary), cuda=False)
            with self.assertRaisesRegex(ValueError, "total byte cap"):
                benchmark_onnx_inference(
                    backend=OnnxBenchmarkBackend.CPU,
                    model_path=paths["model"],
                    backend_config_path=paths["backend"],
                    preprocessing_path=paths["preprocessing"],
                    artifact_paths=(paths["image_a"], paths["image_b"]),
                    dependency_lock_path=paths["lock"],
                    runtime_library_policy_path=paths["runtime_policy"],
                    code_revision="synthetic-cap-smoke",
                    policy=replace(
                        benchmark_policy(fresh_processes=1),
                        maximum_total_artifact_bytes=1,
                    ),
                )

    @unittest.skipUnless(
        OPTIONAL_CPU_ONNX_AVAILABLE,
        "requires the onnx-cpu optional dependency group",
    )
    def test_cpu_fresh_workers_bind_tensor_output_timing_and_rss(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = write_inputs(Path(temporary), cuda=False)
            summary = benchmark_onnx_inference(
                backend=OnnxBenchmarkBackend.CPU,
                model_path=paths["model"],
                backend_config_path=paths["backend"],
                preprocessing_path=paths["preprocessing"],
                artifact_paths=(paths["image_a"], paths["image_b"]),
                dependency_lock_path=paths["lock"],
                runtime_library_policy_path=paths["runtime_policy"],
                code_revision="synthetic-fresh-worker-smoke",
                policy=benchmark_policy(),
            )
        self.assertEqual(summary.fresh_processes, 2)
        self.assertEqual(summary.process_wall_time.samples, 2)
        self.assertEqual(summary.session_construction_time.samples, 2)
        self.assertEqual(summary.warm_preprocessed_inference_time.samples, 4)
        self.assertEqual(summary.end_to_end_inference_time.samples, 2)
        self.assertEqual(summary.output_evaluations, 10)
        self.assertGreater(summary.maximum_worker_ru_maxrss_bytes, 0)
        self.assertIsNone(summary.gpu_telemetry)
        self.assertEqual(
            summary.worker_environment_identity.identity_sha256,
            summary.worker_environment_identity_sha256,
        )
        self.assertEqual(
            set(summary.worker_environment_identity.parent_defined_names),
            {
                name
                for name in summary.worker_environment_identity.sanitized_names
                if name in os.environ
            },
        )
        self.assertEqual(
            summary.interpretation,
            "MEASUREMENT_ONLY_NOT_OPTIMIZATION_PROMOTION_OR_BIOMETRIC_EVIDENCE",
        )
        for item in summary.worker_results:
            self.assertEqual(
                item["supervisor"]["status"],
                "COMPLETED",
            )
            self.assertTrue(item["supervisor"]["stdout_complete"])
            self.assertTrue(item["supervisor"]["stderr_complete"])
        self.assertEqual(
            OnnxInferenceBenchmarkSummary.from_dict(summary.to_dict()),
            summary,
        )
        forged = deepcopy(summary.to_dict())
        forged["worker_results"][0]["measurement"][
            "warm_preprocessed_inference_ns"
        ][0] += 1
        with self.assertRaisesRegex(ValueError, "aggregation differs"):
            OnnxInferenceBenchmarkSummary.from_dict(forged)
        forged_environment = deepcopy(summary.to_dict())
        forged_environment["worker_results"][0]["measurement"][
            "worker_environment_identity_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            ValueError,
            "environment|worker result",
        ):
            OnnxInferenceBenchmarkSummary.from_dict(forged_environment)

    @unittest.skipUnless(
        OPTIONAL_CPU_ONNX_AVAILABLE,
        "requires the onnx-cpu optional dependency group",
    )
    def test_cpu_discovery_policy_strict_rerun_passes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_inputs(root, cuda=False)
            discovery = benchmark_onnx_inference(
                backend=OnnxBenchmarkBackend.CPU,
                model_path=paths["model"],
                backend_config_path=paths["backend"],
                preprocessing_path=paths["preprocessing"],
                artifact_paths=(paths["image_a"], paths["image_b"]),
                dependency_lock_path=paths["lock"],
                runtime_library_policy_path=paths["runtime_policy"],
                code_revision="synthetic-runtime-discovery-cpu",
                policy=benchmark_policy(fresh_processes=1),
            )
            self.assertEqual(discovery.runtime_library_decision, "DISCOVERY_ONLY")
            strict_path = root / "strict-runtime-policy.json"
            write_strict_runtime_policy_from_summary(strict_path, discovery)
            strict = benchmark_onnx_inference(
                backend=OnnxBenchmarkBackend.CPU,
                model_path=paths["model"],
                backend_config_path=paths["backend"],
                preprocessing_path=paths["preprocessing"],
                artifact_paths=(paths["image_a"], paths["image_b"]),
                dependency_lock_path=paths["lock"],
                runtime_library_policy_path=strict_path,
                code_revision="synthetic-runtime-strict-cpu",
                policy=benchmark_policy(fresh_processes=1),
            )
        self.assertEqual(strict.runtime_library_decision, "PASS")
        self.assertGreater(strict.maximum_runtime_library_bytes_hashed, 0)

    @unittest.skipUnless(
        OPTIONAL_CPU_ONNX_AVAILABLE,
        "requires the onnx-cpu optional dependency group",
    )
    def test_cpu_cli_writes_private_nonpromotion_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_inputs(root, cuda=False)
            policy_path = root / "policy.json"
            receipt_path = root / "receipt.json"
            strict_policy_path = root / "strict-runtime-policy.json"
            freeze_receipt_path = root / "runtime-freeze-receipt.json"
            policy_path.write_text(
                json.dumps(
                    benchmark_policy(fresh_processes=1).to_dict()
                )
                + "\n"
            )
            subprocess.run(
                (
                    sys.executable,
                    "tools/benchmark_onnx_inference.py",
                    "--backend",
                    "CPU",
                    "--model",
                    str(paths["model"]),
                    "--backend-config",
                    str(paths["backend"]),
                    "--preprocessing",
                    str(paths["preprocessing"]),
                    "--artifact",
                    str(paths["image_a"]),
                    "--artifact",
                    str(paths["image_b"]),
                    "--dependency-lock",
                    str(paths["lock"]),
                    "--runtime-library-policy",
                    str(paths["runtime_policy"]),
                    "--code-revision",
                    "synthetic-cli-smoke",
                    "--policy",
                    str(policy_path),
                    "--receipt",
                    str(receipt_path),
                ),
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(
                receipt["schema_version"],
                "cvi.onnx_inference_benchmark_receipt.v3",
            )
            self.assertEqual(
                receipt["summary"]["interpretation"],
                "MEASUREMENT_ONLY_NOT_OPTIMIZATION_PROMOTION_OR_BIOMETRIC_EVIDENCE",
            )
            self.assertEqual(
                os.stat(receipt_path).st_mode & 0o777,
                0o600,
            )
            subprocess.run(
                (
                    sys.executable,
                    "tools/freeze_runtime_library_policy.py",
                    "--discovery-receipt",
                    str(receipt_path),
                    "--policy",
                    str(strict_policy_path),
                    "--freeze-receipt",
                    str(freeze_receipt_path),
                ),
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            frozen = RuntimeLibraryPolicy.from_dict(
                json.loads(strict_policy_path.read_text())
            )
            self.assertTrue(frozen.expected_binaries)
            self.assertFalse(frozen.allow_discovery_only)
            self.assertEqual(
                json.loads(freeze_receipt_path.read_text())[
                    "interpretation"
                ],
                "CANDIDATE_POLICY_REQUIRES_REVIEW_AND_STRICT_RERUN",
            )

    @unittest.skipUnless(
        OPTIONAL_CUDA_ONNX_AVAILABLE,
        "requires ONNX Runtime CUDA",
    )
    def test_cuda_worker_retains_device_wide_telemetry_scope(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = write_inputs(Path(temporary), cuda=True)
            summary = benchmark_onnx_inference(
                backend=OnnxBenchmarkBackend.CUDA,
                model_path=paths["model"],
                backend_config_path=paths["backend"],
                preprocessing_path=paths["preprocessing"],
                artifact_paths=(paths["image_a"], paths["image_b"]),
                dependency_lock_path=paths["lock"],
                runtime_library_policy_path=paths["runtime_policy"],
                code_revision="synthetic-fresh-worker-cuda-smoke",
                policy=benchmark_policy(cuda=True, fresh_processes=1),
            )
        self.assertIsNotNone(summary.gpu_telemetry)
        self.assertEqual(summary.gpu_telemetry.scope, "device-wide")
        self.assertTrue(summary.unrelated_gpu_work_excluded_by_operator)
        self.assertEqual(
            summary.worker_results[0]["measurement"]["actual_providers"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    @unittest.skipUnless(
        OPTIONAL_CUDA_ONNX_AVAILABLE,
        "requires ONNX Runtime CUDA",
    )
    def test_cuda_discovery_policy_strict_rerun_passes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_inputs(root, cuda=True)
            discovery = benchmark_onnx_inference(
                backend=OnnxBenchmarkBackend.CUDA,
                model_path=paths["model"],
                backend_config_path=paths["backend"],
                preprocessing_path=paths["preprocessing"],
                artifact_paths=(paths["image_a"], paths["image_b"]),
                dependency_lock_path=paths["lock"],
                runtime_library_policy_path=paths["runtime_policy"],
                code_revision="synthetic-runtime-discovery-cuda",
                policy=benchmark_policy(cuda=True, fresh_processes=1),
            )
            self.assertEqual(discovery.runtime_library_decision, "DISCOVERY_ONLY")
            strict_path = root / "strict-runtime-policy.json"
            write_strict_runtime_policy_from_summary(strict_path, discovery)
            strict = benchmark_onnx_inference(
                backend=OnnxBenchmarkBackend.CUDA,
                model_path=paths["model"],
                backend_config_path=paths["backend"],
                preprocessing_path=paths["preprocessing"],
                artifact_paths=(paths["image_a"], paths["image_b"]),
                dependency_lock_path=paths["lock"],
                runtime_library_policy_path=strict_path,
                code_revision="synthetic-runtime-strict-cuda",
                policy=benchmark_policy(cuda=True, fresh_processes=1),
            )
        self.assertEqual(strict.runtime_library_decision, "PASS")
        self.assertGreater(strict.maximum_runtime_library_bytes_hashed, 0)


if __name__ == "__main__":
    unittest.main()
