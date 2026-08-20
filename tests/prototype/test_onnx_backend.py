from __future__ import annotations

import json
import os
import unittest
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

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

from shared.contracts.runtime_library_provenance import (
    RuntimeLibraryManifest,
    RuntimeLibraryPhase,
)
from data.acquisition import sha256_file
from evaluation.integrity.batch_invariance import (
    BatchInvarianceDecision,
    BatchInvariancePolicy,
    build_batch_invariance_precommitment,
    evaluate_batch_composition_invariance,
)
from evaluation.controls.control_scoring import (
    ArtifactSourceKind,
    ControlScoringInventory,
    EmbeddingCachePolicy,
    ScoringArtifactEntry,
)
from evaluation.integrity.numerical_admission import (
    NumericalAdmissionDecision,
    NumericalDriftPolicy,
    compare_embedding_caches,
)
from shared.foundation.provenance import content_sha256
from prototype.export.embedding_producer import (
    EmbeddingProducerConfig,
    EmbeddingProductionPolicy,
    produce_embedding_cache,
)
from prototype.export.onnx_backend import (
    ImageChannelOrder,
    ImageInterpolation,
    ImagePreprocessingConfig,
    ImageResizePolicy,
    ImageTensorLayout,
    OnnxGraphOptimization,
    OnnxProviderOption,
    OnnxProviderSpec,
    OnnxRuntimeBackendConfig,
    OnnxRuntimeCpuBackend,
    OnnxRuntimeCudaBackend,
    preprocess_image_batch,
)

class _FixtureRuntimeTracker:
    def __init__(self, policy_sha256: str) -> None:
        self.policy = SimpleNamespace(policy_sha256=policy_sha256)
        self.phases = [RuntimeLibraryPhase.DEPENDENCIES_IMPORTED]

    def capture(self, phase: RuntimeLibraryPhase) -> None:
        if phase is not tuple(RuntimeLibraryPhase)[len(self.phases)]:
            raise ValueError("fixture runtime phase order differs")
        self.phases.append(phase)

    def finalize(self) -> RuntimeLibraryManifest:
        return RuntimeLibraryManifest(
            policy_sha256=self.policy.policy_sha256,
            entries=(),
            binary_set_sha256=content_sha256([]),
            maps_snapshots=4,
            maps_bytes_read=0,
            binary_bytes_hashed=0,
            provenance_wall_time_ns=0,
            decision="PASS",
            hard_failures=(),
        )

try:
    version("onnxruntime-gpu")
except PackageNotFoundError:
    OPTIONAL_CUDA_ONNX_AVAILABLE = False
else:
    OPTIONAL_CUDA_ONNX_AVAILABLE = (
        OPTIONAL_ONNX_AVAILABLE
        and "CUDAExecutionProvider" in ort.get_available_providers()
    )

def _preprocessing(
    **overrides: object,
) -> ImagePreprocessingConfig:
    values: dict[str, object] = {
        "width": 2,
        "height": 2,
        "color_mode": "RGB",
        "channel_order": ImageChannelOrder.RGB,
        "layout": ImageTensorLayout.NCHW,
        "resize_policy": ImageResizePolicy.EXACT,
        "interpolation": ImageInterpolation.BILINEAR,
        "value_scale": 1.0 / 255.0,
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
        "maximum_source_width": 8,
        "maximum_source_height": 8,
        "maximum_source_pixels": 64,
        "allowed_source_modes": ("L", "RGB"),
        "decoder_version": PIL.__version__,
        "allowed_formats": ("PNG",),
    }
    values.update(overrides)
    return ImagePreprocessingConfig(**values)

def _backend_config(
    preprocessing: ImagePreprocessingConfig,
    **overrides: object,
) -> OnnxRuntimeBackendConfig:
    values: dict[str, object] = {
        "preprocessing_config_sha256": preprocessing.config_sha256,
        "input_name": "images",
        "output_name": "embedding",
        "input_tensor_type": "tensor(float)",
        "output_tensor_type": "tensor(float)",
        "input_layout": ImageTensorLayout.NCHW,
        "vector_dimension": 2,
        "maximum_batch_size": 4,
        "graph_optimization": OnnxGraphOptimization.DISABLE_ALL,
        "execution_mode": "ORT_SEQUENTIAL",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "allow_intra_op_spinning": False,
        "allow_inter_op_spinning": False,
        "enable_mem_pattern": False,
        "enable_cpu_mem_arena": False,
        "use_deterministic_compute": True,
        "providers": (OnnxProviderSpec("CPUExecutionProvider"),),
        "maximum_model_bytes": 1_000_000,
    }
    values.update(overrides)
    return OnnxRuntimeBackendConfig(**values)

def _cuda_backend_config(
    preprocessing: ImagePreprocessingConfig,
    **overrides: object,
) -> OnnxRuntimeBackendConfig:
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
    return _backend_config(
        preprocessing,
        providers=(
            OnnxProviderSpec(
                "CUDAExecutionProvider",
                tuple(
                    OnnxProviderOption(key, value)
                    for key, value in sorted(options.items())
                ),
            ),
        ),
        **overrides,
    )

def _write_model(
    path: Path,
    *,
    input_name: str = "images",
    output_name: str = "embedding",
    output_dimension: int = 2,
) -> None:
    weights = np.arange(
        1,
        12 * output_dimension + 1,
        dtype=np.float32,
    ).reshape(12, output_dimension)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Flatten",
                [input_name],
                ["flat"],
                axis=1,
            ),
            helper.make_node(
                "MatMul",
                ["flat", "weights"],
                [output_name],
            ),
        ],
        "cvi-test-embedding",
        [
            helper.make_tensor_value_info(
                input_name,
                TensorProto.FLOAT,
                ["batch", 3, 2, 2],
            )
        ],
        [
            helper.make_tensor_value_info(
                output_name,
                TensorProto.FLOAT,
                ["batch", output_dimension],
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

def _write_conv_model(path: Path) -> None:
    weights = (
        np.arange(1, 2 * 3 * 3 * 3 + 1, dtype=np.float32)
        .reshape(2, 3, 3, 3)
        / np.float32(100.0)
    )
    bias = np.array([0.01, -0.02], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["images", "weights", "bias"],
                ["convolution"],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node(
                "Relu",
                ["convolution"],
                ["activation"],
            ),
            helper.make_node(
                "GlobalAveragePool",
                ["activation"],
                ["pooled"],
            ),
            helper.make_node(
                "Flatten",
                ["pooled"],
                ["embedding"],
                axis=1,
            ),
        ],
        "cvi-test-convolution",
        [
            helper.make_tensor_value_info(
                "images",
                TensorProto.FLOAT,
                ["batch", 3, 4, 4],
            )
        ],
        [
            helper.make_tensor_value_info(
                "embedding",
                TensorProto.FLOAT,
                ["batch", 2],
            )
        ],
        [
            numpy_helper.from_array(weights, name="weights"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(model, path)

def _write_rejected_superanimal_contract(path: Path) -> None:
    weights = np.ones((3, 39), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "GlobalAveragePool",
                ["images"],
                ["pooled"],
            ),
            helper.make_node(
                "Flatten",
                ["pooled"],
                ["flat"],
                axis=1,
            ),
            helper.make_node(
                "MatMul",
                ["flat", "weights"],
                ["embedding"],
            ),
        ],
        "renamed-superanimal-replacement",
        [
            helper.make_tensor_value_info(
                "images",
                TensorProto.FLOAT,
                ["batch", 3, 384, 384],
            )
        ],
        [
            helper.make_tensor_value_info(
                "embedding",
                TensorProto.FLOAT,
                ["batch", 39],
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

def _write_rgb(path: Path, *, delta: int = 0) -> None:
    pixels = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120 + delta]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")

@unittest.skipUnless(
    OPTIONAL_ONNX_AVAILABLE,
    "requires the cpu optional dependency group",
)
class OnnxBackendTests(unittest.TestCase):
    def test_superanimal_replacement_contract_is_rejected(self) -> None:
        with TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "renamed-model.onnx"
            _write_rejected_superanimal_contract(model)
            preprocessing = _preprocessing(
                width=384,
                height=384,
                maximum_source_width=384,
                maximum_source_height=384,
                maximum_source_pixels=384 * 384,
            )
            config = _backend_config(
                preprocessing,
                vector_dimension=39,
            )
            with self.assertRaisesRegex(RuntimeError, "replacement ONNX contract"):
                OnnxRuntimeCpuBackend(
                    model_path=model,
                    config=config,
                    preprocessing=preprocessing,
                )

    def _run_batch_invariance_smoke(
        self,
        root: Path,
        backend_class: type,
        backend_config_factory: object,
    ) -> None:
        model = root / "model.onnx"
        _write_model(model)
        preprocessing = _preprocessing()
        backend_config = backend_config_factory(preprocessing)
        backend = backend_class(
            model_path=model,
            config=backend_config,
            preprocessing=preprocessing,
        )
        paths: dict[str, Path] = {}
        entries = []
        for index in range(5):
            token = f"batch-{index}"
            path = root / f"{token}.png"
            _write_rgb(path, delta=index)
            paths[token] = path
            entries.append(
                ScoringArtifactEntry(
                    artifact_token=token,
                    content_sha256=sha256_file(path),
                    byte_size=path.stat().st_size,
                    source_kind=ArtifactSourceKind.BASE,
                )
            )
        inventory = ControlScoringInventory(
            plan_sha256="a" * 64,
            scoring_requests_sha256="b" * 64,
            base_artifact_manifest_sha256="c" * 64,
            base_artifact_verification_sha256="d" * 64,
            control_transform_receipt_sha256="e" * 64,
            entries=tuple(entries),
        )
        preprocessing_file = root / "preprocessing.json"
        preprocessing_file.write_text(
            json.dumps(preprocessing.to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        lineage = root / "lineage.json"
        lineage.write_text('{"license":"synthetic-test"}', encoding="utf-8")
        lock = root / "lock.txt"
        lock.write_text("synthetic-lock", encoding="utf-8")
        producer_config = EmbeddingProducerConfig(
            model_sha256=sha256_file(model),
            model_lineage_sha256=sha256_file(lineage),
            preprocessing_sha256=sha256_file(preprocessing_file),
            preprocessing_semantics_sha256=preprocessing.config_sha256,
            dependency_lock_sha256=sha256_file(lock),
            code_revision="batch-invariance-onnx-smoke",
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
        provenance_paths = {
            "model": model,
            "model_lineage": lineage,
            "preprocessing": preprocessing_file,
            "dependency_lock": lock,
        }
        batch_policy = BatchInvariancePolicy(
            absolute_tolerance=1e-6,
            relative_tolerance=1e-6,
            relative_floor=1e-12,
            maximum_raw_l2_drift=1e-5,
            maximum_raw_norm_drift=1e-5,
            maximum_normalized_l2_drift=1e-6,
            maximum_cosine_drift=1e-7,
            maximum_artifacts=10,
            maximum_vector_dimension=10,
            maximum_backend_calls=100,
            maximum_artifact_evaluations=100,
            maximum_comparison_values=1_000,
            maximum_input_bytes_hashed=1_000_000,
            maximum_provenance_bytes_hashed=2_000_000,
            maximum_anchor_temporary_bytes=1_000,
        )
        runtime_policy_sha256 = "8" * 64
        precommitment = build_batch_invariance_precommitment(
            inventory=inventory,
            artifact_paths=paths,
            producer_config=producer_config,
            provenance_paths=provenance_paths,
            policy=batch_policy,
            runtime_library_policy_sha256=runtime_policy_sha256,
            worker_execution_policy_sha256="a" * 64,
            worker_environment_identity_sha256="b" * 64,
            prior_attempt_ledger_sha256="6" * 64,
            candidate_attempt_token="7" * 64,
            precommitment_sequence=1,
        )
        receipt = evaluate_batch_composition_invariance(
            backend=backend,
            inventory=inventory,
            artifact_paths=paths,
            producer_config=producer_config,
            provenance_paths=provenance_paths,
            policy=batch_policy,
            precommitment=precommitment,
            expected_precommitment_sha256=precommitment.precommitment_sha256,
            runtime_library_tracker=_FixtureRuntimeTracker(
                runtime_policy_sha256
            ),
        )
        self.assertIs(receipt.decision, BatchInvarianceDecision.PASS)

    @unittest.skipUnless(OPTIONAL_ONNX_AVAILABLE, "optional ONNX stack unavailable")
    def test_cpu_batch_composition_invariance_smoke(self) -> None:
        with TemporaryDirectory() as temporary:
            self._run_batch_invariance_smoke(
                Path(temporary),
                OnnxRuntimeCpuBackend,
                _backend_config,
            )

    @unittest.skipUnless(
        OPTIONAL_CUDA_ONNX_AVAILABLE,
        "CUDA ONNX Runtime stack unavailable",
    )
    def test_cuda_batch_composition_invariance_smoke(self) -> None:
        with TemporaryDirectory() as temporary:
            self._run_batch_invariance_smoke(
                Path(temporary),
                OnnxRuntimeCudaBackend,
                _cuda_backend_config,
            )

    def test_preprocessing_config_roundtrip_and_semantic_hash(self) -> None:
        config = _preprocessing()
        self.assertEqual(
            ImagePreprocessingConfig.from_dict(config.to_dict()),
            config,
        )
        changed = replace(
            config,
            operation_order="CONVERT_THEN_RESIZE",
            value_scale=1.0,
        )
        self.assertNotEqual(changed.config_sha256, config.config_sha256)
        unknown = config.to_dict()
        unknown["undeclared"] = True
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            ImagePreprocessingConfig.from_dict(unknown)

    def test_preprocessing_golden_rgb_and_grayscale(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb = root / "rgb.png"
            gray = root / "gray.png"
            _write_rgb(rgb)
            Image.fromarray(
                np.array([[1, 2], [3, 4]], dtype=np.uint8),
                mode="L",
            ).save(gray, format="PNG")
            tensor = preprocess_image_batch((rgb,), _preprocessing())
            expected_hwc = np.array(
                [
                    [[10, 20, 30], [40, 50, 60]],
                    [[70, 80, 90], [100, 110, 120]],
                ],
                dtype=np.float32,
            ) * np.float32(1.0 / 255.0)
            np.testing.assert_array_equal(
                tensor,
                np.transpose(expected_hwc, (2, 0, 1))[None, ...],
            )
            gray_tensor = preprocess_image_batch(
                (gray,),
                _preprocessing(),
            )
            expected_gray = np.array(
                [[1, 2], [3, 4]],
                dtype=np.float32,
            ) * np.float32(1.0 / 255.0)
            np.testing.assert_array_equal(
                gray_tensor[0, 0],
                expected_gray,
            )
            np.testing.assert_array_equal(
                gray_tensor[0, 0],
                gray_tensor[0, 1],
            )
            np.testing.assert_array_equal(
                gray_tensor[0, 1],
                gray_tensor[0, 2],
            )

    def test_shortest_edge_center_crop_preprocessing_is_exact(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "landscape.png"
            pixels = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
            Image.fromarray(pixels, mode="RGB").save(path, format="PNG")
            config = _preprocessing(
                schema_version="cvi.image_preprocessing.v2",
                width=2,
                height=2,
                resize_policy=ImageResizePolicy.SHORTEST_EDGE_CENTER_CROP,
                interpolation=ImageInterpolation.BICUBIC,
                operation_order="CONVERT_THEN_RESIZE_THEN_CENTER_CROP",
                resize_shortest_edge=4,
            )
            self.assertEqual(ImagePreprocessingConfig.from_dict(config.to_dict()), config)
            tensor = preprocess_image_batch((path,), config)
            expected = np.asarray(
                Image.fromarray(pixels, mode="RGB")
                .resize((8, 4), Image.Resampling.BICUBIC)
                .crop((3, 1, 5, 3)),
                dtype=np.float32,
            ) * np.float32(1.0 / 255.0)
            np.testing.assert_array_equal(
                tensor,
                np.transpose(expected, (2, 0, 1))[None],
            )

    def test_preprocessing_rejects_undeclared_image_semantics(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgba = root / "rgba.png"
            too_large = root / "large.png"
            corrupt = root / "corrupt.png"
            Image.new("RGBA", (2, 2), (1, 2, 3, 4)).save(rgba)
            Image.new("RGB", (9, 2), (1, 2, 3)).save(too_large)
            corrupt.write_bytes(b"not-a-png")
            with self.assertRaisesRegex(ValueError, "source mode"):
                preprocess_image_batch((rgba,), _preprocessing())
            with self.assertRaisesRegex(ValueError, "dimension cap"):
                preprocess_image_batch(
                    (too_large,),
                    _preprocessing(
                        resize_policy=ImageResizePolicy.STRETCH,
                    ),
                )
            with self.assertRaises(Exception):
                preprocess_image_batch((corrupt,), _preprocessing())

    def test_cpu_backend_is_exact_provider_and_repeatable(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            image_a = root / "a.png"
            image_b = root / "b.png"
            _write_model(model)
            _write_rgb(image_a)
            _write_rgb(image_b, delta=1)
            preprocessing = _preprocessing()
            config = _backend_config(preprocessing)
            original_session = ort.InferenceSession
            constructor_calls: list[dict[str, object]] = []

            def recording_session(
                *args: object,
                **kwargs: object,
            ) -> object:
                constructor_calls.append(dict(kwargs))
                return original_session(*args, **kwargs)

            with patch.object(
                ort,
                "InferenceSession",
                side_effect=recording_session,
            ):
                backend = OnnxRuntimeCpuBackend(
                    model_path=model,
                    config=config,
                    preprocessing=preprocessing,
                )
            self.assertEqual(len(constructor_calls), 1)
            self.assertEqual(
                constructor_calls[0]["enable_fallback"],
                0,
            )
            self.assertEqual(
                backend.actual_providers,
                ("CPUExecutionProvider",),
            )
            self.assertEqual(
                backend.actual_provider_options,
                {"CPUExecutionProvider": {}},
            )
            self.assertEqual(backend.model_sha256, sha256_file(model))
            first = tuple(
                tuple(row)
                for row in backend.infer_batch((image_a, image_b))
            )
            for _ in range(10):
                current = tuple(
                    tuple(row)
                    for row in backend.infer_batch((image_a, image_b))
                )
                self.assertEqual(current, first)

    def test_cpu_backend_rejects_malformed_numpy_outputs_without_copy(
        self,
    ) -> None:
        class FakeSession:
            def __init__(self, output: np.ndarray) -> None:
                self.output = output
                self.calls = 0

            def run(
                self,
                output_names: list[str],
                inputs: dict[str, np.ndarray],
            ) -> list[np.ndarray]:
                self.calls += 1
                return [self.output]

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            image = root / "image.png"
            _write_model(model)
            _write_rgb(image)
            preprocessing = _preprocessing()
            backend = OnnxRuntimeCpuBackend(
                model_path=model,
                config=_backend_config(preprocessing),
                preprocessing=preprocessing,
            )
            malformed = (
                np.ones((1, 2), dtype=np.float64),
                np.ones((1, 2, 1), dtype=np.float32),
                np.ones((1, 4), dtype=np.float32)[:, ::2],
            )
            messages = ("dtype", "shape", "C-contiguous")
            for output, message in zip(
                malformed,
                messages,
                strict=True,
            ):
                with self.subTest(message=message):
                    fake = FakeSession(output)
                    backend._session = fake
                    with self.assertRaisesRegex(ValueError, message):
                        backend.infer_batch((image,))
                    self.assertEqual(fake.calls, 1)

            class FailingSession:
                def __init__(self) -> None:
                    self.calls = 0

                def run(
                    self,
                    output_names: list[str],
                    inputs: dict[str, np.ndarray],
                ) -> list[np.ndarray]:
                    self.calls += 1
                    raise RuntimeError("synthetic EPFail")

            failing = FailingSession()
            backend._session = failing
            with self.assertRaisesRegex(RuntimeError, "synthetic EPFail"):
                backend.infer_batch((image,))
            self.assertEqual(failing.calls, 1)

    def test_preprocessed_tensor_boundary_is_equivalent_and_strict(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            image_a = root / "a.png"
            image_b = root / "b.png"
            _write_model(model)
            _write_rgb(image_a)
            _write_rgb(image_b, delta=1)
            preprocessing = _preprocessing()
            backend = OnnxRuntimeCpuBackend(
                model_path=model,
                config=_backend_config(preprocessing),
                preprocessing=preprocessing,
            )
            tensor = preprocess_image_batch(
                (image_a, image_b),
                preprocessing,
            )
            path_rows = np.asarray(
                [tuple(row) for row in backend.infer_batch((image_a, image_b))],
                dtype=np.float32,
            )
            tensor_rows = np.asarray(
                [
                    tuple(row)
                    for row in backend.infer_preprocessed_batch(tensor)
                ],
                dtype=np.float32,
            )
            np.testing.assert_array_equal(tensor_rows, path_rows)

            invalid = (
                (tensor.astype(np.float64), "dtype"),
                (tensor[:, :, :, :1], "shape"),
                (tensor[:, :, :, ::-1], "C-contiguous"),
                (np.full_like(tensor, np.nan), "finite"),
                (np.repeat(tensor[:1], 5, axis=0), "batch size"),
            )
            for malformed, message in invalid:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        backend.infer_preprocessed_batch(malformed)
            with self.assertRaisesRegex(TypeError, "NumPy"):
                backend.infer_preprocessed_batch([[1.0]])

    def test_cpu_backend_rejects_metadata_environment_and_external_data(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            _write_model(model)
            preprocessing = _preprocessing()
            with self.assertRaisesRegex(ValueError, "input name"):
                OnnxRuntimeCpuBackend(
                    model_path=model,
                    config=_backend_config(
                        preprocessing,
                        input_name="wrong",
                    ),
                    preprocessing=preprocessing,
                )
            with patch.dict(
                os.environ,
                {"ORT_LOAD_CONFIG_FROM_MODEL": "1"},
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "ORT_LOAD_CONFIG_FROM_MODEL",
                ):
                    OnnxRuntimeCpuBackend(
                        model_path=model,
                        config=_backend_config(preprocessing),
                        preprocessing=preprocessing,
                    )

            external_model = root / "external.onnx"
            external_weights = root / "external.weights"
            _write_model(external_model)
            loaded = onnx.load_model(external_model)
            onnx.save_model(
                loaded,
                external_model,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_weights.name,
                size_threshold=0,
            )
            self.assertTrue(external_weights.exists())
            with self.assertRaisesRegex(ValueError, "external-data"):
                OnnxRuntimeCpuBackend(
                    model_path=external_model,
                    config=_backend_config(preprocessing),
                    preprocessing=preprocessing,
                )

    def test_cpu_backend_integrates_with_atomic_embedding_cache(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            image_a = root / "a.png"
            image_b = root / "b.png"
            lineage = root / "lineage.json"
            preprocessing_path = root / "preprocessing.json"
            lock = root / "uv.lock"
            output = root / "cache"
            output.mkdir()
            _write_model(model)
            _write_rgb(image_a)
            image_b.write_bytes(image_a.read_bytes())
            lineage.write_text('{"source":"synthetic-test"}\n')
            lock.write_text("synthetic-lock\n")
            preprocessing = _preprocessing()
            preprocessing_path.write_text(
                json.dumps(
                    preprocessing.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            backend_config = _backend_config(preprocessing)
            backend = OnnxRuntimeCpuBackend(
                model_path=model,
                config=backend_config,
                preprocessing=preprocessing,
            )
            entries = tuple(
                ScoringArtifactEntry(
                    artifact_token=token,
                    content_sha256=sha256_file(path),
                    byte_size=path.stat().st_size,
                    source_kind=ArtifactSourceKind.BASE,
                )
                for token, path in (
                    ("token-a", image_a),
                    ("token-b", image_b),
                )
            )
            inventory = ControlScoringInventory(
                plan_sha256="a" * 64,
                scoring_requests_sha256="b" * 64,
                base_artifact_manifest_sha256="c" * 64,
                base_artifact_verification_sha256="d" * 64,
                control_transform_receipt_sha256="e" * 64,
                entries=entries,
            )
            producer_config = EmbeddingProducerConfig(
                model_sha256=sha256_file(model),
                model_lineage_sha256=sha256_file(lineage),
                preprocessing_sha256=sha256_file(preprocessing_path),
                preprocessing_semantics_sha256=(
                    preprocessing.config_sha256
                ),
                dependency_lock_sha256=sha256_file(lock),
                code_revision="synthetic-onnx-smoke",
                backend=backend.identity,
                vector_dimension=2,
                batch_size=2,
                input_width=2,
                input_height=2,
                input_channels=3,
                input_value_bytes=4,
                l2_epsilon=1e-12,
                normalization_tolerance=1e-6,
            )
            receipt = produce_embedding_cache(
                inventory=inventory,
                artifact_paths={
                    "token-a": image_a,
                    "token-b": image_b,
                },
                model_path=model,
                model_lineage_path=lineage,
                preprocessing_path=preprocessing_path,
                dependency_lock_path=lock,
                config=producer_config,
                production_policy=EmbeddingProductionPolicy(),
                cache_policy=EmbeddingCachePolicy(),
                backend=backend,
                output_directory=output,
            )
            self.assertEqual(receipt.cost.artifact_bindings, 2)
            self.assertEqual(receipt.cost.unique_content_inputs, 1)
            self.assertEqual(receipt.cost.production_artifact_evaluations, 1)
            self.assertEqual(len(tuple(output.iterdir())), 1)

    def test_cuda_config_rejects_unsafe_or_incomplete_options(self) -> None:
        preprocessing = _preprocessing()
        valid = _cuda_backend_config(preprocessing)
        options = {
            item.key: item.value for item in valid.providers[0].options
        }
        options["use_tf32"] = "1"
        unsafe = replace(
            valid,
            providers=(
                OnnxProviderSpec(
                    "CUDAExecutionProvider",
                    tuple(
                        OnnxProviderOption(key, value)
                        for key, value in sorted(options.items())
                    ),
                ),
            ),
        )
        with TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.onnx"
            _write_model(model)
            with self.assertRaisesRegex(ValueError, "use_tf32"):
                OnnxRuntimeCudaBackend(
                    model_path=model,
                    config=unsafe,
                    preprocessing=preprocessing,
                )
            incomplete = replace(
                valid,
                providers=(
                    OnnxProviderSpec(
                        "CUDAExecutionProvider",
                        valid.providers[0].options[:-1],
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "option set"):
                OnnxRuntimeCudaBackend(
                    model_path=model,
                    config=incomplete,
                    preprocessing=preprocessing,
                )

    @unittest.skipUnless(
        OPTIONAL_CUDA_ONNX_AVAILABLE,
        "requires the cuda optional dependency group",
    )
    def test_cuda_backend_rejects_silent_provider_fallback(self) -> None:
        class SilentCpuSession:
            def disable_fallback(self) -> None:
                return None

            def get_providers(self) -> list[str]:
                return ["CPUExecutionProvider"]

        with TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.onnx"
            _write_model(model)
            preprocessing = _preprocessing()
            constructor_kwargs: dict[str, object] = {}

            def silent_session(
                *args: object,
                **kwargs: object,
            ) -> SilentCpuSession:
                constructor_kwargs.update(kwargs)
                return SilentCpuSession()

            with patch.object(
                ort,
                "InferenceSession",
                side_effect=silent_session,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "strict CUDA session",
                ):
                    OnnxRuntimeCudaBackend(
                        model_path=model,
                        config=_cuda_backend_config(preprocessing),
                        preprocessing=preprocessing,
                    )
            self.assertEqual(constructor_kwargs["enable_fallback"], 0)
            session_options = constructor_kwargs["sess_options"]
            self.assertEqual(
                session_options.get_session_config_entry(
                    "session.disable_cpu_ep_fallback"
                ),
                "1",
            )

    @unittest.skipUnless(
        OPTIONAL_CUDA_ONNX_AVAILABLE,
        "requires the cuda optional dependency group",
    )
    def test_cuda_full_graph_repeatability_and_cpu_drift_smoke(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            image_a = root / "a.png"
            image_b = root / "b.png"
            _write_model(model)
            _write_rgb(image_a)
            _write_rgb(image_b, delta=1)
            preprocessing = _preprocessing()
            cuda = OnnxRuntimeCudaBackend(
                model_path=model,
                config=_cuda_backend_config(preprocessing),
                preprocessing=preprocessing,
            )
            cpu = OnnxRuntimeCpuBackend(
                model_path=model,
                config=_backend_config(preprocessing),
                preprocessing=preprocessing,
            )
            self.assertEqual(
                cuda.actual_providers,
                ("CUDAExecutionProvider", "CPUExecutionProvider"),
            )
            cuda_rows = np.asarray(
                [tuple(row) for row in cuda.infer_batch((image_a, image_b))],
                dtype=np.float32,
            )
            for _ in range(20):
                repeated = np.asarray(
                    [
                        tuple(row)
                        for row in cuda.infer_batch((image_a, image_b))
                    ],
                    dtype=np.float32,
                )
                np.testing.assert_array_equal(repeated, cuda_rows)
            singleton = np.asarray(
                [tuple(row) for row in cuda.infer_batch((image_a,))],
                dtype=np.float32,
            )
            np.testing.assert_array_equal(singleton[0], cuda_rows[0])
            cpu_rows = np.asarray(
                [tuple(row) for row in cpu.infer_batch((image_a, image_b))],
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                cuda_rows,
                cpu_rows,
                rtol=1e-6,
                atol=1e-6,
            )

    @unittest.skipUnless(
        OPTIONAL_CUDA_ONNX_AVAILABLE,
        "requires the cuda optional dependency group",
    )
    def test_cuda_convolution_path_is_dynamic_batch_and_cpu_close(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "convolution.onnx"
            image_a = root / "a.png"
            image_b = root / "b.png"
            _write_conv_model(model)
            Image.new("RGB", (4, 4), (20, 40, 60)).save(image_a)
            Image.new("RGB", (4, 4), (21, 42, 63)).save(image_b)
            preprocessing = _preprocessing(width=4, height=4)
            cuda = OnnxRuntimeCudaBackend(
                model_path=model,
                config=_cuda_backend_config(preprocessing),
                preprocessing=preprocessing,
            )
            cpu = OnnxRuntimeCpuBackend(
                model_path=model,
                config=_backend_config(preprocessing),
                preprocessing=preprocessing,
            )

            cuda_batch = np.asarray(
                [tuple(row) for row in cuda.infer_batch((image_a, image_b))],
                dtype=np.float32,
            )
            tensor = preprocess_image_batch(
                (image_a, image_b),
                preprocessing,
            )
            cuda_tensor_batch = np.asarray(
                [
                    tuple(row)
                    for row in cuda.infer_preprocessed_batch(tensor)
                ],
                dtype=np.float32,
            )
            cuda_single = np.asarray(
                [tuple(row) for row in cuda.infer_batch((image_a,))],
                dtype=np.float32,
            )
            cpu_batch = np.asarray(
                [tuple(row) for row in cpu.infer_batch((image_a, image_b))],
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                cuda_single[0],
                cuda_batch[0],
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_array_equal(cuda_tensor_batch, cuda_batch)
            np.testing.assert_allclose(
                cuda_batch,
                cpu_batch,
                rtol=1e-5,
                atol=1e-6,
            )

    @unittest.skipUnless(
        OPTIONAL_CUDA_ONNX_AVAILABLE,
        "requires the cuda optional dependency group",
    )
    def test_cpu_cuda_canonical_cache_numerical_admission_smoke(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            image_a = root / "a.png"
            image_b = root / "b.png"
            lineage = root / "lineage.json"
            preprocessing_path = root / "preprocessing.json"
            lock = root / "uv.lock"
            reference_root = root / "reference-cache"
            candidate_root = root / "candidate-cache"
            reference_root.mkdir()
            candidate_root.mkdir()
            _write_model(model)
            _write_rgb(image_a)
            _write_rgb(image_b, delta=1)
            lineage.write_text('{"source":"synthetic-test"}\n')
            lock.write_text("synthetic-lock\n")
            preprocessing = _preprocessing()
            preprocessing_path.write_text(
                json.dumps(
                    preprocessing.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            cpu = OnnxRuntimeCpuBackend(
                model_path=model,
                config=_backend_config(preprocessing),
                preprocessing=preprocessing,
            )
            cuda = OnnxRuntimeCudaBackend(
                model_path=model,
                config=_cuda_backend_config(preprocessing),
                preprocessing=preprocessing,
            )
            entries = tuple(
                ScoringArtifactEntry(
                    artifact_token=token,
                    content_sha256=sha256_file(path),
                    byte_size=path.stat().st_size,
                    source_kind=ArtifactSourceKind.BASE,
                )
                for token, path in (
                    ("token-a", image_a),
                    ("token-b", image_b),
                )
            )
            inventory = ControlScoringInventory(
                plan_sha256="a" * 64,
                scoring_requests_sha256="b" * 64,
                base_artifact_manifest_sha256="c" * 64,
                base_artifact_verification_sha256="d" * 64,
                control_transform_receipt_sha256="e" * 64,
                entries=entries,
            )

            def producer_config(backend: object) -> EmbeddingProducerConfig:
                return EmbeddingProducerConfig(
                    model_sha256=sha256_file(model),
                    model_lineage_sha256=sha256_file(lineage),
                    preprocessing_sha256=sha256_file(preprocessing_path),
                    preprocessing_semantics_sha256=(
                        preprocessing.config_sha256
                    ),
                    dependency_lock_sha256=sha256_file(lock),
                    code_revision="synthetic-cpu-cuda-smoke",
                    backend=backend.identity,
                    vector_dimension=2,
                    batch_size=2,
                    input_width=2,
                    input_height=2,
                    input_channels=3,
                    input_value_bytes=4,
                    l2_epsilon=1e-12,
                    normalization_tolerance=1e-6,
                )

            reference_config = producer_config(cpu)
            candidate_config = producer_config(cuda)
            common = {
                "inventory": inventory,
                "artifact_paths": {
                    "token-a": image_a,
                    "token-b": image_b,
                },
                "model_path": model,
                "model_lineage_path": lineage,
                "preprocessing_path": preprocessing_path,
                "dependency_lock_path": lock,
                "production_policy": EmbeddingProductionPolicy(),
                "cache_policy": EmbeddingCachePolicy(),
            }
            reference_receipt = produce_embedding_cache(
                **common,
                config=reference_config,
                backend=cpu,
                output_directory=reference_root,
            )
            candidate_receipt = produce_embedding_cache(
                **common,
                config=candidate_config,
                backend=cuda,
                output_directory=candidate_root,
            )
            admission = compare_embedding_caches(
                reference_manifest=reference_receipt.cache_manifest,
                candidate_manifest=candidate_receipt.cache_manifest,
                reference_config=reference_config,
                candidate_config=candidate_config,
                reference_root=reference_root,
                candidate_root=candidate_root,
                policy=NumericalDriftPolicy(
                    absolute_tolerance=1e-6,
                    relative_tolerance=1e-6,
                    relative_floor=1e-12,
                    maximum_l2_drift=1e-6,
                    maximum_cosine_drift=1e-8,
                ),
            )
            self.assertIs(
                admission.decision,
                NumericalAdmissionDecision.PASS,
            )

if __name__ == "__main__":
    unittest.main()
