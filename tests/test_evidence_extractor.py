from __future__ import annotations

from hashlib import sha256
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from cvi.evidence.model_contract import (
    ConvNeXtModelManifest,
    DogFaceNetModelManifest,
    OnnxEvidenceContractError,
    OnnxEvidenceModelManifest,
    OnnxPreprocessingContract,
    PetReIDModelManifest,
)
from cvi.evidence_extractor import EvidenceExtractorRegistry


def _require_onnx() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def _require_ort() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


class _RecordingSessionOptions:
    def __init__(self) -> None:
        self.entries: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.entries[key] = value


class _RecordingProviderSession:
    def __init__(self, providers: list[str]) -> None:
        self.providers = providers
        self.fallback_disabled = False

    def disable_fallback(self) -> None:
        self.fallback_disabled = True

    def get_providers(self) -> list[str]:
        return list(self.providers)

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(
            name="images", shape=["batch", 3, 8, 8], type="tensor(float)"
        )]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(
            name="embedding", shape=["batch", 4], type="tensor(float)"
        )]


class EvidenceExtractorRegistryTests(unittest.TestCase):
    def test_register_and_get(self) -> None:
        registry = EvidenceExtractorRegistry()
        self.assertEqual(registry.names, [])

    def test_get_unknown_raises(self) -> None:
        registry = EvidenceExtractorRegistry()
        with self.assertRaises(KeyError):
            registry.get("nonexistent")

    def test_property_accessors(self) -> None:
        registry = EvidenceExtractorRegistry()

        class _Fake:
            output_dim = 384
            def extract(self, img): return np.zeros(384)
            def extract_batch(self, imgs): return np.zeros((len(imgs), 384))
            def close(self): pass

        fake = _Fake()
        registry.register("visual", fake)
        registry.register("texture", fake)
        registry.register("structural", fake)
        registry.register("nose", fake)
        self.assertIsNotNone(registry.visual)
        self.assertIsNotNone(registry.texture)
        self.assertIsNotNone(registry.structural)
        self.assertIsNotNone(registry.nose)
        self.assertEqual(registry.names, ["visual", "texture", "structural", "nose"])

    def test_close_calls_all(self) -> None:
        registry = EvidenceExtractorRegistry()
        closed: list[str] = []

        class _Fake:
            def close(self): closed.append("called")

        registry.register("a", _Fake())
        registry.register("b", _Fake())
        registry.close()
        self.assertEqual(len(closed), 2)

    def test_from_onnx_dict_skips_missing(self) -> None:
        registry = EvidenceExtractorRegistry.from_onnx_dict(
            {"visual": Path("/nonexistent/model.onnx")}
        )
        self.assertEqual(registry.names, [])


@unittest.skipUnless(_require_onnx(), "onnx not available")
@unittest.skipUnless(_require_ort(), "onnxruntime not available")
class ExtractorConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        from cvi.evidence_extractor import (  # noqa: late import for skip
            ConvNeXtExtractor, DogFaceNetExtractor, OnnxExtractor, PetReIDExtractor,
        )
        self.ConvNeXtExtractor = ConvNeXtExtractor
        self.DogFaceNetExtractor = DogFaceNetExtractor
        self.OnnxExtractor = OnnxExtractor
        self.PetReIDExtractor = PetReIDExtractor

    def test_generic_contract_and_cpu_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            self._write_dummy_onnx(path, output_dim=4)
            manifest = self._manifest(path, output_dim=4)
            ext = self.OnnxExtractor(path, manifest)
            image = Image.fromarray(np.full((5, 7, 3), 255, dtype=np.uint8))
            tensor = ext.preprocess(image)
            embedding = ext.extract(image)

        self.assertEqual(ext.output_dim, 4)
        self.assertEqual(ext.model_sha256, manifest.model_sha256)
        self.assertEqual(tensor.shape, (1, 3, 8, 8))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 1.0, places=6)
        self.assertEqual(ext._sess.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual(embedding.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)

    def test_legacy_generic_constructor_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            self._write_dummy_onnx(path)
            with self.assertRaisesRegex(
                OnnxEvidenceContractError, "manifest is required"
            ):
                self.OnnxExtractor(path, input_size=8, output_dim=4)

    def test_same_shaped_wrong_artifact_fails_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            expected = Path(tmpdir) / "expected.onnx"
            substituted = Path(tmpdir) / "substituted.onnx"
            self._write_dummy_onnx(expected, weight=1.0)
            self._write_dummy_onnx(substituted, weight=2.0)
            manifest = self._manifest(expected)
            with self.assertRaisesRegex(OnnxEvidenceContractError, "SHA256"):
                self.OnnxExtractor(substituted, manifest)

    def test_wrong_output_dimension_fails_graph_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wrong-dim.onnx"
            self._write_dummy_onnx(path, output_dim=5)
            manifest = self._manifest(path, output_dim=4)
            with self.assertRaisesRegex(OnnxEvidenceContractError, "output shape"):
                self.OnnxExtractor(path, manifest)

    def test_wrong_static_spatial_shape_fails_graph_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wrong-spatial.onnx"
            self._write_dummy_onnx(path, input_shape=("batch", 3, 9, 8))
            manifest = self._manifest(path)
            with self.assertRaisesRegex(OnnxEvidenceContractError, "input shape"):
                self.OnnxExtractor(path, manifest)

    def test_multiple_graph_outputs_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multiple-outputs.onnx"
            self._write_dummy_onnx(path, extra_output=True)
            manifest = self._manifest(path)
            with self.assertRaisesRegex(
                OnnxEvidenceContractError, "exactly one input and one output"
            ):
                self.OnnxExtractor(path, manifest)

    def test_zero_runtime_output_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "zero.onnx"
            self._write_dummy_onnx(path, weight=0.0)
            ext = self.OnnxExtractor(path, self._manifest(path))
            with self.assertRaisesRegex(OnnxEvidenceContractError, "nonzero norm"):
                ext.extract(Image.new("RGB", (8, 8), color="white"))

    def test_nonfinite_runtime_output_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonfinite.onnx"
            self._write_dummy_onnx(path, weight=np.nan)
            ext = self.OnnxExtractor(path, self._manifest(path))
            with self.assertRaisesRegex(OnnxEvidenceContractError, "non-finite"):
                ext.extract(Image.new("RGB", (8, 8), color="white"))

    def test_model_adapters_require_model_specific_manifest_types(self) -> None:
        cases = (
            (self.DogFaceNetExtractor, DogFaceNetModelManifest),
            (self.ConvNeXtExtractor, ConvNeXtModelManifest),
            (self.PetReIDExtractor, PetReIDModelManifest),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            self._write_dummy_onnx(path)
            generic = self._manifest(path)
            for extractor_type, manifest_type in cases:
                with self.subTest(extractor=extractor_type.__name__):
                    with self.assertRaisesRegex(
                        OnnxEvidenceContractError, "requires a .*ModelManifest"
                    ):
                        extractor_type(path, generic)
                    specific = self._manifest(path, manifest_type=manifest_type)
                    self.assertEqual(extractor_type(path, specific).output_dim, 4)

    def test_cuda_must_be_explicitly_available(self) -> None:
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            self._write_dummy_onnx(path)
            manifest = self._manifest(path)
            with patch.object(
                ort, "get_available_providers", return_value=["CPUExecutionProvider"]
            ), self.assertRaisesRegex(
                OnnxEvidenceContractError, "requested but is not available"
            ):
                self.OnnxExtractor(path, manifest, use_cuda=True)

    def test_mocked_cpu_session_requests_and_verifies_cpu_only(self) -> None:
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            self._write_dummy_onnx(path)
            session = _RecordingProviderSession(["CPUExecutionProvider"])
            options = _RecordingSessionOptions()
            constructor_kwargs: dict[str, object] = {}

            def create_session(*args, **kwargs):
                constructor_kwargs.update(kwargs)
                return session

            with (
                patch.object(
                    ort,
                    "get_available_providers",
                    return_value=["CPUExecutionProvider"],
                ),
                patch.object(ort, "SessionOptions", return_value=options),
                patch.object(ort, "InferenceSession", side_effect=create_session),
            ):
                self.OnnxExtractor(path, self._manifest(path))
        self.assertEqual(
            constructor_kwargs["providers"], ["CPUExecutionProvider"]
        )
        self.assertEqual(constructor_kwargs["enable_fallback"], 0)
        self.assertEqual(options.entries, {})
        self.assertTrue(session.fallback_disabled)

    def test_mocked_cuda_session_rejects_provider_substitution(self) -> None:
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            self._write_dummy_onnx(path)
            session = _RecordingProviderSession(["CPUExecutionProvider"])
            options = _RecordingSessionOptions()
            constructor_kwargs: dict[str, object] = {}

            def create_session(*args, **kwargs):
                constructor_kwargs.update(kwargs)
                return session

            with (
                patch.object(
                    ort,
                    "get_available_providers",
                    return_value=[
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                ),
                patch.object(ort, "SessionOptions", return_value=options),
                patch.object(ort, "InferenceSession", side_effect=create_session),
                self.assertRaisesRegex(
                    OnnxEvidenceContractError, "strict CUDA session"
                ),
            ):
                self.OnnxExtractor(path, self._manifest(path), use_cuda=True)
        self.assertEqual(
            constructor_kwargs["providers"], ["CUDAExecutionProvider"]
        )
        self.assertEqual(constructor_kwargs["enable_fallback"], 0)
        self.assertEqual(
            options.entries, {"session.disable_cpu_ep_fallback": "1"}
        )
        self.assertTrue(session.fallback_disabled)

    @staticmethod
    def _manifest(
        path: Path,
        *,
        output_dim: int = 4,
        manifest_type: type[OnnxEvidenceModelManifest] = OnnxEvidenceModelManifest,
    ) -> OnnxEvidenceModelManifest:
        return manifest_type(
            model_id="test-only-fixture",
            model_sha256=sha256(path.read_bytes()).hexdigest(),
            input_name="images",
            input_shape=("batch", 3, 8, 8),
            output_name="embedding",
            output_dim=output_dim,
            preprocessing=OnnxPreprocessingContract(
                color_mode="RGB",
                layout="NCHW",
                dtype="float32",
                resize="bilinear",
                scale=1.0 / 255.0,
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
            ),
        )

    @staticmethod
    def _write_dummy_onnx(
        path: Path,
        *,
        output_dim: int = 4,
        input_shape: tuple[int | str, int, int, int] = ("batch", 3, 8, 8),
        weight: float = 1.0,
        extra_output: bool = False,
    ) -> None:
        import onnx
        from onnx import helper, TensorProto, numpy_helper

        w = numpy_helper.from_array(
            np.full((output_dim, 3), weight, dtype=np.float32), "W"
        )
        nodes = [
            helper.make_node(
                "GlobalAveragePool", inputs=["images"], outputs=["pooled"]
            ),
            helper.make_node("Flatten", inputs=["pooled"], outputs=["flat"], axis=1),
            helper.make_node("Gemm", inputs=["flat", "W"], outputs=["y"],
                             alpha=1.0, beta=0.0, transA=0, transB=1),
            helper.make_node("Identity", inputs=["y"], outputs=["embedding"]),
        ]
        outputs = [
            helper.make_tensor_value_info(
                "embedding", TensorProto.FLOAT, [input_shape[0], output_dim]
            )
        ]
        if extra_output:
            outputs.append(
                helper.make_tensor_value_info(
                    "flat", TensorProto.FLOAT, [input_shape[0], 3]
                )
            )
        graph = helper.make_graph(
            nodes, "test",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, input_shape)],
            outputs,
            initializer=[w],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        onnx.save(model, path)
