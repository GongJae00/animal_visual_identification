from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cvi.evidence_extractor import (
    EvidenceExtractorRegistry,
    SuperAnimalExtractor,
)
from cvi.search_engine import FusePlan, MultiEvidenceEmbedding


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


class MultiEvidenceEmbeddingTests(unittest.TestCase):
    def test_fused_concat(self) -> None:
        emb = MultiEvidenceEmbedding(
            visual=np.ones(384, dtype=np.float32),
            texture=np.ones(128, dtype=np.float32) * 2,
            structural=np.ones(128, dtype=np.float32) * 3,
        )
        fused = emb.fused()
        self.assertEqual(fused.ndim, 1)
        norm = np.linalg.norm(fused)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_fused_partial(self) -> None:
        emb = MultiEvidenceEmbedding(
            visual=np.ones(384, dtype=np.float32),
            texture=None,
            structural=np.ones(128, dtype=np.float32),
        )
        fused = emb.fused()
        self.assertEqual(fused.ndim, 1)

    def test_fused_all_none(self) -> None:
        emb = MultiEvidenceEmbedding()
        fused = emb.fused()
        np.testing.assert_array_equal(fused, np.zeros(640, dtype=np.float32))

    def test_nose_separate(self) -> None:
        emb = MultiEvidenceEmbedding(nose=np.ones(2048, dtype=np.float32))
        ns = emb.nose_separate()
        self.assertIsNotNone(ns)
        self.assertEqual(ns.shape, (2048,))

    def test_fused_with_plan_weights(self) -> None:
        emb = MultiEvidenceEmbedding(
            visual=np.ones(384, dtype=np.float32) * 2,
            texture=np.ones(128, dtype=np.float32),
            structural=None,
        )
        plan = FusePlan(visual_weight=1.0, texture_weight=0.0, structural_weight=0.0)
        fused = emb.fused(plan)
        self.assertAlmostEqual(float(fused[0]), float(fused[10]), places=5)
        norm = np.linalg.norm(fused)
        self.assertAlmostEqual(norm, 1.0, places=5)


class SuperAnimalKeypointsTests(unittest.TestCase):
    @classmethod
    def _make_sae(cls) -> SuperAnimalExtractor:
        sae = SuperAnimalExtractor.__new__(SuperAnimalExtractor)
        sae._num_keypoints = 17
        sae._dim = 256
        return sae

    def test_keypoints_embedding_shape(self) -> None:
        sae = self._make_sae()
        kpts = np.random.randn(34).astype(np.float32)
        emb = sae._keypoints_to_embedding(kpts)
        self.assertEqual(emb.ndim, 1)

    def test_keypoints_embedding_normalised(self) -> None:
        sae = self._make_sae()
        kpts = np.random.randn(34).astype(np.float32)
        emb = sae._keypoints_to_embedding(kpts)
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_keypoints_same_dog_similar(self) -> None:
        sae = self._make_sae()
        base = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                         0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                         0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                         0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
                         0.5, 0.6], dtype=np.float32)
        noisy = base + np.random.randn(34).astype(np.float32) * 0.01
        e1 = sae._keypoints_to_embedding(base)
        e2 = sae._keypoints_to_embedding(noisy)
        cos_sim = float(e1 @ e2)
        self.assertGreater(cos_sim, 0.85)

    def test_zero_keypoints_returns_zero_emb(self) -> None:
        sae = SuperAnimalExtractor.__new__(SuperAnimalExtractor)
        sae._num_keypoints = 2
        sae._dim = 256
        kpts = np.zeros(4, dtype=np.float32)
        emb = sae._keypoints_to_embedding(kpts)
        self.assertAlmostEqual(np.linalg.norm(emb), 0.0, places=5)

    def test_keypoints_geometry_length(self) -> None:
        n_kpts = 17
        n_dists = n_kpts * (n_kpts - 1) // 2
        n_angles = n_dists
        expected_geom = n_dists + n_angles  # = 272
        sae = SuperAnimalExtractor.__new__(SuperAnimalExtractor)
        sae._num_keypoints = n_kpts
        sae._dim = 256
        kpts = np.random.randn(n_kpts * 2).astype(np.float32)
        emb = sae._keypoints_to_embedding(kpts)
        self.assertEqual(len(emb), expected_geom)


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
    def test_dogfacenet_construction(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            self._write_dummy_onnx(tmp)
            ext = self.DogFaceNetExtractor(Path(tmp.name))
            self.assertEqual(ext.output_dim, 384)

    def test_convnext_construction(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            self._write_dummy_onnx(tmp)
            ext = self.ConvNeXtExtractor(Path(tmp.name))
            self.assertEqual(ext.output_dim, 768)

    def test_petreid_construction(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            self._write_dummy_onnx(tmp)
            ext = self.PetReIDExtractor(Path(tmp.name))
            self.assertEqual(ext.output_dim, 2048)

    def test_onnx_extractor_output_dim(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            self._write_dummy_onnx(tmp)
            ext = self.OnnxExtractor(Path(tmp.name), input_size=224, output_dim=640)
            self.assertEqual(ext.output_dim, 640)

    @staticmethod
    def _write_dummy_onnx(tmp) -> None:
        import numpy as np
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        w = numpy_helper.from_array(
            np.random.randn(640, 150528).astype(np.float32), "W"
        )
        nodes = [
            helper.make_node("Flatten", inputs=["x"], outputs=["flat"], axis=1),
            helper.make_node("Gemm", inputs=["flat", "W"], outputs=["y"],
                             alpha=1.0, beta=0.0, transA=0, transB=1),
        ]
        graph = helper.make_graph(
            nodes, "test",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 224, 224])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 640])],
            initializer=[w],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        onnx.save(model, tmp.name)
        tmp.flush()


@unittest.skipUnless(_require_onnx(), "onnx not available")
@unittest.skipUnless(_require_ort(), "onnxruntime not available")
class SearchEngineBackwardCompatTests(unittest.TestCase):
    def test_legacy_feature_extractor_construction(self) -> None:
        from cvi.search_engine import FeatureExtractor
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            self._write_dummy_onnx(tmp)
            ext = FeatureExtractor(Path(tmp.name))
            self.assertEqual(ext.output_dim, 640)

    @staticmethod
    def _write_dummy_onnx(tmp) -> None:
        import numpy as np
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        w = numpy_helper.from_array(
            np.random.randn(640, 150528).astype(np.float32), "W"
        )
        nodes = [
            helper.make_node("Flatten", inputs=["x"], outputs=["flat"], axis=1),
            helper.make_node("Gemm", inputs=["flat", "W"], outputs=["y"],
                             alpha=1.0, beta=0.0, transA=0, transB=1),
        ]
        graph = helper.make_graph(
            nodes, "test",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 224, 224])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 640])],
            initializer=[w],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        onnx.save(model, tmp.name)
        tmp.flush()
