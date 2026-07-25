from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cvi.trainer import (
    ArcFaceHead,
    ArcFaceModel,
    ConvNeXtEmbedding,
    Dinov2Embedding,
    IdentityBalancedSampler,
    TrainConfig,
    _build_label_index,
    _count_parameters,
    _warmup_cosine_schedule,
)


class TrainConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        c = TrainConfig()
        self.assertEqual(c.embedding_dim, 384)
        self.assertEqual(c.arcface_margin, 0.50)
        self.assertIsInstance(c.to_dict(), dict)

    def test_round_trip(self) -> None:
        c = TrainConfig(epochs=100, batch_size=128, lr=1e-3)
        d = c.to_dict()
        restored = TrainConfig.from_dict(d)
        self.assertEqual(restored.epochs, 100)
        self.assertEqual(restored.batch_size, 128)
        self.assertEqual(restored.lr, 1e-3)


class LabelIndexTests(unittest.TestCase):
    def test_build_label_index(self) -> None:
        bindings = [
            {"registered_dog_id": "uuid-a", "identity_token": "a"},
            {"registered_dog_id": "uuid-b", "identity_token": "b"},
            {"registered_dog_id": "uuid-a", "identity_token": "c"},
        ]
        index = _build_label_index(bindings)
        self.assertEqual(len(index), 2)
        self.assertEqual(index["uuid-a"], 0)
        self.assertEqual(index["uuid-b"], 1)

    def test_sorted_order(self) -> None:
        bindings = [
            {"registered_dog_id": "z-id", "identity_token": "z"},
            {"registered_dog_id": "a-id", "identity_token": "a"},
        ]
        index = _build_label_index(bindings)
        self.assertEqual(index["a-id"], 0)
        self.assertEqual(index["z-id"], 1)


class WarmupCosineScheduleTests(unittest.TestCase):
    def test_warmup_linear(self) -> None:
        lr = _warmup_cosine_schedule(0, 50, 5, 1e-3, 1e-6)
        self.assertAlmostEqual(lr, 1e-3 / 5)

    def test_warmup_peak(self) -> None:
        lr = _warmup_cosine_schedule(4, 50, 5, 1e-3, 1e-6)
        self.assertAlmostEqual(lr, 1e-3, places=6)

    def test_cosine_decay_below_max(self) -> None:
        lr = _warmup_cosine_schedule(30, 50, 5, 1e-3, 1e-6)
        self.assertLess(lr, 1e-3)
        self.assertGreaterEqual(lr, 1e-6)

    def test_final_lr_near_min(self) -> None:
        lr = _warmup_cosine_schedule(49, 50, 5, 1e-3, 1e-6)
        self.assertLessEqual(lr, 1e-3)
        self.assertGreaterEqual(lr, 1e-7)


class IdentityBalancedSamplerTests(unittest.TestCase):
    def test_batch_structure(self) -> None:
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        sampler = IdentityBalancedSampler(
            labels, batch_size=3,
            generator=torch.Generator().manual_seed(42),
        )
        batches = list(sampler)
        self.assertTrue(len(batches) > 0)
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            self.assertTrue(all(isinstance(i, int) for i in batch))

    def test_identity_coverage_on_small_data(self) -> None:
        labels = [0, 1]
        sampler = IdentityBalancedSampler(labels, batch_size=2)
        batches = list(sampler)
        all_indices = set()
        for batch in batches:
            all_indices.update(batch)
        self.assertEqual(all_indices, {0, 1})


class CountParametersTests(unittest.TestCase):
    def test_arcface_head_count(self) -> None:
        head = ArcFaceHead(384, 100, 30.0, 0.5)
        counts = _count_parameters(head)
        self.assertIn("total", counts)
        self.assertIn("trainable", counts)
        self.assertGreater(counts["total"], 0)
        self.assertEqual(counts["total"], counts["trainable"])


class ArcFaceHeadForwardTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        import torch
        head = ArcFaceHead(64, 10, 30.0, 0.5)
        features = torch.randn(4, 64)
        features = features / features.norm(dim=1, keepdim=True)
        labels = torch.tensor([0, 1, 2, 3])
        logits = head(features, labels)
        self.assertEqual(logits.shape, (4, 10))

    def test_gradient_flows_through_arcface(self) -> None:
        head = ArcFaceHead(64, 10, 30.0, 0.5)
        features = torch.randn(4, 64)
        features = features / features.norm(dim=1, keepdim=True)
        labels = torch.tensor([0, 1, 2, 3])
        logits = head(features, labels)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        self.assertIsNotNone(head._w.grad)
        self.assertTrue(torch.isfinite(head._w.grad).all())


def _requires_dinov2():
    try:
        from transformers import AutoModel
        AutoModel.from_pretrained("facebook/dinov2-small", attn_implementation="sdpa")
        return False
    except Exception:
        return True


class Dinov2EmbeddingTests(unittest.TestCase):
    @unittest.skipIf(_requires_dinov2(), "DINOv2 model or transformers not available")
    def test_forward_shape_with_dummy_backbone(self) -> None:
        import torch
        model = Dinov2Embedding(embedding_dim=384)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            emb = model(dummy)
        self.assertEqual(emb.shape, (1, 384))

    @unittest.skipIf(_requires_dinov2(), "DINOv2 model or transformers not available")
    def test_l2_normalized(self) -> None:
        import torch
        model = Dinov2Embedding(embedding_dim=384)
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            emb = model(dummy)
        norms = emb.norm(p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class ConvNeXtEmbeddingTests(unittest.TestCase):
    def test_output_dim(self) -> None:
        model = ConvNeXtEmbedding(embedding_dim=768)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            emb = model(dummy)
        self.assertEqual(emb.shape, (1, 768))

    def test_l2_normalized(self) -> None:
        model = ConvNeXtEmbedding(embedding_dim=768)
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            emb = model(dummy)
        norms = emb.norm(p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_projection_dim(self) -> None:
        model = ConvNeXtEmbedding(embedding_dim=256)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            emb = model(dummy)
        self.assertEqual(emb.shape, (1, 256))


class _DummyBackbone(torch.nn.Module):
    def __init__(self, embedding_dim: int = 384,
                 use_gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(224 * 224 * 3, embedding_dim)
        self._embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.reshape(b, -1)
        return torch.nn.functional.normalize(self._linear(flat), p=2, dim=1)


class ArcFaceModelTests(unittest.TestCase):
    def test_forward_training(self) -> None:
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        model.train()
        logits = model(images, labels)
        self.assertEqual(logits.shape, (4, 5))

    def test_forward_eval(self) -> None:
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        model.eval()
        images = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            emb = model(images)
        self.assertEqual(emb.shape, (4, 64))

    def test_extract_embedding(self) -> None:
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        images = torch.randn(2, 3, 224, 224)
        emb_np = model.extract_embedding(images)
        self.assertEqual(emb_np.shape, (2, 64))

    def test_export_to_onnx(self) -> None:
        import tempfile
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            path = Path(f.name)
            model.export_to_onnx(path)
            self.assertTrue(path.exists())

    def test_dinov2_compat_subclass(self) -> None:
        from cvi.trainer import Dinov2ArcFaceModel
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = Dinov2ArcFaceModel(cfg)
        self.assertIsInstance(model, ArcFaceModel)

    def test_gradient_flow(self) -> None:
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        model.train()
        logits = model(images, labels)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        self.assertTrue(has_grad)


class ConvNeXtOnnxExportTests(unittest.TestCase):
    def test_export_and_reload(self) -> None:
        import tempfile
        import numpy as np
        model = ConvNeXtEmbedding(embedding_dim=768)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            emb_before = model(dummy).numpy()
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            path = Path(f.name)
            model.eval()
            torch.onnx.export(
                model, dummy, str(path),
                input_names=["images"], output_names=["embedding"],
                dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
                opset_version=18,
            )
            from cvi.evidence_extractor import OnnxExtractor
            ext = OnnxExtractor(path, input_size=224, output_dim=768,
                                provider="CPUExecutionProvider")
            from PIL import Image
            img = Image.fromarray(
                (dummy[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            emb_after = ext.extract(img)
        self.assertEqual(emb_after.shape, (768,))
        diff = float(abs(np.linalg.norm(emb_after) - 1.0))
        self.assertLess(diff, 1e-4)


if __name__ == "__main__":
    import torch
    unittest.main()
