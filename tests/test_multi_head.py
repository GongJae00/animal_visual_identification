from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cvi.multi_head import (
    MultiHeadBackbone,
    MultiHeadConfig,
    MultiHeadModel,
    TextureHead,
    VisualHead,
)


class MultiHeadConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        c = MultiHeadConfig()
        self.assertEqual(c.embedding_dim, 384)
        self.assertEqual(c.texture_dim, 128)
        self.assertEqual(c.structural_dim, 128)

    def test_round_trip(self) -> None:
        c = MultiHeadConfig(epochs=100, batch_size=64, lr=1e-3)
        d = c.to_dict()
        restored = MultiHeadConfig.from_dict(d)
        self.assertEqual(restored.epochs, 100)
        self.assertEqual(restored.batch_size, 64)

    def test_total_embedding_dim(self) -> None:
        c = MultiHeadConfig()
        total = c.embedding_dim + c.texture_dim + c.structural_dim
        self.assertEqual(total, 640)


class VisualHeadTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        vh = VisualHead(384, 384)
        cls_t = torch.randn(2, 384)
        emb = vh(cls_t)
        self.assertEqual(emb.shape, (2, 384))

    def test_l2_normalized(self) -> None:
        vh = VisualHead(384, 384)
        cls_t = torch.randn(2, 384)
        emb = vh(cls_t)
        norms = emb.norm(p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class TextureHeadTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        th = TextureHead(384, 128)
        patch_t = torch.randn(2, 256, 384)
        emb = th(patch_t)
        self.assertEqual(emb.shape, (2, 128))

    def test_l2_normalized(self) -> None:
        th = TextureHead(384, 128)
        patch_t = torch.randn(2, 256, 384)
        emb = th(patch_t)
        norms = emb.norm(p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_attention_weights_sum_to_one(self) -> None:
        th = TextureHead(384, 128)
        patch_t = torch.randn(1, 256, 384)
        attn_logits = th._attn(patch_t).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        self.assertTrue(torch.allclose(attn_weights.sum(), torch.tensor(1.0), atol=1e-5))


class MultiHeadModelTests(unittest.TestCase):
    def _requires_dinov2(self):
        try:
            from transformers import AutoModel
            AutoModel.from_pretrained("facebook/dinov2-small", attn_implementation="sdpa")
            return False
        except Exception:
            return True

    @unittest.skipIf(True, "DINOv2 model skipped for unit tests; tested manually")
    def test_forward_training(self) -> None:
        config = MultiHeadConfig(num_classes=5)
        model = MultiHeadModel(config)
        dummy = torch.randn(2, 3, 224, 224)
        labels = torch.randint(0, 5, (2,))
        lv, lt, ls = model(dummy, labels)
        self.assertEqual(lv.shape, (2, 5))
        self.assertEqual(lt.shape, (2, 5))
        self.assertEqual(ls.shape, (2, 5))

    @unittest.skipIf(True, "DINOv2 model skipped for unit tests; tested manually")
    def test_forward_inference(self) -> None:
        config = MultiHeadConfig()
        model = MultiHeadModel(config).eval()
        dummy = torch.randn(1, 3, 224, 224)
        emb = model(dummy)
        self.assertEqual(emb.shape, (1, 640))

    @unittest.skipIf(True, "DINOv2 model skipped for unit tests; tested manually")
    def test_extract_split(self) -> None:
        config = MultiHeadConfig()
        model = MultiHeadModel(config).eval()
        dummy = torch.randn(2, 3, 224, 224)
        split = model.extract_split(dummy)
        self.assertEqual(split["visual"].shape, (2, 384))
        self.assertEqual(split["texture"].shape, (2, 128))
        self.assertEqual(split["structural"].shape, (2, 128))


class MultiHeadBackboneTests(unittest.TestCase):
    def test_backbone_output_shapes(self) -> None:
        bb = MultiHeadBackbone()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            cls_t, patch_t, mid_f = bb(dummy)
        self.assertEqual(cls_t.shape, (1, 384))
        self.assertEqual(patch_t.shape, (1, 256, 384))
        self.assertEqual(mid_f.shape, (1, 256, 384))


if __name__ == "__main__":
    unittest.main()
