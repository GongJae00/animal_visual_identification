from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from cvi.face_id.model import FaceIDModel, FaceRegionalEncoder
from cvi.face_id.losses import FaceIDObjective


class _DummyDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))

    def forward(self, *, pixel_values: torch.Tensor, output_hidden_states: bool):
        b = pixel_values.shape[0]
        base = pixel_values.mean(dim=(1, 2, 3)).view(b, 1, 1) + self.param
        return SimpleNamespace(
            hidden_states=tuple(base.expand(b, 257, 384) for _ in range(13))
        )


class FaceIDModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FaceIDModel(
            _DummyDino(),
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
            rescale_factor=1.0 / 255.0,
        ).eval()

    def test_output_shapes(self) -> None:
        rgb = torch.rand((2, 3, 224, 224))
        with torch.no_grad():
            output = self.model(rgb)
        self.assertEqual(output["embedding"].shape, (2, 256))
        self.assertEqual(output["quality"].shape, (2,))
        norm = torch.linalg.vector_norm(output["embedding"], dim=1)
        torch.testing.assert_close(norm, torch.ones(2), atol=1e-5, rtol=1e-5)

    def test_quality_in_range(self) -> None:
        rgb = torch.rand((4, 3, 224, 224))
        with torch.no_grad():
            output = self.model(rgb)
        self.assertTrue(torch.all(output["quality"] >= 0))
        self.assertTrue(torch.all(output["quality"] <= 1))


class FaceIDLossTests(unittest.TestCase):
    def test_all_components_finite(self) -> None:
        objective = FaceIDObjective(256, 16)
        emb = torch.randn((64, 256), requires_grad=True)
        labels = torch.arange(16).repeat_interleave(4)
        sessions = torch.tensor([0, 0, 1, 1] * 16)
        losses = objective(
            {"embedding": emb}, labels, sessions, margin_scale=0.5
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(torch.isfinite(emb.grad).all())


class FaceIDSamplerTests(unittest.TestCase):
    def test_p16_k4_batch(self) -> None:
        from cvi.face_id.sampler import FaceReIDSampler

        ids = [f"dog-{i}" for i in range(20) for _ in range(8)]
        sessions = [f"s{j % 2}" for j in range(160)]
        sampler = FaceReIDSampler(ids, sessions, seed=7)
        batch = next(iter(sampler))
        self.assertEqual(len(batch), 64)
        batch_ids = [ids[i] for i in batch]
        self.assertEqual(len(set(batch_ids)), 16)


if __name__ == "__main__":
    unittest.main()
