from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from embedding.methods.face.losses import (
    FaceIDObjective,
    FaceResidualObjective,
    objective_anchor_coverage,
)
from embedding.methods.face.model import FaceIDModel
from embedding.methods.face.residual_model import FaceIDResidualModel
from legacy.version.face.experiments.face_evaluation import paired_face_retrieval_comparison


class _DummyDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))

    def forward(
        self, *, pixel_values: torch.Tensor, output_hidden_states: bool = False
    ):
        b = pixel_values.shape[0]
        base = pixel_values.mean(dim=(1, 2, 3)).view(b, 1, 1) + self.param
        return SimpleNamespace(
            hidden_states=tuple(base.expand(b, 257, 384) for _ in range(13)),
            pooler_output=base.expand(b, 1, 384).squeeze(1),
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
        self.assertEqual(output["embedding"].shape, (2, 640))
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

    def test_objective_coverage_distinguishes_same_and_cross_session_pairs(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        same_sessions = torch.tensor([0, 0, 1, 1])
        cross_sessions = torch.tensor([0, 1, 2, 3])

        weak = objective_anchor_coverage(
            labels, same_sessions, second_view_available=False
        )
        strong = objective_anchor_coverage(
            labels, cross_sessions, second_view_available=True
        )

        self.assertEqual(float(weak["supcon_valid_anchor_fraction"]), 1.0)
        self.assertEqual(float(weak["cross_session_triplet_valid_anchor_fraction"]), 0.0)
        self.assertEqual(float(weak["second_view_coverage"]), 0.0)
        self.assertEqual(float(strong["cross_session_triplet_valid_anchor_fraction"]), 1.0)
        self.assertEqual(float(strong["second_view_coverage"]), 1.0)

    def test_f5_objective_activates_paired_losses_without_triplet(self) -> None:
        objective = FaceResidualObjective(384, 4)
        embedding = torch.nn.functional.normalize(torch.randn(8, 384), dim=1)
        second = torch.nn.functional.normalize(
            embedding + 0.01 * torch.randn_like(embedding), dim=1
        )
        labels = torch.arange(4).repeat_interleave(2)
        sessions = torch.arange(4).repeat_interleave(2)
        losses = objective(
            {
                "embedding": embedding,
                "baseline_embedding": embedding.detach(),
                "quality": torch.full((8,), 0.5),
            },
            labels,
            sessions,
            second_view_embedding=second,
            quality_target=torch.full((8,), 0.7),
        )

        self.assertGreater(float(losses["supervised_contrastive"]), 0.0)
        self.assertGreater(float(losses["view_consistency"]), 0.0)
        self.assertEqual(float(losses["batch_hard_triplet"]), 0.0)


class FaceIDResidualModelTests(unittest.TestCase):
    def test_zero_initialized_residual_starts_at_frozen_baseline(self) -> None:
        model = FaceIDResidualModel(
            _DummyDino(),
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
            rescale_factor=1.0 / 255.0,
        ).eval()
        with torch.no_grad():
            output = model(torch.rand((2, 3, 224, 224)))

        torch.testing.assert_close(
            output["embedding"],
            output["baseline_embedding"],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_residual_adapter_has_a_hard_norm_bound(self) -> None:
        model = FaceIDResidualModel(
            _DummyDino(),
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
            rescale_factor=1.0 / 255.0,
        )
        with torch.no_grad():
            model.encoder.network[-1].weight.fill_(100.0)
            model.encoder.network[-1].bias.fill_(100.0)
            output = model(torch.rand((4, 3, 224, 224)))

        cosine = torch.nn.functional.cosine_similarity(
            output["embedding"], output["baseline_embedding"], dim=1
        )
        self.assertTrue(torch.all(cosine > 0.99))


class FacePairedComparisonTests(unittest.TestCase):
    def test_exact_query_paired_delta(self) -> None:
        gallery = torch.eye(3).numpy()
        baseline_query = torch.tensor(
            ((0.1, 0.9, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ).numpy()
        candidate_query = gallery.copy()
        identities = torch.tensor((0, 1, 2)).numpy()

        report = paired_face_retrieval_comparison(
            baseline_query_embeddings=baseline_query,
            baseline_gallery_embeddings=gallery,
            candidate_query_embeddings=candidate_query,
            candidate_gallery_embeddings=gallery,
            query_identity_ids=identities,
            gallery_identity_ids=identities,
            resamples=100,
        )

        self.assertEqual(report["baseline_metrics"]["Rank-1"], 2 / 3)
        self.assertEqual(report["candidate_metrics"]["Rank-1"], 1.0)
        self.assertAlmostEqual(
            report["delta_bootstrap_cis"]["Rank-1"]["estimate"], 1 / 3
        )


class FaceIDSamplerTests(unittest.TestCase):
    def test_p16_k4_batch(self) -> None:
        from embedding.methods.face.sampler import FaceReIDSampler

        ids = [f"dog-{i}" for i in range(20) for _ in range(8)]
        sessions = [f"s{j % 2}" for j in range(160)]
        sampler = FaceReIDSampler(ids, sessions, seed=7)
        batch = next(iter(sampler))
        self.assertEqual(len(batch), 64)
        batch_ids = [ids[i] for i in batch]
        self.assertEqual(len(set(batch_ids)), 16)
        with self.assertRaises(ValueError):
            FaceReIDSampler(
                ids,
                sessions,
                identities_per_batch=8,
                samples_per_identity=8,
            )


if __name__ == "__main__":
    unittest.main()
