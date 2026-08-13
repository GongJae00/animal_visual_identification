from __future__ import annotations

import unittest

import torch

from embedding.methods.nose.training.losses import (
    batch_hard_triplet_loss,
    supervised_contrastive_loss,
)


class NoseIDLossTests(unittest.TestCase):
    def test_cross_session_positives_produce_finite_gradients(self) -> None:
        embedding = torch.randn((8, 32), requires_grad=True)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        sessions = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = supervised_contrastive_loss(embedding, labels, sessions)
        loss = loss + batch_hard_triplet_loss(embedding, labels, sessions)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(embedding.grad).all())
        self.assertGreater(float(embedding.grad.abs().sum()), 0.0)

    def test_no_positive_anchor_is_a_differentiable_zero(self) -> None:
        embedding = torch.randn((4, 16), requires_grad=True)
        loss = supervised_contrastive_loss(embedding, torch.arange(4))
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(embedding.grad, torch.zeros_like(embedding)))


if __name__ == "__main__":
    unittest.main()
