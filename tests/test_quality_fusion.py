from __future__ import annotations

import unittest

import torch

from cvi.fusion.quality_fuser import QualityFusionMLP, fuse_channel_scores


class QualityFusionTests(unittest.TestCase):
    def test_unavailable_channel_is_renormalized(self) -> None:
        model = QualityFusionMLP().eval()
        features = torch.ones((1, 27))
        availability = torch.tensor([[True, False, True]])
        weights = model(features, availability)
        self.assertEqual(float(weights[0, 1]), 0.0)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))
        torch.testing.assert_close(weights[0, 0], torch.tensor(0.55 / 0.85))

    def test_score_fusion_ignores_unavailable_nose(self) -> None:
        scores = torch.tensor([[0.4, 0.1, 0.9]])
        weights = torch.tensor([[0.55, 0.15, 0.30]])
        availability = torch.tensor([[True, True, False]])
        fused = fuse_channel_scores(scores, weights, availability)
        expected = (0.55 * 0.4 + 0.15 * 0.1) / 0.70
        torch.testing.assert_close(fused, torch.tensor([expected]))


if __name__ == "__main__":
    unittest.main()
