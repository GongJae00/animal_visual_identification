from __future__ import annotations

import unittest

import torch

from identification.training.nose.augment import NoseIdentityAugment

class NoseIDAugmentationTargetTests(unittest.TestCase):
    def test_views_align_masks_keypoints_and_emit_bounded_targets(self) -> None:
        augment = NoseIdentityAugment(seed=17)
        rgb = torch.full((3, 448, 448), 0.4)
        keypoints = torch.tensor(
            [
                [152.0, 224.0, 1.0], [295.0, 224.0, 1.0],
                [224.0, 107.0, 1.0], [224.0, 340.0, 1.0],
                [80.0, 224.0, 1.0], [367.0, 224.0, 1.0],
            ]
        )
        semantic = torch.zeros((3, 448, 448))
        semantic[0] = 1.0
        semantic[0, 96:352, 96:352] = 0.0
        semantic[1, 96:352, 96:352] = 1.0
        invalid = torch.zeros((1, 448, 448))
        source_valid = torch.ones((1, 448, 448))
        observed_direct_invalid = False
        for _ in range(24):
            view = augment(rgb, keypoints, semantic, invalid, source_valid)
            self.assertEqual(view.degradation_target.shape, (6,))
            self.assertTrue(torch.isfinite(view.degradation_target).all())
            self.assertTrue(((view.degradation_target >= 0) & (view.degradation_target <= 1)).all())
            torch.testing.assert_close(
                view.semantic_probability.sum(dim=0),
                torch.ones((448, 448)),
                atol=1e-5,
                rtol=1e-5,
            )
            outside = view.source_valid_probability < 1e-5
            self.assertTrue((view.invalid_probability[outside] > 0.99).all())
            if view.degradation_target[2] > 0 or view.degradation_target[3] > 0:
                observed_direct_invalid = True
                self.assertGreater(float(view.invalid_probability.sum()), 0.0)
        self.assertTrue(observed_direct_invalid)

if __name__ == "__main__":
    unittest.main()
