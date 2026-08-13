from __future__ import annotations

import unittest

import numpy as np

from embedding.methods.nose.signal.alignment import (
    AlignmentError,
    CANONICAL_KEYPOINTS,
    estimate_similarity_transform,
)


class NoseIDAlignmentTests(unittest.TestCase):
    def test_recovers_known_similarity_transform(self) -> None:
        target = CANONICAL_KEYPOINTS * 447.0
        angle = np.deg2rad(12.0)
        rotation = np.asarray(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
        )
        source = ((target - np.asarray((21.0, -13.0))) @ rotation) / 1.2
        matrix, residual = estimate_similarity_transform(
            np.concatenate([source, np.ones((6, 1))], axis=1)
        )
        recovered = np.concatenate([source, np.ones((6, 1))], axis=1) @ matrix.T
        np.testing.assert_allclose(recovered, target, atol=1e-3)
        self.assertLess(residual, 1e-5)

    def test_reflection_is_not_admitted(self) -> None:
        target = CANONICAL_KEYPOINTS * 447.0
        reflected = target.copy()
        reflected[:, 0] = 447.0 - reflected[:, 0]
        with self.assertRaises(AlignmentError):
            estimate_similarity_transform(
                np.concatenate([reflected, np.ones((6, 1))], axis=1)
            )


if __name__ == "__main__":
    unittest.main()
