from __future__ import annotations

import unittest

import numpy as np
import torch

from data.types import CaptureGroupKind, UnifiedCanidSample
from identity_methods.face.dataset import FaceReIDDataset, align_face_rgb


class FaceDatasetTests(unittest.TestCase):
    def test_dataset_requires_nonempty_rows(self) -> None:
        with self.assertRaises((ValueError, TypeError)):
            FaceReIDDataset(
                root=None,
                rows=(),
                identity_to_index={},
            )

    def test_alignment_uses_eye_eye_nose_and_falls_back_on_low_confidence(self) -> None:
        image = np.zeros((224, 224, 3), dtype=np.float32)
        image[70:75, 65:70] = 1.0
        landmarks = torch.zeros((17, 3), dtype=torch.float32)
        landmarks[:3] = torch.tensor(
            ((0.30, 0.32, 0.9), (0.70, 0.32, 0.9), (0.50, 0.62, 0.9))
        )

        aligned, applied = align_face_rgb(image, landmarks)
        low_confidence = landmarks.clone()
        low_confidence[:, 2] = 0.0
        fallback, fallback_applied = align_face_rgb(image, low_confidence)

        self.assertTrue(applied)
        self.assertEqual(aligned.shape, (224, 224, 3))
        self.assertFalse(fallback_applied)
        self.assertTrue(np.array_equal(fallback, image))


class FaceConfigTests(unittest.TestCase):
    def test_train_config_rejects_invalid_pk(self) -> None:
        from identity_methods.face.config import FaceIDTrainConfig

        with self.assertRaises(ValueError):
            FaceIDTrainConfig(identities_per_batch=8, samples_per_identity=8)


if __name__ == "__main__":
    unittest.main()
