from __future__ import annotations

import unittest

import torch

from cvi.face_id.dataset import FaceReIDDataset
from cvi.canid_data.types import UnifiedCanidSample, CaptureGroupKind


class FaceDatasetTests(unittest.TestCase):
    def test_dataset_requires_nonempty_rows(self) -> None:
        with self.assertRaises((ValueError, TypeError)):
            FaceReIDDataset(
                root=None,
                rows=(),
                identity_to_index={},
            )


class FaceConfigTests(unittest.TestCase):
    def test_train_config_rejects_invalid_pk(self) -> None:
        from cvi.face_id.config import FaceIDTrainConfig

        with self.assertRaises(ValueError):
            FaceIDTrainConfig(identities_per_batch=8, samples_per_identity=8)


if __name__ == "__main__":
    unittest.main()
