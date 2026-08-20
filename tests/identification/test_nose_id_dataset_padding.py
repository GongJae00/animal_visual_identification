from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import tempfile
import unittest
import uuid

import numpy as np
from PIL import Image

from identification.export.nose.signal.alignment import CANONICAL_KEYPOINTS
from identification.export.nose.data.dataset import NoseIDDataset, NoseIDSample

def _write_image(path: Path, value: np.ndarray) -> str:
    Image.fromarray(value).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()

class NoseIDDatasetPaddingTests(unittest.TestCase):
    def test_reflected_rgb_padding_is_invalid_context(self) -> None:
        identity = str(uuid.uuid5(uuid.NAMESPACE_DNS, "padding-dog"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_sha = _write_image(
                root / "image.png", np.full((448, 448, 3), 128, dtype=np.uint8)
            )
            semantic_sha = _write_image(
                root / "semantic.png", np.ones((448, 448), dtype=np.uint8)
            )
            invalid_sha = _write_image(
                root / "invalid.png", np.zeros((448, 448), dtype=np.uint8)
            )
            keypoints = CANONICAL_KEYPOINTS.astype(np.float32) * 447.0 + 100.0
            row = NoseIDSample(
                sample_id="padding-sample",
                image_path=PurePosixPath("image.png"),
                image_sha256=image_sha,
                image_width=448,
                image_height=448,
                registered_dog_id=identity,
                session_id="session-a",
                camera_id="camera-a",
                video_id="video-a",
                frame_index=0,
                timestamp_ms=0,
                nose_bbox_xyxy=(200.0, 200.0, 400.0, 400.0),
                keypoints_xy=keypoints,
                keypoint_visibility=np.full(6, 2, dtype=np.int64),
                semantic_mask_path=PurePosixPath("semantic.png"),
                semantic_mask_sha256=semantic_sha,
                semantic_mask_box_xyxy=(0.0, 0.0, 448.0, 448.0),
                invalid_mask_path=PurePosixPath("invalid.png"),
                invalid_mask_sha256=invalid_sha,
                invalid_mask_box_xyxy=(0.0, 0.0, 448.0, 448.0),
                split_role="TRAIN",
            )
            split = {
                "TRAIN": frozenset((identity,)),
                "DEV": frozenset(),
                "FUSION_CAL": frozenset(),
                "TEST": frozenset(),
            }
            dataset = NoseIDDataset(
                root,
                (row,),
                {identity: 0},
                identity_split=split,
                split_role="TRAIN",
            )
            result = dataset[0]
            padded = result["source_valid_mask"][0] == 0
            self.assertTrue(padded.any())
            self.assertTrue(result["invalid_mask"][0][padded].all())
            self.assertTrue((result["semantic_mask"][padded] == 0).all())
            self.assertTrue(np.isfinite(result["aligned_rgb"].numpy()).all())

if __name__ == "__main__":
    unittest.main()
