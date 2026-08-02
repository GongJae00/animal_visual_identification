from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from identity_methods.face.dataset import RoiFaceReIDDataset
from localization.roi import (
    expand_bbox,
    face_roi_from_dog,
    normalize_source_point_to_square_crop,
    square_padded_crop,
    square_padded_crop_with_mask,
)
from localization.types import DetectionBox


class RoiGeometryTests(unittest.TestCase):
    def test_square_padded_crop_produces_target_size(self) -> None:
        image = Image.new("RGB", (100, 100), (128, 128, 128))
        bbox = DetectionBox(25, 25, 75, 75, 1.0)
        cropped = square_padded_crop(image, bbox, target_size=224)
        self.assertEqual(cropped.size, (224, 224))

    def test_square_crop_returns_source_valid_mask(self) -> None:
        image = Image.new("RGB", (100, 50), (128, 128, 128))
        bbox = DetectionBox(0, 0, 100, 50, 1.0)
        crop, mask, _ = square_padded_crop_with_mask(image, bbox, target_size=100)
        self.assertEqual(crop.size, (100, 100))
        self.assertEqual(mask.getpixel((50, 50)), 255)
        self.assertEqual(mask.getpixel((50, 5)), 0)

    def test_face_roi_from_dog_is_above_center(self) -> None:
        bbox = DetectionBox(0, 0, 100, 100, 1.0)
        face = face_roi_from_dog(bbox)
        self.assertLess(face.y1, bbox.y2 / 2.0)

    def test_expand_bbox_square_output(self) -> None:
        bbox = DetectionBox(40, 40, 60, 100, 1.0)
        x1, y1, x2, y2 = expand_bbox(bbox, scale=1.0, image_width=200, image_height=200)
        self.assertGreater(x2 - x1, 0)
        self.assertGreater(y2 - y1, 0)

    def test_landmark_transform_includes_square_padding(self) -> None:
        normalized = normalize_source_point_to_square_crop(30, 10, [0, 10, 60, 50])
        self.assertEqual(normalized, (0.5, 1.0 / 6.0))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (224, 224), (128, 128, 128)).save(root / "face.jpg")
            dataset = RoiFaceReIDDataset(
                root,
                (
                    {
                        "face_crop_path": "face.jpg",
                        "face_crop_rect_xyxy": [0, 10, 60, 50],
                        "body_keypoints": {"left_eye": [30, 10, 0.8]},
                        "registered_identity_id": "dog-1",
                        "face_quality": {"overall": 0.7},
                        "sample_id": "sample-1",
                        "capture_group_id": None,
                    },
                ),
                {"dog-1": 0},
            )
            landmark = dataset[0]["landmarks"][0]
            self.assertAlmostEqual(float(landmark[0]), 0.5)
            self.assertAlmostEqual(float(landmark[1]), 1.0 / 6.0)
            self.assertAlmostEqual(float(landmark[2]), 0.8)


if __name__ == "__main__":
    unittest.main()
