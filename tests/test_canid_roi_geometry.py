from __future__ import annotations

import unittest

from cvi.localization.roi import expand_bbox, face_roi_from_dog, square_padded_crop

from PIL import Image

from cvi.localization.types import DetectionBox


class RoiGeometryTests(unittest.TestCase):
    def test_square_padded_crop_produces_target_size(self) -> None:
        image = Image.new("RGB", (100, 100), (128, 128, 128))
        bbox = DetectionBox(25, 25, 75, 75, 1.0)
        cropped = square_padded_crop(image, bbox, target_size=224)
        self.assertEqual(cropped.size, (224, 224))

    def test_face_roi_from_dog_is_above_center(self) -> None:
        bbox = DetectionBox(0, 0, 100, 100, 1.0)
        face = face_roi_from_dog(bbox)
        self.assertLess(face.y1, bbox.y2 / 2.0)

    def test_expand_bbox_square_output(self) -> None:
        bbox = DetectionBox(40, 40, 60, 100, 1.0)
        x1, y1, x2, y2 = expand_bbox(
            bbox, scale=1.0, image_width=200, image_height=200
        )
        self.assertGreater(x2 - x1, 0)
        self.assertGreater(y2 - y1, 0)


if __name__ == "__main__":
    unittest.main()
