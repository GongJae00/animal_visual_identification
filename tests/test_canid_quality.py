from __future__ import annotations

import unittest

from parsing.quality import (
    DogQuality,
    FaceQuality,
    NoseQuality,
    score_dog_quality,
    score_face_quality,
    score_nose_quality,
)
from parsing.types import DetectionBox, Keypoint, KeypointSet


class CanidQualityTests(unittest.TestCase):
    def test_dog_quality_to_list_length(self) -> None:
        bbox = DetectionBox(50, 50, 200, 200, 0.9)
        q = score_dog_quality(bbox, image_width=300, image_height=300)
        self.assertEqual(len(q.to_list()), 6)

    def test_face_quality_to_list_length(self) -> None:
        landmarks = KeypointSet(
            keypoints={
                "nose_center": Keypoint(50, 50, 1.0),
                "left_eye": Keypoint(30, 40, 1.0),
                "right_eye": Keypoint(70, 40, 1.0),
            },
            schema="face46",
        )
        q = score_face_quality(landmarks)
        self.assertEqual(len(q.to_list()), 6)

    def test_nose_quality_to_list_length(self) -> None:
        bbox = DetectionBox(100, 100, 200, 200, 0.8)
        q = score_nose_quality(bbox, native_short_side=150.0)
        self.assertEqual(len(q.to_list()), 7)


if __name__ == "__main__":
    unittest.main()
