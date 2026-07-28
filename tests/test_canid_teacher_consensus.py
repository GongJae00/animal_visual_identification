from __future__ import annotations

import unittest

from cvi.localization.consensus import (
    compute_error_correlation,
    consensus_admission,
    consensus_dog_bbox,
    consensus_keypoint,
    robust_weighted_keypoint,
    weighted_box_fusion,
)
from cvi.localization.types import DetectionBox, Keypoint, KeypointSet, LocalizationResult


class ConsensusTests(unittest.TestCase):
    def test_weighted_box_fusion_produces_midpoint(self) -> None:
        a = DetectionBox(0, 0, 10, 10, 0.5)
        b = DetectionBox(10, 10, 20, 20, 0.5)
        fused = weighted_box_fusion([a, b])
        self.assertAlmostEqual(fused.x1, 5.0)
        self.assertAlmostEqual(fused.y1, 5.0)
        self.assertAlmostEqual(fused.x2, 15.0)
        self.assertAlmostEqual(fused.y2, 15.0)

    def test_weighted_box_fusion_confidence_weighted(self) -> None:
        a = DetectionBox(0, 0, 10, 10, 1.0)
        b = DetectionBox(100, 100, 110, 110, 0.0)
        fused = weighted_box_fusion([a, b])
        self.assertAlmostEqual(fused.x1, 0.0, delta=1.0)
        self.assertAlmostEqual(fused.y1, 0.0, delta=1.0)

    def test_robust_keypoint_weights_converge(self) -> None:
        points = [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (99.0, 99.0, 0.1)]
        x, y, conf = robust_weighted_keypoint(points)
        self.assertLess(abs(x), 10.0)
        self.assertLess(abs(y), 10.0)
        self.assertGreater(conf, 0.0)

    def test_consensus_admission_gates_correctly(self) -> None:
        self.assertEqual(consensus_admission(0.90, 0.90), "ACCEPT")
        self.assertEqual(consensus_admission(0.30, 0.90), "REJECT")
        self.assertEqual(consensus_admission(0.60, 0.60), "REVIEW")

    def test_consensus_dog_bbox_returns_fused_box(self) -> None:
        r1 = LocalizationResult(
            image_id="img1", dog_boxes=(DetectionBox(0, 0, 10, 10, 0.9),),
            face_boxes=(), nose_boxes=(), body_keypoints=(), face_landmarks=(),
            model_name="m1", model_family="f1", inference_ms=0.0,
        )
        r2 = LocalizationResult(
            image_id="img1", dog_boxes=(DetectionBox(0, 0, 10, 10, 0.9),),
            face_boxes=(), nose_boxes=(), body_keypoints=(), face_landmarks=(),
            model_name="m2", model_family="f2", inference_ms=0.0,
        )
        bbox, info = consensus_dog_bbox((r1, r2))
        self.assertIsNotNone(bbox)
        self.assertEqual(info["status"], "FUSED")

    def test_consensus_dog_bbox_no_detections(self) -> None:
        r1 = LocalizationResult(
            image_id="img1", dog_boxes=(),
            face_boxes=(), nose_boxes=(), body_keypoints=(), face_landmarks=(),
            model_name="m1", model_family="f1", inference_ms=0.0,
        )
        bbox, info = consensus_dog_bbox((r1,))
        self.assertIsNone(bbox)
        self.assertEqual(info["status"], "NO_DOG_DETECTED")

    def test_error_correlation_matrix_shape(self) -> None:
        results = {
            "m1": (LocalizationResult(
                image_id="img1", dog_boxes=(DetectionBox(0, 0, 10, 10, 0.9),),
                face_boxes=(), nose_boxes=(), body_keypoints=(), face_landmarks=(),
                model_name="m1", model_family="f1", inference_ms=0.0,
            ),),
            "m2": (LocalizationResult(
                image_id="img1", dog_boxes=(),
                face_boxes=(), nose_boxes=(), body_keypoints=(), face_landmarks=(),
                model_name="m2", model_family="f2", inference_ms=0.0,
            ),),
        }
        corr = compute_error_correlation(results)
        self.assertEqual(corr["models"], ["m1", "m2"])
        self.assertEqual(corr["correlation"].shape, (2, 2))


class QualityScoringTests(unittest.TestCase):
    def test_dog_quality_is_normalized(self) -> None:
        from cvi.localization.quality import DogQuality, score_dog_quality

        bbox = DetectionBox(50, 50, 200, 200, 0.9)
        quality = score_dog_quality(bbox, image_width=300, image_height=300)
        self.assertGreaterEqual(quality.overall, 0.0)
        self.assertLessEqual(quality.overall, 1.0)

    def test_face_quality_yaw_proxy(self) -> None:
        from cvi.localization.quality import score_face_quality

        landmarks = KeypointSet(
            keypoints={
                "nose_center": Keypoint(50, 60, 1.0),
                "left_eye": Keypoint(30, 40, 1.0),
                "right_eye": Keypoint(70, 40, 1.0),
            },
            schema="face46",
        )
        quality = score_face_quality(landmarks)
        self.assertGreater(quality.yaw_roll_proxy, 0.0)
        self.assertLessEqual(quality.yaw_roll_proxy, 1.0)

    def test_nose_quality_all_inputs_mapped(self) -> None:
        from cvi.localization.quality import score_nose_quality

        bbox = DetectionBox(100, 100, 200, 200, 0.8, class_name="nose")
        quality = score_nose_quality(
            bbox,
            native_short_side=150.0,
            specular_ratio=0.1,
            image_width=300,
            image_height=300,
        )
        self.assertGreaterEqual(quality.overall, 0.0)
        self.assertLessEqual(quality.overall, 1.0)
        self.assertEqual(len(quality.to_list()), 7)


if __name__ == "__main__":
    unittest.main()
