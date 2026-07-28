from __future__ import annotations

import unittest

from cvi.localization.quality import (
    compute_iou,
    detection_summary,
    greedy_bipartite_match,
    normalized_mean_error,
    pixel_correct_keypoint,
)
from cvi.localization.roi import (
    expand_bbox,
    face_roi_from_dog,
    is_truncated,
    square_padded_crop,
)
from cvi.localization.types import DetectionBox, Keypoint, KeypointSet, LocalizationResult


class DetectionMetricTests(unittest.TestCase):
    def test_iou_perfect_match(self) -> None:
        a = DetectionBox(0, 0, 100, 100, 1.0)
        b = DetectionBox(0, 0, 100, 100, 0.9)
        self.assertAlmostEqual(compute_iou(a, b), 1.0)

    def test_iou_no_overlap(self) -> None:
        a = DetectionBox(0, 0, 10, 10, 1.0)
        b = DetectionBox(20, 20, 30, 30, 1.0)
        self.assertAlmostEqual(compute_iou(a, b), 0.0)

    def test_greedy_matching_one_to_one(self) -> None:
        pred = [DetectionBox(0, 0, 10, 10, 1.0)]
        gt = [DetectionBox(0, 0, 10, 10, 1.0)]
        matches = greedy_bipartite_match(pred, gt)
        self.assertEqual(len(matches), 1)

    def test_detection_summary_aggregates_correctly(self) -> None:
        pred = [DetectionBox(0, 0, 10, 10, 1.0)]
        gt = [DetectionBox(0, 0, 10, 10, 1.0)]
        matches = greedy_bipartite_match(pred, gt)
        summary = detection_summary(matches, len(pred), len(gt))
        self.assertEqual(summary["false_positives"], 0)
        self.assertEqual(summary["false_negatives"], 0)
        self.assertAlmostEqual(summary["AP50_precision"], 1.0)
        self.assertAlmostEqual(summary["AP50_recall"], 1.0)

    def test_pck_is_head_size_normalized(self) -> None:
        pred_kp = Keypoint(5.0, 5.0, 1.0)
        gt_kp = Keypoint(0.0, 0.0, 1.0)
        self.assertTrue(pixel_correct_keypoint(pred_kp, gt_kp, head_size=100.0, threshold=0.10))
        self.assertFalse(pixel_correct_keypoint(pred_kp, gt_kp, head_size=10.0, threshold=0.10))


class RoITests(unittest.TestCase):
    def test_expand_bbox_obeys_image_boundaries(self) -> None:
        bbox = DetectionBox(45, 45, 55, 55, 1.0)
        x1, y1, x2, y2 = expand_bbox(bbox, scale=10.0, image_width=100, image_height=100)
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 0)
        self.assertEqual(x2, 100)
        self.assertEqual(y2, 100)

    def test_truncation_detection(self) -> None:
        bbox_outside = DetectionBox(-40, 0, 60, 100, 1.0)
        self.assertTrue(is_truncated(bbox_outside, image_width=100, image_height=100))
        bbox_inside = DetectionBox(10, 10, 90, 90, 1.0)
        self.assertFalse(is_truncated(bbox_inside, image_width=100, image_height=100))


class TypeValidationTests(unittest.TestCase):
    def test_detection_box_rejects_invalid_confidence(self) -> None:
        with self.assertRaises(ValueError):
            DetectionBox(0, 0, 10, 10, 1.5)
        with self.assertRaises(ValueError):
            DetectionBox(0, 0, 10, 10, -0.1)

    def test_detection_box_rejects_empty_area(self) -> None:
        with self.assertRaises(ValueError):
            DetectionBox(10, 10, 5, 5, 1.0)

    def test_localization_result_requires_nonempty_image_id(self) -> None:
        with self.assertRaises(ValueError):
            LocalizationResult(
                image_id="", dog_boxes=(), face_boxes=(), nose_boxes=(),
                body_keypoints=(), face_landmarks=(),
                model_name="test", model_family="test", inference_ms=0.0,
            )


if __name__ == "__main__":
    unittest.main()
