from __future__ import annotations

import unittest
from pathlib import Path

from evaluation.localization import ap10k_body17_pose_summary
from evaluation.localization_metrics import (
    detection_average_precision,
    detection_summary,
    greedy_bipartite_match,
    pixel_correct_keypoint,
)
from parsing.roi import (
    compute_iou,
    expand_bbox,
    is_truncated,
)
from parsing.types import (
    AP10K_BODY_17_SCHEMA,
    DetectionBox,
    Keypoint,
    KeypointSet,
    LocalizationResult,
)
from workflows.benchmark_canid_localizers import _build_summary


def _point_set(
    points: dict[str, tuple[float, float, float]],
    *,
    schema: str = AP10K_BODY_17_SCHEMA,
) -> KeypointSet:
    return KeypointSet(
        {name: Keypoint(*coordinates) for name, coordinates in points.items()}, schema
    )


def _pose_result(
    boxes: tuple[DetectionBox, ...], point_sets: tuple[KeypointSet, ...]
) -> LocalizationResult:
    return LocalizationResult(
        image_id="image",
        dog_boxes=boxes,
        face_boxes=(),
        nose_boxes=(),
        body_keypoints=point_sets,
        face_landmarks=(),
        model_name="pose",
        model_family="fixture",
        inference_ms=1.0,
    )


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

    def test_detection_average_precision_ranks_confidence(self) -> None:
        ground_truth = {"image": [DetectionBox(0, 0, 10, 10, 1.0)]}
        predictions = {
            "image": [
                DetectionBox(20, 20, 30, 30, 0.1),
                DetectionBox(0, 0, 10, 10, 0.9),
            ]
        }
        result = detection_average_precision(
            predictions, ground_truth, iou_threshold=0.5
        )
        self.assertAlmostEqual(result["AP"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_pck_is_head_size_normalized(self) -> None:
        pred_kp = Keypoint(5.0, 5.0, 1.0)
        gt_kp = Keypoint(0.0, 0.0, 1.0)
        self.assertTrue(
            pixel_correct_keypoint(pred_kp, gt_kp, head_size=100.0, threshold=0.10)
        )
        self.assertFalse(
            pixel_correct_keypoint(pred_kp, gt_kp, head_size=10.0, threshold=0.10)
        )


class AP10KPoseMetricTests(unittest.TestCase):
    def test_confidence_ordered_multi_instance_metrics_and_denominators(self) -> None:
        gt_boxes = {
            "image": [
                DetectionBox(0, 0, 30, 40, 1.0),
                DetectionBox(100, 0, 130, 40, 1.0),
            ]
        }
        gt_points = {
            "image": [
                _point_set(
                    {
                        "nose_center": (10, 10, 1.0),
                        "left_eye": (15, 10, 0.5),
                        "neck": (15, 20, 1.0),
                        "tail_base": (25, 20, 0.0),
                    }
                ),
                _point_set({"nose_center": (110, 10, 1.0)}),
            ]
        }
        result = _pose_result(
            (
                DetectionBox(0, 0, 30, 40, 0.8),
                DetectionBox(100, 0, 130, 40, 0.7),
                DetectionBox(0, 0, 24, 40, 0.9),
            ),
            (
                _point_set({"nose_center": (10, 10, 1.0)}),
                _point_set({"nose_center": (110, 10, 1.0)}),
                _point_set(
                    {
                        "nose_center": (15, 10, 1.0),
                        "left_eye": (15, 10, 0.49),
                    }
                ),
            ),
        )

        summary = ap10k_body17_pose_summary((result,), gt_boxes, gt_points)

        self.assertEqual(summary["matched_instances"], 2)
        self.assertEqual(summary["predicted_instances"], 3)
        self.assertEqual(summary["ground_truth_instances"], 2)
        self.assertEqual(summary["visible_ground_truth_keypoints"], 4)
        self.assertEqual(summary["total_visible_ground_truth_keypoints"], 4)
        self.assertEqual(summary["valid_predicted_keypoints"], 2)
        self.assertEqual(summary["missing_or_low_confidence_keypoints"], 2)
        self.assertEqual(summary["nme_denominator"], 2)
        self.assertEqual(summary["pck_denominator"], 4)
        self.assertAlmostEqual(summary["NME"], 0.05)
        self.assertAlmostEqual(summary["PCK@0.05"], 0.25)
        self.assertAlmostEqual(summary["PCK@0.10"], 0.5)
        self.assertAlmostEqual(summary["end_to_end_PCK@0.10"], 0.5)

    def test_no_instance_matches_reports_null_keypoint_metrics(self) -> None:
        result = _pose_result(
            (DetectionBox(20, 20, 30, 30, 0.9),),
            (_point_set({"nose_center": (25, 25, 1.0)}),),
        )
        summary = ap10k_body17_pose_summary(
            (result,),
            {"image": [DetectionBox(0, 0, 10, 10, 1.0)]},
            {"image": [_point_set({"nose_center": (5, 5, 1.0)})]},
        )

        self.assertEqual(summary["matched_instances"], 0)
        self.assertEqual(summary["pck_denominator"], 0)
        self.assertIsNone(summary["NME"])
        self.assertIsNone(summary["PCK@0.05"])
        self.assertIsNone(summary["PCK@0.10"])
        self.assertEqual(summary["end_to_end_PCK@0.05"], 0.0)
        self.assertEqual(summary["end_to_end_PCK@0.10"], 0.0)

    def test_malformed_alignment_schema_and_geometry_fail_closed(self) -> None:
        box = DetectionBox(0, 0, 10, 10, 1.0)
        result = _pose_result(
            (box,), (_point_set({"nose_center": (5, 5, 1.0)}),)
        )
        with self.assertRaisesRegex(ValueError, "ground-truth keypoint sets"):
            ap10k_body17_pose_summary((result,), {"image": [box]}, {"image": []})

        wrong_schema = _point_set(
            {"nose_center": (5, 5, 1.0)}, schema="not-ap10k"
        )
        with self.assertRaisesRegex(ValueError, "schema must be"):
            ap10k_body17_pose_summary(
                (result,), {"image": [box]}, {"image": [wrong_schema]}
            )

        invalid_box = DetectionBox(0, 0, 10, 10, 1.0)
        object.__setattr__(invalid_box, "x2", 0.0)
        with self.assertRaisesRegex(ValueError, "geometry"):
            ap10k_body17_pose_summary(
                (result,),
                {"image": [invalid_box]},
                {"image": [_point_set({"nose_center": (5, 5, 1.0)})]},
            )

        misaligned_result = _pose_result(
            (box, DetectionBox(20, 20, 30, 30, 0.5)),
            (_point_set({"nose_center": (5, 5, 1.0)}),),
        )
        with self.assertRaisesRegex(ValueError, "align with dog boxes"):
            ap10k_body17_pose_summary(
                (misaligned_result,), {"image": [box]}, {"image": [None]}
            )

    def test_cli_summary_schema_adds_pose_only_for_ap10k(self) -> None:
        box = DetectionBox(0, 0, 10, 10, 1.0)
        points = _point_set({"nose_center": (5, 5, 1.0)})
        result = _pose_result((box,), (points,))
        common = {
            "split_role": "test",
            "results": [result],
            "prediction_cache": Path("predictions.json"),
            "prediction_cache_sha256": "a" * 64,
        }

        ap10k = _build_summary(
            dataset="ap10k-dog",
            ground_truth={"image": [box]},
            ground_truth_keypoints={"image": [points]},
            **common,
        )
        other = _build_summary(
            dataset="dogflw",
            ground_truth=None,
            ground_truth_keypoints=None,
            **common,
        )

        self.assertEqual(
            ap10k["schema_version"], "cvi.canid_localizer_benchmark_summary.v1"
        )
        self.assertEqual(ap10k["detection"]["AP50"]["AP"], 1.0)
        self.assertEqual(
            ap10k["pose"]["schema_version"],
            "cvi.ap10k_body17_pose_evaluation.v1",
        )
        self.assertIn("Custom metric", ap10k["pose"]["metric_note"])
        self.assertNotIn("pose", other)


class RoITests(unittest.TestCase):
    def test_expand_bbox_obeys_image_boundaries(self) -> None:
        bbox = DetectionBox(45, 45, 55, 55, 1.0)
        x1, y1, x2, y2 = expand_bbox(
            bbox, scale=10.0, image_width=100, image_height=100
        )
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
                image_id="",
                dog_boxes=(),
                face_boxes=(),
                nose_boxes=(),
                body_keypoints=(),
                face_landmarks=(),
                model_name="test",
                model_family="test",
                inference_ms=0.0,
            )


if __name__ == "__main__":
    unittest.main()
