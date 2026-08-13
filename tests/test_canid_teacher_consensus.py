from __future__ import annotations

import unittest

from parsing.consensus import (
    compute_error_correlation,
    consensus_admission,
    consensus_dog_bbox,
    consensus_dog_instances,
    robust_weighted_keypoint,
    weighted_box_fusion,
)
from parsing.types import (
    DetectionBox,
    Keypoint,
    KeypointSet,
    LocalizationResult,
)


def _artifact(character: str) -> dict[str, str]:
    return {"artifact_sha256": character * 64}


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
            image_id="img1",
            dog_boxes=(DetectionBox(0, 0, 10, 10, 0.9),),
            face_boxes=(),
            nose_boxes=(),
            body_keypoints=(),
            face_landmarks=(),
            model_name="m1",
            model_family="f1",
            inference_ms=0.0,
            metadata=_artifact("a"),
        )
        r2 = LocalizationResult(
            image_id="img1",
            dog_boxes=(DetectionBox(0, 0, 10, 10, 0.9),),
            face_boxes=(),
            nose_boxes=(),
            body_keypoints=(),
            face_landmarks=(),
            model_name="m2",
            model_family="f2",
            inference_ms=0.0,
            metadata=_artifact("b"),
        )
        bbox, info = consensus_dog_bbox((r1, r2))
        self.assertIsNotNone(bbox)
        self.assertEqual(info["status"], "FUSED")

    def test_consensus_dog_bbox_no_detections(self) -> None:
        r1 = LocalizationResult(
            image_id="img1",
            dog_boxes=(),
            face_boxes=(),
            nose_boxes=(),
            body_keypoints=(),
            face_landmarks=(),
            model_name="m1",
            model_family="f1",
            inference_ms=0.0,
            metadata=_artifact("a"),
        )
        bbox, info = consensus_dog_bbox((r1,))
        self.assertIsNone(bbox)
        self.assertEqual(info["status"], "NO_DOG_DETECTED")

    def test_consensus_preserves_two_dog_instances(self) -> None:
        r1 = LocalizationResult(
            image_id="img1",
            dog_boxes=(
                DetectionBox(0, 0, 20, 20, 0.9),
                DetectionBox(80, 0, 100, 20, 0.8),
            ),
            face_boxes=(),
            nose_boxes=(),
            body_keypoints=(),
            face_landmarks=(),
            model_name="m1",
            model_family="f1",
            inference_ms=0.0,
            metadata=_artifact("a"),
        )
        r2 = LocalizationResult(
            image_id="img1",
            dog_boxes=(
                DetectionBox(1, 0, 21, 20, 0.9),
                DetectionBox(79, 0, 99, 20, 0.8),
            ),
            face_boxes=(),
            nose_boxes=(),
            body_keypoints=(),
            face_landmarks=(),
            model_name="m2",
            model_family="f2",
            inference_ms=0.0,
            metadata=_artifact("b"),
        )
        instances = consensus_dog_instances((r1, r2))
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0].support_models, ("a" * 64, "b" * 64))
        self.assertLess(instances[0].bbox.x2, instances[1].bbox.x1)

    def test_single_teacher_never_auto_accepts(self) -> None:
        result = LocalizationResult(
            image_id="img1",
            dog_boxes=(DetectionBox(0, 0, 20, 20, 0.99),),
            face_boxes=(),
            nose_boxes=(),
            body_keypoints=(),
            face_landmarks=(),
            model_name="m1",
            model_family="f1",
            inference_ms=0.0,
            metadata=_artifact("a"),
        )
        self.assertEqual(consensus_dog_instances((result,))[0].admission, "REVIEW")

    def test_artifact_sha_distinguishes_teachers_with_same_filename(self) -> None:
        results = tuple(
            LocalizationResult(
                image_id="img1",
                dog_boxes=(DetectionBox(0, 0, 20, 20, 0.9),),
                face_boxes=(),
                nose_boxes=(),
                body_keypoints=(),
                face_landmarks=(),
                model_name="best",
                model_family="fixture",
                inference_ms=0.0,
                metadata=_artifact(character),
            )
            for character in ("a", "b")
        )

        instance = consensus_dog_instances(results)[0]
        self.assertEqual(instance.support_models, ("a" * 64, "b" * 64))
        self.assertEqual(instance.agreement, 1.0)

    def test_consensus_rejects_duplicate_teacher_artifact_ids(self) -> None:
        result = LocalizationResult(
            image_id="img1",
            dog_boxes=(DetectionBox(0, 0, 20, 20, 0.9),),
            face_boxes=(),
            nose_boxes=(),
            body_keypoints=(),
            face_landmarks=(),
            model_name="best",
            model_family="fixture",
            inference_ms=0.0,
            metadata=_artifact("a"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate teacher artifact"):
            consensus_dog_instances((result, result))

    def test_consensus_requires_teacher_artifact_id(self) -> None:
        result = LocalizationResult(
            image_id="img1",
            dog_boxes=(DetectionBox(0, 0, 20, 20, 0.9),),
            face_boxes=(),
            nose_boxes=(),
            body_keypoints=(),
            face_landmarks=(),
            model_name="best",
            model_family="fixture",
            inference_ms=0.0,
        )
        with self.assertRaisesRegex(ValueError, "requires artifact_sha256"):
            consensus_dog_instances((result,))

    def test_error_correlation_matrix_shape(self) -> None:
        results = {
            "m1": (
                LocalizationResult(
                    image_id="img1",
                    dog_boxes=(DetectionBox(0, 0, 10, 10, 0.9),),
                    face_boxes=(),
                    nose_boxes=(),
                    body_keypoints=(),
                    face_landmarks=(),
                    model_name="m1",
                    model_family="f1",
                    inference_ms=0.0,
                ),
            ),
            "m2": (
                LocalizationResult(
                    image_id="img1",
                    dog_boxes=(),
                    face_boxes=(),
                    nose_boxes=(),
                    body_keypoints=(),
                    face_landmarks=(),
                    model_name="m2",
                    model_family="f2",
                    inference_ms=0.0,
                ),
            ),
        }
        corr = compute_error_correlation(results)
        self.assertEqual(corr["models"], ["m1", "m2"])
        self.assertEqual(corr["correlation"].shape, (2, 2))


class QualityScoringTests(unittest.TestCase):
    def test_dog_quality_is_normalized(self) -> None:
        from parsing.quality import score_dog_quality

        bbox = DetectionBox(50, 50, 200, 200, 0.9)
        quality = score_dog_quality(bbox, image_width=300, image_height=300)
        self.assertGreaterEqual(quality.overall, 0.0)
        self.assertLessEqual(quality.overall, 1.0)

    def test_face_quality_yaw_proxy(self) -> None:
        from parsing.quality import score_face_quality

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

    def test_face_quality_without_named_anchors_is_finite(self) -> None:
        from parsing.quality import score_face_quality

        landmarks = KeypointSet(
            keypoints={"face46.0": Keypoint(10, 10, 0.8)}, schema="face46"
        )
        quality = score_face_quality(landmarks)
        self.assertEqual(quality.anchor_visibility, 0.0)
        self.assertGreaterEqual(quality.overall, 0.0)

    def test_nose_quality_all_inputs_mapped(self) -> None:
        from parsing.quality import score_nose_quality

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
