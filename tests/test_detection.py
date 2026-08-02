from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from localization.detection import (
    Detection,
    DogDetector,
    DogDetectorConfig,
    FrameSelector,
    QualityMetrics,
    suppress_contained_duplicates,
)


class DetectionTests(unittest.TestCase):
    def test_creation(self) -> None:
        d = Detection(x1=10, y1=20, x2=200, y2=300,
                      confidence=0.95, class_id=16, class_name="dog")
        self.assertEqual(d.width, 190)
        self.assertEqual(d.height, 280)
        self.assertEqual(d.area, 190 * 280)
        self.assertAlmostEqual(d.center[0], 105.0)
        self.assertAlmostEqual(d.center[1], 160.0)

    def test_face_region(self) -> None:
        d = Detection(x1=0, y1=0, x2=200, y2=400,
                      confidence=0.9, class_id=16, class_name="dog")
        face = d.face_region(0.45)
        self.assertEqual(face.x1, 0)
        self.assertEqual(face.y1, 0)
        self.assertEqual(face.x2, 200)
        self.assertEqual(face.y2, 180)
        self.assertEqual(face.height, 180)

    def test_to_dict(self) -> None:
        d = Detection(x1=10, y1=20, x2=200, y2=300,
                      confidence=0.95, class_id=16, class_name="dog")
        dd = d.to_dict()
        self.assertEqual(dd["x1"], 10)
        self.assertEqual(dd["class"], "dog")
        self.assertAlmostEqual(dd["confidence"], 0.95)

    def test_suppresses_contained_duplicate_but_preserves_distinct_dogs(self) -> None:
        detections = [
            Detection(10, 10, 90, 90, 0.9, 16, "dog"),
            Detection(8, 8, 95, 98, 0.7, 16, "dog"),
            Detection(120, 10, 190, 90, 0.8, 16, "dog"),
        ]

        kept = suppress_contained_duplicates(detections, iou_threshold=0.45)

        self.assertEqual([item.confidence for item in kept], [0.9, 0.8])


class QualityMetricsTests(unittest.TestCase):
    def test_good_quality(self) -> None:
        q = QualityMetrics(
            sharpness=200.0, face_coverage=0.5, brightness=0.6,
            is_blurry=False, is_dark=False,
        )
        self.assertTrue(q.acceptable())

    def test_blurry(self) -> None:
        q = QualityMetrics(
            sharpness=10.0, face_coverage=0.5, brightness=0.6,
            is_blurry=True, is_dark=False,
        )
        self.assertFalse(q.acceptable(min_sharpness=50.0))

    def test_low_coverage(self) -> None:
        q = QualityMetrics(
            sharpness=200.0, face_coverage=0.01, brightness=0.6,
            is_blurry=False, is_dark=False,
        )
        self.assertFalse(q.acceptable(min_coverage=0.1))

    def test_dark(self) -> None:
        q = QualityMetrics(
            sharpness=200.0, face_coverage=0.5, brightness=0.01,
            is_blurry=False, is_dark=True,
        )
        self.assertFalse(q.acceptable(min_brightness=0.05))

    def test_to_dict(self) -> None:
        q = QualityMetrics(
            sharpness=150.5, face_coverage=0.3, brightness=0.5,
            is_blurry=False, is_dark=False,
        )
        d = q.to_dict()
        self.assertAlmostEqual(d["sharpness"], 150.5)
        self.assertFalse(d["is_blurry"])


class ComputeQualityTests(unittest.TestCase):
    def test_on_random_image(self) -> None:
        img = Image.fromarray(
            np.random.randint(0, 255, (200, 200), dtype=np.uint8), mode="L"
        )
        q = DogDetector.compute_quality(img)
        self.assertIsInstance(q.sharpness, float)
        self.assertIsInstance(q.face_coverage, float)

    def test_with_detection(self) -> None:
        img = Image.fromarray(
            np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        )
        d = Detection(x1=20, y1=30, x2=180, y2=190,
                      confidence=0.9, class_id=16, class_name="dog")
        q = DogDetector.compute_quality(img, d)
        self.assertGreater(q.face_coverage, 0.0)
        self.assertLessEqual(q.face_coverage, 1.0)


class FrameSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = FrameSelector(top_k_frames=3)
        self.img = Image.fromarray(
            np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        )
        self.det = Detection(
            x1=20, y1=30, x2=180, y2=190,
            confidence=0.9, class_id=16, class_name="dog",
        )

    def test_select_top_k(self) -> None:
        frames = [(0, self.img, self.det),
                  (1, self.img, self.det),
                  (2, self.img, self.det),
                  (3, self.img, self.det),
                  (4, self.img, self.det)]
        selected = self.selector.select(frames)
        self.assertEqual(len(selected), 3)

    def test_select_less_than_k(self) -> None:
        frames = [(0, self.img, self.det)]
        selected = self.selector.select(frames)
        self.assertEqual(len(selected), 1)

    def test_empty(self) -> None:
        selected = self.selector.select([])
        self.assertEqual(selected, [])

    def test_minimum_sharpness_is_applied(self) -> None:
        selector = FrameSelector(top_k_frames=3, min_sharpness=1e9)
        self.assertEqual(selector.select([(0, self.img, self.det)]), [])


class DogDetectorYOLOTests(unittest.TestCase):
    def test_config_defaults(self) -> None:
        cfg = DogDetectorConfig()
        self.assertEqual(cfg.model_size, "n")
        self.assertEqual(cfg.device, "cpu")
        self.assertAlmostEqual(cfg.conf_threshold, 0.25)
        self.assertEqual(cfg.target_size, 224)

    @mock.patch("localization.detection.YOLO")
    def test_detector_creation(self, yolo: mock.Mock) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt") as model:
            model.write(b"pinned detector fixture")
            model.flush()
            digest = hashlib.sha256(b"pinned detector fixture").hexdigest()
            cfg = DogDetectorConfig(
                model_path=model.name,
                model_sha256=digest,
                device="cuda:0",
            )
            det = DogDetector(cfg)
            self.assertIsNotNone(det)
            staged_path = Path(yolo.call_args.args[0])
            self.assertNotEqual(staged_path, Path(model.name))
            self.assertEqual(staged_path.read_bytes(), b"pinned detector fixture")
            yolo.return_value.to.assert_called_once_with("cuda:0")
            det.close()
            self.assertFalse(staged_path.exists())

    @mock.patch("localization.detection.YOLO")
    def test_detector_rejects_implicit_download(self, yolo: mock.Mock) -> None:
        with self.assertRaisesRegex(RuntimeError, "explicit local"):
            DogDetector(DogDetectorConfig(device="cpu"))
        yolo.assert_not_called()
