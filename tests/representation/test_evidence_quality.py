from __future__ import annotations

import builtins
import math
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from representation.evidence.base import AbstractEvidencer
from representation.quality.quality import (
    QualityDiagnostics,
    QualityLimits,
    QualityMapping,
    QualityReason,
    QualityState,
    estimate_blur,
    observe_quality,
)
from representation.channels.extraction import EvidenceExtractionPipeline

class _TestEvidencer(AbstractEvidencer):
    name = "test"
    output_dim = 2

    def __init__(self, minimum_brightness: float | None = None) -> None:
        self._minimum_brightness = minimum_brightness

    def extract(self, image: Image.Image) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return np.stack([self.extract(image) for image in images])

    def map_quality(self, diagnostics: QualityDiagnostics) -> QualityMapping:
        if self._minimum_brightness is None:
            return super().map_quality(diagnostics)
        score = diagnostics.brightness / 255.0
        if diagnostics.brightness >= self._minimum_brightness:
            return QualityMapping(
                QualityState.ELIGIBLE,
                (QualityReason.QUALITY_ACCEPTABLE,),
                score,
            )
        return QualityMapping(
            QualityState.INELIGIBLE,
            (QualityReason.LOW_BRIGHTNESS,),
            score,
        )

class QualityDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_are_finite_and_include_dimensions(self) -> None:
        image = Image.fromarray(
            np.tile(np.array([0, 64, 128, 255], dtype=np.uint8), (4, 1)),
            mode="L",
        )
        observation = observe_quality(image, "appearance")

        self.assertEqual(observation.state, QualityState.UNAVAILABLE)
        self.assertEqual(
            observation.reason_codes,
            (QualityReason.MAPPING_NOT_CONFIGURED,),
        )
        self.assertIsNone(observation.score)
        self.assertIsNotNone(observation.diagnostics)
        diagnostics = observation.diagnostics
        assert diagnostics is not None
        self.assertEqual((diagnostics.width, diagnostics.height), (4, 4))
        self.assertEqual(diagnostics.pixel_count, 16)
        self.assertTrue(math.isfinite(diagnostics.sharpness))
        self.assertTrue(math.isfinite(diagnostics.brightness))
        self.assertTrue(math.isfinite(diagnostics.contrast))

    def test_nonfinite_diagnostics_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            QualityDiagnostics(
                sharpness=float("nan"),
                brightness=100.0,
                contrast=10.0,
                width=4,
                height=4,
                pixel_count=16,
            )

    def test_sharpness_has_no_scipy_runtime_dependency(self) -> None:
        original_import = builtins.__import__

        def reject_scipy(name: str, *args: object, **kwargs: object) -> object:
            if name == "scipy" or name.startswith("scipy."):
                raise AssertionError("SciPy must not be imported")
            return original_import(name, *args, **kwargs)

        image = Image.new("L", (8, 8), color=128)
        with patch("builtins.__import__", side_effect=reject_scipy):
            self.assertEqual(estimate_blur(image), 0.0)

class QualityValidationTests(unittest.TestCase):
    def test_small_and_excessive_images_fail_closed(self) -> None:
        small = observe_quality(Image.new("RGB", (2, 4)), "nose")
        self.assertEqual(small.state, QualityState.UNAVAILABLE)
        self.assertEqual(small.reason_codes, (QualityReason.DIMENSIONS_TOO_SMALL,))

        limits = QualityLimits(min_dimension=3, max_dimension=10, max_pixels=12)
        excessive = observe_quality(
            Image.new("RGB", (4, 4)),
            "nose",
            limits=limits,
        )
        self.assertEqual(
            excessive.reason_codes,
            (QualityReason.PIXEL_LIMIT_EXCEEDED,),
        )

    def test_roi_is_strictly_validated_and_not_clipped(self) -> None:
        image = Image.new("RGB", (10, 8), color=100)
        invalid = observe_quality(image, "face", roi_box=(-1, 0, 5, 5))
        self.assertEqual(invalid.state, QualityState.UNAVAILABLE)
        self.assertEqual(invalid.reason_codes, (QualityReason.ROI_OUT_OF_BOUNDS,))
        self.assertIsNone(invalid.diagnostics)

        valid = observe_quality(image, "face", roi_box=(2, 1, 8, 6))
        self.assertEqual(valid.roi_box, (2, 1, 8, 6))
        assert valid.diagnostics is not None
        self.assertEqual(
            (valid.diagnostics.width, valid.diagnostics.height),
            (6, 5),
        )

    def test_mapping_exceptions_become_unavailable_observations(self) -> None:
        def broken_mapper(diagnostics: QualityDiagnostics) -> QualityMapping:
            raise RuntimeError("broken calibration")

        observation = observe_quality(
            Image.new("RGB", (4, 4)),
            "appearance",
            mapper=broken_mapper,
        )
        self.assertEqual(observation.state, QualityState.UNAVAILABLE)
        self.assertEqual(observation.reason_codes, (QualityReason.MAPPING_ERROR,))
        self.assertIsNone(observation.score)
        self.assertIsNotNone(observation.diagnostics)

class EnrollmentQualityTests(unittest.TestCase):
    def test_missing_quality_never_defaults_to_one(self) -> None:
        pipeline = EvidenceExtractionPipeline({"appearance": _TestEvidencer()})
        observations = pipeline.estimate_quality(Image.new("RGB", (8, 8)))

        observation = observations["appearance"]
        self.assertEqual(observation.state, QualityState.UNAVAILABLE)
        self.assertIsNone(observation.score)
        self.assertNotEqual(observation.score, 1.0)

    def test_channel_hooks_map_the_same_diagnostics_deterministically(self) -> None:
        pipeline = EvidenceExtractionPipeline({
            "permissive": _TestEvidencer(minimum_brightness=100.0),
            "strict": _TestEvidencer(minimum_brightness=200.0),
        })
        image = Image.new("RGB", (8, 8), color=(128, 128, 128))

        first = pipeline.estimate_quality(image)
        second = pipeline.estimate_quality(image)

        self.assertEqual(first, second)
        self.assertEqual(first["permissive"].state, QualityState.ELIGIBLE)
        self.assertEqual(first["strict"].state, QualityState.INELIGIBLE)
        self.assertEqual(first["permissive"].channel, "permissive")
        self.assertEqual(first["strict"].channel, "strict")

    def test_extract_with_quality_returns_typed_observations(self) -> None:
        pipeline = EvidenceExtractionPipeline({
            "appearance": _TestEvidencer(minimum_brightness=100.0),
        })
        embeddings, observations = pipeline.extract_with_quality(
            Image.new("RGB", (8, 8), color=(128, 128, 128))
        )

        np.testing.assert_array_equal(
            embeddings["appearance"],
            np.array([1.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(
            observations["appearance"].state,
            QualityState.ELIGIBLE,
        )

if __name__ == "__main__":
    unittest.main()
