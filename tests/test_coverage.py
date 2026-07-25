from __future__ import annotations

import json
import unittest

from cvi.contracts import Modality
from cvi.coverage import CoverageAccumulator, CoveragePolicy, EvidenceObservation


def policy() -> CoveragePolicy:
    return CoveragePolicy(
        name="test-policy",
        expected_sample_period_ns=100,
        maximum_hold_periods=2.0,
        minimum_dog_height_px=224,
        minimum_head_long_edge_px=128,
        minimum_face_min_edge_px=96,
        minimum_visible_fraction=0.5,
        maximum_occlusion_fraction=0.5,
        maximum_motion_blur_score=0.5,
        maximum_defocus_blur_score=0.5,
        maximum_cage_bar_occlusion_fraction=0.4,
        minimum_localization_confidence=0.5,
        maximum_ir_saturation_fraction=0.2,
        minimum_usable_tracklet_duration_ns=200,
    )


def observation(
    timestamp_ns: int,
    *,
    modality: Modality = Modality.RGB,
    dog_count: int = 1,
    motion_blur_score: float | None = 0.1,
    track_id: str | None = None,
) -> EvidenceObservation:
    return EvidenceObservation(
        timestamp_ns=timestamp_ns,
        modality=modality,
        dog_count=dog_count,
        dog_crop_height_px=256 if dog_count == 1 else None,
        head_long_edge_px=160 if dog_count == 1 else None,
        face_min_edge_px=100 if dog_count == 1 else None,
        visible_fraction=0.9 if dog_count == 1 else None,
        occlusion_fraction=0.1 if dog_count == 1 else None,
        motion_blur_score=motion_blur_score if dog_count == 1 else None,
        defocus_blur_score=0.1 if dog_count == 1 else None,
        cage_bar_occlusion_fraction=0.1 if dog_count == 1 else None,
        localization_confidence=0.9 if dog_count == 1 else None,
        exposure_ok=True if dog_count == 1 else None,
        ir_saturation_fraction=0.1 if modality is not Modality.RGB else None,
        camera_id="camera-1" if track_id is not None else None,
        session_id="session-1" if track_id is not None else None,
        track_id=track_id,
    )


class CoverageTests(unittest.TestCase):
    def test_duration_weighting_caps_hold_and_preserves_unknown_gap(self) -> None:
        accumulator = CoverageAccumulator(policy())
        accumulator.observe(observation(0))
        accumulator.observe(observation(100, motion_blur_score=0.9))
        accumulator.observe(
            observation(500, modality=Modality.IR, dog_count=0)
        )
        result = accumulator.finalize()
        self.assertEqual(result["aggregate"]["observed_duration_ns"], 400)
        self.assertEqual(result["unobserved_gap_duration_ns"], 200)
        self.assertEqual(result["aggregate"]["single_dog_duration_ns"], 300)
        self.assertEqual(result["aggregate"]["full_body_usable_duration_ns"], 100)
        self.assertAlmostEqual(
            result["aggregate"]["full_body_coverage_given_single_dog"],
            1 / 3,
        )
        self.assertEqual(
            result["by_modality"]["IR"]["no_dog_duration_ns"],
            100,
        )

    def test_missing_quality_is_not_imputed_as_usable(self) -> None:
        accumulator = CoverageAccumulator(policy())
        accumulator.observe(observation(0, motion_blur_score=None))
        result = accumulator.finalize()
        self.assertEqual(result["aggregate"]["missing_quality_duration_ns"], 100)
        self.assertEqual(result["aggregate"]["full_body_usable_duration_ns"], 0)

    def test_observations_must_be_strictly_chronological(self) -> None:
        accumulator = CoverageAccumulator(policy())
        accumulator.observe(observation(100))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            accumulator.observe(observation(100))

    def test_policy_round_trip_and_hash_are_stable(self) -> None:
        original = policy()
        restored = CoveragePolicy.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )
        self.assertEqual(restored, original)
        self.assertEqual(restored.policy_sha256, original.policy_sha256)

    def test_pixel_histogram_is_fixed_size(self) -> None:
        accumulator = CoverageAccumulator(policy())
        accumulator.observe(observation(0))
        result = accumulator.finalize()
        histogram = result["aggregate"]["dog_crop_height_histogram_ns"]
        self.assertEqual(len(histogram), 8)
        self.assertEqual(sum(histogram), 100)
        visibility = result["aggregate"]["visibility_histogram_ns"]
        self.assertEqual(len(visibility), 10)
        self.assertEqual(sum(visibility), 100)
        self.assertEqual(
            result["aggregate"]["head_size_availability_given_single_dog"],
            1.0,
        )

    def test_declared_timeline_preserves_leading_and_trailing_unknown(self) -> None:
        accumulator = CoverageAccumulator(policy(), timeline_start_ns=0)
        accumulator.observe(observation(100))
        result = accumulator.finalize(timeline_end_ns=500)
        self.assertEqual(result["aggregate"]["observed_duration_ns"], 200)
        self.assertEqual(result["unobserved_gap_duration_ns"], 300)
        self.assertEqual(result["timeline_start_ns"], 0)
        self.assertEqual(result["timeline_end_ns"], 500)

    def test_empty_timeline_is_entirely_unobserved(self) -> None:
        accumulator = CoverageAccumulator(policy(), timeline_start_ns=100)
        result = accumulator.finalize(timeline_end_ns=500)
        self.assertEqual(result["aggregate"]["observed_duration_ns"], 0)
        self.assertEqual(result["unobserved_gap_duration_ns"], 400)

    def test_contiguous_usable_evidence_forms_tracklet_opportunity(self) -> None:
        accumulator = CoverageAccumulator(policy(), timeline_start_ns=0)
        accumulator.observe(observation(0, track_id="track-1"))
        accumulator.observe(observation(100, track_id="track-1"))
        accumulator.observe(observation(200, track_id="track-1"))
        result = accumulator.finalize(timeline_end_ns=300)
        self.assertEqual(result["usable_tracklet_opportunities"], 1)
        self.assertEqual(result["usable_tracklet_duration_ns"], 300)
        self.assertEqual(
            result["usable_evidence_missing_track_key_duration_ns"],
            0,
        )

    def test_usable_frames_without_track_namespace_are_not_tracklets(self) -> None:
        accumulator = CoverageAccumulator(policy(), timeline_start_ns=0)
        accumulator.observe(observation(0))
        result = accumulator.finalize(timeline_end_ns=200)
        self.assertEqual(result["usable_tracklet_opportunities"], 0)
        self.assertEqual(
            result["usable_evidence_missing_track_key_duration_ns"],
            200,
        )


if __name__ == "__main__":
    unittest.main()
