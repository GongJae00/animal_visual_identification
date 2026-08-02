from __future__ import annotations

import unittest

from evaluation import (
    ClusterBootstrapConfig,
    ClusterUnit,
    FrozenVerificationThreshold,
    ScoredVerificationPair,
    VerificationDirection,
    evaluate_frozen_verification_threshold,
    required_zero_event_trials,
    wilson_rate,
    zero_event_exact_upper_bound,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def threshold() -> FrozenVerificationThreshold:
    return FrozenVerificationThreshold(
        score_threshold=0.8,
        target_fmr=1e-3,
        confidence_level=0.95,
        direction=VerificationDirection.RGB_TO_RGB,
        model_sha256=HASH_A,
        gallery_sha256=HASH_B,
        calibration_manifest_sha256=HASH_C,
    )


def bootstrap() -> ClusterBootstrapConfig:
    return ClusterBootstrapConfig(
        cluster_unit=ClusterUnit.QUERY_DOG,
        resamples=1_000,
        seed=7,
        confidence_level=0.95,
    )


def pair(
    pair_id: str,
    query_dog: str,
    reference_dog: str,
    score: float,
) -> ScoredVerificationPair:
    return ScoredVerificationPair(
        pair_id=pair_id,
        query_track_id=f"track-{pair_id}",
        reference_template_id=f"template-{reference_dog}",
        query_dog_id=query_dog,
        reference_dog_id=reference_dog,
        query_session_id=f"query-session-{pair_id}",
        reference_session_id=f"reference-session-{reference_dog}",
        score=score,
    )


class EvaluationTests(unittest.TestCase):
    def test_threshold_equality_is_consistently_accepted(self) -> None:
        result = evaluate_frozen_verification_threshold(
            (
                pair("positive-pass", "dog-1", "dog-1", 0.8),
                pair("positive-fail", "dog-2", "dog-2", 0.79),
                pair("negative-fail", "dog-3", "dog-4", 0.8),
                pair("negative-pass", "dog-5", "dog-6", 0.1),
            ),
            threshold=threshold(),
            test_manifest_sha256=HASH_D,
            bootstrap=bootstrap(),
        )
        self.assertEqual(result.false_match_rate.events, 1)
        self.assertEqual(result.false_match_rate.trials, 2)
        self.assertEqual(result.false_non_match_rate.events, 1)
        self.assertEqual(result.score_rule, (
            "accept_same_identity_if_score_greater_than_or_equal"
        ))

    def test_calibration_manifest_cannot_be_reused_as_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            evaluate_frozen_verification_threshold(
                (
                    pair("positive", "dog-1", "dog-1", 0.9),
                    pair("negative", "dog-2", "dog-3", 0.1),
                ),
                threshold=threshold(),
                test_manifest_sha256=HASH_C,
                bootstrap=bootstrap(),
            )

    def test_pair_must_be_session_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "session-disjoint"):
            ScoredVerificationPair(
                pair_id="pair",
                query_track_id="track",
                reference_template_id="template",
                query_dog_id="dog",
                reference_dog_id="dog",
                query_session_id="same-session",
                reference_session_id="same-session",
                score=0.9,
            )

    def test_duplicate_pair_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_frozen_verification_threshold(
                (
                    pair("duplicate", "dog-1", "dog-1", 0.9),
                    pair("duplicate", "dog-2", "dog-3", 0.1),
                ),
                threshold=threshold(),
                test_manifest_sha256=HASH_D,
                bootstrap=bootstrap(),
            )

    def test_wilson_interval_does_not_call_zero_events_zero_risk(self) -> None:
        estimate = wilson_rate(0, 100, confidence_level=0.95)
        self.assertEqual(estimate.estimate, 0.0)
        self.assertGreater(estimate.upper_bound, 0.0)

    def test_cluster_bootstrap_is_content_addressed_and_deterministic(self) -> None:
        pairs = (
            pair("positive-pass", "dog-1", "dog-1", 0.9),
            pair("positive-fail", "dog-2", "dog-2", 0.1),
            pair("negative-fail", "dog-3", "dog-4", 0.9),
            pair("negative-pass", "dog-5", "dog-6", 0.1),
        )
        first = evaluate_frozen_verification_threshold(
            pairs,
            threshold=threshold(),
            test_manifest_sha256=HASH_D,
            bootstrap=bootstrap(),
        )
        second = evaluate_frozen_verification_threshold(
            pairs,
            threshold=threshold(),
            test_manifest_sha256=HASH_D,
            bootstrap=bootstrap(),
        )
        self.assertEqual(
            first.false_match_cluster_rate,
            second.false_match_cluster_rate,
        )
        self.assertEqual(first.false_match_cluster_rate.cluster_count, 2)
        self.assertEqual(
            first.cluster_bootstrap_config_sha256,
            bootstrap().config_sha256,
        )

    def test_zero_event_sample_size_for_one_in_ten_thousand(self) -> None:
        trials = required_zero_event_trials(
            1e-4,
            confidence_level=0.95,
        )
        self.assertEqual(trials, 29_956)
        self.assertLessEqual(
            zero_event_exact_upper_bound(
                trials,
                confidence_level=0.95,
            ),
            1e-4,
        )


if __name__ == "__main__":
    unittest.main()
