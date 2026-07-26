from __future__ import annotations

import unittest

import numpy as np

from cvi.evaluation.verification import (
    compute_verification_curve,
    select_threshold_at_far,
    evaluate_at_threshold,
    EvaluationError,
)
from cvi.evaluation._legacy import (
    wilson_rate,
    zero_event_exact_upper_bound,
    required_zero_event_trials,
)


# Calibration fixture: 6 samples, 2 pos 4 neg
# scores: [0.9, 0.8, 0.6, 0.4, 0.3, 0.1], labels: [1, 1, 0, 0, 0, 0]
# Score-derived thresholds: [inf, 0.9, 0.8, 0.6, 0.4, 0.3, 0.1, -inf]
# Target FAR=0.001: only FAR=0 qualifies (indices 0,1,2)
#   FAR=0 thresholds: inf(TAR=0), 0.9(TAR=0.5), 0.8(TAR=1.0)
# Max TAR among FAR<=0.001: t=0.8, TAR=1.0
CAL_SCORES = np.array([0.9, 0.8, 0.6, 0.4, 0.3, 0.1], dtype=np.float64)
CAL_LABELS = np.array([1, 1, 0, 0, 0, 0], dtype=np.int64)

# Test fixture: 4 samples, 2 pos 2 neg
# OLD code (linspace + valid[-1]): threshold=1.0, TAR=0.0
# NEW code (score-derived + max TAR): threshold=0.8, TAR=1.0
TEST_SCORES = np.array([0.9, 0.7, 0.4, 0.2], dtype=np.float64)
TEST_LABELS = np.array([1, 1, 0, 0], dtype=np.int64)


class ThresholdSelectionTest(unittest.TestCase):
    def test_threshold_selects_max_tar(self):
        op = select_threshold_at_far(CAL_SCORES, CAL_LABELS, 0.001)
        self.assertEqual(op.threshold, 0.8)
        self.assertAlmostEqual(op.calibration_tar, 1.0)
        self.assertAlmostEqual(op.calibration_far, 0.0)

    def test_threshold_fit_on_calibration_only(self):
        op = select_threshold_at_far(CAL_SCORES, CAL_LABELS, 0.001)
        self.assertAlmostEqual(op.calibration_tar, 1.0)
        self.assertAlmostEqual(op.calibration_far, 0.0)

    def test_frozen_threshold_applied_to_test(self):
        op = select_threshold_at_far(CAL_SCORES, CAL_LABELS, 0.001)
        test_result = evaluate_at_threshold(TEST_SCORES, TEST_LABELS, op.threshold)
        self.assertEqual(test_result["threshold"], 0.8)
        self.assertIn("TAR", test_result)
        self.assertIn("FAR", test_result)
        self.assertIn("false_accepts", test_result)

    def test_old_behavior_returns_zero_tar(self):
        old_linspace = np.linspace(0.0, 1.0, 1001)
        old_far = np.zeros(1001)
        old_tar = np.zeros(1001)
        for i, t in enumerate(old_linspace):
            pred = (CAL_SCORES >= t).astype(np.int64)
            tp = ((pred == 1) & (CAL_LABELS == 1)).sum()
            fp = ((pred == 1) & (CAL_LABELS == 0)).sum()
            fn = ((pred == 0) & (CAL_LABELS == 1)).sum()
            tn = ((pred == 0) & (CAL_LABELS == 0)).sum()
            old_far[i] = fp / max(fp + tn, 1)
            old_tar[i] = tp / max(tp + fn, 1)
        old_valid = np.where(old_far <= 0.001)[0]
        old_tar_value = float(old_tar[old_valid[-1]]) if len(old_valid) > 0 else 0.0
        self.assertAlmostEqual(old_tar_value, 0.0)
        op = select_threshold_at_far(CAL_SCORES, CAL_LABELS, 0.001)
        new_tar = evaluate_at_threshold(TEST_SCORES, TEST_LABELS, op.threshold)["TAR"]
        self.assertGreater(new_tar, old_tar_value)

    def test_eer_threshold_curve(self):
        overlap_scores = np.array([0.9, 0.4, 0.6, 0.2], dtype=np.float64)
        overlap_labels = np.array([1, 1, 0, 0], dtype=np.int64)
        curve = compute_verification_curve(overlap_scores, overlap_labels)
        eer_idx = int(np.argmin(np.abs(curve.far - curve.frr)))
        self.assertAlmostEqual(curve.thresholds[eer_idx], 0.6)
        self.assertAlmostEqual(curve.far[eer_idx], 0.5)
        self.assertAlmostEqual(curve.frr[eer_idx], 0.5)

    def test_zero_false_accept_upper_bound(self):
        bound = zero_event_exact_upper_bound(100, confidence_level=0.95)
        self.assertGreater(bound, 0.0)
        self.assertLess(bound, 0.05)
        bound2 = zero_event_exact_upper_bound(1000, confidence_level=0.95)
        self.assertLess(bound2, bound)

    def test_insufficient_negative_trials_warning(self):
        required = required_zero_event_trials(0.001, confidence_level=0.95)
        self.assertGreater(required, 0)
        n_neg_cal = int((1 - CAL_LABELS).sum())
        self.assertLess(n_neg_cal, required)

    def test_wilson_rate_known_fixture(self):
        est = wilson_rate(1, 100, confidence_level=0.95)
        self.assertAlmostEqual(est.estimate, 0.01, places=4)
        self.assertGreater(est.upper_bound, est.estimate)
        self.assertLess(est.lower_bound, est.estimate)

    def test_threshold_operating_point(self):
        op = select_threshold_at_far(CAL_SCORES, CAL_LABELS, 1.0)
        result = evaluate_at_threshold(TEST_SCORES, TEST_LABELS, op.threshold)
        self.assertIn("true_accepts", result)
        self.assertIn("false_accepts", result)
        self.assertIn("false_rejects", result)

    def test_exact_counts_no_sentinels(self):
        op = select_threshold_at_far(CAL_SCORES, CAL_LABELS, 0.001)
        self.assertGreaterEqual(op.calibration_num_negative, 0)
        self.assertGreaterEqual(op.calibration_false_accepts, 0)
        self.assertGreaterEqual(op.calibration_num_positive, 0)
        self.assertGreaterEqual(op.calibration_false_rejects, 0)
        self.assertNotEqual(op.calibration_num_negative, -1)

    def test_target_far_range_validation(self):
        with self.assertRaises(EvaluationError):
            select_threshold_at_far(CAL_SCORES, CAL_LABELS, -0.01)
        with self.assertRaises(EvaluationError):
            select_threshold_at_far(CAL_SCORES, CAL_LABELS, 1.5)


class ThresholdRejectInvalidTest(unittest.TestCase):
    def test_rejects_nonfinite_test_scores(self):
        bad = np.array([0.9, np.inf, 0.3, 0.2], dtype=np.float64)
        with self.assertRaises(EvaluationError):
            evaluate_at_threshold(bad, TEST_LABELS, 0.5)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(EvaluationError):
            evaluate_at_threshold(
                np.array([0.9, 0.7], dtype=np.float64),
                np.array([1, 1, 0], dtype=np.int64), 0.5,
            )

    def test_rejects_invalid_labels(self):
        with self.assertRaises(EvaluationError):
            evaluate_at_threshold(
                TEST_SCORES, np.array([1, 2, 0, 0], dtype=np.int64), 0.5,
            )

    def test_rejects_fractional_labels(self):
        with self.assertRaises(EvaluationError):
            evaluate_at_threshold(
                TEST_SCORES, np.array([1, 0.5, 0, 0], dtype=np.float64), 0.5,
            )


if __name__ == "__main__":
    unittest.main()
