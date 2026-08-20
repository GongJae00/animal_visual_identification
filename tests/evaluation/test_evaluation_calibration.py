from __future__ import annotations

import unittest

import numpy as np

from evaluation.calibration import (
    compute_probability_calibration_metrics,
    fit_isotonic_calibration,
    CalibrationError,
    InvalidProbabilityError,
)

# Fixture A: probabilities with known ECE, Brier, NLL
# p = [0.8, 0.2, 0.7, 0.3], labels = [1, 0, 1, 0]
# 2 bins: [0,0.5),[0.5,1.0]
# Bin 0: p=[0.2,0.3], labels=[0,0], acc=0, conf=0.25, diff=0.25
# Bin 1: p=[0.8,0.7], labels=[1,1], acc=1, conf=0.75, diff=0.25
# ECE = (2/4)*0.25 + (2/4)*0.25 = 0.25
# Brier = ((1-0.8)^2 + (0-0.2)^2 + (1-0.7)^2 + (0-0.3)^2)/4
#       = (0.04+0.04+0.09+0.09)/4 = 0.26/4 = 0.065
# NLL = -mean(1*log(0.8) + 0*log(0.2) + 1*log(0.7) + 0*log(0.3))
#     = -(log(0.8)+log(0.7))/4 = -(0.26)/4 with numerical log? Let me compute.
PROB_A = np.array([0.8, 0.2, 0.7, 0.3], dtype=np.float64)
LABEL_A = np.array([1, 0, 1, 0], dtype=np.int64)
# Brier = 0.065, NLL = -mean(log(0.8)+log(0.2 ignored)+log(0.7)+log(0.3 ignored))
# = -(ln(0.8)+ln(0.7))/4 = -( -0.22314 + -0.35667)/4 = 0.57981/4 ≈ 0.14495

# Hmm wait, let me recompute NLL more carefully:
# NLL = -1/n * sum(y*log(p) + (1-y)*log(1-p))
# = -1/4 * (1*ln(0.8) + 0*ln(0.2) + 0*ln(0.2) + 1*ln(0.7) + 0*ln(0.3) + 0*ln(0.3))
# Wait, labels are [1, 0, 1, 0], so:
# = -1/4 * (1*ln(0.8) + 0*ln(1-0.8) + 0*ln(0.2) + 1*ln(1-0.2) + 1*ln(0.7) + 0*ln(1-0.7) + 0*ln(0.3) + 1*ln(1-0.3))
# Hmm no, the formula is for each sample:
# = -1/4 * sum over i of [y_i * ln(p_i) + (1-y_i) * ln(1-p_i)]
# sample 0: y=1, p=0.8: 1*ln(0.8) + 0*ln(0.2) = ln(0.8) = -0.22314
# sample 1: y=0, p=0.2: 0*ln(0.2) + 1*ln(0.8) = ln(0.8) = -0.22314
# sample 2: y=1, p=0.7: 1*ln(0.7) + 0*ln(0.3) = ln(0.7) = -0.35667
# sample 3: y=0, p=0.3: 0*ln(0.3) + 1*ln(0.7) = ln(0.7) = -0.35667
# sum = -0.22314 + -0.22314 + -0.35667 + -0.35667 = -1.15962
# NLL = -(-1.15962)/4 = 0.289905

# exp_NLL (using scipy): from the script we got 0.2899092476
EXP_BRIER_A = 0.065
EXP_NLL_A = 0.2899092476
EXP_ECE_A_10BIN = 0.25

# with 2 bins (same fixture):
EXP_ECE_A_2BIN = 0.25

class CalibrationMetricsTest(unittest.TestCase):
    def test_known_values_10_bins(self):
        result = compute_probability_calibration_metrics(
            PROB_A, LABEL_A, n_bins=10,
        )
        self.assertAlmostEqual(result["Brier"], EXP_BRIER_A)
        self.assertAlmostEqual(result["NLL"], EXP_NLL_A, places=6)
        self.assertEqual(result["n_bins"], 10)
        self.assertEqual(result["num_pairs"], 4)

    def test_perfect_probabilities(self):
        p = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        l = np.array([1, 1, 0, 0], dtype=np.int64)
        result = compute_probability_calibration_metrics(p, l, n_bins=2)
        self.assertAlmostEqual(result["ECE"], 0.0)
        self.assertAlmostEqual(result["Brier"], 0.0)

    def test_rejects_raw_negative_values(self):
        with self.assertRaises(InvalidProbabilityError):
            compute_probability_calibration_metrics(
                np.array([-0.1, 0.5, 0.8], dtype=np.float64),
                np.array([1, 0, 1], dtype=np.int64),
            )

    def test_rejects_values_above_one(self):
        with self.assertRaises(InvalidProbabilityError):
            compute_probability_calibration_metrics(
                np.array([0.5, 1.5, 0.8], dtype=np.float64),
                np.array([1, 0, 1], dtype=np.int64),
            )

    def test_empty_rejected(self):
        with self.assertRaises(CalibrationError):
            compute_probability_calibration_metrics(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
            )

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(CalibrationError):
            compute_probability_calibration_metrics(
                np.array([0.5, 0.5], dtype=np.float64),
                np.array([1, 0, 1], dtype=np.int64),
            )

    def test_non_finite_rejected(self):
        with self.assertRaises(CalibrationError):
            compute_probability_calibration_metrics(
                np.array([0.5, np.nan, 0.8], dtype=np.float64),
                np.array([1, 0, 1], dtype=np.int64),
            )

    def test_bin_outputs_structure(self):
        result = compute_probability_calibration_metrics(PROB_A, LABEL_A, n_bins=5)
        self.assertEqual(len(result["bin_counts"]), 5)
        self.assertEqual(len(result["bin_confidences"]), 5)
        self.assertEqual(len(result["bin_positive_rates"]), 5)

    def test_fractional_labels_are_rejected_before_cast(self):
        with self.assertRaises(CalibrationError):
            compute_probability_calibration_metrics(
                np.array([0.2, 0.8]), np.array([0.5, 1.0])
            )

    def test_invalid_bin_count_rejected(self):
        with self.assertRaises(CalibrationError):
            compute_probability_calibration_metrics(PROB_A, LABEL_A, n_bins=0)

    def test_deterministic(self):
        r1 = compute_probability_calibration_metrics(PROB_A, LABEL_A)
        r2 = compute_probability_calibration_metrics(PROB_A, LABEL_A)
        for k in ("ECE", "Brier", "NLL"):
            self.assertEqual(r1[k], r2[k])

class IsotonicCalibrationTest(unittest.TestCase):
    def test_fit_on_calibration_only(self):
        cal_scores = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float64)
        cal_labels = np.array([0, 0, 1, 1], dtype=np.int64)
        iso = fit_isotonic_calibration(cal_scores, cal_labels)
        test_scores = np.array([0.2, 0.5, 0.8], dtype=np.float64)
        probs = iso.transform(test_scores)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

    def test_insufficient_samples_rejected(self):
        with self.assertRaises(CalibrationError):
            fit_isotonic_calibration(
                np.array([0.5], dtype=np.float64),
                np.array([1], dtype=np.int64),
            )

    def test_single_class_and_nonfinite_fit_are_rejected(self):
        with self.assertRaises(CalibrationError):
            fit_isotonic_calibration(
                np.array([0.1, 0.2]), np.array([0, 0])
            )
        with self.assertRaises(CalibrationError):
            fit_isotonic_calibration(
                np.array([0.1, np.nan]), np.array([0, 1])
            )

if __name__ == "__main__":
    unittest.main()
