from __future__ import annotations

import unittest

import numpy as np

from evaluation.calibration import (
    compute_probability_calibration_metrics,
    fit_isotonic_calibration,
    CalibrationError,
    InvalidProbabilityError,
)

PROB_A = np.array([0.8, 0.2, 0.7, 0.3], dtype=np.float64)
LABEL_A = np.array([1, 0, 1, 0], dtype=np.int64)
EXP_BRIER_A = 0.065
EXP_NLL_A = 0.2899092476
EXP_ECE_A_10BIN = 0.25
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
