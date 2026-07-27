from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class PostSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        from cvi.post_search import (AdaptiveFusion, OpenSetDecision,
                                     ScoreCalibrator, TemporalAggregator)
        self.Fusion = AdaptiveFusion
        self.OpenSet = OpenSetDecision
        self.Calibrator = ScoreCalibrator
        self.TemporalAggregator = TemporalAggregator

    def test_adaptive_fusion(self) -> None:
        fuser = self.Fusion(quality_threshold=0.3)
        scores = {"visual": 0.9, "texture": 0.7}
        fused = fuser.fuse(scores, weights={"visual": 2.0})
        self.assertAlmostEqual(fused, (2.0 * 0.9 + 0.7) / 3.0)

    def test_adaptive_fusion_quality_gating(self) -> None:
        fuser = self.Fusion(quality_threshold=0.5)
        scores = {"visual": 0.9, "texture": 0.7}
        qualities = {"visual": 0.1, "texture": 0.8}
        fused = fuser.fuse(scores, qualities=qualities)
        expected = (0.1 * 0.9 + 1.0 * 0.7) / (0.1 + 1.0)
        self.assertAlmostEqual(fused, expected)

    def test_open_set_accept(self) -> None:
        decider = self.OpenSet(min_similarity=0.5, distance_ratio_threshold=1.0)
        results = [(0, 0.85, {}), (1, 0.3, {})]
        rejected, reason = decider.reject(results[0], results)
        self.assertFalse(rejected)

    def test_open_set_reject_low_sim(self) -> None:
        decider = self.OpenSet(min_similarity=0.5)
        results = [(0, 0.3, {})]
        rejected, reason = decider.reject(results[0], results)
        self.assertTrue(rejected)
        self.assertIn("low_similarity", reason)

    def test_open_set_reject_ratio(self) -> None:
        decider = self.OpenSet(distance_ratio_threshold=0.1)
        results = [(0, 0.6, {}), (1, 0.4, {})]
        dr = self.OpenSet.distance_ratio(0.6, 0.4)
        self.assertAlmostEqual(dr, 0.4/0.6)
        rejected, reason = decider.reject(results[0], results)
        self.assertTrue(rejected)
        self.assertIn("ambiguous", reason)

    def test_temporal_median(self) -> None:
        agg = self.TemporalAggregator("median")
        embs = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([1.5, 3.5])]
        result = agg.aggregate(embs)
        expected = np.array([1.5, 3.5])
        np.testing.assert_array_equal(result, expected)

    def test_temporal_mean(self) -> None:
        agg = self.TemporalAggregator("mean")
        embs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = agg.aggregate(embs)
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_temporal_weighted(self) -> None:
        agg = self.TemporalAggregator("mean")
        embs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        result = agg.aggregate_weighted(embs, [1.0, 3.0])
        expected = np.array([0.25, 0.75])
        np.testing.assert_array_equal(result, expected)

    def test_calibrator_save_load(self) -> None:
        from cvi.post_search import ScoreCalibrator
        cal = ScoreCalibrator()
        cal.fit({"vis": np.array([0.1, 0.5, 0.9])}, np.array([0, 1, 1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrator.json"
            cal.save(path)
            loaded = ScoreCalibrator.load(path)
        val = loaded.calibrate(0.9, "vis")
        self.assertAlmostEqual(val, 1.0, places=1)
        val2 = loaded.calibrate(0.1, "vis")
        self.assertAlmostEqual(val2, 0.0, places=1)

    def test_calibrator_uses_strict_json_not_pickle(self) -> None:
        from cvi.post_search import ScoreCalibrator

        cal = ScoreCalibrator()
        cal.fit({"vis": np.array([0.1, 0.5, 0.9])}, np.array([0, 1, 1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrator.json"
            cal.save(path)
            self.assertTrue(path.read_text(encoding="utf-8").startswith("{"))
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
            payload["schema_version"] = "tampered"
            path.chmod(0o600)
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                ScoreCalibrator.load(path)

    def test_constant_score_calibrator_round_trip(self) -> None:
        from cvi.post_search import ScoreCalibrator

        cal = ScoreCalibrator()
        cal.fit({"vis": np.array([0.5, 0.5])}, np.array([0, 1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrator.json"
            cal.save(path)
            loaded = ScoreCalibrator.load(path)
        self.assertAlmostEqual(loaded.calibrate(0.5, "vis"), 0.5)

    def test_private_json_publication_rejects_nonfinite_values(self) -> None:
        from cvi.protected_io import write_private_json_bundle

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaisesRegex(ValueError, "JSON compliant"):
                write_private_json_bundle(((path, {"threshold": float("inf")}),))
            self.assertFalse(path.exists())

    def test_empty_scores_fusion(self) -> None:
        fuser = self.Fusion()
        self.assertEqual(fuser.fuse({}), 0.0)
