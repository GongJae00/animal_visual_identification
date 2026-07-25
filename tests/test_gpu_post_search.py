from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cvi.gpu_index import GpuIdentityIndex


class GpuIdentityIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.index = GpuIdentityIndex(
            dim=384,
            index_path=self.tmpdir / "index.pt",
            metadata_path=self.tmpdir / "meta.json",
        )

    def tearDown(self) -> None:
        self.index.close()

    def test_empty(self) -> None:
        results = self.index.search(np.random.randn(384).astype(np.float32), top_k=5)
        self.assertEqual(results, [])

    def test_enroll_and_search(self) -> None:
        idx = self.index.enroll(np.random.randn(384).astype(np.float32), "dog_001")
        self.assertEqual(idx, 0)
        self.assertEqual(self.index.size, 1)
        q = self.index._index.reconstruct(0)
        results = self.index.search(q, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

    def test_enroll_batch(self) -> None:
        embs = np.random.randn(3, 384).astype(np.float32)
        ids = ["dog_001", "dog_002", "dog_003"]
        indices = self.index.enroll_batch(embs, ids)
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(self.index.size, 3)

    def test_search_with_evidence(self) -> None:
        emb = np.random.randn(384).astype(np.float32)
        self.index.enroll(emb, "dog_001")
        results = self.index.search_with_evidence(emb, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("evidence", results[0])
        self.assertIn("visual", results[0]["evidence"])

    def test_remove(self) -> None:
        emb = np.random.randn(384).astype(np.float32)
        self.index.enroll(emb, "dog_001")
        self.index.enroll(emb + 0.1, "dog_002")
        self.index.remove(0)
        self.assertEqual(self.index.size, 1)
        meta = self.index._metadata[0]
        self.assertEqual(meta["registered_dog_id"], "dog_002")

    def test_persistence(self) -> None:
        emb = np.random.randn(384).astype(np.float32)
        self.index.enroll(emb, "dog_001")
        self.index.close()
        loaded = GpuIdentityIndex(
            dim=384,
            index_path=self.tmpdir / "index.pt",
            metadata_path=self.tmpdir / "meta.json",
        )
        self.assertEqual(loaded.size, 1)
        loaded.close()

    def test_invalid_dim(self) -> None:
        with self.assertRaises(ValueError):
            self.index.enroll(np.random.randn(128).astype(np.float32), "dog")


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
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            path = Path(f.name)
            cal.save(path)
            loaded = ScoreCalibrator.load(path)
        val = loaded.calibrate(0.9, "vis")
        self.assertAlmostEqual(val, 1.0, places=1)
        val2 = loaded.calibrate(0.1, "vis")
        self.assertAlmostEqual(val2, 0.0, places=1)

    def test_empty_scores_fusion(self) -> None:
        fuser = self.Fusion()
        self.assertEqual(fuser.fuse({}), 0.0)
