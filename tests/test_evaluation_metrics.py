from __future__ import annotations

import unittest

import numpy as np

from cvi.evaluation.verification import (
    compute_verification_metrics,
    compute_verification_curve,
    EvaluationError,
    LengthMismatchError,
    InvalidLabelError,
    NonFiniteScoreError,
    EmptyInputError,
    SingleClassError,
)
from cvi.evaluation.retrieval import (
    compute_retrieval_metrics,
    RetrievalError,
    EmbeddingNormError,
    ClosedSetViolation,
)

# ===== VERIFICATION FIXTURES =====

# Fixture A: perfect separation, 2 pos 2 neg
# AUC=1.0, AP=1.0, d_prime=5.0, EER=0.0
PERFECT_SCORES = np.array([0.9, 0.7, 0.4, 0.2], dtype=np.float64)
PERFECT_LABELS = np.array([1, 1, 0, 0], dtype=np.int64)
EXP_PERFECT = {"ROC_AUC": 1.0, "PR_AUC": 1.0, "d_prime": 5.0, "EER": 0.0}

# Fixture B: overlap, 2 pos 2 neg
# scores descending: 0.9(1), 0.6(0), 0.4(1), 0.2(0)
# AUC=0.75, AP=0.8333, d_prime=1.1043, EER=0.5 at threshold=0.6
OVERLAP_SCORES = np.array([0.9, 0.4, 0.6, 0.2], dtype=np.float64)
OVERLAP_LABELS = np.array([1, 1, 0, 0], dtype=np.int64)
EXP_OVERLAP = {"ROC_AUC": 0.75, "PR_AUC": 0.8333333333333333, "d_prime": 1.1043152607, "EER": 0.5}

# ===== RETRIEVAL FIXTURES =====

# Fixture C: non-perfect ranking
# gallery: [1,0,0](id=0), [-0.3,0.953,0](id=0), [0,1,0](id=1), [0,0,1](id=2)
# query: [1,0,0](id=0)
# after norm: gallery rows already unit; query unit.
# sims = [1, -0.3003, 0, 0]
# order (stable desc): idx0(1,id=0), idx2(0,id=1), idx3(0,id=2), idx1(-0.3,id=0)
# is_pos: [T, F, F, T]
# n_relevant=2
# AP: rank1 p=1/1, rank4 p=2/4 => (1 + 0.5)/2 = 0.75
# mINP: penetration rank/n_rel: rank1:1/2*1=0.5, rank4:4/2*1=2.0 => max=2.0
GALLERY_C = np.array(
    [[1, 0, 0], [-0.3, 0.953, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
)
GALLERY_IDS_C = np.array([0, 0, 1, 2], dtype=np.int64)
QUERY_C = np.array([[1, 0, 0]], dtype=np.float64)
QUERY_IDS_C = np.array([0], dtype=np.int64)
EXP_C = {"mAP": 0.75, "mINP": 2.0, "Rank-1": 1.0, "Rank-5": 1.0, "Rank-10": 1.0}

# Fixture D: perfect ranking
# gallery: [1,0,0](id=0), [0,1,0](id=1), [0,0,1](id=2)
# query: [1,0,0](id=0), [0,1,0](id=1)
# sims q0: [1,0,0], q1: [0,1,0]
# AP q0: 1.0, AP q1: 1.0, mAP=1.0
GALLERY_D = np.eye(3, dtype=np.float64)
GALLERY_IDS_D = np.array([0, 1, 2], dtype=np.int64)
QUERY_D = np.eye(3, dtype=np.float64)[:2]
QUERY_IDS_D = np.array([0, 1], dtype=np.int64)
EXP_D = {"mAP": 1.0, "mINP": 1.0, "Rank-1": 1.0, "Rank-5": 1.0}


class VerificationMetricsTest(unittest.TestCase):
    def test_perfect_separation(self):
        result = compute_verification_metrics(PERFECT_SCORES, PERFECT_LABELS)
        self.assertEqual(result["num_pairs"], 4)
        self.assertEqual(result["num_positive"], 2)
        self.assertEqual(result["num_negative"], 2)
        self.assertAlmostEqual(result["ROC_AUC"], EXP_PERFECT["ROC_AUC"])
        self.assertAlmostEqual(result["PR_AUC"], EXP_PERFECT["PR_AUC"])
        self.assertAlmostEqual(result["d_prime"], EXP_PERFECT["d_prime"], places=5)
        self.assertAlmostEqual(result["EER"], EXP_PERFECT["EER"])

    def test_overlap_known_values(self):
        result = compute_verification_metrics(OVERLAP_SCORES, OVERLAP_LABELS)
        self.assertAlmostEqual(result["ROC_AUC"], EXP_OVERLAP["ROC_AUC"])
        self.assertAlmostEqual(result["PR_AUC"], EXP_OVERLAP["PR_AUC"], places=6)
        self.assertAlmostEqual(result["d_prime"], EXP_OVERLAP["d_prime"], places=5)
        self.assertAlmostEqual(result["EER"], EXP_OVERLAP["EER"])

    def test_overlap_eer_threshold_is_exact(self):
        curve = compute_verification_curve(OVERLAP_SCORES, OVERLAP_LABELS)
        eer_idx = int(np.argmin(np.abs(curve.far - curve.frr)))
        self.assertAlmostEqual(curve.thresholds[eer_idx], 0.6)
        self.assertAlmostEqual(curve.far[eer_idx], 0.5)
        self.assertAlmostEqual(curve.frr[eer_idx], 0.5)

    def test_negative_cosine_scores(self):
        scores = np.array([0.9, -0.5, 0.3, -0.8], dtype=np.float64)
        labels = np.array([1, 1, 0, 0], dtype=np.int64)
        result = compute_verification_metrics(scores, labels)
        self.assertTrue(0 <= result["ROC_AUC"] <= 1)
        self.assertTrue(result["d_prime"] != 0)

    def test_rejects_nonfinite(self):
        scores = np.array([0.9, np.nan, 0.3, 0.2], dtype=np.float64)
        labels = np.array([1, 1, 0, 0], dtype=np.int64)
        with self.assertRaises(NonFiniteScoreError):
            compute_verification_metrics(scores, labels)

    def test_rejects_invalid_labels(self):
        scores = np.array([0.9, 0.7, 0.3, 0.2], dtype=np.float64)
        labels = np.array([1, 2, 0, 0], dtype=np.int64)
        with self.assertRaises(InvalidLabelError):
            compute_verification_metrics(scores, labels)

    def test_rejects_length_mismatch(self):
        scores = np.array([0.9, 0.7], dtype=np.float64)
        labels = np.array([1, 1, 0], dtype=np.int64)
        with self.assertRaises(LengthMismatchError):
            compute_verification_metrics(scores, labels)

    def test_rejects_empty(self):
        with self.assertRaises(EmptyInputError):
            compute_verification_metrics(
                np.array([], dtype=np.float64), np.array([], dtype=np.int64)
            )

    def test_rejects_single_class(self):
        scores = np.array([0.9, 0.7, 0.3], dtype=np.float64)
        labels = np.array([1, 1, 1], dtype=np.int64)
        with self.assertRaises(SingleClassError):
            compute_verification_metrics(scores, labels)

    def test_verification_curve_has_full_coverage(self):
        curve = compute_verification_curve(PERFECT_SCORES, PERFECT_LABELS)
        self.assertIn(np.inf, curve.thresholds)
        self.assertIn(-np.inf, curve.thresholds)
        self.assertEqual(len(curve.thresholds), len(np.unique(PERFECT_SCORES)) + 2)
        self.assertEqual(len(curve.far), len(curve.thresholds))

    def test_no_rounding_in_core(self):
        result = compute_verification_metrics(OVERLAP_SCORES, OVERLAP_LABELS)
        self.assertAlmostEqual(result["d_prime"], 1.1043152607, places=6)

    def test_deterministic(self):
        r1 = compute_verification_metrics(OVERLAP_SCORES, OVERLAP_LABELS)
        r2 = compute_verification_metrics(OVERLAP_SCORES, OVERLAP_LABELS)
        for k in ("ROC_AUC", "PR_AUC", "EER", "d_prime"):
            self.assertEqual(r1[k], r2[k])


class RetrievalMetricsTest(unittest.TestCase):
    def test_non_perfect_ranking(self):
        result = compute_retrieval_metrics(
            QUERY_C, GALLERY_C, QUERY_IDS_C, GALLERY_IDS_C,
            rank_ks=(1, 5, 10),
        )
        self.assertEqual(result["num_queries"], 1)
        self.assertEqual(result["num_gallery"], 4)
        self.assertAlmostEqual(result["mAP"], EXP_C["mAP"])
        self.assertAlmostEqual(result["mINP"], EXP_C["mINP"])
        self.assertAlmostEqual(result["Rank-1"], EXP_C["Rank-1"])

    def test_perfect_ranking(self):
        result = compute_retrieval_metrics(
            QUERY_D, GALLERY_D, QUERY_IDS_D, GALLERY_IDS_D,
            rank_ks=(1, 5, 10),
        )
        self.assertAlmostEqual(result["mAP"], EXP_D["mAP"])
        self.assertAlmostEqual(result["mINP"], EXP_D["mINP"])
        self.assertEqual(result["num_valid_queries"], 2)

    def test_cosine_normalizes_rows(self):
        scaled = GALLERY_D * 3.0
        q = QUERY_D * 2.0
        result = compute_retrieval_metrics(
            q, scaled, QUERY_IDS_D, GALLERY_IDS_D,
            rank_ks=(1,),
        )
        self.assertAlmostEqual(result["Rank-1"], 1.0)

    def test_zero_norm_rejected(self):
        with self.assertRaises(EmbeddingNormError):
            compute_retrieval_metrics(
                np.array([[0, 0, 0]], dtype=np.float64),
                GALLERY_D, np.array([0], dtype=np.int64), GALLERY_IDS_D,
            )

    def test_self_match_excluded(self):
        g = np.array([[1, 0], [0, 1], [0.707, 0.707]], dtype=np.float64)
        g_ids = np.array([0, 1, 0], dtype=np.int64)
        q = np.array([[1, 0]], dtype=np.float64)
        q_ids = np.array([0], dtype=np.int64)
        excl = np.array([[True, False, False]], dtype=bool)
        with_excl = compute_retrieval_metrics(
            q, g, q_ids, g_ids, rank_ks=(1,), exclude_self=excl,
        )
        self.assertEqual(with_excl["num_valid_queries"], 1)
        self.assertTrue(with_excl.get("self_match_excluded"))

    def test_missing_identity_rejected_in_closed_set(self):
        with self.assertRaises(ClosedSetViolation):
            compute_retrieval_metrics(
                np.array([[1, 0, 0]], dtype=np.float64),
                GALLERY_D, np.array([999], dtype=np.int64), GALLERY_IDS_D,
                closed_set=True,
            )

    def test_configurable_rank_ks(self):
        result = compute_retrieval_metrics(
            QUERY_C, GALLERY_C, QUERY_IDS_C, GALLERY_IDS_C,
            rank_ks=(1, 3),
        )
        self.assertIn("Rank-1", result)
        self.assertIn("Rank-3", result)
        self.assertNotIn("Rank-5", result)

    def test_deterministic_tie_policy(self):
        g = np.array([[1, 0], [0.707, 0.707], [0, 1]], dtype=np.float64)
        g_ids = np.array([0, 0, 1], dtype=np.int64)
        q = np.array([[1, 0]], dtype=np.float64)
        q_ids = np.array([0], dtype=np.int64)
        r1 = compute_retrieval_metrics(q, g, q_ids, g_ids, rank_ks=(1,))
        r2 = compute_retrieval_metrics(q, g, q_ids, g_ids, rank_ks=(1,))
        self.assertEqual(r1["Rank-1"], r2["Rank-1"])

    def test_empty_queries_rejected(self):
        with self.assertRaises(RetrievalError):
            compute_retrieval_metrics(
                np.empty((0, 3), dtype=np.float64),
                GALLERY_D, np.array([], dtype=np.int64), GALLERY_IDS_D,
            )

    def test_non_finite_embeddings_rejected(self):
        q = np.array([[1, np.nan, 0]], dtype=np.float64)
        with self.assertRaises(RetrievalError):
            compute_retrieval_metrics(
                q, GALLERY_D, np.array([0], dtype=np.int64), GALLERY_IDS_D,
            )


if __name__ == "__main__":
    unittest.main()
