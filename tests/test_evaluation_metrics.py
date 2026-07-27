from __future__ import annotations

import unittest

import numpy as np

from cvi.evaluation.retrieval import (
    ClosedSetViolation,
    EmbeddingNormError,
    MetricInvariantError,
    RetrievalError,
    SampleIdValidationError,
    compute_retrieval_metrics,
    evaluate_multi_template_closed_set,
    identity_clustered_bootstrap_ci,
)
from cvi.evaluation.verification import (
    EmptyInputError,
    InvalidLabelError,
    LengthMismatchError,
    NonFiniteScoreError,
    SingleClassError,
    EvaluationError,
    compute_verification_curve,
    compute_verification_metrics,
    select_threshold_at_far,
)

PERFECT_SCORES = np.array([0.9, 0.7, 0.4, 0.2], dtype=np.float64)
PERFECT_LABELS = np.array([1, 1, 0, 0], dtype=np.int64)
EXP_PERFECT = {"ROC_AUC": 1.0, "PR_AUC": 1.0, "d_prime": 5.0, "EER": 0.0}

OVERLAP_SCORES = np.array([0.9, 0.4, 0.6, 0.2], dtype=np.float64)
OVERLAP_LABELS = np.array([1, 1, 0, 0], dtype=np.int64)
EXP_OVERLAP = {"ROC_AUC": 0.75, "PR_AUC": 0.8333333333333333, "d_prime": 1.1043152607, "EER": 0.5}

GALLERY_C = np.array(
    [[1, 0, 0], [-0.3, 0.953, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
)
GALLERY_IDS_C = np.array([0, 0, 1, 2], dtype=np.int64)
QUERY_C = np.array([[1, 0, 0]], dtype=np.float64)
QUERY_IDS_C = np.array([0], dtype=np.int64)
EXP_C = {"mAP": 0.75, "mINP": 0.5, "Rank-1": 1.0, "Rank-5": 1.0, "Rank-10": 1.0}

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

    def test_all_tied_scores_have_a_finite_eer_threshold(self):
        result = compute_verification_metrics(
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([0, 1], dtype=np.int64),
        )
        self.assertEqual(result["EER"], 0.5)
        self.assertEqual(result["EER_threshold"], 0.5)

    def test_rejects_nonfinite(self):
        with self.assertRaises(NonFiniteScoreError):
            compute_verification_metrics(
                np.array([0.9, np.nan, 0.3, 0.2], dtype=np.float64),
                np.array([1, 1, 0, 0], dtype=np.int64),
            )

    def test_rejects_invalid_labels(self):
        with self.assertRaises(InvalidLabelError):
            compute_verification_metrics(
                np.array([0.9, 0.7, 0.3, 0.2], dtype=np.float64),
                np.array([1, 2, 0, 0], dtype=np.int64),
            )

    def test_rejects_fractional_labels(self):
        with self.assertRaises(InvalidLabelError):
            compute_verification_metrics(
                np.array([0.9, 0.7, 0.3, 0.2], dtype=np.float64),
                np.array([1, 0.5, 0, 0], dtype=np.float64),
            )

    def test_rejects_length_mismatch(self):
        with self.assertRaises(LengthMismatchError):
            compute_verification_metrics(
                np.array([0.9, 0.7], dtype=np.float64),
                np.array([1, 1, 0], dtype=np.int64),
            )

    def test_rejects_empty(self):
        with self.assertRaises(EmptyInputError):
            compute_verification_metrics(
                np.array([], dtype=np.float64), np.array([], dtype=np.int64),
            )

    def test_rejects_single_class(self):
        with self.assertRaises(SingleClassError):
            compute_verification_metrics(
                np.array([0.9, 0.7, 0.3], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.int64),
            )

    def test_verification_curve_has_full_coverage(self):
        curve = compute_verification_curve(PERFECT_SCORES, PERFECT_LABELS)
        self.assertIn(np.inf, curve.thresholds)
        self.assertIn(-np.inf, curve.thresholds)
        self.assertEqual(curve.n_pos, 2)
        self.assertEqual(curve.n_neg, 2)

    def test_verification_curve_groups_ties_under_greater_equal_rule(self):
        scores = np.array([0.8, 0.8, 0.2, 0.2], dtype=np.float64)
        labels = np.array([1, 0, 1, 0], dtype=np.int64)
        curve = compute_verification_curve(scores, labels)
        tied = int(np.where(curve.thresholds == 0.8)[0][0])
        self.assertEqual(curve.far[tied], 0.5)
        self.assertEqual(curve.tar[tied], 0.5)

    def test_rejects_nonfinite_reject_all_operating_threshold(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float64)
        labels = np.array([0, 1, 1, 0], dtype=np.int64)
        with self.assertRaisesRegex(EvaluationError, "no finite"):
            select_threshold_at_far(scores, labels, target_far=0.0)

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
        result = compute_retrieval_metrics(QUERY_C, GALLERY_C, QUERY_IDS_C, GALLERY_IDS_C)
        self.assertEqual(result["num_queries"], 1)
        self.assertEqual(result["num_gallery"], 4)
        self.assertAlmostEqual(result["mAP"], EXP_C["mAP"])
        self.assertAlmostEqual(result["mINP"], EXP_C["mINP"])
        self.assertAlmostEqual(result["Rank-1"], EXP_C["Rank-1"])

    def test_perfect_ranking(self):
        result = compute_retrieval_metrics(QUERY_D, GALLERY_D, QUERY_IDS_D, GALLERY_IDS_D)
        self.assertAlmostEqual(result["mAP"], EXP_D["mAP"])
        self.assertAlmostEqual(result["mINP"], EXP_D["mINP"])
        self.assertEqual(result["num_valid_queries"], 2)

    def test_cosine_normalizes_rows(self):
        result = compute_retrieval_metrics(
            QUERY_D * 2.0, GALLERY_D * 3.0, QUERY_IDS_D, GALLERY_IDS_D, rank_ks=(1,),
        )
        self.assertAlmostEqual(result["Rank-1"], 1.0)

    def test_zero_norm_rejected(self):
        with self.assertRaises(EmbeddingNormError):
            compute_retrieval_metrics(
                np.array([[0, 0, 0]], dtype=np.float64),
                GALLERY_D, np.array([0], dtype=np.int64), GALLERY_IDS_D,
            )

    def test_self_match_excluded_by_sample_id(self):
        g = np.array([[1, 0], [0, 1], [0.707, 0.707]], dtype=np.float64)
        g_ids = np.array([0, 1, 0], dtype=np.int64)
        g_sids = np.array(["s1", "s2", "s3"])
        q = np.array([[1, 0]], dtype=np.float64)
        q_ids = np.array([0], dtype=np.int64)
        q_sids = np.array(["s1"])
        with_excl = compute_retrieval_metrics(
            q, g, q_ids, g_ids, rank_ks=(1,),
            query_sample_ids=q_sids, gallery_sample_ids=g_sids,
        )
        self.assertEqual(with_excl["num_valid_queries"], 1)
        self.assertTrue(with_excl.get("self_match_excluded"))
        self.assertAlmostEqual(with_excl["mAP"], 1.0)
        self.assertAlmostEqual(with_excl["mINP"], 1.0)

    def test_self_match_same_identity_other_sample_eligible(self):
        g = np.array([[1, 0], [0.707, 0.707]], dtype=np.float64)
        g_ids = np.array([0, 0], dtype=np.int64)
        g_sids = np.array(["s1", "s2"])
        q = np.array([[1, 0]], dtype=np.float64)
        q_ids = np.array([0], dtype=np.int64)
        q_sids = np.array(["s1"])
        result = compute_retrieval_metrics(
            q, g, q_ids, g_ids, rank_ks=(1,),
            query_sample_ids=q_sids, gallery_sample_ids=g_sids,
        )
        self.assertEqual(result["num_valid_queries"], 1)
        self.assertAlmostEqual(result["Rank-1"], 1.0)
        self.assertAlmostEqual(result["mAP"], 1.0)
        self.assertAlmostEqual(result["mINP"], 1.0)

    def test_self_match_string_sample_ids(self):
        g = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float64)
        g_ids = np.array([0, 1, 0], dtype=np.int64)
        g_sids = np.array(["a", "b", "c"])
        q = np.array([[1, 0]], dtype=np.float64)
        q_ids = np.array([0], dtype=np.int64)
        q_sids = np.array(["a"])
        result = compute_retrieval_metrics(
            q, g, q_ids, g_ids, rank_ks=(1,),
            query_sample_ids=q_sids, gallery_sample_ids=g_sids,
        )
        self.assertAlmostEqual(result["Rank-1"], 1.0)

    def test_self_match_all_relevant_removed_raises(self):
        g = np.array([[1, 0]], dtype=np.float64)
        g_ids = np.array([0], dtype=np.int64)
        g_sids = np.array(["s1"])
        q = np.array([[1, 0]], dtype=np.float64)
        q_ids = np.array([0], dtype=np.int64)
        q_sids = np.array(["s1"])
        with self.assertRaises(ClosedSetViolation):
            compute_retrieval_metrics(
                q, g, q_ids, g_ids, closed_set=True,
                query_sample_ids=q_sids, gallery_sample_ids=g_sids,
            )

    def test_duplicate_sample_ids_rejected(self):
        g = np.eye(2, dtype=np.float64)
        g_ids = np.array([0, 1], dtype=np.int64)
        g_sids = np.array(["a", "a"])
        q = np.array([[1, 0]], dtype=np.float64)
        q_ids = np.array([0], dtype=np.int64)
        q_sids = np.array(["b"])
        with self.assertRaises(RetrievalError):
            compute_retrieval_metrics(
                q, g, q_ids, g_ids,
                query_sample_ids=q_sids, gallery_sample_ids=g_sids,
            )

    def test_missing_identity_rejected_in_closed_set(self):
        with self.assertRaises(ClosedSetViolation):
            compute_retrieval_metrics(
                np.array([[1, 0, 0]], dtype=np.float64),
                GALLERY_D, np.array([999], dtype=np.int64), GALLERY_IDS_D,
            )

    def test_configurable_rank_ks(self):
        result = compute_retrieval_metrics(QUERY_C, GALLERY_C, QUERY_IDS_C, GALLERY_IDS_C, rank_ks=(1, 3))
        self.assertIn("Rank-1", result)
        self.assertIn("Rank-3", result)
        self.assertNotIn("Rank-5", result)

    def test_empty_queries_rejected(self):
        with self.assertRaises(RetrievalError):
            compute_retrieval_metrics(
                np.empty((0, 3), dtype=np.float64),
                GALLERY_D, np.array([], dtype=np.int64), GALLERY_IDS_D,
            )

    def test_non_finite_embeddings_rejected(self):
        with self.assertRaises(RetrievalError):
            compute_retrieval_metrics(
                np.array([[1, np.nan, 0]], dtype=np.float64),
                GALLERY_D, np.array([0], dtype=np.int64), GALLERY_IDS_D,
            )

    def test_string_ids_accepted(self):
        g = np.eye(2, dtype=np.float64)
        g_ids = np.array(["a", "b"])
        q = np.array([[1, 0]], dtype=np.float64)
        q_ids = np.array(["a"])
        result = compute_retrieval_metrics(q, g, q_ids, g_ids, rank_ks=(1,))
        self.assertAlmostEqual(result["Rank-1"], 1.0)

    def test_unsupported_metric_rejected(self):
        with self.assertRaises(RetrievalError):
            compute_retrieval_metrics(QUERY_D, GALLERY_D, QUERY_IDS_D, GALLERY_IDS_D, metric="dot")

    def test_mINP_2_relevant_last_at_rank_4(self):
        g = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [0.5, 0.5]], dtype=np.float64)
        g_ids = np.array([0, 1, 2, 0], dtype=np.int64)
        q = np.array([[1, 0]], dtype=np.float64)
        q_ids = np.array([0], dtype=np.int64)
        result = compute_retrieval_metrics(q, g, q_ids, g_ids)
        self.assertAlmostEqual(result["mINP"], 0.5)

    def test_mINP_1_relevant_last_at_rank_3(self):
        g = np.array([[0.9, 0.0], [0.8, 0.0], [-0.5, 0.0]], dtype=np.float64)
        g_ids = np.array([1, 2, 0], dtype=np.int64)
        q = np.array([[1, 0]], dtype=np.float64)
        q_ids = np.array([0], dtype=np.int64)
        result = compute_retrieval_metrics(q, g, q_ids, g_ids)
        self.assertAlmostEqual(result["mINP"], 1.0 / 3.0, places=5)

    def test_mINP_all_relevant_first(self):
        g = np.array([[1, 0], [0, 1]], dtype=np.float64)
        g_ids = np.array([0, 1], dtype=np.int64)
        q = np.array([[1, 0]], dtype=np.float64)
        q_ids = np.array([0], dtype=np.int64)
        result = compute_retrieval_metrics(q, g, q_ids, g_ids)
        self.assertAlmostEqual(result["mINP"], 1.0)

    def test_mINP_never_exceeds_one(self):
        g = np.eye(5, dtype=np.float64)
        g_ids = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        q = np.eye(5, dtype=np.float64)
        q_ids = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        result = compute_retrieval_metrics(q, g, q_ids, g_ids)
        self.assertLessEqual(result["mINP"], 1.0)
        self.assertGreater(result["mINP"], 0.0)

    def test_mINP_no_clipping_raises_on_broken_invariant(self):
        ranked_pos = np.array([True, False, False], dtype=bool)
        from cvi.evaluation.retrieval import _compute_ap_inp
        with self.assertRaises(MetricInvariantError):
            _compute_ap_inp(ranked_pos, n_relevant=5)


class MultiTemplateClosedSetTest(unittest.TestCase):
    def test_frozen_max_aggregation_precedes_distinct_identity_ranking(self):
        result = evaluate_multi_template_closed_set(
            np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float64),
            np.array(["C"]),
            np.array(["A", "A", "C", "B"]),
            self_match_policy="include",
            rank_ks=(1, 2),
        )

        self.assertEqual(result["gallery_identity_order"], ["A", "C", "B"])
        self.assertEqual(result["query_rows"][0]["relevant_rank"], 2)
        self.assertEqual(result["Rank-1"], 0.0)
        self.assertEqual(result["Rank-2"], 1.0)

    def test_max_not_mean_is_the_frozen_identity_aggregation(self):
        result = evaluate_multi_template_closed_set(
            np.array([[0.9, 0.1, 0.8]], dtype=np.float64),
            np.array(["A"]),
            np.array(["A", "A", "B"]),
            self_match_policy="include",
            rank_ks=(1,),
        )

        self.assertEqual(result["aggregation"], "max")
        self.assertEqual(result["Rank-1"], 1.0)

    def test_identity_score_ties_keep_first_gallery_occurrence(self):
        result = evaluate_multi_template_closed_set(
            np.array([[0.8, 0.8, 0.7, 0.1]], dtype=np.float64),
            np.array(["A"]),
            np.array(["B", "A", "B", "C"]),
            self_match_policy="include",
            rank_ks=(1, 2),
        )

        self.assertEqual(result["gallery_identity_order"], ["B", "A", "C"])
        self.assertEqual(result["query_rows"][0]["relevant_rank"], 2)
        self.assertEqual(result["Rank-1"], 0.0)
        self.assertEqual(result["Rank-2"], 1.0)

    def test_per_query_rows_have_exact_identity_level_metrics(self):
        result = evaluate_multi_template_closed_set(
            np.array(
                [
                    [0.9, 0.8, 0.7],
                    [0.6, 0.9, 0.8],
                    [0.9, 0.8, 0.7],
                ],
                dtype=np.float64,
            ),
            np.array(["A", "B", "C"]),
            np.array(["A", "B", "C"]),
            self_match_policy="include",
            rank_ks=(1, 2, 3),
        )

        rows = result["query_rows"]
        self.assertEqual([row["relevant_rank"] for row in rows], [1, 1, 3])
        self.assertEqual(
            [row["bootstrap_cluster_id"] for row in rows], ["A", "B", "C"]
        )
        for row, expected in zip(rows, (1.0, 1.0, 1.0 / 3.0), strict=True):
            self.assertAlmostEqual(row["AP"], expected)
            self.assertAlmostEqual(row["INP"], expected)
            self.assertAlmostEqual(row["reciprocal_rank"], expected)
        self.assertAlmostEqual(result["mAP"], 7.0 / 9.0)
        self.assertAlmostEqual(result["mINP"], 7.0 / 9.0)
        self.assertAlmostEqual(result["MRR"], 7.0 / 9.0)

    def test_explicit_self_match_exclusion_happens_before_max(self):
        kwargs = {
            "query_template_scores": np.array([[0.99, 0.6, 0.8]]),
            "query_identity_ids": np.array(["A"]),
            "gallery_template_identity_ids": np.array(["A", "A", "B"]),
            "query_template_ids": np.array(["same"]),
            "gallery_template_ids": np.array(["same", "other-a", "other-b"]),
            "rank_ks": (1, 2),
        }
        included = evaluate_multi_template_closed_set(
            **kwargs, self_match_policy="include"
        )
        excluded = evaluate_multi_template_closed_set(
            **kwargs, self_match_policy="exclude"
        )

        self.assertEqual(included["query_rows"][0]["relevant_rank"], 1)
        self.assertEqual(excluded["query_rows"][0]["relevant_rank"], 2)

    def test_self_match_exclusion_requires_template_ids(self):
        with self.assertRaises(SampleIdValidationError):
            evaluate_multi_template_closed_set(
                np.array([[1.0]]),
                np.array(["A"]),
                np.array(["A"]),
                self_match_policy="exclude",
            )

    def test_shared_template_id_requires_consistent_identity_even_when_included(self):
        with self.assertRaises(SampleIdValidationError):
            evaluate_multi_template_closed_set(
                np.array([[1.0, 0.5]]),
                np.array(["A"]),
                np.array(["B", "A"]),
                self_match_policy="include",
                query_template_ids=np.array(["same"]),
                gallery_template_ids=np.array(["same", "other"]),
            )

    def test_excluding_only_relevant_template_violates_closed_set(self):
        with self.assertRaises(ClosedSetViolation):
            evaluate_multi_template_closed_set(
                np.array([[1.0, 0.5]]),
                np.array(["A"]),
                np.array(["A", "B"]),
                self_match_policy="exclude",
                query_template_ids=np.array(["same"]),
                gallery_template_ids=np.array(["same", "other"]),
            )

    def test_identity_clustered_bootstrap_is_deterministic(self):
        rows = (
            {"bootstrap_cluster_id": "A", "AP": 0.0},
            {"bootstrap_cluster_id": "A", "AP": 0.0},
            {"bootstrap_cluster_id": "B", "AP": 1.0},
        )
        first = identity_clustered_bootstrap_ci(
            rows, metric="AP", confidence_level=0.8, resamples=200, seed=17
        )
        second = identity_clustered_bootstrap_ci(
            rows, metric="AP", confidence_level=0.8, resamples=200, seed=17
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 1.0 / 3.0)
        self.assertEqual(first["cluster_unit"], "query_identity")
        self.assertEqual(first["cluster_count"], 2)
        self.assertEqual(first["query_row_count"], 3)
        self.assertEqual(first["lower_bound"], 0.0)
        self.assertEqual(first["upper_bound"], 1.0)

    def test_identity_clustered_bootstrap_rejects_missing_metric(self):
        with self.assertRaisesRegex(RetrievalError, "missing metric"):
            identity_clustered_bootstrap_ci(
                (
                    {"bootstrap_cluster_id": "A", "AP": 1.0},
                    {"bootstrap_cluster_id": "B", "INP": 1.0},
                ),
                metric="AP",
                resamples=10,
            )


if __name__ == "__main__":
    unittest.main()
