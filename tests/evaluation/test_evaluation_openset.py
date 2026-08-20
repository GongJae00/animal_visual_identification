from __future__ import annotations

import unittest

import numpy as np

from evaluation.open_set import OpenSetError, evaluate_open_set

GALLERY = np.array([[1, 0], [0, 1]], dtype=np.float64)
GALLERY_IDS = np.array([0, 1], dtype=np.int64)
QUERIES = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float64)
QUERY_IDS = np.array([0, 1, 999], dtype=np.int64)

GALLERY_MIS = np.array([[1, 0], [0, 1]], dtype=np.float64)
GALLERY_IDS_MIS = np.array([0, 1], dtype=np.int64)
QUERIES_MIS = np.array([[0, 1], [1, -1]], dtype=np.float64)
QUERY_IDS_MIS = np.array([0, 999], dtype=np.int64)

CAL_GALLERY = np.array([[1, 0], [0, 1]], dtype=np.float64)
CAL_GALLERY_IDS = np.array([10, 11], dtype=np.int64)
CAL_QUERIES = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float64)
CAL_QUERY_IDS = np.array([10, 11, 9999], dtype=np.int64)

def _calibration_kwargs():
    return {
        "calibration_query_embs": CAL_QUERIES,
        "calibration_gallery_embs": CAL_GALLERY,
        "calibration_query_ids": CAL_QUERY_IDS,
        "calibration_gallery_ids": CAL_GALLERY_IDS,
    }

class OpenSetTest(unittest.TestCase):
    def test_known_values(self):
        result = evaluate_open_set(
            QUERIES, GALLERY, QUERY_IDS, GALLERY_IDS, **_calibration_kwargs()
        )
        self.assertAlmostEqual(result.known_detection_auroc, 1.0)
        self.assertAlmostEqual(result.known_detection_aupr, 1.0)
        self.assertAlmostEqual(result.dir_at_fpir["DIR@FPIR=0.01"], 1.0)
        self.assertAlmostEqual(result.dir_at_fpir["DIR@FPIR=0.001"], 1.0)
        self.assertEqual(result.num_enrolled_queries, 2)
        self.assertEqual(result.num_unknown_queries, 1)

    def test_misidentified_known_query_gives_zero_dir(self):
        result = evaluate_open_set(
            QUERIES_MIS, GALLERY_MIS, QUERY_IDS_MIS, GALLERY_IDS_MIS,
            **_calibration_kwargs(),
        )
        self.assertEqual(result.num_enrolled_queries, 1)
        self.assertEqual(result.num_unknown_queries, 1)
        self.assertAlmostEqual(result.dir_at_fpir["DIR@FPIR=0.01"], 0.0)
        self.assertEqual(result.known_misidentification_count, 1)

    def test_no_unknown_rejected(self):
        with self.assertRaises(OpenSetError):
            evaluate_open_set(
                QUERIES[:2], GALLERY, QUERY_IDS[:2], GALLERY_IDS,
                **_calibration_kwargs(),
            )

    def test_empty_query_rejected(self):
        with self.assertRaises(OpenSetError):
            evaluate_open_set(
                np.empty((0, 2), dtype=np.float64),
                GALLERY, np.array([], dtype=np.int64), GALLERY_IDS,
                **_calibration_kwargs(),
            )

    def test_zero_norm_embedding_rejected(self):
        with self.assertRaises(OpenSetError):
            evaluate_open_set(
                np.array([[0, 0]], dtype=np.float64),
                GALLERY, np.array([999], dtype=np.int64), GALLERY_IDS,
                **_calibration_kwargs(),
            )

    def test_fpir_thresholds_in_result(self):
        result = evaluate_open_set(
            QUERIES, GALLERY, QUERY_IDS, GALLERY_IDS, **_calibration_kwargs()
        )
        self.assertIn("threshold@FPIR=0.01", result.fpir_thresholds)

    def test_per_target_structure(self):
        result = evaluate_open_set(
            QUERIES, GALLERY, QUERY_IDS, GALLERY_IDS, **_calibration_kwargs()
        )
        for target_key, info in result.per_target.items():
            self.assertIn("selected_threshold", info)
            self.assertIn("test", info)
            t = info["test"]
            for f in ("known_queries", "unknown_queries", "DIR", "FPIR"):
                self.assertIn(f, t)

    def test_calibration_test_separate(self):
        cal_q = np.array([[1, 0], [-1, 0]], dtype=np.float64)
        cal_q_ids = np.array([10, 9999], dtype=np.int64)
        cal_g = np.array([[1, 0]], dtype=np.float64)
        cal_g_ids = np.array([10], dtype=np.int64)
        test_q = np.array([[1, 0], [0, 1], [-0.5, 0]], dtype=np.float64)
        test_q_ids = np.array([0, 1, 998], dtype=np.int64)
        test_g = np.array([[1, 0], [0, 1]], dtype=np.float64)
        test_g_ids = np.array([0, 1], dtype=np.int64)
        result = evaluate_open_set(
            test_q, test_g, test_q_ids, test_g_ids,
            fpir_targets=(0.01,),
            calibration_query_embs=cal_q,
            calibration_gallery_embs=cal_g,
            calibration_query_ids=cal_q_ids,
            calibration_gallery_ids=cal_g_ids,
        )
        self.assertGreaterEqual(result.num_enrolled_queries, 1)
        self.assertGreaterEqual(result.num_unknown_queries, 1)
        for info in result.per_target.values():
            self.assertIn("calibration", info)
            self.assertIn("test", info)
            self.assertIn("selected_threshold", info)

    def test_calibration_threshold_frozen(self):
        cal_g = np.array([[1, 0], [0, 1]], dtype=np.float64)
        cal_g_ids = np.array([10, 11], dtype=np.int64)
        cal_q = np.array([[1, 0], [-1, 0]], dtype=np.float64)
        cal_q_ids = np.array([10, 9999], dtype=np.int64)
        test_q1 = np.array([[1, 0], [0, 1], [-0.9, 0]], dtype=np.float64)
        test_q1_ids = np.array([0, 1, 998], dtype=np.int64)
        test_q2 = np.array([[1, 0], [0, 1], [0.9, 0]], dtype=np.float64)
        test_q2_ids = np.array([0, 1, 998], dtype=np.int64)
        test_g = np.array([[1, 0], [0, 1]], dtype=np.float64)
        test_g_ids = np.array([0, 1], dtype=np.int64)
        r1 = evaluate_open_set(
            test_q1, test_g, test_q1_ids, test_g_ids,
            fpir_targets=(0.01,),
            calibration_query_embs=cal_q, calibration_gallery_embs=cal_g,
            calibration_query_ids=cal_q_ids, calibration_gallery_ids=cal_g_ids,
        )
        r2 = evaluate_open_set(
            test_q2, test_g, test_q2_ids, test_g_ids,
            fpir_targets=(0.01,),
            calibration_query_embs=cal_q, calibration_gallery_embs=cal_g,
            calibration_query_ids=cal_q_ids, calibration_gallery_ids=cal_g_ids,
        )
        t1 = r1.per_target["0.01"]["selected_threshold"]
        t2 = r2.per_target["0.01"]["selected_threshold"]
        self.assertEqual(t1, t2)

    def test_test_fallback_calibration_is_rejected(self):
        with self.assertRaisesRegex(OpenSetError, "separate calibration"):
            evaluate_open_set(QUERIES, GALLERY, QUERY_IDS, GALLERY_IDS)

    def test_calibration_and_test_identities_must_be_disjoint(self):
        with self.assertRaisesRegex(OpenSetError, "disjoint"):
            evaluate_open_set(
                QUERIES,
                GALLERY,
                QUERY_IDS,
                GALLERY_IDS,
                calibration_query_embs=QUERIES,
                calibration_gallery_embs=GALLERY,
                calibration_query_ids=QUERY_IDS,
                calibration_gallery_ids=GALLERY_IDS,
            )

    def test_tied_unknown_scores_do_not_understate_fpir(self):
        cal_gallery = np.array([[1.0, 0.0], [0.0, 1.0]])
        cal_gallery_ids = np.array([10, 11])
        cal_queries = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        cal_query_ids = np.array([10, 11, 98, 99])
        result = evaluate_open_set(
            QUERIES,
            GALLERY,
            QUERY_IDS,
            GALLERY_IDS,
            fpir_targets=(0.5,),
            calibration_query_embs=cal_queries,
            calibration_gallery_embs=cal_gallery,
            calibration_query_ids=cal_query_ids,
            calibration_gallery_ids=cal_gallery_ids,
        )
        calibration = result.per_target["0.5"]["calibration"]
        self.assertLessEqual(calibration["FPIR"], 0.5)
        self.assertEqual(calibration["unknown_accepts"], 0)

if __name__ == "__main__":
    unittest.main()
