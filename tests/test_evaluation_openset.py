from __future__ import annotations

import unittest

import numpy as np

from cvi.evaluation.open_set import (
    evaluate_open_set,
    OpenSetError,
    OpenSetResult,
)

# Fixture:
# gallery: [1,0](id=0), [0,1](id=1)
# known queries: [1,0](id=0), [0,1](id=1) -> max sims [1, 1]
# unknown query: [-1,0](id=999) -> max sim [-1 dot 1, -1 dot 0, ...] = 0
# AUROC: known[0]=1>0, known[1]=1>0 => (1+1)/2 = 1.0
# AUPR: labels [1,1,0] scores [1,1,0] => AP=1.0
# DIR@FPIR=0.01: t=1.0, FAR=0, TPR=2/2=1.0
# DIR@FPIR=0.001: t=1.0, FAR=0, TPR=2/2=1.0
# top1 ids: q0->0, q1->1, q2->0 (cos -1>0 with [1,0])
# false_accept: 1 (q2 matched to id 0, != 999)
# false_reject: 0
GALLERY = np.array([[1, 0], [0, 1]], dtype=np.float64)
GALLERY_IDS = np.array([0, 1], dtype=np.int64)
QUERIES = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float64)
QUERY_IDS = np.array([0, 1, 999], dtype=np.int64)


class OpenSetTest(unittest.TestCase):
    def test_known_values(self):
        result = evaluate_open_set(
            QUERIES, GALLERY, QUERY_IDS, GALLERY_IDS,
            closed_set_no_match_policy="allow",
        )
        self.assertAlmostEqual(result.known_vs_unknown_auroc, 1.0)
        self.assertAlmostEqual(result.known_vs_unknown_aupr, 1.0)
        self.assertAlmostEqual(result.dir_at_fpir["DIR@FPIR=1e-02"], 1.0)
        self.assertAlmostEqual(result.dir_at_fpir["DIR@FPIR=1e-03"], 1.0)
        self.assertEqual(result.false_accept_count, 1)
        self.assertEqual(result.false_reject_count, 0)
        self.assertEqual(result.num_enrolled_queries, 2)
        self.assertEqual(result.num_unknown_queries, 1)

    def test_no_unknown_rejected(self):
        q = QUERIES[:2]
        q_ids = QUERY_IDS[:2]
        with self.assertRaises(OpenSetError):
            evaluate_open_set(q, GALLERY, q_ids, GALLERY_IDS)

    def test_empty_query_rejected(self):
        with self.assertRaises(OpenSetError):
            evaluate_open_set(
                np.empty((0, 2), dtype=np.float64),
                GALLERY, np.array([], dtype=np.int64), GALLERY_IDS,
            )

    def test_zero_norm_embedding_rejected(self):
        q = np.array([[0, 0]], dtype=np.float64)
        with self.assertRaises(OpenSetError):
            evaluate_open_set(
                q, GALLERY, np.array([999], dtype=np.int64), GALLERY_IDS,
            )

    def test_non_enrolled_query_rejected_in_closed_mode(self):
        q = np.array([[1, 0, 0]], dtype=np.float64)
        g = np.eye(3, dtype=np.float64)
        with self.assertRaises(OpenSetError):
            evaluate_open_set(
                q, g, np.array([999], dtype=np.int64), np.array([0, 1, 2], dtype=np.int64),
                closed_set_no_match_policy="error",
            )


if __name__ == "__main__":
    unittest.main()
