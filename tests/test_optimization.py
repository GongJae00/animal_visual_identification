from __future__ import annotations

import unittest

from representation_learning.optimization import (
    ImprovementMetric,
    PromotionDecision,
    ProtectedMetric,
    compute_cost,
    encoded_video_bytes,
    evaluate_promotion,
    gallery_bytes,
)
from foundation.provenance import content_sha256


class OptimizationTests(unittest.TestCase):
    def test_promote_requires_safety_and_strict_improvement(self) -> None:
        decision = evaluate_promotion(
            (
                ProtectedMetric("FPIR", 0.0001, -0.0001, 0.0003, 0.0005),
                ProtectedMetric(
                    "usable recall",
                    -0.001,
                    -0.003,
                    0.002,
                    0.005,
                ),
            ),
            (
                ImprovementMetric("p95 latency", -4.0, -2.0),
                ImprovementMetric("peak memory", -50.0, 5.0),
            ),
        )
        self.assertEqual(decision, PromotionDecision.PROMOTE)

    def test_safety_violation_rejects_even_when_cost_improves(self) -> None:
        decision = evaluate_promotion(
            (ProtectedMetric("FPIR", 0.001, 0.0008, 0.002, 0.0005),),
            (ImprovementMetric("p95 latency", -4.0, -2.0),),
        )
        self.assertEqual(decision, PromotionDecision.REJECT)

    def test_unresolved_safety_is_inconclusive_despite_cost_improvement(self) -> None:
        decision = evaluate_promotion(
            (ProtectedMetric("FPIR", 0.0004, -0.0002, 0.001, 0.0005),),
            (ImprovementMetric("p95 latency", -4.0, -2.0),),
        )
        self.assertEqual(decision, PromotionDecision.INCONCLUSIVE)

    def test_unresolved_cost_improvement_is_inconclusive(self) -> None:
        decision = evaluate_promotion(
            (ProtectedMetric("FPIR", 0.0, -0.0002, 0.0002, 0.0005),),
            (ImprovementMetric("p95 latency", -1.0, 0.5),),
        )
        self.assertEqual(decision, PromotionDecision.INCONCLUSIVE)

    def test_missing_metric_classes_are_inconclusive(self) -> None:
        self.assertEqual(
            evaluate_promotion((), (ImprovementMetric("latency", -1.0, -0.1),)),
            PromotionDecision.INCONCLUSIVE,
        )

    def test_compute_model_counts_call_rate_and_per_call_cost(self) -> None:
        total = compute_cost(
            decode_cost=10.0,
            detection_calls=4,
            detection_cost=3.0,
            tracking_calls=10,
            tracking_cost=0.5,
            quality_calls=6,
            quality_cost=0.25,
            embedding_calls=2,
            embedding_cost=4.0,
            search_calls=2,
            search_cost=0.1,
            aggregation_cost=0.3,
        )
        self.assertAlmostEqual(total, 37.0)

    def test_gallery_and_storage_formulae(self) -> None:
        self.assertEqual(gallery_bytes(10_000, 6, 512, 2), 61_440_000)
        self.assertEqual(encoded_video_bytes(4.0, 86_400), 43_200_000_000)

    def test_invalid_negative_cost_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            encoded_video_bytes(-1.0, 10.0)

    def test_config_hash_is_order_invariant(self) -> None:
        left = {"model": "d-fine-n", "shape": [640, 640], "fp16": True}
        right = {"fp16": True, "shape": [640, 640], "model": "d-fine-n"}
        self.assertEqual(content_sha256(left), content_sha256(right))


if __name__ == "__main__":
    unittest.main()
