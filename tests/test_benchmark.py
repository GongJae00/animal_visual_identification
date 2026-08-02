from __future__ import annotations

import unittest

from evaluation.benchmark import (
    BenchmarkReceipt,
    MetricInterval,
    TimingSummary,
    measure_operation,
)
from representation_learning.optimization import PromotionDecision


class BenchmarkTests(unittest.TestCase):
    def test_nearest_rank_timing_summary(self) -> None:
        summary = TimingSummary.from_samples((10, 40, 20, 30, 100))
        self.assertEqual(summary.samples, 5)
        self.assertEqual(summary.minimum_ns, 10)
        self.assertEqual(summary.p50_ns, 30)
        self.assertEqual(summary.p95_ns, 100)
        self.assertEqual(summary.maximum_ns, 100)
        self.assertEqual(summary.mean_ns, 40.0)

    def test_measurement_calls_warmup_repeat_and_sync(self) -> None:
        operation_calls = 0
        synchronization_calls = 0

        def operation() -> int:
            nonlocal operation_calls
            operation_calls += 1
            return operation_calls

        def synchronize() -> None:
            nonlocal synchronization_calls
            synchronization_calls += 1

        summary, result = measure_operation(
            operation,
            warmup=2,
            repeats=3,
            synchronize=synchronize,
        )
        self.assertEqual(operation_calls, 5)
        self.assertEqual(synchronization_calls, 7)
        self.assertEqual(summary.samples, 3)
        self.assertEqual(result, 5)

    def test_receipt_hash_is_stable_across_mapping_order(self) -> None:
        timing = TimingSummary.from_samples((100, 110, 120))
        safety = (MetricInterval("FPIR", 0.001, 0.0008, 0.0012, "ratio", "paired"),)
        resources = (
            MetricInterval("p95 latency", 2.0, 1.8, 2.2, "ms", "bootstrap"),
        )
        common = {
            "reference_config_sha256": "ref",
            "candidate_config_sha256": "candidate",
            "code_revision": "revision",
            "dependency_lock_sha256": "lock",
            "dataset_manifest_sha256": "manifest",
            "split_sha256": "split",
            "calibration_role": "calibration-only",
            "test_role": "frozen-test",
            "warmup_iterations": 5,
            "repeat_iterations": 20,
            "timing": timing,
            "safety_metrics": safety,
            "resource_metrics": resources,
            "decision": PromotionDecision.INCONCLUSIVE,
        }
        left = BenchmarkReceipt(
            hardware={"gpu": "RTX 5080", "driver": "610.62"},
            workload={"streams": 3, "codec": "h264"},
            **common,
        )
        right = BenchmarkReceipt(
            hardware={"driver": "610.62", "gpu": "RTX 5080"},
            workload={"codec": "h264", "streams": 3},
            **common,
        )
        self.assertEqual(left.receipt_sha256, right.receipt_sha256)

    def test_receipt_requires_both_safety_and_resource_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "safety and resource"):
            BenchmarkReceipt(
                reference_config_sha256="ref",
                candidate_config_sha256="candidate",
                code_revision="revision",
                dependency_lock_sha256="lock",
                dataset_manifest_sha256="manifest",
                split_sha256="split",
                calibration_role="calibration-only",
                test_role="frozen-test",
                hardware={"cpu": "test"},
                workload={"streams": 1},
                warmup_iterations=1,
                repeat_iterations=2,
                timing=TimingSummary.from_samples((1, 2)),
                safety_metrics=(),
                resource_metrics=(
                    MetricInterval("latency", 1.0, 0.9, 1.1, "ms", "paired"),
                ),
                decision=PromotionDecision.INCONCLUSIVE,
            )


if __name__ == "__main__":
    unittest.main()
