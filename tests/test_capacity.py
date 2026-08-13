from __future__ import annotations

import unittest

from systems.measurement.capacity import (
    CapacityPlan,
    ComputeResource,
    MemoryComponent,
    ResourceCapacity,
    StageRateBudget,
    adaptive_call_reduction_fraction,
    peak_memory_bytes,
)


class CapacityTests(unittest.TestCase):
    def test_expected_calls_use_state_mix_not_nominal_peak_rate(self) -> None:
        plan = CapacityPlan(
            camera_count=4,
            occupied_fraction=0.25,
            stages=(
                StageRateBudget(
                    "detector",
                    ComputeResource.GPU,
                    idle_calls_per_stream_second=1.0,
                    occupied_calls_per_stream_second=5.0,
                    service_seconds_per_call=0.01,
                ),
            ),
            capacities=(
                ResourceCapacity(ComputeResource.GPU, 1.0, 0.7),
            ),
        )
        self.assertEqual(plan.expected_stage_calls(100)["detector"], 800.0)
        self.assertEqual(plan.peak_state_stage_calls(100)["detector"], 2000.0)

    def test_resource_load_preserves_expected_and_peak_state_limits(self) -> None:
        plan = CapacityPlan(
            camera_count=4,
            occupied_fraction=0.25,
            stages=(
                StageRateBudget(
                    "detector",
                    ComputeResource.GPU,
                    1.0,
                    5.0,
                    0.01,
                ),
            ),
            capacities=(
                ResourceCapacity(ComputeResource.GPU, 1.0, 0.7),
            ),
        )
        load = plan.resource_loads()[0]
        self.assertAlmostEqual(load.expected_utilization, 0.08)
        self.assertAlmostEqual(load.peak_state_utilization, 0.2)
        self.assertEqual(load.maximum_cameras_at_expected_mix, 35)
        self.assertEqual(load.maximum_cameras_at_peak_state, 14)
        self.assertTrue(load.peak_state_within_target)

    def test_adaptive_reduction_is_relative_to_fixed_peak_schedule(self) -> None:
        reduction = adaptive_call_reduction_fraction(
            idle_calls_per_second=1.0,
            occupied_calls_per_second=5.0,
            occupied_fraction=0.25,
        )
        self.assertAlmostEqual(reduction, 0.6)

    def test_memory_terms_are_classified_and_scaled_once(self) -> None:
        total = peak_memory_bytes(
            (
                MemoryComponent(
                    "model",
                    shared_bytes=100,
                    workspace_bytes_per_replica=30,
                    workspace_replicas=2,
                ),
                MemoryComponent(
                    "stream-state",
                    per_stream_bytes=10,
                    per_active_track_bytes=5,
                    per_batch_item_bytes=2,
                ),
            ),
            stream_count=4,
            active_tracks=3,
            batch_items=8,
        )
        self.assertEqual(total, 231)

    def test_missing_resource_capacity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing capacity"):
            CapacityPlan(
                camera_count=1,
                occupied_fraction=0.5,
                stages=(
                    StageRateBudget(
                        "decode",
                        ComputeResource.VIDEO_DECODE,
                        30.0,
                        30.0,
                        0.001,
                    ),
                ),
                capacities=(),
            )

    def test_capacity_config_hash_covers_stage_rates(self) -> None:
        plan = CapacityPlan(
            camera_count=1,
            occupied_fraction=0.5,
            stages=(
                StageRateBudget(
                    "decode",
                    ComputeResource.VIDEO_DECODE,
                    30.0,
                    30.0,
                    0.001,
                ),
            ),
            capacities=(
                ResourceCapacity(ComputeResource.VIDEO_DECODE, 1.0, 0.7),
            ),
        )
        self.assertEqual(len(plan.config_sha256), 64)
        self.assertEqual(
            CapacityPlan.from_dict(plan.to_dict()).config_sha256,
            plan.config_sha256,
        )

    def test_capacity_parser_rejects_unknown_fields(self) -> None:
        payload = {
            "schema_version": "cvi.capacity_plan.v1",
            "camera_count": 1,
            "occupied_fraction": 0.5,
            "stages": [],
            "capacities": [],
            "nominal_fps_shortcut": 30,
        }
        with self.assertRaisesRegex(ValueError, "unknown"):
            CapacityPlan.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
