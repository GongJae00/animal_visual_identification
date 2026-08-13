from __future__ import annotations

import unittest

from systems.measurement.telemetry import (
    GpuTelemetryAccumulator,
    GpuTelemetrySample,
    parse_nvidia_smi_csv,
)


def sample(timestamp_ns: int, power_w: float) -> GpuTelemetrySample:
    return GpuTelemetrySample(
        timestamp_ns=timestamp_ns,
        device_index=0,
        device_name="NVIDIA GeForce RTX 5080",
        driver_version="610.62",
        memory_used_mib=1000.0 + timestamp_ns / 1_000_000_000,
        power_draw_w=power_w,
        power_limit_w=360.0,
        gpu_utilization_pct=50.0,
        memory_utilization_pct=10.0,
        decoder_utilization_pct=25.0,
    )


class TelemetryTests(unittest.TestCase):
    def test_csv_parser_preserves_device_wide_metrics(self) -> None:
        parsed = parse_nvidia_smi_csv(
            "0, NVIDIA GeForce RTX 5080, 610.62, 1574, 46.67, 360.00, 3, 1, 0",
            timestamp_ns=123,
        )
        self.assertEqual(parsed.device_index, 0)
        self.assertEqual(parsed.memory_used_mib, 1574.0)
        self.assertEqual(parsed.power_draw_w, 46.67)
        self.assertEqual(parsed.decoder_utilization_pct, 0.0)

    def test_unsupported_metrics_remain_missing(self) -> None:
        parsed = parse_nvidia_smi_csv(
            "0, GPU, 1.0, N/A, [Not Supported], 100, 0, 0, N/A",
            timestamp_ns=123,
        )
        self.assertIsNone(parsed.memory_used_mib)
        self.assertIsNone(parsed.power_draw_w)
        self.assertIsNone(parsed.decoder_utilization_pct)

    def test_accumulator_integrates_energy_in_constant_state(self) -> None:
        accumulator = GpuTelemetryAccumulator(0.5)
        accumulator.add(sample(0, 40.0))
        accumulator.add(sample(1_000_000_000, 60.0))
        summary = accumulator.finalize()
        self.assertEqual(summary.samples, 2)
        self.assertEqual(summary.sampled_span_seconds, 1.0)
        self.assertEqual(summary.effective_mean_interval_seconds, 1.0)
        self.assertEqual(summary.power_draw_w_mean, 50.0)
        self.assertEqual(summary.power_draw_w_max, 60.0)
        self.assertEqual(summary.sampled_board_energy_joules, 50.0)
        self.assertEqual(summary.scope, "device-wide")
        self.assertEqual(summary.sampler_backend, "injected")


if __name__ == "__main__":
    unittest.main()
