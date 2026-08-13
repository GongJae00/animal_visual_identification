from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from systems.measurement.decode import (
    DecodeBackend,
    DecodeConfig,
    DecodeRun,
    benchmark_decode,
    build_decode_command,
    parse_ffmpeg_benchmark,
)


class DecodeTests(unittest.TestCase):
    def test_ffmpeg_benchmark_parser_preserves_resource_units(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "ffmpeg_benchmark.txt"
        run = parse_ffmpeg_benchmark(
            fixture.read_text(encoding="utf-8"),
            wall_time_ns=20_000_000,
        )
        self.assertEqual(run.decoded_frames, 60)
        self.assertEqual(run.output_time_seconds, 1.96)
        self.assertEqual(run.reported_speed_x, 137.0)
        self.assertEqual(run.process_max_rss_bytes, 84_104 * 1024)
        self.assertEqual(run.wall_time_ns, 20_000_000)

    def test_cpu_command_has_no_hardware_assumption(self) -> None:
        command = build_decode_command(
            Path("/protected/source.mp4"),
            DecodeConfig(DecodeBackend.CPU, 60.0, threads=4),
        )
        self.assertNotIn("-hwaccel", command)
        self.assertIn("-threads", command)

    def test_cuda_command_is_explicit_and_guarded(self) -> None:
        command = build_decode_command(
            Path("/protected/source.mp4"),
            DecodeConfig(DecodeBackend.CUDA, 60.0),
        )
        self.assertIn("-hwaccel", command)
        self.assertIn("cuda", command)
        self.assertIn("-hwaccel_output_format", command)

    def test_gpu_telemetry_is_part_of_hashed_decode_config(self) -> None:
        config = DecodeConfig(
            DecodeBackend.CUDA,
            60.0,
            gpu_device_index=0,
            gpu_telemetry_interval_seconds=0.5,
        )
        self.assertEqual(config.to_dict()["gpu_device_index"], 0)
        self.assertEqual(
            config.to_dict()["gpu_telemetry_interval_seconds"], 0.5
        )
        self.assertEqual(len(config.config_sha256), 64)

    def test_gpu_telemetry_cannot_be_silently_attached_to_cpu(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA"):
            DecodeConfig(
                DecodeBackend.CPU,
                60.0,
                gpu_device_index=0,
                gpu_telemetry_interval_seconds=0.5,
            )

    def test_gpu_telemetry_arguments_must_be_paired(self) -> None:
        with self.assertRaisesRegex(ValueError, "set together"):
            DecodeConfig(
                DecodeBackend.CUDA,
                60.0,
                gpu_device_index=0,
            )

    def test_incomplete_benchmark_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            parse_ffmpeg_benchmark("frame=10", wall_time_ns=1)

    def test_gpu_telemetry_requires_injected_monitor(self) -> None:
        source = Path(self.id().replace(".", "-"))
        source.write_bytes(b"video")
        self.addCleanup(source.unlink, missing_ok=True)
        with self.assertRaisesRegex(ValueError, "injected telemetry monitor"):
            benchmark_decode(
                source,
                source_id="fixture",
                source_sha256=hashlib.sha256(b"video").hexdigest(),
                config=DecodeConfig(
                    DecodeBackend.CUDA,
                    1.0,
                    gpu_device_index=2,
                    gpu_telemetry_interval_seconds=0.5,
                ),
                warmup_runs=0,
                repeat_runs=1,
                unrelated_gpu_work_excluded_by_operator=True,
            )

    def test_injected_gpu_monitor_preserves_summary(self) -> None:
        source = Path(self.id().replace(".", "-"))
        source.write_bytes(b"video")
        self.addCleanup(source.unlink, missing_ok=True)
        seen: dict[str, object] = {}
        telemetry = SimpleNamespace(to_dict=lambda: {"scope": "fixture"})

        def monitor(operation, *, device_index: int, interval_seconds: float):
            seen.update(device_index=device_index, interval_seconds=interval_seconds)
            return operation(), telemetry

        run = DecodeRun(30, 1.0, 1.0, 0.1, 0.1, 1.0, 1024, 1_000_000)
        with (
            patch(
                "systems.measurement.decode.run_decode_once",
                return_value=(run, ("ffmpeg",)),
            ),
            patch(
                "systems.measurement.decode.probe_video_file",
                return_value=SimpleNamespace(average_fps=30.0),
            ),
            patch(
                "systems.measurement.decode.subprocess.run",
                return_value=SimpleNamespace(stdout="ffmpeg version fixture\n"),
            ),
        ):
            summary = benchmark_decode(
                source,
                source_id="fixture",
                source_sha256=hashlib.sha256(b"video").hexdigest(),
                config=DecodeConfig(
                    DecodeBackend.CUDA,
                    1.0,
                    gpu_device_index=2,
                    gpu_telemetry_interval_seconds=0.5,
                ),
                warmup_runs=0,
                repeat_runs=1,
                unrelated_gpu_work_excluded_by_operator=True,
                telemetry_monitor=monitor,
            )

        self.assertEqual(seen, {"device_index": 2, "interval_seconds": 0.5})
        self.assertEqual(summary.to_dict()["gpu_telemetry"], {"scope": "fixture"})


if __name__ == "__main__":
    unittest.main()
