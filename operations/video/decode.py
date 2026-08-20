"""Portable FFmpeg software/CUDA decode benchmark receipts."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns
from typing import Any, Protocol

from data.acquisition import probe_video_file, sha256_file
from shared.foundation.provenance import content_sha256
from shared.foundation.timing import TimingSummary


class TelemetrySummaryView(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


class TelemetryMonitor(Protocol):
    def __call__[T](
        self,
        operation: Callable[[], T],
        *,
        device_index: int,
        interval_seconds: float,
    ) -> tuple[T, TelemetrySummaryView]: ...

_FRAME_PATTERN = re.compile(
    r"frame=\s*(?P<frames>\d+).*?"
    r"time=(?P<time>\d+:\d+:\d+(?:\.\d+)?).*?"
    r"speed=\s*(?P<speed>[0-9.]+)x"
)
_TIME_PATTERN = re.compile(
    r"bench: utime=(?P<user>[0-9.]+)s "
    r"stime=(?P<system>[0-9.]+)s "
    r"rtime=(?P<real>[0-9.]+)s"
)
_RSS_PATTERN = re.compile(r"bench: maxrss=(?P<rss>\d+)kB")


class DecodeBackend(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    backend: DecodeBackend
    duration_seconds: float
    threads: int | None = None
    gpu_device_index: int | None = None
    gpu_telemetry_interval_seconds: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
        ):
            raise ValueError("duration_seconds must be finite and positive")
        if self.threads is not None and (
            isinstance(self.threads, bool)
            or not isinstance(self.threads, int)
            or self.threads <= 0
        ):
            raise ValueError("threads must be a positive integer")
        telemetry_values = (
            self.gpu_device_index,
            self.gpu_telemetry_interval_seconds,
        )
        if any(value is None for value in telemetry_values) and not all(
            value is None for value in telemetry_values
        ):
            raise ValueError(
                "gpu_device_index and gpu_telemetry_interval_seconds "
                "must be set together"
            )
        if self.gpu_device_index is not None:
            if self.backend is not DecodeBackend.CUDA:
                raise ValueError("GPU telemetry requires the CUDA decode backend")
            if (
                isinstance(self.gpu_device_index, bool)
                or not isinstance(self.gpu_device_index, int)
                or self.gpu_device_index < 0
            ):
                raise ValueError("gpu_device_index must be a non-negative integer")
            if (
                isinstance(self.gpu_telemetry_interval_seconds, bool)
                or not isinstance(
                    self.gpu_telemetry_interval_seconds, (int, float)
                )
                or not isfinite(self.gpu_telemetry_interval_seconds)
                or self.gpu_telemetry_interval_seconds < 0.1
            ):
                raise ValueError(
                    "gpu_telemetry_interval_seconds must be finite and at "
                    "least 0.1"
                )

    @property
    def config_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "backend": self.backend.value,
            "duration_seconds": self.duration_seconds,
            "threads": self.threads,
            "gpu_device_index": self.gpu_device_index,
            "gpu_telemetry_interval_seconds": (
                self.gpu_telemetry_interval_seconds
            ),
        }


@dataclass(frozen=True, slots=True)
class DecodeRun:
    decoded_frames: int
    output_time_seconds: float
    reported_speed_x: float
    user_seconds: float
    system_seconds: float
    ffmpeg_real_seconds: float
    process_max_rss_bytes: int
    wall_time_ns: int


@dataclass(frozen=True, slots=True)
class DecodeBenchmarkSummary:
    source_id: str
    source_sha256: str
    config: DecodeConfig
    ffmpeg_version: str
    command: tuple[str, ...]
    warmup_runs: int
    repeat_runs: int
    decoded_frames_per_run: int
    source_average_fps: float
    timing: TimingSummary
    mean_user_seconds: float
    mean_system_seconds: float
    mean_ffmpeg_real_seconds: float
    maximum_process_rss_bytes: int
    gpu_telemetry: TelemetrySummaryView | None
    unrelated_gpu_work_excluded_by_operator: bool | None

    def to_dict(self) -> dict[str, Any]:
        decoded_media_seconds = (
            self.decoded_frames_per_run / self.source_average_fps
        )
        p50_seconds = self.timing.p50_ns / 1_000_000_000
        return {
            "schema_version": "operations.decode_benchmark.v2",
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "config": self.config.to_dict(),
            "config_sha256": self.config.config_sha256,
            "ffmpeg_version": self.ffmpeg_version,
            "command": list(self.command),
            "warmup_runs": self.warmup_runs,
            "repeat_runs": self.repeat_runs,
            "decoded_frames_per_run": self.decoded_frames_per_run,
            "source_average_fps": self.source_average_fps,
            "timing": self.timing.to_dict(),
            "decoded_fps_at_wall_p50": (
                self.decoded_frames_per_run / p50_seconds
            ),
            "realtime_factor_at_wall_p50": decoded_media_seconds / p50_seconds,
            "mean_user_seconds": self.mean_user_seconds,
            "mean_system_seconds": self.mean_system_seconds,
            "mean_ffmpeg_real_seconds": self.mean_ffmpeg_real_seconds,
            "maximum_process_rss_bytes": self.maximum_process_rss_bytes,
            "gpu_telemetry": (
                None
                if self.gpu_telemetry is None
                else self.gpu_telemetry.to_dict()
            ),
            "unrelated_gpu_work_excluded_by_operator": (
                self.unrelated_gpu_work_excluded_by_operator
            ),
        }


def parse_ffmpeg_benchmark(stderr: str, wall_time_ns: int) -> DecodeRun:
    frame_matches = tuple(_FRAME_PATTERN.finditer(stderr))
    time_matches = tuple(_TIME_PATTERN.finditer(stderr))
    rss_matches = tuple(_RSS_PATTERN.finditer(stderr))
    if not frame_matches or not time_matches or not rss_matches:
        raise ValueError("FFmpeg benchmark output is incomplete")
    frame = frame_matches[-1]
    times = time_matches[-1]
    rss = rss_matches[-1]
    return DecodeRun(
        decoded_frames=int(frame.group("frames")),
        output_time_seconds=_parse_clock(frame.group("time")),
        reported_speed_x=float(frame.group("speed")),
        user_seconds=float(times.group("user")),
        system_seconds=float(times.group("system")),
        ffmpeg_real_seconds=float(times.group("real")),
        process_max_rss_bytes=int(rss.group("rss")) * 1024,
        wall_time_ns=wall_time_ns,
    )


def build_decode_command(path: Path, config: DecodeConfig) -> tuple[str, ...]:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-nostats",
        "-benchmark",
    ]
    if config.backend is DecodeBackend.CUDA:
        command.extend(("-hwaccel", "cuda", "-hwaccel_output_format", "cuda"))
    if config.threads is not None:
        command.extend(("-threads", str(config.threads)))
    command.extend(
        (
            "-i",
            str(path),
            "-t",
            f"{config.duration_seconds:.9g}",
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-f",
            "null",
            "-",
        )
    )
    return tuple(command)


def run_decode_once(
    path: Path,
    config: DecodeConfig,
    *,
    timeout_seconds: float | None = None,
) -> tuple[DecodeRun, tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    command = build_decode_command(path, config)
    start = perf_counter_ns()
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    wall_time_ns = perf_counter_ns() - start
    return parse_ffmpeg_benchmark(completed.stderr, wall_time_ns), command


def benchmark_decode(
    path: Path,
    *,
    source_id: str,
    source_sha256: str,
    config: DecodeConfig,
    warmup_runs: int,
    repeat_runs: int,
    timeout_seconds: float | None = None,
    unrelated_gpu_work_excluded_by_operator: bool | None = None,
    telemetry_monitor: TelemetryMonitor | None = None,
) -> DecodeBenchmarkSummary:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not source_id or not source_id.strip():
        raise ValueError("source_id must be non-empty")
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    if (
        isinstance(warmup_runs, bool)
        or not isinstance(warmup_runs, int)
        or warmup_runs < 0
    ):
        raise ValueError("warmup_runs must be a non-negative integer")
    if (
        isinstance(repeat_runs, bool)
        or not isinstance(repeat_runs, int)
        or repeat_runs <= 0
    ):
        raise ValueError("repeat_runs must be a positive integer")
    if timeout_seconds is not None and (
        not isfinite(timeout_seconds) or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")
    telemetry_enabled = config.gpu_device_index is not None
    if not telemetry_enabled and unrelated_gpu_work_excluded_by_operator is not None:
        raise ValueError(
            "GPU-work attestation is only valid when telemetry is enabled"
        )
    if telemetry_enabled and unrelated_gpu_work_excluded_by_operator is None:
        raise ValueError(
            "GPU telemetry requires an explicit clean or contaminated "
            "operator declaration"
        )
    if telemetry_enabled and telemetry_monitor is None:
        raise ValueError("GPU telemetry requires an injected telemetry monitor")
    initial_stat = path.stat()
    for _ in range(warmup_runs):
        run_decode_once(path, config, timeout_seconds=timeout_seconds)

    def measured_repeats() -> tuple[tuple[DecodeRun, tuple[str, ...]], ...]:
        return tuple(
            run_decode_once(path, config, timeout_seconds=timeout_seconds)
            for _ in range(repeat_runs)
        )

    gpu_telemetry: TelemetrySummaryView | None = None
    if telemetry_enabled:
        assert telemetry_monitor is not None
        runs_and_commands, gpu_telemetry = telemetry_monitor(
            measured_repeats,
            device_index=config.gpu_device_index,
            interval_seconds=config.gpu_telemetry_interval_seconds,
        )
    else:
        runs_and_commands = measured_repeats()
    runs = tuple(item[0] for item in runs_and_commands)
    command = runs_and_commands[0][1]
    frame_counts = {run.decoded_frames for run in runs}
    if len(frame_counts) != 1:
        raise RuntimeError("decoded frame count changed across repeat runs")
    probe = probe_video_file(path)
    version = subprocess.run(
        ("ffmpeg", "-version"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    actual_sha256 = sha256_file(path)
    final_stat = path.stat()
    if (
        initial_stat.st_size != final_stat.st_size
        or initial_stat.st_mtime_ns != final_stat.st_mtime_ns
    ):
        raise RuntimeError("source changed during decode benchmark")
    if actual_sha256 != source_sha256:
        raise ValueError("source SHA-256 does not match the admitted manifest")
    return DecodeBenchmarkSummary(
        source_id=source_id,
        source_sha256=source_sha256,
        config=config,
        ffmpeg_version=version,
        command=command,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
        decoded_frames_per_run=runs[0].decoded_frames,
        source_average_fps=probe.average_fps,
        timing=TimingSummary.from_samples(
            tuple(run.wall_time_ns for run in runs)
        ),
        mean_user_seconds=fmean(run.user_seconds for run in runs),
        mean_system_seconds=fmean(run.system_seconds for run in runs),
        mean_ffmpeg_real_seconds=fmean(run.ffmpeg_real_seconds for run in runs),
        maximum_process_rss_bytes=max(
            run.process_max_rss_bytes for run in runs
        ),
        gpu_telemetry=gpu_telemetry,
        unrelated_gpu_work_excluded_by_operator=(
            unrelated_gpu_work_excluded_by_operator
        ),
    )


def _parse_clock(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
