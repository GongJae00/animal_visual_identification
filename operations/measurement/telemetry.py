"""Constant-memory device-wide NVIDIA telemetry."""

from __future__ import annotations

import csv
import subprocess
import threading
from dataclasses import dataclass
from math import isfinite
from time import monotonic_ns
from typing import Callable, TypeVar

T = TypeVar("T")

_QUERY_FIELDS = (
    "index",
    "name",
    "driver_version",
    "memory.used",
    "power.draw",
    "power.limit",
    "utilization.gpu",
    "utilization.memory",
    "utilization.decoder",
)


@dataclass(frozen=True, slots=True)
class GpuTelemetrySample:
    timestamp_ns: int
    device_index: int
    device_name: str
    driver_version: str
    memory_used_mib: float | None
    power_draw_w: float | None
    power_limit_w: float | None
    gpu_utilization_pct: float | None
    memory_utilization_pct: float | None
    decoder_utilization_pct: float | None


@dataclass(frozen=True, slots=True)
class GpuTelemetrySummary:
    scope: str
    sampler_backend: str
    sampler_command: tuple[str, ...]
    requested_interval_seconds: float
    effective_mean_interval_seconds: float | None
    samples: int
    sampled_span_seconds: float
    device_index: int
    device_name: str
    driver_version: str
    power_limit_w: float | None
    memory_used_mib_mean: float | None
    memory_used_mib_max: float | None
    power_draw_w_mean: float | None
    power_draw_w_max: float | None
    gpu_utilization_pct_mean: float | None
    gpu_utilization_pct_max: float | None
    memory_utilization_pct_mean: float | None
    memory_utilization_pct_max: float | None
    decoder_utilization_pct_mean: float | None
    decoder_utilization_pct_max: float | None
    sampled_board_energy_joules: float | None

    def __post_init__(self) -> None:
        if self.scope != "device-wide":
            raise ValueError("GPU telemetry scope must be device-wide")
        if not self.sampler_backend.strip() or (
            self.sampler_backend != "injected" and not self.sampler_command
        ):
            raise ValueError("GPU sampler identity must be non-empty")
        if (
            not isfinite(self.requested_interval_seconds)
            or self.requested_interval_seconds < 0.1
        ):
            raise ValueError("invalid requested GPU telemetry interval")
        if (
            isinstance(self.samples, bool)
            or not isinstance(self.samples, int)
            or self.samples <= 0
            or not isfinite(self.sampled_span_seconds)
            or self.sampled_span_seconds < 0
        ):
            raise ValueError("invalid GPU telemetry sample extent")
        if self.effective_mean_interval_seconds is not None and (
            not isfinite(self.effective_mean_interval_seconds)
            or self.effective_mean_interval_seconds <= 0
        ):
            raise ValueError("invalid effective GPU telemetry interval")
        if (self.samples == 1) != (
            self.effective_mean_interval_seconds is None
        ):
            raise ValueError("GPU telemetry interval and sample count differ")
        if (
            isinstance(self.device_index, bool)
            or not isinstance(self.device_index, int)
            or self.device_index < 0
            or not self.device_name.strip()
            or not self.driver_version.strip()
        ):
            raise ValueError("invalid GPU telemetry device identity")
        for name in (
            "power_limit_w",
            "memory_used_mib_mean",
            "memory_used_mib_max",
            "power_draw_w_mean",
            "power_draw_w_max",
            "gpu_utilization_pct_mean",
            "gpu_utilization_pct_max",
            "memory_utilization_pct_mean",
            "memory_utilization_pct_max",
            "decoder_utilization_pct_mean",
            "decoder_utilization_pct_max",
            "sampled_board_energy_joules",
        ):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"invalid GPU telemetry metric: {name}")
        for mean_name, max_name in (
            ("memory_used_mib_mean", "memory_used_mib_max"),
            ("power_draw_w_mean", "power_draw_w_max"),
            ("gpu_utilization_pct_mean", "gpu_utilization_pct_max"),
            ("memory_utilization_pct_mean", "memory_utilization_pct_max"),
            ("decoder_utilization_pct_mean", "decoder_utilization_pct_max"),
        ):
            mean = getattr(self, mean_name)
            maximum = getattr(self, max_name)
            if (mean is None) != (maximum is None):
                raise ValueError("GPU telemetry mean/max availability differs")
            if mean is not None and mean > maximum:
                raise ValueError("GPU telemetry mean exceeds maximum")

    def to_dict(
        self,
    ) -> dict[str, str | int | float | list[str] | None]:
        payload = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }
        payload["sampler_command"] = list(self.sampler_command)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GpuTelemetrySummary:
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("GPU telemetry summary keys mismatch")
        command = payload["sampler_command"]
        if not isinstance(command, list):
            raise TypeError("GPU sampler command must be a list")
        values = dict(payload)
        values["sampler_command"] = tuple(command)
        return cls(**values)


class GpuTelemetryAccumulator:
    """Streaming sums/maxima and trapezoidal board-energy integration."""

    __slots__ = (
        "_device_index",
        "_device_name",
        "_driver_version",
        "_energy_joules",
        "_first_timestamp_ns",
        "_last_sample",
        "_maxima",
        "_power_limit_w",
        "_requested_interval_seconds",
        "_sampler_backend",
        "_sampler_command",
        "_samples",
        "_sums",
        "_valid_counts",
    )

    _METRICS = (
        "memory_used_mib",
        "power_draw_w",
        "gpu_utilization_pct",
        "memory_utilization_pct",
        "decoder_utilization_pct",
    )

    def __init__(
        self,
        requested_interval_seconds: float,
        *,
        sampler_backend: str = "injected",
        sampler_command: tuple[str, ...] = (),
    ) -> None:
        if (
            not isfinite(requested_interval_seconds)
            or requested_interval_seconds < 0.1
        ):
            raise ValueError(
                "telemetry interval must be finite and at least 0.1 seconds"
            )
        self._requested_interval_seconds = requested_interval_seconds
        if not sampler_backend.strip():
            raise ValueError("sampler_backend must be non-empty")
        self._sampler_backend = sampler_backend
        self._sampler_command = sampler_command
        self._samples = 0
        self._first_timestamp_ns: int | None = None
        self._last_sample: GpuTelemetrySample | None = None
        self._device_index: int | None = None
        self._device_name: str | None = None
        self._driver_version: str | None = None
        self._power_limit_w: float | None = None
        self._sums = {metric: 0.0 for metric in self._METRICS}
        self._valid_counts = {metric: 0 for metric in self._METRICS}
        self._maxima: dict[str, float | None] = {
            metric: None for metric in self._METRICS
        }
        self._energy_joules = 0.0

    def add(self, sample: GpuTelemetrySample) -> None:
        if self._device_index is None:
            self._device_index = sample.device_index
            self._device_name = sample.device_name
            self._driver_version = sample.driver_version
            self._power_limit_w = sample.power_limit_w
            self._first_timestamp_ns = sample.timestamp_ns
        elif (
            sample.device_index != self._device_index
            or sample.device_name != self._device_name
            or sample.driver_version != self._driver_version
        ):
            raise ValueError("GPU identity changed during telemetry")
        if (
            self._last_sample is not None
            and sample.timestamp_ns <= self._last_sample.timestamp_ns
        ):
            raise ValueError("telemetry timestamps must be strictly increasing")
        if (
            self._last_sample is not None
            and self._last_sample.power_draw_w is not None
            and sample.power_draw_w is not None
        ):
            delta_seconds = (
                sample.timestamp_ns - self._last_sample.timestamp_ns
            ) / 1_000_000_000
            self._energy_joules += (
                self._last_sample.power_draw_w + sample.power_draw_w
            ) * 0.5 * delta_seconds
        for metric in self._METRICS:
            value = getattr(sample, metric)
            if value is None:
                continue
            self._sums[metric] += value
            self._valid_counts[metric] += 1
            current_maximum = self._maxima[metric]
            self._maxima[metric] = (
                value if current_maximum is None else max(current_maximum, value)
            )
        self._samples += 1
        self._last_sample = sample

    def finalize(self) -> GpuTelemetrySummary:
        if self._samples == 0 or self._last_sample is None:
            raise ValueError("at least one telemetry sample is required")
        sampled_span_seconds = (
            self._last_sample.timestamp_ns - self._first_timestamp_ns
        ) / 1_000_000_000
        effective_interval = (
            sampled_span_seconds / (self._samples - 1)
            if self._samples >= 2
            else None
        )

        def mean(metric: str) -> float | None:
            count = self._valid_counts[metric]
            return self._sums[metric] / count if count else None

        energy = (
            self._energy_joules
            if self._valid_counts["power_draw_w"] >= 2
            else None
        )
        return GpuTelemetrySummary(
            scope="device-wide",
            sampler_backend=self._sampler_backend,
            sampler_command=self._sampler_command,
            requested_interval_seconds=self._requested_interval_seconds,
            effective_mean_interval_seconds=effective_interval,
            samples=self._samples,
            sampled_span_seconds=sampled_span_seconds,
            device_index=self._device_index,
            device_name=self._device_name,
            driver_version=self._driver_version,
            power_limit_w=self._power_limit_w,
            memory_used_mib_mean=mean("memory_used_mib"),
            memory_used_mib_max=self._maxima["memory_used_mib"],
            power_draw_w_mean=mean("power_draw_w"),
            power_draw_w_max=self._maxima["power_draw_w"],
            gpu_utilization_pct_mean=mean("gpu_utilization_pct"),
            gpu_utilization_pct_max=self._maxima["gpu_utilization_pct"],
            memory_utilization_pct_mean=mean("memory_utilization_pct"),
            memory_utilization_pct_max=self._maxima["memory_utilization_pct"],
            decoder_utilization_pct_mean=mean("decoder_utilization_pct"),
            decoder_utilization_pct_max=self._maxima[
                "decoder_utilization_pct"
            ],
            sampled_board_energy_joules=energy,
        )


def parse_nvidia_smi_csv(
    line: str,
    *,
    timestamp_ns: int,
) -> GpuTelemetrySample:
    rows = tuple(csv.reader((line,)))
    if len(rows) != 1 or len(rows[0]) != len(_QUERY_FIELDS):
        raise ValueError("unexpected nvidia-smi CSV field count")
    values = tuple(value.strip() for value in rows[0])
    return GpuTelemetrySample(
        timestamp_ns=timestamp_ns,
        device_index=int(values[0]),
        device_name=values[1],
        driver_version=values[2],
        memory_used_mib=_optional_number(values[3]),
        power_draw_w=_optional_number(values[4]),
        power_limit_w=_optional_number(values[5]),
        gpu_utilization_pct=_optional_number(values[6]),
        memory_utilization_pct=_optional_number(values[7]),
        decoder_utilization_pct=_optional_number(values[8]),
    )


def monitor_operation(
    operation: Callable[[], T],
    *,
    device_index: int,
    interval_seconds: float,
) -> tuple[T, GpuTelemetrySummary]:
    """Run an operation with one persistent nvidia-smi sampling process."""

    if isinstance(device_index, bool) or not isinstance(device_index, int):
        raise TypeError("device_index must be an integer")
    if device_index < 0:
        raise ValueError("device_index must be non-negative")
    loop_milliseconds = max(100, round(interval_seconds * 1000))
    command = (
        "nvidia-smi",
        f"--id={device_index}",
        f"--query-gpu={','.join(_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
        f"--loop-ms={loop_milliseconds}",
    )
    accumulator = GpuTelemetryAccumulator(
        interval_seconds,
        sampler_backend="nvidia-smi-loop",
        sampler_command=command,
    )
    stop = threading.Event()
    ready = threading.Event()
    errors: list[BaseException] = []

    def worker() -> None:
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("nvidia-smi stdout pipe is unavailable")
            for line in process.stdout:
                if not line.strip():
                    continue
                accumulator.add(
                    parse_nvidia_smi_csv(line, timestamp_ns=monotonic_ns())
                )
                ready.set()
                if stop.is_set():
                    break
            if not stop.is_set():
                stderr = (
                    process.stderr.read()
                    if process.stderr is not None
                    else ""
                )
                raise RuntimeError(
                    "nvidia-smi telemetry stopped unexpectedly: "
                    f"{stderr.strip()}"
                )
        except BaseException as error:
            errors.append(error)
            ready.set()
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.0)
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()

    thread = threading.Thread(target=worker, name="cvi-gpu-telemetry", daemon=True)
    thread.start()
    if not ready.wait(timeout=10.0):
        stop.set()
        thread.join(timeout=5.0)
        raise TimeoutError("initial GPU telemetry sample timed out")
    if errors:
        stop.set()
        thread.join(timeout=5.0)
        raise errors[0]
    try:
        result = operation()
    finally:
        stop.set()
        thread.join(timeout=10.0)
    if thread.is_alive():
        raise TimeoutError("GPU telemetry thread did not stop")
    if errors:
        raise errors[0]
    return result, accumulator.finalize()


def _optional_number(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"n/a", "[n/a]", "not supported", "[not supported]"}:
        return None
    number = float(value)
    if not isfinite(number):
        raise ValueError("nvidia-smi metric must be finite")
    return number
