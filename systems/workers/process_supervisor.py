"""Fresh-process execution with bounded logs, timeout, and scoped RSS."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from time import monotonic_ns, sleep
from typing import Any, BinaryIO, Mapping

from foundation.provenance import content_sha256


class SupervisedProcessStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    OUTPUT_LIMIT_EXCEEDED = "OUTPUT_LIMIT_EXCEEDED"
    SURVIVING_DESCENDANTS = "SURVIVING_DESCENDANTS"


@dataclass(frozen=True, slots=True)
class ProcessSupervisorPolicy:
    timeout_seconds: float
    termination_grace_seconds: float
    poll_interval_seconds: float
    maximum_stdout_bytes: int
    maximum_stderr_bytes: int
    schema_version: str = "cvi.process_supervisor_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.process_supervisor_policy.v1":
            raise ValueError("unsupported process supervisor policy schema")
        for name in (
            "timeout_seconds",
            "termination_grace_seconds",
            "poll_interval_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.poll_interval_seconds > self.timeout_seconds:
            raise ValueError("poll interval must not exceed timeout")
        for name in ("maximum_stdout_bytes", "maximum_stderr_bytes"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "schema_version": self.schema_version,
            "timeout_seconds": self.timeout_seconds,
            "termination_grace_seconds": self.termination_grace_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "maximum_stdout_bytes": self.maximum_stdout_bytes,
            "maximum_stderr_bytes": self.maximum_stderr_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProcessSupervisorPolicy:
        expected = {
            "schema_version",
            "timeout_seconds",
            "termination_grace_seconds",
            "poll_interval_seconds",
            "maximum_stdout_bytes",
            "maximum_stderr_bytes",
        }
        if set(payload) != expected:
            raise ValueError("process supervisor policy keys mismatch")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class SupervisedProcessResult:
    command: tuple[str, ...]
    policy_sha256: str
    status: SupervisedProcessStatus
    return_code: int
    wall_time_ns: int
    sampled_peak_rss_bytes: int | None
    rss_samples: int
    rss_scope: str
    stdout_bytes: int
    stdout_sha256: str
    stderr_bytes: int
    stderr_sha256: str
    stdout_complete: bool
    stderr_complete: bool
    termination_signal_sent: bool
    kill_signal_sent: bool
    fresh_process: bool = True
    schema_version: str = "cvi.supervised_process_result.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.supervised_process_result.v1":
            raise ValueError("unsupported supervised process result schema")
        if not self.command or any(not item for item in self.command):
            raise ValueError("supervised command must be non-empty")
        if isinstance(self.return_code, bool) or not isinstance(
            self.return_code,
            int,
        ):
            raise TypeError("supervised return code must be an integer")
        for name in (
            "stdout_complete",
            "stderr_complete",
            "termination_signal_sent",
            "kill_signal_sent",
            "fresh_process",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.status is SupervisedProcessStatus.COMPLETED and (
            self.return_code != 0
            or not self.stdout_complete
            or not self.stderr_complete
        ):
            raise ValueError("completed process result is inconsistent")
        if self.status is SupervisedProcessStatus.FAILED and self.return_code == 0:
            raise ValueError("failed process result must have nonzero return code")
        if self.wall_time_ns < 0:
            raise ValueError("wall time must be non-negative")
        if self.sampled_peak_rss_bytes is not None:
            if self.sampled_peak_rss_bytes < 0 or self.rss_samples <= 0:
                raise ValueError("invalid sampled RSS result")
        elif self.rss_samples != 0:
            raise ValueError("RSS samples require a sampled peak")
        if not self.rss_scope.strip():
            raise ValueError("RSS scope must be explicit")
        for size in (self.stdout_bytes, self.stderr_bytes):
            if size < 0:
                raise ValueError("captured byte count must be non-negative")
        for digest in (
            self.policy_sha256,
            self.stdout_sha256,
            self.stderr_sha256,
        ):
            _validate_sha256(digest)
        if self.fresh_process is not True:
            raise ValueError("supervised result must describe a fresh process")

    @property
    def result_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "command": list(self.command),
            "policy_sha256": self.policy_sha256,
            "status": self.status.value,
            "return_code": self.return_code,
            "wall_time_ns": self.wall_time_ns,
            "sampled_peak_rss_bytes": self.sampled_peak_rss_bytes,
            "rss_samples": self.rss_samples,
            "rss_scope": self.rss_scope,
            "stdout_bytes": self.stdout_bytes,
            "stdout_sha256": self.stdout_sha256,
            "stderr_bytes": self.stderr_bytes,
            "stderr_sha256": self.stderr_sha256,
            "stdout_complete": self.stdout_complete,
            "stderr_complete": self.stderr_complete,
            "termination_signal_sent": self.termination_signal_sent,
            "kill_signal_sent": self.kill_signal_sent,
            "fresh_process": self.fresh_process,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SupervisedProcessResult:
        expected = {
            "schema_version",
            "command",
            "policy_sha256",
            "status",
            "return_code",
            "wall_time_ns",
            "sampled_peak_rss_bytes",
            "rss_samples",
            "rss_scope",
            "stdout_bytes",
            "stdout_sha256",
            "stderr_bytes",
            "stderr_sha256",
            "stdout_complete",
            "stderr_complete",
            "termination_signal_sent",
            "kill_signal_sent",
            "fresh_process",
        }
        if set(payload) != expected:
            raise ValueError("supervised process result keys mismatch")
        command = payload["command"]
        if not isinstance(command, list):
            raise TypeError("supervised process command must be a list")
        values = dict(payload)
        values["command"] = tuple(command)
        values["status"] = SupervisedProcessStatus(payload["status"])
        return cls(**values)


def run_supervised_process(
    command: tuple[str, ...],
    *,
    policy: ProcessSupervisorPolicy,
    environment: Mapping[str, str] | None = None,
    working_directory: Path | None = None,
) -> SupervisedProcessResult:
    """Run one fresh process without shell interpretation or unbounded pipes."""

    _validate_command(command)
    resolved_working_directory: Path | None = None
    if working_directory is not None:
        resolved_working_directory = working_directory.resolve(strict=True)
        if not resolved_working_directory.is_dir():
            raise NotADirectoryError(resolved_working_directory)
    child_environment: dict[str, str] | None = None
    if environment is not None:
        child_environment = _validate_environment(environment)

    started_ns = monotonic_ns()
    process = subprocess.Popen(
        command,
        cwd=resolved_working_directory,
        env=child_environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        start_new_session=(os.name == "posix"),
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("supervised process pipes are unavailable")
    output_limit_exceeded = threading.Event()
    capture_errors: list[BaseException] = []
    stdout_capture = _PipeCapture()
    stderr_capture = _PipeCapture()
    capture_threads = (
        threading.Thread(
            target=_capture_pipe,
            args=(
                process.stdout,
                policy.maximum_stdout_bytes,
                stdout_capture,
                output_limit_exceeded,
                capture_errors,
            ),
            name="cvi-supervised-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_capture_pipe,
            args=(
                process.stderr,
                policy.maximum_stderr_bytes,
                stderr_capture,
                output_limit_exceeded,
                capture_errors,
            ),
            name="cvi-supervised-stderr",
            daemon=True,
        ),
    )
    for thread in capture_threads:
        thread.start()
    peak_rss: int | None = None
    rss_samples = 0
    termination_sent = False
    kill_sent = False
    status: SupervisedProcessStatus | None = None
    deadline_ns = started_ns + int(policy.timeout_seconds * 1e9)
    while process.poll() is None:
        rss = _read_linux_process_rss_bytes(process.pid)
        if rss is not None:
            peak_rss = rss if peak_rss is None else max(peak_rss, rss)
            rss_samples += 1
        if output_limit_exceeded.is_set():
            status = SupervisedProcessStatus.OUTPUT_LIMIT_EXCEEDED
            termination_sent, kill_sent = _terminate_process(
                process,
                policy.termination_grace_seconds,
            )
            break
        if monotonic_ns() >= deadline_ns:
            status = SupervisedProcessStatus.TIMED_OUT
            termination_sent, kill_sent = _terminate_process(
                process,
                policy.termination_grace_seconds,
            )
            break
        sleep(policy.poll_interval_seconds)
    return_code = process.wait()
    for thread in capture_threads:
        thread.join(timeout=policy.termination_grace_seconds)
    if any(thread.is_alive() for thread in capture_threads):
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.stdout.close()
        process.stderr.close()
        for thread in capture_threads:
            thread.join(timeout=policy.termination_grace_seconds)
        raise RuntimeError("supervised output capture did not reach EOF")
    process.stdout.close()
    process.stderr.close()
    if capture_errors:
        raise RuntimeError("supervised output capture failed") from capture_errors[0]
    if status is None and os.name == "posix":
        descendant_term, descendant_kill = _cleanup_process_group(
            process.pid,
            policy.termination_grace_seconds,
        )
        if descendant_term:
            status = SupervisedProcessStatus.SURVIVING_DESCENDANTS
            termination_sent = True
            kill_sent = kill_sent or descendant_kill
    final_rss = _read_linux_process_rss_bytes(process.pid)
    if final_rss is not None:
        peak_rss = final_rss if peak_rss is None else max(peak_rss, final_rss)
        rss_samples += 1
    finished_ns = monotonic_ns()
    if status is None and output_limit_exceeded.is_set():
        status = SupervisedProcessStatus.OUTPUT_LIMIT_EXCEEDED
    if status is None:
        status = (
            SupervisedProcessStatus.COMPLETED
            if return_code == 0
            else SupervisedProcessStatus.FAILED
        )
    return SupervisedProcessResult(
        command=command,
        policy_sha256=policy.policy_sha256,
        status=status,
        return_code=return_code,
        wall_time_ns=finished_ns - started_ns,
        sampled_peak_rss_bytes=peak_rss,
        rss_samples=rss_samples,
        rss_scope=(
            "linux-proc-worker-main-process-sampled-current-rss"
            if peak_rss is not None
            else "UNAVAILABLE"
        ),
        stdout_bytes=stdout_capture.captured_bytes,
        stdout_sha256=stdout_capture.digest.hexdigest(),
        stderr_bytes=stderr_capture.captured_bytes,
        stderr_sha256=stderr_capture.digest.hexdigest(),
        stdout_complete=stdout_capture.complete,
        stderr_complete=stderr_capture.complete,
        termination_signal_sent=termination_sent,
        kill_signal_sent=kill_sent,
    )


def _terminate_process(
    process: subprocess.Popen[bytes],
    grace_seconds: float,
) -> tuple[bool, bool]:
    if process.poll() is not None:
        return False, False
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return True, False
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=grace_seconds)
        return True, True


def _cleanup_process_group(
    process_group_id: int,
    grace_seconds: float,
) -> tuple[bool, bool]:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False, False
    os.killpg(process_group_id, signal.SIGTERM)
    deadline = monotonic_ns() + int(grace_seconds * 1e9)
    while monotonic_ns() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return True, False
        sleep(min(0.01, grace_seconds))
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return True, False
    return True, True


def _read_linux_process_rss_bytes(pid: int) -> int | None:
    status = Path("/proc") / str(pid) / "status"
    try:
        lines = status.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2] != "kB":
            return None
        return int(fields[1]) * 1024
    return None


class _PipeCapture:
    __slots__ = ("captured_bytes", "complete", "digest")

    def __init__(self) -> None:
        self.captured_bytes = 0
        self.complete = False
        self.digest = hashlib.sha256()


def _capture_pipe(
    stream: BinaryIO,
    maximum_bytes: int,
    capture: _PipeCapture,
    limit_exceeded: threading.Event,
    errors: list[BaseException],
) -> None:
    try:
        while True:
            remaining = maximum_bytes - capture.captured_bytes
            read_size = min(64 * 1024, remaining + 1)
            chunk = stream.read(read_size)
            if not chunk:
                capture.complete = True
                return
            accepted = chunk[:remaining]
            capture.digest.update(accepted)
            capture.captured_bytes += len(accepted)
            if len(chunk) > len(accepted):
                limit_exceeded.set()
                return
    except BaseException as error:
        errors.append(error)
        limit_exceeded.set()


def _validate_command(command: tuple[str, ...]) -> None:
    if not command:
        raise ValueError("supervised command must not be empty")
    for item in command:
        if not isinstance(item, str) or not item or "\x00" in item:
            raise ValueError("supervised command items must be non-empty strings")


def _validate_environment(environment: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in environment.items():
        if (
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\x00" in key
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ValueError("invalid supervised process environment")
        result[key] = value
    return result


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("expected a lowercase SHA-256 digest")
