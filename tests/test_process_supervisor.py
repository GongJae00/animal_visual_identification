from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cvi.process_supervisor import (
    ProcessSupervisorPolicy,
    SupervisedProcessStatus,
    run_supervised_process,
)


def policy(**overrides: object) -> ProcessSupervisorPolicy:
    values: dict[str, object] = {
        "timeout_seconds": 2.0,
        "termination_grace_seconds": 0.2,
        "poll_interval_seconds": 0.01,
        "maximum_stdout_bytes": 4096,
        "maximum_stderr_bytes": 4096,
    }
    values.update(overrides)
    return ProcessSupervisorPolicy(**values)


class ProcessSupervisorTests(unittest.TestCase):
    def test_completed_fresh_process_has_bounded_hashed_logs_and_rss(self) -> None:
        with TemporaryDirectory() as temporary:
            result = run_supervised_process(
                (
                    sys.executable,
                    "-c",
                    "import sys,time; print('out'); "
                    "print('err', file=sys.stderr); time.sleep(0.05)",
                ),
                policy=policy(),
                working_directory=Path(temporary),
            )
        self.assertEqual(result.status, SupervisedProcessStatus.COMPLETED)
        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.stdout_bytes, 4)
        self.assertEqual(result.stderr_bytes, 4)
        self.assertTrue(result.stdout_complete)
        self.assertTrue(result.stderr_complete)
        self.assertIsNotNone(result.sampled_peak_rss_bytes)
        self.assertGreater(result.rss_samples, 0)
        self.assertTrue(result.fresh_process)
        self.assertEqual(result, result.__class__(**{
            "command": result.command,
            "policy_sha256": result.policy_sha256,
            "status": result.status,
            "return_code": result.return_code,
            "wall_time_ns": result.wall_time_ns,
            "sampled_peak_rss_bytes": result.sampled_peak_rss_bytes,
            "rss_samples": result.rss_samples,
            "rss_scope": result.rss_scope,
            "stdout_bytes": result.stdout_bytes,
            "stdout_sha256": result.stdout_sha256,
            "stderr_bytes": result.stderr_bytes,
            "stderr_sha256": result.stderr_sha256,
            "stdout_complete": result.stdout_complete,
            "stderr_complete": result.stderr_complete,
            "termination_signal_sent": result.termination_signal_sent,
            "kill_signal_sent": result.kill_signal_sent,
        }))

    def test_timeout_terminates_the_process_group(self) -> None:
        result = run_supervised_process(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            policy=policy(timeout_seconds=0.1),
        )
        self.assertEqual(result.status, SupervisedProcessStatus.TIMED_OUT)
        self.assertNotEqual(result.return_code, 0)
        self.assertTrue(result.termination_signal_sent)

    def test_timeout_escalates_to_kill_when_term_is_ignored(self) -> None:
        result = run_supervised_process(
            (
                sys.executable,
                "-c",
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
            ),
            policy=policy(
                timeout_seconds=0.1,
                termination_grace_seconds=0.05,
            ),
        )
        self.assertEqual(result.status, SupervisedProcessStatus.TIMED_OUT)
        self.assertTrue(result.termination_signal_sent)
        self.assertTrue(result.kill_signal_sent)

    def test_output_limit_fails_closed_without_pipe_buffering(self) -> None:
        result = run_supervised_process(
            (
                sys.executable,
                "-c",
                "import os,time; os.write(1, b'x' * 1000000); time.sleep(30)",
            ),
            policy=policy(maximum_stdout_bytes=512),
        )
        self.assertEqual(
            result.status,
            SupervisedProcessStatus.OUTPUT_LIMIT_EXCEEDED,
        )
        self.assertTrue(result.termination_signal_sent)
        self.assertEqual(result.stdout_bytes, 512)
        self.assertFalse(result.stdout_complete)

    def test_successful_parent_with_surviving_descendant_is_rejected(self) -> None:
        result = run_supervised_process(
            (
                sys.executable,
                "-c",
                "import subprocess,sys; "
                "subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(30)'],"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)",
            ),
            policy=policy(termination_grace_seconds=0.05),
        )
        self.assertEqual(
            result.status,
            SupervisedProcessStatus.SURVIVING_DESCENDANTS,
        )
        self.assertTrue(result.termination_signal_sent)

    def test_policy_and_command_validation_fail_closed(self) -> None:
        payload = policy().to_dict()
        self.assertEqual(ProcessSupervisorPolicy.from_dict(payload), policy())
        payload["unknown"] = True
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            ProcessSupervisorPolicy.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            run_supervised_process((), policy=policy())
        with self.assertRaisesRegex(ValueError, "environment"):
            run_supervised_process(
                (sys.executable, "-c", "pass"),
                policy=policy(),
                environment={"BAD=KEY": "value"},
            )


if __name__ == "__main__":
    unittest.main()
