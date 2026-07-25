from __future__ import annotations

import os
import sys
import unittest
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from cvi.onnx_backend import onnxruntime_distribution_identity
from cvi.worker_environment import (
    SANITIZED_WORKER_ENVIRONMENT_NAMES,
    WorkerEnvironmentIdentity,
    build_sanitized_worker_environment,
    validate_current_worker_environment,
)


class WorkerEnvironmentTests(unittest.TestCase):
    def test_sanitization_is_value_free_and_does_not_mutate_parent(self) -> None:
        parent = {
            "PATH": "/usr/bin",
            "TOKEN": "not-recorded",
            "LD_LIBRARY_PATH": "/untrusted/cuda",
            "LD_AUDIT": "/untrusted/audit.so",
            "GLIBC_TUNABLES": "glibc.malloc.check=3",
            "CUDA_HOME": "/untrusted/cuda",
            "CUDA_VISIBLE_DEVICES": "99",
            "OMP_NUM_THREADS": "999",
            "ORT_CUDA_TUNABLE_OP_ENABLE": "1",
            "PYTHONPATH": "/untrusted/python",
        }
        original = dict(parent)
        child, identity = build_sanitized_worker_environment(parent)
        self.assertEqual(parent, original)
        self.assertNotIn("TOKEN", child)
        self.assertEqual(child, dict(identity.environment_entries))
        self.assertEqual(
            identity.parent_defined_names,
            (
                "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "GLIBC_TUNABLES",
                "LD_AUDIT", "LD_LIBRARY_PATH", "OMP_NUM_THREADS",
                "ORT_CUDA_TUNABLE_OP_ENABLE", "PYTHONPATH",
            ),
        )
        self.assertNotIn("LD_AUDIT", child)
        self.assertNotIn("GLIBC_TUNABLES", child)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", child)
        self.assertNotIn("ORT_CUDA_TUNABLE_OP_ENABLE", child)
        self.assertEqual(child["OMP_NUM_THREADS"], "1")
        serialized = identity.to_dict()
        self.assertNotIn("/untrusted/cuda", repr(serialized))
        self.assertNotIn("/untrusted/python", repr(serialized))
        self.assertEqual(WorkerEnvironmentIdentity.from_dict(serialized), identity)

    def test_parent_ld_preload_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "forbids parent LD_PRELOAD"):
            build_sanitized_worker_environment({"LD_PRELOAD": "/tmp/x.so"})

    def test_worker_rejects_even_empty_sanitized_variable(self) -> None:
        clean, identity = build_sanitized_worker_environment({})
        clean["LD_LIBRARY_PATH"] = ""
        with patch.dict(os.environ, clean, clear=True):
            with self.assertRaisesRegex(RuntimeError, "differs from allowlist"):
                validate_current_worker_environment(identity)

    def test_worker_accepts_exact_clean_environment_and_python(self) -> None:
        clean, identity = build_sanitized_worker_environment(
            {"CUDA_PATH": "/ignored"},
            python_executable=sys.executable,
        )
        with patch.dict(os.environ, clean, clear=True):
            self.assertEqual(
                validate_current_worker_environment(identity),
                identity,
            )

    def test_onnx_runtime_distribution_lanes_are_exact(self) -> None:
        def installed_cpu(name: str) -> str:
            if name == "onnxruntime":
                return "1.2.3"
            raise PackageNotFoundError(name)

        with patch("importlib.metadata.version", side_effect=installed_cpu):
            self.assertEqual(
                onnxruntime_distribution_identity(require_gpu=False),
                ("onnxruntime", "1.2.3"),
            )
            with self.assertRaisesRegex(RuntimeError, "CUDA backend requires"):
                onnxruntime_distribution_identity(require_gpu=True)

        def installed_gpu(name: str) -> str:
            if name == "onnxruntime-gpu":
                return "4.5.6"
            raise PackageNotFoundError(name)

        with patch("importlib.metadata.version", side_effect=installed_gpu):
            self.assertEqual(
                onnxruntime_distribution_identity(require_gpu=True),
                ("onnxruntime-gpu", "4.5.6"),
            )
            with self.assertRaisesRegex(RuntimeError, "CPU worker requires"):
                onnxruntime_distribution_identity(require_gpu=False)

        with patch("importlib.metadata.version", return_value="7.8.9"):
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                onnxruntime_distribution_identity(require_gpu=False)


if __name__ == "__main__":
    unittest.main()
