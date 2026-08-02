"""Fail-closed allowlisted environment identity for protected workers."""

from __future__ import annotations

import os
import pwd
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from data_pipeline.acquisition import sha256_file
from foundation.provenance import content_sha256


SANITIZED_WORKER_ENVIRONMENT_NAMES = tuple(sorted({
    "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
    "CUDA_CACHE_PATH", "CUDA_CACHE_MAXSIZE", "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING", "CUBLAS_WORKSPACE_CONFIG",
    "CUDNN_LOGDEST_DBG", "CUDNN_LOGLEVEL_DBG", "NVIDIA_TF32_OVERRIDE",
    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_BIND_NOW",
    "GLIBC_TUNABLES", "PYTHONHOME", "PYTHONPATH", "PYTHONINSPECT",
    "PYTHONSTARTUP", "PYTHONWARNINGS", "PYTHONBREAKPOINT", "PYTHONMALLOC",
    "PYTHONASYNCIODEBUG", "OMP_NUM_THREADS", "OMP_DYNAMIC",
    "OMP_WAIT_POLICY", "MKL_NUM_THREADS", "MKL_DYNAMIC",
    "KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK",
    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "ORT_CUDA_TUNABLE_OP_ENABLE", "ORT_CUDA_TUNABLE_OP_TUNING_ENABLE",
    "ORT_LOAD_CONFIG_FROM_MODEL",
}))

ISOLATED_PYTHON_FLAGS = ("-I", "-B")
ISOLATED_WORKER_BOOTSTRAP = """
import json
import importlib.util
import os
import runpy
import sys
import types

module_name = sys.argv.pop(1)
request_path = sys.argv.pop(1)
with open(request_path, encoding="utf-8") as stream:
    request = json.load(stream)
expected = dict(
    request["worker_environment_identity"]["environment_entries"]
)
if dict(os.environ) != expected:
    raise RuntimeError("protected worker initial environment differs from allowlist")
package_name = module_name.partition(".")[0]
package_spec = importlib.util.find_spec(package_name)
if package_spec is None or package_spec.submodule_search_locations is None:
    raise RuntimeError("protected worker package location is unavailable")
package = types.ModuleType(package_name)
package.__package__ = package_name
package.__path__ = list(package_spec.submodule_search_locations)
package.__spec__ = package_spec
sys.modules[package_name] = package
runpy.run_module(module_name, run_name="__main__", alter_sys=True)
""".strip()


def _fixed_worker_environment() -> dict[str, str]:
    home = pwd.getpwuid(os.getuid()).pw_dir
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_MODULE_LOADING": "LAZY",
        "HOME": home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MALLOC_ARENA_MAX": "2",
        "KMP_DUPLICATE_LIB_OK": "True",
        "KMP_INIT_AT_FORK": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "NVIDIA_TF32_OVERRIDE": "0",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": "/tmp",
        "TZ": "UTC",
    }


_SEMANTICS = {
    "schema_version": "cvi.worker_environment_semantics.v2",
    "controlled_parent_names": list(SANITIZED_WORKER_ENVIRONMENT_NAMES),
    "child_construction": "FIXED_ALLOWLIST_NO_PARENT_VALUES",
    "child_environment": _fixed_worker_environment(),
    "parent_value_disclosure": "CONTROLLED_NAMES_ONLY",
    "python_flags": list(ISOLATED_PYTHON_FLAGS),
    "python_executable_identity": "INVOCATION_RESOLVED_PATH_SIZE_SHA256",
}
WORKER_ENVIRONMENT_SEMANTICS_SHA256 = content_sha256(_SEMANTICS)


@dataclass(frozen=True, slots=True)
class WorkerEnvironmentIdentity:
    parent_defined_names: tuple[str, ...]
    environment_entries: tuple[tuple[str, str], ...]
    environment_sha256: str
    python_executable_invocation_path: str
    python_executable_resolved_path: str
    python_executable_bytes: int
    python_executable_sha256: str
    sanitized_names: tuple[str, ...] = SANITIZED_WORKER_ENVIRONMENT_NAMES
    isolated_python_flags: tuple[str, ...] = ISOLATED_PYTHON_FLAGS
    semantics_sha256: str = WORKER_ENVIRONMENT_SEMANTICS_SHA256
    schema_version: str = "cvi.worker_environment_identity.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.worker_environment_identity.v2":
            raise ValueError("unsupported worker environment identity schema")
        if self.sanitized_names != SANITIZED_WORKER_ENVIRONMENT_NAMES:
            raise ValueError("worker controlled environment names differ")
        if self.isolated_python_flags != ISOLATED_PYTHON_FLAGS:
            raise ValueError("worker isolated Python flags differ")
        if self.semantics_sha256 != WORKER_ENVIRONMENT_SEMANTICS_SHA256:
            raise ValueError("worker environment semantics differ")
        if self.parent_defined_names != tuple(
            sorted(set(self.parent_defined_names))
        ) or any(
            name not in self.sanitized_names for name in self.parent_defined_names
        ):
            raise ValueError("parent-defined controlled names differ")
        if self.environment_entries != tuple(sorted(self.environment_entries)):
            raise ValueError("worker environment entries must be key sorted")
        if len({name for name, _ in self.environment_entries}) != len(
            self.environment_entries
        ) or any(not name or "=" in name for name, _ in self.environment_entries):
            raise ValueError("worker environment entry names differ")
        if any("\x00" in name or "\x00" in value for name, value in self.environment_entries):
            raise ValueError("worker environment contains NUL")
        _validate_sha256(self.environment_sha256)
        if self.environment_sha256 != content_sha256(
            [list(item) for item in self.environment_entries]
        ):
            raise ValueError("worker environment hash differs")
        if not self.python_executable_invocation_path or not (
            self.python_executable_resolved_path
        ):
            raise ValueError("Python executable path is empty")
        if (
            isinstance(self.python_executable_bytes, bool)
            or not isinstance(self.python_executable_bytes, int)
            or self.python_executable_bytes <= 0
        ):
            raise ValueError("Python executable byte size must be positive")
        _validate_sha256(self.python_executable_sha256)

    @property
    def identity_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sanitized_names": list(self.sanitized_names),
            "parent_defined_names": list(self.parent_defined_names),
            "environment_entries": [list(item) for item in self.environment_entries],
            "environment_sha256": self.environment_sha256,
            "isolated_python_flags": list(self.isolated_python_flags),
            "semantics_sha256": self.semantics_sha256,
            "python_executable_invocation_path": (
                self.python_executable_invocation_path
            ),
            "python_executable_resolved_path": self.python_executable_resolved_path,
            "python_executable_bytes": self.python_executable_bytes,
            "python_executable_sha256": self.python_executable_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkerEnvironmentIdentity:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("worker environment identity keys mismatch")
        list_names = (
            "sanitized_names", "parent_defined_names", "environment_entries",
            "isolated_python_flags",
        )
        if any(not isinstance(payload[name], list) for name in list_names):
            raise TypeError("worker environment collections must be lists")
        values = dict(payload)
        values["sanitized_names"] = tuple(values["sanitized_names"])
        values["parent_defined_names"] = tuple(values["parent_defined_names"])
        values["environment_entries"] = tuple(
            tuple(item) for item in values["environment_entries"]
        )
        values["isolated_python_flags"] = tuple(
            values["isolated_python_flags"]
        )
        return cls(**values)


def build_sanitized_worker_environment(
    parent_environment: Mapping[str, str],
    *,
    python_executable: str | Path = sys.executable,
) -> tuple[dict[str, str], WorkerEnvironmentIdentity]:
    if parent_environment.get("LD_PRELOAD") not in {None, ""}:
        raise RuntimeError("protected worker benchmark forbids parent LD_PRELOAD")
    child = _fixed_worker_environment()
    parent_defined_names = tuple(
        name for name in SANITIZED_WORKER_ENVIRONMENT_NAMES
        if name in parent_environment
    )
    entries = tuple(sorted(child.items()))
    identity = _python_bound_identity(
        parent_defined_names=parent_defined_names,
        environment_entries=entries,
        python_executable=python_executable,
    )
    return child, identity


def validate_current_worker_environment(
    expected: WorkerEnvironmentIdentity,
) -> WorkerEnvironmentIdentity:
    observed_environment = tuple(sorted(os.environ.items()))
    if observed_environment != expected.environment_entries:
        raise RuntimeError("protected worker environment differs from allowlist")
    observed = _python_bound_identity(
        parent_defined_names=expected.parent_defined_names,
        environment_entries=observed_environment,
        python_executable=sys.executable,
    )
    if observed != expected:
        raise RuntimeError("worker environment identity differs from request")
    return observed


def _python_bound_identity(
    *,
    parent_defined_names: tuple[str, ...],
    environment_entries: tuple[tuple[str, str], ...],
    python_executable: str | Path,
) -> WorkerEnvironmentIdentity:
    invocation = Path(python_executable)
    if not invocation.is_absolute():
        raise ValueError("Python executable path must be absolute")
    resolved = invocation.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Python executable must resolve to a regular file")
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError("Python executable changed while hashing")
    return WorkerEnvironmentIdentity(
        parent_defined_names=tuple(sorted(parent_defined_names)),
        environment_entries=environment_entries,
        environment_sha256=content_sha256(
            [list(item) for item in environment_entries]
        ),
        python_executable_invocation_path=str(invocation),
        python_executable_resolved_path=str(resolved),
        python_executable_bytes=before.st_size,
        python_executable_sha256=digest,
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _validate_sha256(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("expected a lowercase SHA-256 digest")
