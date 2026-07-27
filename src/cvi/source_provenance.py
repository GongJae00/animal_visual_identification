"""Deterministic source and Python runtime provenance for offline tools."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from cvi.provenance import content_sha256


def build_offline_tool_provenance(
    tool_path: Path,
    *,
    additional_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parents[1]
    resolved_tool = tool_path.resolve(strict=True)
    requested = (
        *sorted(package_root.glob("*.py")),
        resolved_tool,
        *(path.resolve(strict=True) for path in additional_paths),
    )
    sources = tuple(dict.fromkeys(requested))
    rows: list[dict[str, str | int]] = []
    for path in sources:
        resolved = path.resolve(strict=True)
        try:
            label = resolved.relative_to(repository_root).as_posix()
        except ValueError as error:
            raise ValueError("offline tool source is outside repository") from error
        if path.is_symlink() or not resolved.is_file():
            raise ValueError("offline tool source must be a real regular file")
        digest, observed = _hash_regular_file_same_read(resolved)
        rows.append(
            {
                "relative_path": label,
                "content_sha256": digest,
                "byte_size": observed,
            }
        )
    rows.sort(key=lambda item: str(item["relative_path"]))
    runtime = {
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "os_name": os.name,
        "numpy_version": _distribution_version("numpy"),
        "jsonschema_version": _distribution_version("jsonschema"),
        "scikit_learn_version": _distribution_version("scikit-learn"),
    }
    return {
        "schema_version": "cvi.offline_tool_provenance.v1",
        "code_source_manifest_sha256": content_sha256(rows),
        "code_source_files": rows,
        "runtime": runtime,
        "runtime_sha256": content_sha256(runtime),
    }


def _hash_regular_file_same_read(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    observed = 0
    try:
        initial = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
            observed += len(chunk)
        final = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(initial) != identity(final) or identity(initial) != identity(named):
        raise RuntimeError("offline tool source changed while hashing")
    if observed != initial.st_size:
        raise RuntimeError("offline tool source byte count differs")
    return digest.hexdigest(), observed


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"
