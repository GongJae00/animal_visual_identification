"""Deterministic source and Python runtime provenance for offline tools."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from pathlib import Path
from typing import Any

from cvi.provenance import content_sha256


def build_offline_tool_provenance(tool_path: Path) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parents[1]
    resolved_tool = tool_path.resolve(strict=True)
    sources = tuple(sorted(package_root.glob("*.py"))) + (resolved_tool,)
    rows: list[dict[str, str | int]] = []
    for path in sources:
        resolved = path.resolve(strict=True)
        try:
            label = resolved.relative_to(repository_root).as_posix()
        except ValueError as error:
            raise ValueError("offline tool source is outside repository") from error
        initial = resolved.stat()
        if path.is_symlink() or not resolved.is_file():
            raise ValueError("offline tool source must be a real regular file")
        digest = hashlib.sha256()
        observed = 0
        with resolved.open("rb") as stream:
            while True:
                payload = stream.read(1_048_576)
                if not payload:
                    break
                digest.update(payload)
                observed += len(payload)
        final = resolved.stat()
        if (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
            initial.st_ctime_ns,
        ) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ) or observed != initial.st_size:
            raise RuntimeError("offline tool source changed while hashing")
        rows.append(
            {
                "relative_path": label,
                "content_sha256": digest.hexdigest(),
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
    }
    return {
        "schema_version": "cvi.offline_tool_provenance.v1",
        "code_source_manifest_sha256": content_sha256(rows),
        "code_source_files": rows,
        "runtime": runtime,
        "runtime_sha256": content_sha256(runtime),
    }
