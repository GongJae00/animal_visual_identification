"""Strict protected JSON I/O shared by command-line boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from cvi.provenance import content_sha256


def read_strict_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"input JSON must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"input JSON must be a regular file: {path}")
    initial = resolved.stat()
    payload = json.loads(
        resolved.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    final = resolved.stat()
    if (
        initial.st_size != final.st_size
        or initial.st_mtime_ns != final.st_mtime_ns
    ):
        raise RuntimeError(f"input JSON changed while reading: {path}")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} root must be an object")
    return payload


def read_content_hashed_json_bundle(
    path: Path,
    *,
    schema_version: str,
    payload_field: str,
    sha256_field: str,
) -> dict[str, Any]:
    bundle = read_strict_json_object(path)
    expected = {"schema_version", payload_field, sha256_field}
    if set(bundle) != expected or bundle["schema_version"] != schema_version:
        raise ValueError("content-hashed JSON bundle schema differs")
    payload = bundle[payload_field]
    if not isinstance(payload, dict):
        raise TypeError("content-hashed JSON bundle payload must be an object")
    if content_sha256(payload) != bundle[sha256_field]:
        raise ValueError("content-hashed JSON bundle digest differs")
    return payload


def write_private_json_bundle(
    outputs: tuple[tuple[Path, dict[str, Any]], ...],
) -> None:
    resolved_items: list[tuple[Path, dict[str, Any]]] = []
    for path, payload in outputs:
        if path.is_symlink():
            raise ValueError(f"output must not be a symlink: {path}")
        parent = path.parent.resolve(strict=True)
        if not parent.is_dir():
            raise NotADirectoryError(parent)
        resolved_items.append((parent / path.name, payload))
    resolved = tuple(resolved_items)
    paths = tuple(path for path, _ in resolved)
    if not paths:
        raise ValueError("protected JSON bundle must not be empty")
    if len(paths) != len(set(paths)):
        raise ValueError("output paths must be distinct")
    parents = {path.parent for path in paths}
    if len(parents) != 1:
        raise ValueError("all outputs must share one protected directory")
    parent = next(iter(parents))
    existing = tuple(path for path in paths if path.exists())
    if existing:
        raise FileExistsError(
            "refusing to overwrite outputs: "
            + ", ".join(str(path) for path in existing)
        )
    created: list[Path] = []
    try:
        with TemporaryDirectory(prefix=".cvi-json-bundle-", dir=parent) as temp:
            temp_root = Path(temp)
            staged: list[tuple[Path, Path]] = []
            for index, (target, payload) in enumerate(resolved):
                temporary = temp_root / f"{index}.json"
                temporary.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.chmod(temporary, 0o600)
                staged.append((temporary, target))
            for temporary, target in staged:
                os.link(temporary, target)
                created.append(target)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result
