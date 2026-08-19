"""Strict protected JSON I/O shared by command-line boundaries."""

from __future__ import annotations

import json
import math
import os
import stat
import hashlib
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from foundation.provenance import content_sha256
from foundation.protected_publication import (
    _stat_identity,
    fsync_directory,
    rename_directory_noreplace,
)


@dataclass(frozen=True, slots=True)
class StrictJsonDocument:
    """One bounded file read and both byte-level and semantic hashes."""

    payload: dict[str, Any]
    raw_sha256: str
    canonical_payload_sha256: str
    byte_size: int


def read_strict_json_document(
    path: Path,
    *,
    maximum_bytes: int = 268_435_456,
    maximum_depth: int = 32,
    maximum_nodes: int = 2_000_000,
    maximum_keys: int = 1_000_000,
    maximum_array_length: int = 1_000_000,
    maximum_string_characters: int = 4_194_304,
    maximum_number_characters: int = 128,
) -> StrictJsonDocument:
    """Read, hash, and parse one regular JSON file under explicit bounds.

    The raw digest is computed from the exact bytes used for parsing. The file
    descriptor and final pathname identity are checked before returning.
    """

    limits = (
        maximum_bytes,
        maximum_depth,
        maximum_nodes,
        maximum_keys,
        maximum_array_length,
        maximum_string_characters,
        maximum_number_characters,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in limits):
        raise ValueError("strict JSON limits must be positive integers")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("strict JSON reading requires O_NOFOLLOW")
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        if path.is_symlink():
            raise ValueError(f"input JSON must not be a symlink: {path}") from exc
        raise
    digest = hashlib.sha256()
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"input JSON must be a regular file: {path}")
        if initial.st_size > maximum_bytes:
            raise ValueError(f"input JSON exceeds byte limit: {path}")
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(descriptor, min(1 << 20, maximum_bytes + 1 - observed)):
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError(f"input JSON exceeds byte limit: {path}")
            digest.update(chunk)
            chunks.append(chunk)
        final = os.fstat(descriptor)
        named = os.stat(absolute, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if _stat_identity(initial) != _stat_identity(final) or (
        named.st_dev,
        named.st_ino,
    ) != (initial.st_dev, initial.st_ino) or observed != initial.st_size:
        raise RuntimeError(f"input JSON changed while reading: {path}")
    raw = b"".join(chunks)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"input JSON must be UTF-8: {path}") from exc

    def parse_int(value: str) -> int:
        if len(value) > maximum_number_characters:
            raise ValueError("JSON integer token exceeds limit")
        return int(value)

    def parse_float(value: str) -> float:
        if len(value) > maximum_number_characters:
            raise ValueError("JSON float token exceeds limit")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("JSON number must be finite")
        return parsed

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON numeric constant: {value}")
            ),
            parse_int=parse_int,
            parse_float=parse_float,
        )
    except (RecursionError, OverflowError) as exc:
        raise ValueError(f"input JSON exceeds parser bounds: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{path} root must be an object")
    _validate_json_structure(
        payload,
        maximum_depth=maximum_depth,
        maximum_nodes=maximum_nodes,
        maximum_keys=maximum_keys,
        maximum_array_length=maximum_array_length,
        maximum_string_characters=maximum_string_characters,
    )
    return StrictJsonDocument(
        payload=payload,
        raw_sha256=digest.hexdigest(),
        canonical_payload_sha256=content_sha256(payload),
        byte_size=observed,
    )


def read_strict_json_object(path: Path) -> dict[str, Any]:
    return read_strict_json_document(path).payload


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
                temporary.write_bytes(json_document_bytes(payload))
                os.chmod(temporary, 0o600)
                staged.append((temporary, target))
            for temporary, target in staged:
                os.link(temporary, target)
                created.append(target)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def json_document_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_private_json_directory_bundle(
    target: Path,
    outputs: tuple[tuple[str, dict[str, Any]], ...],
) -> str:
    """Publish a complete JSON directory with one no-replace rename."""

    if not outputs:
        raise ValueError("protected JSON directory bundle must not be empty")
    names = tuple(name for name, _ in outputs)
    if len(names) != len(set(names)):
        raise ValueError("protected JSON directory names must be distinct")
    for name in names:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("protected JSON directory names must be simple")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    parent = target.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    with TemporaryDirectory(prefix=".cvi-json-directory-", dir=parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir(mode=0o700)
        for name, payload in outputs:
            path = staging / name
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            try:
                view = memoryview(json_document_bytes(payload))
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("protected JSON write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        fsync_directory(staging)
        strategy = rename_directory_noreplace(staging, parent / target.name)
    fsync_directory(parent / target.name)
    fsync_directory(parent)
    return strategy


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _validate_json_structure(
    root: dict[str, Any],
    *,
    maximum_depth: int,
    maximum_nodes: int,
    maximum_keys: int,
    maximum_array_length: int,
    maximum_string_characters: int,
) -> None:
    stack: list[tuple[Any, int]] = [(root, 1)]
    nodes = 0
    keys = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > maximum_nodes or depth > maximum_depth:
            raise ValueError("JSON structure exceeds node or depth limit")
        if isinstance(value, dict):
            keys += len(value)
            if keys > maximum_keys:
                raise ValueError("JSON object key count exceeds limit")
            for key, child in value.items():
                _validate_json_string(key, maximum_string_characters)
                stack.append((child, depth + 1))
        elif isinstance(value, list):
            if len(value) > maximum_array_length:
                raise ValueError("JSON array length exceeds limit")
            stack.extend((child, depth + 1) for child in value)
        elif isinstance(value, str):
            _validate_json_string(value, maximum_string_characters)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("JSON number must be finite")
        elif value is None or isinstance(value, (bool, int)):
            continue
        else:  # pragma: no cover
            raise TypeError("unsupported JSON value")


def _validate_json_string(value: str, maximum: int) -> None:
    if len(value) > maximum:
        raise ValueError("JSON string exceeds limit")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("JSON contains an unpaired surrogate")
