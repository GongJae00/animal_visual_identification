"""Descriptor-retained reads for small provenance and intake boundaries."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


@dataclass(frozen=True, slots=True)
class RetainedFileRead:
    sha256: str
    byte_count: int
    payload: bytes | None


class RetainedFileBinding(TypedDict):
    path: str
    byte_size: int
    content_sha256: str


def retained_regular_file_binding(
    path: Path, *, subject: str
) -> RetainedFileBinding:
    if path.is_symlink():
        raise ValueError(f"{subject} must not be a symlink")
    absolute = Path(os.path.abspath(os.fspath(path)))
    retained = read_retained_regular_file(absolute, subject=subject)
    if retained.byte_count <= 0:
        raise ValueError(f"{subject} must be a nonempty file")
    return {
        "path": str(absolute),
        "byte_size": retained.byte_count,
        "content_sha256": retained.sha256,
    }


def verify_retained_regular_file_binding(
    path: Path,
    binding: Mapping[str, object],
    *,
    subject: str,
) -> None:
    if retained_regular_file_binding(path, subject=subject) != dict(binding):
        raise RuntimeError(f"{subject} changed across execution")


def read_retained_regular_file(
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    maximum_bytes: int | None = None,
    capture_payload: bool = False,
    subject: str = "intake source",
    phase_callback: Callable[[str], None] | None = None,
    phase_label: str = "SOURCE_HASHED",
) -> RetainedFileRead:
    """Read one unchanged named inode through retained parent and file FDs.

    ``capture_payload=False`` keeps memory bounded for large weight files.  Small
    JSON callers opt in only after supplying a hard ``maximum_bytes`` limit.
    """

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(f"{subject} intake requires O_NOFOLLOW support")
    if capture_payload and maximum_bytes is None:
        raise ValueError("captured intake payload requires maximum_bytes")
    for value, name in (
        (expected_bytes, "expected_bytes"),
        (maximum_bytes, "maximum_bytes"),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        expected_bytes is not None
        and maximum_bytes is not None
        and expected_bytes > maximum_bytes
    ):
        raise ValueError(f"{subject} expected byte size exceeds intake maximum")

    absolute = Path(os.path.abspath(os.fspath(path)))
    parent_flags = os.O_RDONLY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_CLOEXEC"):
        parent_flags |= os.O_CLOEXEC
        file_flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        parent_flags |= os.O_DIRECTORY
    parent_fd = os.open(absolute.parent, parent_flags)
    try:
        parent_initial = os.fstat(parent_fd)
        descriptor = os.open(absolute.name, file_flags, dir_fd=parent_fd)
        try:
            initial = os.fstat(descriptor)
            if not stat.S_ISREG(initial.st_mode):
                raise ValueError(f"{subject} must be a regular file")
            if expected_bytes is not None and initial.st_size != expected_bytes:
                raise ValueError(f"{subject} byte size differs from source")
            if maximum_bytes is not None and initial.st_size > maximum_bytes:
                raise ValueError(f"{subject} exceeds intake maximum")

            digest = hashlib.sha256()
            observed = 0
            chunks: list[bytes] | None = [] if capture_payload else None
            while True:
                payload = os.read(descriptor, 1_048_576)
                if not payload:
                    break
                digest.update(payload)
                observed += len(payload)
                if chunks is not None:
                    chunks.append(payload)
                if maximum_bytes is not None and observed > maximum_bytes:
                    raise ValueError(f"{subject} exceeds intake maximum")
            observed_sha256 = digest.hexdigest()
            if observed != initial.st_size:
                raise RuntimeError(f"{subject} hash byte count differs")
            if expected_sha256 is not None and observed_sha256 != expected_sha256:
                raise ValueError(f"{subject} SHA-256 differs from source")
            if phase_callback is not None:
                phase_callback(phase_label)

            final = os.fstat(descriptor)
            named_final = os.stat(
                absolute.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            parent_final = os.fstat(parent_fd)
            named_parent_final = os.stat(
                absolute.parent,
                follow_symlinks=False,
            )
            if _stat_identity(initial) != _stat_identity(final):
                raise RuntimeError(f"{subject} changed during intake")
            if not stat.S_ISREG(named_final.st_mode) or (
                named_final.st_dev,
                named_final.st_ino,
            ) != (final.st_dev, final.st_ino):
                raise RuntimeError(f"{subject} path changed during intake")
            if _directory_identity(parent_initial) != _directory_identity(parent_final):
                raise RuntimeError(f"{subject} parent changed during intake")
            if not stat.S_ISDIR(named_parent_final.st_mode) or (
                named_parent_final.st_dev,
                named_parent_final.st_ino,
            ) != (parent_final.st_dev, parent_final.st_ino):
                raise RuntimeError(f"{subject} parent path changed during intake")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)
    return RetainedFileRead(
        sha256=observed_sha256,
        byte_count=observed,
        payload=b"".join(chunks) if chunks is not None else None,
    )


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
