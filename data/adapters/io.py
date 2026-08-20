"""Shared path and image I/O for dataset layout adapters.

Layout parsers live in sibling modules. This module does not know publisher
folder schemas.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath

from PIL import Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dims(path: Path) -> tuple[int, int]:
    with Image.open(path) as opened:
        return opened.size


def verified_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or relative != path.as_posix()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe relative path: {relative!r}")
    absolute_root = Path(os.path.abspath(os.fspath(root)))
    resolved_root = absolute_root.resolve()
    candidate = absolute_root.joinpath(*path.parts)
    if candidate.is_symlink():
        raise ValueError(f"not a regular file: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"path traversal: {relative}")
    return candidate


def verified_regular_file(path: Path, subject: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.is_symlink() or not absolute.is_file():
        raise ValueError(f"{subject} must be a regular file: {absolute}")
    return absolute


# Names used by tests and in-flight layout modules.
_file_sha256 = file_sha256
_image_dims = image_dims
_verified_path = verified_path
_verified_regular_file = verified_regular_file
