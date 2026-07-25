"""No-replace directory publication helpers for protected artifacts."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_directory_noreplace(source: Path, target: Path) -> str:
    """Atomically publish a directory without replacing an existing target."""

    if os.name != "posix":
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        os.rename(source, target)
        return "PLATFORM_NOREPLACE_RENAME"
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "protected directory publication requires "
            "renameat2(RENAME_NOREPLACE)"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return "RENAMEAT2_NOREPLACE"
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target)
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        return _reserved_empty_directory_rename(source, target)
    raise OSError(error_number, os.strerror(error_number), target)


def _reserved_empty_directory_rename(source: Path, target: Path) -> str:
    """Use atomic mkdir as a cooperative DrvFS publication reservation."""

    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.mkdir(mode=0o700)
    reserved = target.stat()
    try:
        if source.stat().st_dev != reserved.st_dev:
            raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), target)
        observed = target.stat()
        if _stat_identity(observed) != _stat_identity(reserved) or any(
            target.iterdir()
        ):
            raise RuntimeError("protected publication reservation changed")
        os.rename(source, target)
    except BaseException:
        if target.exists() and not target.is_symlink():
            observed = target.stat()
            if _stat_identity(observed) == _stat_identity(reserved) and not any(
                target.iterdir()
            ):
                target.rmdir()
        raise
    return "RESERVED_EMPTY_DIRECTORY_RENAME"


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
