"""Shared content binding for files inside exact local model artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ModelFileBinding:
    relative_path: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or self.relative_path != path.as_posix()
        ):
            raise ValueError("model file relative path is unsafe")
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size <= 0
        ):
            raise ValueError("model file size must be positive")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("model file SHA-256 must be a lowercase digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ModelFileBinding:
        if not isinstance(value, Mapping) or set(value) != {
            "relative_path",
            "byte_size",
            "sha256",
        }:
            raise ValueError("model file binding schema differs")
        return cls(**value)


__all__ = ["ModelFileBinding"]
