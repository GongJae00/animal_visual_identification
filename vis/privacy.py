"""Publication scopes and recursive privacy/path enforcement."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum
from pathlib import PurePosixPath
from typing import Any


class PublicationScope(str, Enum):
    PUBLIC = "public"
    PAPER = "paper"
    PRIVATE = "private"


_SCOPE_RANK = {
    PublicationScope.PUBLIC: 0,
    PublicationScope.PAPER: 1,
    PublicationScope.PRIVATE: 2,
}
_PRIVATE_KEYS = {
    "animal_id",
    "dataset_identity_id",
    "identity_id",
    "identity_token",
    "owner_id",
    "owner_name",
    "private_id",
    "public_subject_token",
    "query_id",
    "sample_id",
    "sample_token",
    "source_path",
    "track_id",
}
_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def scope_allows(target: PublicationScope, data_scope: PublicationScope) -> bool:
    return _SCOPE_RANK[data_scope] <= _SCOPE_RANK[target]


def validate_publishable_value(value: Any, scope: PublicationScope) -> None:
    """Reject path and identifier disclosure from public or paper inputs."""

    if not isinstance(scope, PublicationScope):
        raise TypeError("scope must be a PublicationScope")
    if scope is PublicationScope.PRIVATE:
        return
    stack: list[tuple[Any, tuple[str, ...]]] = [(value, ())]
    while stack:
        current, trail = stack.pop()
        if isinstance(current, Mapping):
            for key, child in current.items():
                normalized = str(key).casefold()
                if normalized in _PRIVATE_KEYS:
                    location = ".".join((*trail, str(key)))
                    raise ValueError(
                        f"{scope.value} data contains private identifier field: {location}"
                    )
                stack.append((child, (*trail, str(key))))
        elif isinstance(current, (list, tuple)):
            stack.extend((child, trail) for child in current)
        elif isinstance(current, str):
            if _is_absolute_path(current):
                raise ValueError(f"{scope.value} data contains an absolute path")
            if _UUID.search(current):
                raise ValueError(f"{scope.value} data contains a private UUID")


def validate_relative_asset_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("asset path must be non-empty text")
    if _is_absolute_path(value) or "\\" in value:
        raise ValueError("asset path must be relative POSIX syntax")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("asset path must not contain traversal segments")
    return value


def _is_absolute_path(value: str) -> bool:
    return value.startswith(("/", "\\\\")) or bool(_WINDOWS_ABSOLUTE.match(value))
