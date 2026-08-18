"""Small, deterministic provenance helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


def git_worktree_provenance(repository: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True, cwd=repository
    ).strip()
    status = subprocess.check_output(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        text=True,
        cwd=repository,
    )
    return {
        "code_commit": commit,
        "worktree_dirty": bool(status.strip()),
        "worktree_status_basis": (
            "git status --porcelain=v1 --untracked-files=normal; includes staged, "
            "unstaged, and untracked path status, not untracked file contents"
        ),
    }


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Serialize a JSON value canonically for stable local receipts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_sha256(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
