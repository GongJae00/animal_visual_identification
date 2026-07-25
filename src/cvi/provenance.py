"""Small, deterministic provenance helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


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
