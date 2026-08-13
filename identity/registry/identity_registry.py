"""Deterministic identity registry mapping public dataset labels to registered dog IDs.

A registered dog ID is a UUIDv5 deterministically derived from a
dataset-identity label. It is a stable identifier, not an anonymity or privacy
boundary.

Namespace isolation
  REGISTERED_DOG_NAMESPACE = uuid5(DNS, "cvi.registered_dog.v1")
  registered_dog_id = uuid5(NAMESPACE, dataset_identity_id)

The registry persists both the identity_token (opaque SHA256 used in the
split infrastructure) and the registered_dog_id so that the protected
split can be bound to registered IDs without passing source labels to model or
evaluator code.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from contracts.identity_ids import (
    REGISTERED_DOG_NAMESPACE,
    compute_identity_token,
    compute_public_subject_token,
    compute_registered_dog_id,
    compute_sample_token,
    compute_sequence_token,
    extract_dataset_name,
)


_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS identity_registry (
    identity_token       TEXT PRIMARY KEY,
    dataset_identity_id  TEXT NOT NULL,
    registered_dog_id    TEXT NOT NULL UNIQUE,
    dataset_name         TEXT NOT NULL,
    image_count          INTEGER NOT NULL DEFAULT 1,
    created_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_registered_dog_id
    ON identity_registry(registered_dog_id);
CREATE INDEX IF NOT EXISTS idx_dataset_identity_id
    ON identity_registry(dataset_identity_id);
"""

_INSERT_RECORD = """\
INSERT INTO identity_registry
    (identity_token, dataset_identity_id, registered_dog_id,
     dataset_name, image_count, created_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(identity_token) DO UPDATE SET
    image_count = image_count + 1;
"""


_SCHEMA_VERSION = "cvi.identity_registry.v1"
_MAXIMUM_REGISTRATIONS = 1_000_000
_MAXIMUM_IMAGE_COUNT = 2**63 - 1
_MAXIMUM_GENERATED_AT_BYTES = 64
_RECORD_FIELDS = {
    "identity_token",
    "dataset_identity_id",
    "registered_dog_id",
    "dataset_name",
    "image_count",
}


@dataclass(frozen=True, slots=True)
class IdentityRegistryRecord:
    identity_token: str
    dataset_identity_id: str
    registered_dog_id: str
    dataset_name: str
    image_count: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "identity_token": self.identity_token,
            "dataset_identity_id": self.dataset_identity_id,
            "registered_dog_id": self.registered_dog_id,
            "dataset_name": self.dataset_name,
            "image_count": self.image_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IdentityRegistryRecord:
        _exact_keys(payload, _RECORD_FIELDS, "identity registry record")
        for field_name in _RECORD_FIELDS - {"image_count"}:
            if not isinstance(payload[field_name], str):
                raise TypeError(
                    f"identity registry record {field_name} must be a string"
                )
        if not isinstance(payload["image_count"], int) or isinstance(
            payload["image_count"], bool
        ):
            raise TypeError("identity registry record image_count must be an integer")
        return cls(
            identity_token=payload["identity_token"],
            dataset_identity_id=payload["dataset_identity_id"],
            registered_dog_id=payload["registered_dog_id"],
            dataset_name=payload["dataset_name"],
            image_count=payload["image_count"],
        )


@dataclass(frozen=True, slots=True)
class IdentityRegistry:
    records: tuple[IdentityRegistryRecord, ...]
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported identity registry schema")
        seen_tokens: set[str] = set()
        seen_ids: set[str] = set()
        for rec in self.records:
            _require_sha256(rec.identity_token, "identity_token")
            _require_sha256(rec.registered_dog_id, "registered_dog_id", allow_uuid=True)
            if rec.identity_token in seen_tokens:
                raise ValueError("duplicate identity_token")
            if rec.registered_dog_id in seen_ids:
                raise ValueError("duplicate registered_dog_id")
            seen_tokens.add(rec.identity_token)
            seen_ids.add(rec.registered_dog_id)

    def to_dict(self) -> dict[str, Any]:
        _validate_records(self.records, require_canonical_order=False)
        records = sorted(
            self.records,
            key=lambda record: (record.dataset_name, record.dataset_identity_id),
        )
        return {
            "schema_version": self.schema_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "namespace_uuid": str(REGISTERED_DOG_NAMESPACE),
            "registrations": [rec.to_dict() for rec in records],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IdentityRegistry:
        expected = {"schema_version", "registrations", "generated_at", "namespace_uuid"}
        _exact_keys(payload, expected, "identity registry")
        if payload["schema_version"] != _SCHEMA_VERSION:
            raise ValueError("unsupported identity registry schema")
        if payload["namespace_uuid"] != str(REGISTERED_DOG_NAMESPACE):
            raise ValueError("identity registry namespace_uuid is invalid")
        _validate_generated_at(payload["generated_at"])
        registrations = payload["registrations"]
        if not isinstance(registrations, list):
            raise TypeError("identity registry registrations must be an array")
        if len(registrations) > _MAXIMUM_REGISTRATIONS:
            raise ValueError("identity registry registrations exceed the size limit")
        records = tuple(
            IdentityRegistryRecord.from_dict(record) for record in registrations
        )
        _validate_records(records, require_canonical_order=True)
        return cls(records=records, schema_version=payload["schema_version"])


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------


def create_registry_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_CREATE_TABLE)
    finally:
        conn.close()


def _validate_registration(dataset_identity_id: str) -> tuple[str, str, str]:
    identity_token = compute_identity_token(dataset_identity_id)
    registered_dog_id = compute_registered_dog_id(dataset_identity_id)
    dataset_name = extract_dataset_name(dataset_identity_id)
    return identity_token, registered_dog_id, dataset_name


def register_identity(
    db_path: Path,
    dataset_identity_id: str,
) -> str:
    identity_token, registered_dog_id, dataset_name = _validate_registration(
        dataset_identity_id
    )
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            _INSERT_RECORD,
            (identity_token, dataset_identity_id, registered_dog_id,
             dataset_name, 1, now),
        )
        record = _lookup_validated_record(conn, identity_token)
        if record is None or record.dataset_identity_id != dataset_identity_id:
            raise ValueError("persisted identity registry row is inconsistent")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return record.registered_dog_id


def register_records(
    db_path: Path,
    dataset_identity_ids: list[str],
) -> dict[str, str]:
    """Register many dataset_identity_ids and return {identity_token: registered_dog_id}.

    Caller should pass each distinct identity ID once.  Duplicates within one
    call are deduplicated transparently; each unique ID is inserted once with
    an image_count of 1, and the ON CONFLICT clause increments the count on
    subsequent calls for the same identity_token.
    """
    seen: set[str] = set()
    unique_ids = [did for did in dataset_identity_ids if did not in seen and not seen.add(did)]
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(db_path))
    mapping: dict[str, str] = {}
    try:
        for did in unique_ids:
            token, rid, dsn = _validate_registration(did)
            conn.execute(
                _INSERT_RECORD,
                (token, did, rid, dsn, 1, now),
            )
            record = _lookup_validated_record(conn, token)
            if record is None or record.dataset_identity_id != did:
                raise ValueError("persisted identity registry row is inconsistent")
            mapping[token] = record.registered_dog_id
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return mapping


def load_registry_manifest(db_path: Path) -> IdentityRegistry:
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "SELECT identity_token, dataset_identity_id, registered_dog_id, "
            "       dataset_name, image_count "
            "FROM identity_registry ORDER BY dataset_name, dataset_identity_id"
        )
        records = tuple(_record_from_row(row) for row in cursor.fetchall())
    finally:
        conn.close()
    _validate_records(records, require_canonical_order=True)
    return IdentityRegistry(records=records)


def lookup_registered_dog_id(db_path: Path, identity_token: str) -> str | None:
    _require_sha256(identity_token, "identity_token")
    conn = sqlite3.connect(str(db_path))
    try:
        record = _lookup_validated_record(conn, identity_token)
        return record.registered_dog_id if record is not None else None
    finally:
        conn.close()


def lookup_by_identity_token(
    db_path: Path, identity_token: str
) -> IdentityRegistryRecord | None:
    _require_sha256(identity_token, "identity_token")
    conn = sqlite3.connect(str(db_path))
    try:
        return _lookup_validated_record(conn, identity_token)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_sha256(value: str, name: str, *, allow_uuid: bool = False) -> None:
    if allow_uuid and _is_canonical_uuid5(value):
        return
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        msg = "a 64-char hex SHA256" if not allow_uuid else "a 64-char hex SHA256 or UUID"
        raise ValueError(f"{name} must be {msg}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA256 digest") from exc


def _is_canonical_uuid5(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return False
    return parsed.version == 5 and str(parsed) == value


def _validate_record(record: IdentityRegistryRecord) -> None:
    if not record.dataset_identity_id:
        raise ValueError("dataset_identity_id must be a non-empty string")
    expected_dataset_name = extract_dataset_name(record.dataset_identity_id)
    if not expected_dataset_name or record.dataset_name != expected_dataset_name:
        raise ValueError("identity registry dataset_name is not canonical")
    if record.identity_token != compute_identity_token(record.dataset_identity_id):
        raise ValueError("identity registry identity_token is not deterministic")
    if not _is_canonical_uuid5(record.registered_dog_id):
        raise ValueError("registered_dog_id must be a canonical UUIDv5 identity")
    if record.registered_dog_id != compute_registered_dog_id(
        record.dataset_identity_id
    ):
        raise ValueError("identity registry registered_dog_id is not deterministic")
    if not 1 <= record.image_count <= _MAXIMUM_IMAGE_COUNT:
        raise ValueError("identity registry image_count is out of bounds")


def _validate_records(
    records: tuple[IdentityRegistryRecord, ...],
    *,
    require_canonical_order: bool,
) -> None:
    seen_tokens: set[str] = set()
    seen_dataset_ids: set[str] = set()
    seen_registered_ids: set[str] = set()
    for record in records:
        _validate_record(record)
        if record.identity_token in seen_tokens:
            raise ValueError("duplicate identity_token")
        if record.dataset_identity_id in seen_dataset_ids:
            raise ValueError("duplicate dataset_identity_id")
        if record.registered_dog_id in seen_registered_ids:
            raise ValueError("duplicate registered_dog_id")
        seen_tokens.add(record.identity_token)
        seen_dataset_ids.add(record.dataset_identity_id)
        seen_registered_ids.add(record.registered_dog_id)
    if require_canonical_order and list(records) != sorted(
        records,
        key=lambda record: (record.dataset_name, record.dataset_identity_id),
    ):
        raise ValueError("identity registry registrations are not canonically ordered")


def _validate_generated_at(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("identity registry generated_at must be a string")
    if not value or len(value.encode("utf-8")) > _MAXIMUM_GENERATED_AT_BYTES:
        raise ValueError("identity registry generated_at is out of bounds")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("identity registry generated_at is invalid") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timedelta(0)
        or parsed.isoformat() != value
    ):
        raise ValueError("identity registry generated_at must be canonical UTC text")


def _record_from_row(row: tuple[Any, ...]) -> IdentityRegistryRecord:
    if len(row) != 5:
        raise ValueError("persisted identity registry row has an invalid schema")
    record = IdentityRegistryRecord.from_dict(dict(zip(_RECORD_FIELDS_IN_ROW, row)))
    _validate_record(record)
    return record


_RECORD_FIELDS_IN_ROW = (
    "identity_token",
    "dataset_identity_id",
    "registered_dog_id",
    "dataset_name",
    "image_count",
)


def _lookup_validated_record(
    conn: sqlite3.Connection, identity_token: str
) -> IdentityRegistryRecord | None:
    row = conn.execute(
        "SELECT identity_token, dataset_identity_id, registered_dog_id, "
        "       dataset_name, image_count "
        "FROM identity_registry WHERE identity_token = ?",
        (identity_token,),
    ).fetchone()
    return None if row is None else _record_from_row(row)


def _exact_keys(
    payload: dict[str, Any], expected: set[str], object_name: str
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{object_name} must be an object")
    actual = set(payload)
    if actual != expected:
        missing = expected - actual
        unknown = actual - expected
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown fields: {', '.join(sorted(unknown))}")
        raise ValueError(
            f"{object_name} has invalid fields ({'; '.join(details)})"
        )
