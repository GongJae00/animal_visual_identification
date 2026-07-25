"""Deterministic identity registry mapping public dataset labels to registered dog IDs.

A registered dog ID is a UUIDv5 deterministically derived from the
dataset-identity label so that the same dog always maps to the same
production-safe identifier without any central coordination.

Namespace isolation
  CVI_REGISTERED_DOG_NAMESPACE = uuid5(DNS, "cvi.registered_dog.v1")
  registered_dog_id = uuid5(NAMESPACE, dataset_identity_id)

The registry persists both the identity_token (opaque SHA256 used in the
split infrastructure) and the registered_dog_id so that the protected
split can be bound to production IDs without leaking source labels to
model or evaluator code.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CVI_REGISTERED_DOG_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "cvi.registered_dog.v1")

_IDENTITY_TOKEN_PREFIX = b"identity\x00"
_SAMPLE_TOKEN_PREFIX = b"sample\x00"
_SEQUENCE_TOKEN_PREFIX = b"sequence\x00"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compute_identity_token(dataset_identity_id: str) -> str:
    return _sha256(_IDENTITY_TOKEN_PREFIX + dataset_identity_id.encode("utf-8"))


def compute_sample_token(source_sample_id: str) -> str:
    return _sha256(_SAMPLE_TOKEN_PREFIX + source_sample_id.encode("utf-8"))


def compute_sequence_token(sequence_id: str | None, identity_token: str) -> str:
    payload = sequence_id if sequence_id is not None else identity_token
    return _sha256(_SEQUENCE_TOKEN_PREFIX + payload.encode("utf-8"))


def compute_registered_dog_id(dataset_identity_id: str) -> str:
    return str(uuid.uuid5(CVI_REGISTERED_DOG_NAMESPACE, dataset_identity_id))


def extract_dataset_name(dataset_identity_id: str) -> str:
    return dataset_identity_id.split(":", 1)[0]


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
        return {
            "schema_version": self.schema_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "namespace_uuid": str(CVI_REGISTERED_DOG_NAMESPACE),
            "registrations": [rec.to_dict() for rec in self.records],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IdentityRegistry:
        expected = {"schema_version", "registrations", "generated_at", "namespace_uuid"}
        _exact_keys(payload, expected, "identity registry")
        return cls(
            records=tuple(
                IdentityRegistryRecord.from_dict(rec)
                for rec in payload["registrations"]
            ),
            schema_version=payload.get("schema_version", _SCHEMA_VERSION),
        )


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
        conn.commit()
    finally:
        conn.close()
    return registered_dog_id


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
            mapping[token] = rid
        conn.commit()
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
        records = tuple(
            IdentityRegistryRecord(
                identity_token=row[0],
                dataset_identity_id=row[1],
                registered_dog_id=row[2],
                dataset_name=row[3],
                image_count=row[4],
            )
            for row in cursor.fetchall()
        )
    finally:
        conn.close()
    return IdentityRegistry(records=records)


def lookup_registered_dog_id(db_path: Path, identity_token: str) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "SELECT registered_dog_id FROM identity_registry "
            "WHERE identity_token = ?",
            (identity_token,),
        )
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def lookup_by_identity_token(
    db_path: Path, identity_token: str
) -> IdentityRegistryRecord | None:
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "SELECT identity_token, dataset_identity_id, registered_dog_id, "
            "       dataset_name, image_count "
            "FROM identity_registry WHERE identity_token = ?",
            (identity_token,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return IdentityRegistryRecord(
            identity_token=row[0],
            dataset_identity_id=row[1],
            registered_dog_id=row[2],
            dataset_name=row[3],
            image_count=row[4],
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_sha256(value: str, name: str, *, allow_uuid: bool = False) -> None:
    if allow_uuid:
        try:
            uuid.UUID(value)
            return
        except ValueError:
            pass
    if not isinstance(value, str) or len(value) != 64:
        msg = "a 64-char hex SHA256" if not allow_uuid else "a 64-char hex SHA256 or UUID"
        raise ValueError(f"{name} must be {msg}")
    int(value, 16)


def _exact_keys(
    payload: dict[str, Any], expected: set[str], object_name: str
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{object_name} must be an object")
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(
            f"{object_name} has unknown fields: {', '.join(sorted(unknown))}"
        )
