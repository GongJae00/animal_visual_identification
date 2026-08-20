"""Provisional generated identities kept separate from registered dog IDs."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from shared.contracts.identity_ids import (
    GENERATED_DOG_NAMESPACE,
    compute_generated_identity_id,
    compute_source_cluster_token,
)

_SCHEMA_VERSION = "enrollment.generated_identity_registry.v1"
_GENERATOR_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAXIMUM_RECORDS = 1_000_000
_MAXIMUM_EVIDENCE_COUNT = 2**63 - 1


class GeneratedIdentityStatus(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    MERGED_TO_REGISTERED = "MERGED_TO_REGISTERED"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class GeneratedIdentityRecord:
    generated_identity_id: str
    generator_id: str
    source_cluster_token: str
    evidence_count: int
    status: GeneratedIdentityStatus = GeneratedIdentityStatus.PROVISIONAL
    registered_identity_id: str | None = None
    superseded_by_generated_identity_id: str | None = None

    def __post_init__(self) -> None:
        _validate_generator_id(self.generator_id)
        _validate_sha256(self.source_cluster_token, "source_cluster_token")
        if self.generated_identity_id != compute_generated_identity_id(
            self.generator_id, self.source_cluster_token
        ):
            raise ValueError("generated_identity_id is not deterministic")
        if (
            not isinstance(self.evidence_count, int)
            or isinstance(self.evidence_count, bool)
            or not 1 <= self.evidence_count <= _MAXIMUM_EVIDENCE_COUNT
        ):
            raise ValueError("evidence_count is out of bounds")
        if not isinstance(self.status, GeneratedIdentityStatus):
            raise TypeError("status must be a GeneratedIdentityStatus")

        has_registered = self.registered_identity_id is not None
        has_successor = self.superseded_by_generated_identity_id is not None
        if self.status is GeneratedIdentityStatus.MERGED_TO_REGISTERED:
            if not has_registered or has_successor:
                raise ValueError(
                    "merged generated identity requires only a registered target"
                )
            _validate_uuid5(self.registered_identity_id, "registered_identity_id")
        elif self.status is GeneratedIdentityStatus.SUPERSEDED:
            if has_registered or not has_successor:
                raise ValueError(
                    "superseded generated identity requires only a generated target"
                )
            _validate_uuid5(
                self.superseded_by_generated_identity_id,
                "superseded_by_generated_identity_id",
            )
            if self.superseded_by_generated_identity_id == self.generated_identity_id:
                raise ValueError("generated identity cannot supersede itself")
        elif has_registered or has_successor:
            raise ValueError("provisional or rejected identity cannot have a target")

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "generated_identity_id": self.generated_identity_id,
            "generator_id": self.generator_id,
            "source_cluster_token": self.source_cluster_token,
            "evidence_count": self.evidence_count,
            "status": self.status.value,
            "registered_identity_id": self.registered_identity_id,
            "superseded_by_generated_identity_id": self.superseded_by_generated_identity_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GeneratedIdentityRecord:
        expected = {
            "generated_identity_id",
            "generator_id",
            "source_cluster_token",
            "evidence_count",
            "status",
            "registered_identity_id",
            "superseded_by_generated_identity_id",
        }
        _require_exact_keys(payload, expected, "generated identity record")
        try:
            status = GeneratedIdentityStatus(payload["status"])
        except (TypeError, ValueError) as exc:
            raise ValueError("generated identity status is invalid") from exc
        return cls(
            generated_identity_id=payload["generated_identity_id"],
            generator_id=payload["generator_id"],
            source_cluster_token=payload["source_cluster_token"],
            evidence_count=payload["evidence_count"],
            status=status,
            registered_identity_id=payload["registered_identity_id"],
            superseded_by_generated_identity_id=payload[
                "superseded_by_generated_identity_id"
            ],
        )


@dataclass(frozen=True, slots=True)
class GeneratedIdentityRegistry:
    records: tuple[GeneratedIdentityRecord, ...]
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported generated identity registry schema")
        if len(self.records) > _MAXIMUM_RECORDS:
            raise ValueError("generated identity registry exceeds the size limit")
        records = tuple(
            sorted(self.records, key=lambda record: record.generated_identity_id)
        )
        object.__setattr__(self, "records", records)
        ids = {record.generated_identity_id for record in records}
        keys = {
            (record.generator_id, record.source_cluster_token) for record in records
        }
        if len(ids) != len(records) or len(keys) != len(records):
            raise ValueError("duplicate generated identity record")
        for record in records:
            successor = record.superseded_by_generated_identity_id
            if successor is not None and successor not in ids:
                raise ValueError("superseded generated identity target is absent")
            if record.registered_identity_id in ids:
                raise ValueError(
                    "registered identity target uses the generated namespace"
                )
        successors = {
            record.generated_identity_id: record.superseded_by_generated_identity_id
            for record in records
            if record.superseded_by_generated_identity_id is not None
        }
        for generated_id in successors:
            visited: set[str] = set()
            current: str | None = generated_id
            while current is not None:
                if current in visited:
                    raise ValueError(
                        "generated identity supersession graph contains a cycle"
                    )
                visited.add(current)
                current = successors.get(current)

    def to_dict(self) -> dict[str, Any]:
        records = sorted(self.records, key=lambda record: record.generated_identity_id)
        return {
            "schema_version": self.schema_version,
            "namespace_uuid": str(GENERATED_DOG_NAMESPACE),
            "generated_identities": [record.to_dict() for record in records],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GeneratedIdentityRegistry:
        _require_exact_keys(
            payload,
            {"schema_version", "namespace_uuid", "generated_identities"},
            "generated identity registry",
        )
        if payload["schema_version"] != _SCHEMA_VERSION:
            raise ValueError("unsupported generated identity registry schema")
        if payload["namespace_uuid"] != str(GENERATED_DOG_NAMESPACE):
            raise ValueError("generated identity namespace_uuid is invalid")
        values = payload["generated_identities"]
        if not isinstance(values, list):
            raise TypeError("generated_identities must be an array")
        records = tuple(GeneratedIdentityRecord.from_dict(value) for value in values)
        if list(records) != sorted(
            records, key=lambda record: record.generated_identity_id
        ):
            raise ValueError("generated identity records are not canonically ordered")
        return cls(records=records, schema_version=payload["schema_version"])


def create_provisional_identity(
    generator_id: str,
    source_cluster_id: str,
    evidence_count: int,
) -> GeneratedIdentityRecord:
    source_cluster_token = compute_source_cluster_token(source_cluster_id)
    return GeneratedIdentityRecord(
        generated_identity_id=compute_generated_identity_id(
            generator_id, source_cluster_token
        ),
        generator_id=generator_id,
        source_cluster_token=source_cluster_token,
        evidence_count=evidence_count,
    )


def transition_generated_identity(
    record: GeneratedIdentityRecord,
    status: GeneratedIdentityStatus,
    *,
    registered_identity_id: str | None = None,
    superseded_by_generated_identity_id: str | None = None,
) -> GeneratedIdentityRecord:
    if record.status is not GeneratedIdentityStatus.PROVISIONAL:
        raise ValueError("only a provisional generated identity can transition")
    if status is GeneratedIdentityStatus.PROVISIONAL:
        raise ValueError("transition target must be a terminal status")
    return GeneratedIdentityRecord(
        generated_identity_id=record.generated_identity_id,
        generator_id=record.generator_id,
        source_cluster_token=record.source_cluster_token,
        evidence_count=record.evidence_count,
        status=status,
        registered_identity_id=registered_identity_id,
        superseded_by_generated_identity_id=superseded_by_generated_identity_id,
    )


def _validate_generator_id(value: object) -> None:
    if not isinstance(value, str) or _GENERATOR_ID.fullmatch(value) is None:
        raise ValueError("generator_id is not canonical")


def _validate_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _validate_uuid5(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical UUIDv5")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical UUIDv5") from exc
    if parsed.version != 5 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUIDv5")


def _require_exact_keys(payload: object, expected: set[str], name: str) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be an object")
    if set(payload) != expected:
        raise ValueError(f"{name} fields differ")


__all__ = [
    "GENERATED_DOG_NAMESPACE",
    "GeneratedIdentityRecord",
    "GeneratedIdentityRegistry",
    "GeneratedIdentityStatus",
    "compute_generated_identity_id",
    "compute_source_cluster_token",
    "create_provisional_identity",
    "transition_generated_identity",
]
