"""Durable, declaration-only history of public-data role exposure.

The ledger records what callers explicitly declare about prior assignment and
evaluation artifacts.  It does not scan storage, infer undeclared use, or let
artifact revocation erase historical exposure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable

from cvi.provenance import content_sha256


_DECLARATION_INTERPRETATION = "EXPLICIT_DECLARATION_ONLY_NO_AUTOMATIC_DISCOVERY"
_LEDGER_INTERPRETATION = "MONOTONIC_HISTORY_INCLUDING_REVOKED_ARTIFACTS"
_RECEIPT_INTERPRETATION = "BINDS_ALL_DECLARED_SOURCE_ARTIFACT_HASHES"
_CANDIDATE_INTERPRETATION = "PROPOSED_ROLE_ONLY_NOT_RECORDED_EXPOSURE"


class ExposureStage(StrEnum):
    """Ordered maximum historical exposure for a public sample or subject."""

    BYTES_EXPORTED = "BYTES_EXPORTED"
    MODEL_TRAINING_USED = "MODEL_TRAINING_USED"
    MODEL_SELECTION_SCORED = "MODEL_SELECTION_SCORED"
    CALIBRATION_SCORED = "CALIBRATION_SCORED"
    FINAL_TEST_SCORED = "FINAL_TEST_SCORED"


_STAGE_RANK = {stage: rank for rank, stage in enumerate(ExposureStage)}


class ExposureDeclarationKind(StrEnum):
    PRIOR_ASSIGNMENT = "PRIOR_ASSIGNMENT"
    PRIOR_EVALUATION = "PRIOR_EVALUATION"


@dataclass(frozen=True, slots=True)
class RoleExposureDeclarationRecord:
    sample_token: str
    identity_token: str
    public_subject_token: str
    stage: ExposureStage
    schema_version: str = "cvi.role_exposure_declaration_record.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.role_exposure_declaration_record.v1":
            raise ValueError("unsupported role exposure declaration record schema")
        _require_entity_tokens(
            self.sample_token,
            self.identity_token,
            self.public_subject_token,
        )
        if not isinstance(self.stage, ExposureStage):
            raise TypeError("declaration stage must be ExposureStage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_token": self.sample_token,
            "identity_token": self.identity_token,
            "public_subject_token": self.public_subject_token,
            "stage": self.stage.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RoleExposureDeclarationRecord:
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            "role exposure declaration record",
        )
        values = dict(payload)
        values["stage"] = ExposureStage(values["stage"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RoleExposureDeclaration:
    source_artifact_sha256: str
    kind: ExposureDeclarationKind
    revoked: bool
    records: tuple[RoleExposureDeclarationRecord, ...]
    interpretation: str = _DECLARATION_INTERPRETATION
    schema_version: str = "cvi.role_exposure_declaration.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.role_exposure_declaration.v1":
            raise ValueError("unsupported role exposure declaration schema")
        _require_sha256(self.source_artifact_sha256, "source_artifact_sha256")
        if not isinstance(self.kind, ExposureDeclarationKind):
            raise TypeError("declaration kind must be ExposureDeclarationKind")
        if not isinstance(self.revoked, bool):
            raise TypeError("revoked must be boolean")
        if self.interpretation != _DECLARATION_INTERPRETATION:
            raise ValueError("role exposure declaration interpretation differs")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("role exposure declaration records must not be empty")
        if any(
            not isinstance(record, RoleExposureDeclarationRecord)
            for record in self.records
        ):
            raise TypeError("declaration records must be RoleExposureDeclarationRecord")
        if tuple(sorted(self.records, key=_declaration_record_key)) != self.records:
            raise ValueError("declaration records must be canonically sorted")
        _require_unique(
            tuple(record.sample_token for record in self.records),
            "declaration sample tokens",
        )
        _validate_token_links(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_artifact_sha256": self.source_artifact_sha256,
            "kind": self.kind.value,
            "revoked": self.revoked,
            "records": [record.to_dict() for record in self.records],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RoleExposureDeclaration:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "source_artifact_sha256",
                "kind",
                "revoked",
                "records",
                "interpretation",
            },
            "role exposure declaration",
        )
        if not isinstance(payload["records"], list):
            raise TypeError("role exposure declaration records must be a list")
        return cls(
            source_artifact_sha256=payload["source_artifact_sha256"],
            kind=ExposureDeclarationKind(payload["kind"]),
            revoked=payload["revoked"],
            records=tuple(
                RoleExposureDeclarationRecord.from_dict(record)
                for record in payload["records"]
            ),
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class RoleExposureRecord:
    sample_token: str
    identity_token: str
    public_subject_token: str
    maximum_historical_stage: ExposureStage
    source_artifact_sha256s: tuple[str, ...]
    schema_version: str = "cvi.role_exposure_record.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.role_exposure_record.v1":
            raise ValueError("unsupported role exposure record schema")
        _require_entity_tokens(
            self.sample_token,
            self.identity_token,
            self.public_subject_token,
        )
        if not isinstance(self.maximum_historical_stage, ExposureStage):
            raise TypeError("maximum_historical_stage must be ExposureStage")
        _require_sorted_sha256s(
            self.source_artifact_sha256s, "record source artifact hashes"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_token": self.sample_token,
            "identity_token": self.identity_token,
            "public_subject_token": self.public_subject_token,
            "maximum_historical_stage": self.maximum_historical_stage.value,
            "source_artifact_sha256s": list(self.source_artifact_sha256s),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RoleExposureRecord:
        _require_exact_keys(
            payload, set(cls.__dataclass_fields__), "role exposure record"
        )
        if not isinstance(payload["source_artifact_sha256s"], list):
            raise TypeError("record source artifact hashes must be a list")
        return cls(
            sample_token=payload["sample_token"],
            identity_token=payload["identity_token"],
            public_subject_token=payload["public_subject_token"],
            maximum_historical_stage=ExposureStage(
                payload["maximum_historical_stage"]
            ),
            source_artifact_sha256s=tuple(payload["source_artifact_sha256s"]),
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class RoleExposureLedger:
    declarations: tuple[RoleExposureDeclaration, ...]
    records: tuple[RoleExposureRecord, ...]
    interpretation: str = _LEDGER_INTERPRETATION
    schema_version: str = "cvi.role_exposure_ledger.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.role_exposure_ledger.v1":
            raise ValueError("unsupported role exposure ledger schema")
        if self.interpretation != _LEDGER_INTERPRETATION:
            raise ValueError("role exposure ledger interpretation differs")
        if not isinstance(self.declarations, tuple) or not self.declarations:
            raise ValueError("role exposure ledger declarations must not be empty")
        if any(
            not isinstance(declaration, RoleExposureDeclaration)
            for declaration in self.declarations
        ):
            raise TypeError("ledger declarations must be RoleExposureDeclaration")
        if tuple(
            sorted(
                self.declarations,
                key=lambda declaration: declaration.source_artifact_sha256,
            )
        ) != self.declarations:
            raise ValueError("ledger declarations must be canonically sorted")
        _require_unique(
            tuple(
                declaration.source_artifact_sha256
                for declaration in self.declarations
            ),
            "ledger source artifact hashes",
        )
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("role exposure ledger records must not be empty")
        if any(not isinstance(record, RoleExposureRecord) for record in self.records):
            raise TypeError("ledger records must be RoleExposureRecord")
        if tuple(sorted(self.records, key=_record_key)) != self.records:
            raise ValueError("ledger records must be canonically sorted")
        _require_unique(
            tuple(record.sample_token for record in self.records),
            "ledger sample tokens",
        )
        if self.records != _aggregate_records(self.declarations):
            raise ValueError("ledger records differ from exact source declarations")

    @property
    def ledger_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "declarations": [item.to_dict() for item in self.declarations],
            "records": [item.to_dict() for item in self.records],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RoleExposureLedger:
        _require_exact_keys(
            payload,
            {"schema_version", "declarations", "records", "interpretation"},
            "role exposure ledger",
        )
        if not isinstance(payload["declarations"], list) or not isinstance(
            payload["records"], list
        ):
            raise TypeError("ledger declarations and records must be lists")
        return cls(
            declarations=tuple(
                RoleExposureDeclaration.from_dict(item)
                for item in payload["declarations"]
            ),
            records=tuple(RoleExposureRecord.from_dict(item) for item in payload["records"]),
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class RoleExposureReceipt:
    ledger_sha256: str
    source_artifact_sha256s: tuple[str, ...]
    revoked_source_artifact_sha256s: tuple[str, ...]
    declaration_count: int
    record_count: int
    interpretation: str = _RECEIPT_INTERPRETATION
    schema_version: str = "cvi.role_exposure_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.role_exposure_receipt.v1":
            raise ValueError("unsupported role exposure receipt schema")
        _require_sha256(self.ledger_sha256, "ledger_sha256")
        _require_sorted_sha256s(
            self.source_artifact_sha256s, "receipt source artifact hashes"
        )
        if not isinstance(self.revoked_source_artifact_sha256s, tuple):
            raise TypeError("revoked source artifact hashes must be a tuple")
        for value in self.revoked_source_artifact_sha256s:
            _require_sha256(value, "revoked source artifact hash")
        if tuple(sorted(self.revoked_source_artifact_sha256s)) != (
            self.revoked_source_artifact_sha256s
        ) or len(self.revoked_source_artifact_sha256s) != len(
            set(self.revoked_source_artifact_sha256s)
        ):
            raise ValueError("revoked source artifact hashes must be sorted and unique")
        if not set(self.revoked_source_artifact_sha256s).issubset(
            self.source_artifact_sha256s
        ):
            raise ValueError("revoked artifact hash is absent from all source hashes")
        _require_positive_int(self.declaration_count, "declaration_count")
        _require_positive_int(self.record_count, "record_count")
        if self.declaration_count != len(self.source_artifact_sha256s):
            raise ValueError("receipt declaration count differs from source hashes")
        if self.interpretation != _RECEIPT_INTERPRETATION:
            raise ValueError("role exposure receipt interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ledger_sha256": self.ledger_sha256,
            "source_artifact_sha256s": list(self.source_artifact_sha256s),
            "revoked_source_artifact_sha256s": list(
                self.revoked_source_artifact_sha256s
            ),
            "declaration_count": self.declaration_count,
            "record_count": self.record_count,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RoleExposureReceipt:
        _require_exact_keys(
            payload, set(cls.__dataclass_fields__), "role exposure receipt"
        )
        if not isinstance(payload["source_artifact_sha256s"], list) or not isinstance(
            payload["revoked_source_artifact_sha256s"], list
        ):
            raise TypeError("receipt artifact hashes must be lists")
        return cls(
            ledger_sha256=payload["ledger_sha256"],
            source_artifact_sha256s=tuple(payload["source_artifact_sha256s"]),
            revoked_source_artifact_sha256s=tuple(
                payload["revoked_source_artifact_sha256s"]
            ),
            declaration_count=payload["declaration_count"],
            record_count=payload["record_count"],
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class CandidateRoleRecord:
    sample_token: str
    identity_token: str
    public_subject_token: str
    assigned_stage: ExposureStage
    schema_version: str = "cvi.candidate_role_record.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.candidate_role_record.v1":
            raise ValueError("unsupported candidate role record schema")
        _require_entity_tokens(
            self.sample_token,
            self.identity_token,
            self.public_subject_token,
        )
        if not isinstance(self.assigned_stage, ExposureStage):
            raise TypeError("assigned_stage must be ExposureStage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_token": self.sample_token,
            "identity_token": self.identity_token,
            "public_subject_token": self.public_subject_token,
            "assigned_stage": self.assigned_stage.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CandidateRoleRecord:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "candidate role record")
        values = dict(payload)
        values["assigned_stage"] = ExposureStage(values["assigned_stage"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CandidateRoleAssignment:
    source_artifact_sha256: str
    records: tuple[CandidateRoleRecord, ...]
    interpretation: str = _CANDIDATE_INTERPRETATION
    schema_version: str = "cvi.candidate_role_assignment.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.candidate_role_assignment.v1":
            raise ValueError("unsupported candidate role assignment schema")
        _require_sha256(self.source_artifact_sha256, "source_artifact_sha256")
        if self.interpretation != _CANDIDATE_INTERPRETATION:
            raise ValueError("candidate role assignment interpretation differs")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("candidate role assignment records must not be empty")
        if any(not isinstance(record, CandidateRoleRecord) for record in self.records):
            raise TypeError("candidate records must be CandidateRoleRecord")
        if tuple(sorted(self.records, key=_candidate_record_key)) != self.records:
            raise ValueError("candidate records must be canonically sorted")
        _require_unique(
            tuple(record.sample_token for record in self.records),
            "candidate sample tokens",
        )
        _validate_token_links(self.records)
        _require_one_stage_per_identity(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_artifact_sha256": self.source_artifact_sha256,
            "records": [record.to_dict() for record in self.records],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CandidateRoleAssignment:
        _require_exact_keys(
            payload,
            {"schema_version", "source_artifact_sha256", "records", "interpretation"},
            "candidate role assignment",
        )
        if not isinstance(payload["records"], list):
            raise TypeError("candidate role assignment records must be a list")
        return cls(
            source_artifact_sha256=payload["source_artifact_sha256"],
            records=tuple(CandidateRoleRecord.from_dict(item) for item in payload["records"]),
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


def merge_role_exposure_declarations(
    declarations: Iterable[RoleExposureDeclaration],
) -> RoleExposureLedger:
    """Merge explicit declarations deterministically, retaining revoked inputs."""

    values = tuple(declarations)
    if not values:
        raise ValueError("at least one explicit role exposure declaration is required")
    if any(not isinstance(value, RoleExposureDeclaration) for value in values):
        raise TypeError("declarations must contain RoleExposureDeclaration values")
    ordered = tuple(sorted(values, key=lambda item: item.source_artifact_sha256))
    _require_unique(
        tuple(item.source_artifact_sha256 for item in ordered),
        "source artifact hashes",
    )
    records = _aggregate_records(ordered)
    return RoleExposureLedger(ordered, records)


def create_role_exposure_receipt(ledger: RoleExposureLedger) -> RoleExposureReceipt:
    if not isinstance(ledger, RoleExposureLedger):
        raise TypeError("ledger must be RoleExposureLedger")
    source_hashes = tuple(
        declaration.source_artifact_sha256 for declaration in ledger.declarations
    )
    return RoleExposureReceipt(
        ledger_sha256=ledger.ledger_sha256,
        source_artifact_sha256s=source_hashes,
        revoked_source_artifact_sha256s=tuple(
            declaration.source_artifact_sha256
            for declaration in ledger.declarations
            if declaration.revoked
        ),
        declaration_count=len(ledger.declarations),
        record_count=len(ledger.records),
    )


def verify_role_exposure_receipt(
    ledger: RoleExposureLedger, receipt: RoleExposureReceipt
) -> None:
    if not isinstance(ledger, RoleExposureLedger):
        raise TypeError("ledger must be RoleExposureLedger")
    if not isinstance(receipt, RoleExposureReceipt):
        raise TypeError("receipt must be RoleExposureReceipt")
    if receipt != create_role_exposure_receipt(ledger):
        raise ValueError("role exposure receipt differs from ledger and source artifacts")


def validate_candidate_assignment(
    ledger: RoleExposureLedger, candidate: CandidateRoleAssignment
) -> None:
    """Reject any sample, identity, or public subject moved to a lower stage."""

    if not isinstance(ledger, RoleExposureLedger):
        raise TypeError("ledger must be RoleExposureLedger")
    if not isinstance(candidate, CandidateRoleAssignment):
        raise TypeError("candidate must be CandidateRoleAssignment")

    sample_history = {record.sample_token: record for record in ledger.records}
    identity_history = _maximum_by_token(ledger.records, "identity_token")
    subject_history = _maximum_by_token(ledger.records, "public_subject_token")
    identity_links = {
        record.identity_token: record.public_subject_token for record in ledger.records
    }
    subject_links = {
        record.public_subject_token: record.identity_token for record in ledger.records
    }

    for record in candidate.records:
        prior_sample = sample_history.get(record.sample_token)
        if prior_sample is not None and (
            prior_sample.identity_token != record.identity_token
            or prior_sample.public_subject_token != record.public_subject_token
        ):
            raise ValueError("candidate sample token conflicts with historical identity links")
        if identity_links.get(record.identity_token, record.public_subject_token) != (
            record.public_subject_token
        ) or subject_links.get(record.public_subject_token, record.identity_token) != (
            record.identity_token
        ):
            raise ValueError("candidate identity and public subject links conflict with history")

        historical = [
            prior_sample.maximum_historical_stage if prior_sample is not None else None,
            identity_history.get(record.identity_token),
            subject_history.get(record.public_subject_token),
        ]
        maximum = _maximum_stage(stage for stage in historical if stage is not None)
        if maximum is not None and _STAGE_RANK[record.assigned_stage] < _STAGE_RANK[maximum]:
            raise ValueError(
                f"candidate role regression for {record.sample_token}: "
                f"{maximum.value} -> {record.assigned_stage.value}"
            )


def _aggregate_records(
    declarations: tuple[RoleExposureDeclaration, ...],
) -> tuple[RoleExposureRecord, ...]:
    links: dict[str, tuple[str, str]] = {}
    identity_links: dict[str, str] = {}
    subject_links: dict[str, str] = {}
    stages: dict[str, ExposureStage] = {}
    source_hashes: dict[str, set[str]] = {}
    for declaration in declarations:
        for record in declaration.records:
            link = (record.identity_token, record.public_subject_token)
            if links.setdefault(record.sample_token, link) != link:
                raise ValueError("sample token has conflicting identity declarations")
            if identity_links.setdefault(
                record.identity_token, record.public_subject_token
            ) != record.public_subject_token:
                raise ValueError("identity token has conflicting public subject declarations")
            if subject_links.setdefault(
                record.public_subject_token, record.identity_token
            ) != record.identity_token:
                raise ValueError("public subject token has conflicting identity declarations")
            prior = stages.get(record.sample_token)
            if prior is None or _STAGE_RANK[record.stage] > _STAGE_RANK[prior]:
                stages[record.sample_token] = record.stage
            source_hashes.setdefault(record.sample_token, set()).add(
                declaration.source_artifact_sha256
            )
    return tuple(
        RoleExposureRecord(
            sample_token=sample_token,
            identity_token=links[sample_token][0],
            public_subject_token=links[sample_token][1],
            maximum_historical_stage=stages[sample_token],
            source_artifact_sha256s=tuple(sorted(source_hashes[sample_token])),
        )
        for sample_token in sorted(links)
    )


def _maximum_by_token(
    records: tuple[RoleExposureRecord, ...], attribute: str
) -> dict[str, ExposureStage]:
    result: dict[str, ExposureStage] = {}
    for record in records:
        token = getattr(record, attribute)
        prior = result.get(token)
        if prior is None or _STAGE_RANK[record.maximum_historical_stage] > _STAGE_RANK[prior]:
            result[token] = record.maximum_historical_stage
    return result


def _maximum_stage(stages: Iterable[ExposureStage]) -> ExposureStage | None:
    values = tuple(stages)
    return max(values, key=_STAGE_RANK.__getitem__) if values else None


def _validate_token_links(records: Iterable[Any]) -> None:
    identity_links: dict[str, str] = {}
    subject_links: dict[str, str] = {}
    for record in records:
        if identity_links.setdefault(
            record.identity_token, record.public_subject_token
        ) != record.public_subject_token:
            raise ValueError("identity token maps to conflicting public subjects")
        if subject_links.setdefault(
            record.public_subject_token, record.identity_token
        ) != record.identity_token:
            raise ValueError("public subject token maps to conflicting identities")


def _require_one_stage_per_identity(records: Iterable[CandidateRoleRecord]) -> None:
    stages: dict[str, ExposureStage] = {}
    for record in records:
        if stages.setdefault(record.identity_token, record.assigned_stage) != (
            record.assigned_stage
        ):
            raise ValueError("candidate identity is assigned to conflicting stages")


def _require_entity_tokens(sample: object, identity: object, subject: object) -> None:
    for value, name in (
        (sample, "sample_token"),
        (identity, "identity_token"),
        (subject, "public_subject_token"),
    ):
        _require_sha256(value, name)
    if len({sample, identity, subject}) != 3:
        raise ValueError("sample, identity, and public subject namespaces must be distinct")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


def _require_sorted_sha256s(values: object, name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    for value in values:
        _require_sha256(value, name)
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{name} must be sorted and unique")


def _require_unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_exact_keys(payload: object, expected: set[str], context: str) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _declaration_record_key(
    record: RoleExposureDeclarationRecord,
) -> tuple[str, str, str]:
    return record.sample_token, record.identity_token, record.public_subject_token


def _record_key(record: RoleExposureRecord) -> tuple[str, str, str]:
    return record.sample_token, record.identity_token, record.public_subject_token


def _candidate_record_key(record: CandidateRoleRecord) -> tuple[str, str, str]:
    return record.sample_token, record.identity_token, record.public_subject_token


__all__ = [
    "CandidateRoleAssignment",
    "CandidateRoleRecord",
    "ExposureDeclarationKind",
    "ExposureStage",
    "RoleExposureDeclaration",
    "RoleExposureDeclarationRecord",
    "RoleExposureLedger",
    "RoleExposureReceipt",
    "RoleExposureRecord",
    "create_role_exposure_receipt",
    "merge_role_exposure_declarations",
    "validate_candidate_assignment",
    "verify_role_exposure_receipt",
]
