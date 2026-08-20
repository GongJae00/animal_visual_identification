"""Role-exposure closure for protected public split construction.

History is accepted only through explicit declarations.  This module verifies
their receipts and source links, constrains proposed split roles, and validates
the complete candidate assignment before it can be published.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from enrollment.registry.identity_registry import compute_public_subject_token
from shared.foundation.provenance import content_sha256
from evaluation.splits.role_exposure import (
    CandidateRoleAssignment,
    CandidateRoleRecord,
    ExposureDeclarationKind,
    ExposureStage,
    RoleExposureDeclaration,
    RoleExposureDeclarationRecord,
    RoleExposureLedger,
    RoleExposureReceipt,
    RoleExposureRecord,
    create_role_exposure_receipt,
    merge_role_exposure_declarations,
    validate_candidate_assignment,
    verify_role_exposure_receipt,
)


_STAGE_RANK = {stage: index for index, stage in enumerate(ExposureStage)}
_ROLE_STAGE = {
    "YT_FIT": ExposureStage.MODEL_TRAINING_USED,
    "YT_DEVELOPMENT": ExposureStage.MODEL_SELECTION_SCORED,
    "YT_CALIBRATION_KNOWN": ExposureStage.CALIBRATION_SCORED,
    "YT_CALIBRATION_UNKNOWN": ExposureStage.CALIBRATION_SCORED,
    "YT_TEST_KNOWN": ExposureStage.FINAL_TEST_SCORED,
    "YT_TEST_UNKNOWN": ExposureStage.FINAL_TEST_SCORED,
    "DOGFACE_FIT": ExposureStage.MODEL_TRAINING_USED,
    "DOGFACE_DEVELOPMENT": ExposureStage.MODEL_SELECTION_SCORED,
    "DOGFACE_CALIBRATION": ExposureStage.CALIBRATION_SCORED,
    "DOGFACE_TEST": ExposureStage.FINAL_TEST_SCORED,
    "MPDD_EXTERNAL_KNOWN": ExposureStage.FINAL_TEST_SCORED,
    "MPDD_EXTERNAL_UNKNOWN": ExposureStage.FINAL_TEST_SCORED,
    "MPDD_EXTERNAL_UNUSED_TRAIN_DOMAIN": ExposureStage.FINAL_TEST_SCORED,
    "SIBETAN_EXTERNAL_KNOWN": ExposureStage.FINAL_TEST_SCORED,
    "SIBETAN_EXTERNAL_UNKNOWN": ExposureStage.FINAL_TEST_SCORED,
}


def verify_split_role_exposure_inputs(
    source_samples: Iterable[Any],
    ledger: RoleExposureLedger,
    receipt: RoleExposureReceipt,
) -> dict[str, ExposureStage]:
    """Authenticate history and return its maximum stage per source identity."""

    verify_role_exposure_receipt(ledger, receipt)
    source_by_sample, source_by_identity, source_by_subject = _source_links(
        source_samples
    )
    maximum: dict[str, ExposureStage] = {}
    for record in ledger.records:
        source = source_by_sample.get(record.sample_token)
        if source is None:
            raise ValueError("role exposure ledger references an unknown source sample")
        subject = compute_public_subject_token(source.dataset_identity_id)
        if (
            record.identity_token != source.identity_token
            or record.public_subject_token != subject
        ):
            raise ValueError("role exposure ledger source links differ")
        if source_by_identity[record.identity_token] != subject or (
            source_by_subject[record.public_subject_token] != record.identity_token
        ):
            raise ValueError("role exposure ledger identity links differ")
        prior = maximum.get(record.identity_token)
        if prior is None or _STAGE_RANK[record.maximum_historical_stage] > _STAGE_RANK[
            prior
        ]:
            maximum[record.identity_token] = record.maximum_historical_stage
    return maximum


def role_allows_historical_stage(
    role: str, historical_stage: ExposureStage | None
) -> bool:
    try:
        assigned_stage = _ROLE_STAGE[role]
    except KeyError as exc:
        raise ValueError(f"unknown protected split role: {role}") from exc
    return historical_stage is None or (
        _STAGE_RANK[assigned_stage] >= _STAGE_RANK[historical_stage]
    )


def validate_split_candidate_assignment(
    *,
    source_samples: Iterable[Any],
    assignment: Mapping[str, Any],
    ledger: RoleExposureLedger,
) -> None:
    """Validate all published records against source labels and exposure history."""

    records = assignment.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("candidate protected split records must not be empty")
    source_by_sample, _, _ = _source_links(source_samples)
    candidate_records: list[CandidateRoleRecord] = []
    seen: set[str] = set()
    for value in records:
        if not isinstance(value, dict):
            raise TypeError("candidate protected split record must be an object")
        sample_token = value.get("sample_token")
        source = source_by_sample.get(sample_token)
        if source is None:
            raise ValueError("candidate protected split references an unknown source sample")
        if sample_token in seen:
            raise ValueError("candidate protected split repeats a source sample")
        seen.add(sample_token)
        if value.get("identity_token") != source.identity_token:
            raise ValueError("candidate protected split source identity differs")
        role = value.get("identity_role")
        if not isinstance(role, str) or role not in _ROLE_STAGE:
            raise ValueError("candidate protected split role differs")
        candidate_records.append(
            CandidateRoleRecord(
                sample_token=sample_token,
                identity_token=source.identity_token,
                public_subject_token=compute_public_subject_token(
                    source.dataset_identity_id
                ),
                assigned_stage=_ROLE_STAGE[role],
            )
        )
    candidate = CandidateRoleAssignment(
        source_artifact_sha256=content_sha256(dict(assignment)),
        records=tuple(
            sorted(
                candidate_records,
                key=lambda item: (
                    item.sample_token,
                    item.identity_token,
                    item.public_subject_token,
                ),
            )
        ),
    )
    validate_candidate_assignment(ledger, candidate)


def verify_declaration_source_links(
    declaration: RoleExposureDeclaration,
    source_artifact: Mapping[str, Any],
) -> None:
    """Verify that an explicit declaration is bound to matching artifact rows."""

    if declaration.source_artifact_sha256 != content_sha256(dict(source_artifact)):
        raise ValueError("role exposure declaration source artifact hash differs")
    records = source_artifact.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("declared source artifact records must not be empty")
    by_sample: dict[str, dict[str, Any]] = {}
    for value in records:
        if not isinstance(value, dict):
            raise TypeError("declared source artifact record must be an object")
        sample_token = value.get("sample_token")
        if not isinstance(sample_token, str) or sample_token in by_sample:
            raise ValueError("declared source artifact sample links differ")
        by_sample[sample_token] = value
    for record in declaration.records:
        source = by_sample.get(record.sample_token)
        if source is None:
            raise ValueError("role exposure declaration source links differ")
        source_identity = source.get("identity_token")
        if source_identity is not None and source_identity != record.identity_token:
            raise ValueError("role exposure declaration source links differ")
        source_subject = source.get("public_subject_token")
        if source_subject is not None and source_subject != record.public_subject_token:
            raise ValueError("role exposure declaration public subject link differs")
        source_stage = source.get("stage")
        if source_stage is not None and source_stage != record.stage.value:
            raise ValueError("role exposure declaration stage link differs")


def _source_links(
    samples: Iterable[Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    by_sample: dict[str, Any] = {}
    subject_by_identity: dict[str, str] = {}
    identity_by_subject: dict[str, str] = {}
    for sample in samples:
        sample_token = getattr(sample, "sample_token", None)
        identity_token = getattr(sample, "identity_token", None)
        dataset_identity_id = getattr(sample, "dataset_identity_id", None)
        if not all(isinstance(value, str) and value for value in (
            sample_token,
            identity_token,
            dataset_identity_id,
        )):
            raise ValueError("public split source link fields differ")
        if sample_token in by_sample:
            raise ValueError("public split source repeats a sample token")
        subject = compute_public_subject_token(dataset_identity_id)
        if len({sample_token, identity_token, subject}) != 3:
            raise ValueError("public split token namespaces are not separated")
        if subject_by_identity.setdefault(identity_token, subject) != subject:
            raise ValueError("public split identity maps to conflicting public subjects")
        if identity_by_subject.setdefault(subject, identity_token) != identity_token:
            raise ValueError("public subject maps to conflicting split identities")
        by_sample[sample_token] = sample
    if not by_sample:
        raise ValueError("public split source must not be empty")
    return by_sample, subject_by_identity, identity_by_subject


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
    "role_allows_historical_stage",
    "validate_split_candidate_assignment",
    "verify_declaration_source_links",
    "verify_role_exposure_receipt",
    "verify_split_role_exposure_inputs",
]
