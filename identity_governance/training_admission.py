"""Fail-closed admission contracts for public-crop model training."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.public_crop_manifest import (
    PublicCropArtifact,
    PublicCropManifest,
    PublicCropVerification,
    verify_public_crop_manifest,
)
from foundation.provenance import content_sha256
from identity_governance.role_exposure import (
    CandidateRoleAssignment,
    CandidateRoleRecord,
    ExposureStage,
    RoleExposureLedger,
    RoleExposureReceipt,
    validate_candidate_assignment,
    verify_role_exposure_receipt,
)

_LANE_ROLE_SUFFIX = {
    "MODEL_TRAINING": "_FIT",
    "MODEL_SELECTION": "_DEVELOPMENT",
}
_FORBIDDEN_MARKERS = ("CALIBRATION", "TEST", "EXTERNAL", "RANDOM_BACKGROUND")


@dataclass(frozen=True, slots=True)
class TrainingCropRow:
    """One admitted sample bound to an exact immutable crop artifact."""

    sample_token: str
    identity_token: str
    public_subject_token: str
    component_token: str
    lane: str
    role: str
    crop_relative_path: str
    crop_artifact_sha256: str
    schema_version: str = "cvi.training_crop_row.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.training_crop_row.v1":
            raise ValueError("unsupported training crop row schema")
        for name in (
            "sample_token",
            "identity_token",
            "public_subject_token",
            "component_token",
            "crop_artifact_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if len({self.sample_token, self.identity_token, self.public_subject_token}) != 3:
            raise ValueError(
                "sample, identity, and public subject namespaces must be distinct"
            )
        if self.lane not in _LANE_ROLE_SUFFIX:
            raise ValueError("training lane must be MODEL_TRAINING or MODEL_SELECTION")
        _require_text(self.role, "role")
        if any(marker in self.role.upper() for marker in _FORBIDDEN_MARKERS):
            raise ValueError("calibration, test, external, and random-background roles are forbidden")
        if not self.role.endswith(_LANE_ROLE_SUFFIX[self.lane]):
            raise ValueError("training role does not match its lane")
        _require_text(self.crop_relative_path, "crop_relative_path", maximum=512)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingCropRow:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "training crop row")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class TrainingAdmissionManifest:
    split_receipt_sha256: str
    crop_manifest_sha256: str
    crop_receipt_sha256: str
    exposure_receipt_sha256: str
    model_receipt_sha256: str
    rows: tuple[TrainingCropRow, ...]
    interpretation: str = "PUBLIC_SUBJECT_TRAINING_ONLY_NO_REGISTERED_DOG_ID"
    schema_version: str = "cvi.training_admission_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.training_admission_manifest.v1":
            raise ValueError("unsupported training admission manifest schema")
        for name in (
            "split_receipt_sha256",
            "crop_manifest_sha256",
            "crop_receipt_sha256",
            "exposure_receipt_sha256",
            "model_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.interpretation != "PUBLIC_SUBJECT_TRAINING_ONLY_NO_REGISTERED_DOG_ID":
            raise ValueError("training admission identity interpretation differs")
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("training admission rows must not be empty")
        if any(not isinstance(row, TrainingCropRow) for row in self.rows):
            raise TypeError("training admission rows must be TrainingCropRow")
        if tuple(sorted(self.rows, key=lambda row: row.sample_token)) != self.rows:
            raise ValueError("training rows must be sorted by sample token")
        _require_unique(tuple(row.sample_token for row in self.rows), "sample tokens")
        _require_unique(tuple(row.crop_relative_path for row in self.rows), "crop paths")
        lanes = {row.lane for row in self.rows}
        if lanes != set(_LANE_ROLE_SUFFIX):
            raise ValueError("training admission requires train and development rows")
        _verify_train_development_disjointness(self.rows)

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_receipt_sha256": self.split_receipt_sha256,
            "crop_manifest_sha256": self.crop_manifest_sha256,
            "crop_receipt_sha256": self.crop_receipt_sha256,
            "exposure_receipt_sha256": self.exposure_receipt_sha256,
            "model_receipt_sha256": self.model_receipt_sha256,
            "rows": [row.to_dict() for row in self.rows],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingAdmissionManifest:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "split_receipt_sha256",
                "crop_manifest_sha256",
                "crop_receipt_sha256",
                "exposure_receipt_sha256",
                "model_receipt_sha256",
                "rows",
                "interpretation",
            },
            "training admission manifest",
        )
        if not isinstance(payload["rows"], list):
            raise TypeError("training admission rows must be a list")
        return cls(
            split_receipt_sha256=payload["split_receipt_sha256"],
            crop_manifest_sha256=payload["crop_manifest_sha256"],
            crop_receipt_sha256=payload["crop_receipt_sha256"],
            exposure_receipt_sha256=payload["exposure_receipt_sha256"],
            model_receipt_sha256=payload["model_receipt_sha256"],
            rows=tuple(TrainingCropRow.from_dict(row) for row in payload["rows"]),
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class TrainingAdmissionReceipt:
    admission_manifest_sha256: str
    split_receipt_sha256: str
    crop_manifest_sha256: str
    crop_receipt_sha256: str
    exposure_receipt_sha256: str
    model_receipt_sha256: str
    crop_verification: PublicCropVerification
    admitted_rows: int
    admitted_public_subjects: int
    state: str = "PASS"
    schema_version: str = "cvi.training_admission_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.training_admission_receipt.v1":
            raise ValueError("unsupported training admission receipt schema")
        for name in (
            "admission_manifest_sha256",
            "split_receipt_sha256",
            "crop_manifest_sha256",
            "crop_receipt_sha256",
            "exposure_receipt_sha256",
            "model_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.crop_verification, PublicCropVerification):
            raise TypeError("crop_verification must be PublicCropVerification")
        if self.crop_verification.state != "PASS" or (
            self.crop_verification.crop_manifest_sha256 != self.crop_manifest_sha256
        ) or self.crop_verification.receipt_sha256 != self.crop_receipt_sha256:
            raise ValueError("training receipt crop verification differs")
        for name in ("admitted_rows", "admitted_public_subjects"):
            _require_positive_int(getattr(self, name), name)
        if self.admitted_public_subjects > self.admitted_rows:
            raise ValueError("admitted subject count exceeds row count")
        if self.state != "PASS":
            raise ValueError("training admission state must be PASS")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "admission_manifest_sha256": self.admission_manifest_sha256,
            "split_receipt_sha256": self.split_receipt_sha256,
            "crop_manifest_sha256": self.crop_manifest_sha256,
            "crop_receipt_sha256": self.crop_receipt_sha256,
            "exposure_receipt_sha256": self.exposure_receipt_sha256,
            "model_receipt_sha256": self.model_receipt_sha256,
            "crop_verification": self.crop_verification.to_dict(),
            "admitted_rows": self.admitted_rows,
            "admitted_public_subjects": self.admitted_public_subjects,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingAdmissionReceipt:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "admission_manifest_sha256",
                "split_receipt_sha256",
                "crop_manifest_sha256",
                "crop_receipt_sha256",
                "exposure_receipt_sha256",
                "model_receipt_sha256",
                "crop_verification",
                "admitted_rows",
                "admitted_public_subjects",
                "state",
            },
            "training admission receipt",
        )
        return cls(
            admission_manifest_sha256=payload["admission_manifest_sha256"],
            split_receipt_sha256=payload["split_receipt_sha256"],
            crop_manifest_sha256=payload["crop_manifest_sha256"],
            crop_receipt_sha256=payload["crop_receipt_sha256"],
            exposure_receipt_sha256=payload["exposure_receipt_sha256"],
            model_receipt_sha256=payload["model_receipt_sha256"],
            crop_verification=PublicCropVerification.from_dict(
                payload["crop_verification"]
            ),
            admitted_rows=payload["admitted_rows"],
            admitted_public_subjects=payload["admitted_public_subjects"],
            state=payload["state"],
            schema_version=payload["schema_version"],
        )


def admit_training(
    manifest: TrainingAdmissionManifest,
    crop_manifest: PublicCropManifest,
    *,
    crop_root: Path,
    exposure_ledger: RoleExposureLedger,
    exposure_receipt: RoleExposureReceipt,
    expected_sample_tokens: tuple[str, ...],
    expected_split_receipt_sha256: str,
    expected_crop_receipt_sha256: str,
    expected_exposure_receipt_sha256: str,
    expected_model_receipt_sha256: str,
    verify_decoded_rgb_pixels: bool = True,
) -> TrainingAdmissionReceipt:
    """Admit only an exact, receipt-bound train/development crop population."""

    if not isinstance(manifest, TrainingAdmissionManifest):
        raise TypeError("manifest must be TrainingAdmissionManifest")
    if not isinstance(crop_manifest, PublicCropManifest):
        raise TypeError("crop_manifest must be PublicCropManifest")
    verify_role_exposure_receipt(exposure_ledger, exposure_receipt)
    for value, name in (
        (expected_split_receipt_sha256, "expected_split_receipt_sha256"),
        (expected_crop_receipt_sha256, "expected_crop_receipt_sha256"),
        (expected_exposure_receipt_sha256, "expected_exposure_receipt_sha256"),
        (expected_model_receipt_sha256, "expected_model_receipt_sha256"),
    ):
        _require_sha256(value, name)
    if manifest.split_receipt_sha256 != expected_split_receipt_sha256:
        raise ValueError("split receipt hash mismatch")
    if manifest.crop_receipt_sha256 != expected_crop_receipt_sha256:
        raise ValueError("crop receipt hash mismatch")
    if manifest.exposure_receipt_sha256 != expected_exposure_receipt_sha256:
        raise ValueError("exposure receipt hash mismatch")
    if exposure_receipt.receipt_sha256 != expected_exposure_receipt_sha256:
        raise ValueError("actual exposure receipt hash mismatch")
    if manifest.model_receipt_sha256 != expected_model_receipt_sha256:
        raise ValueError("model receipt hash mismatch")
    if manifest.crop_manifest_sha256 != crop_manifest.manifest_sha256:
        raise ValueError("crop manifest hash mismatch")

    if not isinstance(expected_sample_tokens, tuple) or not expected_sample_tokens:
        raise ValueError("expected_sample_tokens must be a non-empty tuple")
    for token in expected_sample_tokens:
        _require_sha256(token, "expected sample token")
    _require_unique(expected_sample_tokens, "expected sample tokens")

    artifacts_by_sample = {item.sample_token: item for item in crop_manifest.artifacts}
    rows_by_sample = {row.sample_token: row for row in manifest.rows}
    _require_exact_population(
        "training row/crop sample tokens", set(rows_by_sample), set(artifacts_by_sample)
    )
    _require_exact_population(
        "expected/training sample tokens", set(expected_sample_tokens), set(rows_by_sample)
    )
    crop_paths = {item.relative_path for item in crop_manifest.artifacts}
    row_paths = {row.crop_relative_path for row in manifest.rows}
    _require_exact_population("training row/crop paths", row_paths, crop_paths)

    for sample_token, row in rows_by_sample.items():
        artifact = artifacts_by_sample[sample_token]
        _verify_row_artifact_binding(row, artifact)
        if artifact.source_variant != "original":
            raise ValueError("random-background and non-original crops are forbidden")
    _verify_train_development_disjointness(manifest.rows)
    candidate = CandidateRoleAssignment(
        source_artifact_sha256=manifest.manifest_sha256,
        records=tuple(
            CandidateRoleRecord(
                sample_token=row.sample_token,
                identity_token=row.identity_token,
                public_subject_token=row.public_subject_token,
                assigned_stage=(
                    ExposureStage.MODEL_TRAINING_USED
                    if row.lane == "MODEL_TRAINING"
                    else ExposureStage.MODEL_SELECTION_SCORED
                ),
            )
            for row in manifest.rows
        ),
    )
    validate_candidate_assignment(exposure_ledger, candidate)

    crop_verification = verify_public_crop_manifest(
        crop_root,
        crop_manifest,
        verify_decoded_rgb_pixels=verify_decoded_rgb_pixels,
    )
    if crop_verification.receipt_sha256 != manifest.crop_receipt_sha256:
        raise ValueError("crop verification receipt hash mismatch")
    return TrainingAdmissionReceipt(
        admission_manifest_sha256=manifest.manifest_sha256,
        split_receipt_sha256=manifest.split_receipt_sha256,
        crop_manifest_sha256=manifest.crop_manifest_sha256,
        crop_receipt_sha256=manifest.crop_receipt_sha256,
        exposure_receipt_sha256=manifest.exposure_receipt_sha256,
        model_receipt_sha256=manifest.model_receipt_sha256,
        crop_verification=crop_verification,
        admitted_rows=len(manifest.rows),
        admitted_public_subjects=len(
            {row.public_subject_token for row in manifest.rows}
        ),
    )


def verify_training_admission_receipt(
    manifest: TrainingAdmissionManifest,
    crop_manifest: PublicCropManifest,
    receipt: TrainingAdmissionReceipt,
    *,
    crop_root: Path,
    exposure_ledger: RoleExposureLedger,
    exposure_receipt: RoleExposureReceipt,
    expected_admission_manifest_sha256: str,
    expected_admission_receipt_sha256: str,
    expected_split_receipt_sha256: str,
    expected_crop_receipt_sha256: str,
    expected_exposure_receipt_sha256: str,
    expected_model_receipt_sha256: str,
) -> TrainingAdmissionReceipt:
    """Recompute admission and match it to an externally pinned receipt."""

    if not isinstance(receipt, TrainingAdmissionReceipt):
        raise TypeError("receipt must be TrainingAdmissionReceipt")
    for value, name in (
        (expected_admission_manifest_sha256, "expected_admission_manifest_sha256"),
        (expected_admission_receipt_sha256, "expected_admission_receipt_sha256"),
    ):
        _require_sha256(value, name)
    if manifest.manifest_sha256 != expected_admission_manifest_sha256:
        raise ValueError("training admission manifest hash mismatch")
    if receipt.receipt_sha256 != expected_admission_receipt_sha256:
        raise ValueError("training admission receipt hash mismatch")

    computed = admit_training(
        manifest,
        crop_manifest,
        crop_root=crop_root,
        exposure_ledger=exposure_ledger,
        exposure_receipt=exposure_receipt,
        expected_sample_tokens=tuple(row.sample_token for row in manifest.rows),
        expected_split_receipt_sha256=expected_split_receipt_sha256,
        expected_crop_receipt_sha256=expected_crop_receipt_sha256,
        expected_exposure_receipt_sha256=expected_exposure_receipt_sha256,
        expected_model_receipt_sha256=expected_model_receipt_sha256,
    )
    if computed != receipt:
        raise ValueError("training admission receipt differs from recomputed admission")
    return receipt


def _verify_row_artifact_binding(
    row: TrainingCropRow, artifact: PublicCropArtifact
) -> None:
    if (
        row.public_subject_token != artifact.public_subject_token
        or row.component_token != artifact.component_token
        or row.crop_relative_path != artifact.relative_path
        or row.crop_artifact_sha256 != artifact.artifact_sha256
    ):
        raise ValueError(f"training row crop binding mismatch: {row.sample_token}")


def _verify_train_development_disjointness(rows: tuple[TrainingCropRow, ...]) -> None:
    train = tuple(row for row in rows if row.lane == "MODEL_TRAINING")
    development = tuple(row for row in rows if row.lane == "MODEL_SELECTION")
    for name, getter in (
        ("identity", lambda row: row.identity_token),
        ("public subject", lambda row: row.public_subject_token),
        ("component", lambda row: row.component_token),
    ):
        overlap = {getter(row) for row in train} & {getter(row) for row in development}
        if overlap:
            raise ValueError(f"train/development {name} tokens overlap")


def _require_exact_population(label: str, actual: set[str], expected: set[str]) -> None:
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"{label} mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _require_exact_keys(payload: object, expected: set[str], context: str) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


def _require_text(value: object, name: str, *, maximum: int = 256) -> None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be bounded non-empty text")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains a control character")


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name}")
