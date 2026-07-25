"""Byte-only, license-bound intake for externally pretrained model weights.

This module deliberately does not import a tensor framework or deserialize a
weight file.  A passing receipt binds source metadata to the exact bytes that
were inspected; it is not model, license, safety, or performance admission.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from cvi.provenance import content_sha256
from cvi.retained_file import read_retained_regular_file


class PretrainedWeightUsageLane(StrEnum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    DEPLOYMENT_CANDIDATE = "DEPLOYMENT_CANDIDATE"


class PretrainedWeightChecksumAuthority(StrEnum):
    PUBLISHED_SHA256 = "PUBLISHED_SHA256"
    UNVERIFIED_SHA256 = "UNVERIFIED_SHA256"


class PretrainedWeightFileFormat(StrEnum):
    SAFETENSORS = "SAFETENSORS"
    PYTORCH_STATE_DICT = "PYTORCH_STATE_DICT"


@dataclass(frozen=True, slots=True)
class PretrainedWeightSourceContract:
    source_model_id: str
    source_revision: str
    source_model_page_url: str
    source_file_url: str
    weight_filename: str
    license_id: str
    license_url: str
    license_snapshot_sha256: str
    license_usage_lane: PretrainedWeightUsageLane
    training_description: str
    training_description_url: str
    training_description_snapshot_sha256: str
    expected_file_bytes: int
    expected_sha256: str
    checksum_authority: PretrainedWeightChecksumAuthority
    target_lane: PretrainedWeightUsageLane
    file_format: PretrainedWeightFileFormat
    schema_version: str = "cvi.pretrained_weight_source_contract.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.pretrained_weight_source_contract.v1":
            raise ValueError("unsupported pretrained weight source contract")
        for name in ("source_model_id", "source_revision", "license_id"):
            value = getattr(self, name)
            _require_bounded_text(value, name, maximum=512)
            if value != value.strip() or any(
                ord(character) < 32 for character in value
            ):
                raise ValueError(f"{name} must be canonical text")
        _require_enum(
            self.license_usage_lane,
            PretrainedWeightUsageLane,
            "license_usage_lane",
        )
        _require_enum(
            self.checksum_authority,
            PretrainedWeightChecksumAuthority,
            "checksum_authority",
        )
        _require_enum(self.target_lane, PretrainedWeightUsageLane, "target_lane")
        _require_enum(self.file_format, PretrainedWeightFileFormat, "file_format")
        if self.source_revision.casefold() in {
            "default",
            "head",
            "latest",
            "main",
            "master",
        }:
            raise ValueError("source_revision must identify a frozen revision")
        _require_bounded_text(
            self.training_description,
            "training_description",
            maximum=8_192,
        )
        for name in (
            "source_model_page_url",
            "source_file_url",
            "license_url",
            "training_description_url",
        ):
            _require_https_url(getattr(self, name), name)
        _require_safe_weight_filename(self.weight_filename, self.file_format)
        source_basename = PurePosixPath(urlsplit(self.source_file_url).path).name
        if source_basename != self.weight_filename:
            raise ValueError("source file URL basename differs from weight filename")
        _validate_sha256(
            self.license_snapshot_sha256,
            "license_snapshot_sha256",
        )
        _validate_sha256(
            self.training_description_snapshot_sha256,
            "training_description_snapshot_sha256",
        )
        _validate_sha256(self.expected_sha256, "expected_sha256")
        _require_positive_int(self.expected_file_bytes, "expected_file_bytes")
        if (
            self.license_usage_lane is PretrainedWeightUsageLane.RESEARCH_ONLY
            and self.target_lane
            is PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE
        ):
            raise ValueError(
                "research-only license cannot target the deployment candidate lane"
            )
        if (
            self.checksum_authority
            is PretrainedWeightChecksumAuthority.UNVERIFIED_SHA256
            and self.target_lane
            is PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE
        ):
            raise ValueError(
                "unverified checksum cannot target the deployment candidate lane"
            )

    @property
    def contract_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_model_id": self.source_model_id,
            "source_revision": self.source_revision,
            "source_model_page_url": self.source_model_page_url,
            "source_file_url": self.source_file_url,
            "weight_filename": self.weight_filename,
            "license_id": self.license_id,
            "license_url": self.license_url,
            "license_snapshot_sha256": self.license_snapshot_sha256,
            "license_usage_lane": self.license_usage_lane.value,
            "training_description": self.training_description,
            "training_description_url": self.training_description_url,
            "training_description_snapshot_sha256": (
                self.training_description_snapshot_sha256
            ),
            "expected_file_bytes": self.expected_file_bytes,
            "expected_sha256": self.expected_sha256,
            "checksum_authority": self.checksum_authority.value,
            "target_lane": self.target_lane.value,
            "file_format": self.file_format.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PretrainedWeightSourceContract:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "source contract")
        values = dict(payload)
        values["license_usage_lane"] = PretrainedWeightUsageLane(
            values["license_usage_lane"]
        )
        values["checksum_authority"] = PretrainedWeightChecksumAuthority(
            values["checksum_authority"]
        )
        values["target_lane"] = PretrainedWeightUsageLane(values["target_lane"])
        values["file_format"] = PretrainedWeightFileFormat(values["file_format"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PretrainedWeightIntakeReceipt:
    source_contract_sha256: str
    weight_sha256: str
    weight_bytes: int
    license_snapshot_sha256: str
    training_description_snapshot_sha256: str
    checksum_authority: PretrainedWeightChecksumAuthority
    admitted_lane: PretrainedWeightUsageLane
    file_format: PretrainedWeightFileFormat
    decision: str
    interpretation: str = (
        "WEIGHT_BYTE_INTAKE_ONLY_NOT_DESERIALIZATION_MODEL_OR_PERFORMANCE_ADMISSION"
    )
    schema_version: str = "cvi.pretrained_weight_intake_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.pretrained_weight_intake_receipt.v1":
            raise ValueError("unsupported pretrained weight intake receipt")
        _require_enum(
            self.checksum_authority,
            PretrainedWeightChecksumAuthority,
            "checksum_authority",
        )
        _require_enum(
            self.admitted_lane,
            PretrainedWeightUsageLane,
            "admitted_lane",
        )
        _require_enum(self.file_format, PretrainedWeightFileFormat, "file_format")
        for name in (
            "source_contract_sha256",
            "weight_sha256",
            "license_snapshot_sha256",
            "training_description_snapshot_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        _require_positive_int(self.weight_bytes, "weight_bytes")
        expected_decision = _decision_for(
            self.checksum_authority,
            self.admitted_lane,
        )
        if self.decision != expected_decision:
            raise ValueError("pretrained weight receipt decision differs")
        if (
            self.checksum_authority
            is PretrainedWeightChecksumAuthority.UNVERIFIED_SHA256
            and self.admitted_lane
            is PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE
        ):
            raise ValueError("unverified checksum cannot be deployment-admitted")
        if self.interpretation != (
            "WEIGHT_BYTE_INTAKE_ONLY_NOT_DESERIALIZATION_MODEL_OR_PERFORMANCE_ADMISSION"
        ):
            raise ValueError("pretrained weight receipt interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_contract_sha256": self.source_contract_sha256,
            "weight_sha256": self.weight_sha256,
            "weight_bytes": self.weight_bytes,
            "license_snapshot_sha256": self.license_snapshot_sha256,
            "training_description_snapshot_sha256": (
                self.training_description_snapshot_sha256
            ),
            "checksum_authority": self.checksum_authority.value,
            "admitted_lane": self.admitted_lane.value,
            "file_format": self.file_format.value,
            "decision": self.decision,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PretrainedWeightIntakeReceipt:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "intake receipt")
        values = dict(payload)
        values["checksum_authority"] = PretrainedWeightChecksumAuthority(
            values["checksum_authority"]
        )
        values["admitted_lane"] = PretrainedWeightUsageLane(
            values["admitted_lane"]
        )
        values["file_format"] = PretrainedWeightFileFormat(values["file_format"])
        return cls(**values)


def audit_pretrained_weight_file(
    *,
    weight_path: Path,
    license_snapshot_path: Path,
    training_description_snapshot_path: Path,
    source: PretrainedWeightSourceContract,
    audit_phase_callback: Callable[[str], None] | None = None,
) -> PretrainedWeightIntakeReceipt:
    """Hash source-bound bytes without parsing or deserializing the file."""

    if weight_path.name != source.weight_filename:
        raise ValueError("pretrained weight filename differs from source")
    license_sha256, _ = _hash_retained_nofollow_file(license_snapshot_path)
    if license_sha256 != source.license_snapshot_sha256:
        raise ValueError("pretrained weight license snapshot hash differs")
    training_sha256, _ = _hash_retained_nofollow_file(
        training_description_snapshot_path
    )
    if training_sha256 != source.training_description_snapshot_sha256:
        raise ValueError("pretrained weight training description snapshot hash differs")

    weight_sha256, weight_bytes = _hash_retained_nofollow_file(
        weight_path,
        expected_bytes=source.expected_file_bytes,
        expected_sha256=source.expected_sha256,
        phase_callback=audit_phase_callback,
    )
    return PretrainedWeightIntakeReceipt(
        source_contract_sha256=source.contract_sha256,
        weight_sha256=weight_sha256,
        weight_bytes=weight_bytes,
        license_snapshot_sha256=license_sha256,
        training_description_snapshot_sha256=training_sha256,
        checksum_authority=source.checksum_authority,
        admitted_lane=source.target_lane,
        file_format=source.file_format,
        decision=_decision_for(source.checksum_authority, source.target_lane),
    )


def validate_pretrained_weight_receipt_binding(
    receipt: PretrainedWeightIntakeReceipt,
    source: PretrainedWeightSourceContract,
) -> None:
    """Fail closed if a receipt is replayed against a different source contract."""

    expected = {
        "source_contract_sha256": source.contract_sha256,
        "weight_sha256": source.expected_sha256,
        "weight_bytes": source.expected_file_bytes,
        "license_snapshot_sha256": source.license_snapshot_sha256,
        "training_description_snapshot_sha256": (
            source.training_description_snapshot_sha256
        ),
        "checksum_authority": source.checksum_authority,
        "admitted_lane": source.target_lane,
        "file_format": source.file_format,
        "decision": _decision_for(source.checksum_authority, source.target_lane),
    }
    for field_name, expected_value in expected.items():
        if getattr(receipt, field_name) != expected_value:
            raise ValueError(f"pretrained weight receipt {field_name} differs")


def _hash_retained_nofollow_file(
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    phase_callback: Callable[[str], None] | None = None,
) -> tuple[str, int]:
    result = read_retained_regular_file(
        path,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        capture_payload=False,
        subject="pretrained weight file",
        phase_callback=phase_callback,
        phase_label="WEIGHT_HASHED",
    )
    return result.sha256, result.byte_count


def _decision_for(
    authority: PretrainedWeightChecksumAuthority,
    lane: PretrainedWeightUsageLane,
) -> str:
    prefix = (
        "PASS_PUBLISHED_SHA256"
        if authority is PretrainedWeightChecksumAuthority.PUBLISHED_SHA256
        else "PASS_UNVERIFIED_SHA256"
    )
    return f"{prefix}_{lane.value}"


def _require_safe_weight_filename(
    value: str,
    file_format: PretrainedWeightFileFormat,
) -> None:
    _require_bounded_text(value, "weight_filename", maximum=255)
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError("weight_filename must be a basename")
    if unicodedata.normalize("NFC", value) != value or any(
        ord(character) < 32 or character in '<>:"/\\|?*' for character in value
    ):
        raise ValueError("weight_filename is not a portable path component")
    if value.endswith((".", " ")):
        raise ValueError("weight_filename is not a portable path component")
    stem = value.split(".", 1)[0].casefold()
    if stem in {"aux", "con", "nul", "prn"} or (
        len(stem) == 4
        and stem[:3] in {"com", "lpt"}
        and stem[3] in "123456789"
    ):
        raise ValueError("weight_filename uses a Windows reserved name")
    suffix = Path(value).suffix.casefold()
    expected_suffixes = (
        {".safetensors"}
        if file_format is PretrainedWeightFileFormat.SAFETENSORS
        else {".bin", ".pt", ".pth", ".pyth"}
    )
    if suffix not in expected_suffixes:
        raise ValueError("weight filename suffix differs from declared format")


def _require_https_url(value: str, name: str) -> None:
    _require_bounded_text(value, name, maximum=4_096)
    parsed = urlsplit(value)
    if (
        value != value.strip()
        or any(character.isspace() or ord(character) < 32 for character in value)
        or parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be an anonymous HTTPS URL")


def _require_bounded_text(value: str, name: str, *, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be non-empty and bounded")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_enum(value: object, enum_type: type[StrEnum], name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be a {enum_type.__name__}")


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"pretrained weight {name} fields differ")


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
