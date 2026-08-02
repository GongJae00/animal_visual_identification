"""Strict byte-and-JSON intake for pretrained-model supporting assets.

This boundary never imports a model framework, instantiates a preprocessor, or
interprets configuration fields as executable model behavior.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from artifact_contracts.pretrained_weight_intake import (
    PretrainedWeightIntakeReceipt,
    PretrainedWeightSourceContract,
    PretrainedWeightUsageLane,
    validate_pretrained_weight_receipt_binding,
)
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file


MAXIMUM_ASSET_BYTES = 4_194_304
MAXIMUM_JSON_DEPTH = 32
MAXIMUM_JSON_KEYS = 8_192
MAXIMUM_JSON_ARRAY_LENGTH = 8_192
MAXIMUM_JSON_STRING_CHARACTERS = 65_536
MAXIMUM_JSON_NODES = 32_768
MAXIMUM_JSON_NUMBER_CHARACTERS = 128
_INTERPRETATION = (
    "CONFIG_BYTE_AND_JSON_STRUCTURE_INTAKE_ONLY_NOT_PREPROCESSING_MODEL_OR_"
    "PERFORMANCE_ADMISSION"
)


class PretrainedSupportingAssetKind(StrEnum):
    CONFIG = "CONFIG"
    PREPROCESSOR_CONFIG = "PREPROCESSOR_CONFIG"


@dataclass(frozen=True, slots=True)
class PretrainedSupportingAssetSourceContract:
    source_model_id: str
    source_revision: str
    source_model_page_url: str
    source_file_url: str
    asset_filename: str
    asset_kind: PretrainedSupportingAssetKind
    expected_file_bytes: int
    expected_sha256: str
    license_id: str
    license_url: str
    license_snapshot_sha256: str
    license_usage_lane: PretrainedWeightUsageLane
    associated_pretrained_weight_receipt_sha256: str
    target_lane: PretrainedWeightUsageLane
    schema_version: str = "cvi.pretrained_supporting_asset_source_contract.v1"

    def __post_init__(self) -> None:
        if self.schema_version != (
            "cvi.pretrained_supporting_asset_source_contract.v1"
        ):
            raise ValueError("unsupported pretrained supporting asset contract")
        for name in ("source_model_id", "source_revision", "license_id"):
            _require_canonical_text(getattr(self, name), name, maximum=512)
        if self.source_revision.casefold() in {
            "default",
            "head",
            "latest",
            "main",
            "master",
        }:
            raise ValueError("source_revision must identify a frozen revision")
        for name in (
            "source_model_page_url",
            "source_file_url",
            "license_url",
        ):
            _require_https_url(getattr(self, name), name)
        _require_safe_json_filename(self.asset_filename)
        source_basename = PurePosixPath(urlsplit(self.source_file_url).path).name
        if source_basename != self.asset_filename:
            raise ValueError("source file URL basename differs from asset filename")
        if not isinstance(self.asset_kind, PretrainedSupportingAssetKind):
            raise TypeError("asset_kind must be a PretrainedSupportingAssetKind")
        if (
            isinstance(self.expected_file_bytes, bool)
            or not isinstance(self.expected_file_bytes, int)
            or self.expected_file_bytes <= 0
            or self.expected_file_bytes > MAXIMUM_ASSET_BYTES
        ):
            raise ValueError("expected_file_bytes must be positive and bounded")
        for name in (
            "expected_sha256",
            "license_snapshot_sha256",
            "associated_pretrained_weight_receipt_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for value, name in (
            (self.license_usage_lane, "license_usage_lane"),
            (self.target_lane, "target_lane"),
        ):
            if not isinstance(value, PretrainedWeightUsageLane):
                raise TypeError(f"{name} must be a PretrainedWeightUsageLane")
        if (
            self.license_usage_lane is PretrainedWeightUsageLane.RESEARCH_ONLY
            and self.target_lane is PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE
        ):
            raise ValueError(
                "research-only license cannot target the deployment candidate lane"
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
            "asset_filename": self.asset_filename,
            "asset_kind": self.asset_kind.value,
            "expected_file_bytes": self.expected_file_bytes,
            "expected_sha256": self.expected_sha256,
            "license_id": self.license_id,
            "license_url": self.license_url,
            "license_snapshot_sha256": self.license_snapshot_sha256,
            "license_usage_lane": self.license_usage_lane.value,
            "associated_pretrained_weight_receipt_sha256": (
                self.associated_pretrained_weight_receipt_sha256
            ),
            "target_lane": self.target_lane.value,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> PretrainedSupportingAssetSourceContract:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "source contract")
        values = dict(payload)
        values["asset_kind"] = PretrainedSupportingAssetKind(values["asset_kind"])
        values["license_usage_lane"] = PretrainedWeightUsageLane(
            values["license_usage_lane"]
        )
        values["target_lane"] = PretrainedWeightUsageLane(values["target_lane"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PretrainedSupportingAssetIntakeReceipt:
    source_contract_sha256: str
    asset_sha256: str
    asset_bytes: int
    json_structure_sha256: str
    asset_kind: PretrainedSupportingAssetKind
    license_snapshot_sha256: str
    associated_pretrained_weight_receipt_sha256: str
    admitted_lane: PretrainedWeightUsageLane
    decision: str
    interpretation: str = _INTERPRETATION
    schema_version: str = "cvi.pretrained_supporting_asset_intake_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != (
            "cvi.pretrained_supporting_asset_intake_receipt.v1"
        ):
            raise ValueError("unsupported pretrained supporting asset receipt")
        for name in (
            "source_contract_sha256",
            "asset_sha256",
            "json_structure_sha256",
            "license_snapshot_sha256",
            "associated_pretrained_weight_receipt_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if (
            isinstance(self.asset_bytes, bool)
            or not isinstance(self.asset_bytes, int)
            or self.asset_bytes <= 0
            or self.asset_bytes > MAXIMUM_ASSET_BYTES
        ):
            raise ValueError("asset_bytes must be positive and bounded")
        if not isinstance(self.asset_kind, PretrainedSupportingAssetKind):
            raise TypeError("asset_kind must be a PretrainedSupportingAssetKind")
        if not isinstance(self.admitted_lane, PretrainedWeightUsageLane):
            raise TypeError("admitted_lane must be a PretrainedWeightUsageLane")
        if self.decision != f"PASS_EXACT_BYTE_AND_JSON_{self.admitted_lane.value}":
            raise ValueError("pretrained supporting asset decision differs")
        if self.interpretation != _INTERPRETATION:
            raise ValueError("pretrained supporting asset interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_contract_sha256": self.source_contract_sha256,
            "asset_sha256": self.asset_sha256,
            "asset_bytes": self.asset_bytes,
            "json_structure_sha256": self.json_structure_sha256,
            "asset_kind": self.asset_kind.value,
            "license_snapshot_sha256": self.license_snapshot_sha256,
            "associated_pretrained_weight_receipt_sha256": (
                self.associated_pretrained_weight_receipt_sha256
            ),
            "admitted_lane": self.admitted_lane.value,
            "decision": self.decision,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> PretrainedSupportingAssetIntakeReceipt:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "intake receipt")
        values = dict(payload)
        values["asset_kind"] = PretrainedSupportingAssetKind(values["asset_kind"])
        values["admitted_lane"] = PretrainedWeightUsageLane(
            values["admitted_lane"]
        )
        return cls(**values)


def audit_pretrained_supporting_asset(
    *,
    asset_path: Path,
    license_snapshot_path: Path,
    source: PretrainedSupportingAssetSourceContract,
    associated_weight_source: PretrainedWeightSourceContract,
    associated_weight_receipt: PretrainedWeightIntakeReceipt,
    audit_phase_callback: Callable[[str], None] | None = None,
) -> PretrainedSupportingAssetIntakeReceipt:
    """Bind and structurally validate JSON without interpreting its fields."""

    if asset_path.name not in {
        source.asset_filename,
        f"{source.asset_filename}.partial",
    }:
        raise ValueError("supporting asset local filename differs from source")
    if associated_weight_source.source_model_id != source.source_model_id:
        raise ValueError("associated pretrained weight model ID differs")
    if associated_weight_source.source_revision != source.source_revision:
        raise ValueError("associated pretrained weight revision differs")
    validate_pretrained_weight_receipt_binding(
        associated_weight_receipt,
        associated_weight_source,
    )
    if (
        associated_weight_receipt.receipt_sha256
        != source.associated_pretrained_weight_receipt_sha256
    ):
        raise ValueError("associated pretrained weight receipt SHA-256 differs")
    if (
        associated_weight_receipt.admitted_lane
        is PretrainedWeightUsageLane.RESEARCH_ONLY
        and source.target_lane is PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE
    ):
        raise ValueError("research-only weight cannot support a deployment asset")

    license_result = read_retained_regular_file(
        license_snapshot_path,
        expected_sha256=source.license_snapshot_sha256,
        capture_payload=False,
        subject="pretrained supporting asset license snapshot",
    )
    asset_result = read_retained_regular_file(
        asset_path,
        expected_bytes=source.expected_file_bytes,
        expected_sha256=source.expected_sha256,
        maximum_bytes=MAXIMUM_ASSET_BYTES,
        capture_payload=True,
        subject="pretrained supporting asset",
        phase_callback=audit_phase_callback,
        phase_label="ASSET_HASHED",
    )
    if asset_result.payload is None:  # pragma: no cover - enforced by helper call
        raise RuntimeError("supporting asset payload was not retained")
    parsed = parse_bounded_strict_json_object(asset_result.payload)
    return PretrainedSupportingAssetIntakeReceipt(
        source_contract_sha256=source.contract_sha256,
        asset_sha256=asset_result.sha256,
        asset_bytes=asset_result.byte_count,
        json_structure_sha256=content_sha256(parsed),
        asset_kind=source.asset_kind,
        license_snapshot_sha256=license_result.sha256,
        associated_pretrained_weight_receipt_sha256=(
            associated_weight_receipt.receipt_sha256
        ),
        admitted_lane=source.target_lane,
        decision=f"PASS_EXACT_BYTE_AND_JSON_{source.target_lane.value}",
    )


def validate_pretrained_supporting_asset_receipt_binding(
    receipt: PretrainedSupportingAssetIntakeReceipt,
    source: PretrainedSupportingAssetSourceContract,
) -> None:
    expected = {
        "source_contract_sha256": source.contract_sha256,
        "asset_sha256": source.expected_sha256,
        "asset_bytes": source.expected_file_bytes,
        "asset_kind": source.asset_kind,
        "license_snapshot_sha256": source.license_snapshot_sha256,
        "associated_pretrained_weight_receipt_sha256": (
            source.associated_pretrained_weight_receipt_sha256
        ),
        "admitted_lane": source.target_lane,
        "decision": f"PASS_EXACT_BYTE_AND_JSON_{source.target_lane.value}",
    }
    for field_name, expected_value in expected.items():
        if getattr(receipt, field_name) != expected_value:
            raise ValueError(f"pretrained supporting asset receipt {field_name} differs")


def parse_bounded_strict_json_object(payload: bytes) -> dict[str, Any]:
    """Parse a small UTF-8 JSON object under explicit structural bounds."""

    if not isinstance(payload, bytes) or len(payload) > MAXIMUM_ASSET_BYTES:
        raise ValueError("supporting asset JSON bytes must be bounded")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("supporting asset JSON must be UTF-8") from error
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonstandard_constant,
            parse_int=_parse_bounded_int,
            parse_float=_parse_bounded_float,
        )
    except (RecursionError, OverflowError) as error:
        raise ValueError("supporting asset JSON exceeds parser bounds") from error
    if not isinstance(parsed, dict):
        raise ValueError("supporting asset JSON root must be an object")
    _validate_json_structure(parsed)
    return parsed


def _validate_json_structure(root: dict[str, Any]) -> None:
    stack: list[tuple[Any, int]] = [(root, 1)]
    node_count = 0
    key_count = 0
    while stack:
        value, depth = stack.pop()
        node_count += 1
        if node_count > MAXIMUM_JSON_NODES:
            raise ValueError("supporting asset JSON node count exceeds limit")
        if depth > MAXIMUM_JSON_DEPTH:
            raise ValueError("supporting asset JSON depth exceeds limit")
        if isinstance(value, dict):
            key_count += len(value)
            if key_count > MAXIMUM_JSON_KEYS:
                raise ValueError("supporting asset JSON key count exceeds limit")
            for key, child in value.items():
                _validate_json_string(key)
                stack.append((child, depth + 1))
        elif isinstance(value, list):
            if len(value) > MAXIMUM_JSON_ARRAY_LENGTH:
                raise ValueError("supporting asset JSON array exceeds limit")
            stack.extend((child, depth + 1) for child in value)
        elif isinstance(value, str):
            _validate_json_string(value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("supporting asset JSON number must be finite")
        elif value is None or isinstance(value, (bool, int)):
            continue
        else:  # pragma: no cover - json.loads cannot create another type
            raise TypeError("supporting asset JSON value type is unsupported")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _parse_bounded_int(value: str) -> int:
    if len(value) > MAXIMUM_JSON_NUMBER_CHARACTERS:
        raise ValueError("supporting asset JSON integer token exceeds limit")
    return int(value)


def _parse_bounded_float(value: str) -> float:
    if len(value) > MAXIMUM_JSON_NUMBER_CHARACTERS:
        raise ValueError("supporting asset JSON float token exceeds limit")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("supporting asset JSON number must be finite")
    return parsed


def _validate_json_string(value: str) -> None:
    if len(value) > MAXIMUM_JSON_STRING_CHARACTERS:
        raise ValueError("supporting asset JSON string exceeds limit")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("supporting asset JSON contains an unpaired surrogate")


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"pretrained supporting asset {name} fields differ")


def _require_canonical_text(value: str, name: str, *, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be canonical bounded text")


def _require_https_url(value: str, name: str) -> None:
    _require_canonical_text(value, name, maximum=4_096)
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be an anonymous HTTPS URL")


def _require_safe_json_filename(value: str) -> None:
    _require_canonical_text(value, "asset_filename", maximum=255)
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError("asset_filename must be a basename")
    if any(character in '<>:"/\\|?*' for character in value) or value.endswith(
        (".", " ")
    ):
        raise ValueError("asset_filename is not a portable path component")
    if Path(value).suffix.casefold() != ".json":
        raise ValueError("supporting asset filename must use .json")


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
