"""Typed, externally hash-bound model parity receipts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from contracts.pretrained_supporting_asset_intake import (
    parse_bounded_strict_json_object,
)
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file

_MAXIMUM_RECEIPT_BYTES = 1_048_576
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ModelParityError(RuntimeError):
    """Raised when parity evidence is malformed, unbound, or not passing."""


class ModelUsageLane(StrEnum):
    TEST_FIXTURE = "TEST_FIXTURE"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    DEPLOYMENT_CANDIDATE = "DEPLOYMENT_CANDIDATE"


class ParityFixtureKind(StrEnum):
    SYNTHETIC = "SYNTHETIC"
    RECEIPT_BOUND_CROP = "RECEIPT_BOUND_CROP"


@dataclass(frozen=True, slots=True)
class ParityThresholds:
    maximum_absolute_error: float
    maximum_relative_error: float
    relative_error_floor: float
    minimum_cosine_similarity: float

    def __post_init__(self) -> None:
        _require_finite_nonnegative(
            self.maximum_absolute_error, "maximum_absolute_error"
        )
        _require_finite_nonnegative(
            self.maximum_relative_error, "maximum_relative_error"
        )
        if (
            isinstance(self.relative_error_floor, bool)
            or not isinstance(self.relative_error_floor, (int, float))
            or not math.isfinite(self.relative_error_floor)
            or self.relative_error_floor <= 0
        ):
            raise ModelParityError("relative_error_floor must be finite and positive")
        if (
            isinstance(self.minimum_cosine_similarity, bool)
            or not isinstance(self.minimum_cosine_similarity, (int, float))
            or not math.isfinite(self.minimum_cosine_similarity)
            or not -1.0 <= self.minimum_cosine_similarity <= 1.0
        ):
            raise ModelParityError(
                "minimum_cosine_similarity must be finite and in [-1, 1]"
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "relative_error_floor": self.relative_error_floor,
            "minimum_cosine_similarity": self.minimum_cosine_similarity,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ParityThresholds:
        _require_exact_keys(
            payload,
            {
                "maximum_absolute_error",
                "maximum_relative_error",
                "relative_error_floor",
                "minimum_cosine_similarity",
            },
            "parity thresholds",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ParityFixtureResult:
    fixture_id: str
    fixture_kind: ParityFixtureKind
    input_sha256: str
    reference_output_sha256: str
    candidate_output_sha256: str
    maximum_absolute_error: float
    maximum_relative_error: float
    cosine_similarity: float
    decision: str

    def __post_init__(self) -> None:
        _require_nonempty(self.fixture_id, "fixture_id")
        if not isinstance(self.fixture_kind, ParityFixtureKind):
            raise ModelParityError("fixture_kind must be a ParityFixtureKind")
        for name in (
            "input_sha256",
            "reference_output_sha256",
            "candidate_output_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_finite_nonnegative(
            self.maximum_absolute_error, "maximum_absolute_error"
        )
        _require_finite_nonnegative(
            self.maximum_relative_error, "maximum_relative_error"
        )
        if (
            isinstance(self.cosine_similarity, bool)
            or not isinstance(self.cosine_similarity, (int, float))
            or not math.isfinite(self.cosine_similarity)
            or not -1.0 <= self.cosine_similarity <= 1.0
        ):
            raise ModelParityError("cosine_similarity must be finite and in [-1, 1]")
        if self.decision != "PASS":
            raise ModelParityError("parity fixture decision must be PASS")

    def validate_against(self, thresholds: ParityThresholds) -> None:
        if self.maximum_absolute_error > thresholds.maximum_absolute_error:
            raise ModelParityError("fixture absolute error exceeds its threshold")
        if self.maximum_relative_error > thresholds.maximum_relative_error:
            raise ModelParityError("fixture relative error exceeds its threshold")
        if self.cosine_similarity < thresholds.minimum_cosine_similarity:
            raise ModelParityError("fixture cosine similarity is below its threshold")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "fixture_kind": self.fixture_kind.value,
            "input_sha256": self.input_sha256,
            "reference_output_sha256": self.reference_output_sha256,
            "candidate_output_sha256": self.candidate_output_sha256,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "cosine_similarity": self.cosine_similarity,
            "decision": self.decision,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ParityFixtureResult:
        _require_exact_keys(
            payload,
            {
                "fixture_id",
                "fixture_kind",
                "input_sha256",
                "reference_output_sha256",
                "candidate_output_sha256",
                "maximum_absolute_error",
                "maximum_relative_error",
                "cosine_similarity",
                "decision",
            },
            "parity fixture",
        )
        values = dict(payload)
        values["fixture_kind"] = ParityFixtureKind(values["fixture_kind"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ModelParityReceipt:
    model_id: str
    artifact_sha256: str
    source_weights_sha256: str
    weight_intake_receipt_sha256: str | None
    preprocessing_sha256: str
    preprocessor_intake_receipt_sha256: str | None
    usage_lane: ModelUsageLane
    reference_backend: str
    candidate_backend: str
    thresholds: ParityThresholds
    fixture_panel_receipt_sha256: str | None
    fixtures: tuple[ParityFixtureResult, ...]
    decision: str
    interpretation: str = "NUMERICAL_EXPORT_PARITY_ONLY_NOT_IDENTITY_PERFORMANCE_ADMISSION"
    schema_version: str = "cvi.model_parity_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.model_parity_receipt.v1":
            raise ModelParityError("unsupported model parity receipt schema")
        _require_nonempty(self.model_id, "model_id")
        for name in (
            "artifact_sha256",
            "source_weights_sha256",
            "preprocessing_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in (
            "weight_intake_receipt_sha256",
            "preprocessor_intake_receipt_sha256",
            "fixture_panel_receipt_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, name)
        if not isinstance(self.usage_lane, ModelUsageLane):
            raise ModelParityError("usage_lane must be a ModelUsageLane")
        _require_nonempty(self.reference_backend, "reference_backend")
        _require_nonempty(self.candidate_backend, "candidate_backend")
        if not isinstance(self.thresholds, ParityThresholds):
            raise ModelParityError("thresholds must be ParityThresholds")
        if not self.fixtures:
            raise ModelParityError("parity receipt must contain fixtures")
        fixture_ids = tuple(item.fixture_id for item in self.fixtures)
        if fixture_ids != tuple(sorted(fixture_ids)) or len(fixture_ids) != len(
            set(fixture_ids)
        ):
            raise ModelParityError("parity fixture IDs must be sorted and unique")
        for fixture in self.fixtures:
            fixture.validate_against(self.thresholds)
        has_panel_fixtures = any(
            item.fixture_kind is ParityFixtureKind.RECEIPT_BOUND_CROP
            for item in self.fixtures
        )
        if has_panel_fixtures != (self.fixture_panel_receipt_sha256 is not None):
            raise ModelParityError(
                "receipt-bound crop fixtures and panel receipt hash must appear together"
            )
        if not any(
            item.fixture_kind is ParityFixtureKind.SYNTHETIC
            for item in self.fixtures
        ):
            raise ModelParityError("parity receipt must include a synthetic fixture")
        if self.decision != "PASS":
            raise ModelParityError("model parity receipt decision must be PASS")
        if self.interpretation != (
            "NUMERICAL_EXPORT_PARITY_ONLY_NOT_IDENTITY_PERFORMANCE_ADMISSION"
        ):
            raise ModelParityError("model parity receipt interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "artifact_sha256": self.artifact_sha256,
            "source_weights_sha256": self.source_weights_sha256,
            "weight_intake_receipt_sha256": self.weight_intake_receipt_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "preprocessor_intake_receipt_sha256": (
                self.preprocessor_intake_receipt_sha256
            ),
            "usage_lane": self.usage_lane.value,
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "thresholds": self.thresholds.to_dict(),
            "fixture_panel_receipt_sha256": self.fixture_panel_receipt_sha256,
            "fixtures": [item.to_dict() for item in self.fixtures],
            "decision": self.decision,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelParityReceipt:
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            "model parity receipt",
        )
        thresholds = payload["thresholds"]
        fixtures = payload["fixtures"]
        if not isinstance(thresholds, dict) or not isinstance(fixtures, list):
            raise ModelParityError("parity thresholds and fixtures have wrong types")
        values = dict(payload)
        values["usage_lane"] = ModelUsageLane(values["usage_lane"])
        values["thresholds"] = ParityThresholds.from_dict(thresholds)
        values["fixtures"] = tuple(
            ParityFixtureResult.from_dict(item) for item in fixtures
        )
        return cls(**values)


def load_model_parity_receipt(
    path: Path,
    *,
    expected_sha256: str,
) -> ModelParityReceipt:
    """Load a parity receipt only when an external raw-file digest matches."""

    _require_sha256(expected_sha256, "expected parity receipt SHA256")
    result = read_retained_regular_file(
        path,
        expected_sha256=expected_sha256,
        maximum_bytes=_MAXIMUM_RECEIPT_BYTES,
        capture_payload=True,
        subject="model parity receipt",
    )
    if result.payload is None:  # pragma: no cover - guaranteed by helper call
        raise ModelParityError("model parity receipt payload was not retained")
    try:
        payload = parse_bounded_strict_json_object(result.payload)
        return ModelParityReceipt.from_dict(payload)
    except (TypeError, ValueError, ModelParityError) as exc:
        if isinstance(exc, ModelParityError):
            raise
        raise ModelParityError("model parity receipt content is invalid") from exc


def validate_parity_binding(
    receipt: ModelParityReceipt,
    *,
    model_id: str,
    artifact_sha256: str,
    source_weights_sha256: str,
    preprocessing_sha256: str,
    usage_lane: ModelUsageLane,
    weight_intake_receipt_sha256: str | None = None,
    preprocessor_intake_receipt_sha256: str | None = None,
    public_production: bool = False,
) -> None:
    expected = {
        "model_id": model_id,
        "artifact_sha256": artifact_sha256,
        "source_weights_sha256": source_weights_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "usage_lane": usage_lane,
        "weight_intake_receipt_sha256": weight_intake_receipt_sha256,
        "preprocessor_intake_receipt_sha256": (
            preprocessor_intake_receipt_sha256
        ),
    }
    for name, value in expected.items():
        if getattr(receipt, name) != value:
            raise ModelParityError(f"model parity receipt {name} differs")
    if public_production:
        if usage_lane is ModelUsageLane.TEST_FIXTURE:
            raise ModelParityError(
                "TEST_FIXTURE parity is forbidden for public production"
            )
        if usage_lane is not ModelUsageLane.DEPLOYMENT_CANDIDATE:
            raise ModelParityError(
                "public production requires DEPLOYMENT_CANDIDATE parity"
            )
        if receipt.fixture_panel_receipt_sha256 is None:
            raise ModelParityError(
                "public production requires a receipt-bound crop parity panel"
            )


def _require_exact_keys(
    payload: object, expected: set[str], context: str
) -> None:
    if not isinstance(payload, dict):
        raise ModelParityError(f"{context} must be an object")
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise ModelParityError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ModelParityError(f"{name} must be an exact lowercase SHA256 digest")


def _require_nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelParityError(f"{name} must be a non-empty string")


def _require_finite_nonnegative(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ModelParityError(f"{name} must be finite and non-negative")


__all__ = [
    "ModelParityError",
    "ModelParityReceipt",
    "ModelUsageLane",
    "ParityFixtureKind",
    "ParityFixtureResult",
    "ParityThresholds",
    "load_model_parity_receipt",
    "validate_parity_binding",
]
