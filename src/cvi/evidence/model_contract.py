"""Fail-closed contracts for ONNX evidence models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
import re
from typing import Any


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class OnnxEvidenceContractError(ValueError):
    """Raised when an ONNX artifact or runtime result violates its manifest."""


class OnnxModelUsageLane(StrEnum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    DEPLOYMENT_CANDIDATE = "DEPLOYMENT_CANDIDATE"


class OnnxModelLicenseState(StrEnum):
    VERIFIED = "VERIFIED"
    RESTRICTED = "RESTRICTED"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True, slots=True)
class OnnxPreprocessingContract:
    """Complete image-to-NCHW preprocessing declaration."""

    color_mode: str
    layout: str
    dtype: str
    resize: str
    scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    schema_version: str = "cvi.onnx_preprocessing_contract.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.onnx_preprocessing_contract.v1":
            raise OnnxEvidenceContractError(
                "unsupported ONNX preprocessing contract schema"
            )
        if self.color_mode != "RGB":
            raise OnnxEvidenceContractError("preprocessing color_mode must be 'RGB'")
        if self.layout != "NCHW":
            raise OnnxEvidenceContractError("preprocessing layout must be 'NCHW'")
        if self.dtype != "float32":
            raise OnnxEvidenceContractError("preprocessing dtype must be 'float32'")
        if self.resize not in {"nearest", "bilinear", "bicubic"}:
            raise OnnxEvidenceContractError(
                "preprocessing resize must be nearest, bilinear, or bicubic"
            )
        if (
            not isinstance(self.scale, (int, float))
            or isinstance(self.scale, bool)
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise OnnxEvidenceContractError("preprocessing scale must be finite and positive")
        if (
            not isinstance(self.mean, tuple)
            or len(self.mean) != 3
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in self.mean
            )
        ):
            raise OnnxEvidenceContractError("preprocessing mean must contain three finite values")
        if (
            not isinstance(self.std, tuple)
            or len(self.std) != 3
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
                for value in self.std
            )
        ):
            raise OnnxEvidenceContractError(
                "preprocessing std must contain three finite positive values"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "color_mode": self.color_mode,
            "layout": self.layout,
            "dtype": self.dtype,
            "resize": self.resize,
            "scale": float(self.scale),
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OnnxPreprocessingContract:
        _require_exact_keys(
            payload,
            {
                "schema_version", "color_mode", "layout", "dtype", "resize",
                "scale", "mean", "std",
            },
            "ONNX preprocessing contract",
        )
        mean = _require_three_number_list(payload["mean"], "preprocessing mean")
        std = _require_three_number_list(payload["std"], "preprocessing std")
        return cls(
            schema_version=payload["schema_version"],
            color_mode=payload["color_mode"],
            layout=payload["layout"],
            dtype=payload["dtype"],
            resize=payload["resize"],
            scale=payload["scale"],
            mean=mean,
            std=std,
        )


@dataclass(frozen=True, slots=True)
class OnnxEvidenceModelManifest:
    """Identity, graph, and preprocessing contract for one exact ONNX artifact."""

    model_id: str
    model_sha256: str
    input_name: str
    input_shape: tuple[int | str, int, int, int]
    output_name: str
    output_dim: int
    preprocessing: OnnxPreprocessingContract
    model_kind: str = "generic_onnx"
    usage_lane: OnnxModelUsageLane = OnnxModelUsageLane.RESEARCH_ONLY
    license_state: OnnxModelLicenseState = OnnxModelLicenseState.UNVERIFIED
    schema_version: str = "cvi.onnx_evidence_model_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.onnx_evidence_model_manifest.v1":
            raise OnnxEvidenceContractError(
                "unsupported ONNX evidence model manifest schema"
            )
        if not isinstance(self.model_kind, str) or not self.model_kind:
            raise OnnxEvidenceContractError("model_kind must be non-empty")
        if not isinstance(self.usage_lane, OnnxModelUsageLane):
            raise OnnxEvidenceContractError(
                "usage_lane must be an OnnxModelUsageLane"
            )
        if not isinstance(self.license_state, OnnxModelLicenseState):
            raise OnnxEvidenceContractError(
                "license_state must be an OnnxModelLicenseState"
            )
        if (
            self.usage_lane is OnnxModelUsageLane.DEPLOYMENT_CANDIDATE
            and self.license_state is not OnnxModelLicenseState.VERIFIED
        ):
            raise OnnxEvidenceContractError(
                "deployment-candidate models require a verified license state"
            )
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise OnnxEvidenceContractError("model_id must be non-empty")
        if (
            not isinstance(self.model_sha256, str)
            or _SHA256_RE.fullmatch(self.model_sha256) is None
        ):
            raise OnnxEvidenceContractError(
                "model_sha256 must be an exact lowercase SHA256 digest"
            )
        if (
            not isinstance(self.input_name, str)
            or not self.input_name
            or not isinstance(self.output_name, str)
            or not self.output_name
        ):
            raise OnnxEvidenceContractError("input_name and output_name must be non-empty")
        if not isinstance(self.input_shape, tuple) or len(self.input_shape) != 4:
            raise OnnxEvidenceContractError("input_shape must declare NCHW dimensions")
        batch, channels, height, width = self.input_shape
        if not (
            isinstance(batch, int) and not isinstance(batch, bool) and batch == 1
            or isinstance(batch, str) and bool(batch.strip())
        ):
            raise OnnxEvidenceContractError(
                "input batch dimension must be 1 or a non-empty symbolic name"
            )
        if channels != 3:
            raise OnnxEvidenceContractError(
                "RGB preprocessing requires three declared input channels"
            )
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (channels, height, width)
        ):
            raise OnnxEvidenceContractError(
                "input channel and spatial dimensions must be static positive integers"
            )
        if (
            not isinstance(self.output_dim, int)
            or isinstance(self.output_dim, bool)
            or self.output_dim <= 0
        ):
            raise OnnxEvidenceContractError("output_dim must be a positive integer")
        if not isinstance(self.preprocessing, OnnxPreprocessingContract):
            raise OnnxEvidenceContractError(
                "preprocessing must be an OnnxPreprocessingContract"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_kind": self.model_kind,
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "output_name": self.output_name,
            "output_dim": self.output_dim,
            "preprocessing": self.preprocessing.to_dict(),
            "usage_lane": self.usage_lane.value,
            "license_state": self.license_state.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OnnxEvidenceModelManifest:
        _require_exact_keys(
            payload,
            {
                "schema_version", "model_kind", "model_id", "model_sha256",
                "input_name", "input_shape", "output_name", "output_dim",
                "preprocessing", "usage_lane", "license_state",
            },
            "ONNX evidence model manifest",
        )
        input_shape = payload["input_shape"]
        if not isinstance(input_shape, list) or len(input_shape) != 4:
            raise OnnxEvidenceContractError(
                "manifest input_shape must be a four-item JSON array"
            )
        preprocessing = payload["preprocessing"]
        if not isinstance(preprocessing, dict):
            raise OnnxEvidenceContractError(
                "manifest preprocessing must be an object"
            )
        try:
            usage_lane = OnnxModelUsageLane(payload["usage_lane"])
            license_state = OnnxModelLicenseState(payload["license_state"])
        except (TypeError, ValueError) as exc:
            raise OnnxEvidenceContractError(
                "manifest usage_lane or license_state is unsupported"
            ) from exc
        return cls(
            schema_version=payload["schema_version"],
            model_kind=payload["model_kind"],
            model_id=payload["model_id"],
            model_sha256=payload["model_sha256"],
            input_name=payload["input_name"],
            input_shape=tuple(input_shape),
            output_name=payload["output_name"],
            output_dim=payload["output_dim"],
            preprocessing=OnnxPreprocessingContract.from_dict(preprocessing),
            usage_lane=usage_lane,
            license_state=license_state,
        )


@dataclass(frozen=True, slots=True)
class DogFaceNetModelManifest(OnnxEvidenceModelManifest):
    """Manifest type accepted by :class:`DogFaceNetExtractor`."""

    model_kind: str = "dogfacenet_onnx"

    def __post_init__(self) -> None:
        super(DogFaceNetModelManifest, self).__post_init__()
        _require_model_kind(self.model_kind, "dogfacenet_onnx")


@dataclass(frozen=True, slots=True)
class ConvNeXtModelManifest(OnnxEvidenceModelManifest):
    """Manifest type accepted by :class:`ConvNeXtExtractor`."""

    model_kind: str = "convnext_onnx"

    def __post_init__(self) -> None:
        super(ConvNeXtModelManifest, self).__post_init__()
        _require_model_kind(self.model_kind, "convnext_onnx")


@dataclass(frozen=True, slots=True)
class PetReIDModelManifest(OnnxEvidenceModelManifest):
    """Manifest type accepted by :class:`PetReIDExtractor`."""

    model_kind: str = "petreid_nose_onnx"

    def __post_init__(self) -> None:
        super(PetReIDModelManifest, self).__post_init__()
        _require_model_kind(self.model_kind, "petreid_nose_onnx")


def _require_exact_keys(
    payload: object,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise OnnxEvidenceContractError(f"{label} must use its exact-key schema")


def _require_three_number_list(
    value: object,
    label: str,
) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            for item in value
        )
    ):
        raise OnnxEvidenceContractError(f"{label} must be a three-item finite array")
    return (float(value[0]), float(value[1]), float(value[2]))


def _require_model_kind(actual: str, expected: str) -> None:
    if actual != expected:
        raise OnnxEvidenceContractError(f"model_kind must be {expected!r}")


__all__ = [
    "ConvNeXtModelManifest",
    "DogFaceNetModelManifest",
    "OnnxEvidenceContractError",
    "OnnxEvidenceModelManifest",
    "OnnxModelLicenseState",
    "OnnxModelUsageLane",
    "OnnxPreprocessingContract",
    "PetReIDModelManifest",
]
