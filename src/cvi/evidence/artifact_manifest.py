"""Exact artifact contracts for scaffolded nose and landmark channels."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from hashlib import sha256
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image

from cvi.model_contracts import validated_onnx_bytes


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ArtifactContractError(RuntimeError):
    """Raised when an artifact, manifest, or runtime tensor is not exact."""


class UsageLane(str, Enum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    COMMERCIAL_ALLOWED = "COMMERCIAL_ALLOWED"
    TEST_FIXTURE = "TEST_FIXTURE"


@dataclass(frozen=True, slots=True)
class ArtifactLicense:
    license_id: str
    usage_lane: UsageLane

    def __post_init__(self) -> None:
        if not isinstance(self.license_id, str) or not self.license_id.strip():
            raise ArtifactContractError("license_id must be non-empty")
        if not isinstance(self.usage_lane, UsageLane):
            raise ArtifactContractError("usage_lane must be a UsageLane")

    def to_dict(self) -> dict[str, Any]:
        return {"license_id": self.license_id, "usage_lane": self.usage_lane.value}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactLicense:
        _require_exact_keys(payload, {"license_id", "usage_lane"}, "artifact license")
        return cls(payload["license_id"], UsageLane(payload["usage_lane"]))


@dataclass(frozen=True, slots=True)
class ClaheTransform:
    """Optional local-contrast transform; this is not super-resolution."""

    clip_limit: float
    tile_grid_size: tuple[int, int]

    def __post_init__(self) -> None:
        if (
            isinstance(self.clip_limit, bool)
            or not isinstance(self.clip_limit, (int, float))
            or not math.isfinite(self.clip_limit)
            or self.clip_limit <= 0
        ):
            raise ArtifactContractError("CLAHE clip_limit must be finite and positive")
        if (
            not isinstance(self.tile_grid_size, tuple)
            or len(self.tile_grid_size) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in self.tile_grid_size
            )
        ):
            raise ArtifactContractError("CLAHE tile_grid_size must contain two positive integers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_limit": self.clip_limit,
            "tile_grid_size": list(self.tile_grid_size),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClaheTransform:
        _require_exact_keys(payload, {"clip_limit", "tile_grid_size"}, "CLAHE transform")
        return cls(payload["clip_limit"], _tuple(payload["tile_grid_size"], "tile_grid_size"))


@dataclass(frozen=True, slots=True)
class ImagePreprocessing:
    color_mode: str
    layout: str
    dtype: str
    resize: str
    scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    clahe: ClaheTransform | None

    def __post_init__(self) -> None:
        if self.color_mode != "RGB" or self.layout != "NCHW" or self.dtype != "float32":
            raise ArtifactContractError(
                "image preprocessing must declare RGB, NCHW, and float32"
            )
        if self.resize not in {"nearest", "bilinear", "bicubic"}:
            raise ArtifactContractError("resize must be nearest, bilinear, or bicubic")
        if (
            isinstance(self.scale, bool)
            or not isinstance(self.scale, (int, float))
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ArtifactContractError("preprocessing scale must be finite and positive")
        if not _finite_triplet(self.mean):
            raise ArtifactContractError("preprocessing mean must contain three finite values")
        if not _finite_triplet(self.std, positive=True):
            raise ArtifactContractError(
                "preprocessing std must contain three finite positive values"
            )
        if self.clahe is not None and not isinstance(self.clahe, ClaheTransform):
            raise ArtifactContractError("clahe must be a ClaheTransform or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "color_mode": self.color_mode,
            "layout": self.layout,
            "dtype": self.dtype,
            "resize": self.resize,
            "scale": self.scale,
            "mean": list(self.mean),
            "std": list(self.std),
            "clahe": None if self.clahe is None else self.clahe.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ImagePreprocessing:
        _require_exact_keys(
            payload,
            {"color_mode", "layout", "dtype", "resize", "scale", "mean", "std", "clahe"},
            "image preprocessing",
        )
        clahe = payload["clahe"]
        return cls(
            color_mode=payload["color_mode"],
            layout=payload["layout"],
            dtype=payload["dtype"],
            resize=payload["resize"],
            scale=payload["scale"],
            mean=_tuple(payload["mean"], "mean"),
            std=_tuple(payload["std"], "std"),
            clahe=None if clahe is None else ClaheTransform.from_dict(clahe),
        )


@dataclass(frozen=True, slots=True)
class LandmarkGraphPreprocessing:
    coordinate_space: str
    confidence_range: tuple[float, float]
    visibility_encoding: str

    def __post_init__(self) -> None:
        if self.coordinate_space != "crop_normalized_xy":
            raise ArtifactContractError(
                "graph coordinate_space must be crop_normalized_xy"
            )
        if self.confidence_range != (0.0, 1.0):
            raise ArtifactContractError("graph confidence_range must be (0.0, 1.0)")
        if self.visibility_encoding != "binary_0_1":
            raise ArtifactContractError(
                "graph visibility_encoding must be binary_0_1"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_space": self.coordinate_space,
            "confidence_range": list(self.confidence_range),
            "visibility_encoding": self.visibility_encoding,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LandmarkGraphPreprocessing:
        _require_exact_keys(
            payload,
            {"coordinate_space", "confidence_range", "visibility_encoding"},
            "landmark graph preprocessing",
        )
        return cls(
            payload["coordinate_space"],
            _tuple(payload["confidence_range"], "confidence_range"),
            payload["visibility_encoding"],
        )


@dataclass(frozen=True, slots=True)
class ExactOnnxManifest:
    artifact_id: str
    artifact_sha256: str
    input_name: str
    input_shape: tuple[int, ...]
    output_name: str
    output_shape: tuple[int, ...]
    license: ArtifactLicense

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ArtifactContractError("artifact_id must be non-empty")
        if (
            not isinstance(self.artifact_sha256, str)
            or _SHA256_RE.fullmatch(self.artifact_sha256) is None
        ):
            raise ArtifactContractError(
                "artifact_sha256 must be an exact lowercase SHA256 digest"
            )
        if (
            not isinstance(self.input_name, str)
            or not self.input_name
            or not isinstance(self.output_name, str)
            or not self.output_name
        ):
            raise ArtifactContractError("input_name and output_name must be non-empty")
        _validate_static_shape("input_shape", self.input_shape)
        _validate_static_shape("output_shape", self.output_shape)
        if self.input_shape[0] != 1 or self.output_shape[0] != 1:
            raise ArtifactContractError("artifact batch dimensions must be statically fixed at 1")
        if not isinstance(self.license, ArtifactLicense):
            raise ArtifactContractError("license must be an ArtifactLicense")

    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: _json_value(getattr(self, item.name))
            for item in fields(self)
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExactOnnxManifest:
        expected = {item.name for item in fields(cls)}
        _require_exact_keys(payload, expected, cls.__name__)
        values = dict(payload)
        values["input_shape"] = _tuple(values["input_shape"], "input_shape")
        values["output_shape"] = _tuple(values["output_shape"], "output_shape")
        values["license"] = ArtifactLicense.from_dict(values["license"])
        if "keypoint_order" in values:
            values["keypoint_order"] = _tuple(
                values["keypoint_order"], "keypoint_order"
            )
        if "preprocessing" in values:
            preprocessing_type = (
                LandmarkGraphPreprocessing
                if cls is LandmarkGraphManifest
                else ImagePreprocessing
            )
            values["preprocessing"] = preprocessing_type.from_dict(
                values["preprocessing"]
            )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ExactImageOnnxManifest(ExactOnnxManifest):
    preprocessing: ImagePreprocessing

    def __post_init__(self) -> None:
        ExactOnnxManifest.__post_init__(self)
        if len(self.input_shape) != 4 or self.input_shape[1] != 3:
            raise ArtifactContractError("image artifact input_shape must be [1,3,H,W]")
        if not isinstance(self.preprocessing, ImagePreprocessing):
            raise ArtifactContractError("preprocessing must be ImagePreprocessing")


@dataclass(frozen=True, slots=True)
class NoseDetectorManifest(ExactImageOnnxManifest):
    confidence_threshold: float

    def __post_init__(self) -> None:
        ExactImageOnnxManifest.__post_init__(self)
        if len(self.output_shape) != 3 or self.output_shape[2] != 5:
            raise ArtifactContractError(
                "nose detector output_shape must be [1,detections,5] for normalized xyxy+confidence"
            )
        _validate_probability("confidence_threshold", self.confidence_threshold)


@dataclass(frozen=True, slots=True)
class NoseEmbeddingManifest(ExactImageOnnxManifest):
    def __post_init__(self) -> None:
        ExactImageOnnxManifest.__post_init__(self)
        if len(self.output_shape) != 2:
            raise ArtifactContractError("nose embedding output_shape must be [1,D]")


@dataclass(frozen=True, slots=True)
class NoseMaskManifest(ExactImageOnnxManifest):
    threshold: float

    def __post_init__(self) -> None:
        ExactImageOnnxManifest.__post_init__(self)
        if len(self.output_shape) != 4 or self.output_shape[1] != 1:
            raise ArtifactContractError("nose mask output_shape must be [1,1,H,W]")
        _validate_probability("mask threshold", self.threshold)


@dataclass(frozen=True, slots=True)
class LandmarkKeypointManifest(ExactImageOnnxManifest):
    keypoint_order: tuple[str, ...]
    visibility_threshold: float
    min_visible_keypoints: int

    def __post_init__(self) -> None:
        ExactImageOnnxManifest.__post_init__(self)
        _validate_keypoint_order(self.keypoint_order)
        if len(self.output_shape) != 4:
            raise ArtifactContractError("landmark output_shape must be [1,K,H,W]")
        if self.output_shape[1] != len(self.keypoint_order):
            raise ArtifactContractError(
                "landmark heatmap channel count must match keypoint_order"
            )
        if self.output_shape[2] < 2 or self.output_shape[3] < 2:
            raise ArtifactContractError("landmark heatmap spatial dimensions must be at least 2")
        _validate_probability("visibility_threshold", self.visibility_threshold)
        if (
            not isinstance(self.min_visible_keypoints, int)
            or isinstance(self.min_visible_keypoints, bool)
            or not 1 <= self.min_visible_keypoints <= len(self.keypoint_order)
        ):
            raise ArtifactContractError(
                "min_visible_keypoints must be between 1 and the schema size"
            )


@dataclass(frozen=True, slots=True)
class LandmarkGraphManifest(ExactOnnxManifest):
    keypoint_order: tuple[str, ...]
    preprocessing: LandmarkGraphPreprocessing

    def __post_init__(self) -> None:
        ExactOnnxManifest.__post_init__(self)
        _validate_keypoint_order(self.keypoint_order)
        if not isinstance(self.preprocessing, LandmarkGraphPreprocessing):
            raise ArtifactContractError(
                "graph preprocessing must be LandmarkGraphPreprocessing"
            )
        if self.input_shape != (1, len(self.keypoint_order), 4):
            raise ArtifactContractError(
                "landmark graph input_shape must be [1,K,4] for normalized xy, confidence, visibility"
            )
        if len(self.output_shape) != 2:
            raise ArtifactContractError("landmark graph output_shape must be [1,D]")


class ExactOnnxRuntime:
    """CPU-default runtime loaded only after exact byte and graph validation."""

    def __init__(
        self,
        artifact_path: Path,
        manifest: ExactOnnxManifest,
        *,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(manifest, ExactOnnxManifest):
            raise TypeError("manifest must be an ExactOnnxManifest")
        model_bytes = validated_onnx_bytes(artifact_path)
        if sha256(model_bytes).hexdigest() != manifest.artifact_sha256:
            raise ArtifactContractError("artifact SHA256 does not match its manifest")

        import onnx

        graph = onnx.load_model_from_string(model_bytes).graph
        if len(graph.input) != 1 or len(graph.output) != 1:
            raise ArtifactContractError("ONNX graph must have exactly one input and output")
        model_input, model_output = graph.input[0], graph.output[0]
        if model_input.name != manifest.input_name or model_output.name != manifest.output_name:
            raise ArtifactContractError("ONNX tensor names do not match the manifest")
        if (
            model_input.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or model_output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        ):
            raise ArtifactContractError("ONNX input and output tensors must be float32")
        if _onnx_shape(model_input) != manifest.input_shape:
            raise ArtifactContractError("ONNX input shape does not match the manifest")
        if _onnx_shape(model_output) != manifest.output_shape:
            raise ArtifactContractError("ONNX output shape does not match the manifest")

        import onnxruntime as ort

        available = tuple(ort.get_available_providers())
        provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
        if provider not in available:
            raise ArtifactContractError(f"{provider} was requested but is not available")
        session_options = ort.SessionOptions()
        if use_cuda:
            session_options.add_session_config_entry(
                "session.disable_cpu_ep_fallback", "1"
            )
        self._session = ort.InferenceSession(
            model_bytes,
            sess_options=session_options,
            providers=[provider],
            enable_fallback=0,
        )
        self._session.disable_fallback()
        actual_providers = tuple(self._session.get_providers())
        expected_providers = (
            ("CUDAExecutionProvider", "CPUExecutionProvider")
            if use_cuda
            else ("CPUExecutionProvider",)
        )
        if actual_providers != expected_providers:
            raise ArtifactContractError(
                "actual ONNX providers differ from the requested strict "
                f"{'CUDA' if use_cuda else 'CPU'} session"
            )
        runtime_inputs = self._session.get_inputs()
        runtime_outputs = self._session.get_outputs()
        if len(runtime_inputs) != 1 or len(runtime_outputs) != 1:
            raise ArtifactContractError(
                "ONNX Runtime must expose exactly one input and output"
            )
        runtime_input, runtime_output = runtime_inputs[0], runtime_outputs[0]
        if runtime_input.name != manifest.input_name or runtime_output.name != manifest.output_name:
            raise ArtifactContractError("runtime tensor names do not match the manifest")
        if tuple(runtime_input.shape) != manifest.input_shape:
            raise ArtifactContractError("runtime input shape does not match the manifest")
        if tuple(runtime_output.shape) != manifest.output_shape:
            raise ArtifactContractError("runtime output shape does not match the manifest")
        if (
            getattr(runtime_input, "type", "tensor(float)") != "tensor(float)"
            or getattr(runtime_output, "type", "tensor(float)") != "tensor(float)"
        ):
            raise ArtifactContractError("runtime input and output tensors must be float32")
        self.manifest = manifest

    def run(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.dtype != np.float32 or tensor.shape != self.manifest.input_shape:
            raise ArtifactContractError(
                f"runtime input must be float32 {self.manifest.input_shape}, got "
                f"{tensor.dtype} {tensor.shape}"
            )
        outputs = self._session.run(
            [self.manifest.output_name], {self.manifest.input_name: tensor}
        )
        if len(outputs) != 1 or not isinstance(outputs[0], np.ndarray):
            raise ArtifactContractError("runtime must return exactly one ndarray")
        output = outputs[0]
        if output.dtype != np.float32 or output.shape != self.manifest.output_shape:
            raise ArtifactContractError(
                f"runtime output must be float32 {self.manifest.output_shape}, got "
                f"{output.dtype} {output.shape}"
            )
        if not np.isfinite(output).all():
            raise ArtifactContractError("runtime output contains non-finite values")
        return output


def preprocess_image(image: Image.Image, manifest: ExactImageOnnxManifest) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    preprocessing = manifest.preprocessing
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if preprocessing.clahe is not None:
        lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        transform = cv2.createCLAHE(
            clipLimit=float(preprocessing.clahe.clip_limit),
            tileGridSize=preprocessing.clahe.tile_grid_size,
        )
        array = cv2.cvtColor(
            cv2.merge([transform.apply(lightness), channel_a, channel_b]),
            cv2.COLOR_LAB2RGB,
        )
    _, _, height, width = manifest.input_shape
    resampling = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }[preprocessing.resize]
    resized = Image.fromarray(array, mode="RGB").resize((width, height), resampling)
    values = np.asarray(resized, dtype=np.float32) * preprocessing.scale
    mean = np.asarray(preprocessing.mean, dtype=np.float32)
    std = np.asarray(preprocessing.std, dtype=np.float32)
    values = (values - mean) / std
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None], dtype=np.float32)


def _finite_triplet(values: object, *, positive: bool = False) -> bool:
    return (
        isinstance(values, tuple)
        and len(values) == 3
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and (not positive or value > 0)
            for value in values
        )
    )


def _tuple(value: object, name: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ArtifactContractError(f"{name} must be a JSON array")
    return tuple(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _require_exact_keys(
    payload: object, expected: set[str], context: str
) -> None:
    if not isinstance(payload, dict):
        raise ArtifactContractError(f"{context} must be an object")
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise ArtifactContractError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _validate_static_shape(name: str, shape: object) -> None:
    if (
        not isinstance(shape, tuple)
        or not shape
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in shape
        )
    ):
        raise ArtifactContractError(f"{name} must contain static positive integers")


def _validate_probability(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ArtifactContractError(f"{name} must be finite and between 0 and 1")


def _validate_keypoint_order(order: object) -> None:
    if (
        not isinstance(order, tuple)
        or not order
        or not all(isinstance(name, str) and bool(name.strip()) for name in order)
        or len(set(order)) != len(order)
    ):
        raise ArtifactContractError("keypoint_order must contain unique non-empty names")


def _onnx_shape(value_info: object) -> tuple[int, ...]:
    dimensions: list[int] = []
    for dimension in value_info.type.tensor_type.shape.dim:  # type: ignore[attr-defined]
        if dimension.dim_value <= 0:
            raise ArtifactContractError("ONNX shapes must be fully static")
        dimensions.append(int(dimension.dim_value))
    return tuple(dimensions)


__all__ = [
    "ArtifactContractError",
    "ArtifactLicense",
    "ClaheTransform",
    "ExactOnnxRuntime",
    "ImagePreprocessing",
    "LandmarkGraphManifest",
    "LandmarkGraphPreprocessing",
    "LandmarkKeypointManifest",
    "NoseDetectorManifest",
    "NoseEmbeddingManifest",
    "NoseMaskManifest",
    "UsageLane",
    "preprocess_image",
]
