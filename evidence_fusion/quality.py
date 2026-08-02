from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, TypeAlias

import numpy as np
from PIL import Image


RoiBox: TypeAlias = tuple[int, int, int, int]


class QualityState(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNAVAILABLE = "UNAVAILABLE"


class QualityReason(str, Enum):
    QUALITY_ACCEPTABLE = "QUALITY_ACCEPTABLE"
    LOW_SHARPNESS = "LOW_SHARPNESS"
    LOW_BRIGHTNESS = "LOW_BRIGHTNESS"
    HIGH_BRIGHTNESS = "HIGH_BRIGHTNESS"
    LOW_CONTRAST = "LOW_CONTRAST"
    IMAGE_UNAVAILABLE = "IMAGE_UNAVAILABLE"
    INVALID_DIMENSIONS = "INVALID_DIMENSIONS"
    DIMENSIONS_TOO_SMALL = "DIMENSIONS_TOO_SMALL"
    DIMENSIONS_TOO_LARGE = "DIMENSIONS_TOO_LARGE"
    PIXEL_LIMIT_EXCEEDED = "PIXEL_LIMIT_EXCEEDED"
    INVALID_ROI = "INVALID_ROI"
    ROI_OUT_OF_BOUNDS = "ROI_OUT_OF_BOUNDS"
    ROI_TOO_SMALL = "ROI_TOO_SMALL"
    DIAGNOSTIC_ERROR = "DIAGNOSTIC_ERROR"
    MAPPING_NOT_CONFIGURED = "MAPPING_NOT_CONFIGURED"
    MAPPING_ERROR = "MAPPING_ERROR"
    INVALID_MAPPING_RESULT = "INVALID_MAPPING_RESULT"


# Descriptive alias for callers that name enums by their serialized role.
QualityReasonCode = QualityReason


@dataclass(frozen=True, slots=True)
class QualityLimits:
    min_dimension: int = 3
    max_dimension: int = 16_384
    max_pixels: int = 40_000_000

    def __post_init__(self) -> None:
        values = (self.min_dimension, self.max_dimension, self.max_pixels)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("quality limits must be integers")
        if self.min_dimension < 3:
            raise ValueError("min_dimension must be at least 3")
        if self.max_dimension < self.min_dimension:
            raise ValueError("max_dimension must not be smaller than min_dimension")
        if self.max_pixels < self.min_dimension * self.min_dimension:
            raise ValueError("max_pixels is too small for the minimum dimensions")


DEFAULT_QUALITY_LIMITS = QualityLimits()


@dataclass(frozen=True, slots=True)
class QualityDiagnostics:
    """Uncalibrated image diagnostics; intensity values use the 0..255 scale."""

    sharpness: float
    brightness: float
    contrast: float
    width: int
    height: int
    pixel_count: int

    def __post_init__(self) -> None:
        dimensions = (self.width, self.height, self.pixel_count)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
            raise TypeError("quality diagnostic dimensions must be integers")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("quality diagnostic dimensions must be positive")
        if self.pixel_count != self.width * self.height:
            raise ValueError("pixel_count must equal width * height")
        values = (self.sharpness, self.brightness, self.contrast)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("quality diagnostics must be finite")
        if self.sharpness < 0.0 or not 0.0 <= self.brightness <= 255.0:
            raise ValueError("quality diagnostics are outside their valid range")
        if self.contrast < 0.0:
            raise ValueError("contrast must be non-negative")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sharpness": self.sharpness,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "width": self.width,
            "height": self.height,
            "pixel_count": self.pixel_count,
        }


@dataclass(frozen=True, slots=True)
class QualityMapping:
    state: QualityState
    reason_codes: tuple[QualityReason, ...]
    score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, QualityState):
            raise TypeError("quality mapping state must be a QualityState")
        if not self.reason_codes or not all(
            isinstance(reason, QualityReason) for reason in self.reason_codes
        ):
            raise ValueError("quality mapping requires typed reason codes")
        if self.state is QualityState.UNAVAILABLE:
            if self.score is not None:
                raise ValueError("unavailable quality cannot have a score")
            return
        if self.score is None or not math.isfinite(self.score):
            raise ValueError("eligible and ineligible quality require a finite score")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("quality score must be between 0 and 1")


QualityMapper: TypeAlias = Callable[[QualityDiagnostics], QualityMapping]


@dataclass(frozen=True, slots=True)
class QualityObservation:
    channel: str
    state: QualityState
    reason_codes: tuple[QualityReason, ...]
    diagnostics: QualityDiagnostics | None = None
    score: float | None = None
    roi_box: RoiBox | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError("quality observation channel must be non-empty")
        QualityMapping(self.state, self.reason_codes, self.score)
        if self.state is not QualityState.UNAVAILABLE and self.diagnostics is None:
            raise ValueError("available quality requires diagnostics")

    @property
    def available(self) -> bool:
        return self.state is not QualityState.UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "state": self.state.value,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "score": self.score,
            "roi_box": list(self.roi_box) if self.roi_box is not None else None,
            "diagnostics": (
                self.diagnostics.to_dict() if self.diagnostics is not None else None
            ),
        }


def _unavailable(
    channel: str,
    reason: QualityReason,
    *,
    diagnostics: QualityDiagnostics | None = None,
    roi_box: RoiBox | None = None,
) -> QualityObservation:
    return QualityObservation(
        channel=channel,
        state=QualityState.UNAVAILABLE,
        reason_codes=(reason,),
        diagnostics=diagnostics,
        roi_box=roi_box,
    )


def _validate_dimensions(
    size: object,
    limits: QualityLimits,
) -> QualityReason | None:
    if not isinstance(size, tuple) or len(size) != 2:
        return QualityReason.INVALID_DIMENSIONS
    width, height = size
    if any(isinstance(value, bool) or not isinstance(value, int) for value in size):
        return QualityReason.INVALID_DIMENSIONS
    if width < limits.min_dimension or height < limits.min_dimension:
        return QualityReason.DIMENSIONS_TOO_SMALL
    if width > limits.max_dimension or height > limits.max_dimension:
        return QualityReason.DIMENSIONS_TOO_LARGE
    if width * height > limits.max_pixels:
        return QualityReason.PIXEL_LIMIT_EXCEEDED
    return None


def validate_roi_box(
    roi_box: object,
    image_size: tuple[int, int],
    limits: QualityLimits = DEFAULT_QUALITY_LIMITS,
) -> tuple[RoiBox | None, QualityReason | None]:
    dimension_reason = _validate_dimensions(image_size, limits)
    if dimension_reason is not None:
        return None, dimension_reason
    if not isinstance(roi_box, tuple) or len(roi_box) != 4:
        return None, QualityReason.INVALID_ROI
    if any(isinstance(value, bool) or not isinstance(value, int) for value in roi_box):
        return None, QualityReason.INVALID_ROI
    x0, y0, x1, y1 = roi_box
    width, height = image_size
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None, QualityReason.ROI_OUT_OF_BOUNDS
    if x1 <= x0 or y1 <= y0:
        return None, QualityReason.INVALID_ROI
    if x1 - x0 < limits.min_dimension or y1 - y0 < limits.min_dimension:
        return None, QualityReason.ROI_TOO_SMALL
    return (x0, y0, x1, y1), None


def _quality_diagnostics(
    image: Image.Image,
    limits: QualityLimits = DEFAULT_QUALITY_LIMITS,
) -> QualityDiagnostics:
    dimension_reason = _validate_dimensions(image.size, limits)
    if dimension_reason is not None:
        raise ValueError(dimension_reason.value)
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    if arr.ndim != 2 or arr.shape != (image.height, image.width):
        raise ValueError("grayscale conversion returned unexpected dimensions")
    padded = np.pad(arr, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * arr
    )
    return QualityDiagnostics(
        sharpness=float(np.var(laplacian, dtype=np.float64)),
        brightness=float(np.mean(arr, dtype=np.float64)),
        contrast=float(np.std(arr, dtype=np.float64)),
        width=image.width,
        height=image.height,
        pixel_count=image.width * image.height,
    )


def observe_quality(
    image: Image.Image,
    channel: str,
    *,
    roi_box: RoiBox | None = None,
    mapper: QualityMapper | None = None,
    limits: QualityLimits = DEFAULT_QUALITY_LIMITS,
) -> QualityObservation:
    """Measure an image and apply an explicit channel-owned mapping, if supplied."""
    try:
        image_size = image.size
    except Exception:
        return _unavailable(channel, QualityReason.IMAGE_UNAVAILABLE)

    dimension_reason = _validate_dimensions(image_size, limits)
    if dimension_reason is not None:
        return _unavailable(channel, dimension_reason)

    validated_roi: RoiBox | None = None
    target = image
    if roi_box is not None:
        validated_roi, roi_reason = validate_roi_box(roi_box, image_size, limits)
        if roi_reason is not None:
            return _unavailable(channel, roi_reason)
        assert validated_roi is not None
        try:
            target = image.crop(validated_roi)
        except Exception:
            return _unavailable(
                channel,
                QualityReason.DIAGNOSTIC_ERROR,
                roi_box=validated_roi,
            )

    try:
        diagnostics = _quality_diagnostics(target, limits)
    except Exception:
        return _unavailable(
            channel,
            QualityReason.DIAGNOSTIC_ERROR,
            roi_box=validated_roi,
        )

    if mapper is None:
        return _unavailable(
            channel,
            QualityReason.MAPPING_NOT_CONFIGURED,
            diagnostics=diagnostics,
            roi_box=validated_roi,
        )
    try:
        mapping = mapper(diagnostics)
    except Exception:
        return _unavailable(
            channel,
            QualityReason.MAPPING_ERROR,
            diagnostics=diagnostics,
            roi_box=validated_roi,
        )
    if not isinstance(mapping, QualityMapping):
        return _unavailable(
            channel,
            QualityReason.INVALID_MAPPING_RESULT,
            diagnostics=diagnostics,
            roi_box=validated_roi,
        )
    return QualityObservation(
        channel=channel,
        state=mapping.state,
        reason_codes=mapping.reason_codes,
        diagnostics=diagnostics,
        score=mapping.score,
        roi_box=validated_roi,
    )


def estimate_sharpness(image: Image.Image) -> float:
    return _quality_diagnostics(image).sharpness


def estimate_blur(image: Image.Image) -> float:
    return float(np.clip(estimate_sharpness(image) / 100.0, 0.0, 1.0))


def estimate_brightness(image: Image.Image) -> float:
    return _quality_diagnostics(image).brightness / 255.0


def estimate_contrast(image: Image.Image) -> float:
    return _quality_diagnostics(image).contrast / 128.0


def estimate_occlusion(
    image: Image.Image,
    face_box: RoiBox | None = None,
) -> float:
    if face_box is None:
        return 0.0
    dimension_reason = _validate_dimensions(image.size, DEFAULT_QUALITY_LIMITS)
    if dimension_reason is not None:
        raise ValueError(dimension_reason.value)
    validated, roi_reason = validate_roi_box(face_box, image.size)
    if roi_reason is not None or validated is None:
        raise ValueError((roi_reason or QualityReason.INVALID_ROI).value)
    face = np.asarray(image.crop(validated).convert("L"), dtype=np.float32)
    dark_ratio = float(np.mean(face < 30))
    return min(dark_ratio * 5.0, 1.0)


def overall_quality(image: Image.Image) -> float:
    diagnostics = _quality_diagnostics(image)
    sharpness = float(np.clip(diagnostics.sharpness / 100.0, 0.0, 1.0))
    brightness = diagnostics.brightness / 255.0
    contrast = diagnostics.contrast / 128.0
    score = (
        0.5 * sharpness
        + 0.25 * (1.0 - abs(brightness - 0.5) * 2)
        + 0.25 * min(contrast, 1.0)
    )
    return float(np.clip(score, 0.0, 1.0))
