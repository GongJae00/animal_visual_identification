"""Framework-neutral NoseID-v1 value objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NOSE_KEYPOINTS = (
    "left_nostril_center",
    "right_nostril_center",
    "nasal_root_midline",
    "inferior_midline",
    "left_alar_boundary",
    "right_alar_boundary",
)


def _finite_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float32 with shape {shape}")
    return array


@dataclass(frozen=True, slots=True)
class NoseKeypoints:
    xyc: np.ndarray

    def __post_init__(self) -> None:
        value = _finite_array(self.xyc, (6, 3), "nose keypoints")
        if np.any((value[:, 2] < 0.0) | (value[:, 2] > 1.0)):
            raise ValueError("nose keypoint confidence must be in [0, 1]")
        object.__setattr__(self, "xyc", value)


@dataclass(frozen=True, slots=True)
class NoseDetectionResult:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: NoseKeypoints
    native_short_side: float

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if not all(np.isfinite(self.bbox_xyxy)) or x2 <= x1 or y2 <= y1:
            raise ValueError("nose bbox must be a finite non-empty xyxy box")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("nose confidence must be in [0, 1]")
        if not np.isfinite(self.native_short_side) or self.native_short_side <= 0:
            raise ValueError("native nose short side must be positive")


@dataclass(frozen=True, slots=True)
class AlignedNose:
    rgb: np.ndarray
    keypoints_xyc: np.ndarray
    transform: np.ndarray
    normalized_residual: float
    native_short_side: float

    def __post_init__(self) -> None:
        rgb = _finite_array(self.rgb, (3, 448, 448), "aligned RGB")
        if np.any((rgb < 0.0) | (rgb > 1.0)):
            raise ValueError("aligned RGB must be in [0, 1]")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(
            self,
            "keypoints_xyc",
            _finite_array(self.keypoints_xyc, (6, 3), "aligned keypoints"),
        )
        object.__setattr__(
            self, "transform", _finite_array(self.transform, (2, 3), "transform")
        )
        if not 0.0 <= self.normalized_residual <= 0.18:
            raise ValueError("alignment residual exceeds NoseID-v1 admission")


@dataclass(frozen=True, slots=True)
class NoseIDOutput:
    embedding: np.ndarray
    utility: float
    branch_utilities: np.ndarray
    branch_gates: np.ndarray
    quality_vector: np.ndarray

    def __post_init__(self) -> None:
        embedding = _finite_array(self.embedding, (512,), "nose embedding")
        if not np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4):
            raise ValueError("nose embedding must be L2-normalized")
        object.__setattr__(self, "embedding", embedding)
        for name in ("branch_utilities", "branch_gates"):
            value = _finite_array(getattr(self, name), (3,), name)
            if np.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        quality = _finite_array(self.quality_vector, (14,), "quality vector")
        if np.any((quality < 0.0) | (quality > 1.0)):
            raise ValueError("quality vector must be in [0, 1]")
        object.__setattr__(self, "quality_vector", quality)
        if not 0.0 <= self.utility <= 1.0:
            raise ValueError("nose utility must be in [0, 1]")


__all__ = [
    "AlignedNose",
    "NOSE_KEYPOINTS",
    "NoseDetectionResult",
    "NoseIDOutput",
    "NoseKeypoints",
]
