"""Framework-free localization types: detection, keypoint, landmark, ROI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


AP10K_BODY_17_SCHEMA = "ap10k-dog-17"
AP10K_BODY_17_KEYPOINT_NAMES = (
    "left_eye",
    "right_eye",
    "nose_center",
    "neck",
    "tail_base",
    "left_shoulder",
    "left_elbow",
    "left_front_paw",
    "right_shoulder",
    "right_elbow",
    "right_front_paw",
    "left_hip",
    "left_knee",
    "left_back_paw",
    "right_hip",
    "right_knee",
    "right_back_paw",
)
AP10K_BODY_17_EDGES = (
    ("left_eye", "nose_center"),
    ("right_eye", "nose_center"),
    ("left_eye", "neck"),
    ("right_eye", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_front_paw"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_front_paw"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_back_paw"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_back_paw"),
    ("left_hip", "tail_base"),
    ("right_hip", "tail_base"),
)


@dataclass(frozen=True, slots=True)
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    class_name: str = "dog"

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("detection confidence must be in [0, 1]")
        if not (self.x1 < self.x2 and self.y1 < self.y2):
            raise ValueError("detection bbox must be non-empty")
        for value in (self.x1, self.y1, self.x2, self.y2):
            if not np.isfinite(value):
                raise ValueError("detection coordinates must be finite")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class Keypoint:
    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.x) or not np.isfinite(self.y):
            raise ValueError("keypoint coordinates must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("keypoint confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class KeypointSet:
    keypoints: dict[str, Keypoint]
    schema: str

    def __post_init__(self) -> None:
        if not self.schema or not self.keypoints:
            raise ValueError("keypoint set must have a schema and at least one point")
        for name in self.keypoints:
            if not isinstance(name, str) or not name:
                raise ValueError("keypoint name must be non-empty")

    def named(self, name: str) -> Keypoint | None:
        return self.keypoints.get(name)


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    image_id: str
    dog_boxes: tuple[DetectionBox, ...]
    face_boxes: tuple[DetectionBox, ...]
    nose_boxes: tuple[DetectionBox, ...]
    body_keypoints: tuple[KeypointSet, ...]
    face_landmarks: tuple[KeypointSet, ...]
    model_name: str
    model_family: str
    inference_ms: float
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.image_id or not self.model_name:
            raise ValueError("image_id and model_name must be non-empty")
        if self.inference_ms < 0:
            raise ValueError("inference time must be non-negative")


__all__ = [
    "DetectionBox",
    "AP10K_BODY_17_EDGES",
    "AP10K_BODY_17_KEYPOINT_NAMES",
    "AP10K_BODY_17_SCHEMA",
    "Keypoint",
    "KeypointSet",
    "LocalizationResult",
]
