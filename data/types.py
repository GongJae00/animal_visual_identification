"""Framework-free canid data types with strict validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from contracts.identity_ids import compute_generated_identity_id


class CaptureGroupKind(str, Enum):
    REAL_CAMERA_SESSION = "REAL_CAMERA_SESSION"
    VIDEO_TRACK = "VIDEO_TRACK"
    ALBUM_OR_SOURCE_GROUP = "ALBUM_OR_SOURCE_GROUP"
    POSE_VIEW_CLUSTER = "POSE_VIEW_CLUSTER"
    UNKNOWN = "UNKNOWN"


_DOG_BREED_UNKNOWN = "unknown"


def _require_finite_bbox(
    value: object,
    name: str,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{name} must have four coordinates or be None")
    bbox = tuple(float(v) for v in value)
    if not all(np.isfinite(bbox)) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{name} must be a finite non-empty xyxy box")
    return bbox


def _require_keypoints(
    value: object,
    name: str,
) -> dict[str, tuple[float, float, float]] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict or None")
    result: dict[str, tuple[float, float, float]] = {}
    for key, coordinates in value.items():
        if not isinstance(key, str) or not isinstance(coordinates, (tuple, list)):
            raise ValueError(f"{name} keypoint {key!r} is invalid")
        if len(coordinates) not in (2, 3):
            raise ValueError(f"{name} keypoint {key!r} must have xy or xyc")
        x, y = float(coordinates[0]), float(coordinates[1])
        confidence = float(coordinates[2]) if len(coordinates) == 3 else 0.0
        if not all(np.isfinite((x, y, confidence))):
            raise ValueError(f"{name} keypoint {key!r} must be finite")
        result[key] = (x, y, confidence)
    return result


@dataclass(frozen=True, slots=True)
class UnifiedCanidSample:
    sample_id: str
    dataset_name: str
    dataset_version: str
    source_group_id: str
    image_path: str
    image_sha256: str
    width: int
    height: int
    species: str = "Canis lupus familiaris"
    breed: str | None = None
    registered_identity_id: str | None = None
    generated_identity_id: str | None = None
    raw_identity_id: str | None = None
    dog_boxes_xyxy: tuple[float, float, float, float] | None = None
    body_keypoints: dict[str, tuple[float, float, float]] | None = None
    face_box_xyxy: tuple[float, float, float, float] | None = None
    face_landmarks: dict[str, tuple[float, float, float]] | None = None
    head_roi_xyxy: tuple[float, float, float, float] | None = None
    foreground_mask_path: str | None = None
    nose_mask_path: str | None = None
    capture_group_id: str | None = None
    capture_group_kind: CaptureGroupKind = CaptureGroupKind.UNKNOWN
    camera_id: str | None = None
    timestamp_ms: int | None = None
    split_role: str = "UNASSIGNED"
    label_availability: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.dataset_name:
            raise ValueError("sample_id and dataset_name must be non-empty")
        if self.image_path:
            pass
        for name in (
            "dog_boxes_xyxy",
            "face_box_xyxy",
            "head_roi_xyxy",
        ):
            object.__setattr__(
                self, name, _require_finite_bbox(getattr(self, name), name)
            )
        for name in ("body_keypoints", "face_landmarks"):
            object.__setattr__(
                self, name, _require_keypoints(getattr(self, name), name)
            )
        if not isinstance(self.label_availability, dict):
            raise ValueError("label_availability must be a dict")
        if self.generated_identity_id is not None:
            metadata = self.metadata
            generator_id = metadata.get("generated_identity_generator_id")
            source_cluster_token = metadata.get(
                "generated_identity_source_cluster_token"
            )
            if (
                not isinstance(generator_id, str)
                or not isinstance(source_cluster_token, str)
                or self.generated_identity_id
                != compute_generated_identity_id(generator_id, source_cluster_token)
            ):
                raise ValueError(
                    "generated_identity_id requires matching generator and cluster-token metadata"
                )
        avail = dict(self.label_availability)
        defaults = {
            "identity": self.raw_identity_id is not None,
            "generated_identity": self.generated_identity_id is not None,
            "breed": self.breed is not None and self.breed != _DOG_BREED_UNKNOWN,
            "dog_bbox": self.dog_boxes_xyxy is not None,
            "face_bbox": self.face_box_xyxy is not None,
            "face_landmarks": self.face_landmarks is not None,
            "body_keypoints": self.body_keypoints is not None,
            "nose_mask": self.nose_mask_path is not None,
            "capture_group": self.capture_group_id is not None,
            "camera": self.camera_id is not None,
        }
        for label, present in defaults.items():
            avail.setdefault(label, present)
        object.__setattr__(self, "label_availability", avail)


class DatasetAdmission(str, Enum):
    ADMIT_TRAIN = "ADMIT_TRAIN"
    ADMIT_VALIDATION_ONLY = "ADMIT_VALIDATION_ONLY"
    ADMIT_TEACHER_ONLY = "ADMIT_TEACHER_ONLY"
    BLOCKED_LICENSE = "BLOCKED_LICENSE"
    BLOCKED_ACCESS = "BLOCKED_ACCESS"
    REJECT_LABEL_QUALITY = "REJECT_LABEL_QUALITY"
    REJECT_NOT_CANID = "REJECT_NOT_CANID"


@dataclass(frozen=True, slots=True)
class CanidDatasetRecord:
    canonical_name: str
    official_name: str
    version: str
    license_id: str
    url: str | None
    data_root: str
    sha256_checksums: dict[str, str]
    total_images: int
    total_identities: int
    capture_group_kind: CaptureGroupKind
    has_dog_bbox: bool
    has_face_bbox: bool
    has_face_landmarks: bool
    has_body_keypoints: bool
    has_breed: bool
    has_nose_mask: bool
    admission: DatasetAdmission = DatasetAdmission.ADMIT_TRAIN
    notes: str = ""


__all__ = [
    "CanidDatasetRecord",
    "CaptureGroupKind",
    "DatasetAdmission",
    "UnifiedCanidSample",
]
