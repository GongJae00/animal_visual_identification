"""Strict NoseID-v1 manifest parser and oracle-ROI dataset."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import uuid
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from cvi.nose_id.alignment import align_nose
from cvi.nose_id.types import NOSE_KEYPOINTS, NoseKeypoints


_SPLIT_ROLES = ("TRAIN", "DEV", "FUSION_CAL", "TEST")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_uuid5(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("registered_dog_id must be a string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError("registered_dog_id must be a UUID") from exc
    if parsed.version != 5 or str(parsed) != value:
        raise ValueError("registered_dog_id must be canonical lowercase UUIDv5")
    return value


def _relative_path(value: object, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{name} must be a normalized relative path")
    return path


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


@dataclass(frozen=True, slots=True)
class NoseIDSample:
    sample_id: str
    image_path: PurePosixPath
    image_sha256: str
    image_width: int
    image_height: int
    registered_dog_id: str
    session_id: str
    camera_id: str
    video_id: str
    frame_index: int
    timestamp_ms: int
    nose_bbox_xyxy: tuple[float, float, float, float]
    keypoints_xy: np.ndarray
    keypoint_visibility: np.ndarray
    semantic_mask_path: PurePosixPath
    semantic_mask_sha256: str
    semantic_mask_box_xyxy: tuple[float, float, float, float]
    invalid_mask_path: PurePosixPath
    invalid_mask_sha256: str
    invalid_mask_box_xyxy: tuple[float, float, float, float]
    split_role: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NoseIDSample":
        required = {
            "schema_version", "sample_id", "image", "identity", "capture",
            "annotations", "split_role",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("NoseID sample keys differ")
        if payload["schema_version"] != "cvi.noseid.sample.v1":
            raise ValueError("unsupported NoseID sample schema")
        image = payload["image"]
        identity = payload["identity"]
        capture = payload["capture"]
        annotations = payload["annotations"]
        for name, value in (("image", image), ("identity", identity), ("capture", capture), ("annotations", annotations)):
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be an object")
        if set(image) != {"relative_path", "sha256", "width", "height"}:
            raise ValueError("NoseID image keys differ")
        if not {"registered_dog_id"} <= set(identity) <= {"registered_dog_id", "breed", "coat_group"}:
            raise ValueError("NoseID identity keys differ")
        if set(capture) != {"session_id", "camera_id", "video_id", "frame_index", "timestamp_ms"}:
            raise ValueError("NoseID capture keys differ")
        if set(annotations) != {
            "dog_bbox_xyxy", "face_bbox_xyxy", "nose_bbox_xyxy", "nose_keypoints",
            "semantic_mask", "invalid_mask",
        }:
            raise ValueError("NoseID annotation keys differ")
        keypoints = annotations["nose_keypoints"]
        if not isinstance(keypoints, dict) or set(keypoints) != {"order", "xy", "visibility"}:
            raise ValueError("NoseID keypoint keys differ")
        if tuple(keypoints["order"]) != NOSE_KEYPOINTS:
            raise ValueError("NoseID keypoint order differs")
        xy = np.asarray(keypoints["xy"], dtype=np.float32)
        visibility = np.asarray(keypoints["visibility"], dtype=np.int64)
        if xy.shape != (6, 2) or not np.isfinite(xy).all():
            raise ValueError("nose keypoints xy must have finite shape [6,2]")
        if visibility.shape != (6,) or np.any((visibility < 0) | (visibility > 2)):
            raise ValueError("nose keypoint visibility must contain 0, 1, or 2")
        semantic = annotations["semantic_mask"]
        invalid = annotations["invalid_mask"]
        if not isinstance(semantic, dict) or set(semantic) != {"relative_path", "sha256", "box_xyxy", "classes"}:
            raise ValueError("semantic mask keys differ")
        if semantic["classes"] != {"0": "context", "1": "nasal_surface", "2": "nostril"}:
            raise ValueError("semantic mask classes differ")
        if not isinstance(invalid, dict) or set(invalid) != {"relative_path", "sha256", "box_xyxy", "positive"}:
            raise ValueError("invalid mask keys differ")
        if invalid["positive"] != "specular_or_occluded_or_contaminated":
            raise ValueError("invalid mask semantics differ")
        bbox = tuple(float(value) for value in annotations["nose_bbox_xyxy"])
        if len(bbox) != 4 or not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("nose bbox must be finite non-empty xyxy")
        semantic_box = tuple(float(value) for value in semantic["box_xyxy"])
        invalid_box = tuple(float(value) for value in invalid["box_xyxy"])
        for name, box in (("semantic mask", semantic_box), ("invalid mask", invalid_box)):
            if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"{name} box must be finite non-empty xyxy")
        role = payload["split_role"]
        if role not in _SPLIT_ROLES:
            raise ValueError("unsupported NoseID split role")
        sample_id = payload["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be non-empty")
        width, height = image["width"], image["height"]
        frame_index, timestamp_ms = capture["frame_index"], capture["timestamp_ms"]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (width, height, frame_index, timestamp_ms)) or width == 0 or height == 0:
            raise ValueError("image dimensions and capture indices must be non-negative integers")
        strings = [capture["session_id"], capture["camera_id"], capture["video_id"]]
        if any(not isinstance(value, str) or not value for value in strings):
            raise ValueError("capture identifiers must be non-empty strings")
        return cls(
            sample_id=sample_id,
            image_path=_relative_path(image["relative_path"], "image path"),
            image_sha256=_require_sha256(image["sha256"], "image sha256"),
            image_width=width,
            image_height=height,
            registered_dog_id=_canonical_uuid5(identity["registered_dog_id"]),
            session_id=capture["session_id"],
            camera_id=capture["camera_id"],
            video_id=capture["video_id"],
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            nose_bbox_xyxy=bbox,
            keypoints_xy=xy,
            keypoint_visibility=visibility,
            semantic_mask_path=_relative_path(semantic["relative_path"], "semantic mask path"),
            semantic_mask_sha256=_require_sha256(semantic["sha256"], "semantic mask sha256"),
            semantic_mask_box_xyxy=semantic_box,
            invalid_mask_path=_relative_path(invalid["relative_path"], "invalid mask path"),
            invalid_mask_sha256=_require_sha256(invalid["sha256"], "invalid mask sha256"),
            invalid_mask_box_xyxy=invalid_box,
            split_role=role,
        )


def load_noseid_manifest(path: Path) -> tuple[NoseIDSample, ...]:
    rows: list[NoseIDSample] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(line.encode("utf-8")) > 1_048_576:
                raise ValueError(f"manifest line {line_number} exceeds size cap")
            if not line.strip():
                raise ValueError(f"manifest line {line_number} is empty")
            payload = json.loads(line, object_pairs_hook=_strict_object, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON number: {value}")))
            row = NoseIDSample.from_dict(payload)
            if row.sample_id in seen:
                raise ValueError(f"duplicate NoseID sample_id: {row.sample_id}")
            seen.add(row.sample_id)
            rows.append(row)
    if not rows:
        raise ValueError("NoseID manifest must not be empty")
    return tuple(rows)


def load_identity_split(path: Path) -> dict[str, frozenset[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    if not isinstance(payload, dict) or set(payload) != {"schema_version", *_SPLIT_ROLES}:
        raise ValueError("NoseID split keys differ")
    if payload["schema_version"] != "cvi.noseid.identity_split.v1":
        raise ValueError("unsupported NoseID identity split schema")
    result: dict[str, frozenset[str]] = {}
    observed: set[str] = set()
    for role in _SPLIT_ROLES:
        values = payload[role]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError(f"{role} identities must be a unique array")
        identities = frozenset(_canonical_uuid5(value) for value in values)
        if observed & identities:
            raise ValueError("NoseID identity splits must be disjoint")
        observed.update(identities)
        result[role] = identities
    return result


def validate_manifest_split(
    rows: tuple[NoseIDSample, ...],
    identity_split: dict[str, frozenset[str]],
) -> None:
    if set(identity_split) != set(_SPLIT_ROLES):
        raise ValueError("NoseID identity split roles differ")
    seen_role: dict[str, str] = {}
    for row in rows:
        if row.registered_dog_id not in identity_split[row.split_role]:
            raise ValueError("NoseID manifest row differs from identity split")
        previous = seen_role.setdefault(row.registered_dog_id, row.split_role)
        if previous != row.split_role:
            raise ValueError("NoseID identity appears in multiple manifest roles")


class NoseIDDataset(Dataset):
    """Decode authenticated samples and produce oracle-aligned training tensors."""

    def __init__(
        self,
        root: Path,
        rows: tuple[NoseIDSample, ...],
        identity_to_index: dict[str, int],
        *,
        identity_split: dict[str, frozenset[str]],
        split_role: str,
    ) -> None:
        self.root = Path(os.path.abspath(os.fspath(root)))
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("NoseID data root must be a local directory")
        self.rows = rows
        self.identity_to_index = dict(identity_to_index)
        if not rows:
            raise ValueError("NoseID dataset rows must not be empty")
        validate_manifest_split(rows, identity_split)
        if split_role not in _SPLIT_ROLES or any(row.split_role != split_role for row in rows):
            raise ValueError("NoseID dataset rows must match one explicit split role")
        if set(self.identity_to_index) != {row.registered_dog_id for row in rows}:
            raise ValueError("NoseID identity index population differs from rows")
        indices = list(self.identity_to_index.values())
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in indices)
            or sorted(indices) != list(range(len(indices)))
        ):
            raise ValueError("NoseID identity indices must be contiguous unique integers")

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, relative: PurePosixPath, expected_sha256: str) -> Path:
        path = self.root.joinpath(*relative.parts)
        current = path.parent
        while current != self.root:
            if current.is_symlink():
                raise ValueError(f"NoseID artifact parent is a symlink: {relative}")
            current = current.parent
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
            raise ValueError(f"NoseID artifact differs: {relative}")
        return path

    @staticmethod
    def _expand_mask(
        mask: np.ndarray,
        box: tuple[float, float, float, float],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        if mask.shape == image_shape:
            return mask
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        if x1 < 0 or y1 < 0 or x2 > image_shape[1] or y2 > image_shape[0]:
            raise ValueError("mask box exceeds source image")
        if mask.shape != (y2 - y1, x2 - x1):
            raise ValueError("mask dimensions differ from declared box")
        expanded = np.zeros(image_shape, dtype=mask.dtype)
        expanded[y1:y2, x1:x2] = mask
        return expanded

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self._path(row.image_path, row.image_sha256)
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        if image.shape[:2] != (row.image_height, row.image_width):
            raise ValueError("NoseID decoded image dimensions differ")
        confidence = np.where(row.keypoint_visibility == 2, 1.0, np.where(row.keypoint_visibility == 1, 0.5, 0.0))
        keypoints = NoseKeypoints(np.concatenate([row.keypoints_xy, confidence[:, None]], axis=1))
        short_side = min(row.nose_bbox_xyxy[2] - row.nose_bbox_xyxy[0], row.nose_bbox_xyxy[3] - row.nose_bbox_xyxy[1])
        aligned = align_nose(image, keypoints, native_short_side=short_side)
        semantic_path = self._path(row.semantic_mask_path, row.semantic_mask_sha256)
        invalid_path = self._path(row.invalid_mask_path, row.invalid_mask_sha256)
        with Image.open(semantic_path) as opened:
            semantic = np.asarray(opened.convert("L"), dtype=np.uint8)
        with Image.open(invalid_path) as opened:
            invalid = np.asarray(opened.convert("L"), dtype=np.uint8)
        semantic = self._expand_mask(
            semantic, row.semantic_mask_box_xyxy, image.shape[:2]
        )
        invalid = self._expand_mask(
            invalid, row.invalid_mask_box_xyxy, image.shape[:2]
        )
        semantic_aligned = cv2.warpAffine(semantic, aligned.transform, (448, 448), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        invalid_aligned = cv2.warpAffine(invalid, aligned.transform, (448, 448), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        if np.any(semantic_aligned > 2):
            raise ValueError("semantic mask contains an unknown class")
        invalid_binary = (invalid_aligned > 0).astype(np.float32)
        return {
            "aligned_rgb": torch.from_numpy(aligned.rgb),
            "aligned_kp": torch.from_numpy(aligned.keypoints_xyc),
            "semantic_mask": torch.from_numpy(semantic_aligned.astype(np.int64)),
            "invalid_mask": torch.from_numpy(invalid_binary[None]),
            "identity_index": self.identity_to_index[row.registered_dog_id],
            "registered_dog_id": row.registered_dog_id,
            "session_id": row.session_id,
            "sample_id": row.sample_id,
            "native_short_side": float(short_side),
            "alignment_rms": aligned.normalized_residual,
        }


__all__ = [
    "NoseIDDataset",
    "NoseIDSample",
    "load_identity_split",
    "load_noseid_manifest",
    "validate_manifest_split",
]
