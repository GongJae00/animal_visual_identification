"""Face ReID dataset — loads DogFaceNet 224 crops with provenance."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset

from data_pipeline.adapters import adapt_dogfacenet224
from data_pipeline.types import UnifiedCanidSample
from localization.roi import normalize_source_point_to_square_crop

_POSE_ORDER = (
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
_ALIGNMENT_TARGETS = np.asarray(
    ((0.35, 0.35), (0.65, 0.35), (0.50, 0.58)), dtype=np.float32
)


def align_face_rgb(
    rgb: np.ndarray,
    landmarks: torch.Tensor,
    *,
    minimum_confidence: float = 0.1,
) -> tuple[np.ndarray, bool]:
    """Align eye-eye-nose anchors while preserving a deterministic fallback."""

    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("Face alignment RGB must be finite HxWx3")
    points = landmarks[:3].detach().cpu().numpy()
    if (
        points.shape != (3, 3)
        or np.any(points[:, 2] < minimum_confidence)
        or not np.isfinite(points).all()
    ):
        return image, False
    source = points[:, :2] * np.asarray((image.shape[1] - 1, image.shape[0] - 1))
    target = _ALIGNMENT_TARGETS * np.asarray((223.0, 223.0), dtype=np.float32)
    transform = cv2.getAffineTransform(source.astype(np.float32), target)
    if not np.isfinite(transform).all() or abs(float(np.linalg.det(transform[:, :2]))) < 1e-6:
        return image, False
    aligned = cv2.warpAffine(
        image,
        transform,
        (224, 224),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return np.asarray(aligned, dtype=np.float32), True


class FaceReIDDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: tuple[UnifiedCanidSample, ...],
        identity_to_index: dict[str, int],
        *,
        augment: object | None = None,
    ) -> None:
        self.root = Path(os.path.abspath(os.fspath(root)))
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("FaceID data root must be a local directory")
        self.rows = rows
        self.identity_to_index = dict(identity_to_index)
        self.augment = augment
        if not rows:
            raise ValueError("FaceID dataset must not be empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.root / row.image_path
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1))
        if tensor.shape != (3, 224, 224):
            tensor = torch.nn.functional.interpolate(
                tensor[None], size=(224, 224), mode="bilinear"
            )[0]
        if self.augment is not None:
            tensor = self.augment(tensor)
        return {
            "rgb": tensor.clamp(0, 1),
            "identity_index": self.identity_to_index[row.registered_identity_id],
            "registered_dog_id": row.registered_identity_id,
            "sample_id": row.sample_id,
            "session_id": row.capture_group_id or "unknown",
        }


class RoiFaceReIDDataset(Dataset):
    """Landmark-aware FaceID samples exported by the localization pipeline."""

    def __init__(
        self,
        crop_root: Path,
        records: tuple[dict[str, Any], ...],
        identity_to_index: dict[str, int],
        *,
        augment: object | None = None,
        align: bool = False,
        paired_augment: bool = False,
    ) -> None:
        self.crop_root = Path(crop_root)
        self.records = records
        self.identity_to_index = identity_to_index
        self.augment = augment
        self.align = align
        self.paired_augment = paired_augment
        if paired_augment and augment is None:
            raise ValueError("paired_augment requires an augmentation callable")
        if not records:
            raise ValueError("ROI FaceID dataset must not be empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(self.crop_root / record["face_crop_path"]) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        landmarks = torch.zeros((17, 3), dtype=torch.float32)
        face_crop_rect = record["face_crop_rect_xyxy"]
        if face_crop_rect is not None and record["body_keypoints"] is not None:
            for point_index, name in enumerate(_POSE_ORDER):
                point = record["body_keypoints"].get(name)
                if point is not None:
                    normalized_x, normalized_y = normalize_source_point_to_square_crop(
                        point[0], point[1], face_crop_rect
                    )
                    landmarks[point_index] = torch.tensor(
                        [normalized_x, normalized_y, point[2]]
                    )
        alignment_applied = False
        if self.align:
            rgb, alignment_applied = align_face_rgb(rgb, landmarks)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        second = None
        if self.augment is not None:
            if self.paired_augment:
                second = self.augment(tensor.clone())
            else:
                tensor = self.augment(tensor)
        identity = record["registered_identity_id"]
        result = {
            "rgb": tensor.clamp(0, 1),
            "landmarks": landmarks,
            "quality_target": float(record["face_quality"]["overall"]),
            "identity_index": self.identity_to_index[identity],
            "registered_dog_id": identity,
            "sample_id": record["sample_id"],
            "session_id": record["capture_group_id"] or "unknown",
            "alignment_applied": alignment_applied,
        }
        if second is not None:
            result["second_rgb"] = second.clamp(0, 1)
        return result


def build_dogface_dataset(
    data_root: str,
    *,
    identity_to_index: dict[str, int] | None = None,
    augment: object | None = None,
) -> tuple[FaceReIDDataset, tuple[UnifiedCanidSample, ...]]:
    samples = adapt_dogfacenet224(Path(data_root))
    if identity_to_index is None:
        unique_ids = sorted(
            {s.registered_identity_id for s in samples if s.registered_identity_id}
        )
        identity_to_index = {uid: idx for idx, uid in enumerate(unique_ids)}
    dataset = FaceReIDDataset(
        Path(data_root), samples, identity_to_index, augment=augment
    )
    return dataset, samples


__all__ = [
    "FaceReIDDataset",
    "RoiFaceReIDDataset",
    "align_face_rgb",
    "build_dogface_dataset",
]
