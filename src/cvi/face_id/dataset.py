"""Face ReID dataset — loads DogFaceNet 224 crops with provenance."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cvi.canid_data.adapters import adapt_dogfacenet224
from cvi.canid_data.types import UnifiedCanidSample
from cvi.localization.roi import normalize_source_point_to_square_crop

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
    ) -> None:
        self.crop_root = Path(crop_root)
        self.records = records
        self.identity_to_index = identity_to_index
        self.augment = augment
        if not records:
            raise ValueError("ROI FaceID dataset must not be empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(self.crop_root / record["face_crop_path"]) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1))
        if self.augment is not None:
            tensor = self.augment(tensor)
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
        identity = record["registered_identity_id"]
        return {
            "rgb": tensor.clamp(0, 1),
            "landmarks": landmarks,
            "quality_target": float(record["face_quality"]["overall"]),
            "identity_index": self.identity_to_index[identity],
            "registered_dog_id": identity,
            "sample_id": record["sample_id"],
            "session_id": record["capture_group_id"] or "unknown",
        }


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


__all__ = ["FaceReIDDataset", "RoiFaceReIDDataset", "build_dogface_dataset"]
