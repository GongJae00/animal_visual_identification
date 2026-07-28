"""Face ReID dataset — loads DogFaceNet 224 crops with provenance."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from cvi.canid_data.adapters import adapt_dogfacenet224
from cvi.canid_data.types import UnifiedCanidSample


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
            "registered_dog_id": row.registered_dog_id,
            "sample_id": row.sample_id,
            "session_id": row.capture_group_id or "unknown",
        }


def build_dogface_dataset(
    data_root: str,
    *,
    identity_to_index: dict[str, int] | None = None,
    augment: object | None = None,
) -> tuple[FaceReIDDataset, tuple[UnifiedCanidSample, ...]]:
    samples = adapt_dogfacenet224(Path(data_root))
    if identity_to_index is None:
        unique_ids = sorted({s.registered_identity_id for s in samples if s.registered_identity_id})
        identity_to_index = {uid: idx for idx, uid in enumerate(unique_ids)}
    dataset = FaceReIDDataset(
        Path(data_root), samples, identity_to_index, augment=augment
    )
    return dataset, samples


__all__ = ["FaceReIDDataset", "build_dogface_dataset"]
