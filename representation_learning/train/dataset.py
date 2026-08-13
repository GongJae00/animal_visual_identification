from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.adapters import (
    PetFaceDogSplitSample,
    load_petface_dog_split,
    read_petface_dog_images,
)
from data.source_lock import get_record
from data.types import DatasetAdmission


class PetFaceDataset(Dataset):
    """Map-style view over official PetFace dog metadata and the local dog archive."""

    def __init__(
        self,
        root: Path,
        transform: Any | None = None,
        split: str = "train",
        *,
        maximum_samples: int | None = None,
    ):
        if get_record("petface-dog").admission is not DatasetAdmission.ADMIT_TRAIN:
            raise RuntimeError("PetFace training is blocked by the source admission registry")
        self._root = Path(root)
        self._transform = transform
        self._samples: tuple[PetFaceDogSplitSample, ...] = load_petface_dog_split(
            self._root, split, maximum_samples=maximum_samples
        )
        classes = sorted({sample.raw_identity_id for sample in self._samples})
        self._class_to_idx = {identity: index for index, identity in enumerate(classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self._samples[idx]
        payload = read_petface_dog_images(self._root, (sample.archive_member,))[
            sample.archive_member
        ]
        with Image.open(io.BytesIO(payload)) as opened:
            img = opened.convert("RGB")
        if self._transform:
            img_t = self._transform(img)
        else:
            import torchvision.transforms as T

            t = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            img_t = t(img)
        return img_t, self._class_to_idx[sample.raw_identity_id]

    @property
    def num_classes(self) -> int:
        return len(self._class_to_idx)

    @property
    def classes(self) -> list[str]:
        return list(self._class_to_idx.keys())
