from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler


class PetFaceDataset(Dataset):
    def __init__(self, root: Path, transform: Any | None = None,
                 split: str = "train"):
        self._root = root
        self._transform = transform
        self._samples: list[tuple[Path, int, str]] = []
        self._class_to_idx: dict[str, int] = {}
        self._load_split(split)

    def _load_split(self, split: str) -> None:
        split_file = self._root / f"{split}.txt"
        if split_file.exists():
            for line in split_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 3:
                    rel_path, klass, family = parts[0], parts[1], parts[2]
                    if klass not in self._class_to_idx:
                        self._class_to_idx[klass] = len(self._class_to_idx)
                    full_path = self._root / rel_path
                    if full_path.exists():
                        self._samples.append((full_path, self._class_to_idx[klass], family))
        else:
            for img_path in sorted(self._root.rglob("*.jpg")):
                klass = img_path.parent.name
                if klass not in self._class_to_idx:
                    self._class_to_idx[klass] = len(self._class_to_idx)
                self._samples.append((img_path, self._class_to_idx[klass], "canine"))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label, _ = self._samples[idx]
        img = Image.open(path).convert("RGB")
        if self._transform:
            img_t = self._transform(img)
        else:
            import torchvision.transforms as T
            t = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_t = t(img)
        return img_t, label

    @property
    def num_classes(self) -> int:
        return len(self._class_to_idx)

    @property
    def classes(self) -> list[str]:
        return list(self._class_to_idx.keys())
