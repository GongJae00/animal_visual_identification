"""DogFLW face-crop adapter (canonical name: dogflw).

Publisher tree observed at /mnt/r/Dataset/Animals_Dataset/dogflw:

    dogflw.zip
    DogFLW/Ear Types.docx
    DogFLW/train/{images,labels}/
    DogFLW/test/{images,labels}/

Images are PNG face crops named {ImageNet-synset}_{id}.png. Labels are
same-stem JSON objects with 46 xy landmarks and a 4-value bounding_boxes
xyxy list. Publisher stores those box values as strings; some rows are blank.

Identity unit: none. This dataset has no dog identity. ImageNet synset
prefixes, filename stems, train/test folders, and Ear Types.docx are not
identities. Publisher train/test folders are the split. CC-BY-NC.
Used by parsing regions.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from data.adapters.io import (
    _file_sha256,
    _image_dims,
    _verified_path,
)
from data.types import (
    CaptureGroupKind,
    UnifiedCanidSample,
)
from shared.contracts.identity_ids import compute_sample_token


def adapt_dogflw(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """DogFLW face crops with publisher train/test face46 annotations."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "DogFLW"
    if not base.is_dir():
        raise FileNotFoundError(f"DogFLW base not found: {base}")
    samples: list[UnifiedCanidSample] = []
    for split_role in ("train", "test"):
        image_dir = base / split_role / "images"
        label_dir = base / split_role / "labels"
        for image_path in sorted(image_dir.glob("*.png"), key=lambda path: path.name):
            image_path = _verified_path(root, image_path.relative_to(root).as_posix())
            label_path = label_dir / f"{image_path.stem}.json"
            if not label_path.is_file():
                raise FileNotFoundError(f"DogFLW label missing: {label_path}")
            annotation = json.loads(label_path.read_text(encoding="utf-8"))
            landmarks = annotation.get("landmarks")
            raw_box = annotation.get("bounding_boxes")
            if not isinstance(landmarks, list) or len(landmarks) != 46:
                raise ValueError(f"DogFLW face46 label differs: {label_path}")
            if not isinstance(raw_box, list) or len(raw_box) != 4:
                raise ValueError(f"DogFLW face bbox differs: {label_path}")
            face_landmarks = {}
            for index, point in enumerate(landmarks):
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError(f"DogFLW landmark point differs: {label_path}")
                x, y = float(point[0]), float(point[1])
                if math.isfinite(x) and math.isfinite(y):
                    face_landmarks[f"face46.{index}"] = (x, y, 1.0)
            face_box = None
            if all(value != "" and math.isfinite(float(value)) for value in raw_box):
                face_box = tuple(float(value) for value in raw_box)
            width, height = _image_dims(image_path)
            samples.append(
                UnifiedCanidSample(
                    sample_id=compute_sample_token(
                        f"dogflw:{split_role}:{image_path.stem}"
                    ),
                    dataset_name="dogflw",
                    dataset_version="kaggle-2025-07-02",
                    source_group_id=image_path.stem,
                    image_path=str(image_path.relative_to(root)),
                    image_sha256=_file_sha256(image_path),
                    width=width,
                    height=height,
                    breed=image_path.stem.split("_", 1)[0],
                    face_box_xyxy=face_box,
                    face_landmarks=face_landmarks,
                    capture_group_id=image_path.stem,
                    capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
                    split_role=split_role,
                )
            )
    return tuple(samples)
