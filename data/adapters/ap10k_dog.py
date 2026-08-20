"""AP-10K domestic-dog layout adapter.

Publisher tree under the dataset root (parent of ``ap-10k/``, not ``ap-10k``
itself):

    ap-10k.zip
    ap-10k/data/{image_id:012d}.jpg
    ap-10k/annotations/ap10k-{train,val,test}-split{1,2,3}.json

``ap-10k.zip`` is a sibling archive and is not read. Images are a flat JPEG
directory; there are no identity, breed, or species folders. Annotation JSON
is COCO with 54 species. Official split1 files
(``ap10k-{train,val,test}-split1.json``) are the only splits this adapter
opens. split2 and split3 remain on disk and are unused.

There is no identity unit. ``category_id``, ``image_id``, ``annotation_id``,
``file_name``, ``background``, supercategory, and split role are not
identities. Only category 8 (``dog``) is kept; other Canidae (arctic fox 7,
fox 9, wolf 10) and the remaining species are dropped. ``source_group_id`` is
the image (``ap10k-image:{image_id}``), so multiple dog instances on one JPEG
share a source group.

Each kept instance carries the COCO bbox and 17 body keypoints. Publisher
JSON names the third and fifth keypoints ``nose`` and ``root_of_tail``;
samples store them as ``nose_center`` and ``tail_base``.
"""

from __future__ import annotations

import json
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

_AP10K_KEYPOINTS = (
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


def adapt_ap10k_dog(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """AP-10K split-1 domestic-dog instances with bbox and 17 keypoints."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "ap-10k"
    annotation_root = base / "annotations"
    image_root = base / "data"
    if not annotation_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError(f"AP-10K base not found: {base}")
    samples: list[UnifiedCanidSample] = []
    image_cache: dict[int, tuple[Path, int, int, str]] = {}
    for split_role in ("train", "val", "test"):
        payload = json.loads(
            (annotation_root / f"ap10k-{split_role}-split1.json").read_text(
                encoding="utf-8"
            )
        )
        images = {int(image["id"]): image for image in payload["images"]}
        for annotation in sorted(
            payload["annotations"], key=lambda row: int(row["id"])
        ):
            if int(annotation["category_id"]) != 8:
                continue
            image_id = int(annotation["image_id"])
            image_info = images[image_id]
            if image_id not in image_cache:
                image_path = _verified_path(
                    root,
                    f"ap-10k/data/{image_info['file_name']}",
                )
                width, height = _image_dims(image_path)
                image_cache[image_id] = (
                    image_path,
                    width,
                    height,
                    _file_sha256(image_path),
                )
            image_path, width, height, image_sha256 = image_cache[image_id]
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            raw_keypoints = annotation["keypoints"]
            body_keypoints = {
                name: (
                    float(raw_keypoints[index * 3]),
                    float(raw_keypoints[index * 3 + 1]),
                    float(raw_keypoints[index * 3 + 2]) / 2.0,
                )
                for index, name in enumerate(_AP10K_KEYPOINTS)
                if int(raw_keypoints[index * 3 + 2]) > 0
            }
            samples.append(
                UnifiedCanidSample(
                    sample_id=compute_sample_token(
                        f"ap10k:split1:{split_role}:{annotation['id']}"
                    ),
                    dataset_name="ap10k-dog",
                    dataset_version="official-split1-2021-11-01",
                    source_group_id=f"ap10k-image:{image_id}",
                    image_path=str(image_path.relative_to(root)),
                    image_sha256=image_sha256,
                    width=width,
                    height=height,
                    dog_boxes_xyxy=(x, y, x + box_width, y + box_height),
                    body_keypoints=body_keypoints,
                    capture_group_kind=CaptureGroupKind.UNKNOWN,
                    split_role=split_role,
                    metadata={
                        "annotation_id": int(annotation["id"]),
                        "image_id": image_id,
                    },
                )
            )
    return tuple(samples)


__all__ = ["adapt_ap10k_dog"]
