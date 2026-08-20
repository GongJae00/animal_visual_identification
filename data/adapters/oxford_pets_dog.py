"""Oxford-IIIT Pet dog subset layout adapter.

Publisher tree on disk (do not reorganize):

    /mnt/r/Dataset/Animals_Dataset/oxford-iiit-pet/
      images/                         {stem}.jpg; leftover {stem}.mat unused
      annotations/
        README
        list.txt                      combined catalog; not a split; not identity
        trainval.txt                  official paper split (adapter reads)
        test.txt                      official paper split (adapter reads)
        trimaps/                      {stem}.png; AppleDouble ._ files unused
        xmls/                         optional PASCAL VOC {stem}.xml head ROI

Row schema for trainval.txt, test.txt, and non-comment list.txt rows:

    Image CLASS-ID SPECIES BREED_ID

SPECIES 1=cat, 2=dog. CLASS-ID is the 1..37 breed category. Filenames with a
capital first letter are cats; lowercase first letter are dogs. Breed is the
filename stem before the trailing _<index>. Observed: 12 cat breed IDs, 25 dog
breed IDs, 4978 dog rows across trainval (2492) and test (2486), disjoint stems.

Identity unit: none. Breed, class_id, species_id, breed_id, filename stem,
trimap, and head ROI are not per-animal identities. Parsing evaluation only.

Adapter keeps SPECIES==2 rows from trainval then test, requires the jpg and
trimap, and reads xmls/{stem}.xml only when that path exists.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from data.adapters.io import (
    _file_sha256,
    _image_dims,
    _verified_path,
)
from data.types import CaptureGroupKind, UnifiedCanidSample
from shared.contracts.identity_ids import compute_sample_token


def _oxford_head_annotation(
    xml_path: Path,
    image_name: str,
    width: int,
    height: int,
) -> tuple[tuple[float, float, float, float], dict[str, str]]:
    if xml_path.stat().st_size > 1024 * 1024:
        raise ValueError(f"Oxford head annotation exceeds size limit: {xml_path}")
    payload = xml_path.read_bytes()
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError(f"unsafe Oxford head annotation: {xml_path}")
    try:
        annotation = ET.fromstring(payload)
    except ET.ParseError as error:
        raise ValueError(f"Oxford head annotation is malformed: {xml_path}") from error
    if annotation.findtext("filename") != image_name:
        raise ValueError(f"Oxford head annotation filename differs: {xml_path}")
    objects = annotation.findall("object")
    if len(objects) != 1 or objects[0].findtext("name") != "dog":
        raise ValueError(f"Oxford dog head annotation schema differs: {xml_path}")
    box = objects[0].find("bndbox")
    if box is None:
        raise ValueError(f"Oxford dog head box is missing: {xml_path}")
    try:
        coordinates = tuple(
            float(box.findtext(name, ""))
            for name in ("xmin", "ymin", "xmax", "ymax")
        )
    except ValueError as error:
        raise ValueError(f"Oxford dog head box is malformed: {xml_path}") from error
    x_min, y_min, x_max, y_max = coordinates
    if not (
        all(math.isfinite(value) for value in coordinates)
        and 0 <= x_min < x_max <= width
        and 0 <= y_min < y_max <= height
    ):
        raise ValueError(f"Oxford dog head box is out of bounds: {xml_path}")
    metadata = {
        "head_pose": objects[0].findtext("pose", ""),
        "head_truncated": objects[0].findtext("truncated", ""),
        "head_occluded": objects[0].findtext("occluded", ""),
        "head_difficult": objects[0].findtext("difficult", ""),
    }
    return (x_min, y_min, x_max, y_max), metadata


def adapt_oxford_pets_dog(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """Oxford-IIIT Pet dog subset with official splits, trimaps, and head ROIs."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    image_root = root / "images"
    annotation_root = root / "annotations"
    if not image_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(f"Oxford-IIIT Pet base not found: {root}")
    samples: list[UnifiedCanidSample] = []
    seen_images: set[str] = set()
    for split_role in ("trainval", "test"):
        split_path = annotation_root / f"{split_role}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Oxford official split not found: {split_path}")
        for line_number, line in enumerate(
            split_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(
                    f"Oxford split row schema differs: {split_path}:{line_number}"
                )
            stem, class_id, species_id, breed_id = fields
            if (
                not class_id.isdigit()
                or species_id not in {"1", "2"}
                or not breed_id.isdigit()
            ):
                raise ValueError(
                    f"Oxford split labels differ: {split_path}:{line_number}"
                )
            if stem in seen_images:
                raise ValueError(f"Oxford official splits repeat image: {stem}")
            seen_images.add(stem)
            if species_id != "2":
                continue
            name_parts = stem.rsplit("_", 1)
            if len(name_parts) != 2 or not name_parts[1].isdigit():
                raise ValueError(f"Oxford dog image name differs: {stem}")
            image_path = _verified_path(root, f"images/{stem}.jpg")
            trimap_path = _verified_path(root, f"annotations/trimaps/{stem}.png")
            width, height = _image_dims(image_path)
            xml_relative = f"annotations/xmls/{stem}.xml"
            xml_candidate = root / xml_relative
            head_roi = None
            head_metadata: dict[str, str] = {}
            if xml_candidate.is_file() or xml_candidate.is_symlink():
                xml_path = _verified_path(root, xml_relative)
                head_roi, head_metadata = _oxford_head_annotation(
                    xml_path, image_path.name, width, height
                )
            samples.append(
                UnifiedCanidSample(
                    sample_id=compute_sample_token(
                        f"oxford-pets-dog:official:{split_role}:{stem}"
                    ),
                    dataset_name="oxford-pets-dog",
                    dataset_version="publisher-splits-v1",
                    source_group_id=stem,
                    image_path=str(image_path.relative_to(root)),
                    image_sha256=_file_sha256(image_path),
                    width=width,
                    height=height,
                    breed=name_parts[0],
                    head_roi_xyxy=head_roi,
                    foreground_mask_path=str(trimap_path.relative_to(root)),
                    capture_group_kind=CaptureGroupKind.UNKNOWN,
                    split_role=split_role,
                    metadata={
                        "class_id": int(class_id),
                        "species_id": int(species_id),
                        "breed_id": int(breed_id),
                        "trimap_values": {
                            "foreground": 1,
                            "background": 2,
                            "not_classified": 3,
                        },
                        **head_metadata,
                    },
                )
            )
    if not samples:
        raise ValueError("Oxford official splits contain no dog images")
    return tuple(samples)
