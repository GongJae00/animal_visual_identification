"""Adapters from existing dataset parsers to UnifiedCanidSample."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image

from data.public.public_canine_manifest import (
    DOGFACE_TEST_MD5,
    DOGFACE_TEST_SHA256,
    DOGFACE_TRAIN_MD5,
    DOGFACE_TRAIN_SHA256,
    _read_published_class_file,
)
from data.types import (
    CaptureGroupKind,
    UnifiedCanidSample,
)
from contracts.identity_ids import (
    compute_registered_dog_id,
    compute_sample_token,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_dims(path: Path) -> tuple[int, int]:
    with Image.open(path) as opened:
        return opened.size


def _verified_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or relative != path.as_posix()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe relative path: {relative!r}")
    absolute_root = Path(os.path.abspath(os.fspath(root)))
    resolved_root = absolute_root.resolve()
    candidate = absolute_root.joinpath(*path.parts)
    if candidate.is_symlink():
        raise ValueError(f"not a regular file: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"path traversal: {relative}")
    # Preserve the caller's absolute root spelling so relative paths remain
    # stable when CANINE_IDENTITY_DATA_DIR itself is a symlink to protected storage.
    return candidate


def _verified_regular_file(path: Path, subject: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.is_symlink() or not absolute.is_file():
        raise ValueError(f"{subject} must be a regular file: {absolute}")
    return absolute


_PETFACE_SPLIT_MEMBERS = {
    "train": "PetFace/split/dog/train.csv",
    "val": "PetFace/split/dog/val.txt",
    "test": "PetFace/split/dog/test.txt",
    "reidentification": "PetFace/split/dog/reidentification.csv",
}
_PETFACE_README_MEMBER = "PetFace/README.md"
_PETFACE_IMAGE_ARCHIVE = "dog-001.tar.gz"
_PETFACE_MAX_ZIP_ARCHIVES = 16
_PETFACE_MAX_METADATA_BYTES = 64 * 1024 * 1024
_PETFACE_MAX_SPLIT_ROWS = 500_000
_PETFACE_MAX_TAR_MEMBERS = 2_000_000
_PETFACE_MAX_IMAGE_BYTES = 16 * 1024 * 1024
_PETFACE_MAX_INTAKE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PetFaceDogSplitSample:
    """One publisher-declared PetFace dog split row."""

    archive_member: str
    raw_identity_id: str


def _read_unique_petface_zip_member(root: Path, member_name: str) -> bytes:
    zip_paths = sorted(root.glob("*.zip"), key=lambda path: path.name)
    if not zip_paths:
        raise FileNotFoundError(f"PetFace metadata archives not found: {root}")
    if len(zip_paths) > _PETFACE_MAX_ZIP_ARCHIVES:
        raise ValueError("PetFace metadata archive count exceeds intake limit")
    matches: list[tuple[str, bytes]] = []
    for unverified_archive in zip_paths:
        archive_path = _verified_path(root, unverified_archive.name)
        with zipfile.ZipFile(archive_path) as archive:
            infos = [
                info for info in archive.infolist() if info.filename == member_name
            ]
            if len(infos) > 1:
                raise ValueError(
                    f"PetFace metadata member is duplicated in {archive_path.name}"
                )
            if not infos:
                continue
            info = infos[0]
            mode = info.external_attr >> 16
            if (
                info.is_dir()
                or stat.S_ISLNK(mode)
                or info.flag_bits & 0x1
                or info.file_size > _PETFACE_MAX_METADATA_BYTES
            ):
                raise ValueError(f"unsafe PetFace metadata member: {member_name}")
            payload = archive.read(info)
            if len(payload) != info.file_size:
                raise ValueError(f"PetFace metadata member size differs: {member_name}")
            matches.append((archive_path.name, payload))
    if not matches:
        raise FileNotFoundError(f"PetFace metadata member not found: {member_name}")
    if len(matches) != 1:
        archives = ", ".join(name for name, _ in matches)
        raise ValueError(
            f"PetFace metadata member is ambiguous across archives: {archives}"
        )
    return matches[0][1]


def _petface_member_identity(relative: str) -> str:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or relative != path.as_posix()
        or "\\" in relative
        or len(path.parts) != 3
        or path.parts[0] != "dog"
        or not path.parts[1].isdigit()
        or path.suffix.lower() != ".png"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe PetFace dog image member: {relative!r}")
    return path.parts[1]


def load_petface_dog_split(
    data_root: Path,
    split: str,
    *,
    maximum_samples: int | None = None,
) -> tuple[PetFaceDogSplitSample, ...]:
    """Read a bounded official dog split directly from PetFace metadata archives."""

    if split not in _PETFACE_SPLIT_MEMBERS:
        raise ValueError(f"unsupported PetFace dog split: {split!r}")
    if maximum_samples is not None and (
        isinstance(maximum_samples, bool)
        or not isinstance(maximum_samples, int)
        or maximum_samples <= 0
    ):
        raise ValueError("maximum_samples must be a positive integer or None")
    root = Path(os.path.abspath(os.fspath(data_root)))
    if not root.is_dir():
        raise FileNotFoundError(f"PetFace archive root not found: {root}")
    readme = _read_unique_petface_zip_member(root, _PETFACE_README_MEMBER).decode(
        "utf-8", errors="strict"
    )
    required_terms = (
        "provided for research purposes only",
        "prohibited from redistributing or sharing this dataset",
    )
    if any(term not in readme.lower() for term in required_terms):
        raise ValueError("PetFace local license metadata differs")

    member_name = _PETFACE_SPLIT_MEMBERS[split]
    text = _read_unique_petface_zip_member(root, member_name).decode(
        "utf-8-sig", errors="strict"
    )
    if "\x00" in text:
        raise ValueError(f"PetFace split contains NUL bytes: {member_name}")
    rows: list[PetFaceDogSplitSample] = []
    seen_members: set[str] = set()
    header_seen = False
    for line_number, raw_row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if not raw_row or all(not value.strip() for value in raw_row):
            continue
        expected_columns = 2 if member_name.endswith(".csv") else 1
        if expected_columns == 2 and not header_seen:
            if raw_row != ["filename", "label"]:
                raise ValueError(f"PetFace split header differs: {member_name}")
            header_seen = True
            continue
        if len(raw_row) != expected_columns:
            raise ValueError(
                f"PetFace split row {line_number} has {len(raw_row)} columns"
            )
        relative = raw_row[0].strip()
        identity = _petface_member_identity(relative)
        if expected_columns == 2:
            label = raw_row[1].strip()
            if not label.isdigit() or int(label) != int(identity):
                raise ValueError(
                    f"PetFace split identity differs at row {line_number}"
                )
        if relative in seen_members:
            raise ValueError(f"PetFace split repeats image member: {relative}")
        seen_members.add(relative)
        rows.append(PetFaceDogSplitSample(relative, identity))
        if len(rows) > _PETFACE_MAX_SPLIT_ROWS:
            raise ValueError("PetFace split row count exceeds intake limit")
    if not rows:
        raise ValueError(f"PetFace split is empty: {member_name}")
    if maximum_samples is not None:
        rows = rows[:maximum_samples]
    return tuple(rows)


def read_petface_dog_images(
    data_root: Path,
    members: tuple[str, ...],
) -> dict[str, bytes]:
    """Read selected regular image members without extracting the PetFace tar archive."""

    if not members or len(set(members)) != len(members):
        raise ValueError("PetFace image request must be non-empty and unique")
    for member_name in members:
        _petface_member_identity(member_name)
    root = Path(os.path.abspath(os.fspath(data_root)))
    archive_path = _verified_path(root, _PETFACE_IMAGE_ARCHIVE)
    requested = set(members)
    result: dict[str, bytes] = {}
    total_bytes = 0
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member_index, member in enumerate(archive, start=1):
            if member_index > _PETFACE_MAX_TAR_MEMBERS:
                raise ValueError("PetFace image archive member count exceeds intake limit")
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or member.name != path.as_posix()
                or "\\" in member.name
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError(f"unsafe PetFace tar member: {member.name!r}")
            if member.name not in requested:
                continue
            if member.name in result:
                raise ValueError(f"PetFace image member is duplicated: {member.name}")
            if not member.isfile() or member.size > _PETFACE_MAX_IMAGE_BYTES:
                raise ValueError(f"unsafe PetFace image member: {member.name}")
            total_bytes += member.size
            if total_bytes > _PETFACE_MAX_INTAKE_BYTES:
                raise ValueError("PetFace selected image bytes exceed intake limit")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"PetFace image member cannot be read: {member.name}")
            payload = stream.read(_PETFACE_MAX_IMAGE_BYTES + 1)
            if len(payload) != member.size:
                raise ValueError(f"PetFace image member size differs: {member.name}")
            result[member.name] = payload
    missing = requested - set(result)
    if missing:
        raise FileNotFoundError(
            f"PetFace split image missing from dog archive: {min(missing)}"
        )
    return result


def adapt_petface_dog(
    data_root: Path,
    *,
    split_role: str,
    maximum_samples: int,
) -> tuple[UnifiedCanidSample, ...]:
    """Bounded research intake for PetFace dog archives; not an admitted adapter."""

    rows = load_petface_dog_split(
        data_root, split_role, maximum_samples=maximum_samples
    )
    images = read_petface_dog_images(
        data_root, tuple(row.archive_member for row in rows)
    )
    split_member = _PETFACE_SPLIT_MEMBERS[split_role]
    samples: list[UnifiedCanidSample] = []
    for row in rows:
        payload = images[row.archive_member]
        with Image.open(io.BytesIO(payload)) as opened:
            width, height = opened.size
            opened.verify()
        identity = row.raw_identity_id
        samples.append(
            UnifiedCanidSample(
                sample_id=compute_sample_token(
                    f"petface-dog:official:{split_role}:{row.archive_member}"
                ),
                dataset_name="petface-dog",
                dataset_version="eccv-2024-local-archive-intake-v1",
                source_group_id=identity,
                image_path=f"{_PETFACE_IMAGE_ARCHIVE}::{row.archive_member}",
                image_sha256=hashlib.sha256(payload).hexdigest(),
                width=width,
                height=height,
                registered_identity_id=compute_registered_dog_id(
                    f"petface-dog:v1:official-folder:{identity}"
                ),
                raw_identity_id=identity,
                capture_group_id=identity,
                capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
                split_role=split_role,
                metadata={
                    "archive_member": row.archive_member,
                    "image_archive": _PETFACE_IMAGE_ARCHIVE,
                    "official_split_member": split_member,
                    "source_intake_only": True,
                },
            )
        )
    return tuple(samples)


def adapt_dogfacenet224(
    data_root: Path,
    *,
    classes_train_path: Path | None = None,
    classes_test_path: Path | None = None,
) -> tuple[UnifiedCanidSample, ...]:
    """DogFaceNet 224 web-folder crops with the authenticated publisher split."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "after_4_bis"
    if not base.is_dir():
        raise FileNotFoundError(f"DogFaceNet base not found: {base}")
    if (classes_train_path is None) != (classes_test_path is None):
        raise ValueError("DogFace train and test class paths must be provided together")
    if classes_train_path is None:
        classes_train_path = _verified_path(root, "classes_train.txt")
        classes_test_path = _verified_path(root, "classes_test.txt")
    else:
        classes_train_path = _verified_regular_file(
            classes_train_path, "DogFace train class file"
        )
        classes_test_path = _verified_regular_file(
            classes_test_path, "DogFace test class file"
        )
    train_values, _, _ = _read_published_class_file(
        classes_train_path,
        expected_sha256=DOGFACE_TRAIN_SHA256,
        expected_md5=DOGFACE_TRAIN_MD5,
    )
    test_values, _, _ = _read_published_class_file(
        classes_test_path,
        expected_sha256=DOGFACE_TEST_SHA256,
        expected_md5=DOGFACE_TEST_MD5,
    )
    train_counts = Counter(train_values)
    test_counts = Counter(test_values)
    if set(train_counts) & set(test_counts):
        raise ValueError("DogFace class-file identity partition differs")

    images_by_identity: list[tuple[str, tuple[Path, ...]]] = []
    observed_counts: Counter[int] = Counter()
    for identity_dir in sorted(base.iterdir(), key=lambda path: path.name):
        if not identity_dir.is_dir():
            continue
        folder_id = identity_dir.name
        if not folder_id.isdigit():
            raise ValueError(f"DogFaceNet identity folder differs: {identity_dir}")
        image_files = tuple(
            image_file
            for image_file in sorted(identity_dir.iterdir())
            if image_file.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if image_files:
            images_by_identity.append((folder_id, image_files))
            observed_counts[int(folder_id)] = len(image_files)
    if set(train_counts) | set(test_counts) != set(observed_counts):
        raise ValueError("DogFace class-file identity partition differs")
    if train_counts + test_counts != observed_counts:
        raise ValueError("DogFace class-file multiplicities differ from extracted images")

    split_by_identity = {
        **{identity: "train" for identity in train_counts},
        **{identity: "test" for identity in test_counts},
    }
    result: list[UnifiedCanidSample] = []
    for folder_id, image_files in images_by_identity:
        dataset_identity = f"dogfacenet224:v1:web-folder:{folder_id}"
        reg_id = compute_registered_dog_id(dataset_identity)
        for image_file in image_files:
            image_file = _verified_path(root, image_file.relative_to(root).as_posix())
            sample_id = compute_sample_token(
                f"dogfacenet224:{folder_id}:{image_file.stem}"
            )
            width, height = _image_dims(image_file)
            sha = _file_sha256(image_file)
            relative = str(image_file.relative_to(root))
            result.append(
                UnifiedCanidSample(
                    sample_id=sample_id,
                    dataset_name="dogfacenet224",
                    dataset_version="zenodo-12578449-v1",
                    source_group_id=folder_id,
                    image_path=relative,
                    image_sha256=sha,
                    width=width,
                    height=height,
                    registered_identity_id=reg_id,
                    raw_identity_id=folder_id,
                    capture_group_id=folder_id,
                    capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
                    split_role=split_by_identity[int(folder_id)],
                )
            )
    return tuple(result)


def adapt_mpdd(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """MPDD: pose captures with explicit splits and unverified filename tokens."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "MPDD" / "pytorch"
    if not base.is_dir():
        raise FileNotFoundError(f"MPDD base not found: {base}")
    result: list[UnifiedCanidSample] = []
    for split_role in ("train", "val", "query", "gallery"):
        split_dir = base / split_role
        if not split_dir.is_dir():
            continue
        for image_file in sorted(split_dir.iterdir()):
            if image_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            image_file = _verified_path(root, image_file.relative_to(root).as_posix())
            stem = image_file.stem
            match = re.fullmatch(
                r"(?P<identity>\d+)_c(?P<camera>\d+)_?s(?P<sequence>\d+)_(?P<frame>\d+)",
                stem,
            )
            if match is None:
                raise ValueError(f"MPDD image filename schema differs: {image_file.name}")
            identity_str = match.group("identity")
            camera_str = f"c{match.group('camera')}"
            sequence_str = f"s{match.group('sequence')}"
            dataset_identity = f"mpdd:v1:device-capture:{identity_str}"
            reg_id = compute_registered_dog_id(dataset_identity)
            sample_id = compute_sample_token(f"mpdd:{stem}:{split_role}")
            width, height = _image_dims(image_file)
            sha = _file_sha256(image_file)
            relative = str(image_file.relative_to(root))
            capture_id = (
                f"{identity_str}{chr(0)}{camera_str}{chr(0)}{sequence_str}"
            )
            result.append(
                UnifiedCanidSample(
                    sample_id=sample_id,
                    dataset_name="mpdd",
                    dataset_version="mendeley-v5j6m8dzhv-v1",
                    source_group_id=identity_str,
                    image_path=relative,
                    image_sha256=sha,
                    width=width,
                    height=height,
                    registered_identity_id=reg_id,
                    raw_identity_id=identity_str,
                    capture_group_id=capture_id,
                    capture_group_kind=CaptureGroupKind.POSE_VIEW_CLUSTER,
                    split_role=split_role,
                    metadata={
                        "unverified_camera_token": camera_str,
                        "unverified_sequence_token": sequence_str,
                    },
                )
            )
    return tuple(result)


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


def adapt_sibetan(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """Sibetan: camera-trap sequences joined to publisher GT dog identities."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "Sibetan"
    if not base.is_dir():
        raise FileNotFoundError(f"Sibetan base not found: {base}")
    gt_path = base / "gt_sibetan.json"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Sibetan identity GT not found: {gt_path}")
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise ValueError("Sibetan identity GT must be a non-empty object")
    identity_by_cluster: dict[str, str] = {}
    for dog_identity, raw_clusters in ground_truth.items():
        if (
            not isinstance(dog_identity, str)
            or not dog_identity.isdigit()
            or not isinstance(raw_clusters, list)
            or not raw_clusters
        ):
            raise ValueError("Sibetan identity GT schema differs")
        for cluster in raw_clusters:
            if isinstance(cluster, bool) or not isinstance(cluster, int) or cluster < 0:
                raise ValueError("Sibetan identity GT cluster differs")
            cluster_id = str(cluster)
            if cluster_id in identity_by_cluster:
                raise ValueError("Sibetan identity GT repeats a sequence cluster")
            identity_by_cluster[cluster_id] = dog_identity
    result: list[UnifiedCanidSample] = []
    observed_clusters: set[str] = set()
    for cluster_dir in sorted(base.iterdir()):
        if not cluster_dir.is_dir() or not cluster_dir.name.isdigit():
            continue
        cluster_id = cluster_dir.name
        dog_identity = identity_by_cluster.get(cluster_id)
        if dog_identity is None:
            raise ValueError("Sibetan image cluster is absent from identity GT")
        observed_clusters.add(cluster_id)
        dataset_identity = f"sibetan:v1:gt-json:{dog_identity}"
        reg_id = compute_registered_dog_id(dataset_identity)
        for image_file in sorted(cluster_dir.iterdir()):
            if image_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            image_file = _verified_path(root, image_file.relative_to(root).as_posix())
            stem = image_file.stem
            parts = stem.split("_")
            camera_part = next((p for p in parts if p.startswith("C")), "")
            sample_id = compute_sample_token(f"sibetan:{cluster_id}:{stem}")
            width, height = _image_dims(image_file)
            sha = _file_sha256(image_file)
            relative = str(image_file.relative_to(root))
            result.append(
                UnifiedCanidSample(
                    sample_id=sample_id,
                    dataset_name="sibetan",
                    dataset_version="publisher-v1-2025-10-27",
                    source_group_id=cluster_id,
                    image_path=relative,
                    image_sha256=sha,
                    width=width,
                    height=height,
                    registered_identity_id=reg_id,
                    raw_identity_id=dog_identity,
                    capture_group_id=cluster_id,
                    capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
                    split_role="UNASSIGNED",
                    metadata={"unverified_camera_token": camera_part or None},
                )
            )
    if observed_clusters != set(identity_by_cluster):
        raise ValueError("Sibetan identity GT and sequence directories differ")
    return tuple(result)


def adapt_yt_bb_dog(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """YT-BB-Dog: video-track identities with train/test split."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "YT-BB-dog" / "YT-BB-Dog"
    if not base.is_dir():
        raise FileNotFoundError(f"YT-BB-Dog base not found: {base}")
    result: list[UnifiedCanidSample] = []
    for split_role in ("train", "test"):
        split_dir = base / split_role
        if not split_dir.is_dir():
            continue
        for identity_dir in sorted(split_dir.iterdir(), key=lambda p: p.name):
            if not identity_dir.is_dir():
                continue
            identity_str = identity_dir.name
            dataset_identity = f"yt-bb-dog:v1:video-track:{identity_str}"
            reg_id = compute_registered_dog_id(dataset_identity)
            for image_file in sorted(identity_dir.iterdir()):
                if image_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                image_file = _verified_path(
                    root, image_file.relative_to(root).as_posix()
                )
                stem = image_file.stem
                sample_id = compute_sample_token(
                    f"yt-bb-dog:{identity_str}:{split_role}:{stem}"
                )
                width, height = _image_dims(image_file)
                sha = _file_sha256(image_file)
                relative = str(image_file.relative_to(root))
                result.append(
                    UnifiedCanidSample(
                        sample_id=sample_id,
                        dataset_name="yt-bb-dog",
                        dataset_version="publisher-v1-2025-10-27",
                        source_group_id=identity_str,
                        image_path=relative,
                        image_sha256=sha,
                        width=width,
                        height=height,
                        registered_identity_id=reg_id,
                        raw_identity_id=identity_str,
                        capture_group_id=identity_str,
                        capture_group_kind=CaptureGroupKind.VIDEO_TRACK,
                        split_role=split_role,
                    )
                )
    return tuple(result)


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


ADAPTERS = {
    "ap10k-dog": adapt_ap10k_dog,
    "dogflw": adapt_dogflw,
    "dogfacenet224": adapt_dogfacenet224,
    "mpdd": adapt_mpdd,
    "oxford-pets-dog": adapt_oxford_pets_dog,
    "sibetan": adapt_sibetan,
    "yt-bb-dog": adapt_yt_bb_dog,
}

# PetFace is intentionally separate until source provenance is admitted.
RESEARCH_INTAKE_ADAPTERS = {"petface-dog": adapt_petface_dog}


__all__ = [
    "ADAPTERS",
    "RESEARCH_INTAKE_ADAPTERS",
    "PetFaceDogSplitSample",
    "adapt_ap10k_dog",
    "adapt_dogfacenet224",
    "adapt_dogflw",
    "adapt_mpdd",
    "adapt_oxford_pets_dog",
    "adapt_petface_dog",
    "adapt_sibetan",
    "adapt_yt_bb_dog",
    "load_petface_dog_split",
    "read_petface_dog_images",
]
