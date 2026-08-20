"""PetFace dog subset layout adapter.

Registry data_root is ``petface``. On this host that path is the same inode as
publisher ``PetFace`` (DrvFS case fold); do not rename the disk.

Publisher folders:
- ``README.md``: research-only, no-redistribution terms.
- ``dog/<zero-padded-id>/<nn>.png``: identity unit. The folder name is the
  publisher dog id; each PNG is a face crop of that dog.
- ``split/dog/train.csv``: ``filename,label``. Label must equal the folder id
  as an integer. ``val.txt`` and ``test.txt``: filename only.
  ``reidentification.csv``: ``filename,label``. ``verification.csv``: image
  pairs, not an identity and not a supported split.
- ``archives/``: Drive zip shards ``PetFace-*-003.zip`` … ``-006.zip`` (members
  under ``PetFace/``), ``dog-001.tar.gz`` (members ``dog/<id>/<nn>.png``),
  ``cat-002.tar.gz``, and other-species tars under ``archives/images/``. Dog
  images may remain archived.
- ``cat/``, ``images/<species>/``, ``annotations/*.csv`` (Name/Breed/Gender),
  ``src_points/*.npy``, and every non-dog ``split/<species>/``: not dog
  identities.

Intake reads ``PetFace/README.md`` and ``PetFace/split/dog/...`` from ``*.zip``
files at the adapter root, and image bytes from ``dog-001.tar.gz`` at that root.
Canonical name ``petface-dog``. RESEARCH_INTAKE_ADAPTERS only; BLOCKED_ACCESS.
License README terms are required.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image

from data.adapters.io import _verified_path
from data.types import CaptureGroupKind, UnifiedCanidSample
from shared.contracts.identity_ids import (
    compute_registered_dog_id,
    compute_sample_token,
)

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


__all__ = [
    "PetFaceDogSplitSample",
    "adapt_petface_dog",
    "load_petface_dog_split",
    "read_petface_dog_images",
]
