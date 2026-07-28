"""Adapters from existing dataset parsers to UnifiedCanidSample."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from PIL import Image

from cvi.canid_data.types import (
    CaptureGroupKind,
    UnifiedCanidSample,
)
from cvi.identity_registry import (
    compute_identity_token,
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
    resolved = (root / relative).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    if not str(resolved).startswith(str(root.resolve())):
        raise ValueError(f"path traversal: {relative}")
    return resolved


def adapt_dogfacenet224(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """DogFaceNet 224: per-identity web-folder crops, no bbox/keypoint/breed."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "after_4_bis"
    if not base.is_dir():
        raise FileNotFoundError(f"DogFaceNet base not found: {base}")
    result: list[UnifiedCanidSample] = []
    for identity_dir in sorted(base.iterdir(), key=lambda p: p.name):
        if not identity_dir.is_dir():
            continue
        folder_id = identity_dir.name
        dataset_identity = f"dogfacenet224:v1:web-folder:{folder_id}"
        reg_id = compute_registered_dog_id(dataset_identity)
        for image_file in sorted(identity_dir.iterdir()):
            if not image_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                continue
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
                    split_role="UNASSIGNED",
                )
            )
    return tuple(result)


def adapt_mpdd(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """MPDD: camera-session captures with explicit split roles."""

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
            if not image_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                continue
            stem = image_file.stem
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            identity_str = parts[0]
            camera_str = parts[1] if parts[1].startswith("c") else ""
            seq_str = ""
            for part in parts[2:]:
                if part.startswith("s"):
                    seq_str = part
                    break
            dataset_identity = f"mpdd:v1:device-capture:{identity_str}"
            reg_id = compute_registered_dog_id(dataset_identity)
            sample_id = compute_sample_token(f"mpdd:{stem}:{split_role}")
            width, height = _image_dims(image_file)
            sha = _file_sha256(image_file)
            relative = str(image_file.relative_to(root))
            capture_id = (
                f"{identity_str}{chr(0)}{camera_str}{chr(0)}{split_role}"
                if camera_str
                else identity_str
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
                    capture_group_kind=CaptureGroupKind.REAL_CAMERA_SESSION,
                    camera_id=camera_str or None,
                    split_role=split_role,
                )
            )
    return tuple(result)


def adapt_sibetan(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """Sibetan: camera-trapping clusters, no split, cluster GT JSON."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "Sibetan"
    if not base.is_dir():
        raise FileNotFoundError(f"Sibetan base not found: {base}")
    result: list[UnifiedCanidSample] = []
    for cluster_dir in sorted(base.iterdir()):
        if not cluster_dir.is_dir() or not cluster_dir.name.isdigit():
            continue
        cluster_id = cluster_dir.name
        dataset_identity = f"sibetan:v1:gt-json:dog_{cluster_id}"
        reg_id = compute_registered_dog_id(dataset_identity)
        for image_file in sorted(cluster_dir.iterdir()):
            if not image_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                continue
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
                    dataset_version="official-2026-07-22",
                    source_group_id=cluster_id,
                    image_path=relative,
                    image_sha256=sha,
                    width=width,
                    height=height,
                    registered_identity_id=reg_id,
                    raw_identity_id=cluster_id,
                    capture_group_id=cluster_id,
                    capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
                    camera_id=camera_part or None,
                    split_role="UNASSIGNED",
                )
            )
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
                if not image_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    continue
                stem = image_file.stem
                frame_part = stem.split("_")[-1] if "_" in stem else "0"
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
                        dataset_version="outer-official-2026-07-22",
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


ADAPTERS = {
    "dogfacenet224": adapt_dogfacenet224,
    "mpdd": adapt_mpdd,
    "sibetan": adapt_sibetan,
    "yt-bb-dog": adapt_yt_bb_dog,
}


__all__ = [
    "ADAPTERS",
    "adapt_dogfacenet224",
    "adapt_mpdd",
    "adapt_sibetan",
    "adapt_yt_bb_dog",
]
