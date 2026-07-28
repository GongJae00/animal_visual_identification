"""Adapters from existing dataset parsers to UnifiedCanidSample."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

from PIL import Image

from cvi.canid_data.types import (
    CaptureGroupKind,
    UnifiedCanidSample,
)
from cvi.identity_registry import (
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
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*path.parts)
    if candidate.is_symlink():
        raise ValueError(f"not a regular file: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")
    if not resolved.is_relative_to(resolved_root):
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
            if image_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
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
            if image_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            image_file = _verified_path(root, image_file.relative_to(root).as_posix())
            stem = image_file.stem
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            identity_str = parts[0]
            camera_str = parts[1] if parts[1].startswith("c") else ""
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
    "sibetan": adapt_sibetan,
    "yt-bb-dog": adapt_yt_bb_dog,
}


__all__ = [
    "ADAPTERS",
    "adapt_dogfacenet224",
    "adapt_ap10k_dog",
    "adapt_dogflw",
    "adapt_mpdd",
    "adapt_sibetan",
    "adapt_yt_bb_dog",
]
