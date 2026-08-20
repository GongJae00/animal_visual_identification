"""Nose-localizer training datasets and losses.

Identity embedding trainers live under identification/training/nose/.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence
from zipfile import ZipFile, ZipInfo

from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from parsing.export.regions.localizer import (
    AP10K_SUPPORTED_INDICES,
    DOGFLW_DERIVATION,
    INPUT_SIZE,
    KEYPOINT_ORDER,
    image_to_tensor,
)

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_IMAGE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class NoseKeypointRecord:
    dataset: str
    split: str
    archive_path: Path
    image_member: str
    sample_id: str
    crop_xyxy: tuple[float, float, float, float]
    points: tuple[tuple[float, float] | None, ...]
    supported: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.points) != len(KEYPOINT_ORDER) or len(self.supported) != len(
            KEYPOINT_ORDER
        ):
            raise ValueError("record keypoints must match KEYPOINT_ORDER")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("record split must be train, val, or test")
        x0, y0, x1, y1 = self.crop_xyxy
        if not all(math.isfinite(value) for value in self.crop_xyxy) or not (
            x0 < x1 and y0 < y1
        ):
            raise ValueError("record crop must be finite non-empty xyxy")


class ZipNoseKeypointDataset(Dataset[dict[str, Any]]):
    """Decode crops directly from immutable ZIP archives without extraction."""

    def __init__(
        self, records: Sequence[NoseKeypointRecord], *, input_size: int = INPUT_SIZE
    ) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        self.records = tuple(records)
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with ZipFile(record.archive_path) as archive:
            image_bytes = _read_member(archive, record.image_member, _MAX_IMAGE_BYTES)
        with Image.open(BytesIO(image_bytes)) as opened:
            image = opened.convert("RGB")
        try:
            crop, integer_box = _crop_image(image, record.crop_xyxy)
        except ValueError:
            if record.dataset != "dogflw":
                raise
            # Two publisher rows carry face boxes in a coordinate system that
            # does not match their released images, while all 46 landmarks are
            # finite and image-relative. Preserve their supervision by using
            # the complete released image.
            integer_box = (0, 0, image.width, image.height)
            crop = image
        resized = crop.resize(
            (self.input_size, self.input_size), Image.Resampling.BILINEAR
        )

        targets = torch.zeros((len(KEYPOINT_ORDER), 2), dtype=torch.float32)
        visibility = torch.zeros(len(KEYPOINT_ORDER), dtype=torch.bool)
        support = torch.tensor(record.supported, dtype=torch.bool)
        left, top, _, _ = integer_box
        for keypoint_index, point in enumerate(record.points):
            if point is None or not support[keypoint_index]:
                continue
            normalized = (
                (point[0] - left) / crop.width,
                (point[1] - top) / crop.height,
            )
            targets[keypoint_index] = torch.tensor(normalized, dtype=torch.float32)
            visibility[keypoint_index] = all(0.0 <= value <= 1.0 for value in normalized)
        return {
            "image": image_to_tensor(resized),
            "target": targets,
            "visibility": visibility,
            "support": support,
            "normalizer": torch.tensor(math.sqrt(2.0), dtype=torch.float32),
            "dataset": record.dataset,
            "split": record.split,
            "sample_id": record.sample_id,
        }


def partial_keypoint_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    support: torch.Tensor,
    *,
    confidence_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Masked coordinate loss plus supported-channel visibility confidence loss."""

    expected = prediction.shape[:-1]
    if prediction.ndim != 3 or prediction.shape[-2:] != (len(KEYPOINT_ORDER), 3):
        raise ValueError("prediction must have shape [B,8,3]")
    if target.shape != (*expected, 2):
        raise ValueError("target must have shape [B,8,2]")
    if visibility.shape != expected or support.shape != expected:
        raise ValueError("visibility and support must have shape [B,8]")
    visible = visibility.bool() & support.bool()
    supported = support.bool()
    if visible.any():
        coordinate = F.smooth_l1_loss(
            prediction[..., :2][visible], target[visible], beta=0.02
        )
    else:
        coordinate = prediction[..., :2].sum() * 0.0
    if supported.any():
        confidence = F.binary_cross_entropy(
            prediction[..., 2][supported], visibility.float()[supported]
        )
    else:
        confidence = prediction[..., 2].sum() * 0.0
    total = coordinate + confidence_weight * confidence
    return {"total": total, "coordinate": coordinate, "confidence": confidence}


def keypoint_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    normalizer: torch.Tensor,
    *,
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute bbox-diagonal NME and confidence-threshold coverage."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if prediction.ndim != 3 or prediction.shape[-2:] != (len(KEYPOINT_ORDER), 3):
        raise ValueError("prediction must have shape [B,8,3]")
    if target.shape != prediction[..., :2].shape or visibility.shape != prediction.shape[:2]:
        raise ValueError("metric target shapes differ")
    if normalizer.shape != (prediction.shape[0],) or torch.any(normalizer <= 0):
        raise ValueError("normalizer must have shape [B] and be positive")
    visible = visibility.bool()
    covered = visible & (prediction[..., 2] >= confidence_threshold)
    distances = torch.linalg.vector_norm(prediction[..., :2] - target, dim=-1)
    normalized = distances / normalizer[:, None]
    eligible_count = int(visible.sum().item())
    covered_count = int(covered.sum().item())
    return {
        "normalization": "ground_truth_crop_content_diagonal",
        "confidence_threshold": confidence_threshold,
        "eligible_keypoints": eligible_count,
        "covered_keypoints": covered_count,
        "coverage": covered_count / eligible_count if eligible_count else None,
        "NME": float(normalized[visible].mean().item()) if eligible_count else None,
        "covered_NME": (
            float(normalized[covered].mean().item()) if covered_count else None
        ),
    }


def parse_ap10k_zip(archive_path: Path) -> dict[str, tuple[NoseKeypointRecord, ...]]:
    """Parse AP-10K official split 1 dog instances directly from its ZIP."""

    path = _regular_zip_path(archive_path)
    result: dict[str, list[NoseKeypointRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    with ZipFile(path) as archive:
        members = _safe_member_map(archive)
        for split in result:
            annotation_member = _unique_suffix_member(
                members, f"annotations/ap10k-{split}-split1.json"
            )
            payload = _read_json(archive, members[annotation_member])
            if not isinstance(payload, dict):
                raise ValueError("AP-10K annotation root must be an object")
            images = _ap10k_images(payload)
            annotations = payload.get("annotations")
            if not isinstance(annotations, list):
                raise ValueError("AP-10K annotations must be an array")
            for annotation in sorted(annotations, key=lambda row: int(row["id"])):
                if int(annotation.get("category_id", -1)) != 8:
                    continue
                image_info = images[int(annotation["image_id"])]
                relative = _safe_relative(str(image_info["file_name"]))
                image_member = _unique_suffix_member(
                    members, f"data/{relative.as_posix()}"
                )
                bbox = annotation.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError("AP-10K bbox must contain xywh")
                x, y, width, height = (float(value) for value in bbox)
                keypoints = annotation.get("keypoints")
                if not isinstance(keypoints, list) or len(keypoints) != 51:
                    raise ValueError("AP-10K keypoints must contain 17 xyv triples")
                points: list[tuple[float, float] | None] = [None] * len(KEYPOINT_ORDER)
                for source_index, target_index in ((0, 0), (1, 1), (2, 3)):
                    if int(keypoints[source_index * 3 + 2]) > 0:
                        point = (
                            float(keypoints[source_index * 3]),
                            float(keypoints[source_index * 3 + 1]),
                        )
                        if all(math.isfinite(value) for value in point):
                            points[target_index] = point
                result[split].append(
                    NoseKeypointRecord(
                        dataset="ap10k",
                        split=split,
                        archive_path=path,
                        image_member=image_member,
                        sample_id=f"ap10k:split1:{split}:{int(annotation['id'])}",
                        crop_xyxy=(x, y, x + width, y + height),
                        points=tuple(points),
                        supported=tuple(
                            index in AP10K_SUPPORTED_INDICES
                            for index in range(len(KEYPOINT_ORDER))
                        ),
                    )
                )
    return {split: tuple(records) for split, records in result.items()}


def dogflw_points(
    landmarks: Sequence[object],
) -> tuple[tuple[float, float] | None, ...]:
    """Derive the documented eye centers and six NoseID points from face46."""

    if len(landmarks) != 46:
        raise ValueError("DogFLW landmarks must contain 46 points")
    parsed: list[tuple[float, float] | None] = []
    for point in landmarks:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each DogFLW landmark must be an xy pair")
        try:
            xy = (float(point[0]), float(point[1]))
        except (TypeError, ValueError):
            xy = (math.nan, math.nan)
        parsed.append(xy if all(math.isfinite(value) for value in xy) else None)

    derived: list[tuple[float, float] | None] = []
    for name in KEYPOINT_ORDER:
        source_indices = DOGFLW_DERIVATION[name]
        source_points = [parsed[index] for index in source_indices]
        if any(point is None for point in source_points):
            derived.append(None)
            continue
        concrete = [point for point in source_points if point is not None]
        derived.append(
            (
                sum(point[0] for point in concrete) / len(concrete),
                sum(point[1] for point in concrete) / len(concrete),
            )
        )
    return tuple(derived)


def parse_dogflw_zip(archive_path: Path) -> dict[str, tuple[NoseKeypointRecord, ...]]:
    """Parse publisher train/test face crops and face46 labels from a ZIP."""

    path = _regular_zip_path(archive_path)
    result: dict[str, list[NoseKeypointRecord]] = {"train": [], "test": []}
    with ZipFile(path) as archive:
        members = _safe_member_map(archive)
        for split in result:
            marker = f"DogFLW/{split}/labels/"
            label_members = sorted(
                name
                for name in members
                if marker in name and name.endswith(".json")
            )
            if not label_members:
                raise ValueError(f"DogFLW ZIP has no publisher {split} labels")
            for label_member in label_members:
                if not label_member.endswith(f"/{marker.split('/', 1)[1]}{PurePosixPath(label_member).name}"):
                    continue
                payload = _read_json(
                    archive, members[label_member], allow_publisher_nan=True
                )
                if not isinstance(payload, dict):
                    raise ValueError("DogFLW label must be an object")
                landmarks = payload.get("landmarks")
                if not isinstance(landmarks, list):
                    raise ValueError("DogFLW landmarks must be an array")
                bbox = payload.get("bounding_boxes")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError("DogFLW bounding_boxes must contain xyxy")
                stem = PurePosixPath(label_member).stem
                image_member = _dogflw_image_member(members, split, stem)
                try:
                    crop = tuple(float(value) for value in bbox)
                    if not all(math.isfinite(value) for value in crop):
                        raise ValueError
                except (TypeError, ValueError):
                    # A few publisher rows have a blank face box but valid
                    # landmarks. Bind those rows to the full decoded image.
                    image_bytes = _read_member(
                        archive, members[image_member], _MAX_IMAGE_BYTES
                    )
                    with Image.open(BytesIO(image_bytes)) as opened:
                        crop = (0.0, 0.0, float(opened.width), float(opened.height))
                result[split].append(
                    NoseKeypointRecord(
                        dataset="dogflw",
                        split=split,
                        archive_path=path,
                        image_member=image_member,
                        sample_id=f"dogflw:{split}:{stem}",
                        crop_xyxy=crop,
                        points=dogflw_points(landmarks),
                        supported=(True,) * len(KEYPOINT_ORDER),
                    )
                )
    return {split: tuple(records) for split, records in result.items()}


def _regular_zip_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("dataset archive must be a regular ZIP file")
    return candidate.resolve()


def _safe_member_map(archive: ZipFile) -> dict[str, ZipInfo]:
    members: dict[str, ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        file_type = (info.external_attr >> 16) & 0o170000
        if file_type == stat.S_IFLNK:
            raise ValueError(f"ZIP symlink members are not allowed: {info.filename}")
        relative = _safe_relative(info.filename)
        name = relative.as_posix()
        if name in members:
            raise ValueError(f"duplicate ZIP member: {name}")
        members[name] = info
    return members


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe ZIP-relative path: {value!r}")
    return path


def _unique_suffix_member(members: Mapping[str, ZipInfo], suffix: str) -> str:
    matches = [name for name in members if name == suffix or name.endswith(f"/{suffix}")]
    if len(matches) != 1:
        raise ValueError(f"ZIP must contain exactly one {suffix}; found {len(matches)}")
    return matches[0]


def _read_member(archive: ZipFile, member: str | ZipInfo, limit: int) -> bytes:
    info = archive.getinfo(member) if isinstance(member, str) else member
    if info.file_size > limit:
        raise ValueError(f"ZIP member exceeds {limit} bytes: {info.filename}")
    with archive.open(info) as stream:
        value = stream.read(limit + 1)
    if len(value) > limit or len(value) != info.file_size:
        raise ValueError(f"ZIP member size differs: {info.filename}")
    return value


def _read_json(
    archive: ZipFile, info: ZipInfo, *, allow_publisher_nan: bool = False
) -> Any:
    try:
        encoded = _read_member(archive, info, _MAX_JSON_BYTES)
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=(
                (lambda value: math.nan if value == "NaN" else _reject_json_constant(value))
                if allow_publisher_nan
                else lambda value: _reject_json_constant(value)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON member: {info.filename}") from exc


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _ap10k_images(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw_images = payload.get("images")
    if not isinstance(raw_images, list):
        raise ValueError("AP-10K images must be an array")
    images: dict[int, Mapping[str, Any]] = {}
    for image in raw_images:
        if not isinstance(image, dict) or "id" not in image or "file_name" not in image:
            raise ValueError("AP-10K image record differs")
        image_id = int(image["id"])
        if image_id in images:
            raise ValueError("AP-10K image IDs must be unique")
        images[image_id] = image
    return images


def _dogflw_image_member(
    members: Mapping[str, ZipInfo], split: str, stem: str
) -> str:
    matches = [
        name
        for name in members
        if any(
            name == f"DogFLW/{split}/images/{stem}{suffix}"
            or name.endswith(f"/DogFLW/{split}/images/{stem}{suffix}")
            for suffix in (".png", ".jpg", ".jpeg")
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"DogFLW image for {split}/{stem} differs")
    return matches[0]


def _crop_image(
    image: Image.Image, bbox: tuple[float, float, float, float]
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox
    left = max(0, min(image.width, int(math.floor(x0))))
    top = max(0, min(image.height, int(math.floor(y0))))
    right = max(0, min(image.width, int(math.ceil(x1))))
    bottom = max(0, min(image.height, int(math.ceil(y1))))
    if left >= right or top >= bottom:
        raise ValueError("annotation crop is empty after image clipping")
    integer_box = (left, top, right, bottom)
    return image.crop(integer_box), integer_box


__all__ = [
    "NoseKeypointRecord",
    "ZipNoseKeypointDataset",
    "dogflw_points",
    "keypoint_metrics",
    "parse_ap10k_zip",
    "parse_dogflw_zip",
    "partial_keypoint_loss",
]
