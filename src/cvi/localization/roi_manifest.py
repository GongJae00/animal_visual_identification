"""Export detector predictions as instance-level ReID crop manifests."""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from cvi.canid_data.types import UnifiedCanidSample
from cvi.localization.consensus import consensus_dog_instances
from cvi.localization.prediction_cache import cache_record_to_result
from cvi.localization.quality import (
    compute_iou,
    estimate_blur,
    score_dog_quality,
    score_face_quality,
)
from cvi.localization.roi import (
    face_and_weak_nose_rois_from_pose,
    square_padded_crop_with_mask,
)
from cvi.protected_io import read_strict_json_document
from cvi.provenance import content_sha256

_BUNDLE_SCHEMA = "cvi.canid_roi_manifest_bundle.v2"
_MANIFEST_SCHEMA = "cvi.canid_roi_manifest.v2"
_SHA256_LENGTH = 64


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_roi_manifest(
    samples: Sequence[UnifiedCanidSample],
    caches: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    output_dir: Path,
    margin: float = 1.15,
    target_size: int = 224,
) -> dict[str, Any]:
    if not caches:
        raise ValueError("ROI export requires at least one prediction cache")
    if not samples:
        raise ValueError("ROI export requires at least one sample")
    if (
        isinstance(target_size, bool)
        or not isinstance(target_size, int)
        or target_size <= 0
    ):
        raise ValueError("ROI target size must be a positive integer")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(margin)
        or margin <= 0
    ):
        raise ValueError("ROI margin must be finite and positive")
    resolved_data_root = data_root.resolve(strict=True)
    if not resolved_data_root.is_dir():
        raise ValueError("ROI data root must be a directory")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    cache_maps = []
    for cache in caches:
        if cache["dataset_name"] != samples[0].dataset_name:
            raise ValueError("prediction cache dataset differs from samples")
        cache_maps.append({record["sample_id"]: record for record in cache["records"]})
    sample_ids = set(cache_maps[0])
    if any(set(records) != sample_ids for records in cache_maps[1:]):
        raise ValueError("prediction caches must cover the same sample set")
    if not sample_ids.issubset(sample_by_id):
        raise ValueError("prediction cache contains unknown samples")

    crop_dir = output_dir / "dog_crops"
    mask_dir = output_dir / "source_valid_masks"
    face_dir = output_dir / "face_crops"
    face_mask_dir = output_dir / "face_source_valid_masks"
    nose_dir = output_dir / "weak_nose_crops"
    nose_mask_dir = output_dir / "weak_nose_source_valid_masks"
    crop_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)
    face_mask_dir.mkdir(parents=True, exist_ok=True)
    nose_dir.mkdir(parents=True, exist_ok=True)
    nose_mask_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for sample_id in sorted(sample_ids):
        sample = sample_by_id[sample_id]
        raw_records = [cache_map[sample_id] for cache_map in cache_maps]
        for raw in raw_records:
            if (
                raw["image_path"] != sample.image_path
                or raw["image_sha256"] != sample.image_sha256
                or raw["width"] != sample.width
                or raw["height"] != sample.height
            ):
                raise ValueError("prediction cache image binding differs")
        relative_image_path = _require_relative_path(sample.image_path, "image_path")
        image_path = resolved_data_root.joinpath(*relative_image_path.parts)
        try:
            image_path = image_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("source image does not exist or is unsafe") from exc
        if (
            not image_path.is_relative_to(resolved_data_root)
            or image_path.relative_to(resolved_data_root).as_posix()
            != relative_image_path.as_posix()
            or not image_path.is_file()
        ):
            raise ValueError("source image must be a regular file under data_root")
        if _file_sha256(image_path) != sample.image_sha256:
            raise ValueError("source image hash differs during ROI export")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        results = tuple(
            cache_record_to_result(raw, cache["model"])
            for raw, cache in zip(raw_records, caches, strict=True)
        )
        instances = consensus_dog_instances(results)
        registered_identity_id = (
            sample.registered_identity_id if len(instances) == 1 else None
        )
        for instance_index, instance in enumerate(instances):
            instance_id = content_sha256(
                {
                    "sample_id": sample_id,
                    "instance_index": instance_index,
                    "bbox": [
                        round(value, 4)
                        for value in (
                            instance.bbox.x1,
                            instance.bbox.y1,
                            instance.bbox.x2,
                            instance.bbox.y2,
                        )
                    ],
                    "support_models": instance.support_models,
                }
            )[:32]
            crop, source_valid, crop_rect = square_padded_crop_with_mask(
                image, instance.bbox, margin=margin, target_size=target_size
            )
            crop_path = crop_dir / f"{instance_id}.jpg"
            mask_path = mask_dir / f"{instance_id}.png"
            if crop_path.exists() or mask_path.exists():
                raise FileExistsError(f"ROI output already exists: {instance_id}")
            crop.save(crop_path, quality=95)
            source_valid.save(mask_path)
            crop_sha256 = _file_sha256(crop_path)
            mask_sha256 = _file_sha256(mask_path)
            quality = score_dog_quality(
                instance.bbox,
                model_agreement=instance.agreement,
                multi_dog_boxes=max(len(instances), 1),
                image_width=sample.width,
                image_height=sample.height,
                blur=estimate_blur(crop),
            )
            pose_set = None
            pose_iou = 0.0
            for result in results:
                for predicted_box, keypoints in zip(
                    result.dog_boxes, result.body_keypoints, strict=False
                ):
                    overlap = compute_iou(instance.bbox, predicted_box)
                    if overlap > pose_iou:
                        pose_iou = overlap
                        pose_set = keypoints
            face_roi = None
            nose_roi = None
            face_crop_path = None
            face_mask_path = None
            face_crop_sha256 = None
            face_mask_sha256 = None
            face_crop_rect = None
            face_quality = None
            nose_crop_path = None
            nose_mask_path = None
            nose_crop_sha256 = None
            nose_mask_sha256 = None
            if pose_set is not None and pose_iou >= 0.3:
                face_roi, nose_roi = face_and_weak_nose_rois_from_pose(
                    pose_set,
                    instance.bbox,
                    image_width=sample.width,
                    image_height=sample.height,
                )
            if face_roi is not None:
                face_crop, face_valid, face_crop_rect = square_padded_crop_with_mask(
                    image, face_roi, margin=1.15, target_size=target_size
                )
                face_target = face_dir / f"{instance_id}.jpg"
                face_mask_target = face_mask_dir / f"{instance_id}.png"
                if face_target.exists() or face_mask_target.exists():
                    raise FileExistsError(
                        f"face ROI output already exists: {instance_id}"
                    )
                face_crop.save(face_target, quality=95)
                face_valid.save(face_mask_target)
                face_crop_path = str(face_target.relative_to(output_dir))
                face_mask_path = str(face_mask_target.relative_to(output_dir))
                face_crop_sha256 = _file_sha256(face_target)
                face_mask_sha256 = _file_sha256(face_mask_target)
                face_quality = score_face_quality(
                    pose_set,
                    face_roi,
                    image_width=sample.width,
                    image_height=sample.height,
                    blur=estimate_blur(face_crop),
                )
            if nose_roi is not None:
                nose_crop, nose_valid, _ = square_padded_crop_with_mask(
                    image, nose_roi, margin=1.20, target_size=target_size
                )
                nose_target = nose_dir / f"{instance_id}.jpg"
                nose_mask_target = nose_mask_dir / f"{instance_id}.png"
                if nose_target.exists() or nose_mask_target.exists():
                    raise FileExistsError(
                        f"nose ROI output already exists: {instance_id}"
                    )
                nose_crop.save(nose_target, quality=95)
                nose_valid.save(nose_mask_target)
                nose_crop_path = str(nose_target.relative_to(output_dir))
                nose_mask_path = str(nose_mask_target.relative_to(output_dir))
                nose_crop_sha256 = _file_sha256(nose_target)
                nose_mask_sha256 = _file_sha256(nose_mask_target)
            records.append(
                {
                    "sample_id": sample_id,
                    "instance_id": instance_id,
                    "dataset_name": sample.dataset_name,
                    "dataset_version": sample.dataset_version,
                    "image_path": sample.image_path,
                    "image_sha256": sample.image_sha256,
                    "image_width": sample.width,
                    "image_height": sample.height,
                    "dog_bbox_xyxy": [
                        instance.bbox.x1,
                        instance.bbox.y1,
                        instance.bbox.x2,
                        instance.bbox.y2,
                    ],
                    "dog_crop_path": str(crop_path.relative_to(output_dir)),
                    "dog_crop_sha256": crop_sha256,
                    "source_valid_mask_path": str(mask_path.relative_to(output_dir)),
                    "source_valid_mask_sha256": mask_sha256,
                    "crop_rect_xyxy": list(crop_rect),
                    "crop_margin": margin,
                    "crop_size": target_size,
                    "face_roi_xyxy": (
                        [face_roi.x1, face_roi.y1, face_roi.x2, face_roi.y2]
                        if face_roi is not None
                        else None
                    ),
                    "face_crop_path": face_crop_path,
                    "face_crop_sha256": face_crop_sha256,
                    "face_source_valid_mask_path": face_mask_path,
                    "face_source_valid_mask_sha256": face_mask_sha256,
                    "face_crop_rect_xyxy": (
                        list(face_crop_rect) if face_crop_rect is not None else None
                    ),
                    "face_quality": (
                        asdict(face_quality) if face_quality is not None else None
                    ),
                    "nose_roi_xyxy": (
                        [nose_roi.x1, nose_roi.y1, nose_roi.x2, nose_roi.y2]
                        if nose_roi is not None
                        else None
                    ),
                    "weak_nose_crop_path": nose_crop_path,
                    "weak_nose_crop_sha256": nose_crop_sha256,
                    "weak_nose_source_valid_mask_path": nose_mask_path,
                    "weak_nose_source_valid_mask_sha256": nose_mask_sha256,
                    "body_keypoints": (
                        {
                            name: [point.x, point.y, point.confidence]
                            for name, point in sorted(pose_set.keypoints.items())
                        }
                        if pose_set is not None
                        else None
                    ),
                    "quality": asdict(quality),
                    "teacher_support_models": list(instance.support_models),
                    "teacher_agreement": instance.agreement,
                    "review_state": instance.admission,
                    "registered_identity_id": registered_identity_id,
                    "capture_group_id": sample.capture_group_id,
                    "capture_group_kind": sample.capture_group_kind.value,
                    "camera_id": sample.camera_id,
                    "timestamp_ms": sample.timestamp_ms,
                    "split_role": sample.split_role,
                }
            )
    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "dataset_name": samples[0].dataset_name,
        "dataset_version": samples[0].dataset_version,
        "source_sample_ids": sorted(sample_ids),
        "prediction_cache_sha256s": [content_sha256(cache) for cache in caches],
        "records": records,
    }
    return {
        "schema_version": _BUNDLE_SCHEMA,
        "manifest_sha256": content_sha256(manifest),
        "manifest": manifest,
    }


_RECORD_FIELDS = {
    "sample_id",
    "instance_id",
    "dataset_name",
    "dataset_version",
    "image_path",
    "image_sha256",
    "image_width",
    "image_height",
    "dog_bbox_xyxy",
    "dog_crop_path",
    "dog_crop_sha256",
    "source_valid_mask_path",
    "source_valid_mask_sha256",
    "crop_rect_xyxy",
    "crop_margin",
    "crop_size",
    "face_roi_xyxy",
    "face_crop_path",
    "face_crop_sha256",
    "face_source_valid_mask_path",
    "face_source_valid_mask_sha256",
    "face_crop_rect_xyxy",
    "face_quality",
    "nose_roi_xyxy",
    "weak_nose_crop_path",
    "weak_nose_crop_sha256",
    "weak_nose_source_valid_mask_path",
    "weak_nose_source_valid_mask_sha256",
    "body_keypoints",
    "quality",
    "teacher_support_models",
    "teacher_agreement",
    "review_state",
    "registered_identity_id",
    "capture_group_id",
    "capture_group_kind",
    "camera_id",
    "timestamp_ms",
    "split_role",
}
_QUALITY_FIELDS = {
    "detector_confidence",
    "model_agreement",
    "truncation",
    "native_resolution",
    "multi_dog_contamination",
    "blur_estimate",
    "overall",
}
_FACE_QUALITY_FIELDS = {
    "landmark_confidence",
    "anchor_visibility",
    "yaw_roll_proxy",
    "resolution",
    "truncation",
    "blur_estimate",
    "overall",
}


def _require_text(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_uuid5(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical UUIDv5")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical UUIDv5") from exc
    if parsed.version != 5 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUIDv5")
    return value


def _require_number(value: Any, name: str, *, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return float(value)


def _require_bbox(value: Any, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must be a four-coordinate list")
    coordinates = tuple(
        _require_number(item, name, minimum=-float("inf"), maximum=float("inf"))
        for item in value
    )
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        raise ValueError(f"{name} must be non-empty")
    return coordinates


def _require_crop_rect(
    value: Any,
    name: str,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must be a four-integer list")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
        raise ValueError(
            f"{name} must be a non-empty rectangle within the source image"
        )
    return (x1, y1, x2, y2)


def _require_relative_path(value: Any, name: str) -> PurePosixPath:
    text = _require_text(value, name)
    assert text is not None
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or text != path.as_posix()
        or "\\" in text
        or path.parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a canonical safe relative path")
    return path


def _verify_artifact(
    root: Path,
    *,
    relative_path: Any,
    expected_sha256: Any,
    expected_size: int,
    path_name: str,
    digest_name: str,
    expected_mode: str,
) -> None:
    pure_path = _require_relative_path(relative_path, path_name)
    digest = _require_sha256(expected_sha256, digest_name)
    candidate = root.joinpath(*pure_path.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{path_name} artifact does not exist or is unsafe") from exc
    if (
        not resolved.is_relative_to(root)
        or resolved.relative_to(root).as_posix() != pure_path.as_posix()
        or not resolved.is_file()
    ):
        raise ValueError(
            f"{path_name} must resolve to a regular file under the manifest parent"
        )
    if _file_sha256(resolved) != digest:
        raise ValueError(f"{digest_name} differs from the artifact")
    try:
        with Image.open(resolved) as opened:
            if opened.size != (expected_size, expected_size):
                raise ValueError(f"{path_name} dimensions differ from crop_size")
            if opened.mode != expected_mode:
                raise ValueError(f"{path_name} image mode differs")
            opened.load()
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"{path_name} is not a valid image artifact") from exc


def _validate_optional_artifacts(
    record: Mapping[str, Any],
    root: Path,
    *,
    prefix: str,
    roi_field: str,
    crop_path_field: str,
    crop_hash_field: str,
    mask_path_field: str,
    mask_hash_field: str,
    crop_size: int,
    image_width: int,
    image_height: int,
    crop_rect_field: str | None = None,
) -> None:
    fields = [
        roi_field,
        crop_path_field,
        crop_hash_field,
        mask_path_field,
        mask_hash_field,
    ]
    if crop_rect_field is not None:
        fields.append(crop_rect_field)
    values = [record[field] for field in fields]
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(
            f"ROI manifest {prefix} artifact fields must be all present or all null"
        )
    roi = _require_bbox(record[roi_field], roi_field)
    if not (
        0.0 <= roi[0] < roi[2] <= image_width and 0.0 <= roi[1] < roi[3] <= image_height
    ):
        raise ValueError(f"{roi_field} must be within the source image")
    if crop_rect_field is not None:
        _require_crop_rect(
            record[crop_rect_field],
            crop_rect_field,
            image_width=image_width,
            image_height=image_height,
        )
    _verify_artifact(
        root,
        relative_path=record[crop_path_field],
        expected_sha256=record[crop_hash_field],
        expected_size=crop_size,
        path_name=crop_path_field,
        digest_name=crop_hash_field,
        expected_mode="RGB",
    )
    _verify_artifact(
        root,
        relative_path=record[mask_path_field],
        expected_sha256=record[mask_hash_field],
        expected_size=crop_size,
        path_name=mask_path_field,
        digest_name=mask_hash_field,
        expected_mode="L",
    )


def _validate_record(record: Any, root: Path, manifest: Mapping[str, Any]) -> None:
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise ValueError("ROI manifest record schema differs")
    for field in ("sample_id", "dataset_name", "dataset_version", "split_role"):
        _require_text(record[field], field)
    instance_id = _require_text(record["instance_id"], "instance_id")
    if len(instance_id or "") != 32 or any(
        char not in "0123456789abcdef" for char in instance_id or ""
    ):
        raise ValueError("instance_id must be 32 lowercase hexadecimal characters")
    if (
        record["dataset_name"] != manifest["dataset_name"]
        or record["dataset_version"] != manifest["dataset_version"]
        or record["sample_id"] not in manifest["source_sample_ids"]
    ):
        raise ValueError("ROI manifest record dataset or sample binding differs")
    _require_relative_path(record["image_path"], "image_path")
    _require_sha256(record["image_sha256"], "image_sha256")
    dimensions = (record["image_width"], record["image_height"], record["crop_size"])
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in dimensions
    ):
        raise ValueError(
            "ROI manifest image and crop dimensions must be positive integers"
        )
    image_width, image_height, crop_size = dimensions
    _require_bbox(record["dog_bbox_xyxy"], "dog_bbox_xyxy")
    _require_crop_rect(
        record["crop_rect_xyxy"],
        "crop_rect_xyxy",
        image_width=image_width,
        image_height=image_height,
    )
    _require_number(
        record["crop_margin"],
        "crop_margin",
        minimum=0.0,
        maximum=float("inf"),
    )
    if record["crop_margin"] <= 0:
        raise ValueError("crop_margin must be positive")
    _verify_artifact(
        root,
        relative_path=record["dog_crop_path"],
        expected_sha256=record["dog_crop_sha256"],
        expected_size=crop_size,
        path_name="dog_crop_path",
        digest_name="dog_crop_sha256",
        expected_mode="RGB",
    )
    _verify_artifact(
        root,
        relative_path=record["source_valid_mask_path"],
        expected_sha256=record["source_valid_mask_sha256"],
        expected_size=crop_size,
        path_name="source_valid_mask_path",
        digest_name="source_valid_mask_sha256",
        expected_mode="L",
    )
    _validate_optional_artifacts(
        record,
        root,
        prefix="face",
        roi_field="face_roi_xyxy",
        crop_path_field="face_crop_path",
        crop_hash_field="face_crop_sha256",
        mask_path_field="face_source_valid_mask_path",
        mask_hash_field="face_source_valid_mask_sha256",
        crop_rect_field="face_crop_rect_xyxy",
        crop_size=crop_size,
        image_width=image_width,
        image_height=image_height,
    )
    _validate_optional_artifacts(
        record,
        root,
        prefix="weak-nose",
        roi_field="nose_roi_xyxy",
        crop_path_field="weak_nose_crop_path",
        crop_hash_field="weak_nose_crop_sha256",
        mask_path_field="weak_nose_source_valid_mask_path",
        mask_hash_field="weak_nose_source_valid_mask_sha256",
        crop_size=crop_size,
        image_width=image_width,
        image_height=image_height,
    )
    keypoints = record["body_keypoints"]
    if keypoints is not None:
        if not isinstance(keypoints, dict) or not keypoints:
            raise ValueError("body_keypoints must be a non-empty object or null")
        for name, point in keypoints.items():
            _require_text(name, "body keypoint name")
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError("body keypoint values must be [x, y, confidence]")
            _require_number(
                point[0],
                "body keypoint x",
                minimum=-float("inf"),
                maximum=float("inf"),
            )
            _require_number(
                point[1],
                "body keypoint y",
                minimum=-float("inf"),
                maximum=float("inf"),
            )
            _require_number(
                point[2],
                "body keypoint confidence",
                minimum=0.0,
                maximum=1.0,
            )
    if record["face_roi_xyxy"] is not None and keypoints is None:
        raise ValueError("face artifacts require body_keypoints")
    face_quality = record["face_quality"]
    if record["face_crop_path"] is None:
        if face_quality is not None:
            raise ValueError("face_quality requires face artifacts")
    else:
        if (
            not isinstance(face_quality, dict)
            or set(face_quality) != _FACE_QUALITY_FIELDS
        ):
            raise ValueError("ROI manifest face_quality schema differs")
        for name, value in face_quality.items():
            _require_number(value, f"face_quality.{name}", minimum=0.0, maximum=1.0)
    quality = record["quality"]
    if not isinstance(quality, dict) or set(quality) != _QUALITY_FIELDS:
        raise ValueError("ROI manifest quality schema differs")
    for name, value in quality.items():
        _require_number(value, f"quality.{name}", minimum=0.0, maximum=1.0)
    support_models = record["teacher_support_models"]
    if not isinstance(support_models, list) or not support_models:
        raise ValueError(
            "teacher_support_models must be a sorted unique non-empty list"
        )
    for model_name in support_models:
        _require_text(model_name, "teacher support model")
    if support_models != sorted(support_models) or len(support_models) != len(
        set(support_models)
    ):
        raise ValueError(
            "teacher_support_models must be a sorted unique non-empty list"
        )
    _require_number(
        record["teacher_agreement"],
        "teacher_agreement",
        minimum=0.0,
        maximum=1.0,
    )
    review_state = _require_text(record["review_state"], "review_state")
    if review_state not in {"ACCEPT", "REVIEW", "REJECT"}:
        raise ValueError("ROI manifest review_state differs")
    if record["registered_identity_id"] is not None:
        _require_uuid5(record["registered_identity_id"], "registered_identity_id")
    for field in ("capture_group_id", "camera_id"):
        _require_text(record[field], field, optional=True)
    if record["timestamp_ms"] is not None and (
        isinstance(record["timestamp_ms"], bool)
        or not isinstance(record["timestamp_ms"], int)
    ):
        raise ValueError("timestamp_ms must be an integer or null")
    capture_group_kind = _require_text(
        record["capture_group_kind"], "capture_group_kind"
    )
    if capture_group_kind not in {
        "REAL_CAMERA_SESSION",
        "VIDEO_TRACK",
        "ALBUM_OR_SOURCE_GROUP",
        "POSE_VIEW_CLUSTER",
        "UNKNOWN",
    }:
        raise ValueError("capture_group_kind differs")


def _validate_manifest(manifest: Any, root: Path) -> None:
    expected = {
        "schema_version",
        "dataset_name",
        "dataset_version",
        "source_sample_ids",
        "prediction_cache_sha256s",
        "records",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError("ROI manifest schema differs")
    if manifest["schema_version"] != _MANIFEST_SCHEMA:
        raise ValueError("ROI manifest schema version differs")
    _require_text(manifest["dataset_name"], "dataset_name")
    _require_text(manifest["dataset_version"], "dataset_version")
    source_ids = manifest["source_sample_ids"]
    if not isinstance(source_ids, list) or not source_ids:
        raise ValueError("source_sample_ids must be a sorted unique non-empty list")
    for sample_id in source_ids:
        _require_text(sample_id, "source sample ID")
    if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
        raise ValueError("source_sample_ids must be a sorted unique non-empty list")
    cache_hashes = manifest["prediction_cache_sha256s"]
    if not isinstance(cache_hashes, list) or not cache_hashes:
        raise ValueError("prediction_cache_sha256s must be a non-empty list")
    for digest in cache_hashes:
        _require_sha256(digest, "prediction cache SHA-256")
    records = manifest["records"]
    if not isinstance(records, list):
        raise ValueError("ROI manifest records must be a list")
    instance_ids: set[str] = set()
    records_per_sample: dict[str, int] = {}
    for record in records:
        _validate_record(record, root, manifest)
        instance_id = record["instance_id"]
        if instance_id in instance_ids:
            raise ValueError("ROI manifest instance IDs must be unique")
        instance_ids.add(instance_id)
        sample_id = record["sample_id"]
        records_per_sample[sample_id] = records_per_sample.get(sample_id, 0) + 1
    for record in records:
        if (
            records_per_sample[record["sample_id"]] != 1
            and record["registered_identity_id"] is not None
        ):
            raise ValueError(
                "multi-instance samples must not carry registered identity"
            )


def read_roi_manifest(path: Path) -> dict[str, Any]:
    document = read_strict_json_document(
        path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    bundle = document.payload
    expected = {"schema_version", "manifest_sha256", "manifest"}
    if set(bundle) != expected or bundle["schema_version"] != _BUNDLE_SCHEMA:
        raise ValueError("ROI manifest bundle schema differs")
    manifest = bundle["manifest"]
    _require_sha256(bundle["manifest_sha256"], "manifest_sha256")
    if (
        not isinstance(manifest, dict)
        or content_sha256(manifest) != bundle["manifest_sha256"]
    ):
        raise ValueError("ROI manifest bundle digest differs")
    root = path.parent.resolve(strict=True)
    _validate_manifest(manifest, root)
    return manifest


__all__ = ["build_roi_manifest", "read_roi_manifest"]
