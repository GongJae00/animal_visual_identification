"""Content-bound localization prediction caches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.types import UnifiedCanidSample
from foundation.protected_io import (
    read_content_hashed_json_bundle,
    write_private_json_bundle,
)
from foundation.provenance import content_sha256
from parsing.types import DetectionBox, LocalizationResult

_BUNDLE_SCHEMA = "localization_prediction_cache_bundle.v1"
_CACHE_SCHEMA = "localization_prediction_cache.v1"
_MODEL_FIELDS = {
    "family",
    "name",
    "artifact_sha256",
    "artifact_size_bytes",
    "license_id",
    "device",
}
_RECORD_FIELDS = {
    "sample_id",
    "image_path",
    "image_sha256",
    "width",
    "height",
    "dog_boxes",
    "body_keypoints",
    "inference_ms",
}
_BOX_FIELDS = {
    "detection_index",
    "xyxy",
    "confidence",
    "class_id",
    "class_name",
}
_KEYPOINT_SET_FIELDS = {"schema", "points"}


def build_prediction_cache(
    samples: Sequence[UnifiedCanidSample],
    results: Sequence[LocalizationResult],
    *,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_model(dict(model))
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("prediction cache samples must have unique IDs")
    if len(results) != len(samples) or {result.image_id for result in results} != set(
        sample_by_id
    ):
        raise ValueError("prediction results must exactly cover the sample set")

    records: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: item.image_id):
        sample = sample_by_id[result.image_id]
        if result.model_family != model["family"] or result.model_name != model["name"]:
            raise ValueError(
                "prediction result model identity differs from cache model"
            )
        if result.metadata is not None and (
            result.metadata.get("artifact_sha256") != model["artifact_sha256"]
        ):
            raise ValueError(
                "prediction result artifact identity differs from cache model"
            )
        if result.body_keypoints and len(result.body_keypoints) != len(
            result.dog_boxes
        ):
            raise ValueError("body keypoint sets must align with dog boxes")
        box_order = sorted(
            range(len(result.dog_boxes)),
            key=lambda index: (
                -result.dog_boxes[index].confidence,
                result.dog_boxes[index].x1,
                result.dog_boxes[index].y1,
                result.dog_boxes[index].x2,
                result.dog_boxes[index].y2,
            ),
        )
        boxes = [result.dog_boxes[index] for index in box_order]
        records.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "image_sha256": sample.image_sha256,
                "width": sample.width,
                "height": sample.height,
                "dog_boxes": [
                    {
                        "detection_index": index,
                        "xyxy": [box.x1, box.y1, box.x2, box.y2],
                        "confidence": box.confidence,
                        "class_id": box.class_id,
                        "class_name": box.class_name,
                    }
                    for index, box in enumerate(boxes)
                ],
                "body_keypoints": [
                    {
                        "schema": point_set.schema,
                        "points": {
                            name: [point.x, point.y, point.confidence]
                            for name, point in sorted(point_set.keypoints.items())
                        },
                    }
                    for point_set in (
                        [result.body_keypoints[index] for index in box_order]
                        if result.body_keypoints
                        else []
                    )
                ],
                "inference_ms": result.inference_ms,
            }
        )
    sample_set = [
        [record["sample_id"], record["image_sha256"], record["width"], record["height"]]
        for record in records
    ]
    cache = {
        "schema_version": _CACHE_SCHEMA,
        "dataset_name": samples[0].dataset_name if samples else "",
        "dataset_version": samples[0].dataset_version if samples else "",
        "sample_set_sha256": content_sha256(sample_set),
        "model": dict(model),
        "records": records,
    }
    validate_prediction_cache(cache)
    return {
        "schema_version": _BUNDLE_SCHEMA,
        "cache_sha256": content_sha256(cache),
        "cache": cache,
    }


def validate_prediction_cache(cache: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "dataset_name",
        "dataset_version",
        "sample_set_sha256",
        "model",
        "records",
    }
    if (
        not isinstance(cache, dict)
        or set(cache) != required
        or cache["schema_version"] != _CACHE_SCHEMA
    ):
        raise ValueError("localization prediction cache schema differs")
    for field in ("dataset_name", "dataset_version"):
        if not isinstance(cache[field], str):
            raise TypeError(f"prediction cache {field} must be a string")
    _validate_sha256(cache["sample_set_sha256"], "prediction cache sample-set")
    _validate_model(cache["model"])
    records = cache["records"]
    if not isinstance(records, list):
        raise TypeError("prediction cache records must be a list")
    seen: set[str] = set()
    sample_set: list[list[Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise ValueError("prediction cache record schema differs")
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("prediction cache sample IDs must be unique")
        seen.add(sample_id)
        if not isinstance(record["image_path"], str) or not record["image_path"]:
            raise ValueError("prediction cache image paths must be non-empty strings")
        _validate_sha256(record["image_sha256"], "prediction cache image")
        for field in ("width", "height"):
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"prediction cache {field} must be a positive integer")
        if not _is_finite_number(record["inference_ms"]) or record["inference_ms"] < 0:
            raise ValueError("prediction latency must be finite and non-negative")
        raw_boxes = record["dog_boxes"]
        if not isinstance(raw_boxes, list):
            raise TypeError("prediction cache dog_boxes must be a list")
        for index, raw_box in enumerate(raw_boxes):
            if not isinstance(raw_box, dict) or set(raw_box) != _BOX_FIELDS:
                raise ValueError("prediction cache detection box schema differs")
            if (
                isinstance(raw_box["detection_index"], bool)
                or not isinstance(raw_box["detection_index"], int)
                or raw_box["detection_index"] != index
            ):
                raise ValueError("prediction detection indices must be ordered")
            coordinates = raw_box["xyxy"]
            if (
                not isinstance(coordinates, list)
                or len(coordinates) != 4
                or not all(_is_finite_number(value) for value in coordinates)
            ):
                raise ValueError("prediction box xyxy must contain four finite numbers")
            if not _is_finite_number(raw_box["confidence"]):
                raise ValueError("prediction box confidence must be finite")
            if isinstance(raw_box["class_id"], bool) or not isinstance(
                raw_box["class_id"], int
            ):
                raise TypeError("prediction box class_id must be an integer")
            if not isinstance(raw_box["class_name"], str) or not raw_box["class_name"]:
                raise ValueError("prediction box class_name must be a non-empty string")
            DetectionBox(
                *coordinates,
                raw_box["confidence"],
                raw_box["class_id"],
                raw_box["class_name"],
            )
        raw_keypoint_sets = record["body_keypoints"]
        if not isinstance(raw_keypoint_sets, list):
            raise TypeError("prediction cache body_keypoints must be a list")
        if raw_keypoint_sets and len(raw_keypoint_sets) != len(raw_boxes):
            raise ValueError("body keypoint sets must align with dog boxes")
        for raw_set in raw_keypoint_sets:
            _validate_keypoint_set(raw_set)
        sample_set.append(
            [sample_id, record["image_sha256"], record["width"], record["height"]]
        )
    if [record["sample_id"] for record in records] != sorted(seen):
        raise ValueError("prediction cache records must be ordered by sample ID")
    if content_sha256(sample_set) != cache["sample_set_sha256"]:
        raise ValueError("prediction cache sample-set digest differs")


def write_prediction_cache(path: Path, bundle: dict[str, Any]) -> None:
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"schema_version", "cache_sha256", "cache"}
        or bundle["schema_version"] != _BUNDLE_SCHEMA
    ):
        raise ValueError("localization prediction cache bundle schema differs")
    _validate_sha256(bundle["cache_sha256"], "prediction cache bundle")
    if not isinstance(bundle["cache"], dict):
        raise TypeError("prediction cache bundle cache must be an object")
    validate_prediction_cache(bundle["cache"])
    if content_sha256(bundle["cache"]) != bundle["cache_sha256"]:
        raise ValueError("prediction cache bundle digest differs")
    write_private_json_bundle(((path, bundle),))


def read_prediction_cache(path: Path) -> dict[str, Any]:
    cache = read_content_hashed_json_bundle(
        path,
        schema_version=_BUNDLE_SCHEMA,
        payload_field="cache",
        sha256_field="cache_sha256",
    )
    validate_prediction_cache(cache)
    return cache


def cache_record_to_result(
    record: Mapping[str, Any], model: Mapping[str, Any]
) -> LocalizationResult:
    from parsing.types import Keypoint, KeypointSet

    boxes = tuple(
        DetectionBox(
            *raw["xyxy"], raw["confidence"], raw["class_id"], raw["class_name"]
        )
        for raw in record["dog_boxes"]
    )
    point_sets = tuple(
        KeypointSet(
            {
                name: Keypoint(*coordinates)
                for name, coordinates in raw_set["points"].items()
            },
            raw_set["schema"],
        )
        for raw_set in record["body_keypoints"]
    )
    return LocalizationResult(
        image_id=record["sample_id"],
        dog_boxes=boxes,
        face_boxes=(),
        nose_boxes=(),
        body_keypoints=point_sets,
        face_landmarks=(),
        model_name=model["name"],
        model_family=model["family"],
        inference_ms=record["inference_ms"],
        metadata={"artifact_sha256": model["artifact_sha256"]},
    )


def _validate_model(model: Any) -> None:
    if not isinstance(model, dict) or set(model) != _MODEL_FIELDS:
        raise ValueError("prediction cache model schema differs")
    for field in ("family", "name", "license_id", "device"):
        if not isinstance(model[field], str) or not model[field]:
            raise ValueError(
                f"prediction cache model {field} must be a non-empty string"
            )
    _validate_sha256(model["artifact_sha256"], "prediction cache model artifact")
    size = model["artifact_size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("prediction cache model artifact_size_bytes must be positive")


def _validate_keypoint_set(raw_set: Any) -> None:
    if not isinstance(raw_set, dict) or set(raw_set) != _KEYPOINT_SET_FIELDS:
        raise ValueError("prediction cache keypoint set schema differs")
    if not isinstance(raw_set["schema"], str) or not raw_set["schema"]:
        raise ValueError("prediction cache keypoint schema must be a non-empty string")
    points = raw_set["points"]
    if not isinstance(points, dict) or not points:
        raise ValueError("prediction cache keypoint points must be a non-empty object")
    for name, coordinates in points.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "prediction cache keypoint names must be non-empty strings"
            )
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 3
            or not all(_is_finite_number(value) for value in coordinates)
        ):
            raise ValueError(
                "prediction cache keypoints must contain three finite numbers"
            )
        if not 0.0 <= coordinates[2] <= 1.0:
            raise ValueError("prediction cache keypoint confidence must be in [0, 1]")


def _validate_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} SHA256 must be a lowercase digest")


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and bool(np.isfinite(value))
    )


__all__ = [
    "build_prediction_cache",
    "cache_record_to_result",
    "read_prediction_cache",
    "validate_prediction_cache",
    "write_prediction_cache",
]
