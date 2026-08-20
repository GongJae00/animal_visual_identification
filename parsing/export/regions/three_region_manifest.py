"""Classify existing ROI evidence into the fail-closed A/F/N contract."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from shared.contracts.three_region_artifact import (
    BUNDLE_SCHEMA,
    INTERPRETATION,
    MANIFEST_SCHEMA,
    completion_for_record,
    unavailable,
    validate_three_region_artifact_bundle,
)
from shared.foundation.provenance import content_sha256
from shared.foundation.retained_file import read_retained_regular_file
from parsing.export.types import AP10K_BODY_17_EDGES


def build_three_region_artifact_bundle(
    roi_manifest: Mapping[str, Any],
    *,
    root: Path,
    roi_bundle_raw_sha256: str,
    roi_manifest_sha256: str,
) -> dict[str, Any]:
    """Build an evidence inventory without upgrading proposals into semantics."""

    resolved_root = root.resolve(strict=True)
    records = [
        _build_record(record, root=resolved_root, producer_sha256=roi_manifest_sha256)
        for record in roi_manifest["records"]
    ]
    records.sort(key=lambda item: (item["sample_id"], item["instance_id"]))
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "interpretation": INTERPRETATION,
        "input_bindings": {
            "roi_bundle_raw_sha256": roi_bundle_raw_sha256,
            "roi_manifest_sha256": roi_manifest_sha256,
            "prediction_cache_sha256s": sorted(
                set(roi_manifest["prediction_cache_sha256s"])
            ),
        },
        "dataset_name": roi_manifest["dataset_name"],
        "dataset_version": roi_manifest["dataset_version"],
        "records": records,
    }
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "manifest_sha256": content_sha256(manifest),
        "manifest": manifest,
    }
    validate_three_region_artifact_bundle(bundle, root=resolved_root)
    return bundle


def _build_record(
    record: Mapping[str, Any], *, root: Path, producer_sha256: str
) -> dict[str, Any]:
    body_points = record["body_keypoints"]
    if body_points is None:
        skeleton = unavailable("INSUFFICIENT_LANDMARKS")
    else:
        names = set(body_points)
        skeleton = _geometry(
            qualification="MODEL_GENERATED_CANDIDATE",
            kind="SKELETON",
            schema="UNBOUND_ROI_MANIFEST_V2_BODY_KEYPOINTS",
            payload={
                "keypoints": body_points,
                "edges": [
                    [first, second]
                    for first, second in AP10K_BODY_17_EDGES
                    if first in names and second in names
                ],
            },
            producer_sha256=producer_sha256,
        )
    face_box = record["face_roi_xyxy"]
    nose_box = record["nose_roi_xyxy"]
    regions = {
        "A": {
            "semantic_mask": unavailable("GENERATOR_NOT_CONFIGURED"),
            "source_validity_mask": _source_validity_mask(
                root,
                path=record["source_valid_mask_path"],
                digest=record["source_valid_mask_sha256"],
                producer_sha256=producer_sha256,
                source_crop_rect=record["crop_rect_xyxy"],
            ),
            "skeleton": skeleton,
        },
        "F": {
            "semantic_mask": unavailable("GENERATOR_NOT_CONFIGURED"),
            "source_validity_mask": (
                unavailable("NO_ROI")
                if record["face_source_valid_mask_path"] is None
                else _source_validity_mask(
                    root,
                    path=record["face_source_valid_mask_path"],
                    digest=record["face_source_valid_mask_sha256"],
                    producer_sha256=producer_sha256,
                    source_crop_rect=record["face_crop_rect_xyxy"],
                )
            ),
            "proposal_box": (
                unavailable("NO_ROI")
                if face_box is None
                else _box_geometry(face_box, producer_sha256=producer_sha256)
            ),
            "landmarks": unavailable("SCHEMA_INCOMPLETE"),
        },
        "N": {
            "semantic_mask": unavailable("GENERATOR_NOT_CONFIGURED"),
            "source_validity_mask": (
                unavailable("NO_ROI")
                if record["weak_nose_source_valid_mask_path"] is None
                else unavailable("SCHEMA_INCOMPLETE")
            ),
            "proposal_box": (
                unavailable("NO_ROI")
                if nose_box is None
                else _box_geometry(nose_box, producer_sha256=producer_sha256)
            ),
            "native_geometry": unavailable("NATIVE_SOURCE_GEOMETRY_UNAVAILABLE"),
        },
    }
    output = {
        "sample_id": record["sample_id"],
        "instance_id": record["instance_id"],
        "registered_identity_id": record["registered_identity_id"],
        "source": {
            "dataset_name": record["dataset_name"],
            "dataset_version": record["dataset_version"],
            "image_path": record["image_path"],
            "image_sha256": record["image_sha256"],
            "width": record["image_width"],
            "height": record["image_height"],
            "coordinate_space": "SOURCE_IMAGE_PIXELS",
        },
        "regions": regions,
        "completion": {},
    }
    output["completion"] = completion_for_record(output)
    return output


def _source_validity_mask(
    root: Path,
    *,
    path: str,
    digest: str,
    producer_sha256: str,
    source_crop_rect: list[int],
) -> dict[str, Any]:
    pure_path = PurePosixPath(path)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or pure_path.as_posix() != path
        or "\\" in path
        or pure_path.parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise ValueError("ROI source-validity mask path is unsafe")
    try:
        artifact_path = root.joinpath(*pure_path.parts).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("ROI source-validity mask does not exist or is unsafe") from exc
    if (
        not artifact_path.is_relative_to(root)
        or artifact_path.relative_to(root).as_posix() != pure_path.as_posix()
        or not artifact_path.is_file()
    ):
        raise ValueError("ROI source-validity mask must be a regular file under root")
    retained = read_retained_regular_file(
        artifact_path,
        expected_sha256=digest,
        maximum_bytes=67_108_864,
        capture_payload=True,
        subject="ROI source-validity mask",
    )
    payload = retained.payload
    assert payload is not None
    with Image.open(BytesIO(payload)) as opened:
        if opened.format != "PNG" or opened.mode != "L":
            raise ValueError("ROI source-validity mask must be an L-mode PNG")
        width, height = opened.size
        if width * height > 33_554_432:
            raise ValueError("ROI source-validity mask pixel count exceeds policy")
        if width != height:
            raise ValueError("ROI source-validity mask must be square")
        observed = sorted(
            int(value) for value in np.unique(np.asarray(opened, dtype=np.uint8))
        )
    if any(value not in {0, 255} for value in observed):
        raise ValueError("ROI source-validity mask must contain only 0 and 255")
    return {
        "state": "AVAILABLE",
        "qualification": "SOURCE_VALIDITY",
        "semantic_target": "SOURCE_VALIDITY",
        "artifact": {
            "relative_path": path,
            "sha256": digest,
            "byte_size": len(payload),
            "width": width,
            "height": height,
            "encoding": "PNG",
            "pixel_mode": "L",
            "coordinate_space": "CROP_PIXELS",
            "observed_pixel_values": observed,
            "source_mapping": _source_mapping(source_crop_rect),
        },
        "class_map": {"0": "padding", "255": "source_pixel"},
        "producer_reference_sha256": producer_sha256,
        "review_reference_sha256": None,
        "pixel_verification_reference_sha256": None,
        "verification_binding_sha256": None,
        "review_receipt_path": None,
        "review_receipt_sha256": None,
    }


def _source_mapping(source_crop_rect: list[int]) -> dict[str, Any]:
    x1, y1, x2, y2 = source_crop_rect
    width, height = x2 - x1, y2 - y1
    side = max(width, height)
    return {
        "source_crop_rect_xyxy": source_crop_rect,
        "square_side": side,
        "offset_xy": [(side - width) // 2, (side - height) // 2],
        "resize_interpolation": "NEAREST",
    }


def _box_geometry(
    coordinates: list[float], *, producer_sha256: str
) -> dict[str, Any]:
    return _geometry(
        qualification="GEOMETRIC_PROXY",
        kind="BOUNDING_BOX",
        schema="XYXY_FLOAT.v1",
        payload={"xyxy": coordinates},
        producer_sha256=producer_sha256,
    )


def _geometry(
    *,
    qualification: str,
    kind: str,
    schema: str,
    payload: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    return {
        "state": "AVAILABLE",
        "qualification": qualification,
        "kind": kind,
        "schema": schema,
        "coordinate_space": "SOURCE_IMAGE_PIXELS",
        "payload": payload,
        "producer_reference_sha256": producer_sha256,
    }


__all__ = ["build_three_region_artifact_bundle"]
