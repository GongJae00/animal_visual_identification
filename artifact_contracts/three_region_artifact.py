"""Fail-closed artifact contract for Appearance, Face, and Nose evidence."""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from foundation.protected_io import read_strict_json_document
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file

BUNDLE_SCHEMA = "cvi.three_region_artifact_bundle.v1"
MANIFEST_SCHEMA = "cvi.three_region_artifact_manifest.v1"
INTERPRETATION = (
    "REGION_EVIDENCE_ARTIFACT_NOT_SEGMENTATION_OR_BIOMETRIC_VALIDATION"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REGIONS = ("A", "F", "N")
_MASK_QUALIFICATIONS = {
    "VERIFIED_SEMANTIC",
    "MODEL_GENERATED_CANDIDATE",
    "GEOMETRIC_PROXY",
    "SOURCE_VALIDITY",
}
_GEOMETRY_QUALIFICATIONS = {
    "VERIFIED_ANNOTATION",
    "MODEL_GENERATED_CANDIDATE",
    "GEOMETRIC_PROXY",
}
_UNAVAILABLE_REASONS = {
    "AMBIGUOUS_MASKS",
    "GENERATOR_NOT_CONFIGURED",
    "HUMAN_REVIEW_ABSENT",
    "HUMAN_REVIEW_REJECTED",
    "INSUFFICIENT_LANDMARKS",
    "NATIVE_SOURCE_GEOMETRY_UNAVAILABLE",
    "NO_ELIGIBLE_MASK",
    "NO_ROI",
    "SCHEMA_INCOMPLETE",
}
_REGION_FIELDS = {
    "A": {"semantic_mask", "source_validity_mask", "skeleton"},
    "F": {
        "semantic_mask",
        "source_validity_mask",
        "proposal_box",
        "landmarks",
    },
    "N": {
        "semantic_mask",
        "source_validity_mask",
        "proposal_box",
        "native_geometry",
    },
}
_MASK_TARGETS = {"A": "FULL_BODY_DOG", "F": "EARS_FACE_NECK", "N": "NOSE"}
_VERIFIED_CLASS_MAPS = {
    "FULL_BODY_DOG": {"0": "background", "1": "dog"},
    "EARS_FACE_NECK": {
        "0": "background",
        "1": "ears",
        "2": "face",
        "3": "neck",
    },
    "NOSE": {"0": "context", "1": "nasal_surface", "2": "nostril"},
}
_REQUIRED_VERIFIED_VALUES = {
    "FULL_BODY_DOG": {1},
    "EARS_FACE_NECK": {1, 2, 3},
    "NOSE": {1},
}
_MAX_MASK_BYTES = 67_108_864
_MAX_MASK_PIXELS = 33_554_432


def unavailable(reason: str) -> dict[str, str]:
    """Create an explicit unavailable evidence slot."""

    if reason not in _UNAVAILABLE_REASONS:
        raise ValueError("three-region unavailable reason differs")
    return {"state": "UNAVAILABLE", "reason": reason}


def completion_for_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Derive completion from evidence; callers cannot assert it independently."""

    regions = record.get("regions")
    if not isinstance(regions, Mapping):
        raise TypeError("three-region record regions must be an object")
    missing: list[str] = []
    if not _is_verified_mask(regions.get("A", {}).get("semantic_mask")):
        missing.append("A.VERIFIED_SEMANTIC_MASK")
    skeleton = regions.get("A", {}).get("skeleton")
    if not _is_available_geometry(skeleton) or str(skeleton.get("schema", "")).startswith(
        "UNBOUND_"
    ):
        missing.append("A.SCHEMA_BOUND_SKELETON")
    if not _is_verified_mask(regions.get("F", {}).get("semantic_mask")):
        missing.append("F.VERIFIED_SEMANTIC_MASK")
    landmarks = regions.get("F", {}).get("landmarks")
    if not _has_face_coverage(landmarks):
        missing.append("F.EAR_FACE_NECK_LANDMARKS")
    if not _is_verified_mask(regions.get("N", {}).get("semantic_mask")):
        missing.append("N.VERIFIED_SEMANTIC_MASK")
    native_geometry = regions.get("N", {}).get("native_geometry")
    if not _is_available_geometry(native_geometry):
        missing.append("N.NATIVE_GEOMETRY")
    return {
        "state": "COMPLETE" if not missing else "INCOMPLETE_REQUIRED_EVIDENCE",
        "missing": missing,
    }


def validate_three_region_artifact_bundle(
    bundle: object, *, root: Path
) -> dict[str, Any]:
    """Validate one already-read bundle and every referenced image artifact."""

    expected = {"schema_version", "manifest_sha256", "manifest"}
    if (
        not isinstance(bundle, dict)
        or set(bundle) != expected
        or bundle["schema_version"] != BUNDLE_SCHEMA
    ):
        raise ValueError("three-region artifact bundle schema differs")
    _require_sha256(bundle["manifest_sha256"], "manifest_sha256")
    manifest = bundle["manifest"]
    if not isinstance(manifest, dict) or content_sha256(manifest) != bundle["manifest_sha256"]:
        raise ValueError("three-region artifact bundle digest differs")
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("three-region artifact root must be a directory")
    _validate_manifest(manifest, resolved_root)
    return manifest


def read_three_region_artifact(path: Path) -> dict[str, Any]:
    document = read_strict_json_document(
        path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    return validate_three_region_artifact_bundle(document.payload, root=path.parent)


def _validate_manifest(manifest: dict[str, Any], root: Path) -> None:
    expected = {
        "schema_version",
        "interpretation",
        "input_bindings",
        "dataset_name",
        "dataset_version",
        "records",
    }
    if set(manifest) != expected or manifest["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError("three-region artifact manifest schema differs")
    if manifest["interpretation"] != INTERPRETATION:
        raise ValueError("three-region artifact interpretation differs")
    for field in ("dataset_name", "dataset_version"):
        _require_text(manifest[field], field)
    bindings = manifest["input_bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {
        "roi_bundle_raw_sha256",
        "roi_manifest_sha256",
        "prediction_cache_sha256s",
    }:
        raise ValueError("three-region input bindings schema differs")
    _require_sha256(bindings["roi_bundle_raw_sha256"], "ROI bundle raw SHA-256")
    _require_sha256(bindings["roi_manifest_sha256"], "ROI manifest SHA-256")
    cache_hashes = bindings["prediction_cache_sha256s"]
    if not isinstance(cache_hashes, list) or not cache_hashes:
        raise ValueError("prediction cache SHA-256s must be a non-empty list")
    for digest in cache_hashes:
        _require_sha256(digest, "prediction cache SHA-256")
    if cache_hashes != sorted(cache_hashes) or len(cache_hashes) != len(set(cache_hashes)):
        raise ValueError("prediction cache SHA-256s must be sorted and unique")
    records = manifest["records"]
    if not isinstance(records, list):
        raise TypeError("three-region records must be a list")
    identities: list[tuple[str, str]] = []
    for record in records:
        _validate_record(record, root, manifest)
        identities.append((record["sample_id"], record["instance_id"]))
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("three-region records must be ordered and unique")


def _validate_record(
    record: object, root: Path, manifest: Mapping[str, Any]
) -> None:
    if not isinstance(record, dict) or set(record) != {
        "sample_id",
        "instance_id",
        "registered_identity_id",
        "source",
        "regions",
        "completion",
    }:
        raise ValueError("three-region record schema differs")
    _require_text(record["sample_id"], "sample_id")
    instance_id = _require_text(record["instance_id"], "instance_id")
    if len(instance_id) != 32 or any(character not in "0123456789abcdef" for character in instance_id):
        raise ValueError("instance_id must be 32 lowercase hexadecimal characters")
    if record["registered_identity_id"] is not None:
        _require_uuid5(record["registered_identity_id"], "registered_identity_id")
    source = record["source"]
    if not isinstance(source, dict) or set(source) != {
        "dataset_name",
        "dataset_version",
        "image_path",
        "image_sha256",
        "width",
        "height",
        "coordinate_space",
    }:
        raise ValueError("three-region source schema differs")
    if (
        source["dataset_name"] != manifest["dataset_name"]
        or source["dataset_version"] != manifest["dataset_version"]
    ):
        raise ValueError("three-region source dataset binding differs")
    _require_relative_path(source["image_path"], "source image_path")
    _require_sha256(source["image_sha256"], "source image SHA-256")
    width = _require_positive_integer(source["width"], "source width")
    height = _require_positive_integer(source["height"], "source height")
    if source["coordinate_space"] != "SOURCE_IMAGE_PIXELS":
        raise ValueError("source coordinate space differs")
    regions = record["regions"]
    if not isinstance(regions, dict) or set(regions) != set(_REGIONS):
        raise ValueError("three-region region schema differs")
    for region_name in _REGIONS:
        region = regions[region_name]
        if not isinstance(region, dict) or set(region) != _REGION_FIELDS[region_name]:
            raise ValueError(f"three-region {region_name} schema differs")
        _validate_mask(
            region["semantic_mask"],
            root,
            expected_target=_MASK_TARGETS[region_name],
            semantic_slot=True,
            source_width=width,
            source_height=height,
            sample_id=record["sample_id"],
            instance_id=record["instance_id"],
            source_image_sha256=source["image_sha256"],
        )
        _validate_mask(
            region["source_validity_mask"],
            root,
            expected_target="SOURCE_VALIDITY",
            semantic_slot=False,
            source_width=width,
            source_height=height,
            sample_id=record["sample_id"],
            instance_id=record["instance_id"],
            source_image_sha256=source["image_sha256"],
        )
        semantic = region["semantic_mask"]
        validity = region["source_validity_mask"]
        if (
            semantic.get("state") == validity.get("state") == "AVAILABLE"
            and semantic["artifact"]["sha256"] == validity["artifact"]["sha256"]
        ):
            raise ValueError("semantic and source-validity masks must be distinct artifacts")
    semantic_digests = [
        regions[region]["semantic_mask"]["artifact"]["sha256"]
        for region in _REGIONS
        if regions[region]["semantic_mask"].get("state") == "AVAILABLE"
    ]
    if len(semantic_digests) != len(set(semantic_digests)):
        raise ValueError("A/F/N semantic masks must be distinct artifacts")
    _validate_geometry(regions["A"]["skeleton"], width=width, height=height, expected_kind="SKELETON")
    _validate_geometry(regions["F"]["proposal_box"], width=width, height=height, expected_kind="BOUNDING_BOX")
    _validate_geometry(regions["F"]["landmarks"], width=width, height=height, expected_kind="LANDMARKS")
    _validate_geometry(regions["N"]["proposal_box"], width=width, height=height, expected_kind="BOUNDING_BOX")
    _validate_geometry(regions["N"]["native_geometry"], width=width, height=height, expected_kind="NATIVE_NOSE_GEOMETRY")
    if record["completion"] != completion_for_record(record):
        raise ValueError("three-region completion differs from available evidence")


def _validate_mask(
    value: object,
    root: Path,
    *,
    expected_target: str,
    semantic_slot: bool,
    source_width: int,
    source_height: int,
    sample_id: str,
    instance_id: str,
    source_image_sha256: str,
) -> None:
    if _validate_unavailable(value):
        return
    expected = {
        "state",
        "qualification",
        "semantic_target",
        "artifact",
        "class_map",
        "producer_reference_sha256",
        "review_reference_sha256",
        "pixel_verification_reference_sha256",
        "verification_binding_sha256",
        "review_receipt_path",
        "review_receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected or value["state"] != "AVAILABLE":
        raise ValueError("three-region mask schema differs")
    qualification = value["qualification"]
    if qualification not in _MASK_QUALIFICATIONS:
        raise ValueError("three-region mask qualification differs")
    if value["semantic_target"] != expected_target:
        raise ValueError("three-region mask semantic target differs")
    if semantic_slot and qualification == "SOURCE_VALIDITY":
        raise ValueError("source-validity mask cannot occupy a semantic-mask slot")
    if not semantic_slot and qualification != "SOURCE_VALIDITY":
        raise ValueError("source-validity slot requires SOURCE_VALIDITY qualification")
    _require_sha256(value["producer_reference_sha256"], "mask producer reference")
    review = value["review_reference_sha256"]
    pixel_receipt = value["pixel_verification_reference_sha256"]
    verification_binding = value["verification_binding_sha256"]
    review_receipt_path = value["review_receipt_path"]
    review_receipt_sha256 = value["review_receipt_sha256"]
    if qualification == "VERIFIED_SEMANTIC":
        _require_sha256(review, "mask review reference")
        _require_sha256(pixel_receipt, "mask pixel-verification reference")
    elif review is not None:
        raise ValueError("unverified mask must not carry a review reference")
    elif pixel_receipt is not None:
        _require_sha256(pixel_receipt, "mask pixel-verification reference")
    class_map = value["class_map"]
    if not isinstance(class_map, dict) or not class_map:
        raise ValueError("three-region mask class_map must be non-empty")
    class_values: set[int] = set()
    for raw_value, name in class_map.items():
        if not isinstance(raw_value, str) or not raw_value.isdigit():
            raise ValueError("three-region mask class values must be decimal strings")
        class_value = int(raw_value)
        if not 0 <= class_value <= 255 or class_value in class_values:
            raise ValueError("three-region mask class values must be unique uint8 values")
        class_values.add(class_value)
        _require_text(name, "mask class name")
    observed = _validate_image_artifact(
        value["artifact"],
        root,
        source_width=source_width,
        source_height=source_height,
    )
    if observed - class_values:
        raise ValueError("mask artifact contains values outside class_map")
    if qualification == "VERIFIED_SEMANTIC":
        if class_map != _VERIFIED_CLASS_MAPS[expected_target]:
            raise ValueError("verified semantic mask class_map differs from target")
        if not _REQUIRED_VERIFIED_VALUES[expected_target].issubset(observed):
            raise ValueError("verified semantic mask lacks required foreground classes")
        receipt_digest = _validate_review_receipt(
            root,
            relative_path=review_receipt_path,
            expected_sha256=review_receipt_sha256,
            sample_id=sample_id,
            instance_id=instance_id,
            source_image_sha256=source_image_sha256,
            semantic_target=expected_target,
            artifact=value["artifact"],
            class_map=class_map,
            review_reference_sha256=review,
            pixel_verification_reference_sha256=pixel_receipt,
        )
        expected_binding = content_sha256(
            {
                "sample_id": sample_id,
                "instance_id": instance_id,
                "source_image_sha256": source_image_sha256,
                "semantic_target": expected_target,
                "artifact": {
                    key: value["artifact"][key]
                    for key in (
                        "sha256",
                        "width",
                        "height",
                        "coordinate_space",
                        "source_mapping",
                    )
                },
                "class_map": class_map,
                "review_reference_sha256": review,
                "pixel_verification_reference_sha256": pixel_receipt,
                "review_receipt_sha256": receipt_digest,
            }
        )
        if verification_binding != expected_binding:
            raise ValueError("semantic mask verification binding differs")
    elif any(
        item is not None
        for item in (
            verification_binding,
            review_receipt_path,
            review_receipt_sha256,
        )
    ):
        raise ValueError("unverified mask must not carry verification evidence")
    if qualification == "SOURCE_VALIDITY":
        _validate_source_validity_pixels(
            value["artifact"], root, source_width=source_width, source_height=source_height
        )


def _validate_image_artifact(
    value: object,
    root: Path,
    *,
    source_width: int,
    source_height: int,
) -> set[int]:
    expected = {
        "relative_path",
        "sha256",
        "byte_size",
        "width",
        "height",
        "encoding",
        "pixel_mode",
        "coordinate_space",
        "observed_pixel_values",
        "source_mapping",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("three-region image artifact schema differs")
    relative = _require_relative_path(value["relative_path"], "artifact relative_path")
    digest = _require_sha256(value["sha256"], "artifact SHA-256")
    byte_size = _require_positive_integer(value["byte_size"], "artifact byte_size")
    width = _require_positive_integer(value["width"], "artifact width")
    height = _require_positive_integer(value["height"], "artifact height")
    if width * height > _MAX_MASK_PIXELS:
        raise ValueError("mask artifact pixel count exceeds policy")
    if value["encoding"] != "PNG" or value["pixel_mode"] != "L":
        raise ValueError("mask artifact must declare PNG encoding and L mode")
    if value["coordinate_space"] not in {"SOURCE_IMAGE_PIXELS", "CROP_PIXELS"}:
        raise ValueError("mask artifact coordinate space differs")
    if value["coordinate_space"] == "SOURCE_IMAGE_PIXELS":
        if value["source_mapping"] is not None or (width, height) != (
            source_width,
            source_height,
        ):
            raise ValueError("source-aligned mask geometry differs")
    else:
        _validate_source_mapping(
            value["source_mapping"],
            source_width=source_width,
            source_height=source_height,
        )
    observed = value["observed_pixel_values"]
    if (
        not isinstance(observed, list)
        or not observed
        or observed != sorted(set(observed))
        or any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255 for item in observed)
    ):
        raise ValueError("observed mask pixel values must be sorted unique uint8 values")
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("mask artifact does not exist or is unsafe") from exc
    if (
        not resolved.is_relative_to(root)
        or resolved.relative_to(root).as_posix() != relative.as_posix()
        or not resolved.is_file()
    ):
        raise ValueError("mask artifact must be a regular file under the artifact root")
    retained = read_retained_regular_file(
        resolved,
        expected_bytes=byte_size,
        expected_sha256=digest,
        maximum_bytes=_MAX_MASK_BYTES,
        capture_payload=True,
        subject="three-region mask artifact",
    )
    payload = retained.payload
    assert payload is not None
    try:
        with Image.open(BytesIO(payload)) as opened:
            if opened.format != "PNG" or opened.mode != "L" or opened.size != (width, height):
                raise ValueError("mask artifact image metadata differs")
            actual = sorted(int(item) for item in np.unique(np.asarray(opened, dtype=np.uint8)))
    except (OSError, SyntaxError) as exc:
        raise ValueError("mask artifact is not a valid image") from exc
    if actual != observed:
        raise ValueError("observed mask pixel values differ from artifact")
    return set(actual)


def _validate_source_mapping(
    value: object, *, source_width: int, source_height: int
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "source_crop_rect_xyxy",
        "square_side",
        "offset_xy",
        "resize_interpolation",
    }:
        raise ValueError("crop mask source mapping schema differs")
    rect = value["source_crop_rect_xyxy"]
    if (
        not isinstance(rect, list)
        or len(rect) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in rect)
    ):
        raise ValueError("crop mask source rectangle must contain four integers")
    x1, y1, x2, y2 = rect
    if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
        raise ValueError("crop mask source rectangle differs")
    side = max(x2 - x1, y2 - y1)
    if value["square_side"] != side:
        raise ValueError("crop mask square side differs")
    expected_offset = [(side - (x2 - x1)) // 2, (side - (y2 - y1)) // 2]
    if value["offset_xy"] != expected_offset:
        raise ValueError("crop mask padding offset differs")
    if value["resize_interpolation"] != "NEAREST":
        raise ValueError("crop mask resize interpolation differs")


def _validate_source_validity_pixels(
    artifact: Mapping[str, Any],
    root: Path,
    *,
    source_width: int,
    source_height: int,
) -> None:
    if artifact["coordinate_space"] != "CROP_PIXELS" or artifact["width"] != artifact["height"]:
        raise ValueError("source-validity mask must be a square crop mask")
    mapping = artifact["source_mapping"]
    _validate_source_mapping(
        mapping, source_width=source_width, source_height=source_height
    )
    x1, y1, x2, y2 = mapping["source_crop_rect_xyxy"]
    offset_x, offset_y = mapping["offset_xy"]
    side = mapping["square_side"]
    source_x = np.minimum(
        side - 1,
        np.floor(
            (np.arange(artifact["width"], dtype=np.float64) + 0.5)
            * side
            / artifact["width"]
        ).astype(np.int64),
    )
    source_y = np.minimum(
        side - 1,
        np.floor(
            (np.arange(artifact["height"], dtype=np.float64) + 0.5)
            * side
            / artifact["height"]
        ).astype(np.int64),
    )
    valid_x = (source_x >= offset_x) & (source_x < offset_x + (x2 - x1))
    valid_y = (source_y >= offset_y) & (source_y < offset_y + (y2 - y1))
    expected = (valid_y[:, None] & valid_x[None, :]).astype(np.uint8) * 255
    relative = _require_relative_path(artifact["relative_path"], "artifact relative_path")
    retained = read_retained_regular_file(
        root.joinpath(*relative.parts).resolve(strict=True),
        expected_bytes=artifact["byte_size"],
        expected_sha256=artifact["sha256"],
        maximum_bytes=_MAX_MASK_BYTES,
        capture_payload=True,
        subject="three-region source-validity mask",
    )
    assert retained.payload is not None
    with Image.open(BytesIO(retained.payload)) as opened:
        actual = np.asarray(opened, dtype=np.uint8)
    if not np.array_equal(actual, expected):
        raise ValueError("source-validity mask pixels differ from source mapping")


def _validate_review_receipt(
    root: Path,
    *,
    relative_path: object,
    expected_sha256: object,
    sample_id: str,
    instance_id: str,
    source_image_sha256: str,
    semantic_target: str,
    artifact: Mapping[str, Any],
    class_map: Mapping[str, str],
    review_reference_sha256: str,
    pixel_verification_reference_sha256: str,
) -> str:
    relative = _require_relative_path(relative_path, "review receipt path")
    digest = _require_sha256(expected_sha256, "review receipt SHA-256")
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("review receipt does not exist or is unsafe") from exc
    if (
        not resolved.is_relative_to(root)
        or resolved.relative_to(root).as_posix() != relative.as_posix()
        or not resolved.is_file()
    ):
        raise ValueError("review receipt must be a regular file under artifact root")
    document = read_strict_json_document(resolved, maximum_bytes=1_048_576)
    if document.raw_sha256 != digest:
        raise ValueError("review receipt byte digest differs")
    receipt = document.payload
    expected = {
        "schema_version": "cvi.semantic_mask_review_receipt.v1",
        "decision": "VERIFIED",
        "sample_id": sample_id,
        "instance_id": instance_id,
        "source_image_sha256": source_image_sha256,
        "semantic_target": semantic_target,
        "mask_artifact_sha256": artifact["sha256"],
        "mask_width": artifact["width"],
        "mask_height": artifact["height"],
        "coordinate_space": artifact["coordinate_space"],
        "source_mapping_sha256": content_sha256(artifact["source_mapping"]),
        "class_map_sha256": content_sha256(class_map),
        "review_reference_sha256": review_reference_sha256,
        "pixel_verification_reference_sha256": pixel_verification_reference_sha256,
    }
    if receipt != expected:
        raise ValueError("semantic mask review receipt binding differs")
    return digest


def _validate_geometry(
    value: object, *, width: int, height: int, expected_kind: str
) -> None:
    if _validate_unavailable(value):
        return
    expected = {
        "state",
        "qualification",
        "kind",
        "schema",
        "coordinate_space",
        "payload",
        "producer_reference_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected or value["state"] != "AVAILABLE":
        raise ValueError("three-region geometry schema differs")
    if value["qualification"] not in _GEOMETRY_QUALIFICATIONS:
        raise ValueError("three-region geometry qualification differs")
    if value["kind"] != expected_kind:
        raise ValueError("three-region geometry kind differs")
    _require_text(value["schema"], "geometry schema")
    if value["coordinate_space"] != "SOURCE_IMAGE_PIXELS":
        raise ValueError("geometry coordinate space differs")
    _require_sha256(value["producer_reference_sha256"], "geometry producer reference")
    payload = value["payload"]
    if expected_kind == "BOUNDING_BOX":
        if not isinstance(payload, dict) or set(payload) != {"xyxy"}:
            raise ValueError("bounding-box payload schema differs")
        _validate_box(payload["xyxy"], width=width, height=height, integers=False)
    elif expected_kind == "SKELETON":
        _validate_points_and_edges(payload, width=width, height=height)
    elif expected_kind == "LANDMARKS":
        if not isinstance(payload, dict) or set(payload) != {"points", "coverage"}:
            raise ValueError("landmark payload schema differs")
        _validate_points(payload["points"], width=width, height=height)
        coverage = payload["coverage"]
        if (
            not isinstance(coverage, list)
            or coverage != sorted(set(coverage))
            or any(item not in {"EARS", "FACE", "NECK"} for item in coverage)
        ):
            raise ValueError("landmark coverage differs")
    else:
        if not isinstance(payload, dict) or set(payload) != {
            "source_box_xyxy",
            "crop_width",
            "crop_height",
            "keypoints",
        }:
            raise ValueError("native Nose geometry payload schema differs")
        x1, y1, x2, y2 = _validate_box(
            payload["source_box_xyxy"], width=width, height=height, integers=True
        )
        if (
            _require_positive_integer(payload["crop_width"], "native crop width") != x2 - x1
            or _require_positive_integer(payload["crop_height"], "native crop height") != y2 - y1
        ):
            raise ValueError("native Nose crop dimensions differ from source box")
        _validate_points(payload["keypoints"], width=width, height=height)


def _validate_points_and_edges(value: object, *, width: int, height: int) -> None:
    if not isinstance(value, dict) or set(value) != {"keypoints", "edges"}:
        raise ValueError("skeleton payload schema differs")
    names = _validate_points(value["keypoints"], width=width, height=height)
    edges = value["edges"]
    if not isinstance(edges, list) or len(edges) != len({tuple(edge) for edge in edges if isinstance(edge, list)}):
        raise ValueError("skeleton edges must be a unique list")
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or edge[0] not in names or edge[1] not in names or edge[0] == edge[1]:
            raise ValueError("skeleton edge must reference two distinct keypoints")


def _validate_points(value: object, *, width: int, height: int) -> set[str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("geometry points must be a non-empty object")
    for name, point in value.items():
        _require_text(name, "geometry point name")
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError("geometry points must contain [x, y, confidence]")
        x = _require_number(point[0], "geometry point x")
        y = _require_number(point[1], "geometry point y")
        confidence = _require_number(point[2], "geometry point confidence")
        if not 0.0 <= x <= width or not 0.0 <= y <= height or not 0.0 <= confidence <= 1.0:
            raise ValueError("geometry point lies outside its valid range")
    return set(value)


def _validate_box(
    value: object, *, width: int, height: int, integers: bool
) -> tuple[Any, Any, Any, Any]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("geometry box must contain four coordinates")
    if integers and any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("native geometry box must contain integers")
    coordinates = tuple(_require_number(item, "geometry box coordinate") for item in value)
    x1, y1, x2, y2 = coordinates
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError("geometry box must be non-empty and within the source image")
    return coordinates


def _validate_unavailable(value: object) -> bool:
    if not isinstance(value, dict) or value.get("state") != "UNAVAILABLE":
        return False
    if set(value) != {"state", "reason"} or value["reason"] not in _UNAVAILABLE_REASONS:
        raise ValueError("unavailable evidence schema differs")
    return True


def _is_verified_mask(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("state") == "AVAILABLE" and value.get("qualification") == "VERIFIED_SEMANTIC"


def _is_available_geometry(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("state") == "AVAILABLE"


def _has_face_coverage(value: object) -> bool:
    return _is_available_geometry(value) and value.get("kind") == "LANDMARKS" and set(value.get("payload", {}).get("coverage", ())) == {"EARS", "FACE", "NECK"}


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_uuid5(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a canonical UUIDv5")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical UUIDv5") from exc
    if parsed.version != 5 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUIDv5")
    return value


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _require_relative_path(value: object, name: str) -> PurePosixPath:
    text = _require_text(value, name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or path.as_posix() != text
        or "\\" in text
        or path.parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a canonical safe relative path")
    return path


__all__ = [
    "BUNDLE_SCHEMA",
    "INTERPRETATION",
    "MANIFEST_SCHEMA",
    "completion_for_record",
    "read_three_region_artifact",
    "unavailable",
    "validate_three_region_artifact_bundle",
]
