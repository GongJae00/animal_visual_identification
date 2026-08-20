"""Strict per-source Face and native Nose outcomes for SiBeTan diagnostics."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from shared.foundation.provenance import content_sha256


BUNDLE_SCHEMA = "cvi.sibetan_multievidence_bundle.v1"
MANIFEST_SCHEMA = "cvi.sibetan_multievidence_manifest.v1"
V2_BUNDLE_SCHEMA = "cvi.sibetan_multievidence_bundle.v2"
V2_MANIFEST_SCHEMA = "cvi.sibetan_multievidence_manifest.v2"
STATES = {"AVAILABLE", "UNAVAILABLE"}
REASONS = {
    "NO_TARGET_DOG_INSTANCE",
    "MULTIPLE_DOG_INSTANCES",
    "FACE_GEOMETRY_UNAVAILABLE",
    "LOW_LOCALIZATION_CONFIDENCE",
    "LOW_FRONTALITY",
    "LOW_NATIVE_RESOLUTION",
}


def build_evidence_bundle(
    *,
    records: Sequence[Mapping[str, Any]],
    input_bindings: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [dict(row) for row in records]
    if rows != sorted(rows, key=lambda row: row["sample_id"]):
        raise ValueError("SiBeTan evidence records must be sorted by sample_id")
    counts = {
        branch: dict(sorted(Counter(row[branch]["state"] for row in rows).items()))
        for branch in ("face", "nose")
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_name": "sibetan",
        "input_bindings": dict(input_bindings),
        "policy": dict(policy),
        "records": rows,
        "record_count": len(rows),
        "state_counts": counts,
        "interpretation": (
            "EXPOSED_RESEARCH_EVIDENCE_MATERIALIZATION_NOT_FINAL_OR_BIOMETRIC_VALIDATION"
        ),
    }
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "manifest_sha256": content_sha256(manifest),
        "manifest": manifest,
    }
    validate_evidence_bundle(bundle)
    return bundle


def validate_evidence_bundle(
    bundle: object,
    *,
    nose_root: Path | None = None,
    face_root: Path | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"schema_version", "manifest_sha256", "manifest"}
        or bundle["schema_version"] != BUNDLE_SCHEMA
    ):
        raise ValueError("SiBeTan evidence bundle schema differs")
    manifest = bundle["manifest"]
    expected = {
        "schema_version", "dataset_name", "input_bindings", "policy", "records",
        "record_count", "state_counts", "interpretation",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected
        or manifest["schema_version"] != MANIFEST_SCHEMA
        or manifest["dataset_name"] != "sibetan"
        or content_sha256(manifest) != bundle["manifest_sha256"]
    ):
        raise ValueError("SiBeTan evidence manifest schema or digest differs")
    policy = manifest["policy"]
    if policy != {
        "minimum_detector_confidence": 0.5,
        "minimum_frontality": 0.5,
        "minimum_native_short_side": 32,
        "face_margin": 1.15,
        "face_size": 224,
        "target_association": "EXACTLY_ONE_POSE_DOG_INSTANCE",
    }:
        raise ValueError("SiBeTan evidence policy differs")
    rows = manifest["records"]
    if (
        not isinstance(rows, list)
        or not rows
        or rows != sorted(rows, key=lambda row: row["sample_id"])
        or len({row["sample_id"] for row in rows}) != len(rows)
        or manifest["record_count"] != len(rows)
    ):
        raise ValueError("SiBeTan evidence record population differs")
    for row in rows:
        _validate_record(row, nose_root=nose_root, face_root=face_root)
    expected_counts = {
        branch: dict(sorted(Counter(row[branch]["state"] for row in rows).items()))
        for branch in ("face", "nose")
    }
    if manifest["state_counts"] != expected_counts:
        raise ValueError("SiBeTan evidence state counts differ")
    return manifest


def _validate_record(
    row: object, *, nose_root: Path | None, face_root: Path | None
) -> None:
    expected = {
        "sample_id", "registered_identity_id", "source_group_id", "image_path",
        "image_sha256", "source_width", "source_height", "face", "nose",
    }
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError("SiBeTan evidence record schema differs")
    for field in ("sample_id", "registered_identity_id", "source_group_id", "image_path"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"SiBeTan evidence {field} differs")
    _sha(row["image_sha256"], "image_sha256")
    for field in ("source_width", "source_height"):
        if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] <= 0:
            raise ValueError(f"SiBeTan evidence {field} differs")
    _validate_outcome(row["face"], "face", face_root)
    _validate_outcome(row["nose"], "nose", nose_root)


def _validate_outcome(value: object, branch: str, root: Path | None) -> None:
    expected = {"state", "reasons", "crop_path", "crop_sha256", "crop_width", "crop_height"}
    if not isinstance(value, dict) or set(value) != expected or value["state"] not in STATES:
        raise ValueError(f"SiBeTan {branch} outcome schema differs")
    reasons = value["reasons"]
    if not isinstance(reasons, list) or reasons != sorted(set(reasons)) or any(reason not in REASONS for reason in reasons):
        raise ValueError(f"SiBeTan {branch} reasons differ")
    available = value["state"] == "AVAILABLE"
    artifact_fields = ("crop_path", "crop_sha256", "crop_width", "crop_height")
    if available != (not reasons) or available != all(value[field] is not None for field in artifact_fields):
        raise ValueError(f"SiBeTan {branch} availability differs")
    if not available:
        if any(value[field] is not None for field in artifact_fields):
            raise ValueError(f"SiBeTan unavailable {branch} carries an artifact")
        return
    _sha(value["crop_sha256"], f"{branch}.crop_sha256")
    if any(isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] <= 0 for field in ("crop_width", "crop_height")):
        raise ValueError(f"SiBeTan {branch} crop dimensions differ")
    path = PurePosixPath(value["crop_path"])
    if path.is_absolute() or path.as_posix() != value["crop_path"] or ".." in path.parts:
        raise ValueError(f"SiBeTan {branch} crop path is unsafe")
    if root is not None:
        candidate = root.joinpath(*path.parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"SiBeTan {branch} crop is unavailable")
        import hashlib
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != value["crop_sha256"]:
            raise ValueError(f"SiBeTan {branch} crop hash differs")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase SHA-256")


__all__ = [
    "BUNDLE_SCHEMA",
    "MANIFEST_SCHEMA",
    "build_evidence_bundle",
    "validate_evidence_bundle",
    "build_evidence_bundle_v2",
    "validate_evidence_bundle_v2",
]


def build_evidence_bundle_v2(
    *, records: Sequence[Mapping[str, Any]], input_bindings: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [dict(row) for row in records]
    if rows != sorted(rows, key=lambda row: row["sample_id"]):
        raise ValueError("SiBeTan v2 evidence records must be sorted")
    state_counts = {
        branch: dict(sorted(Counter(row[branch]["state"] for row in rows).items()))
        for branch in ("face", "nose")
    }
    manifest = {
        "schema_version": V2_MANIFEST_SCHEMA,
        "dataset_name": "sibetan",
        "input_bindings": dict(input_bindings),
        "policy": dict(policy),
        "records": rows,
        "record_count": len(rows),
        "state_counts": state_counts,
        "interpretation": "SOURCE_COORDINATE_TWO_STAGE_RESEARCH_EVIDENCE_NOT_BIOMETRIC_VALIDATION",
    }
    bundle = {
        "schema_version": V2_BUNDLE_SCHEMA,
        "manifest_sha256": content_sha256(manifest),
        "manifest": manifest,
    }
    validate_evidence_bundle_v2(bundle)
    return bundle


def validate_evidence_bundle_v2(bundle: object, *, root: Path | None = None) -> dict[str, Any]:
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"schema_version", "manifest_sha256", "manifest"}
        or bundle["schema_version"] != V2_BUNDLE_SCHEMA
    ):
        raise ValueError("SiBeTan v2 evidence bundle schema differs")
    manifest = bundle["manifest"]
    expected = {
        "schema_version", "dataset_name", "input_bindings", "policy", "records",
        "record_count", "state_counts", "interpretation",
    }
    if (
        not isinstance(manifest, dict) or set(manifest) != expected
        or manifest["schema_version"] != V2_MANIFEST_SCHEMA
        or manifest["dataset_name"] != "sibetan"
        or content_sha256(manifest) != bundle["manifest_sha256"]
    ):
        raise ValueError("SiBeTan v2 evidence manifest differs")
    expected_policy = {
        "target_association": "EXACTLY_ONE_POSE_DOG_INSTANCE",
        "head_geometry": "ROI_MANIFEST_FACE_CROP_RECT_XYXY",
        "localizer_input": "RAW_SOURCE_RGB_HEAD_RECT",
        "localizer_resize": "BILINEAR_STRETCH_224X224",
        "nose_margin": 0.08,
        "minimum_localizer_confidence": 0.5,
        "frontality_admission": "NONE_CONTINUOUS_QUALITY",
        "native_short_side_admission": "NONE_CONTINUOUS_QUALITY",
        "crop_encoding": "PNG_RGB_LOSSLESS_FROM_DECODED_SOURCE",
    }
    if manifest["policy"] != expected_policy:
        raise ValueError("SiBeTan v2 evidence policy differs")
    rows = manifest["records"]
    if (
        not isinstance(rows, list) or not rows
        or rows != sorted(rows, key=lambda row: row["sample_id"])
        or len({row["sample_id"] for row in rows}) != len(rows)
        or len({row["image_path"] for row in rows}) != len(rows)
        or manifest["record_count"] != len(rows)
    ):
        raise ValueError("SiBeTan v2 evidence population differs")
    artifact_paths: set[str] = set()
    for row in rows:
        _validate_v2_record(row, root, artifact_paths)
    counts = {
        branch: dict(sorted(Counter(row[branch]["state"] for row in rows).items()))
        for branch in ("face", "nose")
    }
    if manifest["state_counts"] != counts:
        raise ValueError("SiBeTan v2 state counts differ")
    return manifest


def _validate_v2_record(row: object, root: Path | None, artifact_paths: set[str]) -> None:
    expected = {
        "sample_id", "registered_identity_id", "source_group_id", "image_path",
        "image_sha256", "source_width", "source_height", "face", "nose",
    }
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError("SiBeTan v2 evidence record differs")
    _sha(row["image_sha256"], "image_sha256")
    width, height = row["source_width"], row["source_height"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (width, height)):
        raise ValueError("SiBeTan v2 source dimensions differ")
    _validate_v2_outcome(row["face"], "face", width, height, root, artifact_paths)
    _validate_v2_outcome(row["nose"], "nose", width, height, root, artifact_paths)


def _validate_v2_outcome(
    value: object, branch: str, source_width: int, source_height: int,
    root: Path | None, artifact_paths: set[str],
) -> None:
    common = {"state", "reasons", "source_box_xyxy", "crop_path", "crop_sha256", "crop_width", "crop_height", "quality"}
    expected = common | (
        {"proposal_box_xyxy", "upstream_quality"}
        if branch == "face" else
        {"head_relative_box_xyxy", "head_box_xyxy", "keypoints", "localizer_confidence", "frontality", "native_short_side"}
    )
    if not isinstance(value, dict) or set(value) != expected or value["state"] not in STATES:
        raise ValueError(f"SiBeTan v2 {branch} outcome differs")
    reasons = value["reasons"]
    if not isinstance(reasons, list) or reasons != sorted(set(reasons)):
        raise ValueError(f"SiBeTan v2 {branch} reasons differ")
    available = value["state"] == "AVAILABLE"
    if available != (not reasons):
        raise ValueError(f"SiBeTan v2 {branch} state/reasons differ")
    artifacts = ("crop_path", "crop_sha256", "crop_width", "crop_height")
    if branch == "face" and value["source_box_xyxy"] is not None:
        _validate_box(value["source_box_xyxy"], source_width, source_height, "face source box")
        proposal = value["proposal_box_xyxy"]
        if not isinstance(proposal, list) or len(proposal) != 4 or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in proposal
        ):
            raise ValueError("SiBeTan v2 face proposal differs")
    if branch == "nose" and value["head_box_xyxy"] is not None:
        head = value["head_box_xyxy"]
        local = value["head_relative_box_xyxy"]
        source = value["source_box_xyxy"]
        _validate_box(head, source_width, source_height, "nose head box")
        _validate_box(local, head[2] - head[0], head[3] - head[1], "nose head-relative box")
        _validate_box(source, source_width, source_height, "nose source box")
        if source != [head[0] + local[0], head[1] + local[1], head[0] + local[2], head[1] + local[3]]:
            raise ValueError("SiBeTan v2 nose coordinate mapping differs")
        if value["native_short_side"] != min(source[2] - source[0], source[3] - source[1]):
            raise ValueError("SiBeTan v2 nose native short side differs")
        for name in ("localizer_confidence", "frontality"):
            number = value[name]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"SiBeTan v2 nose {name} differs")
        keypoints = value["keypoints"]
        if not isinstance(keypoints, dict) or set(keypoints) != {"normalized_head", "source_xyc"}:
            raise ValueError("SiBeTan v2 nose keypoints differ")
        if any(not isinstance(points, list) or len(points) != 8 for points in keypoints.values()):
            raise ValueError("SiBeTan v2 nose keypoint count differs")
    if available != all(value[field] is not None for field in artifacts):
        raise ValueError(f"SiBeTan v2 {branch} artifact availability differs")
    box = value["source_box_xyxy"]
    if available:
        _validate_box(box, source_width, source_height, f"{branch} source box")
        if value["crop_width"] != box[2] - box[0] or value["crop_height"] != box[3] - box[1]:
            raise ValueError(f"SiBeTan v2 {branch} crop geometry differs")
        _sha(value["crop_sha256"], f"{branch}.crop_sha256")
        path = PurePosixPath(value["crop_path"])
        if path.is_absolute() or ".." in path.parts or path.as_posix() in artifact_paths:
            raise ValueError(f"SiBeTan v2 {branch} crop path differs")
        artifact_paths.add(path.as_posix())
        if root is not None:
            candidate = root.joinpath(*path.parts).resolve(strict=True)
            if not candidate.is_relative_to(root.resolve(strict=True)) or candidate.is_symlink():
                raise ValueError(f"SiBeTan v2 {branch} crop path is unsafe")
            import hashlib
            from PIL import Image
            payload = candidate.read_bytes()
            if hashlib.sha256(payload).hexdigest() != value["crop_sha256"]:
                raise ValueError(f"SiBeTan v2 {branch} crop hash differs")
            with Image.open(candidate) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (value["crop_width"], value["crop_height"]):
                    raise ValueError(f"SiBeTan v2 {branch} crop encoding differs")
    elif any(value[field] is not None for field in artifacts):
        raise ValueError(f"SiBeTan v2 unavailable {branch} carries artifact")
    if not isinstance(value["quality"], dict):
        raise ValueError(f"SiBeTan v2 {branch} quality differs")


def _validate_box(value: object, width: int, height: int, name: str) -> None:
    if (
        not isinstance(value, list) or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or not (0 <= value[0] < value[2] <= width and 0 <= value[1] < value[3] <= height)
    ):
        raise ValueError(f"SiBeTan v2 {name} differs")
