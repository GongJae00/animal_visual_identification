"""Strict, content-bound manifests for materialized nose-region crops."""

from __future__ import annotations

import hashlib
import io
import math
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from shared.foundation.protected_io import read_strict_json_document
from shared.foundation.provenance import content_sha256


BUNDLE_SCHEMA = "identification.nose.nose_region_crop_manifest_bundle.v1"
MANIFEST_SCHEMA = "identification.nose.nose_region_crop_manifest.v1"
PLAN_SCHEMA = "identification.nose.nose_region_crop_protocol_plan.v1"
SUMMARY_SCHEMA = "identification.nose.nose_region_crop_summary.v1"

ROLE_TO_SPLIT = {
    "DOGFACE_FIT": "TRAIN",
    "YT_FIT": "TRAIN",
    "MPDD_EXTERNAL_UNUSED_TRAIN_DOMAIN": "DEV",
}
ROLE_TO_DATASET = {
    "DOGFACE_FIT": "dogfacenet224",
    "YT_FIT": "yt-bb-dog",
    "MPDD_EXTERNAL_UNUSED_TRAIN_DOMAIN": "mpdd",
}
LICENSING_LANES = {
    "dogfacenet224": "RESEARCH_ONLY_CC_BY_NC_4_0_DERIVED_LOCALIZER",
    "yt-bb-dog": "RESEARCH_ONLY_CC_BY_NC_4_0_DERIVED_LOCALIZER",
    "mpdd": "RESEARCH_ONLY_VALIDATION_CC_BY_NC_4_0_DERIVED_LOCALIZER",
}
REQUIRED_DATASET_SPLITS = {
    "dogfacenet224": "TRAIN",
    "yt-bb-dog": "TRAIN",
    "mpdd": "DEV",
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_FIELDS = {
    "dataset_name",
    "sample_token",
    "identity_token",
    "registered_dog_id",
    "capture_session_token",
    "source_sha256",
    "source_width",
    "source_height",
    "crop_path",
    "crop_sha256",
    "crop_width",
    "crop_height",
    "detector_confidence",
    "frontality",
    "nose_box_xyxy",
    "source_role",
    "split_role",
    "licensing_lane",
}
_POLICY_FIELDS = {
    "minimum_detector_confidence",
    "minimum_frontality",
    "minimum_native_short_side",
    "frontality_metric",
    "crop_encoding",
    "path_layout",
}
_COUNT_FIELDS = {"candidates", "admitted", "rejected", "rejection_reasons"}
_MANIFEST_FIELDS = {
    "schema_version",
    "input_sha256s",
    "policy",
    "records",
    "summary",
    "interpretation",
}


def admitted_split_for_role(role: object, dataset_name: object) -> str | None:
    """Return the only admitted split for a protected role, or reject it."""

    if not isinstance(role, str) or not role:
        raise ValueError("protected source role must be non-empty text")
    split = ROLE_TO_SPLIT.get(role)
    if split is None:
        return None
    if dataset_name != ROLE_TO_DATASET[role]:
        raise ValueError("protected source role and dataset differ")
    return split


def normalized_box_to_pixel_box(
    box: Sequence[object], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Clip normalized xyxy coordinates and conservatively cover source pixels."""

    _positive_int(image_width, "image_width")
    _positive_int(image_height, "image_height")
    if isinstance(box, (str, bytes)) or len(box) != 4:
        raise ValueError("normalized nose box must have four coordinates")
    values = tuple(_finite_number(value, "normalized nose box") for value in box)
    x1, y1, x2, y2 = (min(1.0, max(0.0, value)) for value in values)
    pixel_box = (
        math.floor(x1 * image_width),
        math.floor(y1 * image_height),
        math.ceil(x2 * image_width),
        math.ceil(y2 * image_height),
    )
    _pixel_box(pixel_box, image_width, image_height, "nose_box_xyxy")
    return pixel_box


def encode_png_crop(
    image: Image.Image, box: Sequence[object]
) -> tuple[bytes, tuple[int, int]]:
    """Encode exactly the predicted pixel box as deterministic RGB PNG bytes."""

    if not isinstance(image, Image.Image):
        raise TypeError("source crop must be a PIL image")
    pixel_box = _pixel_box(box, image.width, image.height, "nose_box_xyxy")
    crop = image.convert("RGB").crop(pixel_box)
    output = io.BytesIO()
    crop.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue(), crop.size


def frontality_components_from_keypoints(
    keypoints: Sequence[Sequence[object]],
) -> dict[str, float]:
    """Separate geometric pose evidence from localizer confidence."""
    if len(keypoints) != 8 or any(len(point) != 3 for point in keypoints):
        raise ValueError("frontality requires eight normalized x/y/confidence points")
    parsed = tuple(
        tuple(_finite_number(value, "frontality keypoint") for value in point)
        for point in keypoints
    )
    if any(
        not 0.0 <= value <= 1.0
        for point in parsed
        for value in point
    ):
        raise ValueError("frontality keypoints must be normalized to [0, 1]")
    left_eye, right_eye, nasal_root, nasal_inferior = parsed[:4]
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    eye_distance = math.hypot(eye_dx, eye_dy)
    if eye_distance <= 1e-8:
        return {
            "symmetry": 0.0,
            "level": 0.0,
            "geometric_frontality": 0.0,
            "anchor_confidence": min(point[2] for point in parsed[:4]),
            "combined_frontality": 0.0,
        }
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    nose_x = (nasal_root[0] + nasal_inferior[0]) / 2.0
    offset = abs(nose_x - eye_mid_x) / eye_distance
    roll = abs(eye_dy) / eye_distance
    symmetry = max(0.0, 1.0 - offset / 0.5)
    level = max(0.0, 1.0 - roll / 0.5)
    confidence = min(point[2] for point in parsed[:4])
    geometry = min(1.0, max(0.0, symmetry * level))
    if geometry <= 1e-12:
        geometry = 0.0
    combined = min(1.0, max(0.0, geometry * confidence))
    return {
        "symmetry": symmetry,
        "level": level,
        "geometric_frontality": geometry,
        "anchor_confidence": confidence,
        "combined_frontality": 0.0 if combined <= 1e-12 else combined,
    }


def frontality_from_keypoints(keypoints: Sequence[Sequence[object]]) -> float:
    """Return the legacy geometry-times-confidence frontality score."""

    return frontality_components_from_keypoints(keypoints)["combined_frontality"]


def build_protocol_plan(
    *,
    input_sha256s: Mapping[str, str],
    policy: Mapping[str, Any],
    dataset_counts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the immutable pre-materialization plan printed before inference."""

    hashes = _input_hashes(input_sha256s)
    normalized_policy = _policy(policy)
    counts = _dataset_counts(dataset_counts, final=False)
    for dataset_name in REQUIRED_DATASET_SPLITS:
        if counts.get(dataset_name, {}).get("candidates", 0) <= 0:
            raise ValueError(f"required dataset has no candidate crops: {dataset_name}")
    plan = {
        "schema_version": PLAN_SCHEMA,
        "admitted_roles": dict(sorted(ROLE_TO_SPLIT.items())),
        "input_sha256s": hashes,
        "policy": normalized_policy,
        "dataset_counts": counts,
        "interpretation": "PLAN_ONLY_NO_CROPS_WRITTEN",
    }
    plan["plan_sha256"] = content_sha256(plan)
    return plan


def build_summary(
    *,
    input_sha256s: Mapping[str, str],
    dataset_counts: Mapping[str, Mapping[str, Any]],
    protocol_plan_sha256: str,
) -> dict[str, Any]:
    hashes = _input_hashes(input_sha256s)
    counts = _dataset_counts(dataset_counts, final=True)
    _require_sha256(protocol_plan_sha256, "protocol_plan_sha256")
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "input_sha256s": hashes,
        "protocol_plan_sha256": protocol_plan_sha256,
        "dataset_counts": counts,
    }
    summary["summary_sha256"] = content_sha256(summary)
    return summary


def build_nose_region_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    input_sha256s: Mapping[str, str],
    policy: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate a content-hashed crop-manifest bundle."""

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "input_sha256s": _input_hashes(input_sha256s),
        "policy": _policy(policy),
        "records": [dict(record) for record in records],
        "summary": dict(summary),
        "interpretation": "DERIVED_NOSE_CROPS_ONLY_NOT_RAW_DATA_OR_BIOMETRIC_VALIDATION",
    }
    _validate_manifest(manifest, root=None)
    return {
        "schema_version": BUNDLE_SCHEMA,
        "manifest_sha256": content_sha256(manifest),
        "manifest": manifest,
    }


def validate_nose_region_manifest_bundle(
    bundle: object, *, root: Path | None = None
) -> dict[str, Any]:
    expected = {"schema_version", "manifest_sha256", "manifest"}
    if not isinstance(bundle, dict) or set(bundle) != expected:
        raise ValueError("nose-region manifest bundle schema differs")
    if bundle["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError("unsupported nose-region manifest bundle schema")
    _require_sha256(bundle["manifest_sha256"], "manifest_sha256")
    manifest = bundle["manifest"]
    if not isinstance(manifest, dict) or content_sha256(manifest) != bundle[
        "manifest_sha256"
    ]:
        raise ValueError("nose-region manifest content digest differs")
    resolved_root = None
    if root is not None:
        resolved_root = root.resolve(strict=True)
        if not resolved_root.is_dir():
            raise ValueError("nose-region manifest root must be a directory")
    _validate_manifest(manifest, root=resolved_root)
    return manifest


def read_nose_region_manifest(path: Path) -> dict[str, Any]:
    document = read_strict_json_document(
        path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    return validate_nose_region_manifest_bundle(
        document.payload, root=path.parent
    )


def _validate_manifest(manifest: object, root: Path | None) -> None:
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("nose-region manifest schema differs")
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError("unsupported nose-region manifest schema")
    if manifest["interpretation"] != (
        "DERIVED_NOSE_CROPS_ONLY_NOT_RAW_DATA_OR_BIOMETRIC_VALIDATION"
    ):
        raise ValueError("nose-region manifest interpretation differs")
    hashes = _input_hashes(manifest["input_sha256s"])
    policy = _policy(manifest["policy"])
    if hashes != manifest["input_sha256s"] or policy != manifest["policy"]:
        raise ValueError("nose-region manifest values are not canonical")
    summary = _validate_summary(manifest["summary"], hashes)
    records = manifest["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("nose-region manifest records must be non-empty")

    samples: set[str] = set()
    identities: set[str] = set()
    identity_contracts: dict[str, tuple[str, str, str]] = {}
    registered_contracts: dict[str, str] = {}
    paths: set[str] = set()
    admitted_counts: Counter[str] = Counter()
    split_identities: dict[str, set[str]] = defaultdict(set)
    previous_key: tuple[str, str] | None = None
    for record in records:
        _validate_record(record, root, policy)
        sample = record["sample_token"]
        identity = record["identity_token"]
        if sample in samples:
            raise ValueError("nose-region manifest repeats a sample token")
        if sample in identities or identity in samples or sample == identity:
            raise ValueError("sample and identity token domains collide")
        samples.add(sample)
        identities.add(identity)
        contract = (
            record["registered_dog_id"],
            record["source_role"],
            record["split_role"],
        )
        if identity_contracts.setdefault(identity, contract) != contract:
            raise ValueError("one identity has conflicting manifest contracts")
        registered = record["registered_dog_id"]
        if registered_contracts.setdefault(registered, identity) != identity:
            raise ValueError("registered UUID aliases multiple identity tokens")
        if record["crop_path"] in paths:
            raise ValueError("nose-region manifest repeats a crop path")
        paths.add(record["crop_path"])
        key = (record["dataset_name"], sample)
        if previous_key is not None and key <= previous_key:
            raise ValueError("nose-region records must be canonically sorted")
        previous_key = key
        admitted_counts[record["dataset_name"]] += 1
        split_identities[record["split_role"]].add(identity)

    if split_identities["TRAIN"] & split_identities["DEV"]:
        raise ValueError("TRAIN and DEV identities must be disjoint")
    for dataset_name, split_role in REQUIRED_DATASET_SPLITS.items():
        if admitted_counts[dataset_name] <= 0:
            raise ValueError(f"required materialized dataset is absent: {dataset_name}")
        if any(
            record["split_role"] != split_role
            for record in records
            if record["dataset_name"] == dataset_name
        ):
            raise ValueError("dataset appears in the wrong materialized split")
    count_payload = summary["dataset_counts"]
    for dataset_name, values in count_payload.items():
        if values["admitted"] != admitted_counts[dataset_name]:
            raise ValueError("summary admitted counts differ from records")
    if set(admitted_counts) - set(count_payload):
        raise ValueError("summary omits a materialized dataset")


def _validate_record(
    record: object, root: Path | None, policy: Mapping[str, Any]
) -> None:
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise ValueError("nose-region record schema differs")
    dataset = _text(record["dataset_name"], "dataset_name")
    if dataset not in REQUIRED_DATASET_SPLITS:
        raise ValueError("nose-region record dataset is unsupported")
    for field in (
        "sample_token",
        "identity_token",
        "capture_session_token",
        "source_sha256",
        "crop_sha256",
    ):
        _require_sha256(record[field], field)
    _canonical_uuid5(record["registered_dog_id"])
    source_width = _positive_int(record["source_width"], "source_width")
    source_height = _positive_int(record["source_height"], "source_height")
    crop_width = _positive_int(record["crop_width"], "crop_width")
    crop_height = _positive_int(record["crop_height"], "crop_height")
    box = _pixel_box(
        record["nose_box_xyxy"], source_width, source_height, "nose_box_xyxy"
    )
    if (crop_width, crop_height) != (box[2] - box[0], box[3] - box[1]):
        raise ValueError("crop dimensions differ from predicted nose box")
    confidence = _probability(record["detector_confidence"], "detector_confidence")
    frontality = _probability(record["frontality"], "frontality")
    if confidence < policy["minimum_detector_confidence"]:
        raise ValueError("admitted crop is below the detector confidence threshold")
    if frontality < policy["minimum_frontality"]:
        raise ValueError("admitted crop is below the frontality threshold")
    if min(crop_width, crop_height) < policy["minimum_native_short_side"]:
        raise ValueError("admitted crop is below the native-resolution threshold")
    role = _text(record["source_role"], "source_role")
    expected_split = admitted_split_for_role(role, dataset)
    if expected_split is None or record["split_role"] != expected_split:
        raise ValueError("nose-region record carries a protected rejected role")
    if record["licensing_lane"] != LICENSING_LANES[dataset]:
        raise ValueError("nose-region licensing lane differs")
    path = _relative_crop_path(record["crop_path"], record["sample_token"])

    if root is not None:
        candidate = root.joinpath(*path.parts)
        if candidate.is_symlink():
            raise ValueError("nose-region crop must not be a symlink")
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("nose-region crop does not exist or is unsafe") from exc
        if (
            not resolved.is_relative_to(root)
            or resolved.relative_to(root).as_posix() != path.as_posix()
            or not resolved.is_file()
        ):
            raise ValueError("nose-region crop escapes its manifest root")
        payload = resolved.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["crop_sha256"]:
            raise ValueError("nose-region crop hash differs")
        try:
            with Image.open(io.BytesIO(payload)) as opened:
                if opened.format != "PNG" or opened.mode != "RGB":
                    raise ValueError("nose-region crop must be an RGB PNG")
                if opened.size != (crop_width, crop_height):
                    raise ValueError("nose-region crop dimensions differ")
                opened.load()
        except (OSError, SyntaxError) as exc:
            raise ValueError("nose-region crop is not a valid PNG") from exc
    del confidence, frontality


def _validate_summary(summary: object, input_hashes: dict[str, str]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "input_sha256s",
        "protocol_plan_sha256",
        "dataset_counts",
        "summary_sha256",
    }
    if not isinstance(summary, dict) or set(summary) != fields:
        raise ValueError("nose-region summary schema differs")
    if summary["schema_version"] != SUMMARY_SCHEMA:
        raise ValueError("unsupported nose-region summary schema")
    if _input_hashes(summary["input_sha256s"]) != input_hashes:
        raise ValueError("nose-region summary input hashes differ")
    _require_sha256(summary["protocol_plan_sha256"], "protocol_plan_sha256")
    counts = _dataset_counts(summary["dataset_counts"], final=True)
    if counts != summary["dataset_counts"]:
        raise ValueError("nose-region summary counts are not canonical")
    _require_sha256(summary["summary_sha256"], "summary_sha256")
    expected = content_sha256({
        key: value for key, value in summary.items() if key != "summary_sha256"
    })
    if summary["summary_sha256"] != expected:
        raise ValueError("nose-region summary digest differs")
    return summary


def _input_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("input_sha256s must be a non-empty object")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not name or any(ord(char) < 32 for char in name):
            raise ValueError("input hash names must be non-empty text")
        _require_sha256(digest, f"input_sha256s.{name}")
        result[name] = digest
    return dict(sorted(result.items()))


def _policy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _POLICY_FIELDS:
        raise ValueError("nose-region crop policy schema differs")
    result = dict(value)
    _probability(result["minimum_detector_confidence"], "minimum_detector_confidence")
    _probability(result["minimum_frontality"], "minimum_frontality")
    _positive_int(result["minimum_native_short_side"], "minimum_native_short_side")
    if result["frontality_metric"] != (
        "EYE_MIDLINE_NOSE_OFFSET_ROLL_WITH_KEYPOINT_CONFIDENCE_V1"
    ):
        raise ValueError("nose-region frontality metric differs")
    if result["crop_encoding"] != "PNG_RGB_LOSSLESS":
        raise ValueError("nose-region crop encoding differs")
    if result["path_layout"] != "FLAT_SAMPLE_TOKEN_HASH":
        raise ValueError("nose-region crop path layout differs")
    return result


def _dataset_counts(value: object, *, final: bool) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("dataset_counts must be a non-empty object")
    result: dict[str, dict[str, Any]] = {}
    expected_fields = _COUNT_FIELDS if final else {"candidates", "rejected", "rejection_reasons"}
    for dataset_name, raw in value.items():
        _text(dataset_name, "dataset count name")
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ValueError("dataset count schema differs")
        counts = dict(raw)
        for field in expected_fields - {"rejection_reasons"}:
            _nonnegative_int(counts[field], f"dataset_counts.{dataset_name}.{field}")
        reasons = counts["rejection_reasons"]
        if not isinstance(reasons, Mapping):
            raise ValueError("rejection_reasons must be an object")
        normalized_reasons: dict[str, int] = {}
        for reason, count in reasons.items():
            _text(reason, "rejection reason")
            normalized_reasons[reason] = _nonnegative_int(
                count, f"rejection_reasons.{reason}"
            )
        counts["rejection_reasons"] = dict(sorted(normalized_reasons.items()))
        if counts["rejected"] != sum(normalized_reasons.values()):
            raise ValueError("rejected count differs from rejection reasons")
        if final and counts["candidates"] != counts["admitted"] + sum(
            count
            for reason, count in normalized_reasons.items()
            if reason != "PROTECTED_ROLE_REJECTED"
        ):
            raise ValueError("candidate count differs from admission outcomes")
        result[dataset_name] = counts
    return dict(sorted(result.items()))


def _relative_crop_path(value: object, sample_token: str) -> PurePosixPath:
    text = _text(value, "crop_path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parts != ("crops", f"{sample_token}.png")
    ):
        raise ValueError("crop_path must use the flat sample-token hash layout")
    return path


def _pixel_box(
    value: Sequence[object], width: int, height: int, name: str
) -> tuple[int, int, int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{name} must have four integer coordinates")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must have four integer coordinates")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{name} must be non-empty and within the source crop")
    return x1, y1, x2, y2


def _canonical_uuid5(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("registered_dog_id must be a canonical UUIDv5")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError("registered_dog_id must be a canonical UUIDv5") from exc
    if parsed.version != 5 or str(parsed) != value:
        raise ValueError("registered_dog_id must be a canonical UUIDv5")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{name} must be bounded non-empty text")
    return value


def _finite_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _probability(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


__all__ = [
    "BUNDLE_SCHEMA",
    "LICENSING_LANES",
    "MANIFEST_SCHEMA",
    "PLAN_SCHEMA",
    "REQUIRED_DATASET_SPLITS",
    "ROLE_TO_DATASET",
    "ROLE_TO_SPLIT",
    "SUMMARY_SCHEMA",
    "admitted_split_for_role",
    "build_nose_region_manifest",
    "build_protocol_plan",
    "build_summary",
    "encode_png_crop",
    "frontality_from_keypoints",
    "normalized_box_to_pixel_box",
    "read_nose_region_manifest",
    "validate_nose_region_manifest_bundle",
]
