"""Strict macro nose-print acquisition and annotation admission contracts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Callable, Iterable
import uuid

import numpy as np
from PIL import Image

from cvi.nose_id.types import NOSE_KEYPOINTS
from cvi.provenance import content_sha256


ACQUISITION_SCHEMA = "cvi.noseid.acquisition.v1"
ANNOTATION_SCHEMA = "cvi.noseid.annotation.v1"
ADMISSION_RECEIPT_SCHEMA = "cvi.noseid.annotation_admission_receipt.v1"
ANNOTATION_TEMPLATE_SCHEMA = "cvi.noseid.annotation_template.v1"
MINIMUM_NATIVE_NOSE_SHORT_SIDE = 224
SPLIT_ROLES = ("TRAIN", "DEV", "FUSION_CAL", "TEST")
USAGE_LANES = ("RESEARCH_ONLY", "COMMERCIAL_ALLOWED")
SEMANTIC_MASK_CLASSES = {
    "0": "context",
    "1": "nasal_surface",
    "2": "nostril",
}
INVALID_MASK_CLASSES = {
    "0": "valid",
    "1": "specular",
    "2": "occluded",
    "3": "contaminated",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OPAQUE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]{7,127}")
_MAX_JSONL_BYTES = 268_435_456
_MAX_JSONL_LINE_BYTES = 1_048_576
_MAX_IMAGE_BYTES = 268_435_456
_MAX_MASK_BYTES = 67_108_864


def _exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    if set(value) != expected:
        raise ValueError(f"{name} keys differ")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\0" in value:
        raise ValueError(f"{name} must be a non-empty trimmed string without NUL")
    return value


def _opaque_token(value: Any, name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an opaque 8-128 character token")
    return value


def _canonical_uuid5(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("registered_dog_id must be a string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError("registered_dog_id must be a canonical UUIDv5") from exc
    if parsed.version != 5 or str(parsed) != value:
        raise ValueError("registered_dog_id must be a canonical UUIDv5")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _relative_path(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError(f"{name} must be a normalized POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a normalized POSIX relative path")
    return path


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer_box(value: Any, name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must contain four integer xyxy coordinates")
    box = tuple(value)
    if box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{name} must be a non-empty non-negative xyxy box")
    return box  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class OriginalImage:
    relative_path: PurePosixPath
    sha256: str
    width: int
    height: int

    @classmethod
    def from_dict(cls, payload: Any) -> "OriginalImage":
        value = _exact_keys(
            payload, {"relative_path", "sha256", "width", "height"}, "original image"
        )
        return cls(
            _relative_path(value["relative_path"], "original image path"),
            _sha256(value["sha256"], "original image sha256"),
            _positive_integer(value["width"], "original image width"),
            _positive_integer(value["height"], "original image height"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path.as_posix(),
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionRecord:
    sample_id: str
    registered_dog_id: str
    session_id: str
    camera_id: str
    capture_id: str
    original_image: OriginalImage
    consent_token: str
    license_id: str
    usage_lane: str
    split_role: str

    @classmethod
    def from_dict(cls, payload: Any) -> "AcquisitionRecord":
        value = _exact_keys(
            payload,
            {
                "schema_version",
                "sample_id",
                "registered_dog_id",
                "session_id",
                "camera_id",
                "capture_id",
                "original_image",
                "consent",
                "license",
                "split_role",
            },
            "acquisition record",
        )
        if value["schema_version"] != ACQUISITION_SCHEMA:
            raise ValueError("unsupported acquisition record schema")
        consent = _exact_keys(
            value["consent"], {"token", "usage_lane"}, "acquisition consent"
        )
        license_value = _exact_keys(
            value["license"], {"license_id", "usage_lane"}, "acquisition license"
        )
        consent_lane = consent["usage_lane"]
        license_lane = license_value["usage_lane"]
        if consent_lane not in USAGE_LANES or license_lane not in USAGE_LANES:
            raise ValueError("unsupported acquisition usage lane")
        if consent_lane != license_lane:
            raise ValueError("consent and license usage lanes must match")
        role = value["split_role"]
        if role not in SPLIT_ROLES:
            raise ValueError("unsupported acquisition split role")
        session_id = _opaque_token(value["session_id"], "session_id")
        camera_id = _opaque_token(value["camera_id"], "camera_id")
        capture_id = _opaque_token(value["capture_id"], "capture_id")
        if len({session_id, camera_id, capture_id}) != 3:
            raise ValueError("session_id, camera_id, and capture_id must be distinct")
        return cls(
            sample_id=_opaque_token(value["sample_id"], "sample_id"),
            registered_dog_id=_canonical_uuid5(value["registered_dog_id"]),
            session_id=session_id,
            camera_id=camera_id,
            capture_id=capture_id,
            original_image=OriginalImage.from_dict(value["original_image"]),
            consent_token=_opaque_token(consent["token"], "consent token"),
            license_id=_nonempty(license_value["license_id"], "license_id"),
            usage_lane=consent_lane,
            split_role=role,
        )

    @property
    def record_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACQUISITION_SCHEMA,
            "sample_id": self.sample_id,
            "registered_dog_id": self.registered_dog_id,
            "session_id": self.session_id,
            "camera_id": self.camera_id,
            "capture_id": self.capture_id,
            "original_image": self.original_image.to_dict(),
            "consent": {"token": self.consent_token, "usage_lane": self.usage_lane},
            "license": {"license_id": self.license_id, "usage_lane": self.usage_lane},
            "split_role": self.split_role,
        }


@dataclass(frozen=True, slots=True)
class MaskReference:
    relative_path: PurePosixPath
    sha256: str
    box_xyxy: tuple[int, int, int, int]
    classes: dict[str, str]

    @classmethod
    def from_dict(
        cls, payload: Any, *, name: str, expected_classes: dict[str, str]
    ) -> "MaskReference":
        value = _exact_keys(
            payload, {"relative_path", "sha256", "box_xyxy", "classes"}, name
        )
        if value["classes"] != expected_classes:
            raise ValueError(f"{name} classes differ")
        return cls(
            _relative_path(value["relative_path"], f"{name} path"),
            _sha256(value["sha256"], f"{name} sha256"),
            _integer_box(value["box_xyxy"], f"{name} box"),
            dict(expected_classes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path.as_posix(),
            "sha256": self.sha256,
            "box_xyxy": list(self.box_xyxy),
            "classes": dict(self.classes),
        }


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    sample_id: str
    acquisition_sha256: str
    nose_bbox_xyxy: tuple[int, int, int, int]
    native_nose_short_side: int
    keypoints_xy: tuple[tuple[float, float], ...]
    keypoint_visibility: tuple[int, ...]
    semantic_mask: MaskReference
    invalid_mask: MaskReference
    annotator_token: str
    reviewer_token: str

    @classmethod
    def from_dict(cls, payload: Any) -> "AnnotationRecord":
        value = _exact_keys(
            payload,
            {
                "schema_version",
                "sample_id",
                "acquisition_sha256",
                "nose_bbox_xyxy",
                "native_nose_short_side",
                "nose_points",
                "semantic_mask",
                "invalid_mask",
                "annotator_token",
                "reviewer_token",
                "review_status",
            },
            "annotation record",
        )
        if value["schema_version"] != ANNOTATION_SCHEMA:
            raise ValueError("unsupported annotation record schema")
        if value["review_status"] != "APPROVED":
            raise ValueError("completed annotation review_status must be APPROVED")
        points = _exact_keys(
            value["nose_points"], {"order", "xy", "visibility"}, "nose points"
        )
        if points["order"] != list(NOSE_KEYPOINTS):
            raise ValueError("nose point order differs")
        xy = points["xy"]
        visibility = points["visibility"]
        if not isinstance(xy, list) or len(xy) != len(NOSE_KEYPOINTS):
            raise ValueError("nose point xy must have shape [6,2]")
        parsed_xy: list[tuple[float, float]] = []
        for index, point in enumerate(xy):
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("nose point xy must have shape [6,2]")
            parsed_xy.append(
                (
                    _finite_number(point[0], f"nose point {index} x"),
                    _finite_number(point[1], f"nose point {index} y"),
                )
            )
        if (
            not isinstance(visibility, list)
            or len(visibility) != len(NOSE_KEYPOINTS)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item not in {0, 1, 2}
                for item in visibility
            )
        ):
            raise ValueError("nose point visibility must contain six values in {0,1,2}")
        bbox = _integer_box(value["nose_bbox_xyxy"], "nose bbox")
        native_short_side = _positive_integer(
            value["native_nose_short_side"], "native nose short side"
        )
        observed_short_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
        if native_short_side != observed_short_side:
            raise ValueError("native nose short side differs from nose bbox")
        if native_short_side < MINIMUM_NATIVE_NOSE_SHORT_SIDE:
            raise ValueError(
                f"native nose short side must be at least {MINIMUM_NATIVE_NOSE_SHORT_SIDE}"
            )
        annotator = _opaque_token(value["annotator_token"], "annotator_token")
        reviewer = _opaque_token(value["reviewer_token"], "reviewer_token")
        if annotator == reviewer:
            raise ValueError("annotator_token and reviewer_token must be distinct")
        return cls(
            sample_id=_opaque_token(value["sample_id"], "sample_id"),
            acquisition_sha256=_sha256(
                value["acquisition_sha256"], "acquisition_sha256"
            ),
            nose_bbox_xyxy=bbox,
            native_nose_short_side=native_short_side,
            keypoints_xy=tuple(parsed_xy),
            keypoint_visibility=tuple(visibility),
            semantic_mask=MaskReference.from_dict(
                value["semantic_mask"],
                name="semantic mask",
                expected_classes=SEMANTIC_MASK_CLASSES,
            ),
            invalid_mask=MaskReference.from_dict(
                value["invalid_mask"],
                name="invalid mask",
                expected_classes=INVALID_MASK_CLASSES,
            ),
            annotator_token=annotator,
            reviewer_token=reviewer,
        )

    @property
    def record_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ANNOTATION_SCHEMA,
            "sample_id": self.sample_id,
            "acquisition_sha256": self.acquisition_sha256,
            "nose_bbox_xyxy": list(self.nose_bbox_xyxy),
            "native_nose_short_side": self.native_nose_short_side,
            "nose_points": {
                "order": list(NOSE_KEYPOINTS),
                "xy": [list(point) for point in self.keypoints_xy],
                "visibility": list(self.keypoint_visibility),
            },
            "semantic_mask": self.semantic_mask.to_dict(),
            "invalid_mask": self.invalid_mask.to_dict(),
            "annotator_token": self.annotator_token,
            "reviewer_token": self.reviewer_token,
            "review_status": "APPROVED",
        }


def _read_regular_bytes(path: Path, maximum_bytes: int, name: str) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("authenticated reads require O_NOFOLLOW")
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        if absolute.is_symlink():
            raise ValueError(f"{name} must not be a symlink: {path}") from exc
        raise
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"{name} must be a regular file: {path}")
        if initial.st_size > maximum_bytes:
            raise ValueError(f"{name} exceeds byte limit: {path}")
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(descriptor, min(1 << 20, maximum_bytes + 1 - observed)):
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError(f"{name} exceeds byte limit: {path}")
            chunks.append(chunk)
        final = os.fstat(descriptor)
        named = os.stat(absolute, follow_symlinks=False)
    finally:
        os.close(descriptor)
    initial_identity = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
    final_identity = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
    if (
        initial_identity != final_identity
        or (named.st_dev, named.st_ino) != (initial.st_dev, initial.st_ino)
        or observed != initial.st_size
    ):
        raise RuntimeError(f"{name} changed while reading: {path}")
    return b"".join(chunks)


def _secure_root(root: Path, name: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(root)))
    if absolute.is_symlink() or not absolute.is_dir():
        raise ValueError(f"{name} must be a non-symlink directory")
    return absolute


def secure_relative_file(
    root: Path,
    relative: PurePosixPath,
    *,
    expected_sha256: str,
    maximum_bytes: int,
    name: str,
) -> bytes:
    """Read one hash-bound regular file without following any path symlink."""

    absolute_root = _secure_root(root, f"{name} root")
    current = absolute_root
    for part in relative.parts[:-1]:
        current /= part
        try:
            current_stat = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            raise ValueError(f"{name} parent does not exist: {relative}") from None
        if stat.S_ISLNK(current_stat.st_mode) or not stat.S_ISDIR(current_stat.st_mode):
            raise ValueError(f"{name} parent must be a real directory: {relative}")
    payload = _read_regular_bytes(
        absolute_root.joinpath(*relative.parts), maximum_bytes, name
    )
    if sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{name} SHA256 differs: {relative}")
    return payload


def _load_jsonl(
    path: Path, parser: Callable[[Any], Any], name: str
) -> tuple[Any, ...]:
    raw = _read_regular_bytes(path, _MAX_JSONL_BYTES, name)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} must be UTF-8") from exc
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"{name} must not be empty")
    rows: list[Any] = []
    for line_number, line in enumerate(lines, 1):
        if not line or not line.strip():
            raise ValueError(f"{name} line {line_number} is empty")
        if len(line.encode("utf-8")) > _MAX_JSONL_LINE_BYTES:
            raise ValueError(f"{name} line {line_number} exceeds byte limit")
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON number: {value}")
                ),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {name} JSON on line {line_number}") from exc
        rows.append(parser(payload))
    return tuple(rows)


def load_acquisition_jsonl(path: Path) -> tuple[AcquisitionRecord, ...]:
    return _load_jsonl(path, AcquisitionRecord.from_dict, "acquisition JSONL")


def load_annotation_jsonl(path: Path) -> tuple[AnnotationRecord, ...]:
    return _load_jsonl(path, AnnotationRecord.from_dict, "annotation JSONL")


def canonical_jsonl_bytes(records: Iterable[AcquisitionRecord | AnnotationRecord]) -> bytes:
    return b"".join(
        json.dumps(
            record.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


def _validate_identity_policy(records: Iterable[AcquisitionRecord]) -> None:
    identity_roles: dict[str, str] = {}
    train_sessions: dict[str, set[str]] = {}
    for record in records:
        previous = identity_roles.setdefault(record.registered_dog_id, record.split_role)
        if previous != record.split_role:
            raise ValueError("nose identities must be disjoint across split roles")
        if record.split_role == "TRAIN":
            train_sessions.setdefault(record.registered_dog_id, set()).add(record.session_id)
    insufficient = sorted(
        identity for identity, sessions in train_sessions.items() if len(sessions) < 2
    )
    if insufficient:
        raise ValueError("every TRAIN identity must have at least two distinct sessions")


def validate_acquisition_records(
    records: tuple[AcquisitionRecord, ...], data_root: Path
) -> None:
    if not records:
        raise ValueError("acquisition records must not be empty")
    root = _secure_root(data_root, "acquisition data root")
    sample_ids: set[str] = set()
    capture_bindings: dict[str, tuple[str, str, str]] = {}
    for record in records:
        if not isinstance(record, AcquisitionRecord):
            raise TypeError("acquisition records must contain AcquisitionRecord values")
        if AcquisitionRecord.from_dict(record.to_dict()) != record:
            raise ValueError("acquisition record is not canonical")
        if record.sample_id in sample_ids:
            raise ValueError(f"duplicate acquisition sample_id: {record.sample_id}")
        sample_ids.add(record.sample_id)
        binding = (
            record.registered_dog_id,
            record.session_id,
            record.camera_id,
        )
        previous_binding = capture_bindings.setdefault(record.capture_id, binding)
        if previous_binding != binding:
            raise ValueError(
                "one capture_id must bind exactly one identity, session, and camera"
            )
        image_bytes = secure_relative_file(
            root,
            record.original_image.relative_path,
            expected_sha256=record.original_image.sha256,
            maximum_bytes=_MAX_IMAGE_BYTES,
            name="original image",
        )
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                size = image.size
        except Exception as exc:
            raise ValueError(
                f"original image cannot be decoded: {record.original_image.relative_path}"
            ) from exc
        if size != (record.original_image.width, record.original_image.height):
            raise ValueError("decoded original image dimensions differ")
    _validate_identity_policy(records)


def _validate_mask(
    root: Path,
    reference: MaskReference,
    *,
    expected_box: tuple[int, int, int, int],
    allowed_values: set[int],
    name: str,
) -> None:
    if reference.box_xyxy != expected_box:
        raise ValueError(f"{name} box must exactly match nose bbox")
    if reference.relative_path.suffix.lower() != ".png":
        raise ValueError(f"{name} must be a lossless PNG")
    payload = secure_relative_file(
        root,
        reference.relative_path,
        expected_sha256=reference.sha256,
        maximum_bytes=_MAX_MASK_BYTES,
        name=name,
    )
    try:
        with Image.open(BytesIO(payload)) as opened:
            if opened.format != "PNG":
                raise ValueError(f"{name} must be a PNG")
            opened.load()
            mask = np.asarray(opened)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"{name} cannot be decoded") from exc
    expected_shape = (expected_box[3] - expected_box[1], expected_box[2] - expected_box[0])
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != expected_shape:
        raise ValueError(f"{name} must be uint8 with shape {expected_shape}")
    observed = {int(value) for value in np.unique(mask)}
    if not observed <= allowed_values:
        raise ValueError(f"{name} contains an unknown class value")


def validate_annotation_records(
    acquisitions: tuple[AcquisitionRecord, ...],
    annotations: tuple[AnnotationRecord, ...],
    *,
    data_root: Path,
    annotation_root: Path,
) -> None:
    validate_acquisition_records(acquisitions, data_root)
    if not annotations:
        raise ValueError("completed annotations must not be empty")
    mask_root = _secure_root(annotation_root, "annotation root")
    by_sample = {record.sample_id: record for record in acquisitions}
    seen: set[str] = set()
    admitted_acquisitions: list[AcquisitionRecord] = []
    for annotation in annotations:
        if not isinstance(annotation, AnnotationRecord):
            raise TypeError("annotations must contain AnnotationRecord values")
        if AnnotationRecord.from_dict(annotation.to_dict()) != annotation:
            raise ValueError("annotation record is not canonical")
        if annotation.sample_id in seen:
            raise ValueError(f"duplicate annotation sample_id: {annotation.sample_id}")
        seen.add(annotation.sample_id)
        acquisition = by_sample.get(annotation.sample_id)
        if acquisition is None:
            raise ValueError("annotation does not bind an acquisition sample")
        if annotation.acquisition_sha256 != acquisition.record_sha256:
            raise ValueError("annotation acquisition_sha256 differs")
        x0, y0, x1, y1 = annotation.nose_bbox_xyxy
        if x1 > acquisition.original_image.width or y1 > acquisition.original_image.height:
            raise ValueError("nose bbox exceeds original image")
        for x, y in annotation.keypoints_xy:
            if not (0.0 <= x < acquisition.original_image.width) or not (
                0.0 <= y < acquisition.original_image.height
            ):
                raise ValueError("nose point lies outside original image")
        _validate_mask(
            mask_root,
            annotation.semantic_mask,
            expected_box=annotation.nose_bbox_xyxy,
            allowed_values={0, 1, 2},
            name="semantic mask",
        )
        _validate_mask(
            mask_root,
            annotation.invalid_mask,
            expected_box=annotation.nose_bbox_xyxy,
            allowed_values={0, 1, 2, 3},
            name="invalid mask",
        )
        admitted_acquisitions.append(acquisition)
    _validate_identity_policy(admitted_acquisitions)


def build_admission_receipt(
    acquisitions: tuple[AcquisitionRecord, ...],
    annotations: tuple[AnnotationRecord, ...],
    *,
    acquisition_jsonl_sha256: str,
    annotation_jsonl_sha256: str,
    batch_manifest_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic receipt after ``validate_annotation_records`` passes."""

    for value, name in (
        (acquisition_jsonl_sha256, "acquisition_jsonl_sha256"),
        (annotation_jsonl_sha256, "annotation_jsonl_sha256"),
        (batch_manifest_sha256, "batch_manifest_sha256"),
    ):
        _sha256(value, name)
    by_sample = {record.sample_id: record for record in acquisitions}
    ordered = sorted(annotations, key=lambda item: item.sample_id)
    records = [
        {
            "sample_id": annotation.sample_id,
            "registered_dog_id": by_sample[annotation.sample_id].registered_dog_id,
            "split_role": by_sample[annotation.sample_id].split_role,
            "acquisition_sha256": annotation.acquisition_sha256,
            "annotation_sha256": annotation.record_sha256,
            "original_image_sha256": by_sample[
                annotation.sample_id
            ].original_image.sha256,
            "semantic_mask_sha256": annotation.semantic_mask.sha256,
            "invalid_mask_sha256": annotation.invalid_mask.sha256,
        }
        for annotation in ordered
    ]
    role_counts = {
        role: sum(item["split_role"] == role for item in records) for role in SPLIT_ROLES
    }
    receipt = {
        "schema_version": ADMISSION_RECEIPT_SCHEMA,
        "decision": "ADMITTED_VERIFIED_HUMAN_ANNOTATIONS",
        "contracts": {
            "acquisition_schema": ACQUISITION_SCHEMA,
            "annotation_schema": ANNOTATION_SCHEMA,
            "minimum_native_nose_short_side": MINIMUM_NATIVE_NOSE_SHORT_SIDE,
            "nose_point_order": list(NOSE_KEYPOINTS),
            "semantic_mask_classes": dict(SEMANTIC_MASK_CLASSES),
            "invalid_mask_classes": dict(INVALID_MASK_CLASSES),
        },
        "acquisition_jsonl_sha256": acquisition_jsonl_sha256,
        "annotation_jsonl_sha256": annotation_jsonl_sha256,
        "batch_manifest_sha256": batch_manifest_sha256,
        "admitted_records_sha256": content_sha256(records),
        "admitted_count": len(records),
        "admitted_identity_count": len(
            {item["registered_dog_id"] for item in records}
        ),
        "split_role_counts": role_counts,
        "records": records,
    }
    return receipt


__all__ = [
    "ACQUISITION_SCHEMA",
    "ADMISSION_RECEIPT_SCHEMA",
    "ANNOTATION_SCHEMA",
    "ANNOTATION_TEMPLATE_SCHEMA",
    "AcquisitionRecord",
    "AnnotationRecord",
    "INVALID_MASK_CLASSES",
    "MINIMUM_NATIVE_NOSE_SHORT_SIDE",
    "MaskReference",
    "OriginalImage",
    "SEMANTIC_MASK_CLASSES",
    "SPLIT_ROLES",
    "USAGE_LANES",
    "build_admission_receipt",
    "canonical_jsonl_bytes",
    "load_acquisition_jsonl",
    "load_annotation_jsonl",
    "secure_relative_file",
    "validate_acquisition_records",
    "validate_annotation_records",
]
