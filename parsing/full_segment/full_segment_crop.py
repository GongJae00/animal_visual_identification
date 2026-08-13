"""Materialize content-bound, neutral-background square Full crops."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from foundation.provenance import content_sha256
from parsing.full_segment.full_segment_cache import (
    thaw_animal_parsing_prediction,
    validate_frozen_animal_parsing,
)
from parsing.full_segment.full_segment_contracts import (
    AnimalAssociation,
    BodyMaskPolicy,
    ObservationRoute,
)

CROP_SCHEMA = "cvi.full_segment_crop.v1"
_MAX_SOURCE_PIXELS = 33_554_432
_CROP_FIELDS = {
    "schema_version",
    "source_sha256",
    "source_width",
    "source_height",
    "route",
    "parsing_prediction_sha256",
    "association",
    "authoritative_mask_sha256",
    "body_mask_policy",
    "mask_policy_sha256",
    "instance_index",
    "parsing_quality_state",
    "source_crop_box_xyxy",
    "target_size",
    "background_rgb",
    "full_rgb_encoding",
    "full_rgb_sha256",
    "full_rgb_byte_size",
    "full_mask_encoding",
    "full_mask_sha256",
    "full_mask_byte_size",
    "crop_record_sha256",
}


@dataclass(frozen=True, slots=True)
class MaterializedFullCrop:
    full_rgb_png: bytes
    full_mask_png: bytes
    record: dict[str, Any]


def materialize_full_crop(
    source_bytes: bytes,
    *,
    expected_source_sha256: str,
    frozen_parsing: object,
    association: AnimalAssociation,
    target_size: int = 224,
    context_fraction: float = 0.05,
    background_rgb: tuple[int, int, int] = (127, 127, 127),
) -> MaterializedFullCrop:
    """Materialize one selected body parse; never infer an association."""

    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise ValueError("Full crop source bytes must be non-empty")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != expected_source_sha256:
        raise ValueError("Full crop source digest differs")
    _validate_policy(target_size, context_fraction, background_rgb)
    bundle = validate_frozen_animal_parsing(frozen_parsing)
    prediction = thaw_animal_parsing_prediction(bundle)
    association.validate_instance_count(len(prediction.instances))
    instance = prediction.instances[association.instance_index]
    if instance.quality.state not in {"USABLE", "REVIEW"}:
        raise ValueError("Full crop requires a usable or review parsing prediction")
    if instance.mask_box_xyxy is None:
        raise ValueError("Full crop requires non-empty parser support")
    try:
        with Image.open(io.BytesIO(source_bytes)) as opened:
            if opened.width * opened.height > _MAX_SOURCE_PIXELS:
                raise ValueError("Full crop source pixel count exceeds policy")
            opened.load()
            source = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Full crop source is not a supported image") from exc
    if source.size != (prediction.source_width, prediction.source_height):
        raise ValueError("Full crop source dimensions differ from frozen parsing")

    box = _expand_box(
        instance.mask_box_xyxy,
        width=source.width,
        height=source.height,
        fraction=context_fraction,
    )
    x1, y1, x2, y2 = box
    source_values = np.asarray(source, dtype=np.uint8)
    mask_values = np.ascontiguousarray(instance.hard_mask[y1:y2, x1:x2])
    crop_values = np.empty((y2 - y1, x2 - x1, 3), dtype=np.uint8)
    crop_values[...] = background_rgb
    foreground = mask_values.astype(bool)
    crop_values[foreground] = source_values[y1:y2, x1:x2][foreground]

    rgb_png, mask_png = _square_artifacts(
        crop_values,
        mask_values,
        target_size=target_size,
        background_rgb=background_rgb,
    )
    payload = {
        "schema_version": CROP_SCHEMA,
        "source_sha256": source_sha256,
        "source_width": source.width,
        "source_height": source.height,
        "route": ObservationRoute.BODY_PARSING.value,
        "parsing_prediction_sha256": bundle["prediction_sha256"],
        "association": association.to_dict(),
        "authoritative_mask_sha256": None,
        "body_mask_policy": None,
        "mask_policy_sha256": None,
        "instance_index": instance.instance_index,
        "parsing_quality_state": instance.quality.state,
        "source_crop_box_xyxy": list(box),
        "target_size": target_size,
        "background_rgb": list(background_rgb),
        "full_rgb_encoding": "PNG_RGB_LOSSLESS",
        "full_rgb_sha256": hashlib.sha256(rgb_png).hexdigest(),
        "full_rgb_byte_size": len(rgb_png),
        "full_mask_encoding": "PNG_L_BINARY_0_255_LOSSLESS",
        "full_mask_sha256": hashlib.sha256(mask_png).hexdigest(),
        "full_mask_byte_size": len(mask_png),
    }
    record = {**payload, "crop_record_sha256": content_sha256(payload)}
    validate_full_crop_record(record)
    return MaterializedFullCrop(rgb_png, mask_png, record)


def materialize_body_mask_full_crop(
    source_bytes: bytes,
    authoritative_mask_bytes: bytes,
    *,
    expected_source_sha256: str,
    expected_authoritative_mask_sha256: str,
    policy: BodyMaskPolicy,
    target_size: int = 224,
    context_fraction: float = 0.05,
    background_rgb: tuple[int, int, int] = (127, 127, 127),
) -> MaterializedFullCrop:
    """Materialize Full from an authoritative source-aligned Oxford trimap."""

    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise ValueError("Full crop source bytes must be non-empty")
    if not isinstance(authoritative_mask_bytes, bytes) or not authoritative_mask_bytes:
        raise ValueError("authoritative body mask bytes must be non-empty")
    if not isinstance(policy, BodyMaskPolicy):
        raise TypeError("authoritative body mask requires a body mask policy")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != expected_source_sha256:
        raise ValueError("Full crop source digest differs")
    mask_sha256 = hashlib.sha256(authoritative_mask_bytes).hexdigest()
    if mask_sha256 != expected_authoritative_mask_sha256:
        raise ValueError("authoritative body mask digest differs")
    _validate_policy(target_size, context_fraction, background_rgb)

    try:
        with Image.open(io.BytesIO(source_bytes)) as opened:
            if opened.width * opened.height > _MAX_SOURCE_PIXELS:
                raise ValueError("Full crop source pixel count exceeds policy")
            opened.load()
            source = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Full crop source is not a supported image") from exc
    try:
        with Image.open(io.BytesIO(authoritative_mask_bytes)) as opened:
            if opened.width * opened.height > _MAX_SOURCE_PIXELS:
                raise ValueError("authoritative body mask pixel count exceeds policy")
            if len(opened.getbands()) != 1:
                raise ValueError("authoritative body mask must be single-channel")
            opened.load()
            mask_values = np.asarray(opened)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("authoritative body mask is not a supported image") from exc
    if mask_values.ndim != 2 or mask_values.dtype.kind not in {"i", "u"}:
        raise ValueError("authoritative body mask labels must be integer single-channel")
    if source.size != (mask_values.shape[1], mask_values.shape[0]):
        raise ValueError("authoritative body mask dimensions differ from source")

    observed_labels = {int(label) for label in np.unique(mask_values)}
    expected_labels = {1, 2, 3}
    if not observed_labels.issubset(expected_labels):
        raise ValueError("authoritative body mask contains invalid Oxford trimap labels")
    if 1 not in observed_labels:
        raise ValueError("authoritative body mask is all-background")
    if policy.permitted_labels is None:
        if observed_labels != expected_labels:
            raise ValueError("authoritative body mask is missing expected Oxford labels")
    elif not observed_labels.issubset(set(policy.permitted_labels)):
        raise ValueError("authoritative body mask exceeds explicitly permitted labels")

    hard_mask = np.ascontiguousarray(mask_values == 1, dtype=np.uint8)
    y_values, x_values = np.nonzero(hard_mask)
    mask_box = (
        int(x_values.min()),
        int(y_values.min()),
        int(x_values.max()) + 1,
        int(y_values.max()) + 1,
    )
    box = _expand_box(
        mask_box,
        width=source.width,
        height=source.height,
        fraction=context_fraction,
    )
    x1, y1, x2, y2 = box
    source_values = np.asarray(source, dtype=np.uint8)
    cropped_mask = hard_mask[y1:y2, x1:x2]
    crop_values = np.empty((y2 - y1, x2 - x1, 3), dtype=np.uint8)
    crop_values[...] = background_rgb
    foreground = cropped_mask.astype(bool)
    crop_values[foreground] = source_values[y1:y2, x1:x2][foreground]
    rgb_png, mask_png = _square_artifacts(
        crop_values,
        cropped_mask,
        target_size=target_size,
        background_rgb=background_rgb,
    )
    payload = {
        "schema_version": CROP_SCHEMA,
        "source_sha256": source_sha256,
        "source_width": source.width,
        "source_height": source.height,
        "route": ObservationRoute.BODY_MASK.value,
        "parsing_prediction_sha256": None,
        "association": None,
        "authoritative_mask_sha256": mask_sha256,
        "body_mask_policy": policy.to_dict(),
        "mask_policy_sha256": policy.policy_sha256,
        "instance_index": None,
        "parsing_quality_state": None,
        "source_crop_box_xyxy": list(box),
        "target_size": target_size,
        "background_rgb": list(background_rgb),
        "full_rgb_encoding": "PNG_RGB_LOSSLESS",
        "full_rgb_sha256": hashlib.sha256(rgb_png).hexdigest(),
        "full_rgb_byte_size": len(rgb_png),
        "full_mask_encoding": "PNG_L_BINARY_0_255_LOSSLESS",
        "full_mask_sha256": hashlib.sha256(mask_png).hexdigest(),
        "full_mask_byte_size": len(mask_png),
    }
    record = {**payload, "crop_record_sha256": content_sha256(payload)}
    validate_full_crop_record(record)
    return MaterializedFullCrop(rgb_png, mask_png, record)


def materialize_native_full_crop(
    source_bytes: bytes,
    *,
    expected_source_sha256: str,
    route: ObservationRoute,
    target_size: int = 224,
    background_rgb: tuple[int, int, int] = (127, 127, 127),
) -> MaterializedFullCrop:
    """Materialize a native Face/Head source as an explicit Full appearance view."""

    if route not in {ObservationRoute.NATIVE_FACE, ObservationRoute.NATIVE_HEAD}:
        raise ValueError("native Full crop requires a native Face or Head route")
    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise ValueError("Full crop source bytes must be non-empty")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != expected_source_sha256:
        raise ValueError("Full crop source digest differs")
    _validate_policy(target_size, 0.0, background_rgb)
    try:
        with Image.open(io.BytesIO(source_bytes)) as opened:
            if opened.width * opened.height > _MAX_SOURCE_PIXELS:
                raise ValueError("Full crop source pixel count exceeds policy")
            opened.load()
            source = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Full crop source is not a supported image") from exc

    source_values = np.asarray(source, dtype=np.uint8)
    source_mask = np.ones((source.height, source.width), dtype=np.uint8)
    rgb_png, mask_png = _square_artifacts(
        source_values,
        source_mask,
        target_size=target_size,
        background_rgb=background_rgb,
    )
    payload = {
        "schema_version": CROP_SCHEMA,
        "source_sha256": source_sha256,
        "source_width": source.width,
        "source_height": source.height,
        "route": route.value,
        "parsing_prediction_sha256": None,
        "association": None,
        "authoritative_mask_sha256": None,
        "body_mask_policy": None,
        "mask_policy_sha256": None,
        "instance_index": None,
        "parsing_quality_state": "NATIVE",
        "source_crop_box_xyxy": [0, 0, source.width, source.height],
        "target_size": target_size,
        "background_rgb": list(background_rgb),
        "full_rgb_encoding": "PNG_RGB_LOSSLESS",
        "full_rgb_sha256": hashlib.sha256(rgb_png).hexdigest(),
        "full_rgb_byte_size": len(rgb_png),
        "full_mask_encoding": "PNG_L_BINARY_0_255_LOSSLESS",
        "full_mask_sha256": hashlib.sha256(mask_png).hexdigest(),
        "full_mask_byte_size": len(mask_png),
    }
    record = {**payload, "crop_record_sha256": content_sha256(payload)}
    validate_full_crop_record(record)
    return MaterializedFullCrop(rgb_png, mask_png, record)


def validate_full_crop_record(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CROP_FIELDS:
        raise ValueError("Full crop record schema differs")
    if value["schema_version"] != CROP_SCHEMA:
        raise ValueError("unsupported Full crop record schema")
    for field in ("source_sha256", "full_rgb_sha256", "full_mask_sha256", "crop_record_sha256"):
        _require_sha256(value[field], field)
    payload = {key: item for key, item in value.items() if key != "crop_record_sha256"}
    if content_sha256(payload) != value["crop_record_sha256"]:
        raise ValueError("Full crop record digest differs")
    width = _positive_int(value["source_width"], "Full crop source width")
    height = _positive_int(value["source_height"], "Full crop source height")
    try:
        route = ObservationRoute(value["route"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Full crop route differs") from exc
    if route is ObservationRoute.BODY_PARSING:
        _require_sha256(value["parsing_prediction_sha256"], "parsing_prediction_sha256")
        association = AnimalAssociation.from_dict(value["association"])
        if association.instance_index != value["instance_index"]:
            raise ValueError("Full crop association index differs")
        if value["parsing_quality_state"] not in {"USABLE", "REVIEW"}:
            raise ValueError("Full crop parsing quality differs")
        if any(
            value[field] is not None
            for field in (
                "authoritative_mask_sha256",
                "body_mask_policy",
                "mask_policy_sha256",
            )
        ):
            raise ValueError("parsed Full crop cannot claim authoritative mask provenance")
    elif route is ObservationRoute.BODY_MASK:
        if any(
            value[field] is not None
            for field in (
                "parsing_prediction_sha256",
                "association",
                "instance_index",
                "parsing_quality_state",
            )
        ):
            raise ValueError("body-mask Full crop cannot claim parsing provenance")
        _require_sha256(
            value["authoritative_mask_sha256"], "authoritative_mask_sha256"
        )
        _require_sha256(value["mask_policy_sha256"], "mask_policy_sha256")
        body_mask_policy = BodyMaskPolicy.from_dict(value["body_mask_policy"])
        if body_mask_policy.policy_sha256 != value["mask_policy_sha256"]:
            raise ValueError("Full crop body mask policy digest differs")
    elif route in {ObservationRoute.NATIVE_FACE, ObservationRoute.NATIVE_HEAD}:
        if any(
            value[field] is not None
            for field in (
                "parsing_prediction_sha256",
                "association",
                "authoritative_mask_sha256",
                "body_mask_policy",
                "mask_policy_sha256",
                "instance_index",
            )
        ) or value["parsing_quality_state"] != "NATIVE":
            raise ValueError("native Full crop cannot claim body provenance")
    else:
        raise ValueError("Full crop route cannot materialize an artifact")
    _pixel_box(value["source_crop_box_xyxy"], width, height)
    target_size = _positive_int(value["target_size"], "Full crop target size")
    background = value["background_rgb"]
    if not isinstance(background, list):
        raise TypeError("Full crop background must be an array")
    _validate_policy(target_size, 0.0, tuple(background))
    if value["full_rgb_encoding"] != "PNG_RGB_LOSSLESS":
        raise ValueError("Full RGB encoding differs")
    if value["full_mask_encoding"] != "PNG_L_BINARY_0_255_LOSSLESS":
        raise ValueError("Full mask encoding differs")
    _positive_int(value["full_rgb_byte_size"], "Full RGB byte size")
    _positive_int(value["full_mask_byte_size"], "Full mask byte size")
    return value


def verify_full_crop_artifacts(
    crop: MaterializedFullCrop | dict[str, Any],
    full_rgb_png: bytes | None = None,
    full_mask_png: bytes | None = None,
) -> None:
    if isinstance(crop, MaterializedFullCrop):
        record = validate_full_crop_record(crop.record)
        rgb_payload = crop.full_rgb_png
        mask_payload = crop.full_mask_png
    else:
        record = validate_full_crop_record(crop)
        if full_rgb_png is None or full_mask_png is None:
            raise ValueError("Full crop verification requires both artifacts")
        rgb_payload, mask_payload = full_rgb_png, full_mask_png
    for payload, prefix in ((rgb_payload, "full_rgb"), (mask_payload, "full_mask")):
        if len(payload) != record[f"{prefix}_byte_size"]:
            raise ValueError(f"{prefix} artifact byte size differs")
        if hashlib.sha256(payload).hexdigest() != record[f"{prefix}_sha256"]:
            raise ValueError(f"{prefix} artifact digest differs")
    with Image.open(io.BytesIO(rgb_payload)) as rgb:
        if rgb.format != "PNG" or rgb.mode != "RGB" or rgb.size != (
            record["target_size"],
            record["target_size"],
        ):
            raise ValueError("Full RGB artifact contract differs")
    with Image.open(io.BytesIO(mask_payload)) as mask:
        if mask.format != "PNG" or mask.mode != "L" or mask.size != (
            record["target_size"],
            record["target_size"],
        ):
            raise ValueError("Full mask artifact contract differs")
        if not set(np.unique(np.asarray(mask, dtype=np.uint8))).issubset({0, 255}):
            raise ValueError("Full mask artifact must be binary")


def _encode_png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _square_artifacts(
    rgb_values: np.ndarray,
    mask_values: np.ndarray,
    *,
    target_size: int,
    background_rgb: tuple[int, int, int],
) -> tuple[bytes, bytes]:
    side = max(rgb_values.shape[:2])
    offset_x = (side - rgb_values.shape[1]) // 2
    offset_y = (side - rgb_values.shape[0]) // 2
    square_rgb = np.empty((side, side, 3), dtype=np.uint8)
    square_rgb[...] = background_rgb
    square_mask = np.zeros((side, side), dtype=np.uint8)
    square_rgb[
        offset_y : offset_y + rgb_values.shape[0],
        offset_x : offset_x + rgb_values.shape[1],
    ] = rgb_values
    square_mask[
        offset_y : offset_y + mask_values.shape[0],
        offset_x : offset_x + mask_values.shape[1],
    ] = mask_values * 255
    rgb = Image.fromarray(square_rgb, mode="RGB").resize(
        (target_size, target_size), Image.Resampling.BILINEAR
    )
    mask = Image.fromarray(square_mask, mode="L").resize(
        (target_size, target_size), Image.Resampling.NEAREST
    )
    resized_values = np.asarray(rgb, dtype=np.uint8).copy()
    resized_mask = np.asarray(mask, dtype=np.uint8)
    if not resized_mask.any():
        raise ValueError("Full mask artifact has no foreground")
    resized_values[resized_mask == 0] = background_rgb
    return _encode_png(Image.fromarray(resized_values, mode="RGB")), _encode_png(mask)


def _expand_box(
    box: tuple[int, int, int, int], *, width: int, height: int, fraction: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x = math.ceil((x2 - x1) * fraction)
    pad_y = math.ceil((y2 - y1) * fraction)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _validate_policy(
    target_size: object,
    context_fraction: object,
    background_rgb: tuple[object, ...],
) -> None:
    if (
        isinstance(target_size, bool)
        or not isinstance(target_size, int)
        or not 1 <= target_size <= 4096
        or isinstance(context_fraction, bool)
        or not isinstance(context_fraction, (int, float))
        or not math.isfinite(context_fraction)
        or not 0.0 <= context_fraction <= 1.0
        or len(background_rgb) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
            for item in background_rgb
        )
    ):
        raise ValueError("Full crop materialization policy differs")


def _pixel_box(value: object, width: int, height: int) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError("Full crop source box differs")
    x1, y1, x2, y2 = value
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("Full crop source box exceeds source")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")


__all__ = [
    "CROP_SCHEMA",
    "MaterializedFullCrop",
    "materialize_body_mask_full_crop",
    "materialize_full_crop",
    "materialize_native_full_crop",
    "validate_full_crop_record",
    "verify_full_crop_artifacts",
]
