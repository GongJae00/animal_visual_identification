"""Deterministic packed parsing and cache records for the Full segment."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from foundation.provenance import content_sha256
from localization.animal_parsing import (
    AnimalParsingPrediction,
    ParsedAnimalInstance,
    ParsedAnimalQuality,
)
from localization.full_segment_contracts import (
    AnimalAssociation,
    FullSegmentObservation,
    FullStatus,
    ObservationRoute,
    SegmentRole,
    SegmentRoleRecord,
    SourceViewScope,
    TerminalObservability,
)

FROZEN_PARSING_SCHEMA = "cvi.full_segment_frozen_animal_parsing.v1"
FROZEN_PARSING_BINDING_SCHEMA = "cvi.full_segment_frozen_parsing_binding.v1"
CACHE_SCHEMA = "cvi.full_segment_cache.v2"
CACHE_BUNDLE_SCHEMA = "cvi.full_segment_cache_bundle.v2"
LEGACY_CACHE_SCHEMA = "cvi.full_segment_cache.v1"
LEGACY_CACHE_BUNDLE_SCHEMA = "cvi.full_segment_cache_bundle.v1"

_MAX_SOURCE_PIXELS = 33_554_432
_MAX_INSTANCES = 32
_INSTANCE_FIELDS = {
    "instance_index",
    "query_index",
    "class_id",
    "class_name",
    "class_score",
    "detector_box_xyxy",
    "refinement_box_xyxy",
    "mask_box_xyxy",
    "quality",
    "instance_probability",
    "foreground_probability",
    "ownership_probability",
    "hard_mask",
}
_QUALITY_FIELDS = {
    "state",
    "reasons",
    "flags",
    "semantic_shape_iou",
    "ownership_retention",
    "foreground_pixels",
    "component_count",
    "touches_source_border",
}
_ARRAY_FIELDS = {"dtype", "shape", "encoding", "data", "raw_sha256"}
_FROZEN_BINDING_FIELDS = {
    "schema_version",
    "frozen_schema_version",
    "prediction_sha256",
    "frozen_content_sha256",
    "frozen_json_sha256",
    "policy_sha256",
    "source_id",
    "source_sha256",
    "source_width",
    "source_height",
    "instance_count",
    "association",
}


def freeze_animal_parsing_prediction(
    prediction: AnimalParsingPrediction,
) -> dict[str, Any]:
    """Pack a complete parser result without rerunning or weakening the parser."""

    if not isinstance(prediction, AnimalParsingPrediction):
        raise TypeError("Full segment freezing requires AnimalParsingPrediction")
    payload = {
        "source_width": prediction.source_width,
        "source_height": prediction.source_height,
        "parser_schema_version": prediction.schema_version,
        "ontology": prediction.ontology,
        "ontology_description": prediction.ontology_description,
        "policy_sha256": prediction.policy_sha256,
        "instances": [_pack_instance(instance) for instance in prediction.instances],
    }
    bundle = {
        "schema_version": FROZEN_PARSING_SCHEMA,
        "prediction_sha256": content_sha256(payload),
        "prediction": payload,
    }
    validate_frozen_animal_parsing(bundle)
    return bundle


def validate_frozen_animal_parsing(value: object) -> dict[str, Any]:
    fields = {"schema_version", "prediction_sha256", "prediction"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("frozen animal parsing bundle schema differs")
    if value["schema_version"] != FROZEN_PARSING_SCHEMA:
        raise ValueError("unsupported frozen animal parsing schema")
    _require_sha256(value["prediction_sha256"], "frozen parsing prediction")
    payload = value["prediction"]
    payload_fields = {
        "source_width",
        "source_height",
        "parser_schema_version",
        "ontology",
        "ontology_description",
        "policy_sha256",
        "instances",
    }
    if not isinstance(payload, dict) or set(payload) != payload_fields:
        raise ValueError("frozen animal parsing payload schema differs")
    if content_sha256(payload) != value["prediction_sha256"]:
        raise ValueError("frozen animal parsing prediction digest differs")
    width = _positive_int(payload["source_width"], "frozen parsing source width")
    height = _positive_int(payload["source_height"], "frozen parsing source height")
    if width * height > _MAX_SOURCE_PIXELS:
        raise ValueError("frozen parsing source pixel count exceeds policy")
    for field in ("parser_schema_version", "ontology", "ontology_description"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"frozen parsing {field} must be non-empty text")
    _require_sha256(payload["policy_sha256"], "frozen parsing policy")
    instances = payload["instances"]
    if not isinstance(instances, list) or len(instances) > _MAX_INSTANCES:
        raise ValueError("frozen parsing instances differ")
    for index, instance in enumerate(instances):
        _validate_packed_instance(instance, index=index, width=width, height=height)
    return value


def thaw_animal_parsing_prediction(value: object) -> AnimalParsingPrediction:
    bundle = validate_frozen_animal_parsing(value)
    payload = bundle["prediction"]
    instances = tuple(_unpack_instance(instance) for instance in payload["instances"])
    return AnimalParsingPrediction(
        source_width=payload["source_width"],
        source_height=payload["source_height"],
        instances=instances,
        policy_sha256=payload["policy_sha256"],
        ontology=payload["ontology"],
        ontology_description=payload["ontology_description"],
        schema_version=payload["parser_schema_version"],
    )


def build_frozen_parsing_binding(
    frozen_parsing: object,
    *,
    frozen_json_sha256: str,
    source_id: str,
    source_sha256: str,
    association: AnimalAssociation,
) -> dict[str, Any]:
    """Build a compact, path-independent reference to shared frozen parsing."""

    bundle = validate_frozen_animal_parsing(frozen_parsing)
    payload = bundle["prediction"]
    association.validate_instance_count(len(payload["instances"]))
    binding = {
        "schema_version": FROZEN_PARSING_BINDING_SCHEMA,
        "frozen_schema_version": bundle["schema_version"],
        "prediction_sha256": bundle["prediction_sha256"],
        "frozen_content_sha256": content_sha256(bundle),
        "frozen_json_sha256": frozen_json_sha256,
        "policy_sha256": payload["policy_sha256"],
        "source_id": source_id,
        "source_sha256": source_sha256,
        "source_width": payload["source_width"],
        "source_height": payload["source_height"],
        "instance_count": len(payload["instances"]),
        "association": association.to_dict(),
    }
    validate_frozen_parsing_binding(binding)
    return binding


def validate_frozen_parsing_binding(value: object) -> dict[str, Any]:
    """Validate a compact frozen-parsing reference without opening shared arrays."""

    if not isinstance(value, dict) or set(value) != _FROZEN_BINDING_FIELDS:
        raise ValueError("frozen parsing binding schema differs")
    if value["schema_version"] != FROZEN_PARSING_BINDING_SCHEMA:
        raise ValueError("unsupported frozen parsing binding schema")
    if value["frozen_schema_version"] != FROZEN_PARSING_SCHEMA:
        raise ValueError("frozen parsing binding frozen schema differs")
    for field in (
        "prediction_sha256",
        "frozen_content_sha256",
        "frozen_json_sha256",
        "policy_sha256",
        "source_sha256",
    ):
        _require_sha256(value[field], f"frozen parsing binding {field}")
    if not isinstance(value["source_id"], str) or not value["source_id"]:
        raise ValueError("frozen parsing binding source ID must be non-empty")
    width = _positive_int(value["source_width"], "frozen parsing binding width")
    height = _positive_int(value["source_height"], "frozen parsing binding height")
    if width * height > _MAX_SOURCE_PIXELS:
        raise ValueError("frozen parsing binding source pixel count exceeds policy")
    count = value["instance_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 <= count <= _MAX_INSTANCES
    ):
        raise ValueError("frozen parsing binding instance count differs")
    association = AnimalAssociation.from_dict(value["association"])
    association.validate_instance_count(count)
    return value


def build_body_observation(
    *,
    source_id: str,
    source_sha256: str,
    source_view_scope: SourceViewScope,
    frozen_parsing: object,
    association: AnimalAssociation,
    face_observability: TerminalObservability = TerminalObservability.NOT_RUN,
    nose_observability: TerminalObservability = TerminalObservability.NOT_RUN,
    full_rgb_sha256: str | None = None,
) -> FullSegmentObservation:
    """Bind one frozen parser instance to a body observation without inference."""

    if source_view_scope not in {
        SourceViewScope.BODY_AVAILABLE,
        SourceViewScope.BODY_TRUNCATED,
    }:
        raise ValueError("body observation requires a body source scope")
    if TerminalObservability.NATIVE in {face_observability, nose_observability}:
        raise ValueError("body observation cannot claim native Face/Nose evidence")
    bundle = validate_frozen_animal_parsing(frozen_parsing)
    prediction = thaw_animal_parsing_prediction(bundle)
    association.validate_instance_count(len(prediction.instances))
    quality_state = prediction.instances[association.instance_index].quality.state
    full_status = FullStatus(quality_state)
    if (
        source_view_scope is SourceViewScope.BODY_TRUNCATED
        and full_status is FullStatus.USABLE
    ):
        full_status = FullStatus.REVIEW
    if full_rgb_sha256 is not None:
        _require_sha256(full_rgb_sha256, "Full RGB artifact")
        if full_status not in {FullStatus.USABLE, FullStatus.REVIEW}:
            raise ValueError("unusable body observation cannot bind a Full crop")
    prediction_sha256 = bundle["prediction_sha256"]
    roles = (
        SegmentRoleRecord(
            SegmentRole.FULL,
            full_status.value,
            ObservationRoute.BODY_PARSING,
            prediction_sha256,
            full_rgb_sha256,
        ),
        SegmentRoleRecord(
            SegmentRole.FACE,
            face_observability.value,
            ObservationRoute.BODY_PARSING,
            None,
            None,
        ),
        SegmentRoleRecord(
            SegmentRole.NOSE,
            nose_observability.value,
            ObservationRoute.BODY_PARSING,
            None,
            None,
        ),
    )
    return FullSegmentObservation(
        source_id=source_id,
        source_sha256=source_sha256,
        source_width=prediction.source_width,
        source_height=prediction.source_height,
        source_view_scope=source_view_scope,
        full_status=full_status,
        face_observability=face_observability,
        nose_observability=nose_observability,
        route=ObservationRoute.BODY_PARSING,
        parsing_prediction_sha256=prediction_sha256,
        association=association,
        authoritative_mask_sha256=None,
        mask_policy_sha256=None,
        roles=roles,
    )


def build_body_mask_observation(
    *,
    source_id: str,
    source_sha256: str,
    source_width: int,
    source_height: int,
    source_view_scope: SourceViewScope,
    authoritative_mask_sha256: str,
    mask_policy_sha256: str,
    full_rgb_sha256: str,
    face_observability: TerminalObservability = TerminalObservability.NOT_RUN,
    nose_observability: TerminalObservability = TerminalObservability.NOT_RUN,
) -> FullSegmentObservation:
    """Bind authoritative mask materialization without claiming parser output."""

    if source_view_scope not in {
        SourceViewScope.BODY_AVAILABLE,
        SourceViewScope.BODY_TRUNCATED,
    }:
        raise ValueError("body mask observation requires a body source scope")
    if TerminalObservability.NATIVE in {face_observability, nose_observability}:
        raise ValueError("body mask observation cannot claim native Face/Nose evidence")
    _require_sha256(authoritative_mask_sha256, "authoritative body mask")
    _require_sha256(mask_policy_sha256, "body mask policy")
    _require_sha256(full_rgb_sha256, "Full RGB artifact")
    full_status = (
        FullStatus.REVIEW
        if source_view_scope is SourceViewScope.BODY_TRUNCATED
        else FullStatus.USABLE
    )
    roles = (
        SegmentRoleRecord(
            SegmentRole.FULL,
            full_status.value,
            ObservationRoute.BODY_MASK,
            mask_policy_sha256,
            full_rgb_sha256,
        ),
        SegmentRoleRecord(
            SegmentRole.FACE,
            face_observability.value,
            ObservationRoute.BODY_MASK,
            None,
            None,
        ),
        SegmentRoleRecord(
            SegmentRole.NOSE,
            nose_observability.value,
            ObservationRoute.BODY_MASK,
            None,
            None,
        ),
    )
    return FullSegmentObservation(
        source_id=source_id,
        source_sha256=source_sha256,
        source_width=source_width,
        source_height=source_height,
        source_view_scope=source_view_scope,
        full_status=full_status,
        face_observability=face_observability,
        nose_observability=nose_observability,
        route=ObservationRoute.BODY_MASK,
        parsing_prediction_sha256=None,
        association=None,
        authoritative_mask_sha256=authoritative_mask_sha256,
        mask_policy_sha256=mask_policy_sha256,
        roles=roles,
    )


def build_full_segment_cache(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = [dict(record) for record in records]
    normalized.sort(key=lambda record: record.get("source_id", ""))
    schema = (
        LEGACY_CACHE_SCHEMA
        if any(
            isinstance(record.get("frozen_parsing"), dict)
            and record["frozen_parsing"].get("schema_version") == FROZEN_PARSING_SCHEMA
            for record in normalized
        )
        else CACHE_SCHEMA
    )
    cache = {"schema_version": schema, "records": normalized}
    validate_full_segment_cache(cache)
    return {
        "schema_version": (
            LEGACY_CACHE_BUNDLE_SCHEMA
            if schema == LEGACY_CACHE_SCHEMA
            else CACHE_BUNDLE_SCHEMA
        ),
        "cache_sha256": content_sha256(cache),
        "cache": cache,
    }


def validate_full_segment_cache_bundle(value: object) -> dict[str, Any]:
    fields = {"schema_version", "cache_sha256", "cache"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Full segment cache bundle schema differs")
    if value["schema_version"] not in {
        CACHE_BUNDLE_SCHEMA,
        LEGACY_CACHE_BUNDLE_SCHEMA,
    }:
        raise ValueError("unsupported Full segment cache bundle schema")
    _require_sha256(value["cache_sha256"], "Full segment cache")
    cache = validate_full_segment_cache(value["cache"])
    expected_bundle_schema = (
        LEGACY_CACHE_BUNDLE_SCHEMA
        if cache["schema_version"] == LEGACY_CACHE_SCHEMA
        else CACHE_BUNDLE_SCHEMA
    )
    if value["schema_version"] != expected_bundle_schema:
        raise ValueError("Full segment cache and bundle schema versions differ")
    if content_sha256(cache) != value["cache_sha256"]:
        raise ValueError("Full segment cache digest differs")
    return cache


def validate_full_segment_cache(value: object) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "records"}
        or value["schema_version"] not in {CACHE_SCHEMA, LEGACY_CACHE_SCHEMA}
    ):
        raise ValueError("Full segment cache schema differs")
    records = value["records"]
    if not isinstance(records, list):
        raise TypeError("Full segment cache records must be an array")
    previous: str | None = None
    for record in records:
        fields = {"source_id", "observation", "frozen_parsing", "crop"}
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("Full segment cache record schema differs")
        source_id = record["source_id"]
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Full segment cache source ID must be non-empty")
        if previous is not None and source_id <= previous:
            raise ValueError(
                "Full segment cache records must have unique sorted source IDs"
            )
        previous = source_id
        observation = FullSegmentObservation.from_dict(record["observation"])
        if observation.source_id != source_id:
            raise ValueError("Full segment cache source ID differs from observation")
        frozen = record["frozen_parsing"]
        if observation.parsing_prediction_sha256 is None:
            if frozen is not None:
                raise ValueError("non-body cache record cannot retain frozen parsing")
        else:
            if value["schema_version"] == LEGACY_CACHE_SCHEMA:
                validate_frozen_animal_parsing(frozen)
                if frozen["prediction_sha256"] != observation.parsing_prediction_sha256:
                    raise ValueError("cache parsing digest differs from observation")
            else:
                binding = validate_frozen_parsing_binding(frozen)
                if (
                    binding["prediction_sha256"]
                    != observation.parsing_prediction_sha256
                    or binding["source_id"] != observation.source_id
                    or binding["source_sha256"] != observation.source_sha256
                    or (binding["source_width"], binding["source_height"])
                    != (observation.source_width, observation.source_height)
                    or AnimalAssociation.from_dict(binding["association"])
                    != observation.association
                ):
                    raise ValueError("cache parsing binding differs from observation")
        crop = record["crop"]
        full_artifact_sha256 = observation.roles[0].artifact_sha256
        if full_artifact_sha256 is not None and crop is None:
            raise ValueError("cache observation binds a missing Full crop")
        if crop is not None:
            from localization.full_segment_crop import validate_full_crop_record

            validate_full_crop_record(crop)
            if crop["source_sha256"] != observation.source_sha256:
                raise ValueError("cache crop source differs from observation")
            if (crop["source_width"], crop["source_height"]) != (
                observation.source_width,
                observation.source_height,
            ):
                raise ValueError("cache crop dimensions differ from observation")
            if crop["route"] != observation.route.value:
                raise ValueError("cache crop route differs from observation")
            if (
                crop["parsing_prediction_sha256"]
                != observation.parsing_prediction_sha256
            ):
                raise ValueError("cache crop parsing differs from observation")
            if (
                crop["authoritative_mask_sha256"]
                != observation.authoritative_mask_sha256
            ):
                raise ValueError("cache authoritative mask differs from observation")
            if crop["mask_policy_sha256"] != observation.mask_policy_sha256:
                raise ValueError("cache body mask policy differs from observation")
            if crop["full_rgb_sha256"] != full_artifact_sha256:
                raise ValueError("cache Full artifact differs from observation")
    return value


def _pack_instance(instance: ParsedAnimalInstance) -> dict[str, Any]:
    quality = instance.quality
    return {
        "instance_index": instance.instance_index,
        "query_index": instance.query_index,
        "class_id": instance.class_id,
        "class_name": instance.class_name,
        "class_score": instance.class_score,
        "detector_box_xyxy": list(instance.detector_box_xyxy),
        "refinement_box_xyxy": list(instance.refinement_box_xyxy),
        "mask_box_xyxy": (
            None if instance.mask_box_xyxy is None else list(instance.mask_box_xyxy)
        ),
        "quality": {
            "state": quality.state,
            "reasons": list(quality.reasons),
            "flags": list(quality.flags),
            "semantic_shape_iou": quality.semantic_shape_iou,
            "ownership_retention": quality.ownership_retention,
            "foreground_pixels": quality.foreground_pixels,
            "component_count": quality.component_count,
            "touches_source_border": quality.touches_source_border,
        },
        "instance_probability": _pack_array(instance.instance_probability),
        "foreground_probability": _pack_array(instance.foreground_probability),
        "ownership_probability": _pack_array(instance.ownership_probability),
        "hard_mask": _pack_array(instance.hard_mask),
    }


def _pack_array(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype == np.float32:
        dtype = "float32-le"
        raw = contiguous.astype("<f4", copy=False).tobytes(order="C")
    elif contiguous.dtype == np.uint8:
        dtype = "uint8"
        raw = contiguous.tobytes(order="C")
    else:
        raise ValueError("frozen parsing array dtype differs")
    # Level 1 retains exact bytes while avoiding high CPU cost for negligible
    # additional compression on dense probability maps.
    compressed = zlib.compress(raw, level=1)
    return {
        "dtype": dtype,
        "shape": list(contiguous.shape),
        "encoding": "BASE64_ZLIB_C_ORDER",
        "data": base64.b64encode(compressed).decode("ascii"),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_packed_instance(
    value: object, *, index: int, width: int, height: int
) -> None:
    if not isinstance(value, dict) or set(value) != _INSTANCE_FIELDS:
        raise ValueError("frozen parsing instance schema differs")
    if value["instance_index"] != index:
        raise ValueError("frozen parsing instance indices differ")
    for field in ("instance_index", "query_index", "class_id"):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"frozen parsing {field} must be non-negative")
    if not isinstance(value["class_name"], str) or not value["class_name"]:
        raise ValueError("frozen parsing class name must be non-empty")
    score = value["class_score"]
    if not _finite_number(score) or not 0.0 <= score <= 1.0:
        raise ValueError("frozen parsing class score differs")
    _pixel_box(value["detector_box_xyxy"], width, height, "detector box")
    _pixel_box(value["refinement_box_xyxy"], width, height, "refinement box")
    if value["mask_box_xyxy"] is not None:
        _pixel_box(value["mask_box_xyxy"], width, height, "mask box")
    _validate_quality(value["quality"])
    shape = (height, width)
    for field in (
        "instance_probability",
        "foreground_probability",
        "ownership_probability",
    ):
        _unpack_array(value[field], expected_dtype="float32-le", expected_shape=shape)
    mask = _unpack_array(
        value["hard_mask"], expected_dtype="uint8", expected_shape=shape
    )
    if not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError("frozen parsing hard mask must be binary")
    if bool(mask.any()) != (value["mask_box_xyxy"] is not None):
        raise ValueError("frozen parsing mask box differs from support")
    if int(mask.sum()) != value["quality"]["foreground_pixels"]:
        raise ValueError("frozen parsing mask support differs from quality")


def _validate_quality(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _QUALITY_FIELDS:
        raise ValueError("frozen parsing quality schema differs")
    if value["state"] not in {"USABLE", "REVIEW", "UNUSABLE"}:
        raise ValueError("frozen parsing quality state differs")
    for field in ("reasons", "flags"):
        items = value[field]
        if (
            not isinstance(items, list)
            or any(not isinstance(item, str) or not item for item in items)
            or len(items) != len(set(items))
        ):
            raise ValueError(f"frozen parsing quality {field} differ")
    for field in ("semantic_shape_iou", "ownership_retention"):
        item = value[field]
        if not _finite_number(item) or not 0.0 <= item <= 1.0:
            raise ValueError(f"frozen parsing quality {field} differs")
    for field in ("foreground_pixels", "component_count"):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"frozen parsing quality {field} differs")
    if not isinstance(value["touches_source_border"], bool):
        raise TypeError("frozen parsing border quality must be boolean")
    ParsedAnimalQuality(
        state=value["state"],
        reasons=tuple(value["reasons"]),
        flags=tuple(value["flags"]),
        semantic_shape_iou=value["semantic_shape_iou"],
        ownership_retention=value["ownership_retention"],
        foreground_pixels=value["foreground_pixels"],
        component_count=value["component_count"],
        touches_source_border=value["touches_source_border"],
    )


def _unpack_instance(value: dict[str, Any]) -> ParsedAnimalInstance:
    quality = value["quality"]
    return ParsedAnimalInstance(
        instance_index=value["instance_index"],
        query_index=value["query_index"],
        class_id=value["class_id"],
        class_name=value["class_name"],
        class_score=value["class_score"],
        detector_box_xyxy=tuple(value["detector_box_xyxy"]),
        refinement_box_xyxy=tuple(value["refinement_box_xyxy"]),
        mask_box_xyxy=(
            None if value["mask_box_xyxy"] is None else tuple(value["mask_box_xyxy"])
        ),
        instance_probability=_unpack_array(
            value["instance_probability"],
            expected_dtype="float32-le",
            expected_shape=tuple(value["instance_probability"]["shape"]),
        ),
        foreground_probability=_unpack_array(
            value["foreground_probability"],
            expected_dtype="float32-le",
            expected_shape=tuple(value["foreground_probability"]["shape"]),
        ),
        ownership_probability=_unpack_array(
            value["ownership_probability"],
            expected_dtype="float32-le",
            expected_shape=tuple(value["ownership_probability"]["shape"]),
        ),
        hard_mask=_unpack_array(
            value["hard_mask"],
            expected_dtype="uint8",
            expected_shape=tuple(value["hard_mask"]["shape"]),
        ),
        quality=ParsedAnimalQuality(
            state=quality["state"],
            reasons=tuple(quality["reasons"]),
            flags=tuple(quality["flags"]),
            semantic_shape_iou=quality["semantic_shape_iou"],
            ownership_retention=quality["ownership_retention"],
            foreground_pixels=quality["foreground_pixels"],
            component_count=quality["component_count"],
            touches_source_border=quality["touches_source_border"],
        ),
    )


def _unpack_array(
    value: object, *, expected_dtype: str, expected_shape: tuple[int, int]
) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != _ARRAY_FIELDS:
        raise ValueError("frozen parsing packed array schema differs")
    if value["dtype"] != expected_dtype or value["encoding"] != "BASE64_ZLIB_C_ORDER":
        raise ValueError("frozen parsing packed array encoding differs")
    if value["shape"] != list(expected_shape):
        raise ValueError("frozen parsing packed array shape differs")
    _require_sha256(value["raw_sha256"], "frozen parsing raw array")
    item_size = 4 if expected_dtype == "float32-le" else 1
    expected_bytes = math.prod(expected_shape) * item_size
    data = value["data"]
    if not isinstance(data, str) or len(data) > ((expected_bytes + 128) * 4 // 3 + 4):
        raise ValueError("frozen parsing packed array size differs")
    try:
        compressed = base64.b64decode(data, validate=True)
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected_bytes + 1)
    except (binascii.Error, zlib.error) as exc:
        raise ValueError("frozen parsing packed array payload differs") from exc
    if (
        len(raw) != expected_bytes
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or hashlib.sha256(raw).hexdigest() != value["raw_sha256"]
    ):
        raise ValueError("frozen parsing packed array digest or length differs")
    dtype = np.dtype("<f4") if expected_dtype == "float32-le" else np.dtype(np.uint8)
    array = np.frombuffer(raw, dtype=dtype).reshape(expected_shape).copy(order="C")
    if expected_dtype == "float32-le":
        array = array.astype(np.float32, copy=False)
        if (
            not np.isfinite(array).all()
            or float(array.min()) < 0.0
            or float(array.max()) > 1.0
        ):
            raise ValueError("frozen parsing probability values differ")
    return array


def _pixel_box(value: object, width: int, height: int, label: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"frozen parsing {label} differs")
    x1, y1, x2, y2 = value
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError(f"frozen parsing {label} exceeds source")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


__all__ = [
    "CACHE_BUNDLE_SCHEMA",
    "CACHE_SCHEMA",
    "FROZEN_PARSING_BINDING_SCHEMA",
    "FROZEN_PARSING_SCHEMA",
    "LEGACY_CACHE_BUNDLE_SCHEMA",
    "LEGACY_CACHE_SCHEMA",
    "build_body_mask_observation",
    "build_body_observation",
    "build_frozen_parsing_binding",
    "build_full_segment_cache",
    "freeze_animal_parsing_prediction",
    "thaw_animal_parsing_prediction",
    "validate_frozen_animal_parsing",
    "validate_frozen_parsing_binding",
    "validate_full_segment_cache",
    "validate_full_segment_cache_bundle",
]
