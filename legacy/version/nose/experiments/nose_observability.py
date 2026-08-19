"""Manifest-only Nose crop observability and correspondence proxies."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from legacy.version.common.experiments.sibetan_evidence import (
    BUNDLE_SCHEMA as SIBETAN_V1_BUNDLE_SCHEMA,
)
from legacy.version.common.experiments.sibetan_evidence import (
    V2_BUNDLE_SCHEMA as SIBETAN_V2_BUNDLE_SCHEMA,
)
from legacy.version.common.experiments.sibetan_evidence import (
    validate_evidence_bundle,
    validate_evidence_bundle_v2,
)
from parsing.nose_region.native_yt import (
    BUNDLE_SCHEMA as YT_BUNDLE_SCHEMA,
)
from parsing.nose_region.native_yt import (
    validate_manifest_bundle,
)

REPORT_SCHEMA = "cvi.nose_observability_proxy_report.v1"
REPORT_BUNDLE_SCHEMA = "cvi.nose_observability_proxy_report_bundle.v1"
REFERENCE_RESIZE_SIDE = 224
SUPPORTED_BUNDLE_SCHEMAS = (
    YT_BUNDLE_SCHEMA,
    SIBETAN_V1_BUNDLE_SCHEMA,
    SIBETAN_V2_BUNDLE_SCHEMA,
)

_UNIT_INTERVAL_QUALITY_FIELDS = (
    "blur_score",
    "saturation_mean",
    "clipped_pixel_fraction",
    "specular_fraction",
    "contrast_score",
    "jpeg_blocking_score",
    "noise_score",
    "mask_uncertainty",
)


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"available": False, "count": 0, "reason": "NO_VALUES"}
    ordered = sorted(_number(value, "summary value") for value in values)

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "available": True,
        "count": len(ordered),
        "minimum": ordered[0],
        "p05": quantile(0.05),
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "p75": quantile(0.75),
        "p95": quantile(0.95),
        "maximum": ordered[-1],
        "mean": math.fsum(ordered) / len(ordered),
    }


def _unit_strata(values: Sequence[float]) -> dict[str, int]:
    counts = Counter(
        "LT_0_25"
        if value < 0.25
        else "GE_0_25_LT_0_50"
        if value < 0.5
        else "GE_0_50_LT_0_75"
        if value < 0.75
        else "GE_0_75_LE_1_00"
        for value in values
    )
    return {
        label: counts[label]
        for label in (
            "LT_0_25",
            "GE_0_25_LT_0_50",
            "GE_0_50_LT_0_75",
            "GE_0_75_LE_1_00",
        )
    }


def _resolution_strata(values: Sequence[int]) -> dict[str, int]:
    counts = Counter(
        "LT_16_PX"
        if value < 16
        else "GE_16_LT_32_PX"
        if value < 32
        else "GE_32_LT_64_PX"
        if value < 64
        else "GE_64_LT_96_PX"
        if value < 96
        else "GE_96_LT_160_PX"
        if value < 160
        else "GE_160_LT_224_PX"
        if value < 224
        else "GE_224_PX"
        for value in values
    )
    labels = (
        "LT_16_PX",
        "GE_16_LT_32_PX",
        "GE_32_LT_64_PX",
        "GE_64_LT_96_PX",
        "GE_96_LT_160_PX",
        "GE_160_LT_224_PX",
        "GE_224_PX",
    )
    return {label: counts[label] for label in labels}


def _require_lineage(value: object, name: str, *, hashes_are_values: bool) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} required lineage is missing")
    if hashes_are_values:
        hashes = list(value.values())
    else:
        hashes = []
        stack = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, Mapping):
                for key, child in current.items():
                    if isinstance(key, str) and key.lower().endswith("sha256"):
                        hashes.append(child)
                    stack.append(child)
            elif isinstance(current, list):
                stack.extend(current)
    if not hashes or any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in hashes
    ):
        raise ValueError(f"{name} required SHA-256 lineage is missing or malformed")


def _reject_duplicate_tokens(bundle: Mapping[str, Any], field: str) -> None:
    manifest = bundle.get("manifest")
    records = manifest.get("records") if isinstance(manifest, Mapping) else None
    if not isinstance(records, list):
        return
    tokens = [record.get(field) for record in records if isinstance(record, Mapping)]
    if len(tokens) == len(records) and len(tokens) != len(set(tokens)):
        raise ValueError("evidence manifest repeats a sample token")


def _adapt_bundle(bundle: object) -> tuple[str, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(bundle, dict):
        raise TypeError("observability audit input must be a JSON object")
    schema = bundle.get("schema_version")
    if schema not in SUPPORTED_BUNDLE_SCHEMAS:
        raise ValueError(f"unsupported Nose evidence bundle schema: {schema!r}")

    rows: list[dict[str, Any]] = []
    if schema == YT_BUNDLE_SCHEMA:
        _reject_duplicate_tokens(bundle, "sample_token")
        manifest = validate_manifest_bundle(bundle)
        _require_lineage(manifest["input_sha256s"], "native YT input", hashes_are_values=True)
        if not isinstance(manifest["tool_provenance"], Mapping) or not manifest["tool_provenance"]:
            raise ValueError("native YT required tool provenance is missing")
        for record in manifest["records"]:
            quality = record["quality"]
            crop_size = None
            if record["crop_width"] is not None or record["crop_height"] is not None:
                crop_size = (
                    _positive_dimension(record["crop_width"], "crop_width"),
                    _positive_dimension(record["crop_height"], "crop_height"),
                )
            rows.append(
                {
                    "token": record["sample_token"],
                    "group": record["track_token"],
                    "state": record["record_state"],
                    "reasons": list(record["quality_flags"]),
                    "source_size": (record["source_width"], record["source_height"]),
                    "crop_size": crop_size,
                    "box": record["nose_box_xyxy"],
                    "confidence": quality["detector_confidence"],
                    "frontality": quality["frontality"],
                    "quality": quality,
                }
            )
        branches = {
            "nose": {
                "state_counts": dict(sorted(Counter(row["state"] for row in rows).items())),
                "rejection_reason_counts": dict(
                    sorted(Counter(reason for row in rows for reason in row["reasons"]).items())
                ),
            }
        }
        return "YT_NATIVE_NOSE_V1", manifest, rows, branches

    _reject_duplicate_tokens(bundle, "sample_id")
    if schema == SIBETAN_V1_BUNDLE_SCHEMA:
        manifest = validate_evidence_bundle(bundle)
        format_name = "SIBETAN_MULTIEVIDENCE_V1"
    else:
        manifest = validate_evidence_bundle_v2(bundle)
        format_name = "SIBETAN_MULTIEVIDENCE_V2"
    _require_lineage(manifest["input_bindings"], "SiBeTan input", hashes_are_values=False)
    for record in manifest["records"]:
        nose = record["nose"]
        crop_size = None
        if nose["crop_width"] is not None or nose["crop_height"] is not None:
            crop_size = (
                _positive_dimension(nose["crop_width"], "nose.crop_width"),
                _positive_dimension(nose["crop_height"], "nose.crop_height"),
            )
        rows.append(
            {
                "token": record["sample_id"],
                "group": record["source_group_id"],
                "state": nose["state"],
                "reasons": list(nose["reasons"]),
                "source_size": (record["source_width"], record["source_height"]),
                "crop_size": crop_size,
                "box": nose.get("source_box_xyxy"),
                "confidence": nose.get("localizer_confidence"),
                "frontality": nose.get("frontality"),
                "quality": nose.get("quality", {}),
            }
        )
    branches = {
        branch: {
            "state_counts": dict(
                sorted(Counter(record[branch]["state"] for record in manifest["records"]).items())
            ),
            "rejection_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for record in manifest["records"]
                        for reason in record[branch]["reasons"]
                    ).items()
                )
            ),
        }
        for branch in ("face", "nose")
    }
    return format_name, manifest, rows, branches


def _box_geometry(row: Mapping[str, Any]) -> dict[str, float] | None:
    box = row["box"]
    if box is None:
        return None
    source_width = _positive_dimension(row["source_size"][0], "source_width")
    source_height = _positive_dimension(row["source_size"][1], "source_height")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in box)
        or not (0 <= box[0] < box[2] <= source_width)
        or not (0 <= box[1] < box[3] <= source_height)
    ):
        raise ValueError("nose source-coordinate box dimensions are malformed")
    width = box[2] - box[0]
    height = box[3] - box[1]
    return {
        "center_x": (box[0] + box[2]) / (2.0 * source_width),
        "center_y": (box[1] + box[3]) / (2.0 * source_height),
        "width": width / source_width,
        "height": height / source_height,
        "area": width * height / (source_width * source_height),
        "aspect_ratio": width / height,
        "pixel_width": float(width),
        "pixel_height": float(height),
    }


def audit_nose_observability(bundle: object) -> dict[str, Any]:
    """Build label-blind pixel-extent proxies from a supported evidence bundle."""

    input_format, manifest, rows, branch_availability = _adapt_bundle(bundle)
    crop_widths: list[float] = []
    crop_heights: list[float] = []
    crop_short_sides: list[int] = []
    crop_areas: list[float] = []
    crop_aspects: list[float] = []
    width_scale: list[float] = []
    height_scale: list[float] = []
    short_side_scale: list[float] = []
    upsampling: list[float] = []
    anisotropy: list[float] = []
    confidences: list[float] = []
    frontalities: list[float] = []
    quality_values: dict[str, list[float]] = {
        field: [] for field in _UNIT_INTERVAL_QUALITY_FIELDS
    }
    geometry_by_group: dict[str, list[dict[str, float]]] = defaultdict(list)
    topology_proxy_counts: Counter[str] = Counter()

    for row in rows:
        source_width, source_height = row["source_size"]
        _positive_dimension(source_width, "source_width")
        _positive_dimension(source_height, "source_height")
        geometry = _box_geometry(row)
        if geometry is not None:
            geometry_by_group[row["group"]].append(geometry)

        crop_size = row["crop_size"]
        if crop_size is not None:
            width, height = crop_size
            crop_widths.append(float(width))
            crop_heights.append(float(height))
            crop_short_sides.append(min(width, height))
            crop_areas.append(float(width * height))
            crop_aspects.append(width / height)
            x_scale = REFERENCE_RESIZE_SIDE / width
            y_scale = REFERENCE_RESIZE_SIDE / height
            width_scale.append(x_scale)
            height_scale.append(y_scale)
            short_side_scale.append(REFERENCE_RESIZE_SIDE / min(width, height))
            upsampling.append(max(1.0, REFERENCE_RESIZE_SIDE / min(width, height)))
            anisotropy.append(max(x_scale, y_scale) / min(x_scale, y_scale))

        proxy_size = (
            None
            if geometry is None and crop_size is None
            else min(crop_size)
            if geometry is None
            else int(min(geometry["pixel_width"], geometry["pixel_height"]))
        )
        category = (
            "TOPOLOGY_OBSERVABILITY_PROXY_NO_PIXEL_EXTENT"
            if proxy_size is None
            else "TOPOLOGY_OBSERVABILITY_PROXY_NATIVE_SHORT_SIDE_LT_32_PX"
            if proxy_size < 32
            else "TOPOLOGY_OBSERVABILITY_PROXY_NATIVE_SHORT_SIDE_32_TO_63_PX"
            if proxy_size < 64
            else "TOPOLOGY_OBSERVABILITY_PROXY_NATIVE_SHORT_SIDE_64_TO_127_PX"
            if proxy_size < 128
            else "TOPOLOGY_OBSERVABILITY_PROXY_NATIVE_SHORT_SIDE_GE_128_PX"
        )
        topology_proxy_counts[category] += 1

        for field, destination in (
            ("confidence", confidences),
            ("frontality", frontalities),
        ):
            value = row[field]
            if value is not None:
                number = _number(value, field, minimum=0.0)
                if number > 1.0:
                    raise ValueError(f"{field} must be in [0, 1]")
                destination.append(number)
        quality = row["quality"]
        if not isinstance(quality, Mapping):
            raise TypeError("nose quality lineage must be an object")
        for field in _UNIT_INTERVAL_QUALITY_FIELDS:
            value = quality.get(field)
            if value is None:
                continue
            number = _number(value, f"quality.{field}", minimum=0.0)
            if number > 1.0:
                raise ValueError(f"quality.{field} must be in [0, 1]")
            quality_values[field].append(number)

    group_ranges: dict[str, list[float]] = {
        "normalized_center_bounding_diagonal": [],
        "normalized_width_range": [],
        "normalized_height_range": [],
        "normalized_area_fold_range": [],
        "aspect_ratio_fold_range": [],
    }
    repeated_groups = 0
    repeated_records = 0
    for geometries in geometry_by_group.values():
        if len(geometries) < 2:
            continue
        repeated_groups += 1
        repeated_records += len(geometries)
        centers_x = [item["center_x"] for item in geometries]
        centers_y = [item["center_y"] for item in geometries]
        widths = [item["width"] for item in geometries]
        heights = [item["height"] for item in geometries]
        areas = [item["area"] for item in geometries]
        aspects = [item["aspect_ratio"] for item in geometries]
        group_ranges["normalized_center_bounding_diagonal"].append(
            math.hypot(max(centers_x) - min(centers_x), max(centers_y) - min(centers_y))
        )
        group_ranges["normalized_width_range"].append(max(widths) - min(widths))
        group_ranges["normalized_height_range"].append(max(heights) - min(heights))
        group_ranges["normalized_area_fold_range"].append(max(areas) / min(areas))
        group_ranges["aspect_ratio_fold_range"].append(max(aspects) / min(aspects))

    coordinate_count = sum(len(values) for values in geometry_by_group.values())
    correspondence_available = repeated_groups > 0
    topology_labels = (
        "TOPOLOGY_OBSERVABILITY_PROXY_NO_PIXEL_EXTENT",
        "TOPOLOGY_OBSERVABILITY_PROXY_NATIVE_SHORT_SIDE_LT_32_PX",
        "TOPOLOGY_OBSERVABILITY_PROXY_NATIVE_SHORT_SIDE_32_TO_63_PX",
        "TOPOLOGY_OBSERVABILITY_PROXY_NATIVE_SHORT_SIDE_64_TO_127_PX",
        "TOPOLOGY_OBSERVABILITY_PROXY_NATIVE_SHORT_SIDE_GE_128_PX",
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_MANIFEST_ONLY_NOSE_OBSERVABILITY_PROXY_AUDIT",
        "interpretation": (
            "PIXEL_EXTENT_AND_CROP_CORRESPONDENCE_PROXIES_ONLY_NOT_VISIBLE_RIDGE_"
            "TOPOLOGY_OR_BIOMETRIC_VALIDATION"
        ),
        "input_contract": {
            "input_format": input_format,
            "input_bundle_schema": bundle["schema_version"],
            "input_manifest_schema": manifest["schema_version"],
            "dataset_name": manifest["dataset_name"],
            "manifest_sha256": bundle["manifest_sha256"],
            "sample_token_field": "sample_token" if input_format == "YT_NATIVE_NOSE_V1" else "sample_id",
            "record_count": len(rows),
            "identity_labels_used": False,
            "image_pixels_inspected": False,
        },
        "availability_and_rejections": branch_availability,
        "native_materialized_crop_size": {
            "materialized_crop_count": len(crop_short_sides),
            "width_pixels": _summary(crop_widths),
            "height_pixels": _summary(crop_heights),
            "short_side_pixels": _summary(crop_short_sides),
            "area_pixels": _summary(crop_areas),
            "aspect_ratio_width_over_height": _summary(crop_aspects),
            "short_side_fixed_strata": _resolution_strata(crop_short_sides),
        },
        "reference_resize_scale_proxy": {
            "reference_canvas": {"width": REFERENCE_RESIZE_SIDE, "height": REFERENCE_RESIZE_SIDE},
            "reference_only_not_asserted_preprocessing": True,
            "does_not_create_or_demonstrate_observed_detail": True,
            "width_scale_factor_to_reference": _summary(width_scale),
            "height_scale_factor_to_reference": _summary(height_scale),
            "short_side_scale_factor_to_reference": _summary(short_side_scale),
            "short_side_upsampling_factor_floor_one": _summary(upsampling),
            "square_resize_anisotropic_scale_ratio": _summary(anisotropy),
        },
        "quality_frontality_confidence_strata": {
            "confidence": {"summary": _summary(confidences), "fixed_strata": _unit_strata(confidences)},
            "frontality": {"summary": _summary(frontalities), "fixed_strata": _unit_strata(frontalities)},
            "quality_metrics": {
                field: {"summary": _summary(values), "fixed_strata": _unit_strata(values)}
                for field, values in quality_values.items()
            },
            "no_synthetic_overall_quality_score": True,
        },
        "crop_correspondence_proxy": {
            "available": correspondence_available,
            "reason": None if correspondence_available else "NO_SOURCE_GROUP_WITH_TWO_COORDINATE_BOXES",
            "coordinate_record_count": coordinate_count,
            "repeated_source_group_count": repeated_groups,
            "records_in_repeated_source_groups": repeated_records,
            "coordinate_normalization": "SOURCE_WIDTH_AND_HEIGHT",
            "grouping_field": "track_token" if input_format == "YT_NATIVE_NOSE_V1" else "source_group_id",
            "metrics_are_crop_instability_proxies_not_anatomical_correspondence": True,
            "group_metric_summaries": {
                name: _summary(values) for name, values in group_ranges.items()
            },
        },
        "topology_observability_proxy": {
            "category_counts": {
                label: topology_proxy_counts[label] for label in topology_labels
            },
            "basis": (
                "NATIVE_SOURCE_COORDINATE_PIXEL_EXTENT_WHERE_PRESENT_OTHERWISE_"
                "MATERIALIZED_CROP_DIMENSIONS"
            ),
            "categories_are_pixel_extent_proxies_only": True,
            "pixels_never_equated_with_visible_ridge_topology": True,
            "manual_annotation_required_for_any_topology_claim": True,
            "claim_boundary": (
                "THESE CATEGORIES NEITHER ESTABLISH NOR REFUTE VISIBLE NASAL RIDGE "
                "TOPOLOGY; ANY TOPOLOGY CLAIM REQUIRES MANUAL ANNOTATION UNDER A "
                "DECLARED PROTOCOL"
            ),
        },
        "limitations": [
            "The audit reads manifest metadata and does not inspect image pixels.",
            "Reference resize factors quantify geometric scaling only and do not imply recovered detail.",
            "Crop-box stability is not anatomical point correspondence.",
            "Pixel dimensions are not evidence that nasal ridge topology is visible.",
            "Manual annotation under a declared protocol is required for every topology claim.",
        ],
    }


__all__ = [
    "REFERENCE_RESIZE_SIDE",
    "REPORT_BUNDLE_SCHEMA",
    "REPORT_SCHEMA",
    "SUPPORTED_BUNDLE_SCHEMAS",
    "audit_nose_observability",
]
