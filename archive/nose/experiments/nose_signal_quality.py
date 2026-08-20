"""Label-blind signal and mask diagnostics for native Nose/muzzle crops."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from parsing.export.regions.manifest import frontality_components_from_keypoints


NOSE_SIGNAL_QUALITY_SCHEMA = "identification.nose.nose_signal_quality_report.v1"
_SIGNAL_FIELDS = (
    "blur_laplacian_variance",
    "blur_score",
    "saturation_mean",
    "clipped_pixel_fraction",
    "specular_fraction",
    "contrast_rms",
    "contrast_score",
    "jpeg_blocking_score",
    "noise_score",
    "native_short_side",
    "mask_uncertainty",
    "detector_confidence",
    "frontality",
)


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"available": False, "reason": "NO_VALUES", "count": 0}
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("signal summary values must be finite")
    quantiles = np.quantile(array, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "available": True,
        "count": len(array),
        "minimum": float(array.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
    }


def _fixed_bins(values: list[float], edges: tuple[float, ...]) -> list[dict[str, Any]]:
    counts = np.histogram(np.asarray(values, dtype=np.float64), bins=np.asarray(edges))[0]
    return [
        {
            "lower_inclusive": edges[index],
            "upper_exclusive": (
                edges[index + 1] if math.isfinite(edges[index + 1]) else None
            ),
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]


def mask_topology(binary_mask: np.ndarray, soft_mask: np.ndarray) -> dict[str, float | int]:
    """Measure support integrity without asserting that the mask is anatomically correct."""

    binary = np.asarray(binary_mask)
    soft = np.asarray(soft_mask)
    if binary.ndim != 2 or soft.shape != binary.shape or binary.size == 0:
        raise ValueError("binary and soft masks must be non-empty same-shaped 2D arrays")
    if not np.isfinite(binary).all() or not np.isfinite(soft).all():
        raise ValueError("mask arrays must be finite")
    support = binary >= 128
    area = int(support.sum())
    area_fraction = area / support.size
    if area == 0:
        return {
            "area_fraction": 0.0,
            "component_count": 0,
            "largest_component_fraction": 0.0,
            "border_support_fraction": 0.0,
            "compactness": 0.0,
            "centroid_offset": 1.0,
            "soft_boundary_fraction": float(np.mean((soft > 5) & (soft < 250))),
        }

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_index = int(np.argmax(component_areas)) + 1
    largest = labels == largest_index
    contours, _ = cv2.findContours(
        largest.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    compactness = 0.0 if perimeter <= 0.0 else min(
        1.0, 4.0 * math.pi * float(component_areas.max()) / (perimeter * perimeter)
    )
    border = np.concatenate((support[0], support[-1], support[:, 0], support[:, -1]))
    center_x = (binary.shape[1] - 1) / 2.0
    center_y = (binary.shape[0] - 1) / 2.0
    centroid_x, centroid_y = centroids[largest_index]
    half_diagonal = max(math.hypot(center_x, center_y), 1.0)
    return {
        "area_fraction": float(area_fraction),
        "component_count": int(component_count - 1),
        "largest_component_fraction": float(component_areas.max() / area),
        "border_support_fraction": float(border.mean()),
        "compactness": float(compactness),
        "centroid_offset": float(
            min(1.0, math.hypot(centroid_x - center_x, centroid_y - center_y) / half_diagonal)
        ),
        "soft_boundary_fraction": float(np.mean((soft > 5) & (soft < 250))),
    }


def analyze_native_nose_signal(
    manifest: Mapping[str, Any], *, artifacts_root: Path | None = None
) -> dict[str, Any]:
    """Aggregate quality attrition and mask topology without using identity labels."""

    if manifest.get("schema_version") != "archive.nose.yt_native_nose_manifest.v1":
        raise ValueError("native Nose signal audit requires a validated YT v1 manifest")
    records = manifest.get("records")
    policy = manifest.get("policy")
    if not isinstance(records, list) or not records or not isinstance(policy, Mapping):
        raise ValueError("native Nose signal audit input differs")
    root = None
    if artifacts_root is not None:
        root = artifacts_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("artifacts_root must be a directory")

    states: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    signals: dict[str, list[float]] = {name: [] for name in _SIGNAL_FIELDS}
    components: dict[str, list[float]] = {
        name: []
        for name in (
            "symmetry",
            "level",
            "geometric_frontality",
            "anchor_confidence",
            "combined_frontality",
            "nose_keypoint_confidence",
        )
    }
    topology: dict[str, list[float]] = {}
    combined_mismatch = 0.0
    localized = 0
    hard_frontality_failures = 0
    confidence_limited_failures = 0
    geometry_limited_failures = 0
    continuous_frontality_pool = 0

    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("native Nose signal audit record differs")
        states[str(record["record_state"])] += 1
        flags.update(record["quality_flags"])
        quality = record["quality"]
        for name in _SIGNAL_FIELDS:
            value = quality[name]
            if value is not None:
                signals[name].append(float(value))
        if record["keypoints"] is None:
            continue
        localized += 1
        points = [
            [point["normalized_x"], point["normalized_y"], point["confidence"]]
            for point in record["keypoints"]
        ]
        decomposition = frontality_components_from_keypoints(points)
        for name, value in decomposition.items():
            components[name].append(value)
        nose_confidence = float(np.mean([point[2] for point in points[2:]]))
        components["nose_keypoint_confidence"].append(nose_confidence)
        combined_mismatch = max(
            combined_mismatch,
            abs(decomposition["combined_frontality"] - float(quality["frontality"])),
        )
        threshold = float(policy["minimum_frontality"])
        if decomposition["combined_frontality"] < threshold:
            hard_frontality_failures += 1
            if decomposition["geometric_frontality"] >= threshold:
                confidence_limited_failures += 1
            else:
                geometry_limited_failures += 1
        if (
            float(quality["detector_confidence"]) >= float(policy["minimum_detector_confidence"])
            and int(quality["native_short_side"]) >= int(policy["minimum_native_short_side"])
            and float(quality["mask_uncertainty"]) <= float(policy["maximum_mask_uncertainty"])
        ):
            continuous_frontality_pool += 1

        if root is not None:
            binary_path = root / record["binary_mask_path"]
            soft_path = root / record["soft_mask_path"]
            with Image.open(binary_path) as opened:
                binary = np.asarray(opened.convert("L"))
            with Image.open(soft_path) as opened:
                soft = np.asarray(opened.convert("L"))
            values = mask_topology(binary, soft)
            for name, value in values.items():
                topology.setdefault(name, []).append(float(value))

    frontality_edges = (-1e-12, 0.25, 0.5, 0.75, 1.000000000001)
    resolution_edges = (-1e-12, 8.0, 16.0, 32.0, 64.0, float("inf"))
    report = {
        "schema_version": NOSE_SIGNAL_QUALITY_SCHEMA,
        "status": "PASS_LABEL_BLIND_NOSE_SIGNAL_AUDIT",
        "interpretation": (
            "LABEL_BLIND_SIGNAL_AND_MASK_DIAGNOSTIC_NOT_IDENTITY_VALUE_OR_BIOMETRIC_VALIDATION"
        ),
        "population": {
            "record_count": len(records),
            "localized_record_count": localized,
            "state_counts": dict(sorted(states.items())),
            "quality_flag_counts": dict(sorted(flags.items())),
        },
        "frontality_decomposition": {
            **{name: _summary(values) for name, values in components.items()},
            "maximum_legacy_score_mismatch": combined_mismatch,
            "hard_frontality_failure_count": hard_frontality_failures,
            "confidence_limited_failure_count": confidence_limited_failures,
            "geometry_limited_failure_count": geometry_limited_failures,
            "geometric_frontality_fixed_bins": _fixed_bins(
                components["geometric_frontality"], frontality_edges
            ),
            "combined_frontality_fixed_bins": _fixed_bins(
                components["combined_frontality"], frontality_edges
            ),
        },
        "policy_attrition": {
            "original_available_count": states["AVAILABLE"],
            "localized_count": localized,
            "localized_passing_non_frontality_gates_count": continuous_frontality_pool,
            "frontality_policy_is_counterfactual_only": True,
        },
        "signal_quality": {name: _summary(values) for name, values in signals.items()},
        "native_resolution_fixed_bins": _fixed_bins(
            signals["native_short_side"], resolution_edges
        ),
        "mask_topology": (
            {"available": False, "reason": "ARTIFACTS_ROOT_NOT_PROVIDED"}
            if root is None
            else {
                "available": True,
                "sample_count": localized,
                **{name: _summary(values) for name, values in topology.items()},
            }
        ),
        "limitations": [
            "No identity labels or retrieval scores are used.",
            "Mask topology measures support integrity, not anatomical correctness.",
            "Counterfactual coverage does not admit low-frontality evidence for training or evaluation.",
        ],
    }
    return report


__all__ = [
    "NOSE_SIGNAL_QUALITY_SCHEMA",
    "analyze_native_nose_signal",
    "mask_topology",
]
