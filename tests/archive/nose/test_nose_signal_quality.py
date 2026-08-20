from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from archive.nose.experiments.nose_signal_quality import (
    analyze_native_nose_signal,
    mask_topology,
)
from parsing.export.regions.manifest import (
    frontality_components_from_keypoints,
    frontality_from_keypoints,
)

def _points(confidence: float = 0.8) -> list[list[float]]:
    return [
        [0.3, 0.3, confidence],
        [0.7, 0.3, confidence],
        [0.5, 0.5, confidence],
        [0.5, 0.7, confidence],
        [0.4, 0.6, confidence],
        [0.6, 0.6, confidence],
        [0.35, 0.6, confidence],
        [0.65, 0.6, confidence],
    ]

def _record(*, confidence: float, state: str) -> dict:
    points = _points(confidence)
    frontality = frontality_from_keypoints(points)
    return {
        "record_state": state,
        "quality_flags": [] if state == "AVAILABLE" else ["LOW_FRONTALITY"],
        "keypoints": [
            {
                "normalized_x": point[0],
                "normalized_y": point[1],
                "confidence": point[2],
            }
            for point in points
        ],
        "binary_mask_path": "binary.png",
        "soft_mask_path": "soft.png",
        "quality": {
            "blur_laplacian_variance": 100.0,
            "blur_score": 0.2,
            "saturation_mean": 0.3,
            "clipped_pixel_fraction": 0.01,
            "specular_fraction": 0.02,
            "contrast_rms": 20.0,
            "contrast_score": 0.4,
            "jpeg_blocking_score": 0.1,
            "noise_score": 0.05,
            "native_short_side": 48,
            "mask_uncertainty": 0.1,
            "detector_confidence": confidence,
            "frontality": frontality,
        },
    }

def test_frontality_components_separate_geometry_and_confidence() -> None:
    components = frontality_components_from_keypoints(_points(0.4))

    assert components["geometric_frontality"] == pytest.approx(1.0)
    assert components["anchor_confidence"] == pytest.approx(0.4)
    assert components["combined_frontality"] == pytest.approx(0.4)
    assert frontality_from_keypoints(_points(0.4)) == pytest.approx(0.4)

    yawed = _points()
    yawed[2][0] = yawed[3][0] = 0.7
    assert frontality_components_from_keypoints(yawed)["geometric_frontality"] == 0.0

def test_label_blind_audit_exposes_confidence_limited_attrition() -> None:
    manifest = {
        "schema_version": "archive.nose.yt_native_nose_manifest.v1",
        "policy": {
            "minimum_detector_confidence": 0.3,
            "minimum_frontality": 0.5,
            "minimum_native_short_side": 32,
            "maximum_mask_uncertainty": 0.5,
        },
        "records": [
            _record(confidence=0.8, state="AVAILABLE"),
            _record(confidence=0.4, state="LOW_QUALITY"),
        ],
    }

    report = analyze_native_nose_signal(manifest)

    assert report["population"]["localized_record_count"] == 2
    assert report["frontality_decomposition"]["hard_frontality_failure_count"] == 1
    assert report["frontality_decomposition"]["confidence_limited_failure_count"] == 1
    assert report["frontality_decomposition"]["geometry_limited_failure_count"] == 0
    assert report["policy_attrition"]["localized_passing_non_frontality_gates_count"] == 2
    assert report["mask_topology"]["available"] is False

def test_mask_topology_reports_fragmentation_and_support_geometry() -> None:
    binary = np.zeros((20, 20), dtype=np.uint8)
    binary[4:12, 4:12] = 255
    binary[16:18, 16:18] = 255
    soft = binary.copy()
    soft[3, 4:12] = 64

    result = mask_topology(binary, soft)

    assert result["component_count"] == 2
    assert result["largest_component_fraction"] == pytest.approx(64 / 68)
    assert 0.0 < result["area_fraction"] < 1.0
    assert result["border_support_fraction"] == 0.0
    assert result["soft_boundary_fraction"] > 0.0

def test_frontality_component_validation_matches_legacy_contract() -> None:
    invalid = deepcopy(_points())
    invalid[0][2] = 2.0
    with pytest.raises(ValueError, match=r"normalized to \[0, 1\]"):
        frontality_components_from_keypoints(invalid)
