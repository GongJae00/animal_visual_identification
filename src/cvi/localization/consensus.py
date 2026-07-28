"""Teacher consensus: error correlation, weighted aggregation, admission gating.

All computations are deterministic and annotation-free — they operate on
the predictions themselves to measure agreement, not against ground truth.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from cvi.localization.types import DetectionBox, Keypoint, KeypointSet, LocalizationResult


@dataclass(frozen=True, slots=True)
class FailureVector:
    image_id: str
    model_name: str
    missed_dog: bool = False
    wrong_instance: bool = False
    bbox_truncation: bool = False
    low_confidence: bool = False



def compute_error_correlation(
    results_by_model: dict[str, tuple[LocalizationResult, ...]],
) -> dict[str, np.ndarray]:
    """Return per-model-pair error correlation matrix.

    Each entry measures how often two models fail on the same image
    vs fail independently.  High correlation + similar performance
    → one model can be removed from the ensemble.
    """

    models = sorted(results_by_model)
    n_models = len(models)
    if n_models < 2:
        raise ValueError("error correlation requires at least two models")

    image_ids = sorted({r.image_id for results in results_by_model.values() for r in results})

    has_dog = np.zeros((n_models, len(image_ids)), dtype=bool)
    for model_idx, model in enumerate(models):
        for result in results_by_model[model]:
            img_idx = image_ids.index(result.image_id)
            has_dog[model_idx, img_idx] = len(result.dog_boxes) > 0

    correlation = np.eye(n_models)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            both = (has_dog[i] & has_dog[j]).sum()
            either = (has_dog[i] | has_dog[j]).sum()
            agreement = both / max(either, 1)
            correlation[i, j] = agreement
            correlation[j, i] = agreement
    return {"models": models, "correlation": correlation}


def weighted_box_fusion(
    boxes: list[DetectionBox],
    *,
    weights: list[float] | None = None,
) -> DetectionBox:
    """Weighted average of bounding boxes. High-confidence boxes have more influence."""

    if not boxes:
        raise ValueError("box fusion requires at least one box")
    if weights is None:
        weights = [b.confidence for b in boxes]
    if len(weights) != len(boxes) or any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative and match box count")

    total = max(sum(weights), 1e-6)
    x1 = sum(b.x1 * w for b, w in zip(boxes, weights, strict=True)) / total
    y1 = sum(b.y1 * w for b, w in zip(boxes, weights, strict=True)) / total
    x2 = sum(b.x2 * w for b, w in zip(boxes, weights, strict=True)) / total
    y2 = sum(b.y2 * w for b, w in zip(boxes, weights, strict=True)) / total
    confidence = min(1.0, sum(w for w in weights) / len(weights))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("fused box collapsed to zero area")
    return DetectionBox(x1, y1, x2, y2, confidence)


def robust_weighted_keypoint(
    points: list[tuple[float, float, float]],
    *,
    weights: list[float] | None = None,
) -> tuple[float, float, float]:
    """Robust weighted mean of keypoints. Outliers trimmed via median distance."""

    if not points:
        raise ValueError("keypoint fusion requires at least one point")
    if any(p[2] < 0 for p in points):
        raise ValueError("keypoint confidence must be non-negative")

    if weights is None:
        weights = [p[2] for p in points]
    if len(weights) != len(points):
        raise ValueError("weights must match point count")

    total_weight = max(sum(weights), 1e-6)
    x = sum(p[0] * w for p, w in zip(points, weights, strict=True)) / total_weight
    y = sum(p[1] * w for p, w in zip(points, weights, strict=True)) / total_weight
    distances = np.asarray(
        [np.linalg.norm((p[0] - x, p[1] - y)) for p in points]
    )
    median_dist = float(np.median(distances))
    if median_dist < 1e-6:
        return (x, y, float(np.mean(weights)))

    robust_weights = [
        w * min(1.0, 1.5 * median_dist / max(d, 1e-8)) for w, d in zip(weights, distances)
    ]
    robust_total = max(sum(robust_weights), 1e-6)
    robust_x = sum(p[0] * rw for p, rw in zip(points, robust_weights, strict=True)) / robust_total
    robust_y = sum(p[1] * rw for p, rw in zip(points, robust_weights, strict=True)) / robust_total
    confidence = float(np.clip(np.mean(robust_weights), 0.0, 1.0))

    return (robust_x, robust_y, confidence)


def consensus_admission(
    agreement_score: float,
    confidence: float,
    *,
    high_threshold: float = 0.80,
    low_threshold: float = 0.50,
) -> str:
    """Classify a consensus prediction as ACCEPT / REVIEW / REJECT."""

    if not 0.0 <= agreement_score <= 1.0 or not 0.0 <= confidence <= 1.0:
        raise ValueError("agreement and confidence must be in [0, 1]")

    if confidence >= high_threshold and agreement_score >= high_threshold:
        return "ACCEPT"
    if confidence < low_threshold or agreement_score < low_threshold:
        return "REJECT"
    return "REVIEW"


def consensus_dog_bbox(
    results: tuple[LocalizationResult, ...],
) -> tuple[DetectionBox | None, dict[str, Any]]:
    """Produce one consensus dog bbox from multiple model predictions."""

    boxes: list[DetectionBox] = []
    for result in results:
        boxes.extend(result.dog_boxes)

    if not boxes:
        return None, {"status": "NO_DOG_DETECTED", "model_count": 0, "box_count": 0}

    model_count = len({r.model_name for r in results})
    agreement = len(boxes) / max(model_count, 1)

    try:
        fused = weighted_box_fusion(boxes)
        admission = consensus_admission(
            min(1.0, agreement), fused.confidence
        )
        return fused, {
            "status": "FUSED",
            "admission": admission,
            "model_count": model_count,
            "box_count": len(boxes),
            "agreement": agreement,
        }
    except ValueError:
        return boxes[0], {
            "status": "SINGLE_BOX",
            "admission": "REVIEW",
            "model_count": model_count,
            "box_count": 1,
            "agreement": 1.0 / model_count,
        }


def consensus_keypoint(
    results: tuple[LocalizationResult, ...],
    keypoint_name: str,
    source: str = "body_keypoints",
) -> tuple[Keypoint | None, dict[str, Any]]:
    """Produce one consensus keypoint from multiple model predictions."""

    points: list[tuple[float, float, float]] = []
    for result in results:
        sets = (
            result.body_keypoints
            if source == "body_keypoints"
            else result.face_landmarks
        )
        for kps in sets:
            kp = kps.named(keypoint_name)
            if kp is not None and kp.confidence > 0.0:
                points.append((kp.x, kp.y, kp.confidence))

    if len(points) < 2:
        return None, {"status": "INSUFFICIENT_POINTS", "point_count": len(points)}

    x, y, confidence = robust_weighted_keypoint(points)
    agreement = len(points) / max(
        len({r.model_name for r in results}), 1
    )
    admission = consensus_admission(min(1.0, agreement), confidence)

    return (
        Keypoint(x, y, confidence),
        {
            "status": "CONSENSUS",
            "admission": admission,
            "point_count": len(points),
            "agreement": agreement,
        },
    )


__all__ = [
    "FailureVector",
    "compute_error_correlation",
    "consensus_admission",
    "consensus_dog_bbox",
    "consensus_keypoint",
    "robust_weighted_keypoint",
    "weighted_box_fusion",
]
