"""Teacher consensus: prediction agreement, aggregation, and review routing.

These annotation-free computations characterize prediction agreement, not
errors or empirical accuracy against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from parsing.roi import compute_iou
from parsing.types import (
    DetectionBox,
    Keypoint,
    LocalizationResult,
)


@dataclass(frozen=True, slots=True)
class FailureVector:
    image_id: str
    model_name: str
    missed_dog: bool = False
    wrong_instance: bool = False
    bbox_truncation: bool = False
    low_confidence: bool = False


@dataclass(frozen=True, slots=True)
class ConsensusDogInstance:
    """One fused instance; ``support_models`` contains teacher artifact SHA256s."""

    bbox: DetectionBox
    support_models: tuple[str, ...]
    agreement: float
    admission: str


def compute_error_correlation(
    results_by_model: dict[str, tuple[LocalizationResult, ...]],
) -> dict[str, np.ndarray]:
    """Return pairwise Jaccard agreement of dog-detection presence.

    The historical function name is retained for compatibility. This
    annotation-free statistic is not error correlation or an accuracy estimate.
    """

    models = sorted(results_by_model)
    n_models = len(models)
    if n_models < 2:
        raise ValueError("detection agreement requires at least two models")

    image_ids = sorted(
        {r.image_id for results in results_by_model.values() for r in results}
    )

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
    distances = np.asarray([np.linalg.norm((p[0] - x, p[1] - y)) for p in points])
    median_dist = float(np.median(distances))
    if median_dist < 1e-6:
        return (x, y, float(np.mean(weights)))

    robust_weights = [
        w * min(1.0, 1.5 * median_dist / max(d, 1e-8))
        for w, d in zip(weights, distances)
    ]
    robust_total = max(sum(robust_weights), 1e-6)
    robust_x = (
        sum(p[0] * rw for p, rw in zip(points, robust_weights, strict=True))
        / robust_total
    )
    robust_y = (
        sum(p[1] * rw for p, rw in zip(points, robust_weights, strict=True))
        / robust_total
    )
    confidence = float(np.clip(np.mean(robust_weights), 0.0, 1.0))

    return (robust_x, robust_y, confidence)


def consensus_admission(
    agreement_score: float,
    confidence: float,
    *,
    high_threshold: float = 0.80,
    low_threshold: float = 0.50,
) -> str:
    """Route a consensus prediction as ACCEPT / REVIEW / REJECT.

    These labels are review-routing states, not ground-truth admission evidence.
    """

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

    instances = consensus_dog_instances(results)
    if not instances:
        return None, {"status": "NO_DOG_DETECTED", "model_count": 0, "box_count": 0}
    if len(instances) != 1:
        return None, {
            "status": "MULTIPLE_INSTANCES",
            "model_count": len({_teacher_identity(result) for result in results}),
            "box_count": sum(len(result.dog_boxes) for result in results),
        }
    instance = instances[0]
    return instance.bbox, {
        "status": "FUSED",
        "admission": instance.admission,
        "model_count": len({_teacher_identity(result) for result in results}),
        "box_count": len(instance.support_models),
        "agreement": instance.agreement,
    }


def consensus_dog_instances(
    results: tuple[LocalizationResult, ...], *, iou_threshold: float = 0.5
) -> tuple[ConsensusDogInstance, ...]:
    """Match and fuse dog instances without merging distinct dogs."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if results and len({result.image_id for result in results}) != 1:
        raise ValueError("consensus inputs must describe one image")
    teacher_ids = _unique_teacher_identities(results)
    candidates = sorted(
        (
            (_teacher_identity(result), box)
            for result in results
            for box in result.dog_boxes
        ),
        key=lambda item: (
            item[0],
            -item[1].confidence,
            item[1].x1,
            item[1].y1,
            item[1].x2,
            item[1].y2,
        ),
    )
    clusters: list[dict[str, DetectionBox]] = []
    for teacher_id, box in candidates:
        matches: list[tuple[float, int]] = []
        for cluster_index, cluster in enumerate(clusters):
            if teacher_id in cluster:
                continue
            overlaps = [compute_iou(box, existing) for existing in cluster.values()]
            if overlaps and min(overlaps) >= iou_threshold:
                matches.append((float(np.mean(overlaps)), cluster_index))
        if matches:
            _, cluster_index = max(matches, key=lambda item: (item[0], -item[1]))
            clusters[cluster_index][teacher_id] = box
        else:
            clusters.append({teacher_id: box})

    instances: list[ConsensusDogInstance] = []
    total_models = max(len(teacher_ids), 1)
    for cluster in clusters:
        support_models = tuple(sorted(cluster))
        fused = weighted_box_fusion([cluster[name] for name in support_models])
        agreement = len(support_models) / total_models
        admission = (
            consensus_admission(agreement, fused.confidence)
            if len(support_models) >= 2
            else "REVIEW"
        )
        instances.append(
            ConsensusDogInstance(fused, support_models, agreement, admission)
        )
    instances.sort(
        key=lambda item: (
            (item.bbox.x1 + item.bbox.x2) / 2.0,
            (item.bbox.y1 + item.bbox.y2) / 2.0,
        )
    )
    return tuple(instances)


def consensus_keypoint(
    results: tuple[LocalizationResult, ...],
    keypoint_name: str,
    source: str = "body_keypoints",
) -> tuple[Keypoint | None, dict[str, Any]]:
    """Produce one consensus keypoint from multiple model predictions."""

    teacher_ids = _unique_teacher_identities(results)
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
    agreement = len(points) / max(len(teacher_ids), 1)
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


def _unique_teacher_identities(
    results: tuple[LocalizationResult, ...],
) -> tuple[str, ...]:
    identities = tuple(_teacher_identity(result) for result in results)
    if len(set(identities)) != len(identities):
        raise ValueError(
            "consensus inputs contain duplicate teacher artifact IDs or caches"
        )
    return tuple(sorted(identities))


def _teacher_identity(result: LocalizationResult) -> str:
    metadata = result.metadata
    if metadata is None or "artifact_sha256" not in metadata:
        raise ValueError("consensus teacher metadata requires artifact_sha256")
    artifact_sha256 = metadata["artifact_sha256"]
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_sha256)
    ):
        raise ValueError("teacher artifact SHA256 must be a lowercase digest")
    return artifact_sha256


__all__ = [
    "FailureVector",
    "ConsensusDogInstance",
    "compute_error_correlation",
    "consensus_admission",
    "consensus_dog_bbox",
    "consensus_dog_instances",
    "consensus_keypoint",
    "robust_weighted_keypoint",
    "weighted_box_fusion",
]
