"""Detection quality metrics: IoU, precision, recall, PCK, NME."""

from __future__ import annotations

from typing import Any

import numpy as np

from cvi.localization.types import DetectionBox, Keypoint, KeypointSet


def compute_iou(predicted: DetectionBox, ground_truth: DetectionBox) -> float:
    x1 = max(predicted.x1, ground_truth.x1)
    y1 = max(predicted.y1, ground_truth.y1)
    x2 = min(predicted.x2, ground_truth.x2)
    y2 = min(predicted.y2, ground_truth.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = predicted.area + ground_truth.area - intersection
    return intersection / max(union, 1e-6)


def pixel_correct_keypoint(
    predicted: Keypoint,
    ground_truth: Keypoint,
    *,
    head_size: float,
    threshold: float = 0.10,
) -> bool:
    distance = float(
        np.linalg.norm(
            (predicted.x - ground_truth.x, predicted.y - ground_truth.y)
        )
    )
    return distance <= threshold * head_size


def normalized_mean_error(
    predicted: KeypointSet,
    ground_truth: KeypointSet,
    *,
    normalization: float = 1.0,
    visible_only: bool = True,
) -> dict[str, Any]:
    errors: list[float] = []
    per_point: dict[str, float] = {}
    for name in ground_truth.keypoints:
        gt = ground_truth.named(name)
        pred = predicted.named(name)
        if gt is None or pred is None:
            continue
        if visible_only and gt.confidence < 0.5:
            continue
        distance = float(
            np.linalg.norm((pred.x - gt.x, pred.y - gt.y))
        ) / max(normalization, 1e-6)
        errors.append(distance)
        per_point[name] = distance
    return {
        "nme": float(np.mean(errors)) if errors else float("nan"),
        "count": len(errors),
        "per_point": per_point,
    }


def detection_summary(
    matches: list[tuple[DetectionBox, DetectionBox, float]],
    total_predicted: int,
    total_ground_truth: int,
    *,
    iou_thresholds: tuple[float, ...] = (0.50, 0.75),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total_predicted": total_predicted,
        "total_ground_truth": total_ground_truth,
    }
    matched = len(matches)
    result["matched"] = matched
    result["false_positives"] = total_predicted - matched
    result["false_negatives"] = total_ground_truth - matched
    for threshold in iou_thresholds:
        above = sum(1 for _, _, iou in matches if iou >= threshold)
        precision = above / max(total_predicted, 1)
        recall = above / max(total_ground_truth, 1)
        result[f"AP{int(threshold * 100)}_precision"] = precision
        result[f"AP{int(threshold * 100)}_recall"] = recall
        result[f"AP{int(threshold * 100)}_f1"] = (
            2.0 * precision * recall / max(precision + recall, 1e-6)
        )
    return result


def greedy_bipartite_match(
    predicted: list[DetectionBox],
    ground_truth: list[DetectionBox],
    *,
    iou_threshold: float = 0.50,
) -> list[tuple[DetectionBox, DetectionBox, float]]:
    """Greedy bipartite matching by IoU, one-to-one."""
    available_pred = list(range(len(predicted)))
    available_gt = list(range(len(ground_truth)))
    pairs: list[tuple[int, int, float]] = []
    for pred_idx, pred in enumerate(predicted):
        for gt_idx, gt in enumerate(ground_truth):
            iou = compute_iou(pred, gt)
            if iou >= iou_threshold:
                pairs.append((pred_idx, gt_idx, iou))
    pairs.sort(key=lambda entry: -entry[2])
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    result: list[tuple[DetectionBox, DetectionBox, float]] = []
    for pred_idx, gt_idx, iou in pairs:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            result.append((predicted[pred_idx], ground_truth[gt_idx], iou))
    return result


__all__ = [
    "compute_iou",
    "detection_summary",
    "greedy_bipartite_match",
    "normalized_mean_error",
    "pixel_correct_keypoint",
]
