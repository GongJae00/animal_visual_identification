"""Supervised localization metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from parsing.roi import compute_iou
from parsing.types import DetectionBox, Keypoint, KeypointSet


def pixel_correct_keypoint(
    predicted: Keypoint,
    ground_truth: Keypoint,
    *,
    head_size: float,
    threshold: float = 0.10,
) -> bool:
    distance = float(
        np.linalg.norm((predicted.x - ground_truth.x, predicted.y - ground_truth.y))
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
        distance = float(np.linalg.norm((pred.x - gt.x, pred.y - gt.y))) / max(
            normalization, 1e-6
        )
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


def detection_average_precision(
    predictions: dict[str, list[DetectionBox]],
    ground_truth: dict[str, list[DetectionBox]],
    *,
    iou_threshold: float,
) -> dict[str, float | int]:
    """Compute 101-point interpolated AP for one class and IoU threshold."""

    ranked = sorted(
        (
            (box.confidence, image_id, box)
            for image_id, boxes in predictions.items()
            for box in boxes
        ),
        key=lambda item: -item[0],
    )
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    matched: dict[str, set[int]] = defaultdict(set)
    true_positive: list[float] = []
    false_positive: list[float] = []
    for _, image_id, predicted in ranked:
        candidates = ground_truth.get(image_id, [])
        best_index = -1
        best_iou = 0.0
        for gt_index, target in enumerate(candidates):
            if gt_index in matched[image_id]:
                continue
            iou = compute_iou(predicted, target)
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index
        is_match = best_index >= 0 and best_iou >= iou_threshold
        if is_match:
            matched[image_id].add(best_index)
        true_positive.append(float(is_match))
        false_positive.append(float(not is_match))
    if total_gt == 0:
        raise ValueError("detection AP requires ground-truth boxes")
    if not ranked:
        return {"AP": 0.0, "recall": 0.0, "precision": 0.0, "ground_truth": total_gt}
    tp = np.cumsum(np.asarray(true_positive))
    fp = np.cumsum(np.asarray(false_positive))
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1.0)
    interpolated = [
        float(np.max(precision[recall >= threshold]))
        if np.any(recall >= threshold)
        else 0.0
        for threshold in np.linspace(0.0, 1.0, 101)
    ]
    return {
        "AP": float(np.mean(interpolated)),
        "recall": float(recall[-1]),
        "precision": float(precision[-1]),
        "ground_truth": total_gt,
    }


__all__ = [
    "detection_average_precision",
    "detection_summary",
    "greedy_bipartite_match",
    "normalized_mean_error",
    "pixel_correct_keypoint",
]
