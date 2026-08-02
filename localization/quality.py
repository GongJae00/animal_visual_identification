"""Detection metrics (supervised) and quality scoring (unsupervised, ROI-level).

Quality features are computed from the image and predictions without
ground-truth annotations. Composite ``overall`` scores are normalized to
``[0, 1]`` with higher values preferred; raw diagnostic feature direction is
defined by each dataclass field.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from localization.roi import is_truncated
from localization.types import DetectionBox, Keypoint, KeypointSet

# ── Supervised detection metrics ──────────────────────────────────────


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


# ── ROI quality scoring (unsupervised) ────────────────────────────────


@dataclass(frozen=True, slots=True)
class DogQuality:
    detector_confidence: float
    model_agreement: float
    truncation: float
    native_resolution: float
    multi_dog_contamination: float
    blur_estimate: float
    overall: float

    def to_list(self) -> list[float]:
        return [
            self.detector_confidence,
            self.model_agreement,
            self.truncation,
            self.native_resolution,
            self.multi_dog_contamination,
            self.blur_estimate,
        ]


@dataclass(frozen=True, slots=True)
class FaceQuality:
    landmark_confidence: float
    anchor_visibility: float
    yaw_roll_proxy: float
    resolution: float
    truncation: float
    blur_estimate: float
    overall: float

    def to_list(self) -> list[float]:
        return [
            self.landmark_confidence,
            self.anchor_visibility,
            self.yaw_roll_proxy,
            self.resolution,
            self.truncation,
            self.blur_estimate,
        ]


@dataclass(frozen=True, slots=True)
class NoseQuality:
    anchor_agreement: float
    native_resolution: float
    blur_estimate: float
    specular_ratio: float
    truncation: float
    muzzle_contamination: float
    support_coverage: float
    overall: float

    def to_list(self) -> list[float]:
        return [
            self.anchor_agreement,
            self.native_resolution,
            self.blur_estimate,
            self.specular_ratio,
            self.truncation,
            self.muzzle_contamination,
            self.support_coverage,
        ]


def estimate_blur(image: Image.Image) -> float:
    import cv2

    gray = np.asarray(image.convert("L"), dtype=np.float64)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return min(1.0, variance / 500.0)


def score_dog_quality(
    bbox: DetectionBox,
    *,
    model_agreement: float = 1.0,
    multi_dog_boxes: int = 1,
    image_width: int = 0,
    image_height: int = 0,
    blur: float = 0.5,
) -> DogQuality:
    trunc = 0.0
    if image_width and image_height:
        trunc = 1.0 - float(
            is_truncated(bbox, image_width=image_width, image_height=image_height)
        )
    native = min(1.0, max(bbox.width / 224.0, bbox.height / 224.0))
    contamination = 1.0 / max(multi_dog_boxes, 1)
    overall = float(
        np.mean([bbox.confidence, model_agreement, trunc, native, contamination, blur])
    )
    return DogQuality(
        detector_confidence=bbox.confidence,
        model_agreement=model_agreement,
        truncation=trunc,
        native_resolution=native,
        multi_dog_contamination=contamination,
        blur_estimate=blur,
        overall=overall,
    )


def score_face_quality(
    landmarks: KeypointSet | None,
    bbox: DetectionBox | None = None,
    *,
    image_width: int = 0,
    image_height: int = 0,
    blur: float = 0.5,
) -> FaceQuality:
    landmark_conf = 0.0
    anchor_vis = 0.0
    if landmarks is not None:
        confidences = [kp.confidence for kp in landmarks.keypoints.values()]
        landmark_conf = float(np.mean(confidences)) if confidences else 0.0
        anchors = ("nose_center", "left_eye", "right_eye")
        anchor_confidences = [
            point.confidence
            for anchor in anchors
            if (point := landmarks.named(anchor)) is not None
        ]
        anchor_vis = float(np.mean(anchor_confidences)) if anchor_confidences else 0.0
    yaw = 1.0
    if landmarks is not None:
        left = landmarks.named("left_eye")
        right = landmarks.named("right_eye")
        nose = landmarks.named("nose_center")
        if left and right and nose:
            eye_mid = ((left.x + right.x) / 2.0, (left.y + right.y) / 2.0)
            offset = abs(nose.x - eye_mid[0]) / max(abs(right.x - left.x), 1.0)
            yaw = max(0.0, 1.0 - offset / 0.5)
    trunc = 1.0
    if bbox is not None and image_width and image_height:
        trunc = 1.0 - float(
            is_truncated(bbox, image_width=image_width, image_height=image_height)
        )
    resolution = 1.0
    if bbox is not None:
        resolution = min(1.0, max(bbox.width, bbox.height) / 112.0)
    overall = float(np.mean([landmark_conf, anchor_vis, yaw, resolution, trunc, blur]))
    return FaceQuality(
        landmark_confidence=landmark_conf,
        anchor_visibility=anchor_vis,
        yaw_roll_proxy=yaw,
        resolution=resolution,
        truncation=trunc,
        blur_estimate=blur,
        overall=overall,
    )


def score_nose_quality(
    nose_bbox: DetectionBox,
    *,
    anchor_agreement: float = 1.0,
    native_short_side: float = 0.0,
    blur: float = 0.5,
    specular_ratio: float = 0.0,
    image_width: int = 0,
    image_height: int = 0,
    muzzle_contamination: float = 0.0,
    support_coverage: float = 1.0,
) -> NoseQuality:
    trunc = 1.0
    if image_width and image_height:
        trunc = 1.0 - float(
            is_truncated(nose_bbox, image_width=image_width, image_height=image_height)
        )
    native = min(1.0, native_short_side / 224.0) if native_short_side > 0 else 0.5
    overall = float(
        np.mean(
            [
                anchor_agreement,
                native,
                blur,
                1.0 - specular_ratio,
                trunc,
                1.0 - muzzle_contamination,
                support_coverage,
            ]
        )
    )
    return NoseQuality(
        anchor_agreement=anchor_agreement,
        native_resolution=native,
        blur_estimate=blur,
        specular_ratio=specular_ratio,
        truncation=trunc,
        muzzle_contamination=muzzle_contamination,
        support_coverage=support_coverage,
        overall=overall,
    )


__all__ = [
    "DogQuality",
    "FaceQuality",
    "NoseQuality",
    "compute_iou",
    "detection_summary",
    "estimate_blur",
    "greedy_bipartite_match",
    "normalized_mean_error",
    "pixel_correct_keypoint",
    "score_dog_quality",
    "score_face_quality",
    "score_nose_quality",
]
