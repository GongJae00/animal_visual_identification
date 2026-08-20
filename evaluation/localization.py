"""Deterministic localization benchmark runner and diagnostic renderer."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from data.types import UnifiedCanidSample
from parsing.export.detection.adapters import AbstractLocalizationAdapter
from evaluation.localization_metrics import (
    detection_summary,
    greedy_bipartite_match,
)
from parsing.export.crops.roi import compute_iou
from parsing.export.types import (
    AP10K_BODY_17_KEYPOINT_NAMES,
    AP10K_BODY_17_SCHEMA,
    DetectionBox,
    KeypointSet,
    LocalizationResult,
)

_POSE_EVALUATION_SCHEMA = "cvi.ap10k_body17_pose_evaluation.v1"
_POSE_MATCH_IOU = 0.50
_POSE_KEYPOINT_CONFIDENCE = 0.50


def ap10k_body17_pose_summary(
    results: Sequence[LocalizationResult],
    ground_truth_boxes: Mapping[str, Sequence[DetectionBox]],
    ground_truth_keypoints: Mapping[str, Sequence[KeypointSet | None]],
) -> dict[str, Any]:
    """Compute custom, detector-conditioned AP-10K body-17 pose metrics."""

    image_ids = [result.image_id for result in results]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("pose evaluation result image IDs must be unique")
    if set(ground_truth_boxes) != set(ground_truth_keypoints):
        raise ValueError("ground-truth bbox and keypoint image IDs must align")

    matched_instances = 0
    predicted_instances = 0
    ground_truth_instances = 0
    visible_ground_truth = 0
    total_visible_ground_truth = 0
    valid_predictions = 0
    missing_predictions = 0
    normalized_errors: list[float] = []
    pck_correct = {0.05: 0, 0.10: 0}

    for result in results:
        predicted_instances += len(result.dog_boxes)
        if result.body_keypoints and len(result.body_keypoints) != len(
            result.dog_boxes
        ):
            raise ValueError("body keypoint sets must align with dog boxes")
        for point_set in result.body_keypoints:
            _validate_ap10k_keypoint_set(point_set, "predicted")

        gt_boxes = list(ground_truth_boxes.get(result.image_id, ()))
        gt_point_sets = list(ground_truth_keypoints.get(result.image_id, ()))
        if len(gt_boxes) != len(gt_point_sets):
            raise ValueError(
                "ground-truth keypoint sets must align with ground-truth dog boxes"
            )
        ground_truth_instances += len(gt_boxes)
        diagonals = [_ground_truth_bbox_diagonal(box) for box in gt_boxes]
        for point_set in gt_point_sets:
            if point_set is not None:
                _validate_ap10k_keypoint_set(point_set, "ground-truth")
                total_visible_ground_truth += sum(
                    point.confidence > 0.0 for point in point_set.keypoints.values()
                )

        for pred_index, gt_index in _confidence_ordered_pose_matches(
            result.dog_boxes, gt_boxes
        ):
            matched_instances += 1
            predicted = (
                result.body_keypoints[pred_index] if result.body_keypoints else None
            )
            ground_truth = gt_point_sets[gt_index]
            if ground_truth is None:
                continue
            normalization = diagonals[gt_index]
            for name in AP10K_BODY_17_KEYPOINT_NAMES:
                target = ground_truth.named(name)
                if target is None or target.confidence <= 0.0:
                    continue
                visible_ground_truth += 1
                point = predicted.named(name) if predicted is not None else None
                if point is None or point.confidence < _POSE_KEYPOINT_CONFIDENCE:
                    missing_predictions += 1
                    continue
                error = float(np.hypot(point.x - target.x, point.y - target.y))
                normalized_error = error / normalization
                if not np.isfinite(normalized_error):
                    raise ValueError("pose normalized error must be finite")
                valid_predictions += 1
                normalized_errors.append(normalized_error)
                for threshold in pck_correct:
                    pck_correct[threshold] += int(normalized_error <= threshold)

    pck_denominator = visible_ground_truth
    nme_denominator = valid_predictions
    return {
        "schema_version": _POSE_EVALUATION_SCHEMA,
        "metric": "custom_ap10k_body17_pose",
        "matching": "prediction_confidence_ordered_one_to_one_iou",
        "matching_iou_threshold": _POSE_MATCH_IOU,
        "normalization": "ground_truth_bbox_diagonal",
        "prediction_keypoint_confidence_threshold": _POSE_KEYPOINT_CONFIDENCE,
        "predicted_instances": predicted_instances,
        "ground_truth_instances": ground_truth_instances,
        "matched_instances": matched_instances,
        "visible_ground_truth_keypoints": visible_ground_truth,
        "total_visible_ground_truth_keypoints": total_visible_ground_truth,
        "valid_predicted_keypoints": valid_predictions,
        "missing_or_low_confidence_keypoints": missing_predictions,
        "nme_denominator": nme_denominator,
        "pck_denominator": pck_denominator,
        "NME": float(np.mean(normalized_errors)) if normalized_errors else None,
        "PCK@0.05": (
            pck_correct[0.05] / pck_denominator if pck_denominator else None
        ),
        "PCK@0.10": (
            pck_correct[0.10] / pck_denominator if pck_denominator else None
        ),
        "end_to_end_PCK@0.05": (
            pck_correct[0.05] / total_visible_ground_truth
            if total_visible_ground_truth
            else None
        ),
        "end_to_end_PCK@0.10": (
            pck_correct[0.10] / total_visible_ground_truth
            if total_visible_ground_truth
            else None
        ),
        "metric_note": (
            "Custom metric, not official AP-10K OKS/mAP. Predictions are processed "
            "by descending dog-box confidence and matched one-to-one to the "
            "highest-IoU unmatched GT dog at IoU >= 0.50. Distances are normalized "
            "by the matched GT bbox diagonal. GT keypoints with visibility > 0 are "
            "eligible. NME averages only eligible points with prediction confidence "
            ">= 0.50; PCK denominators include every eligible point on matched "
            "instances, so missing or lower-confidence predictions are incorrect. "
            "Detector-conditioned PCK excludes unmatched GT instances; end-to-end "
            "PCK includes all visible GT keypoints and counts unmatched instances as "
            "incorrect. NME remains detector- and confidence-conditioned."
        ),
    }


def _confidence_ordered_pose_matches(
    predicted: Sequence[DetectionBox], ground_truth: Sequence[DetectionBox]
) -> list[tuple[int, int]]:
    matched_ground_truth: set[int] = set()
    matches: list[tuple[int, int]] = []
    prediction_order = sorted(
        range(len(predicted)), key=lambda index: (-predicted[index].confidence, index)
    )
    for pred_index in prediction_order:
        candidates = [
            (compute_iou(predicted[pred_index], target), gt_index)
            for gt_index, target in enumerate(ground_truth)
            if gt_index not in matched_ground_truth
        ]
        if not candidates:
            continue
        best_iou, gt_index = max(candidates, key=lambda item: (item[0], -item[1]))
        if best_iou < _POSE_MATCH_IOU:
            continue
        matched_ground_truth.add(gt_index)
        matches.append((pred_index, gt_index))
    return matches


def _ground_truth_bbox_diagonal(box: DetectionBox) -> float:
    if (
        not np.isfinite(box.width)
        or not np.isfinite(box.height)
        or box.width <= 0.0
        or box.height <= 0.0
    ):
        raise ValueError("ground-truth bbox geometry must be finite and positive")
    diagonal = float(np.hypot(box.width, box.height))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("ground-truth bbox diagonal must be finite and positive")
    return diagonal


def _validate_ap10k_keypoint_set(point_set: KeypointSet, label: str) -> None:
    if point_set.schema != AP10K_BODY_17_SCHEMA:
        raise ValueError(f"{label} keypoint schema must be {AP10K_BODY_17_SCHEMA}")
    unexpected = set(point_set.keypoints) - set(AP10K_BODY_17_KEYPOINT_NAMES)
    if unexpected:
        raise ValueError(f"{label} keypoint set contains names outside body-17")


def run_benchmark(
    adapter: AbstractLocalizationAdapter,
    samples: tuple[UnifiedCanidSample, ...],
    data_root: Path,
    *,
    maximum_images: int = 1024,
    ground_truth: dict[str, list[DetectionBox]] | None = None,
    ground_truth_keypoints: dict[str, list[KeypointSet | None]] | None = None,
) -> dict[str, Any]:
    """Run localization inference, optionally scoring against ground truth."""

    if ground_truth_keypoints is not None and ground_truth is None:
        raise ValueError("keypoint scoring requires aligned ground-truth dog boxes")
    if ground_truth_keypoints is not None and any(
        sample.dataset_name != "ap10k-dog" for sample in samples
    ):
        raise ValueError("pose metrics are only defined for ap10k-dog")

    selected = samples[: min(len(samples), maximum_images)]
    results: list[LocalizationResult] = []
    timings: list[float] = []

    for sample in selected:
        image_path = data_root / sample.image_path
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        started = time.perf_counter()
        result = adapter.detect(image, image_id=sample.sample_id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        object.__setattr__(result, "inference_ms", elapsed_ms)
        results.append(result)
        timings.append(elapsed_ms)

    report: dict[str, Any] = {
        "model": adapter.to_dict(),
        "images_processed": len(results),
        "dog_boxes_total": sum(len(r.dog_boxes) for r in results),
        "face_boxes_total": sum(len(r.face_boxes) for r in results),
        "nose_boxes_total": sum(len(r.nose_boxes) for r in results),
        "latency_ms": {
            "mean": float(np.mean(timings)) if timings else 0.0,
            "median": float(np.median(timings)) if timings else 0.0,
            "p99": float(np.percentile(timings, 99)) if timings else 0.0,
        },
    }

    if ground_truth is not None:
        all_matches: list[tuple[DetectionBox, DetectionBox, float]] = []
        total_pred, total_gt = 0, 0
        for result in results:
            gt_boxes = ground_truth.get(result.image_id, [])
            pred_boxes = list(result.dog_boxes)
            total_pred += len(pred_boxes)
            total_gt += len(gt_boxes)
            all_matches.extend(greedy_bipartite_match(pred_boxes, gt_boxes))
        report["detection"] = detection_summary(all_matches, total_pred, total_gt)

    if ground_truth_keypoints is not None and ground_truth is not None:
        report["pose"] = ap10k_body17_pose_summary(
            results, ground_truth, ground_truth_keypoints
        )

    return report


def build_contact_sheet(
    results: tuple[LocalizationResult, ...],
    samples: tuple[UnifiedCanidSample, ...],
    data_root: Path,
    output_dir: Path,
    *,
    grid_size: int = 8,
) -> None:
    """Write contact sheet images with bbox/keypoint overlays to output_dir."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_map = {s.sample_id: s for s in samples}
    canvas: Image.Image | None = None
    row_count = 0

    for index, result in enumerate(results):
        sample = sample_map.get(result.image_id)
        if sample is None:
            continue
        image_path = data_root / sample.image_path
        if not image_path.is_file():
            continue
        original = Image.open(image_path).convert("RGB")
        scale_x = 224.0 / original.width
        scale_y = 224.0 / original.height
        image = original.resize((224, 224))
        draw = ImageDraw.Draw(image)
        for box in result.dog_boxes:
            draw.rectangle(
                (
                    box.x1 * scale_x,
                    box.y1 * scale_y,
                    box.x2 * scale_x,
                    box.y2 * scale_y,
                ),
                outline=(0, 255, 0),
                width=3,
            )
            draw.text(
                (box.x1 * scale_x, max(0.0, box.y1 * scale_y - 12)),
                f"dog {box.confidence:.2f}",
                fill=(0, 255, 0),
            )

        if canvas is None:
            canvas = Image.new("RGB", (224 * grid_size, 224 * grid_size), (0, 0, 0))

        col = index % grid_size
        row = index // grid_size
        canvas.paste(image, (col * 224, row * 224))
        row_count = row + 1
        if row_count >= grid_size:
            break

    if canvas is not None:
        canvas.save(output_dir / "contact_sheet.jpg", quality=90)


__all__ = ["ap10k_body17_pose_summary", "build_contact_sheet", "run_benchmark"]
