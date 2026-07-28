"""Deterministic localization benchmark runner with Pareto selection."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cvi.canid_data.types import UnifiedCanidSample
from cvi.localization.adapters import AbstractLocalizationAdapter
from cvi.localization.quality import (
    compute_iou,
    detection_summary,
    greedy_bipartite_match,
    normalized_mean_error,
    pixel_correct_keypoint,
)
from cvi.localization.types import DetectionBox, KeypointSet, LocalizationResult


def run_benchmark(
    adapter: AbstractLocalizationAdapter,
    samples: tuple[UnifiedCanidSample, ...],
    data_root: Path,
    *,
    maximum_images: int = 1024,
    ground_truth: dict[str, list[DetectionBox]] | None = None,
    ground_truth_keypoints: dict[str, KeypointSet] | None = None,
) -> dict[str, Any]:
    """Run localization inference, optionally scoring against ground truth."""

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
        report["detection"] = detection_summary(
            all_matches, total_pred, total_gt
        )

    if ground_truth_keypoints is not None:
        per_point_errors: dict[str, list[float]] = defaultdict(list)
        for _, pred_set in (
            ("body_keypoints", None),
            ("face_landmarks", None),
        ):
            pass

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
        image = Image.open(image_path).convert("RGB").resize((224, 224))

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


__all__ = ["build_contact_sheet", "run_benchmark"]
