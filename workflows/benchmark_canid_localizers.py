"""Run a pinned dog detector and emit a content-bound prediction cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from data.adapters import ADAPTERS
from data.source_lock import get_record
from parsing.adapters import (
    TorchvisionFasterRCNNDogAdapter,
    UltralyticsDogAdapter,
    UltralyticsDogPoseAdapter,
)
from evaluation.localization import (
    ap10k_body17_pose_summary,
    build_contact_sheet,
)
from parsing.prediction_cache import (
    build_prediction_cache,
    write_prediction_cache,
)
from evaluation.localization_metrics import detection_average_precision
from parsing.types import (
    AP10K_BODY_17_SCHEMA,
    DetectionBox,
    Keypoint,
    KeypointSet,
    LocalizationResult,
)

_SUMMARY_SCHEMA = "cvi.canid_localizer_benchmark_summary.v1"


def _build_summary(
    *,
    dataset: str,
    split_role: str | None,
    results: list[LocalizationResult],
    prediction_cache: Path,
    prediction_cache_sha256: str,
    ground_truth: dict[str, list[DetectionBox]] | None,
    ground_truth_keypoints: dict[str, list[KeypointSet | None]] | None,
) -> dict[str, object]:
    timings = np.asarray([result.inference_ms for result in results], dtype=np.float64)
    report: dict[str, object] = {
        "schema_version": _SUMMARY_SCHEMA,
        "dataset": dataset,
        "split_role": split_role,
        "images": len(results),
        "dog_detections": sum(len(result.dog_boxes) for result in results),
        "detection_rate": sum(bool(result.dog_boxes) for result in results)
        / len(results),
        "latency_ms_mean": float(timings.mean()),
        "latency_ms_median": float(np.median(timings)),
        "prediction_cache": str(prediction_cache),
        "prediction_cache_sha256": prediction_cache_sha256,
    }
    if ground_truth is not None:
        predicted = {result.image_id: list(result.dog_boxes) for result in results}
        report["detection"] = {
            "AP50": detection_average_precision(
                predicted, ground_truth, iou_threshold=0.50
            ),
            "AP75": detection_average_precision(
                predicted, ground_truth, iou_threshold=0.75
            ),
        }
    if dataset == "ap10k-dog":
        if ground_truth is None or ground_truth_keypoints is None:
            raise ValueError("AP-10K summary requires bbox and body-17 ground truth")
        report["pose"] = ap10k_body17_pose_summary(
            results, ground_truth, ground_truth_keypoints
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="yt-bb-dog", choices=sorted(ADAPTERS))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument(
        "--backend",
        choices=("ultralytics", "torchvision-fasterrcnn", "ultralytics-pose"),
        default="ultralytics",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--split-role")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-cache", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    record = get_record(args.dataset)
    root = Path(record.data_root)
    all_samples = ADAPTERS[args.dataset](root)
    ground_truth: dict[str, list[DetectionBox]] | None = None
    ground_truth_keypoints: dict[str, list[KeypointSet | None]] | None = None
    effective_split_role = args.split_role
    if args.dataset == "ap10k-dog":
        effective_split_role = args.split_role or "test"
        grouped: dict[str, list] = {}
        for sample in all_samples:
            if sample.split_role == effective_split_role:
                grouped.setdefault(sample.source_group_id, []).append(sample)
        samples = tuple(rows[0] for _, rows in sorted(grouped.items()))[
            : args.max_images
        ]
        ground_truth = {
            rows[0].sample_id: [
                DetectionBox(*sample.dog_boxes_xyxy, 1.0)
                for sample in rows
                if sample.dog_boxes_xyxy is not None
            ]
            for _, rows in sorted(grouped.items())[: args.max_images]
        }
        ground_truth_keypoints = {
            rows[0].sample_id: [
                (
                    KeypointSet(
                        {
                            name: Keypoint(x, y, visibility)
                            for name, (x, y, visibility) in (
                                sample.body_keypoints.items()
                            )
                        },
                        AP10K_BODY_17_SCHEMA,
                    )
                    if sample.body_keypoints
                    else None
                )
                for sample in rows
                if sample.dog_boxes_xyxy is not None
            ]
            for _, rows in sorted(grouped.items())[: args.max_images]
        }
    else:
        selected = (
            tuple(
                sample for sample in all_samples if sample.split_role == args.split_role
            )
            if args.split_role
            else all_samples
        )
        samples = selected[: args.max_images]
    if not samples:
        raise RuntimeError("selected dataset has no samples")

    adapter_class = {
        "ultralytics": UltralyticsDogAdapter,
        "torchvision-fasterrcnn": TorchvisionFasterRCNNDogAdapter,
        "ultralytics-pose": UltralyticsDogPoseAdapter,
    }[args.backend]
    adapter = adapter_class(args.model_path, args.model_sha256, device=args.device)
    results = []
    try:
        for sample in samples:
            image = Image.open(root / sample.image_path).convert("RGB")
            started = time.perf_counter()
            result = adapter.detect(image, image_id=sample.sample_id)
            object.__setattr__(
                result, "inference_ms", (time.perf_counter() - started) * 1000.0
            )
            results.append(result)
    finally:
        adapter.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_cache.parent.mkdir(parents=True, exist_ok=True)
    build_contact_sheet(tuple(results), tuple(samples), root, args.output_dir)
    model = adapter.to_dict()
    bundle = build_prediction_cache(samples, results, model=model)
    write_prediction_cache(args.prediction_cache, bundle)
    report = _build_summary(
        dataset=args.dataset,
        split_role=effective_split_role,
        results=results,
        prediction_cache=args.prediction_cache,
        prediction_cache_sha256=bundle["cache_sha256"],
        ground_truth=ground_truth,
        ground_truth_keypoints=ground_truth_keypoints,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
