"""Evaluate an AP-10K dog ROI manifest against publisher bounding boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.adapters import adapt_ap10k_dog
from foundation.protected_io import write_private_json_bundle
from localization.quality import greedy_bipartite_match
from localization.roi_manifest import read_roi_manifest
from localization.types import DetectionBox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _score(
    predictions: dict[str, list[DetectionBox]],
    ground_truth: dict[str, list[DetectionBox]],
    threshold: float,
) -> dict[str, float | int]:
    matches = 0
    prediction_count = sum(len(boxes) for boxes in predictions.values())
    ground_truth_count = sum(len(boxes) for boxes in ground_truth.values())
    for sample_id, target_boxes in ground_truth.items():
        matches += len(
            greedy_bipartite_match(
                predictions.get(sample_id, []), target_boxes, iou_threshold=threshold
            )
        )
    return {
        "matches": matches,
        "predictions": prediction_count,
        "ground_truth": ground_truth_count,
        "precision": matches / max(prediction_count, 1),
        "recall": matches / max(ground_truth_count, 1),
    }


def main() -> None:
    args = parse_args()
    manifest = read_roi_manifest(args.manifest)
    samples = adapt_ap10k_dog(args.data_root)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    groups: dict[str, list] = {}
    for sample in samples:
        if sample.split_role == "test":
            groups.setdefault(sample.source_group_id, []).append(sample)
    ground_truth: dict[str, list[DetectionBox]] = {}
    for sample_id in manifest["source_sample_ids"]:
        sample = sample_by_id[sample_id]
        ground_truth[sample_id] = [
            DetectionBox(*row.dog_boxes_xyxy, 1.0)
            for row in groups[sample.source_group_id]
            if row.dog_boxes_xyxy is not None
        ]
    predictions_all: dict[str, list[DetectionBox]] = {}
    predictions_accepted: dict[str, list[DetectionBox]] = {}
    for record in manifest["records"]:
        box = DetectionBox(
            *record["dog_bbox_xyxy"], record["quality"]["detector_confidence"]
        )
        predictions_all.setdefault(record["sample_id"], []).append(box)
        if record["review_state"] == "ACCEPT":
            predictions_accepted.setdefault(record["sample_id"], []).append(box)
    total_images = len(ground_truth)
    report = {
        "schema_version": "cvi.canid_roi_manifest_evaluation.v2",
        "images": total_images,
        "instances": len(manifest["records"]),
        "automatic_accept_review_state_instances": sum(
            record["review_state"] == "ACCEPT" for record in manifest["records"]
        ),
        "all": {
            "IoU50": _score(predictions_all, ground_truth, 0.50),
            "IoU75": _score(predictions_all, ground_truth, 0.75),
        },
        "automatic_accept_review_state": {
            "image_coverage": len(predictions_accepted) / max(total_images, 1),
            "IoU50": _score(predictions_accepted, ground_truth, 0.50),
            "IoU75": _score(predictions_accepted, ground_truth, 0.75),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_private_json_bundle(((args.output, report),))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
