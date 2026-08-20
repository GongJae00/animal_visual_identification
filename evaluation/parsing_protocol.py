"""Parsing evaluation catalog: stage × data × metric × extraction.

This module is a protocol ledger, not a results table. It does not record
measured values. Backbone choice is out of scope; the five export substages
stay if the detector or segmenter changes.

CLI: ``uv run python -m evaluation.commands.evaluate parsing-protocol --help``
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.foundation.protected_io import write_private_json_bundle

PROTOCOL_SCHEMA = "evaluation.parsing_protocol.v1"
INTERPRETATION = (
    "PARSING_EVALUATION_CATALOG_NOT_MEASURED_VALUES_NOT_BIOMETRIC_VALIDATION"
)
_KINDS = frozenset({"supervised", "unsupervised", "census", "control", "review"})
_STAGE_IDS = ("detection", "segmentation", "regions", "quality", "crops")


@dataclass(frozen=True, slots=True)
class ParsingStage:
    stage_id: str
    vis_substage: str
    label: str
    output: str

    def to_dict(self) -> dict[str, str]:
        return {
            "stage_id": self.stage_id,
            "vis_substage": self.vis_substage,
            "label": self.label,
            "output": self.output,
        }


@dataclass(frozen=True, slots=True)
class ParsingMetric:
    metric_id: str
    stage_id: str
    label: str
    data: str
    extraction: str
    owner: str
    report_schema: str
    kind: str
    command: str

    def __post_init__(self) -> None:
        if self.stage_id not in _STAGE_IDS:
            raise ValueError(f"unknown parsing stage: {self.stage_id}")
        if self.kind not in _KINDS:
            raise ValueError(f"unknown parsing metric kind: {self.kind}")
        if not self.metric_id or not self.label or not self.data or not self.extraction:
            raise ValueError("parsing metric fields must be non-empty")
        if not self.owner or not self.command:
            raise ValueError("parsing metric owner and command must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "metric_id": self.metric_id,
            "stage_id": self.stage_id,
            "label": self.label,
            "data": self.data,
            "extraction": self.extraction,
            "owner": self.owner,
            "report_schema": self.report_schema,
            "kind": self.kind,
            "command": self.command,
        }

    def figure_row(self, stage_label: str) -> dict[str, str]:
        return {
            "stage": stage_label,
            "data": self.data,
            "metric": self.label,
            "extraction": self.extraction,
        }


STAGES: tuple[ParsingStage, ...] = (
    ParsingStage("detection", "00_detection", "Detection", "boxes"),
    ParsingStage("segmentation", "01_segmentation", "Segmentation", "masks"),
    ParsingStage("regions", "02_regions", "Regions", "face / nose / body"),
    ParsingStage("quality", "03_quality", "Quality", "admission scores"),
    ParsingStage("crops", "04_crops", "Crops", "crops"),
)

_STAGE_BY_ID = {stage.stage_id: stage for stage in STAGES}

METRICS: tuple[ParsingMetric, ...] = (
    ParsingMetric(
        metric_id="detection.ap50",
        stage_id="detection",
        label="AP@0.50",
        data="AP-10K dog boxes",
        extraction="101-point interpolated AP; greedy max-IoU match per image",
        owner="evaluation.localization_metrics.detection_average_precision",
        report_schema="parsing.canid_localizer_benchmark_summary.v1",
        kind="supervised",
        command="evaluation.commands.evaluate localization-benchmark",
    ),
    ParsingMetric(
        metric_id="detection.ap75",
        stage_id="detection",
        label="AP@0.75",
        data="AP-10K dog boxes",
        extraction="Same AP as @0.50 with IoU threshold 0.75",
        owner="evaluation.localization_metrics.detection_average_precision",
        report_schema="parsing.canid_localizer_benchmark_summary.v1",
        kind="supervised",
        command="evaluation.commands.evaluate localization-benchmark",
    ),
    ParsingMetric(
        metric_id="detection.prf50",
        stage_id="detection",
        label="P / R / F1 @ IoU 0.50",
        data="AP-10K dog boxes",
        extraction="Greedy one-to-one bipartite match, then TP/pred and TP/gt",
        owner="evaluation.localization_metrics.detection_summary",
        report_schema="parsing.canid_localizer_benchmark_summary.v1",
        kind="supervised",
        command="evaluation.commands.evaluate localization-benchmark",
    ),
    ParsingMetric(
        metric_id="detection.rate",
        stage_id="detection",
        label="Detection rate",
        data="AP-10K dog images",
        extraction="Images with at least one dog box / images",
        owner="evaluation.localization_benchmark",
        report_schema="parsing.canid_localizer_benchmark_summary.v1",
        kind="census",
        command="evaluation.commands.evaluate localization-benchmark",
    ),
    ParsingMetric(
        metric_id="detection.counts",
        stage_id="detection",
        label="Image and box counts",
        data="AP-10K dog images",
        extraction="Images and dog_detections from the prediction cache",
        owner="evaluation.localization_benchmark",
        report_schema="parsing.canid_localizer_benchmark_summary.v1",
        kind="census",
        command="evaluation.commands.evaluate localization-benchmark",
    ),
    ParsingMetric(
        metric_id="segmentation.iou",
        stage_id="segmentation",
        label="Classified-pixel IoU",
        data="Oxford-IIIT Pet trimaps",
        extraction="TP/(TP+FP+FN) on trimap {1,2}; class 3 excluded",
        owner="evaluation.oxford_pet_foreground",
        report_schema="evaluation.oxford_pet_foreground_evaluation.v3",
        kind="supervised",
        command="evaluation.commands.evaluate oxford-pet",
    ),
    ParsingMetric(
        metric_id="segmentation.dice",
        stage_id="segmentation",
        label="Classified-pixel Dice",
        data="Oxford-IIIT Pet trimaps",
        extraction="2TP/(2TP+FP+FN) on the same classified pixels",
        owner="evaluation.oxford_pet_foreground",
        report_schema="evaluation.oxford_pet_foreground_evaluation.v3",
        kind="supervised",
        command="evaluation.commands.evaluate oxford-pet",
    ),
    ParsingMetric(
        metric_id="segmentation.foreground_recall",
        stage_id="segmentation",
        label="Foreground recall",
        data="Oxford-IIIT Pet trimaps",
        extraction="TP/(TP+FN) on classified foreground pixels",
        owner="evaluation.oxford_pet_foreground",
        report_schema="evaluation.oxford_pet_foreground_evaluation.v3",
        kind="supervised",
        command="evaluation.commands.evaluate oxford-pet",
    ),
    ParsingMetric(
        metric_id="segmentation.background_leakage",
        stage_id="segmentation",
        label="Background leakage",
        data="Oxford-IIIT Pet trimaps",
        extraction="FP/(FP+TN) on classified background pixels",
        owner="evaluation.oxford_pet_foreground",
        report_schema="evaluation.oxford_pet_foreground_evaluation.v3",
        kind="supervised",
        command="evaluation.commands.evaluate oxford-pet",
    ),
    ParsingMetric(
        metric_id="segmentation.averages",
        stage_id="segmentation",
        label="Macro / micro averages",
        data="Oxford-IIIT Pet trimaps",
        extraction="Unweighted per-image mean; pixel counts pooled across images",
        owner="evaluation.oxford_pet_foreground",
        report_schema="evaluation.oxford_pet_foreground_evaluation.v3",
        kind="supervised",
        command="evaluation.commands.evaluate oxford-pet",
    ),
    ParsingMetric(
        metric_id="segmentation.panel",
        stage_id="segmentation",
        label="Unassisted panel",
        data="AP-10K images",
        extraction="Visible-instance candidates; not human-mask verified",
        owner="parsing.export.panel",
        report_schema="parsing.panel.v1",
        kind="review",
        command="parsing.commands.parse panel",
    ),
    ParsingMetric(
        metric_id="regions.nme",
        stage_id="regions",
        label="NME",
        data="AP-10K body-17 keypoints",
        extraction="Detector-conditioned; distance / GT-box diagonal; not official OKS/mAP",
        owner="evaluation.localization.ap10k_body17_pose_summary",
        report_schema="evaluation.ap10k_body17_pose_evaluation.v1",
        kind="supervised",
        command="evaluation.commands.evaluate localization-benchmark",
    ),
    ParsingMetric(
        metric_id="regions.pck",
        stage_id="regions",
        label="PCK@0.05 / PCK@0.10",
        data="AP-10K body-17 keypoints",
        extraction="Matched instances; missing or low-confidence points count as incorrect",
        owner="evaluation.localization.ap10k_body17_pose_summary",
        report_schema="evaluation.ap10k_body17_pose_evaluation.v1",
        kind="supervised",
        command="evaluation.commands.evaluate localization-benchmark",
    ),
    ParsingMetric(
        metric_id="regions.e2e_pck",
        stage_id="regions",
        label="End-to-end PCK",
        data="AP-10K body-17 keypoints",
        extraction="All visible GT keypoints; unmatched instances count as incorrect",
        owner="evaluation.localization.ap10k_body17_pose_summary",
        report_schema="evaluation.ap10k_body17_pose_evaluation.v1",
        kind="supervised",
        command="evaluation.commands.evaluate localization-benchmark",
    ),
    ParsingMetric(
        metric_id="regions.three_region",
        stage_id="regions",
        label="A/F/N completeness",
        data="ROI bundle",
        extraction="Fail-closed COMPLETE vs incomplete records after three-region export",
        owner="parsing.export.regions.three_region_export",
        report_schema="parsing.three_region_artifact_bundle.v1",
        kind="census",
        command="parsing.commands.parse three-region",
    ),
    ParsingMetric(
        metric_id="quality.dog",
        stage_id="quality",
        label="DogQuality",
        data="Source image + dog box",
        extraction="Mean of confidence, agreement, truncation, resolution, contamination, blur",
        owner="parsing.export.quality.score_dog_quality",
        report_schema="",
        kind="unsupervised",
        command="parsing.export.quality",
    ),
    ParsingMetric(
        metric_id="quality.face",
        stage_id="quality",
        label="FaceQuality",
        data="Face box + landmarks",
        extraction="Mean of landmark conf, anchor vis, yaw proxy, resolution, truncation, blur",
        owner="parsing.export.quality.score_face_quality",
        report_schema="",
        kind="unsupervised",
        command="parsing.export.quality",
    ),
    ParsingMetric(
        metric_id="quality.nose",
        stage_id="quality",
        label="NoseQuality",
        data="Nose box + support mask",
        extraction="Mean of agreement, resolution, blur, 1−specular, truncation, 1−muzzle, coverage",
        owner="parsing.export.quality.score_nose_quality",
        report_schema="",
        kind="unsupervised",
        command="parsing.export.quality",
    ),
    ParsingMetric(
        metric_id="crops.route_census",
        stage_id="crops",
        label="Route census",
        data="Materialization receipts × route-plan",
        extraction="Counts by actual_route and terminal_reason",
        owner="parsing.export.compare.summarize",
        report_schema="parsing.parser_materialization_summary.v1",
        kind="census",
        command="parsing.commands.parse compare",
    ),
    ParsingMetric(
        metric_id="crops.comparison",
        stage_id="crops",
        label="Materialization comparison",
        data="Two route-plans + materialization roots",
        extraction="Per-token route change, crop recoveries, byte identity",
        owner="parsing.export.compare",
        report_schema="parsing.parser_materialization_comparison.v1",
        kind="census",
        command="parsing.commands.parse compare",
    ),
    ParsingMetric(
        metric_id="crops.oracle",
        stage_id="crops",
        label="Oracle crops",
        data="Protected pair bundle",
        extraction="Token-only crops from authenticated pairs; control, not parser accuracy",
        owner="evaluation.controls.oracle_crop_export",
        report_schema="",
        kind="control",
        command="evaluation.commands.evaluate oracle-crops",
    ),
)


def _metric_ids() -> tuple[str, ...]:
    return tuple(metric.metric_id for metric in METRICS)


def protocol_document() -> dict[str, Any]:
    ids = _metric_ids()
    if len(ids) != len(set(ids)):
        raise RuntimeError("parsing metric_id values must be unique")
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "interpretation": INTERPRETATION,
        "backbone_independent": True,
        "columns": ["stage", "data", "metric", "extraction"],
        "stages": [stage.to_dict() for stage in STAGES],
        "metrics": [metric.to_dict() for metric in METRICS],
        "figure_rows": [
            metric.figure_row(_STAGE_BY_ID[metric.stage_id].label) for metric in METRICS
        ],
    }


def metrics_for_stage(stage_id: str) -> tuple[ParsingMetric, ...]:
    return tuple(metric for metric in METRICS if metric.stage_id == stage_id)


def visualization_trace() -> dict[str, Any]:
    document = protocol_document()
    substages: dict[str, Any] = {}
    for stage in STAGES:
        rows = [
            metric.figure_row(stage.label) for metric in metrics_for_stage(stage.stage_id)
        ]
        substages[stage.vis_substage] = {
            "summary": stage.output,
            "metrics": rows,
        }
    return {
        "stage": "parsing",
        "protocol": document,
        "substages": substages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.commands.evaluate parsing-protocol",
        description=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Visualization-trace JSON (protocol + per-substage metric tables).",
    )
    args = parser.parse_args(argv)
    trace = visualization_trace()
    write_private_json_bundle(((args.output, trace),))
    print(
        json.dumps(
            {
                "event": "parsing_protocol_written",
                "schema_version": PROTOCOL_SCHEMA,
                "output": str(args.output),
                "stages": len(STAGES),
                "metrics": len(METRICS),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
