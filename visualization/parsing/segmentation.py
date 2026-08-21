"""Segmentation plates for ``Visualization/vis/00_parsing/01_segmentation``."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from visualization.parsing.assets import AssetLoader
from visualization.parsing.plates import format_table, save_grid
from visualization.rendering.style import paper_matplotlib_rc

_DATASET_FILENAME = re.compile(r"^[A-Za-z0-9_-]+$")
_GRID_ROWS = 6
_GRID_COLUMNS = 3
_Sample = tuple[Any, Any, Any | None]
_Metric = tuple[str, int, int, int, int, int, float, float]


def render(
    payload: Mapping[str, Any], target: Path, loader: AssetLoader
) -> tuple[tuple[Path, ...], Mapping[str, Any]]:
    written: list[Path] = []
    datasets = payload.get("samples", {})
    if datasets is None:
        datasets = {}
    if not isinstance(datasets, Mapping):
        raise ValueError("segmentation samples must be an object")
    for dataset_name in sorted(datasets):
        _validate_dataset_name(dataset_name)
        raw_samples = datasets[dataset_name]
        if not isinstance(raw_samples, list):
            raise ValueError("segmentation samples must be arrays")
        if not raw_samples:
            continue
        samples = tuple(
            _load_sample(value, loader) for value in raw_samples[:_GRID_ROWS]
        )
        path = target / f"segmentation_{dataset_name}.pdf"
        _draw_samples(path, samples)
        written.append(path)

    metrics = _read_metrics(payload.get("segmentation_metrics"))
    if metrics:
        parser_backbones = _backbone_label(payload.get("parser_backbones"))
        path = target / "segmentation_metrics.pdf"
        _draw_metrics(path, metrics, parser_backbones)
        written.append(path)
    return tuple(written), {}


def _validate_dataset_name(value: Any) -> None:
    if not isinstance(value, str) or not _DATASET_FILENAME.fullmatch(value):
        raise ValueError("segmentation dataset name is not a safe filename")


def _load_sample(value: Any, loader: AssetLoader) -> _Sample:
    if not isinstance(value, Mapping):
        raise ValueError("segmentation sample must be an object")
    background_value = value.get("background_image")
    return (
        loader.image(value.get("segment_input_image")),
        loader.image(value.get("segment_output_image")),
        loader.image(background_value) if background_value is not None else None,
    )


def _read_metrics(value: Any) -> tuple[_Metric, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("segmentation metrics must be an array")
    result: list[_Metric] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("segmentation metric must be an object")
        dataset_name = item.get("dataset_name")
        _validate_dataset_name(dataset_name)
        counts = tuple(
            item.get(name)
            for name in (
                "detected_inputs",
                "parsed_instances",
                "usable",
                "review",
                "unusable",
            )
        )
        means = tuple(
            item.get(name) for name in ("mean_shape_iou", "mean_ownership_retention")
        )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts
        ):
            raise ValueError("segmentation metric counts must be nonnegative integers")
        if counts[2] + counts[3] + counts[4] != counts[1]:
            raise ValueError("segmentation metric quality counts are inconsistent")
        if any(
            isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not math.isfinite(mean)
            or not 0.0 <= mean <= 1.0
            for mean in means
        ):
            raise ValueError("segmentation metric means must lie in [0, 1]")
        result.append((dataset_name, *counts, *means))
    return tuple(result)


def _backbone_label(value: Any) -> str:
    if value is None:
        return "unspecified"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("parser_backbones must be non-empty model ids")
    return value


def _draw_samples(path: Path, samples: tuple[_Sample, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import numpy as np
    from matplotlib import pyplot as plt

    rc = paper_matplotlib_rc()
    with matplotlib.rc_context(rc):
        figure, axes = plt.subplots(
            _GRID_ROWS,
            _GRID_COLUMNS,
            figsize=(9.0, 15.0),
            squeeze=False,
        )
        try:
            for row_index, row in enumerate(axes):
                for axis in row:
                    axis.set_axis_off()
                if row_index >= len(samples):
                    continue
                segment_input, segment_output, background = samples[row_index]
                row[0].imshow(segment_input)
                row[1].imshow(segment_input)
                mask = np.asarray(segment_output.convert("L")) > 0
                if mask.any() and not mask.all():
                    row[1].contour(
                        mask,
                        levels=(0.5,),
                        colors=("#B6423C",),
                        linewidths=1.5,
                    )
                if background is not None:
                    row[2].imshow(background)
            save_grid(figure, path, rc)
        finally:
            plt.close(figure)


def _draw_metrics(
    path: Path, metrics: tuple[_Metric, ...], parser_backbones: str
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    total_instances = sum(row[2] for row in metrics)
    total_shape_iou = sum(row[2] * row[6] for row in metrics)
    total_retention = sum(row[2] * row[7] for row in metrics)
    rows = [
        _metric_row(dataset, parser_backbones, *values)
        for dataset, *values in metrics
    ]
    rows.append(
        _metric_row(
            "TOTAL",
            parser_backbones,
            sum(row[1] for row in metrics),
            total_instances,
            sum(row[3] for row in metrics),
            sum(row[4] for row in metrics),
            sum(row[5] for row in metrics),
            total_shape_iou / total_instances if total_instances else 0.0,
            total_retention / total_instances if total_instances else 0.0,
        )
    )
    rc = paper_matplotlib_rc()
    with matplotlib.rc_context(rc):
        figure, axis = plt.subplots(figsize=(20.0, 2.4))
        try:
            axis.set_axis_off()
            table = axis.table(
                cellText=rows,
                colLabels=(
                    "dataset",
                    "parser_backbones",
                    "detected_inputs",
                    "parsed_instances",
                    "usable",
                    "review",
                    "unusable",
                    "usable_rate",
                    "mean_shape_iou",
                    "mean_ownership_retention",
                ),
                cellLoc="center",
                loc="center",
                colWidths=(
                    0.09,
                    0.20,
                    0.08,
                    0.09,
                    0.06,
                    0.06,
                    0.07,
                    0.08,
                    0.11,
                    0.16,
                ),
            )
            format_table(table, len(rows), size=7.5)
            figure.savefig(
                path,
                format="pdf",
                dpi=rc["savefig.dpi"],
                facecolor=rc["savefig.facecolor"],
                bbox_inches="tight",
                pad_inches=0.04,
            )
        finally:
            plt.close(figure)


def _metric_row(
    dataset: str,
    parser_backbones: str,
    detected: int,
    instances: int,
    usable: int,
    review: int,
    unusable: int,
    shape_iou: float,
    retention: float,
) -> tuple[str, ...]:
    return (
        dataset,
        parser_backbones,
        str(detected),
        str(instances),
        str(usable),
        str(review),
        str(unusable),
        f"{usable / instances:.3f}" if instances else "0.000",
        f"{shape_iou:.3f}",
        f"{retention:.3f}",
    )
