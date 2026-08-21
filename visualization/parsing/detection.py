"""Detection plates for ``Visualization/vis/00_parsing/00_detection``."""

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
_Sample = tuple[Any, tuple[tuple[float, float, float, float], ...], Any | None]
_Metric = tuple[str, int, int, int, int]


def render(
    payload: Mapping[str, Any], target: Path, loader: AssetLoader
) -> tuple[tuple[Path, ...], Mapping[str, Any]]:
    datasets = payload.get("samples")
    if datasets is None:
        return (), {}
    if not isinstance(datasets, Mapping):
        raise ValueError("detection samples must be an object")
    written: list[Path] = []
    sample_metrics: list[_Metric] = []
    for dataset_name in sorted(datasets):
        _validate_dataset_name(dataset_name, "detection")
        raw_samples = datasets[dataset_name]
        if not isinstance(raw_samples, list):
            raise ValueError("detection samples must be arrays")
        if not raw_samples:
            continue
        samples = tuple(
            _load_sample(value, loader) for value in raw_samples[:_GRID_ROWS]
        )
        detected = sum(bool(boxes) for _source, boxes, _segment in samples)
        multi_box = sum(len(boxes) > 1 for _source, boxes, _segment in samples)
        sample_metrics.append(
            (
                dataset_name,
                len(samples),
                detected,
                len(samples) - detected,
                multi_box,
            )
        )
        path = target / f"detection_box_{dataset_name}.pdf"
        _draw_samples(path, samples)
        written.append(path)

    metrics = _read_metrics(payload.get("metrics"), sample_metrics)
    if metrics:
        backbone = payload.get("detection_backbone")
        if backbone is not None and (
            not isinstance(backbone, str) or not backbone.strip()
        ):
            raise ValueError("detection_backbone must be non-empty text")
        path = target / "detection_metrics.pdf"
        _draw_metrics(path, metrics, backbone or "unspecified")
        written.append(path)
    return tuple(written), {}


def _validate_dataset_name(value: Any, kind: str) -> None:
    if not isinstance(value, str) or not _DATASET_FILENAME.fullmatch(value):
        raise ValueError(f"{kind} dataset name is not a safe filename")


def _load_sample(value: Any, loader: AssetLoader) -> _Sample:
    if not isinstance(value, Mapping):
        raise ValueError("detection sample must be an object")
    segment_value = value.get("segment_image")
    return (
        loader.image(value.get("source_image")),
        tuple(_box(box) for box in _boxes(value.get("detector_boxes"))),
        loader.image(segment_value) if segment_value is not None else None,
    )


def _boxes(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("detection detector_boxes must be an array")
    return value


def _box(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("detection box must contain four coordinates")
    coordinates = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in coordinates):
        raise ValueError("detection box coordinates must be finite")
    x1, y1, x2, y2 = coordinates
    if not x1 < x2 or not y1 < y2:
        raise ValueError("detection box must be non-empty")
    return coordinates


def _read_metrics(value: Any, fallback: list[_Metric]) -> tuple[_Metric, ...]:
    if value is None:
        return tuple(fallback)
    if not isinstance(value, list):
        raise ValueError("detection metrics must be an array")
    result: list[_Metric] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("detection metric must be an object")
        dataset_name = item.get("dataset_name")
        _validate_dataset_name(dataset_name, "detection metric")
        fields = tuple(
            item.get(name)
            for name in (
                "input_images",
                "detected_samples",
                "undetected_samples",
                "multi_box_samples",
            )
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in fields
        ):
            raise ValueError("detection metric counts must be nonnegative integers")
        input_images, detected, undetected, multi_box = fields
        if detected + undetected != input_images or multi_box > detected:
            raise ValueError("detection metric counts are inconsistent")
        result.append((dataset_name, input_images, detected, undetected, multi_box))
    return tuple(result)


def _draw_samples(path: Path, samples: tuple[_Sample, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

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
                source, boxes, segment = samples[row_index]
                row[0].imshow(source)
                row[1].imshow(source)
                for x1, y1, x2, y2 in boxes:
                    row[1].add_patch(
                        Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            fill=False,
                            edgecolor="#B6423C",
                            linewidth=2.0,
                        )
                    )
                if segment is not None:
                    row[2].imshow(segment)
            save_grid(figure, path, rc)
        finally:
            plt.close(figure)


def _draw_metrics(path: Path, metrics: tuple[_Metric, ...], backbone: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    total = tuple(sum(row[index] for row in metrics) for index in range(1, 5))
    rows = [
        _metric_row(dataset, backbone, input_images, detected, undetected, multi_box)
        for dataset, input_images, detected, undetected, multi_box in metrics
    ]
    rows.append(_metric_row("TOTAL", backbone, *total))
    rc = paper_matplotlib_rc()
    with matplotlib.rc_context(rc):
        figure, axis = plt.subplots(figsize=(11.5, 2.0))
        try:
            axis.set_axis_off()
            table = axis.table(
                cellText=rows,
                colLabels=(
                    "dataset",
                    "detection_backbone",
                    "input_images",
                    "detected_samples",
                    "undetected_samples",
                    "detection_rate",
                    "multi_box_samples",
                ),
                cellLoc="center",
                loc="center",
                colWidths=(0.10, 0.24, 0.09, 0.13, 0.15, 0.14, 0.15),
            )
            format_table(table, len(rows), size=8)
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
    backbone: str,
    input_images: int,
    detected: int,
    undetected: int,
    multi_box: int,
) -> tuple[str, ...]:
    return (
        dataset,
        backbone,
        str(input_images),
        str(detected),
        str(undetected),
        f"{detected / input_images:.3f}" if input_images else "0.000",
        str(multi_box),
    )
