"""Pipeline observer plates. Paper FIGURE_REGISTRY 00-17 is a different sequence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visualization.privacy import PublicationScope, validate_publishable_value
from visualization.rendering.style import paper_matplotlib_rc

STAGE_LAYOUT: dict[str, tuple[str, tuple[str, ...]]] = {
    "parsing": (
        "00_parsing",
        (
            "00_detection",
            "01_segmentation",
            "02_regions",
            "03_quality",
            "04_crops",
        ),
    ),
    "identification": (
        "01_identification",
        ("00_appearance", "01_face", "02_nose"),
    ),
    "representation": (
        "02_representation",
        ("00_evidence", "01_channels", "02_quality"),
    ),
    "enrollment": (
        "03_enrollment",
        ("00_registry", "01_write"),
    ),
    "gallery": ("04_gallery", ("00_store",)),
    "search": ("05_search", ("00_scoring", "01_matching")),
}

_ACTIVATIONS_ABSENT = "activations absent"
_OPTIMIZATION_SCHEMA = "evaluation.optimization_protocol.v1"


def vis_directory(output_root: Path, stage: str) -> Path:
    vis_dir, _substages = STAGE_LAYOUT[stage]
    return output_root / "vis" / vis_dir


def render_stage(
    stage: str,
    trace: dict[str, Any],
    output_root: Path,
    *,
    asset_root: Path | None = None,
    scope: PublicationScope = PublicationScope.PRIVATE,
) -> tuple[Path, ...]:
    """Write inner 00_/01_ substages under Visualization/vis/0N_<stage>/."""

    if stage not in STAGE_LAYOUT:
        raise ValueError(f"unknown visualization stage: {stage}")
    catalog = write_optimization_catalog(stage, trace, output_root)
    if catalog:
        return catalog
    declared = trace.get("stage")
    if declared is not None and declared != stage:
        raise ValueError(f"trace stage {declared!r} does not match {stage!r}")
    validate_publishable_value(trace, scope)
    vis_dir, substages = STAGE_LAYOUT[stage]
    stage_root = output_root / "vis" / vis_dir
    stage_root.mkdir(parents=True, exist_ok=True)
    payload = trace.get("substages")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("trace substages must be an object")
    written: list[Path] = []
    for name in substages:
        sub = payload.get(name)
        if sub is None:
            sub = {}
        if not isinstance(sub, dict):
            raise ValueError(f"substage {name} must be an object")
        target = stage_root / name
        target.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": stage,
            "substage": name,
            "summary": sub.get("summary") or "absent",
            "activations": _activation_status(sub.get("activations")),
        }
        json_path = target / "trace.json"
        json_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(json_path)
        image_path = (
            _resolve_image(sub.get("image"), asset_root)
            if sub.get("image") is not None
            else None
        )
        if image_path is not None:
            png_path = target / "trace.png"
            _draw_image(png_path, image_path)
            written.append(png_path)
    return tuple(written)


def write_optimization_catalog(
    stage: str,
    trace: dict[str, Any],
    output_root: Path,
) -> tuple[Path, ...]:
    """Write vis/0N_<stage>/optimization.json from an optimization protocol trace.

    Runtime rows (prototype, operations, evaluation.integrity) stay on the
    protocol document, not a vis/06 directory. JSON only; no plates.
    """

    payload = _optimization_stage_substages(stage, trace)
    if payload is None:
        return ()
    protocol = trace["protocol"]
    vis_dir, _substages = STAGE_LAYOUT[stage]
    stage_root = output_root / "vis" / vis_dir
    stage_root.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": protocol["schema_version"],
        "interpretation": protocol["interpretation"],
        "stage": stage,
        "substages": payload,
    }
    path = stage_root / "optimization.json"
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (path,)


def _optimization_stage_substages(
    stage: str,
    trace: dict[str, Any],
) -> dict[str, list[Any]] | None:
    protocol = trace.get("protocol")
    if not isinstance(protocol, dict):
        return None
    if protocol.get("schema_version") != _OPTIMIZATION_SCHEMA:
        return None
    if "runtime" not in trace:
        return None
    nested = trace.get("substages")
    if nested is None:
        nested = {}
    if not isinstance(nested, dict):
        raise ValueError("trace substages must be an object")
    _vis_dir, names = STAGE_LAYOUT[stage]
    if stage in nested and isinstance(nested[stage], dict):
        source = nested[stage]
    else:
        source = nested
    payload: dict[str, list[Any]] = {}
    for name in names:
        rows = source.get(name)
        if rows is None:
            payload[name] = []
            continue
        if not isinstance(rows, list):
            raise ValueError(f"optimization substage {name} must be a list")
        payload[name] = rows
    return payload


def _activation_status(value: Any) -> str:
    if value is None:
        return _ACTIVATIONS_ABSENT
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, (list, tuple)) and first:
            return "spatial activation map present"
        return _ACTIVATIONS_ABSENT
    return _ACTIVATIONS_ABSENT


def _resolve_image(value: Any, asset_root: Path | None) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("trace image must be a relative path string")
    if asset_root is None:
        raise ValueError("trace image requires --asset-root")
    from visualization.privacy import validate_relative_asset_path

    relative = validate_relative_asset_path(value)
    path = (asset_root.resolve(strict=True) / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _draw_image(target: Path, image_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from PIL import Image

    rc = paper_matplotlib_rc()
    with matplotlib.rc_context(rc):
        figure = plt.figure(figsize=(4.0, 4.0))
        try:
            ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
            ax.set_axis_off()
            ax.imshow(Image.open(image_path).convert("RGB"))
            figure.savefig(
                target,
                format="png",
                dpi=rc["savefig.dpi"],
                facecolor=rc["savefig.facecolor"],
                bbox_inches="tight",
                pad_inches=0.0,
            )
        finally:
            plt.close(figure)
