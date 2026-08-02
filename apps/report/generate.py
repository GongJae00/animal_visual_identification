"""Generate publication-oriented, privacy-safe IdentityEngine report visualizations."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data_pipeline.source_lock import get_record


WIDTH = 1800
HEIGHT = 1050
REPORT_SCHEMA = "cvi.dogface_holdout_fusion_evaluation.v1"
REPORT_STATUS = "PASS_DOGFACE_HOLDOUT_FUSION_EVALUATION"
PROVENANCE_SCHEMA = "cvi.report_visualizations.v1"
ROI_BUNDLE_SCHEMA = "cvi.canid_roi_manifest_bundle.v2"
ROI_MANIFEST_SCHEMA = "cvi.canid_roi_manifest.v2"
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

INK = "#07121f"
NAVY = "#0d1d2d"
PANEL = "#11283a"
IVORY = "#f4eddc"
MUTED = "#91a4ae"
GRID = "#264052"
CORAL = "#ff775f"
ORANGE = "#f4a259"
CYAN = "#53d8e8"
LIME = "#b9e769"
WHITE = "#ffffff"

DATASET_SPECS: tuple[tuple[str, str, str, Callable[[Path], tuple[Any, ...]]], ...]

REQUIRED_OUTPUTS = (
    "INDEX.md",
    "provenance.json",
    "00_dataset/00_dataset_overview.png",
    "01_preprocessing/01_preprocessing_pipeline.png",
    "01_preprocessing/01_pipeline_boundary.svg",
    "01_preprocessing/01_pipeline_boundary.png",
    "02_embedding/02_embedding_architecture.svg",
    "02_embedding/02_embedding_architecture.png",
    "02_embedding/02_embedding_geometry.svg",
    "02_embedding/02_embedding_geometry.png",
    "03_retrieval/03_gallery_retrieval.svg",
    "03_retrieval/03_gallery_retrieval.png",
    "04_calibration_fusion/04_oof_calibration_fusion.svg",
    "04_calibration_fusion/04_oof_calibration_fusion.png",
    "04_calibration_fusion/04_calibration_selection.svg",
    "04_calibration_fusion/04_calibration_selection.png",
    "05_evaluation/05_final_results.svg",
    "05_evaluation/05_final_results.png",
    "06_diagnostics/06_failure_diagnostics.svg",
    "06_diagnostics/06_failure_diagnostics.png",
)

_FORBIDDEN_REPORT_KEYS = {
    "sample_id",
    "sample_ids",
    "sample_token",
    "sample_tokens",
    "identity_token",
    "identity_tokens",
    "public_subject_token",
    "dataset_identity_id",
    "query_rows",
    "gallery_identity_order",
    "embeddings",
    "vectors",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError("input paths must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"input is not a regular file: {path}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"input is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def _finite_metric(value: Any, name: str, *, unit_interval: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (unit_interval and not 0.0 <= result <= 1.0):
        raise ValueError(f"{name} is outside its valid range")
    return result


def _walk_privacy(value: Any, trail: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _FORBIDDEN_REPORT_KEYS:
                raise ValueError(f"evaluation report contains forbidden private field: {key}")
            _walk_privacy(child, (*trail, str(key)))
    elif isinstance(value, list):
        for child in value:
            _walk_privacy(child, trail)
    elif isinstance(value, str):
        if value.startswith(("/", "\\\\")) or (len(value) > 2 and value[1:3] in {":\\", ":/"}):
            raise ValueError("evaluation report contains an absolute path")


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("evaluation report schema differs")
    if report.get("status") != REPORT_STATUS:
        raise ValueError("evaluation report did not pass")
    privacy = report.get("privacy")
    if not isinstance(privacy, dict) or any(
        privacy.get(key) is not False
        for key in (
            "per_pair_values_serialized",
            "per_sample_ids_serialized",
            "per_sample_vectors_serialized",
            "query_rows_serialized",
        )
    ):
        raise ValueError("evaluation report privacy declaration differs")
    _walk_privacy(report)
    extract_metrics(report)


def _metric_block(value: Any, name: str) -> dict[str, float | int]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    result: dict[str, float | int] = {
        "rank1": _finite_metric(value.get("Rank-1"), f"{name}.Rank-1"),
        "mrr": _finite_metric(value.get("MRR"), f"{name}.MRR"),
    }
    for key in ("num_queries", "num_gallery_identities", "num_gallery_templates"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{name}.{key} must be a positive integer")
        result[key] = item
    if value.get("closed_set") is not True:
        raise ValueError(f"{name} must be closed-set")
    return result


def extract_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only aggregate report values consumed by figures."""

    try:
        calibration = report["calibration_results"]
        final = report["final_results"]
        selection = report["frozen_selection"]
        diagnostic_root = report["embedding_diagnostics"]["final"]
    except (KeyError, TypeError) as exc:
        raise ValueError("evaluation report is missing required aggregate sections") from exc
    one_oof = calibration.get("one_shot_oof")
    candidates = calibration.get("three_shot_oof_candidates")
    simplex = calibration.get("simplex")
    fold_protocol = calibration.get("fold_protocol")
    if not all(isinstance(value, dict) for value in (one_oof, candidates, simplex, fold_protocol)):
        raise ValueError("calibration report structure differs")
    folds = fold_protocol.get("n_splits")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("calibration fold count differs")
    names = simplex.get("channel_names")
    weights = simplex.get("weights")
    if names != ["Appearance-v3", "Face-v4"] or not isinstance(weights, list) or len(weights) != 2:
        raise ValueError("fusion channel contract differs")
    parsed_weights = [_finite_metric(value, "simplex weight") for value in weights]
    if not math.isclose(sum(parsed_weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("simplex weights do not sum to one")
    selected_id = selection.get("selected_aggregation_id")
    if not isinstance(selected_id, str) or selected_id not in candidates:
        raise ValueError("selected aggregation is not a calibration candidate")
    if selection.get("simplex_weights") != weights:
        raise ValueError("frozen and calibration weights differ")

    final_output: dict[str, Any] = {}
    for shot in ("one_shot", "three_shot"):
        block = final.get(shot)
        if not isinstance(block, dict) or not isinstance(block.get("channel_context"), dict):
            raise ValueError(f"final {shot} structure differs")
        context = block["channel_context"]
        parsed = {
            "appearance": _metric_block(context.get("Appearance-v3-max"), f"final.{shot}.appearance"),
            "face": _metric_block(context.get("Face-v4-max"), f"final.{shot}.face"),
            "fused": _metric_block(block.get("fused"), f"final.{shot}.fused"),
        }
        ci_root = block.get("fused_identity_clustered_ci", {}).get("Rank-1")
        if not isinstance(ci_root, dict) or ci_root.get("confidence_level") != 0.95:
            raise ValueError(f"final {shot} Rank-1 confidence interval differs")
        parsed["rank1_ci"] = {
            "lower": _finite_metric(ci_root.get("lower_bound"), "CI lower"),
            "upper": _finite_metric(ci_root.get("upper_bound"), "CI upper"),
            "cohorts": ci_root.get("cluster_count"),
            "queries": ci_root.get("query_row_count"),
        }
        if not all(isinstance(parsed["rank1_ci"][key], int) and parsed["rank1_ci"][key] > 0 for key in ("cohorts", "queries")):
            raise ValueError("confidence interval counts differ")
        final_output[shot] = parsed

    diagnostics: dict[str, Any] = {}
    diagnostic_names = {
        "appearance": "Appearance-v3-pre-L2-CLS",
        "baseline": "Face-v4-DINO-baseline-384D",
        "face": "Face-v4-final-640D",
        "regional": "Face-v4-regional-projection-256D",
    }
    for short, full in diagnostic_names.items():
        block = diagnostic_root.get(full)
        if not isinstance(block, dict) or block.get("schema_version") != "cvi.embedding_diagnostics.v1":
            raise ValueError(f"missing diagnostic block: {full}")
        normalized = block.get("normalized_centered_covariance_spectrum")
        geometry = block.get("normalized_directional_geometry")
        norms = block.get("raw_norm_summary")
        if not all(isinstance(value, dict) for value in (normalized, geometry, norms)):
            raise ValueError(f"diagnostic structure differs: {full}")
        nominal = block.get("nominal_dimension")
        if isinstance(nominal, bool) or not isinstance(nominal, int) or nominal <= 0:
            raise ValueError(f"diagnostic nominal dimension differs: {full}")
        diagnostics[short] = {
            "nominal": nominal,
            "effective_rank": _finite_metric(
                normalized.get("effective_rank_entropy"),
                f"{full}.effective_rank",
                unit_interval=False,
            ),
            "centroid_norm": _finite_metric(geometry.get("centroid_norm"), f"{full}.centroid_norm"),
            "raw_norm_mean": _finite_metric(norms.get("mean"), f"{full}.raw_norm_mean", unit_interval=False),
            "raw_norm_std": _finite_metric(norms.get("standard_deviation"), f"{full}.raw_norm_std", unit_interval=False),
        }
    return {
        "folds": folds,
        "weights": parsed_weights,
        "selected_aggregation": selected_id,
        "calibration_one": {
            "appearance": _metric_block(one_oof.get("Appearance-v3"), "calibration.appearance"),
            "face": _metric_block(one_oof.get("Face-v4"), "calibration.face"),
            "fused": _metric_block(one_oof.get("fused"), "calibration.fused"),
        },
        "calibration_candidates": {
            key: _metric_block(value, f"calibration.candidate.{key}")
            for key, value in sorted(candidates.items())
        },
        "final": final_output,
        "diagnostics": diagnostics,
    }


def svg_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


@dataclass(frozen=True)
class Op:
    kind: str
    values: tuple[Any, ...]


class Scene:
    def __init__(self, figure: str, title: str, subtitle: str, footer: str) -> None:
        self.ops: list[Op] = []
        self.rect(0, 0, WIDTH, HEIGHT, INK)
        for x in range(40, WIDTH, 80):
            self.line(x, 0, x, HEIGHT, GRID, 1)
        for y in range(40, HEIGHT, 80):
            self.line(0, y, WIDTH, y, GRID, 1)
        self.text(70, 55, figure, 20, LIME, bold=True)
        self.text(70, 95, title, 48, IVORY, bold=True)
        self.text(72, 160, subtitle, 22, MUTED)
        self.line(70, 205, 1730, 205, GRID, 2)
        self.line(70, 990, 1730, 990, GRID, 2)
        self.text(70, 1005, footer, 16, MUTED)

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str | None = None, width: int = 1, radius: int = 0, dash: bool = False) -> None:
        self.ops.append(Op("rect", (x, y, w, h, fill, stroke, width, radius, dash)))

    def line(self, x1: float, y1: float, x2: float, y2: float, fill: str, width: int = 2, dash: bool = False) -> None:
        self.ops.append(Op("line", (x1, y1, x2, y2, fill, width, dash)))

    def text(self, x: float, y: float, value: object, size: int, fill: str = IVORY, *, bold: bool = False, anchor: str = "start") -> None:
        self.ops.append(Op("text", (x, y, str(value), size, fill, bold, anchor)))

    def circle(self, x: float, y: float, radius: float, fill: str, stroke: str | None = None, width: int = 1) -> None:
        self.ops.append(Op("circle", (x, y, radius, fill, stroke, width)))

    def polygon(self, points: Sequence[tuple[float, float]], fill: str) -> None:
        self.ops.append(Op("polygon", (tuple(points), fill)))


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if not path.is_file():
        raise FileNotFoundError(f"required font is unavailable: {path}")
    return ImageFont.truetype(str(path), size=size)


def _hex(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _dash_line(draw: ImageDraw.ImageDraw, coords: tuple[float, float, float, float], fill: str, width: int) -> None:
    x1, y1, x2, y2 = coords
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    for start in np.arange(0, length, 18):
        stop = min(start + 10, length)
        a = start / length
        b = stop / length
        draw.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=fill, width=width)


def render_png(scene: Scene, path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), _hex(INK))
    draw = ImageDraw.Draw(image)
    for op in scene.ops:
        values = op.values
        if op.kind == "rect":
            x, y, w, h, fill, stroke, width, radius, dash = values
            box = (x, y, x + w, y + h)
            draw.rounded_rectangle(box, radius=radius, fill=fill, outline=None if dash else stroke, width=width)
            if dash and stroke:
                _dash_line(draw, (x, y, x + w, y), stroke, width)
                _dash_line(draw, (x + w, y, x + w, y + h), stroke, width)
                _dash_line(draw, (x + w, y + h, x, y + h), stroke, width)
                _dash_line(draw, (x, y + h, x, y), stroke, width)
        elif op.kind == "line":
            x1, y1, x2, y2, fill, width, dash = values
            if dash:
                _dash_line(draw, (x1, y1, x2, y2), fill, width)
            else:
                draw.line((x1, y1, x2, y2), fill=fill, width=width)
        elif op.kind == "text":
            x, y, value, size, fill, bold, anchor = values
            draw.text((x, y), value, font=_font(size, bold), fill=fill, anchor={"start": "la", "middle": "ma", "end": "ra"}[anchor])
        elif op.kind == "circle":
            x, y, radius, fill, stroke, width = values
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=stroke, width=width)
        elif op.kind == "polygon":
            points, fill = values
            draw.polygon(points, fill=fill)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=9, optimize=False)


def render_svg(scene: Scene, path: Path) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
    ]
    for op in scene.ops:
        values = op.values
        if op.kind == "rect":
            x, y, w, h, fill, stroke, width, radius, dash = values
            attrs = f'x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}"'
            if stroke:
                attrs += f' stroke="{stroke}" stroke-width="{width}"'
            if dash:
                attrs += ' stroke-dasharray="10 8"'
            lines.append(f"<rect {attrs}/>")
        elif op.kind == "line":
            x1, y1, x2, y2, fill, width, dash = values
            extra = ' stroke-dasharray="10 8"' if dash else ""
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{fill}" stroke-width="{width}"{extra}/>' )
        elif op.kind == "text":
            x, y, value, size, fill, bold, anchor = values
            weight = "700" if bold else "400"
            lines.append(f'<text x="{x}" y="{y + size}" fill="{fill}" font-family="DejaVu Sans, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{svg_escape(value)}</text>')
        elif op.kind == "circle":
            x, y, radius, fill, stroke, width = values
            outline = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
            lines.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}"{outline}/>' )
        elif op.kind == "polygon":
            points, fill = values
            encoded = " ".join(f"{x},{y}" for x, y in points)
            lines.append(f'<polygon points="{encoded}" fill="{fill}"/>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_scene(scene: Scene, root: Path, stem: str) -> None:
    render_svg(scene, root / f"{stem}.svg")
    render_png(scene, root / f"{stem}.png")


def _label(scene: Scene, x: float, y: float, value: str, color: str) -> None:
    width = max(100, len(value) * 13 + 28)
    scene.rect(x, y, width, 34, NAVY, color, 2, 17)
    scene.text(x + 14, y + 5, value, 16, color, bold=True)


def _arrow(scene: Scene, x1: float, y1: float, x2: float, y2: float, color: str = MUTED, dash: bool = False) -> None:
    scene.line(x1, y1, x2, y2, color, 3, dash)
    angle = math.atan2(y2 - y1, x2 - x1)
    scene.polygon(
        ((x2, y2), (x2 - 16 * math.cos(angle - 0.5), y2 - 16 * math.sin(angle - 0.5)), (x2 - 16 * math.cos(angle + 0.5), y2 - 16 * math.sin(angle + 0.5))),
        color,
    )


def _box(scene: Scene, x: float, y: float, w: float, h: float, title: str, lines: Sequence[str], color: str, *, dash: bool = False) -> None:
    scene.rect(x, y, w, h, PANEL, color, 2, 18, dash)
    scene.text(x + 24, y + 18, title, 24, color, bold=True)
    for index, line in enumerate(lines):
        scene.text(x + 24, y + 65 + index * 31, line, 17, IVORY if index == 0 else MUTED)


def _pipeline_boundary_scene() -> Scene:
    scene = Scene("FIG 03", "Runtime boundary, made explicit", "Research localization is upstream; the canonical public API is crop-level.", "Source: README + api.py + localization modules | Dashed branch is not connected product capability")
    _label(scene, 90, 245, "UPSTREAM EXPERIMENTAL", CYAN)
    _box(scene, 90, 310, 270, 170, "Detector", ("dog bounding box", "research adapter"), CYAN, dash=True)
    _box(scene, 420, 310, 270, 170, "Pose", ("17 AP-10K points", "research adapter"), CYAN, dash=True)
    _box(scene, 750, 310, 270, 170, "ROI", ("dog / face / nose", "crop derivation"), CYAN, dash=True)
    _arrow(scene, 360, 395, 420, 395, CYAN, True)
    _arrow(scene, 690, 395, 750, 395, CYAN, True)
    _arrow(scene, 1020, 395, 1140, 535, CYAN, True)
    scene.line(1100, 235, 1100, 940, LIME, 3)
    scene.text(1120, 250, "CANONICAL IdentityEngine", 18, LIME, bold=True)
    _box(scene, 1140, 450, 560, 170, "Caller-provided crop", ("IdentityEngine starts here", "PIL.Image enrollment / search"), LIME)
    _arrow(scene, 1420, 620, 1420, 685, LIME)
    _box(scene, 1140, 685, 560, 170, "Closed-set candidates", ("enrolled gallery only", "not an identity decision"), LIME)
    scene.rect(90, 610, 930, 245, NAVY, GRID, 2, 18)
    scene.text(120, 640, "DOGFACE FINAL EVALUATION", 20, ORANGE, bold=True)
    scene.text(120, 690, "Publisher face crops", 31, IVORY, bold=True)
    scene.text(120, 745, "No detector / pose / ROI stage in the final holdout.", 20, MUTED)
    scene.text(120, 790, "This evaluation is research evidence, not bundled runtime performance.", 18, MUTED)
    return scene


def _embedding_architecture_scene(metrics: Mapping[str, Any]) -> Scene:
    weights = metrics["weights"]
    scene = Scene("FIG 04", "Two embedding paths, one frozen selection", "Exact dimensional contracts and the calibration-frozen channel weights.", "Source: trainer.py + face_id/model.py + final evaluation report | Dimensions are architecture contracts")
    _label(scene, 90, 245, "APPEARANCE", CORAL)
    _box(scene, 90, 310, 360, 160, "Appearance-v3", ("DINOv2 CLS", "384D -> L2"), CORAL)
    _arrow(scene, 450, 390, 570, 390, CORAL)
    _box(scene, 570, 310, 250, 160, "384D", ("direction", "cosine-ready"), CORAL)
    _label(scene, 90, 545, "FACE", CYAN)
    _box(scene, 90, 610, 310, 170, "Baseline", ("DINO pooler", "384D"), CYAN)
    _box(scene, 470, 610, 310, 170, "Regional", ("anatomical pooling", "256D x 0.25"), CYAN)
    _arrow(scene, 400, 695, 470, 695, CYAN)
    _arrow(scene, 780, 695, 900, 695, CYAN)
    _box(scene, 900, 610, 270, 170, "Concatenate", ("384 + 256", "640D"), CYAN)
    _arrow(scene, 1170, 695, 1270, 695, CYAN)
    _box(scene, 1270, 610, 220, 170, "L2", ("640D", "unit norm"), LIME)
    scene.rect(1030, 290, 660, 230, NAVY, LIME, 3, 20)
    scene.text(1065, 320, "FROZEN FUSION", 19, LIME, bold=True)
    scene.text(1065, 375, f"Appearance  {weights[0]:.2f}", 34, CORAL, bold=True)
    scene.text(1065, 425, f"Face            {weights[1]:.2f}", 34, CYAN, bold=True)
    scene.text(1065, 475, "selected before final scoring", 17, MUTED)
    return scene


def _embedding_geometry_scene(metrics: Mapping[str, Any]) -> Scene:
    diagnostics = metrics["diagnostics"]
    scene = Scene("FIG 05", "Embedding geometry at final evaluation", "Nominal dimension is not the same as effective directional diversity.", "Source: aggregate final embedding diagnostics | Effective rank is entropy rank; no eigenvalue spectrum is reconstructed")
    rows = (("Appearance 384D", "appearance", CORAL), ("Face baseline 384D", "baseline", CYAN), ("Face final 640D", "face", LIME), ("Regional 256D", "regional", ORANGE))
    scene.text(100, 260, "REPRESENTATION", 16, MUTED, bold=True)
    scene.text(650, 260, "EFFECTIVE / NOMINAL RANK", 16, MUTED, bold=True)
    scene.text(1490, 260, "CENTROID NORM", 16, MUTED, bold=True)
    for index, (label, key, color) in enumerate(rows):
        y = 330 + index * 145
        data = diagnostics[key]
        ratio = data["effective_rank"] / data["nominal"]
        scene.text(100, y, label, 25, color, bold=True)
        scene.rect(650, y + 5, 430, 28, NAVY, GRID, 1, 14)
        scene.rect(650, y + 5, 430 * ratio, 28, color, None, 1, 14)
        scene.text(1100, y, f'{data["effective_rank"]:.1f} / {data["nominal"]}  ({ratio:.1%})', 20, IVORY)
        scene.text(1510, y, f'{data["centroid_norm"]:.4f}', 25, color, bold=True)
    scene.rect(100, 875, 1590, 70, NAVY, ORANGE, 2, 14)
    scene.text(125, 892, "Regional directions are nearly co-linear in this cohort; diagnostics show collapse, not its cause.", 20, IVORY)
    return scene


def _retrieval_scene() -> Scene:
    scene = Scene("FIG 06", "Gallery retrieval contract", "Template scores become stable identity-level closed-set candidates.", "Source: evaluation/retrieval.py | Canonical aggregation=max; alternatives are research-only and not selected")
    boxes = (
        (80, "Query crop", ("embedding", "L2 direction"), CORAL),
        (390, "Gallery templates", ("many per identity", "enrolled set"), CYAN),
        (720, "Cosine matrix", ("query x template", "exact scores"), IVORY),
        (1050, "Identity max", ("canonical max", "per identity"), LIME),
        (1380, "Stable ranking", ("first occurrence", "closed-set output"), LIME),
    )
    for x, title, lines, color in boxes:
        _box(scene, x, 360, 270, 200, title, lines, color)
    for x in (350, 680, 1010, 1340):
        _arrow(scene, x, 460, x + 40, 460, MUTED)
    scene.rect(260, 680, 1280, 180, NAVY, GRID, 2, 18)
    scene.text(300, 710, "RESEARCH ALTERNATIVES", 18, MUTED, bold=True)
    scene.text(300, 765, "mean  /  median  /  top-k mean  /  log-mean-exp  /  quality-weighted mean", 24, IVORY)
    scene.text(300, 815, "Available for evaluation; calibration selected max for the reported three-shot result.", 18, MUTED)
    return scene


def _oof_scene(metrics: Mapping[str, Any]) -> Scene:
    folds = metrics["folds"]
    scene = Scene("FIG 07", "OOF calibration and fusion freeze", "Identity-disjoint folds fit held-out probabilities before the one-time final boundary.", "Source: calibration fold protocol + oof_simplex.py | OOF provenance is caller-attested in the report")
    scene.text(90, 255, f"{folds} IDENTITY-DISJOINT FOLDS", 18, CYAN, bold=True)
    for index in range(folds):
        x = 90 + index * 205
        scene.rect(x, 315, 165, 180, PANEL, CYAN, 2, 15)
        scene.text(x + 82, 335, f"FOLD {index + 1}", 18, CYAN, bold=True, anchor="middle")
        scene.text(x + 82, 390, "fit on 4", 17, IVORY, anchor="middle")
        scene.text(x + 82, 430, "score held-out", 15, MUTED, anchor="middle")
    _arrow(scene, 1120, 405, 1230, 405, MUTED)
    _box(scene, 1230, 315, 470, 180, "Isotonic OOF", ("fold-specific calibration", "no held-out fold in fit"), ORANGE)
    _arrow(scene, 1465, 495, 1465, 590, MUTED)
    _box(scene, 1230, 590, 470, 170, "Quality gate + simplex", ("available channels only", "weighted Brier grid"), LIME)
    _arrow(scene, 1230, 675, 1080, 675, LIME)
    _box(scene, 610, 590, 470, 170, "Freeze", ("weights + aggregation", "before final inference"), LIME)
    scene.line(535, 550, 535, 925, LIME, 4)
    scene.text(555, 850, "ONE-TIME FINAL", 20, LIME, bold=True)
    scene.text(555, 888, "No refit or reselection", 17, MUTED)
    return scene


def _bar_chart(scene: Scene, rows: Sequence[tuple[str, float, str]], *, x: int, y: int, width: int, step: int = 70) -> None:
    for index, (label, value, color) in enumerate(rows):
        yy = y + index * step
        scene.text(x, yy, label, 17, IVORY)
        scene.rect(x + 230, yy + 2, width, 24, NAVY, GRID, 1, 12)
        scene.rect(x + 230, yy + 2, width * value, 24, color, None, 1, 12)
        scene.text(x + 245 + width, yy - 2, f"{value:.3f}", 18, color, bold=True)


def _calibration_selection_scene(metrics: Mapping[str, Any]) -> Scene:
    one = metrics["calibration_one"]
    candidates = metrics["calibration_candidates"]
    selected = metrics["selected_aggregation"]
    scene = Scene("FIG 08", "Calibration selection", "One-shot channel context and three-shot OOF aggregation candidates.", "Source: actual calibration report | Rank-1/MRR are aggregate closed-set calibration results")
    scene.text(90, 255, "ONE-SHOT OOF", 18, MUTED, bold=True)
    _bar_chart(scene, (("Appearance Rank-1", one["appearance"]["rank1"], CORAL), ("Face Rank-1", one["face"]["rank1"], CYAN), ("Fused Rank-1", one["fused"]["rank1"], LIME), ("Appearance MRR", one["appearance"]["mrr"], CORAL), ("Face MRR", one["face"]["mrr"], CYAN), ("Fused MRR", one["fused"]["mrr"], LIME)), x=90, y=305, width=420, step=62)
    scene.line(890, 245, 890, 920, GRID, 2)
    scene.text(950, 255, "THREE-SHOT CANDIDATE RANK-1", 18, MUTED, bold=True)
    rows = []
    for key, value in candidates.items():
        rows.append((key.replace("log_mean_exp_", "LME ").replace("top_k_mean_", "top-k "), value["rank1"], LIME if key == selected else ORANGE))
    _bar_chart(scene, rows, x=950, y=305, width=380, step=68)
    scene.text(950, 875, f"SELECTED  {selected}   |   weights A/F = {metrics['weights'][0]:.2f}/{metrics['weights'][1]:.2f}", 20, LIME, bold=True)
    return scene


def _final_results_scene(metrics: Mapping[str, Any]) -> Scene:
    scene = Scene("FIG 09", "Final closed-set holdout", "One-shot and three-shot channel context with frozen fusion and identity-clustered uncertainty.", "Source: actual final report | Public DogFace holdout; not lifelong biometric or deployment validation")
    for shot_index, shot in enumerate(("one_shot", "three_shot")):
        data = metrics["final"][shot]
        x = 90 + shot_index * 850
        title = "ONE-SHOT" if shot == "one_shot" else "THREE-SHOT"
        scene.text(x, 255, title, 22, LIME, bold=True)
        scene.text(x, 295, f"{data['rank1_ci']['cohorts']} identities  /  {data['rank1_ci']['queries']} queries", 17, MUTED)
        rows = []
        for channel, color in (("appearance", CORAL), ("face", CYAN), ("fused", LIME)):
            rows.extend(((f"{channel.title()} Rank-1", data[channel]["rank1"], color), (f"{channel.title()} MRR", data[channel]["mrr"], color)))
        _bar_chart(scene, rows, x=x, y=350, width=410, step=68)
        ci = data["rank1_ci"]
        scene.rect(x, 810, 760, 100, NAVY, LIME, 2, 14)
        scene.text(x + 25, 828, "FUSED RANK-1 95% CI", 17, MUTED, bold=True)
        scene.text(x + 25, 863, f"{data['fused']['rank1']:.3f}   [{ci['lower']:.3f}, {ci['upper']:.3f}]", 25, LIME, bold=True)
    return scene


def _diagnostics_scene(metrics: Mapping[str, Any]) -> Scene:
    diagnostics = metrics["diagnostics"]
    scene = Scene("FIG 10", "Failure diagnostics and next actions", "Aggregate geometry isolates a regional collapse signal without inventing examples or causes.", "Source: actual final aggregate diagnostics | No raw vectors, per-query rows, spectra, or failure examples shown")
    rows = (("Appearance", "appearance", CORAL), ("Face baseline", "baseline", CYAN), ("Face final", "face", LIME), ("Regional", "regional", ORANGE))
    scene.text(90, 260, "EFFECTIVE-RANK UTILIZATION", 17, MUTED, bold=True)
    for index, (label, key, color) in enumerate(rows):
        y = 315 + index * 92
        data = diagnostics[key]
        ratio = data["effective_rank"] / data["nominal"]
        scene.text(90, y, label, 19, color, bold=True)
        scene.rect(300, y + 3, 330, 22, NAVY, GRID, 1, 11)
        scene.rect(300, y + 3, 330 * ratio, 22, color, None, 1, 11)
        scene.text(650, y - 2, f"{ratio:.1%}", 18, IVORY)
    scene.text(90, 735, "RAW NORM BEHAVIOR", 17, MUTED, bold=True)
    scene.text(90, 780, f"Face final: {diagnostics['face']['raw_norm_mean']:.6f} +/- {diagnostics['face']['raw_norm_std']:.2e} (post-L2)", 19, LIME)
    scene.text(90, 825, f"Regional pre-L2: {diagnostics['regional']['raw_norm_mean']:.3f} +/- {diagnostics['regional']['raw_norm_std']:.3f}", 19, ORANGE)
    scene.line(850, 245, 850, 925, GRID, 2)
    scene.text(920, 260, "SIGNAL", 17, ORANGE, bold=True)
    scene.text(920, 310, f"Regional centroid norm  {diagnostics['regional']['centroid_norm']:.6f}", 28, ORANGE, bold=True)
    scene.text(920, 365, "Near-unity centroid + low rank utilization", 20, IVORY)
    scene.text(920, 405, "is consistent with directional collapse.", 20, IVORY)
    scene.text(920, 485, "ACTIONABLE NEXT STEPS", 17, LIME, bold=True)
    for index, line in enumerate(("1  Audit regional training gradients and loss balance", "2  Track pre/post-normalization rank during training", "3  Use a new identity-disjoint calibration cohort", "4  Re-evaluate fusion after a frozen selection protocol")):
        scene.text(920, 535 + index * 62, line, 20, IVORY if index < 2 else MUTED)
    scene.text(920, 825, "Causality is not established by these diagnostics.", 18, MUTED)
    return scene


@dataclass(frozen=True)
class SelectedImage:
    canonical_name: str
    path: Path
    sha256: str
    role: str


@dataclass(frozen=True)
class _LocatedImage:
    image_path: str
    image_sha256: str


def _image_files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(directory.iterdir(), key=lambda value: value.name)
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _located(root: Path, path: Path) -> tuple[_LocatedImage, ...]:
    verified = _safe_relative(root, path.relative_to(root).as_posix())
    with Image.open(verified) as opened:
        opened.verify()
    return (_LocatedImage(path.relative_to(root).as_posix(), _sha256_file(verified)),)


def _locate_dogfacenet224(root: Path) -> tuple[_LocatedImage, ...]:
    base = root / "after_4_bis"
    if not base.is_dir():
        raise FileNotFoundError(f"DogFaceNet base not found: {base}")
    for identity_dir in sorted(base.iterdir(), key=lambda value: value.name):
        if identity_dir.is_dir():
            images = _image_files(identity_dir)
            if images:
                return _located(root, images[0])
    return ()


def _locate_mpdd(root: Path) -> tuple[_LocatedImage, ...]:
    base = root / "MPDD" / "pytorch"
    if not base.is_dir():
        raise FileNotFoundError(f"MPDD base not found: {base}")
    for split in ("train", "val", "query", "gallery"):
        for path in _image_files(base / split):
            if len(path.stem.split("_")) >= 3:
                return _located(root, path)
    return ()


def _locate_sibetan(root: Path) -> tuple[_LocatedImage, ...]:
    base = root / "Sibetan"
    if not base.is_dir():
        raise FileNotFoundError(f"Sibetan base not found: {base}")
    for cluster in sorted(base.iterdir(), key=lambda value: value.name):
        if cluster.is_dir() and cluster.name.isdigit():
            images = _image_files(cluster)
            if images:
                return _located(root, images[0])
    return ()


def _locate_yt_bb_dog(root: Path) -> tuple[_LocatedImage, ...]:
    base = root / "YT-BB-dog" / "YT-BB-Dog"
    if not base.is_dir():
        raise FileNotFoundError(f"YT-BB-Dog base not found: {base}")
    for split in ("train", "test"):
        split_root = base / split
        if not split_root.is_dir():
            continue
        for identity_dir in sorted(split_root.iterdir(), key=lambda value: value.name):
            if identity_dir.is_dir():
                images = _image_files(identity_dir)
                if images:
                    return _located(root, images[0])
    return ()


DATASET_SPECS = (
    ("dogfacenet224", "dogfacenet224", "TRAIN", _locate_dogfacenet224),
    ("mpdd", "mpdd", "VALIDATION", _locate_mpdd),
    ("sibetan", "sibetan", "VALIDATION", _locate_sibetan),
    ("yt-bb-dog", "yt-bb-dog", "TRAIN", _locate_yt_bb_dog),
)


def _safe_relative(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or relative != pure.as_posix() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("unsafe relative image path")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*pure.parts).resolve(strict=True)
    if not candidate.is_relative_to(resolved_root) or not candidate.is_file() or candidate.is_symlink():
        raise ValueError("image path does not resolve to a safe regular file")
    return candidate


def load_roi_record(roi_manifest_path: Path, ap10k_root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    bundle = _load_json(roi_manifest_path)
    if bundle.get("schema_version") != ROI_BUNDLE_SCHEMA or set(bundle) != {"schema_version", "manifest_sha256", "manifest"}:
        raise ValueError("ROI manifest bundle schema differs")
    manifest = bundle.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ROI_MANIFEST_SCHEMA:
        raise ValueError("ROI manifest schema differs")
    if _canonical_sha256(manifest) != bundle.get("manifest_sha256"):
        raise ValueError("ROI manifest digest differs")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("ROI records must be a list")
    required = (
        "image_path",
        "image_sha256",
        "dog_crop_path",
        "dog_crop_sha256",
        "face_crop_path",
        "face_crop_sha256",
        "weak_nose_crop_path",
        "weak_nose_crop_sha256",
        "source_valid_mask_path",
        "source_valid_mask_sha256",
        "body_keypoints",
        "dog_bbox_xyxy",
    )
    eligible = [record for record in records if isinstance(record, dict) and all(record.get(key) is not None for key in required)]
    if not eligible:
        raise ValueError("ROI manifest has no complete dog/face/nose/mask record")
    record = min(eligible, key=lambda value: str(value["image_sha256"]))
    manifest_root = roi_manifest_path.resolve(strict=True).parent
    paths = {
        "source": _safe_relative(ap10k_root, record["image_path"]),
        "dog": _safe_relative(manifest_root, record["dog_crop_path"]),
        "face": _safe_relative(manifest_root, record["face_crop_path"]),
        "nose": _safe_relative(manifest_root, record["weak_nose_crop_path"]),
        "mask": _safe_relative(manifest_root, record["source_valid_mask_path"]),
    }
    hashes = {
        "source": record["image_sha256"],
        "dog": record["dog_crop_sha256"],
        "face": record["face_crop_sha256"],
        "nose": record["weak_nose_crop_sha256"],
        "mask": record["source_valid_mask_sha256"],
    }
    for key, path in paths.items():
        expected = hashes[key]
        if not isinstance(expected, str) or len(expected) != 64 or _sha256_file(path) != expected:
            raise ValueError(f"ROI {key} image hash differs")
        with Image.open(path) as opened:
            opened.verify()
    points = record["body_keypoints"]
    if not isinstance(points, dict) or len(points) != 17:
        raise ValueError("ROI record must contain all 17 body keypoints")
    return record, paths


def select_dataset_images(
    data_root: Path,
    ap_source_relative: str,
    ap_source_hash: str,
) -> tuple[SelectedImage, ...]:
    datasets_root = data_root.resolve(strict=True) / "datasets"
    if not datasets_root.is_dir():
        raise ValueError("data root must contain datasets/")
    ap_root = datasets_root / "ap10k"
    ap_path = _safe_relative(ap_root, ap_source_relative)
    if _sha256_file(ap_path) != ap_source_hash:
        raise ValueError("AP10K ROI source hash differs during dataset selection")
    selected: list[SelectedImage] = [
        SelectedImage("ap10k-dog", ap_path, ap_source_hash, "LOCALIZATION")
    ]
    for canonical_name, directory, role, adapter in DATASET_SPECS:
        root = datasets_root / directory
        samples = adapter(root)
        if not samples:
            raise ValueError(f"dataset adapter found no publication sample: {canonical_name}")
        sample = min(samples, key=lambda value: (value.image_sha256, value.image_path))
        path = _safe_relative(root, sample.image_path)
        if _sha256_file(path) != sample.image_sha256:
            raise ValueError(f"dataset adapter image hash differs: {canonical_name}")
        selected.append(SelectedImage(canonical_name, path, sample.image_sha256, role))
    return tuple(selected)


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize((math.ceil(image.width * scale), math.ceil(image.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _contain(
    image: Image.Image,
    size: tuple[int, int],
) -> tuple[Image.Image, float, float, float]:
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    offset_x = (target_w - resized.width) / 2
    offset_y = (target_h - resized.height) / 2
    canvas = Image.new("RGB", size, _hex(NAVY))
    canvas.paste(resized, (round(offset_x), round(offset_y)))
    return canvas, scale, offset_x, offset_y


def _base_raster(figure: str, title: str, subtitle: str, footer: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), _hex(INK))
    draw = ImageDraw.Draw(image)
    for x in range(40, WIDTH, 80):
        draw.line((x, 0, x, HEIGHT), fill=GRID, width=1)
    for y in range(40, HEIGHT, 80):
        draw.line((0, y, WIDTH, y), fill=GRID, width=1)
    draw.text((70, 55), figure, font=_font(20, True), fill=LIME)
    draw.text((70, 95), title, font=_font(48, True), fill=IVORY)
    draw.text((72, 160), subtitle, font=_font(22), fill=MUTED)
    draw.line((70, 205, 1730, 205), fill=GRID, width=2)
    draw.line((70, 990, 1730, 990), fill=GRID, width=2)
    draw.text((70, 1005), footer, font=_font(16), fill=MUTED)
    return image, draw


def render_dataset_overview(selected: Sequence[SelectedImage], path: Path) -> None:
    image, draw = _base_raster("FIG 01", "Five public datasets, five distinct roles", "Actual CC-BY-4.0 source images shown only as low-resolution composite derivatives.", "Source: registry-recorded public datasets | No identity labels, sample tokens, paths, or owner data")
    card_w, card_h, gap = 310, 650, 25
    for index, item in enumerate(selected):
        x = 75 + index * (card_w + gap)
        y = 265
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=18, fill=PANEL, outline=GRID, width=2)
        with Image.open(item.path) as opened:
            photo = _cover(opened.convert("RGB"), (card_w - 24, 440))
        image.paste(photo, (x + 12, y + 12))
        record = get_record(item.canonical_name)
        draw.text((x + 18, y + 478), record.official_name, font=_font(19, True), fill=IVORY)
        draw.text((x + 18, y + 530), item.role, font=_font(17, True), fill=LIME if item.role != "VALIDATION" else CYAN)
        draw.text((x + 18, y + 573), record.license_id, font=_font(16), fill=MUTED)
        draw.text((x + 18, y + 610), "ACTUAL PUBLIC SAMPLE", font=_font(13, True), fill=CORAL)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=9, optimize=False)


_SKELETON = (
    ("left_eye", "nose_center"), ("right_eye", "nose_center"), ("left_eye", "neck"), ("right_eye", "neck"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_front_paw"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_front_paw"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_back_paw"), ("right_hip", "right_knee"), ("right_knee", "right_back_paw"),
    ("left_hip", "tail_base"), ("right_hip", "tail_base"),
)


def render_preprocessing(record: Mapping[str, Any], paths: Mapping[str, Path], path: Path) -> None:
    image, draw = _base_raster("FIG 02", "Experimental preprocessing trace", "One hash-verified AP-10K record: source -> dog -> face -> weak nose -> source-valid mask.", "Source: AP-10K + experimental ROI manifest | Localization is upstream research, not canonical IdentityEngine runtime")
    with Image.open(paths["source"]) as opened:
        source = opened.convert("RGB")
    source_box = (70, 270, 700, 870)
    rendered, scale, offset_x, offset_y = _contain(
        source,
        (source_box[2] - source_box[0], source_box[3] - source_box[1]),
    )
    image.paste(rendered, source_box[:2])
    def point(raw_x: float, raw_y: float) -> tuple[float, float]:
        return (
            source_box[0] + offset_x + raw_x * scale,
            source_box[1] + offset_y + raw_y * scale,
        )
    bbox = record["dog_bbox_xyxy"]
    p1, p2 = point(bbox[0], bbox[1]), point(bbox[2], bbox[3])
    draw.rectangle((*p1, *p2), outline=CORAL, width=5)
    keypoints = record["body_keypoints"]
    for first, second in _SKELETON:
        if first in keypoints and second in keypoints:
            draw.line((*point(*keypoints[first][:2]), *point(*keypoints[second][:2])), fill=CYAN, width=3)
    for coordinates in keypoints.values():
        x, y = point(*coordinates[:2])
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=LIME)
    draw.rounded_rectangle((85, 285, 280, 325), radius=18, fill=INK)
    draw.text((105, 294), "01 SOURCE + POSE", font=_font(15, True), fill=IVORY)
    cards = (("02 DOG CROP", "dog", CORAL), ("03 FACE CROP", "face", CYAN), ("04 WEAK NOSE", "nose", ORANGE), ("05 VALID MASK", "mask", LIME))
    for index, (label, key, color) in enumerate(cards):
        x = 780 + (index % 2) * 450
        y = 270 + (index // 2) * 340
        draw.rounded_rectangle((x, y, x + 390, y + 285), radius=18, fill=PANEL, outline=color, width=3)
        with Image.open(paths[key]) as opened:
            tile = _cover(opened.convert("RGB"), (250, 250))
        image.paste(tile, (x + 18, y + 18))
        draw.text((x + 285, y + 50), label, font=_font(17, True), fill=color)
        description = {"dog": "square padded", "face": "pose-derived", "nose": "weak ROI", "mask": "source pixels"}[key]
        draw.text((x + 285, y + 105), description, font=_font(15), fill=MUTED)
        if index < len(cards) - 1:
            draw.text((x + 285, y + 180), "->", font=_font(28, True), fill=IVORY)
    draw.text((780, 900), "All displayed crops are derivatives; no raw source file is copied into the report.", font=_font(17), fill=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=9, optimize=False)


def _index_text() -> str:
    return """# IdentityEngine Report Visualization Index

이 디렉터리는 보고서 편집을 위한 재현 가능한 시각화 산출물입니다. 사진이 포함된 두 도판은 승인된 CC-BY-4.0 공개 데이터의 저해상도 합성 파생물이며, 나머지 수치는 실제 최종 평가 보고서의 집계값에서 생성됩니다. 개체 라벨, 샘플 식별자, 소유자 정보, 절대 경로, 원시 임베딩, 질의별 결과는 포함하지 않습니다.

## Figure Index

1. **Dataset overview** (`00_dataset/00_dataset_overview.png`)  
   다섯 공개 데이터셋의 실제 대표 이미지와 연구 역할을 비교합니다.  
   *Five actual public samples with train, validation, and localization roles.*
2. **Experimental preprocessing** (`01_preprocessing/01_preprocessing_pipeline.png`)  
   AP-10K 원본에서 검출 상자, 17개 포즈 점, dog/face/weak-nose crop, 유효 마스크까지의 실험적 전처리를 보입니다.  
   *A hash-verified AP-10K source-to-ROI trace.*
3. **Runtime boundary** (`01_preprocessing/01_pipeline_boundary.svg`, `.png`)  
   검출·포즈·ROI는 상류 실험 분기이고, 공개 IdentityEngine는 호출자가 제공한 crop에서 시작함을 구분합니다.  
   *Experimental localization is upstream; canonical IdentityEngine starts at a caller-provided crop.*
4. **Embedding architecture** (`02_embedding/02_embedding_architecture.svg`, `.png`)  
   Appearance 384D와 Face 384D + 256D regional 경로, 640D L2 출력, 동결된 융합 가중치를 표시합니다.  
   *Exact embedding dimensions and frozen Appearance/Face fusion weights.*
5. **Embedding geometry** (`02_embedding/02_embedding_geometry.svg`, `.png`)  
   명목 차원, entropy effective rank, 방향 중심 크기를 비교하여 regional collapse 신호를 제한적으로 해석합니다.  
   *Nominal versus effective rank and directional centroid metrics.*
6. **Gallery retrieval** (`03_retrieval/03_gallery_retrieval.svg`, `.png`)  
   cosine template score, identity-level max, 안정 정렬, closed-set 후보 출력의 계약을 설명합니다.  
   *Canonical max aggregation and stable closed-set identity ranking.*
7. **OOF calibration and fusion** (`04_calibration_fusion/04_oof_calibration_fusion.svg`, `.png`)  
   identity-disjoint 5-fold isotonic OOF 보정과 최종 평가 전 동결 경계를 나타냅니다.  
   *Five-fold OOF calibration, quality gate, simplex, and pre-final freeze.*
8. **Calibration selection** (`04_calibration_fusion/04_calibration_selection.svg`, `.png`)  
   실제 one-shot A/F/fused 결과와 three-shot 후보 Rank-1, 선택된 max를 보여줍니다.  
   *Actual calibration metrics and the selected max aggregation.*
9. **Final results** (`05_evaluation/05_final_results.svg`, `.png`)  
   one-/three-shot A/F/frozen-fusion Rank-1·MRR와 융합 Rank-1 95% CI를 보고합니다.  
   *Actual final closed-set metrics, cohort/query counts, and fused Rank-1 CIs.*
10. **Failure diagnostics** (`06_diagnostics/06_failure_diagnostics.svg`, `.png`)  
    effective-rank 활용률, 중심 크기, norm 거동, regional collapse 신호와 후속 조치를 정리합니다.  
    *Aggregate diagnostics and bounded next actions; no invented spectra or examples.*

## Capability Boundary

**Actual evidence:** 다섯 공개 데이터셋의 대표 이미지, 검증된 AP-10K ROI 기록, 최종 DogFace holdout 보고서의 집계 지표.  
**Evaluation-only / upstream research:** detector, pose, ROI localization은 upstream 실험 경로이며 canonical `canine_identity.IdentityEngine`에 연결되어 있지 않습니다. OOF calibration과 A/F fusion은 DogFace holdout 평가에서 실제 실행됐지만 canonical runtime에는 연결되지 않았습니다. 공개 런타임은 caller-provided crop의 closed-set enrollment/retrieval만 수행하며, 최종 DogFace 평가는 publisher face crop에서 시작해 검출·포즈 단계를 포함하지 않습니다.

## Attribution

| Dataset | License | Official source |
|---|---|---|
| AP-10K domestic dog subset | CC-BY-4.0 | https://github.com/AlexTheBad/AP-10K |
| DogFaceNet 224 (resized) | CC-BY-4.0 | https://zenodo.org/records/12578449 |
| Multi-pose dog dataset | CC-BY-4.0 | https://data.mendeley.com/datasets/v5j6m8dzhv/1 |
| Sibetan | CC-BY-4.0 | https://www.lirmm.fr/YT-BB-Dog_Sibetan/ |
| YT-BB-Dog | CC-BY-4.0 | https://www.lirmm.fr/YT-BB-Dog_Sibetan/ |

이미지 파생물은 각 원 데이터셋의 CC-BY-4.0 조건을 유지하며 위 출처에 귀속됩니다. 저장소 코드의 Apache-2.0 라이선스가 데이터셋 권리를 대체하지 않습니다.

## Reproduction

```bash
export CANINE_IDENTITY_SECURE_DATA_ROOT=/path/to/canine_video_identity_secure
export CANINE_IDENTITY_ROI_MANIFEST=/path/to/roi_manifest.json
export CANINE_IDENTITY_EVALUATION_REPORT=/path/to/evaluation.json

uv run python apps/report/generate.py \\
  --data-root "$CANINE_IDENTITY_SECURE_DATA_ROOT" \\
  --roi-manifest "$CANINE_IDENTITY_ROI_MANIFEST" \\
  --evaluation-report "$CANINE_IDENTITY_EVALUATION_REPORT" \\
  --output-dir /path/outside/repository/identity-report
```

생성기는 기존 출력 디렉터리를 덮어쓰지 않습니다. 동일한 입력 바이트, 소스 코드, Pillow/폰트 환경에서 결정적으로 생성되며 `provenance.json`에 입력·출력 SHA-256을 기록합니다.
"""


def _write_provenance(root: Path, selected: Sequence[SelectedImage], report_path: Path) -> None:
    generator_path = Path(__file__).resolve(strict=True)
    outputs = {}
    for relative in REQUIRED_OUTPUTS:
        if relative == "provenance.json":
            continue
        path = root / relative
        outputs[relative] = {"sha256": _sha256_file(path), "byte_size": path.stat().st_size}
    datasets = []
    for item in selected:
        record = get_record(item.canonical_name)
        datasets.append({
            "canonical_name": record.canonical_name,
            "official_name": record.official_name,
            "license": record.license_id,
            "source_url": record.url,
            "source_image_sha256": item.sha256,
        })
    payload = {
        "schema_version": PROVENANCE_SCHEMA,
        "generator": {
            "source": "apps/report/generate.py",
            "source_sha256": _sha256_file(generator_path),
        },
        "evaluation_report_sha256": _sha256_file(report_path),
        "datasets": datasets,
        "outputs": outputs,
        "privacy": {
            "aggregate_only": True,
            "contains_absolute_paths": False,
            "contains_identity_or_sample_fields": False,
            "contains_raw_vectors_or_per_query_results": False,
        },
        "derivative_and_attribution_statement": (
            "Photo panels are low-resolution composite derivatives of the listed "
            "CC-BY-4.0 datasets and remain attributable to their official sources; "
            "no standalone source image is redistributed."
        ),
    }
    (root / "provenance.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def generate(data_root: Path, roi_manifest: Path, evaluation_report: Path, output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink() or os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if not output_dir.parent.resolve(strict=True).is_dir():
        raise ValueError("output directory parent must exist")
    report = _load_json(evaluation_report)
    validate_report(report)
    metrics = extract_metrics(report)
    ap10k_root = data_root.resolve(strict=True) / "datasets" / "ap10k"
    record, roi_paths = load_roi_record(roi_manifest, ap10k_root)
    selected = select_dataset_images(
        data_root,
        record["image_path"],
        record["image_sha256"],
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        render_dataset_overview(selected, temporary / "00_dataset/00_dataset_overview.png")
        render_preprocessing(record, roi_paths, temporary / "01_preprocessing/01_preprocessing_pipeline.png")
        _save_scene(_pipeline_boundary_scene(), temporary, "01_preprocessing/01_pipeline_boundary")
        _save_scene(_embedding_architecture_scene(metrics), temporary, "02_embedding/02_embedding_architecture")
        _save_scene(_embedding_geometry_scene(metrics), temporary, "02_embedding/02_embedding_geometry")
        _save_scene(_retrieval_scene(), temporary, "03_retrieval/03_gallery_retrieval")
        _save_scene(_oof_scene(metrics), temporary, "04_calibration_fusion/04_oof_calibration_fusion")
        _save_scene(_calibration_selection_scene(metrics), temporary, "04_calibration_fusion/04_calibration_selection")
        _save_scene(_final_results_scene(metrics), temporary, "05_evaluation/05_final_results")
        _save_scene(_diagnostics_scene(metrics), temporary, "06_diagnostics/06_failure_diagnostics")
        (temporary / "INDEX.md").write_text(_index_text(), encoding="utf-8")
        _write_provenance(temporary, selected, evaluation_report)
        missing = [relative for relative in REQUIRED_OUTPUTS if not (temporary / relative).is_file()]
        if missing:
            raise RuntimeError(f"generator inventory is incomplete: {missing}")
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--roi-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    generate(args.data_root, args.roi_manifest, args.evaluation_report, args.output_dir)
    print(json.dumps({"status": "PASS_REPORT_VISUALIZATIONS", "output_count": len(REQUIRED_OUTPUTS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
