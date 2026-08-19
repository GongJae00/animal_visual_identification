"""Lazy deterministic Matplotlib renderer for static publication formats."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visualization.contracts import FigureData
from visualization.rendering.recipes import draw_recipe, validate_recipe
from visualization.registry import FIGURE_BY_ID
from visualization.rendering.style import FIGURE_SIZE, matplotlib_rc

_FIXED_TIME = datetime(2000, 1, 1, tzinfo=UTC)


def render_static_figure(
    data: FigureData,
    output_root: Path,
    *,
    asset_root: Path | None = None,
) -> tuple[str, ...]:
    """Render SVG, PDF, and PNG without importing Matplotlib at package import."""

    validate_recipe(data)
    spec = FIGURE_BY_ID[data.figure_id]
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    output_root.mkdir(parents=True, exist_ok=True)
    relative_paths: list[str] = []
    rc = matplotlib_rc()
    with matplotlib.rc_context(rc):
        figure = plt.figure(figsize=FIGURE_SIZE, constrained_layout=False)
        try:
            figure.suptitle(
                data.title, x=0.06, y=0.975, ha="left", fontsize=13, fontweight="bold"
            )
            draw_recipe(figure, data, asset_root=asset_root)
            successor_results = (
                data.kind == "result_forest" and "absolute_rows" in data.payload
            )
            bottom_margin = {
                "embedding_diagnostics": 0.25,
                "ranked_retrieval": 0.22,
            }.get(data.kind, 0.18)
            if successor_results:
                bottom_margin = 0.25
            if data.kind == "result_forest" and not successor_results:
                left_margin = 0.34
            elif data.kind == "census":
                left_margin = 0.30
            else:
                left_margin = 0.12
            figure.subplots_adjust(
                left=left_margin, right=0.94, top=0.84, bottom=bottom_margin
            )
            figure.text(0.06, 0.035, data.limitations[0], fontsize=6.5, color="#61707D")
            for extension in spec.primary_formats:
                relative = f"figures/{data.figure_id}.{extension}"
                target = output_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(
                    target,
                    format=extension,
                    metadata=_metadata(extension),
                    dpi=rc["savefig.dpi"],
                )
                relative_paths.append(relative)
        finally:
            plt.close(figure)
    return tuple(relative_paths)


def matplotlib_version() -> str:
    import matplotlib

    return str(matplotlib.__version__)


def _metadata(extension: str) -> dict[str, Any]:
    if extension == "svg":
        return {"Creator": "cvi.vis.renderer.v1", "Date": None}
    if extension == "pdf":
        return {
            "Creator": "cvi.vis.renderer.v1",
            "Producer": "cvi.vis.renderer.v1",
            "CreationDate": _FIXED_TIME,
            "ModDate": _FIXED_TIME,
        }
    if extension == "png":
        return {"Software": "cvi.vis.renderer.v1"}
    raise ValueError(f"unsupported static format: {extension}")
