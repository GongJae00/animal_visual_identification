"""Lazy deterministic Matplotlib renderer for static publication formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visualization.contracts import FigureData
from visualization.rendering.recipes import draw_recipe, validate_recipe
from visualization.registry import FIGURE_BY_ID
from visualization.rendering.style import FIGURE_SIZE, matplotlib_rc


def render_static_figure(
    data: FigureData,
    output_root: Path,
    *,
    asset_root: Path | None = None,
) -> tuple[str, ...]:
    """Render PNG without importing Matplotlib at package import."""

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
            draw_recipe(figure, data, asset_root=asset_root)
            successor_results = (
                data.kind == "result_forest" and "absolute_rows" in data.payload
            )
            bottom_margin = {
                "ranked_retrieval": 0.10,
            }.get(data.kind, 0.12)
            if successor_results:
                bottom_margin = 0.14
            if data.kind == "result_forest" and not successor_results:
                left_margin = 0.34
            elif data.kind == "census":
                left_margin = 0.30
            else:
                left_margin = 0.12
            figure.subplots_adjust(
                left=left_margin, right=0.96, top=0.96, bottom=bottom_margin
            )
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
    if extension != "png":
        raise ValueError(f"unsupported static format: {extension}")
    return {"Software": "visualization.renderer.v1"}
