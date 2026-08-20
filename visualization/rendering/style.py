"""Fixed publication style shared by all Matplotlib recipes."""

from __future__ import annotations

from typing import Any

from shared.foundation.provenance import content_sha256

STYLE_VERSION = "visualization.style.v1"
FONT_FAMILY = "DejaVu Sans"
COLORS = {
    "ink": "#17212B",
    "muted": "#61707D",
    "grid": "#D8DEE3",
    "paper": "#FAF8F3",
    "blue": "#2667A9",
    "teal": "#16857B",
    "orange": "#D97925",
    "red": "#B6423C",
    "purple": "#72559D",
    "gold": "#B6952E",
}
SERIES_COLORS = (
    COLORS["blue"],
    COLORS["orange"],
    COLORS["teal"],
    COLORS["purple"],
    COLORS["red"],
    COLORS["gold"],
)
FIGURE_SIZE = (8.0, 4.5)
DPI = 160
FIXED_STYLE: dict[str, Any] = {
    "version": STYLE_VERSION,
    "font_family": FONT_FAMILY,
    "colors": COLORS,
    "series_colors": SERIES_COLORS,
    "figure_size_inches": FIGURE_SIZE,
    "dpi": DPI,
    "line_width": 1.6,
    "marker_size": 5.0,
}
STYLE_FINGERPRINT = content_sha256(FIXED_STYLE)

PAPER_STYLE_VERSION = "visualization.paper_style.v1"
PAPER_COLORS = {
    "ink": "#111111",
    "muted": "#4A4A4A",
    "rule": "#111111",
    "paper": "#FFFFFF",
}
PAPER_DPI = 300
PAPER_STYLE: dict[str, Any] = {
    "version": PAPER_STYLE_VERSION,
    "font_family": FONT_FAMILY,
    "colors": PAPER_COLORS,
    "figure_width_inches": 7.2,
    "dpi": PAPER_DPI,
}


def matplotlib_rc() -> dict[str, Any]:
    return {
        "axes.edgecolor": COLORS["ink"],
        "axes.facecolor": COLORS["paper"],
        "axes.grid": True,
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": 0.8,
        "figure.dpi": DPI,
        "figure.facecolor": COLORS["paper"],
        "font.family": FONT_FAMILY,
        "font.size": 9,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": FIXED_STYLE["line_width"],
        "savefig.dpi": DPI,
        "savefig.facecolor": COLORS["paper"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
    }


def paper_matplotlib_rc() -> dict[str, Any]:
    """White, caption-free publication rc. Does not alter observer STYLE_FINGERPRINT."""

    return {
        "axes.edgecolor": PAPER_COLORS["ink"],
        "axes.facecolor": PAPER_COLORS["paper"],
        "axes.grid": False,
        "axes.labelcolor": PAPER_COLORS["ink"],
        "axes.linewidth": 0.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": False,
        "axes.spines.left": False,
        "axes.titlepad": 0.0,
        "axes.titlesize": 0.0,
        "figure.dpi": PAPER_DPI,
        "figure.facecolor": PAPER_COLORS["paper"],
        "font.family": FONT_FAMILY,
        "font.size": 8,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": PAPER_DPI,
        "savefig.facecolor": PAPER_COLORS["paper"],
        "savefig.pad_inches": 0.04,
        "text.color": PAPER_COLORS["ink"],
        "xtick.bottom": False,
        "xtick.labelbottom": False,
        "ytick.left": False,
        "ytick.labelleft": False,
    }
