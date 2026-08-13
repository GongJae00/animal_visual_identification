"""Fixed publication style shared by all Matplotlib recipes."""

from __future__ import annotations

from typing import Any

from foundation.provenance import content_sha256

STYLE_VERSION = "cvi.vis.style.v1"
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
        "pdf.fonttype": 42,
        "savefig.dpi": DPI,
        "savefig.facecolor": COLORS["paper"],
        "svg.fonttype": "none",
        "svg.hashsalt": "cvi-vis-v1",
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
    }
