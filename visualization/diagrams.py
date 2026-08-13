"""Deterministic architecture and evidence-ladder diagram recipes."""

from __future__ import annotations

import textwrap
from collections import defaultdict
from typing import Any

from visualization.style import COLORS, SERIES_COLORS


def draw_architecture(ax: Any, payload: dict[str, Any]) -> None:
    nodes = payload["nodes"]
    layers: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, node in enumerate(nodes):
        layers[node["layer"]].append((index, node))
    positions: dict[int, tuple[float, float]] = {}
    maximum_layer = max(layers, default=0)
    for layer in sorted(layers):
        items = layers[layer]
        x = 0.08 + 0.84 * layer / max(maximum_layer, 1)
        for row, (index, node) in enumerate(items):
            y = 0.85 - 0.7 * (row + 0.5) / len(items)
            positions[index] = (x, y)
            color = SERIES_COLORS[node["group_index"] % len(SERIES_COLORS)]
            ax.text(
                x,
                y,
                _wrapped(node["label"], width=18),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7.2,
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": COLORS["paper"],
                    "edgecolor": color,
                    "linewidth": 1.5,
                },
                zorder=3,
            )
    for edge in payload["edges"]:
        start = positions[edge["source"]]
        stop = positions[edge["target"]]
        ax.annotate(
            "",
            xy=stop,
            xytext=start,
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["muted"],
                "linewidth": 1.1,
                "shrinkA": 24,
                "shrinkB": 24,
            },
            zorder=1,
        )
        if edge["label"]:
            ax.text(
                (start[0] + stop[0]) / 2,
                (start[1] + stop[1]) / 2 + 0.025,
                edge["label"],
                color=COLORS["muted"],
                fontsize=6.5,
                ha="center",
                transform=ax.transAxes,
            )
    ax.set_axis_off()


def draw_ladder(ax: Any, payload: dict[str, Any]) -> None:
    from matplotlib.patches import Rectangle

    status_colors = {
        "established": COLORS["teal"],
        "conditional": COLORS["orange"],
        "out_of_scope": COLORS["muted"],
    }
    steps = payload["steps"]
    row_gap = 0.82 / len(steps)
    bar_height = min(0.09, row_gap * 0.68)
    for index, step in enumerate(steps):
        y = 0.90 - (index + 0.5) * row_gap
        color = status_colors[step["status"]]
        ax.add_patch(
            Rectangle(
                (0.02, y - bar_height / 2),
                0.32,
                bar_height,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                alpha=0.92,
            )
        )
        ax.text(
            0.03,
            y,
            _wrapped(step["label"], width=28),
            transform=ax.transAxes,
            va="center",
            ha="left",
            color="white" if step["status"] != "out_of_scope" else COLORS["ink"],
            fontsize=7.8 if len(steps) > 8 else 8.4,
            fontweight="bold",
        )
        ax.text(
            0.35,
            y,
            _wrapped(step["detail"], width=70 if len(steps) <= 7 else 84),
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=6.6 if len(steps) > 8 else 7.0,
            color=COLORS["muted"],
        )
    ax.set_axis_off()


def _wrapped(value: str, *, width: int) -> str:
    return "\n".join(
        textwrap.fill(line, width=width, break_long_words=False, break_on_hyphens=False)
        for line in value.splitlines()
    )
