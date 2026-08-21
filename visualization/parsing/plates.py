"""Small rendering primitives shared by parsing plates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def save_grid(figure: Any, path: Path, rc: Mapping[str, Any]) -> None:
    figure.subplots_adjust(0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    figure.savefig(
        path,
        format="pdf",
        dpi=rc["savefig.dpi"],
        facecolor=rc["savefig.facecolor"],
        bbox_inches="tight",
        pad_inches=0.0,
    )


def format_table(table: Any, rows: int, *, size: float) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    table.scale(1.0, 1.8)
    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#111111")
        cell.set_linewidth(0.5)
        if row_index in (0, rows):
            cell.set_text_props(weight="bold")
