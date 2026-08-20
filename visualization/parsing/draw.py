"""Parsing observer. Writes Visualization/vis/00_parsing/."""

from pathlib import Path
from typing import Any

from visualization.rendering.pipeline import STAGE_LAYOUT, render_stage

STAGE = "parsing"
VIS_DIR, SUBSTAGES = STAGE_LAYOUT[STAGE]


def render(trace: dict[str, Any], output_root: Path, **kwargs: Any) -> tuple[Path, ...]:
    return render_stage(STAGE, trace, output_root, **kwargs)
