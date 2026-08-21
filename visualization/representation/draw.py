"""Representation stage adapter."""

from pathlib import Path
from typing import Any

from visualization.rendering.pipeline import STAGE_LAYOUT, render_stage
from visualization.representation.channels import render as render_channels

STAGE = "representation"
VIS_DIR, _ = STAGE_LAYOUT[STAGE]
SUBSTAGES = ("01_channels",)
_RENDERERS = {"01_channels": render_channels}


def render(trace: dict[str, Any], output_root: Path, **kwargs: Any) -> tuple[Path, ...]:
    return render_stage(
        STAGE,
        trace,
        output_root,
        **kwargs,
        renderers=_RENDERERS,
        substage_names=SUBSTAGES,
    )
