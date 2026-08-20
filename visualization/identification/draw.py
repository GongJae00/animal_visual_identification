"""Identification observer. Writes Visualization/vis/01_identification/.

Imports identification.export only, never identification.training.
Caption-free PCA and cosine views of channel embeddings.
"""

from pathlib import Path
from typing import Any

from visualization.rendering.embeddings import render_vector_stage
from visualization.rendering.pipeline import STAGE_LAYOUT

STAGE = "identification"
VIS_DIR, SUBSTAGES = STAGE_LAYOUT[STAGE]


def render(trace: dict[str, Any], output_root: Path, **kwargs: Any) -> tuple[Path, ...]:
    return render_vector_stage(
        STAGE,
        trace,
        output_root,
        substages=SUBSTAGES,
        vis_dir=VIS_DIR,
        **kwargs,
    )
