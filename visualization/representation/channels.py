"""Channel diagnostics for ``Visualization/vis/01_representation/01_channels``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def render(
    payload: Mapping[str, Any], target: Path, _context: Any
) -> tuple[tuple[Path, ...], Mapping[str, Any]]:
    from visualization.rendering.embeddings import render_embedding_views

    files, record = render_embedding_views(payload, target)
    return tuple(files), record
