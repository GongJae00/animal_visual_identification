"""Parsing stage adapter and protocol receipt writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visualization.parsing import detection, segmentation
from visualization.parsing.assets import AssetLoader
from visualization.privacy import PublicationScope
from visualization.rendering.pipeline import (
    STAGE_LAYOUT,
    SubstageRenderer,
    render_stage,
    vis_directory,
)

STAGE = "parsing"
VIS_DIR, SUBSTAGES = STAGE_LAYOUT[STAGE]
_RENDERERS: dict[str, SubstageRenderer] = {
    "00_detection": detection.render,
    "01_segmentation": segmentation.render,
}


def render(
    trace: dict[str, Any],
    output_root: Path,
    *,
    asset_root: Path | None = None,
    scope: PublicationScope = PublicationScope.PRIVATE,
) -> tuple[Path, ...]:
    written = list(
        render_stage(
            STAGE,
            trace,
            output_root,
            asset_root=asset_root,
            scope=scope,
            renderers=_RENDERERS,
            render_context=AssetLoader(asset_root),
        )
    )
    protocol = trace.get("protocol")
    if protocol is None:
        return tuple(written)
    if not isinstance(protocol, dict):
        raise ValueError("trace protocol must be an object")
    if protocol.get("schema_version") != "evaluation.parsing_protocol.v1":
        return tuple(written)
    path = vis_directory(output_root, STAGE) / "protocol.json"
    path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written.append(path)
    return tuple(written)
