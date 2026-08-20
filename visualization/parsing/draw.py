"""Parsing observer. Writes Visualization/vis/00_parsing/.

JSON traces only unless a source image is supplied. No titles, captions,
flow diagrams, or catalog plates.
"""

from pathlib import Path
from typing import Any

from visualization.rendering.pipeline import STAGE_LAYOUT, render_stage

STAGE = "parsing"
VIS_DIR, SUBSTAGES = STAGE_LAYOUT[STAGE]


def render(trace: dict[str, Any], output_root: Path, **kwargs: Any) -> tuple[Path, ...]:
    written = list(render_stage(STAGE, trace, output_root, **kwargs))
    protocol = trace.get("protocol")
    if protocol is None:
        return tuple(written)
    if not isinstance(protocol, dict):
        raise ValueError("trace protocol must be an object")
    if protocol.get("schema_version") != "evaluation.parsing_protocol.v1":
        return tuple(written)
    import json

    path = output_root / "vis" / VIS_DIR / "protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(path)
    return tuple(written)
