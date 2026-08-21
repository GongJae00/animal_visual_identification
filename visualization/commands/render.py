"""Render pipeline observer plates or paper figure bundles.

Run: ``uv run python -m visualization.commands.render --help``

``--stage`` writes gitignored ``Visualization/vis/00_parsing`` … ``05_search``.
Paper ``FIGURE_REGISTRY`` 00–17 is a different sequence; ``--paper`` writes
``Visualization/paper/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from visualization.privacy import PublicationScope
from visualization.rendering.pipeline import STAGE_LAYOUT

_CALLERS = {
    "parsing": "visualization.parsing",
    "representation": "visualization.representation",
    "enrollment": "visualization.enrollment",
    "gallery": "visualization.gallery",
    "search": "visualization.search",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m visualization.commands.render",
        description=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--stage",
        choices=tuple(STAGE_LAYOUT),
        help="Pipeline observer stage. Writes vis/0N_<stage>/ inner substages.",
    )
    mode.add_argument(
        "--paper",
        action="store_true",
        help="Publish a figure-data bundle using FIGURE_REGISTRY 00-17.",
    )
    parser.add_argument("--trace", type=Path, help="JSON trace for --stage")
    parser.add_argument("--input", type=Path, help="Figure-data bundle for --paper")
    parser.add_argument(
        "--output",
        type=Path,
        help="Root for --stage (default Visualization) or paper directory.",
    )
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing pipeline visualization tree before rendering.",
    )
    parser.add_argument(
        "--scope",
        choices=[item.value for item in PublicationScope],
        default=PublicationScope.PRIVATE.value,
    )
    args = parser.parse_args(argv)
    if args.stage is not None:
        return _render_stage(args)
    return _render_paper(args)


def _render_stage(args: argparse.Namespace) -> int:
    if args.trace is None:
        raise SystemExit("--trace is required with --stage")
    output = args.output or Path("Visualization")
    if args.clean:
        from visualization.rendering.pipeline import clear_visualizations

        clear_visualizations(output)
    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("trace must be a JSON object")
    import importlib

    module = importlib.import_module(_CALLERS[args.stage])
    written = module.render(
        payload,
        output,
        asset_root=args.asset_root,
        scope=PublicationScope(args.scope),
    )
    print(
        json.dumps(
            {
                "event": "pipeline_visualization_rendered",
                "stage": args.stage,
                "directory": str(
                    (output / "vis" / STAGE_LAYOUT[args.stage][0]).resolve()
                ),
                "files": [str(path) for path in written],
            },
            sort_keys=True,
        )
    )
    return 0


def _render_paper(args: argparse.Namespace) -> int:
    if args.clean:
        raise SystemExit("--clean is only valid with --stage")
    if args.input is None:
        raise SystemExit("--input is required with --paper")
    from visualization.contracts import FigureData
    from visualization.publishing.publication import publish

    output = args.output or Path("Visualization") / "paper"
    bundle = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict):
        raise ValueError("paper input must be a figure-data bundle object")
    figures = (FigureData.from_bundle(bundle),)
    target_scope = PublicationScope(args.scope)
    if target_scope is PublicationScope.PRIVATE:
        target_scope = PublicationScope.PAPER
    receipt = publish(
        figures,
        output,
        target_scope=target_scope,
        asset_root=args.asset_root,
    )
    print(
        json.dumps(
            {"event": "research_visualizations_rendered", **receipt},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
