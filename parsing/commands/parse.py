"""Parsing CLI.

Run: ``uv run python -m parsing.commands.parse --help``
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m parsing.commands.parse",
        description="Parsing runtime commands.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("materialize", help="Materialize one Full-segment record")
    sub.add_parser("manifest", help="Bind parser models, policy, and reports")
    sub.add_parser("panel", help="Run an unassisted parsing panel")
    sub.add_parser("compare", help="Compare parser materializations")
    sub.add_parser(
        "three-region",
        help="Export A/F/N evidence from one ROI bundle",
    )
    sub.add_parser("benchmark-batches", help="Benchmark parser batches")
    sub.add_parser("benchmark-localizers", help="Benchmark dog detectors")
    sub.add_parser("unified-manifest", help="Build a unified canid JSONL manifest")
    sub.add_parser("oracle-crops", help="Export token-only oracle crops")
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command, rest = argv[0], argv[1:]
    if command == "materialize":
        from parsing.export.crops.sample_materialization import (
            _read_json_object,
            run,
        )

        materialize = argparse.ArgumentParser(prog="parse materialize")
        materialize.add_argument("--request", required=True, type=Path)
        materialize.add_argument("--output-dir", required=True, type=Path)
        args = materialize.parse_args(rest)
        request = _read_json_object(
            args.request.resolve(), label="Full segment request"
        )
        run(request, output_dir=args.output_dir.resolve())
        return 0
    if command == "manifest":
        from parsing.export.manifest import main as run_manifest

        return run_manifest(rest)
    if command == "panel":
        from parsing.export.panel import main as run_panel

        return run_panel(rest)
    if command == "compare":
        from parsing.export.compare import main as run_compare

        return run_compare(rest)
    if command in {"three-region", "three_region"}:
        from parsing.export.regions.three_region_export import main as run_export

        return run_export(rest)
    absorbed = {
        "benchmark-batches": "archive.full128.evaluation.parsing_batch_benchmark",
        "benchmark-localizers": "evaluation.localization_benchmark",
        "unified-manifest": "data.unified_manifest",
        "oracle-crops": "evaluation.controls.oracle_crop_export",
    }
    if command in absorbed:
        import importlib

        module = importlib.import_module(absorbed[command])
        previous = sys.argv
        sys.argv = [previous[0], *rest]
        try:
            result = module.main(rest) if command == "unified-manifest" else module.main()
        finally:
            sys.argv = previous
        return 0 if result is None else int(result)
    parser.error(f"unknown command {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
