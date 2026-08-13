"""Render content-bound research figures into an atomic static publication."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from foundation.protected_io import read_strict_json_document, read_strict_json_object
from vis.adapters import (
    adapt_common_evaluation_report,
    adapt_master_results_table,
    adapt_protected_evaluation_v3,
)
from vis.contracts import FigureData
from vis.privacy import PublicationScope
from vis.publication import publish

_LARGE_JSON = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}
from vis.successor_family import adapt_successor_family


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--scope",
        choices=tuple(scope.value for scope in PublicationScope),
        default=PublicationScope.PUBLIC.value,
    )
    parser.add_argument(
        "--adapter",
        choices=(
            "figure-data",
            "master-results",
            "common-report",
            "protected-report-v3",
            "successor-family",
        ),
        default="figure-data",
    )
    parser.add_argument("--figure-id", action="append")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--public-report", type=Path)
    parser.add_argument("--private-report", type=Path)
    parser.add_argument("--face-protocol-v2", type=Path)
    parser.add_argument("--gallery-query-panel", type=Path)
    parser.add_argument("--successor-inventory", type=Path)
    parser.add_argument("--cache-descriptor", action="append", type=Path)
    args = parser.parse_args(argv)

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite publication: {args.output}")
    scope = PublicationScope(args.scope)
    if args.adapter == "successor-family":
        if args.input:
            parser.error(
                "successor-family mode uses named artifact options, not --input"
            )
        required = {
            "--public-report": args.public_report,
            "--face-protocol-v2": args.face_protocol_v2,
            "--gallery-query-panel": args.gallery_query_panel,
            "--successor-inventory": args.successor_inventory,
        }
        missing = [name for name, path in required.items() if path is None]
        if missing:
            parser.error("successor-family mode requires " + ", ".join(missing))
        figures = adapt_successor_family(
            read_strict_json_document(args.public_report, **_LARGE_JSON).payload,
            read_strict_json_document(args.face_protocol_v2, **_LARGE_JSON).payload,
            read_strict_json_document(args.gallery_query_panel, **_LARGE_JSON).payload,
            read_strict_json_document(args.successor_inventory, **_LARGE_JSON).payload,
            target_scope=scope,
            private_report=(
                None
                if args.private_report is None
                else read_strict_json_document(
                    args.private_report, **_LARGE_JSON
                ).payload
            ),
            cache_descriptors=tuple(
                read_strict_json_object(path) for path in (args.cache_descriptor or ())
            ),
            asset_root=args.asset_root,
        )
    else:
        if not args.input:
            parser.error(f"{args.adapter} mode requires at least one --input")
        if (
            any(
                path is not None
                for path in (
                    args.public_report,
                    args.private_report,
                    args.face_protocol_v2,
                    args.gallery_query_panel,
                    args.successor_inventory,
                )
            )
            or args.cache_descriptor
        ):
            parser.error("named successor artifacts require --adapter successor-family")
        documents = [read_strict_json_object(path) for path in args.input]
    if args.adapter == "figure-data":
        figures = tuple(FigureData.from_bundle(document) for document in documents)
    elif args.adapter != "successor-family":
        if len(documents) != 1:
            parser.error("schema-specific adapters accept exactly one --input")
        figure_id = (
            args.figure_id[0] if args.figure_id else "13_primary_results_paired_deltas"
        )
        if args.figure_id and len(args.figure_id) != 1:
            parser.error("schema-specific adapters accept one --figure-id")
        adapter = {
            "master-results": adapt_master_results_table,
            "common-report": adapt_common_evaluation_report,
            "protected-report-v3": adapt_protected_evaluation_v3,
        }[args.adapter]
        figures = (adapter(documents[0], figure_id=figure_id, scope=scope),)
    receipt = publish(
        figures,
        args.output,
        target_scope=scope,
        asset_root=args.asset_root,
        figure_ids=tuple(args.figure_id) if args.figure_id else None,
    )
    print(
        json.dumps(
            {"event": "research_visualizations_rendered", **receipt}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
