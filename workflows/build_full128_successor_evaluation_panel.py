"""Bind the governance-v2 gallery panel to materialized successor availability."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evaluation.full_segment.full128_successors import build_authoritative_fixed_evaluation_panel
from foundation.protected_io import read_strict_json_document, write_private_json_bundle

_LIMITS = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successor-inventory", required=True, type=Path)
    parser.add_argument("--face-protocol-v2", required=True, type=Path)
    parser.add_argument("--gallery-query-panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite successor evaluation panel")
    read = lambda path: read_strict_json_document(path, **_LIMITS).payload
    panel = build_authoritative_fixed_evaluation_panel(
        read(args.successor_inventory),
        read(args.face_protocol_v2),
        read(args.gallery_query_panel),
    )
    write_private_json_bundle(((args.output, panel),))
    available = sum(row["status"] == "AVAILABLE" for row in panel["cohorts"])
    print(
        json.dumps(
            {
                "status": "CREATED_FULL128_SUCCESSOR_EVALUATION_PANEL",
                "panel_sha256": panel["panel_sha256"],
                "available_cohort_count": available,
                "unavailable_cohort_count": len(panel["cohorts"]) - available,
                "required_sample_count": len(panel["required_sample_tokens"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
