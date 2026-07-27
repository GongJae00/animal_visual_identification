"""Analyze frozen duplicate-component quota capacity without allocating a split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.duplicate_graph_capacity import analyze_duplicate_graph_capacity
from cvi.protected_io import read_strict_json_object, write_private_json_bundle
from cvi.protected_public_split import (
    FrozenPublicSplitEvidenceGraph,
    ProtectedPublicSplitPolicy,
    PublicSplitSourceBundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--evidence-graph", required=True, type=Path)
    parser.add_argument("--split-policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_duplicate_graph_capacity(
        source=PublicSplitSourceBundle.from_dict(
            read_strict_json_object(args.source_bundle)
        ),
        graph=FrozenPublicSplitEvidenceGraph.from_dict(
            read_strict_json_object(args.evidence_graph)
        ),
        policy=ProtectedPublicSplitPolicy.from_dict(
            read_strict_json_object(args.split_policy)
        ),
    )
    write_private_json_bundle(((args.output, report),))
    print(json.dumps({
        "status": report["status"],
        "largest_allocation_block_identity_count": report[
            "largest_allocation_block_identity_count"
        ],
        "quarantined_identity_count": report["quarantined_identity_count"],
        "failed_quota_lanes": report["failed_quota_lanes"],
        "report_sha256": report["report_sha256"],
        "output": str(args.output),
    }, sort_keys=True))
    return 2 if report["failed_quota_lanes"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
