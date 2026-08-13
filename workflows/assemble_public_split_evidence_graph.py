"""Promote a complete duplicate-adjudication ledger to a frozen graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts.source_provenance import build_offline_tool_provenance
from data_pipeline.public_duplicate_adjudication import (
    assemble_frozen_evidence_graph,
    read_adjudication_ledger,
    read_source_bundle,
)
from foundation.protected_io import write_private_json_bundle
from foundation.provenance import content_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--adjudication-ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()

    source = read_source_bundle(args.source_bundle)
    ledger = read_adjudication_ledger(args.adjudication_ledger)
    if ledger.promotion_status != "READY_FOR_GRAPH_PROMOTION":
        print(json.dumps({
            "status": "BLOCKED",
            "candidate_count": len(ledger.records),
            "outcome_counts": dict(ledger.outcome_counts),
            "global_blockers": list(ledger.global_blockers),
            "output_created": False,
        }, sort_keys=True))
        return 2
    graph = assemble_frozen_evidence_graph(source=source, ledger=ledger)
    provenance = build_offline_tool_provenance(Path(__file__))
    receipt = {
        "schema_version": "cvi.public_split_evidence_graph_assembly_receipt.v1",
        "source_bundle_sha256": source.bundle_sha256,
        "adjudication_ledger_sha256": ledger.ledger_sha256,
        "candidate_set_sha256": ledger.candidate_set_sha256,
        "candidate_count": len(ledger.records),
        "outcome_counts": dict(ledger.outcome_counts),
        "unresolved_candidate_count": 0,
        "unbound_candidate_count": ledger.unbound_candidate_count,
        "adjudication_mode": ledger.mode.value,
        "graph_sha256": graph.graph_sha256,
        "edge_count": len(graph.edges),
        "tool_provenance": provenance,
        "tool_provenance_sha256": content_sha256(provenance),
        "decision": "PASS_COMPLETE_ADJUDICATION_GRAPH_PROMOTION",
    }
    receipt["receipt_sha256"] = content_sha256(receipt)
    write_private_json_bundle((
        (args.output, graph.to_dict()),
        (args.receipt_output, receipt),
    ))
    print(json.dumps({
        "status": receipt["decision"],
        "graph_sha256": graph.graph_sha256,
        "edge_count": len(graph.edges),
        "candidate_count": len(ledger.records),
        "adjudication_mode": ledger.mode.value,
        "receipt_sha256": receipt["receipt_sha256"],
        "output": str(args.output),
        "receipt_output": str(args.receipt_output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
