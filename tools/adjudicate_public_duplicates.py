"""Build exact evidence and resumable public duplicate adjudication ledgers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cvi.protected_io import read_strict_json_object
from cvi.provenance import content_sha256
from cvi.public_duplicate_adjudication import (
    AdjudicationMode,
    build_adjudication_chunk,
    build_duplicate_evidence_source_generation,
    build_exact_duplicate_graph,
    build_review_queue,
    merge_adjudication_chunks,
    publish_adjudication_chunk,
    publish_adjudication_ledger,
    publish_exact_graph,
    publish_review_queue,
    publish_source_generation,
    read_adjudication_chunk,
    read_adjudication_ledger,
    read_exact_graph,
    read_geometric_chunks,
    read_source_bundle,
)
from cvi.source_provenance import build_offline_tool_provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    exact = commands.add_parser("exact", help="derive authenticated exact-pixel pairs")
    exact.add_argument("--source-bundle", required=True, type=Path)
    exact.add_argument("--image-content-receipts", required=True, type=Path)
    exact.add_argument("--opaque-binding", required=True, type=Path)
    exact.add_argument("--output", required=True, type=Path)

    chunk = commands.add_parser("chunk", help="adjudicate one deterministic candidate range")
    chunk.add_argument("--source-bundle", required=True, type=Path)
    chunk.add_argument("--exact-graph", required=True, type=Path)
    chunk.add_argument("--phash-evidence", required=True, type=Path)
    chunk.add_argument("--opaque-binding", required=True, type=Path)
    chunk.add_argument("--pdq-evidence", type=Path)
    chunk.add_argument("--pdq-transform-admission", type=Path)
    chunk.add_argument("--dinov2-filter-evidence", type=Path)
    chunk.add_argument("--geometric-evidence", type=Path, action="append", default=[])
    chunk.add_argument("--geometric-admission-receipt", type=Path)
    chunk.add_argument("--review-adjudication", type=Path)
    chunk.add_argument("--start-index", required=True, type=int)
    chunk.add_argument("--maximum-candidates", required=True, type=int)
    chunk.add_argument("--output", required=True, type=Path)
    chunk.add_argument(
        "--mode",
        choices=[item.value for item in AdjudicationMode],
        default=AdjudicationMode.STANDARD.value,
    )

    generation = commands.add_parser(
        "source-generation",
        help="publish a new no-overwrite source bundle with core evidence bindings",
    )
    generation.add_argument("--source-bundle", required=True, type=Path)
    generation.add_argument("--exact-graph", required=True, type=Path)
    generation.add_argument("--phash-evidence", required=True, type=Path)
    generation.add_argument("--pdq-evidence", required=True, type=Path)
    generation.add_argument("--opaque-binding", required=True, type=Path)
    generation.add_argument("--output", required=True, type=Path)

    merge = commands.add_parser("merge", help="merge a complete contiguous chunk partition")
    merge.add_argument("--chunk", required=True, type=Path, action="append")
    merge.add_argument("--output", required=True, type=Path)

    review = commands.add_parser(
        "review-queue", help="export unresolved pairs without assigning outcomes"
    )
    review.add_argument("--source-bundle", required=True, type=Path)
    review.add_argument("--adjudication-ledger", required=True, type=Path)
    review.add_argument("--image-content-receipts", required=True, type=Path)
    review.add_argument("--phash-evidence", required=True, type=Path)
    review.add_argument("--opaque-binding", required=True, type=Path)
    review.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    provenance = build_offline_tool_provenance(Path(__file__))
    if args.command == "exact":
        graph = build_exact_duplicate_graph(
            source=read_source_bundle(args.source_bundle),
            image_receipts=read_strict_json_object(args.image_content_receipts),
            opaque_binding_bundle=read_strict_json_object(args.opaque_binding),
        )
        bundle_sha256 = publish_exact_graph(
            args.output, graph, tool_provenance=provenance
        )
        result = {
            "status": "PASS_AUTHENTICATED_PIXEL_EXACT_GRAPH",
            "sample_count": graph.sample_count,
            "exact_pair_count": len(graph.pairs),
            "graph_sha256": graph.graph_sha256,
            "bundle_sha256": bundle_sha256,
            "output": str(args.output),
        }
    elif args.command == "source-generation":
        exact_bundle = read_strict_json_object(args.exact_graph)
        source = build_duplicate_evidence_source_generation(
            source=read_source_bundle(args.source_bundle),
            exact_graph=read_exact_graph(args.exact_graph),
            exact_graph_bundle=exact_bundle,
            phash_evidence_bundle=read_strict_json_object(args.phash_evidence),
            pdq_evidence_bundle=read_strict_json_object(args.pdq_evidence),
            opaque_binding_bundle=read_strict_json_object(args.opaque_binding),
        )
        bundle_sha256 = publish_source_generation(args.output, source)
        result = {
            "status": "PASS_IMMUTABLE_CORE_DUPLICATE_EVIDENCE_SOURCE_GENERATION",
            "sample_count": len(source.samples),
            "source_bundle_sha256": bundle_sha256,
            "evidence_bindings": dict(source.evidence_bindings),
            "output": str(args.output),
        }
    elif args.command == "chunk":
        chunk = build_adjudication_chunk(
            source=read_source_bundle(args.source_bundle),
            exact_graph=read_exact_graph(args.exact_graph),
            exact_graph_artifact_sha256=content_sha256(
                read_strict_json_object(args.exact_graph)
            ),
            phash_evidence_bundle=read_strict_json_object(args.phash_evidence),
            opaque_binding_bundle=read_strict_json_object(args.opaque_binding),
            pdq_evidence_bundle=(
                None
                if args.pdq_evidence is None
                else read_strict_json_object(args.pdq_evidence)
            ),
            pdq_transform_admission=(
                None
                if args.pdq_transform_admission is None
                else read_strict_json_object(args.pdq_transform_admission)
            ),
            dinov2_filter_evidence=(
                None
                if args.dinov2_filter_evidence is None
                else read_strict_json_object(args.dinov2_filter_evidence)
            ),
            geometric_evidence=read_geometric_chunks(args.geometric_evidence),
            geometric_admission_receipt=(
                None
                if args.geometric_admission_receipt is None
                else read_strict_json_object(args.geometric_admission_receipt)
            ),
            review_bundle=(
                None
                if args.review_adjudication is None
                else read_strict_json_object(args.review_adjudication)
            ),
            mode=AdjudicationMode(args.mode),
            start_index=args.start_index,
            maximum_candidates=args.maximum_candidates,
        )
        bundle_sha256 = publish_adjudication_chunk(
            args.output, chunk, tool_provenance=provenance
        )
        result = {
            "status": "CHUNK_CREATED",
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "total_candidate_count": chunk.total_candidate_count,
            "global_blockers": list(chunk.global_blockers),
            "mode": chunk.mode.value,
            "unbound_candidate_count": chunk.unbound_candidate_count,
            "outcome_counts": dict(sorted(
                Counter(item.outcome.value for item in chunk.records).items()
            )),
            "chunk_sha256": chunk.chunk_sha256,
            "bundle_sha256": bundle_sha256,
            "output": str(args.output),
        }
    elif args.command == "merge":
        ledger = merge_adjudication_chunks(tuple(
            read_adjudication_chunk(path) for path in args.chunk
        ))
        bundle_sha256 = publish_adjudication_ledger(
            args.output, ledger, tool_provenance=provenance
        )
        result = {
            "status": ledger.promotion_status,
            "candidate_count": len(ledger.records),
            "outcome_counts": dict(ledger.outcome_counts),
            "global_blockers": list(ledger.global_blockers),
            "mode": ledger.mode.value,
            "unbound_candidate_count": ledger.unbound_candidate_count,
            "ledger_sha256": ledger.ledger_sha256,
            "bundle_sha256": bundle_sha256,
            "output": str(args.output),
        }
    else:
        queue = build_review_queue(
            source=read_source_bundle(args.source_bundle),
            ledger=read_adjudication_ledger(args.adjudication_ledger),
            image_receipts=read_strict_json_object(args.image_content_receipts),
            phash_evidence_bundle=read_strict_json_object(args.phash_evidence),
            opaque_binding_bundle=read_strict_json_object(args.opaque_binding),
        )
        bundle_sha256 = publish_review_queue(
            args.output, queue, tool_provenance=provenance
        )
        result = {
            "status": queue["decision"],
            "record_count": queue["record_count"],
            "queue_sha256": content_sha256(queue),
            "bundle_sha256": bundle_sha256,
            "output": str(args.output),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
