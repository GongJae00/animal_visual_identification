"""Run resumable corpus-wide Meta PDQ fingerprinting and bounded MIH search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts.source_provenance import build_offline_tool_provenance
from foundation.protected_io import read_strict_json_object
from foundation.provenance import content_sha256
from identity_methods.classical.pdq_contracts import PDQSearchPolicy
from identity_methods.classical.public_canine_pdq_audit import (
    build_pdq_evidence_bundle,
    merge_pdq_fingerprint_chunks,
    prepare_pdq_audit_context,
    publish_pdq_evidence_bundle,
    publish_pdq_fingerprint_manifest,
    read_pdq_fingerprint_chunk,
    read_pdq_fingerprint_manifest,
    run_resumable_fingerprint_chunks,
)
from identity_methods.classical.public_canine_phash_audit import (
    read_public_canine_phash_policy,
    read_public_canine_phash_sources,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fingerprint = commands.add_parser("fingerprint")
    for command in (fingerprint,):
        command.add_argument("--source-spec", required=True, type=Path)
        command.add_argument("--decode-policy", required=True, type=Path)
        command.add_argument("--native-worker-directory", required=True, type=Path)
        command.add_argument("--official-regression-receipt", required=True, type=Path)
    fingerprint.add_argument("--output-directory", required=True, type=Path)
    fingerprint.add_argument("--chunk-size", required=True, type=int)
    fingerprint.add_argument("--maximum-new-chunks", type=int)

    merge = commands.add_parser("merge")
    for command in (merge,):
        command.add_argument("--source-spec", required=True, type=Path)
        command.add_argument("--decode-policy", required=True, type=Path)
        command.add_argument("--native-worker-directory", required=True, type=Path)
        command.add_argument("--official-regression-receipt", required=True, type=Path)
    merge.add_argument("--chunk-directory", required=True, type=Path)
    merge.add_argument("--output", required=True, type=Path)

    candidates = commands.add_parser("candidates")
    candidates.add_argument("--fingerprint-manifest", required=True, type=Path)
    candidates.add_argument("--search-policy", required=True, type=Path)
    candidates.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    provenance = build_offline_tool_provenance(Path(__file__))
    if args.command in {"fingerprint", "merge"}:
        decode_policy = read_public_canine_phash_policy(args.decode_policy)
        context = prepare_pdq_audit_context(
            sources=read_public_canine_phash_sources(args.source_spec),
            decode_policy=decode_policy,
            native_worker_directory=args.native_worker_directory,
            official_regression_receipt_path=args.official_regression_receipt,
        )
    if args.command == "fingerprint":
        created, reused = run_resumable_fingerprint_chunks(
            context=context,
            decode_policy=decode_policy,
            output_directory=args.output_directory,
            chunk_size=args.chunk_size,
            tool_provenance=provenance,
            maximum_new_chunks=args.maximum_new_chunks,
        )
        result = {
            "status": "FINGERPRINT_CHUNKS_RESUMABLE",
            "corpus_sample_count": len(context.items),
            "created_chunk_count": created,
            "reused_chunk_count": reused,
            "output_directory": str(args.output_directory),
        }
    elif args.command == "merge":
        paths = tuple(sorted(args.chunk_directory.glob("pdq-fingerprints-*.json")))
        manifest = merge_pdq_fingerprint_chunks(
            context=context,
            chunks=tuple(read_pdq_fingerprint_chunk(path) for path in paths),
        )
        bundle_sha256 = publish_pdq_fingerprint_manifest(
            args.output, manifest, tool_provenance=provenance
        )
        result = {
            "status": manifest["decision"],
            "fingerprint_count": manifest["fingerprint_count"],
            "chunk_count": manifest["chunk_count"],
            "manifest_sha256": content_sha256(manifest),
            "bundle_sha256": bundle_sha256,
            "output": str(args.output),
        }
    else:
        policy = PDQSearchPolicy.from_dict(read_strict_json_object(args.search_policy))
        bundle = build_pdq_evidence_bundle(
            fingerprint_manifest=read_pdq_fingerprint_manifest(
                args.fingerprint_manifest
            ),
            policy=policy,
        )
        artifact_sha256 = publish_pdq_evidence_bundle(args.output, bundle)
        search = bundle["evidence"]["search_result"]
        result = {
            "status": bundle["evidence"]["decision"],
            "fingerprint_count": bundle["evidence"]["fingerprint_count"],
            "eligible_sample_count": len(search["eligible_sample_ids"]),
            "low_quality_sample_count": len(search["ineligible_low_quality_sample_ids"]),
            "candidate_count": len(search["candidates"]),
            "preflight_raw_posting_visits": search["preflight_raw_posting_visits"],
            "unique_orientation_inspections": search["unique_orientation_inspections"],
            "artifact_sha256": artifact_sha256,
            "output": str(args.output),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
