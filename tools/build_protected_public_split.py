"""Build a no-overwrite protected public canine split artifact set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.protected_io import read_strict_json_object, write_private_json_bundle
from cvi.protected_public_split import (
    FrozenPublicSplitEvidenceGraph,
    ProtectedPublicSplitPolicy,
    PublicSplitSourceBundle,
    build_protected_public_split,
    create_split_secret,
    read_split_secret,
    validate_protected_split_output_paths,
)
from cvi.source_provenance import build_offline_tool_provenance
from cvi.provenance import content_sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct protected score-blind public RGB roles only after the "
            "duplicate/review/dependency graph is frozen."
        )
    )
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--evidence-graph", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--secret", required=True, type=Path)
    parser.add_argument(
        "--create-secret",
        action="store_true",
        help="create --secret once with 32 random bytes and mode 0600",
    )
    parser.add_argument("--assignment-output", required=True, type=Path)
    parser.add_argument("--evaluator-binding-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()

    validate_protected_split_output_paths(
        (
            args.assignment_output,
            args.evaluator_binding_output,
            args.receipt_output,
        )
    )

    source_payload = read_strict_json_object(args.source_bundle)
    graph_payload = read_strict_json_object(args.evidence_graph)
    policy_payload = read_strict_json_object(args.policy)
    source = PublicSplitSourceBundle.from_dict(source_payload)
    graph = FrozenPublicSplitEvidenceGraph.from_dict(graph_payload)
    policy = ProtectedPublicSplitPolicy.from_dict(policy_payload)
    secret = (
        create_split_secret(args.secret)
        if args.create_secret
        else read_split_secret(args.secret)
    )
    input_hashes = tuple(sorted((
        ("evidence_graph_payload_sha256", content_sha256(graph_payload)),
        ("policy_payload_sha256", content_sha256(policy_payload)),
        ("source_bundle_payload_sha256", content_sha256(source_payload)),
    )))
    result = build_protected_public_split(
        source=source,
        graph=graph,
        policy=policy,
        secret=secret,
        input_file_sha256s=input_hashes,
        tool_provenance=build_offline_tool_provenance(Path(__file__)),
    )
    write_private_json_bundle((
        (args.assignment_output, result.assignment),
        (args.evaluator_binding_output, result.evaluator_binding),
        (args.receipt_output, result.receipt),
    ))
    print(json.dumps({
        "status": result.status,
        "assignment_sha256": result.receipt["assignment_sha256"],
        "evaluator_binding_sha256": result.receipt["evaluator_binding_sha256"],
        "receipt_sha256": result.receipt["receipt_sha256"],
        "seed_commitment": result.receipt["seed_commitment"],
        "interpretation": result.receipt["interpretation"],
    }, sort_keys=True))
    if not result.status.startswith("PASS"):
        raise SystemExit(2)
if __name__ == "__main__":
    main()
