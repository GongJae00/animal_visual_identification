"""Run the protected four-corpus label-blind pHash candidate audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.public_canine_phash_audit import (
    read_public_canine_phash_policy,
    read_public_canine_phash_sources,
    run_public_canine_phash_audit,
)
from cvi.source_provenance import build_offline_tool_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-spec", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--evidence-output", required=True, type=Path)
    parser.add_argument("--binding-output", required=True, type=Path)
    args = parser.parse_args()

    policy = read_public_canine_phash_policy(args.policy)
    sources = read_public_canine_phash_sources(
        args.source_spec, maximum_bytes=policy.maximum_source_spec_bytes
    )
    evidence_sha256, binding_sha256 = run_public_canine_phash_audit(
        sources=sources,
        policy=policy,
        evidence_output=args.evidence_output,
        binding_output=args.binding_output,
        tool_provenance=build_offline_tool_provenance(Path(__file__)),
    )
    print(json.dumps({
        "status": "PASS_LABEL_BLIND_PHASH_CANDIDATE_GENERATION",
        "evidence_sha256": evidence_sha256,
        "binding_sha256": binding_sha256,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
