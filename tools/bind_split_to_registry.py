"""Bind a protected split assignment to the identity registry.

Joins identity_token-based assignment with registered_dog_id, validates
coverage, and writes a binding manifest for downstream training/eval.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cvi.protected_io import read_strict_json_object, write_private_json_bundle
from cvi.split_registry_binding import build_binding
from cvi.source_provenance import build_offline_tool_provenance
from cvi.provenance import content_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--split-receipt", required=True, type=Path)
    parser.add_argument("--registry-db", required=True, type=Path)
    parser.add_argument("--registry-manifest", required=True, type=Path)
    parser.add_argument("--expected-split-receipt-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    t0 = time.time()

    assignment = read_strict_json_object(args.assignment)
    split_receipt = read_strict_json_object(args.split_receipt)
    registry_manifest = read_strict_json_object(args.registry_manifest)

    binding = build_binding(
        assignment,
        args.registry_db,
        split_receipt,
        registry_manifest,
        args.expected_split_receipt_sha256,
    )

    manifest = binding.to_dict()
    manifest["assignment_sha256"] = split_receipt["assignment_sha256"]
    manifest["split_receipt_sha256"] = split_receipt["receipt_sha256"]
    manifest["tool_provenance"] = build_offline_tool_provenance(Path(__file__))
    manifest["manifest_sha256"] = content_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_private_json_bundle(((args.output, manifest),))

    elapsed = time.time() - t0
    status = "VALID" if binding.is_valid else "INVALID"
    print(
        json.dumps(
            {
                "status": status,
                "elapsed_seconds": round(elapsed, 2),
                "total_identities": binding.total_identities,
                "total_samples": binding.total_samples,
                "unregistered_count": len(binding.unregistered_tokens),
                "output": str(args.output),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if not binding.is_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
