"""Bind a protected split assignment to the identity registry.

Joins identity_token-based assignment with registered_dog_id, validates
coverage, and writes a binding manifest for downstream training/eval.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cvi.identity_registry import load_registry_manifest
from cvi.protected_io import read_strict_json_object
from cvi.split_registry_binding import build_binding
from cvi.source_provenance import build_offline_tool_provenance
from cvi.provenance import content_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--registry-db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    t0 = time.time()

    assignment = read_strict_json_object(args.assignment)
    registry = load_registry_manifest(args.registry_db)

    binding = build_binding(assignment, args.registry_db)

    manifest = binding.to_dict()
    manifest["assignment_sha256"] = assignment.get(
        "assignment_sha256",
        assignment.get("receipt_sha256", ""),
    )
    manifest["registry_sha256"] = content_sha256(registry.to_dict())
    manifest["tool_provenance"] = build_offline_tool_provenance(Path(__file__))
    manifest["manifest_sha256"] = content_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )

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
