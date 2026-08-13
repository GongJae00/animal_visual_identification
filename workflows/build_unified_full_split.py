"""Allocate and census one observation-complete unified Full split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from identity_governance.full_split_census import (
    FullSplitAllocationPolicy,
    UnifiedFullObservation,
    allocate_unified_full_split,
    build_unified_full_census,
    unified_full_split_bundle,
)

REQUEST_SCHEMA = "cvi.unified_full_split_request.v1"
_LIMITS = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}


def run(payload: object) -> dict[str, object]:
    expected = {"schema_version", "allocation_name", "policy", "observations"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("unified Full split request fields differ")
    if payload["schema_version"] != REQUEST_SCHEMA:
        raise ValueError("unified Full split request schema differs")
    if not isinstance(payload["observations"], list):
        raise TypeError("unified Full split request observations must be an array")
    policy = FullSplitAllocationPolicy.from_dict(payload["policy"])
    observations = tuple(
        UnifiedFullObservation.from_dict(item) for item in payload["observations"]
    )
    manifest = allocate_unified_full_split(
        allocation_name=payload["allocation_name"],
        observations=observations,
        policy=policy,
    )
    census = build_unified_full_census(manifest)
    return unified_full_split_bundle(manifest, census)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Full split output: {args.output}")
    request = read_strict_json_document(args.request, **_LIMITS).payload
    bundle = run(request)
    write_private_json_bundle(((args.output, bundle),))
    print(
        json.dumps(
            {
                "status": "CREATED_UNIFIED_FULL_SPLIT",
                "manifest_sha256": bundle["manifest_sha256"],
                "census_sha256": bundle["census_sha256"],
                "observation_count": bundle["census"]["observation_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
