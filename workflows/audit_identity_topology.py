"""Audit normalized embedding-space identity topology from a bound manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from artifact_contracts.source_provenance import build_source_provenance
from experiments.identity_topology import (
    IdentityTopologyConfig,
    audit_identity_topology,
)
from foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from foundation.provenance import content_sha256


REPORT_BUNDLE_SCHEMA_VERSION = "cvi.embedding_identity_topology_audit_bundle.v1"
_DEFAULT_CONFIG = IdentityTopologyConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--connectivity-threshold",
        type=float,
        default=_DEFAULT_CONFIG.connectivity_cosine_distance_threshold,
        help="maximum cosine distance for an edge between session prototypes",
    )
    parser.add_argument(
        "--normalization-tolerance",
        type=float,
        default=_DEFAULT_CONFIG.normalization_tolerance,
    )
    parser.add_argument(
        "--minimum-prototype-norm",
        type=float,
        default=_DEFAULT_CONFIG.minimum_prototype_norm,
    )
    parser.add_argument(
        "--hubness-k", type=int, default=_DEFAULT_CONFIG.hubness_k
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    output = Path(os.path.abspath(os.fspath(args.output)))
    output = output.parent.resolve(strict=True) / output.name
    if output.is_relative_to(repository):
        raise ValueError("topology audit must be written outside the Git repository")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite topology audit: {output}")
    document = read_strict_json_document(
        args.manifest,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    config = IdentityTopologyConfig(
        connectivity_cosine_distance_threshold=args.connectivity_threshold,
        normalization_tolerance=args.normalization_tolerance,
        minimum_prototype_norm=args.minimum_prototype_norm,
        hubness_k=args.hubness_k,
    )
    source_provenance = build_source_provenance(
        (repository / "workflows/audit_identity_topology.py",)
    )
    code_sha256s = {
        row["relative_path"]: row["content_sha256"]
        for row in source_provenance["code_source_files"]
    }
    report = audit_identity_topology(
        document.payload,
        config=config,
        code_sha256s=code_sha256s,
    )
    report["provenance"]["input_file_sha256"] = document.raw_sha256
    report["provenance"]["input_byte_size"] = document.byte_size
    bundle = {
        "schema_version": REPORT_BUNDLE_SCHEMA_VERSION,
        "report_sha256": content_sha256(report),
        "report": report,
    }
    write_private_json_bundle(((output, bundle),))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": os.fspath(output),
                "report_sha256": bundle["report_sha256"],
                "branches": report["population"]["branch_count"],
                "identities": report["population"]["identity_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
