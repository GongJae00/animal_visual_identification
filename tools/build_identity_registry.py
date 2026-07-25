"""Build the deterministic identity registry from a public-dataset source bundle.

Reads a PublicSplitSourceBundle, computes registered_dog_id (UUIDv5) for
each unique dataset_identity_id, and writes both an SQLite database and
a JSON manifest for downstream training and evaluation pipelines.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cvi.identity_registry import (
    IdentityRegistryRecord,
    create_registry_database,
    load_registry_manifest,
    register_records,
)
from cvi.protected_public_split import PublicSplitSourceBundle
from cvi.protected_io import read_strict_json_object
from cvi.source_provenance import build_offline_tool_provenance
from cvi.provenance import content_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--db-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()

    t0 = time.time()

    source_payload = read_strict_json_object(args.source_bundle)
    source = PublicSplitSourceBundle.from_dict(source_payload)
    print(
        json.dumps(
            {
                "event": "loaded_source_bundle",
                "sample_count": len(source.samples),
                "source_bundle_sha256": source.bundle_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    unique_identities: dict[str, str] = {}
    for sample in source.samples:
        did = sample.dataset_identity_id
        if did not in unique_identities:
            unique_identities[did] = sample.identity_token
    print(
        json.dumps(
            {
                "event": "extracted_identities",
                "unique_identity_count": len(unique_identities),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    create_registry_database(args.db_output)
    mapping = register_records(args.db_output, list(unique_identities.keys()))
    print(
        json.dumps(
            {
                "event": "registered_identities",
                "registration_count": len(mapping),
                "db_output": str(args.db_output),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    registry = load_registry_manifest(args.db_output)
    manifest = registry.to_dict()
    manifest["source_bundle_sha256"] = source.bundle_sha256
    manifest["schema_version"] = "cvi.identity_registry_manifest.v1"
    manifest["tool_provenance"] = build_offline_tool_provenance(Path(__file__))
    manifest["manifest_sha256"] = content_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    elapsed = time.time() - t0
    print(
        json.dumps(
            {
                "status": "DONE",
                "elapsed_seconds": round(elapsed, 2),
                "identity_count": len(mapping),
                "db_output": str(args.db_output),
                "manifest_output": str(args.manifest_output),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
