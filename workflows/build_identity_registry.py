"""Build the deterministic identity registry from a public-dataset source bundle.

Reads a PublicSplitSourceBundle, computes registered_dog_id (UUIDv5) for
each unique dataset_identity_id, and writes both an SQLite database and
a JSON manifest for downstream training and evaluation pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from contracts.source_provenance import build_offline_tool_provenance
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from foundation.provenance import content_sha256
from identity.registry.identity_registry import (
    IdentityRegistryRecord,
    create_registry_database,
    load_registry_manifest,
    register_records,
)
from identity.splits.protected_public_split import PublicSplitSourceBundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--db-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()

    args.db_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    final_db = args.db_output.absolute()
    final_manifest = args.manifest_output.absolute()
    outputs = (final_db, final_manifest)
    if final_db == final_manifest:
        parser.error("registry database and manifest outputs must be distinct")
    existing = [path for path in outputs if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "refusing to extend or overwrite registry outputs: "
            + ", ".join(str(path) for path in existing)
        )

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

    with TemporaryDirectory(prefix=".cvi-registry-", dir=final_db.parent) as temporary:
        working_db = Path(temporary) / "identity_registry.db"
        create_registry_database(working_db)
        mapping = register_records(working_db, list(unique_identities.keys()))
        print(
            json.dumps(
                {
                    "event": "registered_identities",
                    "registration_count": len(mapping),
                    "db_output": str(final_db),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        registry = load_registry_manifest(working_db)
        expected_identity_ids = set(unique_identities)
        observed_identity_ids = {
            record.dataset_identity_id for record in registry.records
        }
        if observed_identity_ids != expected_identity_ids or any(
            record.image_count != 1 for record in registry.records
        ):
            raise RuntimeError(
                "registry database differs from source-bundle identities"
            )
        manifest = registry.to_dict()
        manifest["source_bundle_sha256"] = source.bundle_sha256
        manifest["schema_version"] = "cvi.identity_registry_manifest.v1"
        manifest["tool_provenance"] = build_offline_tool_provenance(Path(__file__))
        manifest["manifest_sha256"] = content_sha256(
            {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        )
        published_db = False
        try:
            os.link(working_db, final_db)
            published_db = True
            write_private_json_bundle(((final_manifest, manifest),))
        except BaseException:
            if published_db:
                final_db.unlink(missing_ok=True)
            raise
    elapsed = time.time() - t0
    print(
        json.dumps(
            {
                "status": "DONE",
                "elapsed_seconds": round(elapsed, 2),
                "identity_count": len(mapping),
                "db_output": str(final_db),
                "manifest_output": str(final_manifest),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
