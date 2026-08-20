"""Registry build/bind and split-manifest check.

Run through ``evaluation.commands.evaluate``:
``registry-build``, ``registry-bind``, ``split-check``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from enrollment.registry.identity_registry import (
    create_registry_database,
    load_registry_manifest,
    register_records,
)
from evaluation.splits.leakage import association_audit
from evaluation.splits.protected_public_split import PublicSplitSourceBundle
from evaluation.splits.split_registry_binding import build_binding
from evaluation.splits.tracklet_split import SplitManifest
from shared.contracts.source_provenance import build_offline_tool_provenance
from shared.foundation.protected_io import read_strict_json_object, write_private_json_bundle
from shared.foundation.provenance import content_sha256


def _run_build(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Build the identity registry from a public-dataset source bundle."
    )
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--db-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args(argv)

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

    unique_identities: dict[str, None] = {}
    for sample in source.samples:
        unique_identities.setdefault(sample.dataset_identity_id)
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

    with TemporaryDirectory(prefix=".registry-", dir=final_db.parent) as temporary:
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
        manifest["schema_version"] = "enrollment.registry_manifest.v1"
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


def _run_bind(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Bind a protected split assignment to the identity registry."
    )
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--split-receipt", required=True, type=Path)
    parser.add_argument("--registry-db", required=True, type=Path)
    parser.add_argument("--registry-manifest", required=True, type=Path)
    parser.add_argument("--expected-split-receipt-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

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


def _run_check(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Check a split manifest.")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)

    payload = read_strict_json_object(args.manifest)
    manifest = SplitManifest.from_dict(payload)
    blockers = manifest.gate_blockers()
    result = {
        "schema_version": manifest.schema_version,
        "manifest_sha256": manifest.manifest_sha256,
        "policy": manifest.policy.name,
        "tracklets": len(manifest.records),
        "gate_status": "PASS" if not blockers else "BLOCKED",
        "blockers": list(blockers),
        "association_audits": {
            key: association_audit(manifest.records, key).to_dict()
            for key in ("camera_id", "cage_id", "site_id")
        }
        if manifest.records
        else {},
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    if blockers:
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "build"
    if argv and argv[0] in {"build", "bind", "check"}:
        command = argv[0]
        argv = argv[1:]
    {
        "build": _run_build,
        "bind": _run_bind,
        "check": _run_check,
    }[command](argv)
