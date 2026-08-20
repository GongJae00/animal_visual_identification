"""Audit Nose/YT or SiBeTan manifest observability proxies without reading pixels."""

from __future__ import annotations

from archive.root import repository_root as find_repo_root
import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shared.contracts.source_provenance import build_source_provenance
from archive.nose.experiments.nose_observability import (
    REPORT_BUNDLE_SCHEMA,
    audit_nose_observability,
)
from shared.foundation.protected_io import read_strict_json_document, write_private_json_bundle
from shared.foundation.provenance import content_sha256


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(os.path.abspath(os.fspath(args.output)))
    repository = find_repo_root(__file__)
    output = output.parent.resolve(strict=True) / output.name
    if output.is_relative_to(repository):
        raise ValueError("observability audit must be written outside the Git repository")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite observability audit: {output}")
    expected_file_sha256 = _require_sha256(
        args.manifest_file_sha256, "manifest file SHA-256"
    )
    document = read_strict_json_document(
        args.manifest,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    if document.raw_sha256 != expected_file_sha256:
        raise ValueError("Nose evidence manifest file differs from external pin")

    report = audit_nose_observability(document.payload)
    source_provenance = build_source_provenance(
        (repository / "archive/nose/commands/audit_nose_observability.py",)
    )
    report["source_binding"] = {
        "path": os.fspath(args.manifest),
        "file_sha256": document.raw_sha256,
        "canonical_payload_sha256": document.canonical_payload_sha256,
        "manifest_sha256": document.payload["manifest_sha256"],
        "byte_size": document.byte_size,
    }
    report["tool_provenance"] = {
        "schema_version": "archive.nose.nose_observability_tool_provenance.v1",
        "code_sha256s": {
            row["relative_path"]: row["content_sha256"]
            for row in source_provenance["code_source_files"]
        },
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    bundle = {
        "schema_version": REPORT_BUNDLE_SCHEMA,
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
                "input_format": report["input_contract"]["input_format"],
                "record_count": report["input_contract"]["record_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return bundle


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
