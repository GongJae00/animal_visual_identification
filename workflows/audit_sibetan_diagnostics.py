"""Extract strict descriptive confound diagnostics from a final SiBeTan report."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from artifact_contracts.source_provenance import build_source_provenance
from experiments.sibetan_diagnostics import (
    build_sibetan_diagnostic,
    bundle_sibetan_diagnostic,
)
from foundation.protected_io import read_strict_json_document, write_private_json_bundle


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    output = Path(os.path.abspath(os.fspath(args.output)))
    output = output.parent.resolve(strict=True) / output.name
    if output.is_relative_to(repository):
        raise ValueError("SiBeTan diagnostic must be written outside the Git repository")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite SiBeTan diagnostic: {output}")

    source = read_strict_json_document(args.input)
    source_provenance = build_source_provenance(
        (repository / "workflows/audit_sibetan_diagnostics.py",)
    )
    report = build_sibetan_diagnostic(
        source.payload,
        source_file_sha256=source.raw_sha256,
        source_canonical_sha256=source.canonical_payload_sha256,
        code_sha256s={
            row["relative_path"]: row["content_sha256"]
            for row in source_provenance["code_source_files"]
        },
    )
    bundle = bundle_sibetan_diagnostic(report)
    write_private_json_bundle(((output, bundle),))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": os.fspath(output),
                "report_sha256": bundle["report_sha256"],
                "source_report_sha256": report["source_report"]["report_sha256"],
                "unavailable_field_count": len(report["unavailable_fields"]),
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
