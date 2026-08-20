#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.contracts.source_provenance import build_offline_tool_provenance
from data.audit.pdq.native import build_native_pdq_worker


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the fixed portable IdentityEngine PDQ worker")
    parser.add_argument("--source-bundle-directory", required=True, type=Path)
    parser.add_argument("--worker-source", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--compiler", required=True, type=Path)
    args = parser.parse_args()
    receipt, strategy = build_native_pdq_worker(
        source_bundle_directory=args.source_bundle_directory,
        worker_source=args.worker_source,
        output_directory=args.output_directory,
        compiler=args.compiler,
        builder_tool_provenance=build_offline_tool_provenance(Path(__file__)),
    )
    print(json.dumps({
        "schema_version": "cvi.pdq_native_build_tool_output.v4",
        "receipt_sha256": receipt.receipt_sha256,
        "binary_sha256": receipt.binary_sha256,
        "publication_strategy": strategy,
        "interpretation": receipt.interpretation,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
