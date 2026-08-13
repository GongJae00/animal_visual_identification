#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts.source_provenance import build_offline_tool_provenance
from identity_methods.classical.pdq_official_regression import (
    publish_official_pdq_regression,
    run_official_pdq_regression,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Admit the fixed Meta PDQ official bridge regression"
    )
    parser.add_argument("--regression-bundle-directory", required=True, type=Path)
    parser.add_argument("--native-worker-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    provenance = build_offline_tool_provenance(Path(__file__))
    receipt = run_official_pdq_regression(
        regression_bundle_directory=args.regression_bundle_directory,
        native_worker_directory=args.native_worker_directory,
        tool_provenance=provenance,
    )
    publish_official_pdq_regression(
        receipt=receipt,
        tool_provenance=provenance,
        output_path=args.output,
    )
    print(json.dumps({
        "schema_version": "cvi.pdq_official_regression_tool_output.v1",
        "receipt_sha256": receipt.receipt_sha256,
        "decision": receipt.decision,
        "interpretation": receipt.interpretation,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

