"""Verify protected evaluation artifacts against an external final anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.protected_evaluation import verify_protected_evaluation_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--expected-plan-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-advanced-exposure-declaration-sha256", required=True
    )
    parser.add_argument("--expected-output-receipt-sha256", required=True)
    args = parser.parse_args()
    receipt = verify_protected_evaluation_output(
        preparation_directory=args.preparation_directory,
        output_directory=args.output_directory,
        expected_plan_receipt_sha256=args.expected_plan_receipt_sha256,
        expected_advanced_exposure_declaration_sha256=(
            args.expected_advanced_exposure_declaration_sha256
        ),
        expected_output_receipt_sha256=args.expected_output_receipt_sha256,
    )
    print(json.dumps({
        "status": "VERIFIED",
        "plan_receipt_sha256": receipt.plan_receipt_sha256,
        "report_raw_sha256": receipt.report_raw_sha256,
        "report_canonical_payload_sha256": receipt.report_canonical_payload_sha256,
        "output_receipt_sha256": receipt.receipt_sha256,
    }, sort_keys=True))
