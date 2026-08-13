"""Verify a score-drift receipt against externally archived hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foundation.protected_io import read_content_hashed_json_bundle
from evaluation.integrity.score_drift_admission import (
    ScoreDriftAdmissionReceipt,
    verify_score_drift_receipt_external_anchors,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-precommitment-sha256", required=True)
    parser.add_argument("--expected-admission-plan-sha256", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    args = parser.parse_args()

    receipt = ScoreDriftAdmissionReceipt.from_dict(
        read_content_hashed_json_bundle(
            args.receipt,
            schema_version="cvi.score_drift_admission_bundle.v2",
            payload_field="receipt",
            sha256_field="receipt_sha256",
        )
    )
    verify_score_drift_receipt_external_anchors(
        receipt,
        expected_precommitment_sha256=args.expected_precommitment_sha256,
        expected_admission_plan_sha256=(
            args.expected_admission_plan_sha256
        ),
        expected_receipt_sha256=args.expected_receipt_sha256,
    )
    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "decision": receipt.decision.value,
                "receipt_sha256": receipt.receipt_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
