"""Verify a batch receipt against separately archived immutable anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from operations.workers.batch_invariance_runner import BatchFreshWorkerReceipt
from shared.foundation.protected_io import read_content_hashed_json_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-precommitment-sha256", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    args = parser.parse_args()

    receipt = BatchFreshWorkerReceipt.from_dict(
        read_content_hashed_json_bundle(
            args.receipt,
            schema_version="evaluation.batch_invariance_bundle.v4",
            payload_field="receipt",
            sha256_field="receipt_sha256",
        )
    )
    if receipt.receipt_sha256 != args.expected_receipt_sha256:
        raise ValueError("batch receipt differs from external final anchor")
    if receipt.batch_receipt.precommitment_sha256 != (
        args.expected_precommitment_sha256
    ):
        raise ValueError("batch receipt precommitment differs from external anchor")
    print(json.dumps({
        "status": "VERIFIED",
        "decision": receipt.batch_receipt.decision.value,
        "precommitment_sha256": (
            receipt.batch_receipt.precommitment_sha256
        ),
        "receipt_sha256": receipt.receipt_sha256,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
