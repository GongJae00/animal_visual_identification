"""Verify a batch receipt against separately archived immutable anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from operations.batch_invariance_runner import BatchFreshWorkerReceipt
from foundation.protected_io import read_strict_json_object


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-precommitment-sha256", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    args = parser.parse_args()

    payload = read_strict_json_object(args.receipt)
    if set(payload) != {"schema_version", "receipt_sha256", "receipt"} or payload[
        "schema_version"
    ] != "cvi.batch_invariance_bundle.v4":
        raise ValueError("batch receipt bundle schema differs")
    receipt = BatchFreshWorkerReceipt.from_dict(payload["receipt"])
    if receipt.receipt_sha256 != payload["receipt_sha256"]:
        raise ValueError("batch receipt bundle hash differs")
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
