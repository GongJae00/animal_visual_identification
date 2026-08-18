"""Verify protected embedding production against external immutable anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from systems.workers.embedding_production_runner import (
    read_embedding_production_outer_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-precommitment-sha256", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-completed-attempt-ledger-head-sha256",
        required=True,
    )
    args = parser.parse_args()
    receipt = read_embedding_production_outer_bundle(
        args.receipt,
        expected_receipt_sha256=args.expected_receipt_sha256,
        expected_completed_attempt_ledger_head_sha256=(
            args.expected_completed_attempt_ledger_head_sha256
        ),
    )
    if receipt.precommitment_sha256 != args.expected_precommitment_sha256:
        raise ValueError("embedding receipt differs from external precommitment")
    print(json.dumps({
        "status": "VERIFIED",
        "precommitment_sha256": receipt.precommitment_sha256,
        "receipt_sha256": receipt.receipt_sha256,
        "production_receipt_sha256": receipt.production_receipt_sha256,
        "completed_attempt_ledger_head_sha256": (
            receipt.completed_attempt_ledger_head_sha256
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
