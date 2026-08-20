"""Audit source-bound pretrained weight bytes without deserializing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.contracts.intake.pretrained_weight_intake import (
    PretrainedWeightSourceContract,
    audit_pretrained_weight_file,
)
from shared.contracts.source_provenance import build_offline_tool_provenance
from shared.foundation.protected_io import read_strict_json_object, write_private_json_bundle
from shared.foundation.provenance import content_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--weight", required=True, type=Path)
    parser.add_argument("--license-snapshot", required=True, type=Path)
    parser.add_argument("--training-snapshot", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    source = PretrainedWeightSourceContract.from_dict(
        read_strict_json_object(args.source_contract)
    )
    receipt = audit_pretrained_weight_file(
        weight_path=args.weight,
        license_snapshot_path=args.license_snapshot,
        training_description_snapshot_path=args.training_snapshot,
        source=source,
    )
    tool_provenance = build_offline_tool_provenance(Path(__file__))
    bundle = {
        "schema_version": "shared.pretrained_weight_intake_bundle.v1",
        "source_contract_sha256": source.contract_sha256,
        "source_contract": source.to_dict(),
        "receipt_sha256": receipt.receipt_sha256,
        "receipt": receipt.to_dict(),
        "tool_provenance": tool_provenance,
        "tool_provenance_sha256": content_sha256(tool_provenance),
    }
    write_private_json_bundle(((args.receipt, bundle),))
    print(
        json.dumps(
            {
                "status": receipt.decision,
                "source_model_id": source.source_model_id,
                "weight_bytes": receipt.weight_bytes,
                "weight_sha256": receipt.weight_sha256,
                "receipt_sha256": receipt.receipt_sha256,
                "interpretation": receipt.interpretation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
