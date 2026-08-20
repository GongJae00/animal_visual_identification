"""Audit immutable model JSON bytes without importing or executing a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.contracts.intake.pretrained_supporting_asset_intake import (
    PretrainedSupportingAssetSourceContract,
    audit_pretrained_supporting_asset,
)
from shared.contracts.intake.pretrained_weight_intake import (
    PretrainedWeightIntakeReceipt,
    PretrainedWeightSourceContract,
)
from shared.contracts.source_provenance import build_offline_tool_provenance
from shared.foundation.protected_io import read_strict_json_object, write_private_json_bundle
from shared.foundation.provenance import content_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--license-snapshot", required=True, type=Path)
    parser.add_argument("--weight-intake-receipt", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    source = PretrainedSupportingAssetSourceContract.from_dict(
        read_strict_json_object(args.source_contract)
    )
    weight_bundle = read_strict_json_object(args.weight_intake_receipt)
    expected_bundle_keys = {
        "schema_version",
        "source_contract_sha256",
        "source_contract",
        "receipt_sha256",
        "receipt",
        "tool_provenance",
        "tool_provenance_sha256",
    }
    if set(weight_bundle) != expected_bundle_keys or weight_bundle[
        "schema_version"
    ] != "cvi.pretrained_weight_intake_bundle.v1":
        raise ValueError("associated pretrained weight bundle schema differs")
    weight_source = PretrainedWeightSourceContract.from_dict(
        weight_bundle["source_contract"]
    )
    weight_receipt = PretrainedWeightIntakeReceipt.from_dict(
        weight_bundle["receipt"]
    )
    if weight_source.contract_sha256 != weight_bundle["source_contract_sha256"]:
        raise ValueError("associated pretrained weight source digest differs")
    if weight_receipt.receipt_sha256 != weight_bundle["receipt_sha256"]:
        raise ValueError("associated pretrained weight receipt digest differs")
    if content_sha256(weight_bundle["tool_provenance"]) != weight_bundle[
        "tool_provenance_sha256"
    ]:
        raise ValueError("associated pretrained weight tool provenance differs")

    receipt = audit_pretrained_supporting_asset(
        asset_path=args.asset,
        license_snapshot_path=args.license_snapshot,
        source=source,
        associated_weight_source=weight_source,
        associated_weight_receipt=weight_receipt,
    )
    tool_provenance = build_offline_tool_provenance(Path(__file__))
    bundle = {
        "schema_version": "cvi.pretrained_supporting_asset_intake_bundle.v1",
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
                "asset_kind": source.asset_kind.value,
                "asset_bytes": receipt.asset_bytes,
                "asset_sha256": receipt.asset_sha256,
                "receipt_sha256": receipt.receipt_sha256,
                "interpretation": receipt.interpretation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
