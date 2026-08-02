"""Strict readers for protected public-dataset archive receipt bundles."""

from __future__ import annotations

from pathlib import Path

from foundation.protected_io import read_strict_json_object
from foundation.provenance import content_sha256
from data_pipeline.public_dataset import PublicDatasetArchiveReceipt


def read_public_archive_receipt_bundle(path: Path) -> PublicDatasetArchiveReceipt:
    bundle = read_strict_json_object(path)
    expected = {
        "schema_version",
        "source_contract_sha256",
        "archive_policy_sha256",
        "receipt_sha256",
        "receipt",
        "tool_provenance",
        "tool_provenance_sha256",
    }
    if set(bundle) != expected or bundle["schema_version"] != (
        "cvi.public_dataset_archive_bundle.v2"
    ):
        raise ValueError("public archive receipt bundle fields differ")
    if content_sha256(bundle["tool_provenance"]) != bundle["tool_provenance_sha256"]:
        raise ValueError("public archive receipt provenance differs")
    receipt_payload = bundle["receipt"]
    if not isinstance(receipt_payload, dict):
        raise TypeError("public archive receipt payload must be an object")
    receipt = PublicDatasetArchiveReceipt.from_dict(receipt_payload)
    if receipt.receipt_sha256 != bundle["receipt_sha256"]:
        raise ValueError("public archive receipt digest differs")
    if receipt.source_contract_sha256 != bundle["source_contract_sha256"]:
        raise ValueError("public archive receipt source binding differs")
    if receipt.archive_policy_sha256 != bundle["archive_policy_sha256"]:
        raise ValueError("public archive receipt policy binding differs")
    return receipt
