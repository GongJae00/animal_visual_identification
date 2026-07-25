"""Safely extract an archive already admitted by the public intake gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.protected_io import read_strict_json_object, write_private_json_bundle
from cvi.provenance import content_sha256
from cvi.public_dataset import (
    PublicDatasetArchivePolicy,
    PublicDatasetArchiveReceipt,
    PublicDatasetSourceContract,
)
from cvi.public_dataset_extraction import extract_audited_public_dataset_zip
from cvi.source_provenance import build_offline_tool_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--archive-policy", required=True, type=Path)
    parser.add_argument("--archive-receipt", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--extraction-receipt", required=True, type=Path)
    parser.add_argument("--file-manifest", required=True, type=Path)
    args = parser.parse_args()

    source = PublicDatasetSourceContract.from_dict(
        read_strict_json_object(args.source_contract)
    )
    policy = PublicDatasetArchivePolicy.from_dict(
        read_strict_json_object(args.archive_policy)
    )
    archive_bundle = read_strict_json_object(args.archive_receipt)
    expected_archive_bundle_fields = {
        "schema_version",
        "source_contract_sha256",
        "archive_policy_sha256",
        "receipt_sha256",
        "receipt",
        "tool_provenance",
        "tool_provenance_sha256",
    }
    if (
        set(archive_bundle) != expected_archive_bundle_fields
        or archive_bundle["schema_version"] != "cvi.public_dataset_archive_bundle.v2"
    ):
        raise ValueError("public dataset archive bundle fields differ")
    if content_sha256(archive_bundle["tool_provenance"]) != archive_bundle[
        "tool_provenance_sha256"
    ]:
        raise ValueError("public dataset archive tool provenance differs")
    archive_receipt = PublicDatasetArchiveReceipt.from_dict(
        archive_bundle["receipt"]
    )
    if archive_receipt.receipt_sha256 != archive_bundle["receipt_sha256"]:
        raise ValueError("public dataset archive bundle digest differs")
    if source.contract_sha256 != archive_bundle["source_contract_sha256"]:
        raise ValueError("public dataset archive bundle source differs")
    if policy.policy_sha256 != archive_bundle["archive_policy_sha256"]:
        raise ValueError("public dataset archive bundle policy differs")

    receipt, files = extract_audited_public_dataset_zip(
        archive_path=args.archive,
        source=source,
        archive_policy=policy,
        archive_receipt=archive_receipt,
        output_directory=args.output_directory,
    )
    tool_provenance = build_offline_tool_provenance(Path(__file__))
    receipt_bundle = {
        "schema_version": "cvi.public_dataset_extraction_bundle.v2",
        "receipt_sha256": receipt.receipt_sha256,
        "receipt": receipt.to_dict(),
        "tool_provenance": tool_provenance,
        "tool_provenance_sha256": content_sha256(tool_provenance),
    }
    manifest_payload = {
        "schema_version": "cvi.public_dataset_file_manifest.v1",
        "source_contract_sha256": source.contract_sha256,
        "archive_receipt_sha256": archive_receipt.receipt_sha256,
        "file_content_manifest_sha256": receipt.file_content_manifest_sha256,
        "files": [item.to_dict() for item in files],
    }
    write_private_json_bundle(
        (
            (args.extraction_receipt, receipt_bundle),
            (args.file_manifest, manifest_payload),
        )
    )
    print(
        json.dumps(
            {
                "status": receipt.decision,
                "files": receipt.extracted_regular_files,
                "bytes": receipt.extracted_bytes,
                "publication_strategy": receipt.publication_strategy,
                "receipt_sha256": receipt.receipt_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
