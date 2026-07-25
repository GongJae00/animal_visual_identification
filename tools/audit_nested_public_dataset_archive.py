"""Audit a nested ZIP against its protected parent extraction manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.nested_public_dataset import audit_parent_bound_nested_public_zip
from cvi.protected_io import read_strict_json_object, write_private_json_bundle
from cvi.provenance import content_sha256
from cvi.public_dataset import PublicDatasetArchivePolicy, PublicDatasetSourceContract
from cvi.public_dataset_extraction import (
    ExtractedPublicDatasetFile,
    PublicDatasetExtractionReceipt,
)
from cvi.source_provenance import build_offline_tool_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-output-directory", required=True, type=Path)
    parser.add_argument("--parent-extraction-receipt", required=True, type=Path)
    parser.add_argument("--parent-file-manifest", required=True, type=Path)
    parser.add_argument("--parent-member-relative-path", required=True)
    parser.add_argument("--terms-snapshot", required=True, type=Path)
    parser.add_argument("--nested-source-contract", required=True, type=Path)
    parser.add_argument("--nested-policy", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    extraction_bundle = read_strict_json_object(args.parent_extraction_receipt)
    if (
        set(extraction_bundle)
        != {
            "schema_version",
            "receipt_sha256",
            "receipt",
            "tool_provenance",
            "tool_provenance_sha256",
        }
        or extraction_bundle["schema_version"]
        != "cvi.public_dataset_extraction_bundle.v2"
    ):
        raise ValueError("parent extraction bundle fields differ")
    if content_sha256(extraction_bundle["tool_provenance"]) != extraction_bundle[
        "tool_provenance_sha256"
    ]:
        raise ValueError("parent extraction provenance differs")
    parent_receipt = PublicDatasetExtractionReceipt.from_dict(
        extraction_bundle["receipt"]
    )
    if parent_receipt.receipt_sha256 != extraction_bundle["receipt_sha256"]:
        raise ValueError("parent extraction receipt digest differs")

    file_manifest = read_strict_json_object(args.parent_file_manifest)
    if set(file_manifest) != {
        "schema_version",
        "source_contract_sha256",
        "archive_receipt_sha256",
        "file_content_manifest_sha256",
        "files",
    } or file_manifest["schema_version"] != "cvi.public_dataset_file_manifest.v1":
        raise ValueError("parent file manifest fields differ")
    files = file_manifest["files"]
    if not isinstance(files, list):
        raise TypeError("parent file manifest files must be a list")
    parent_files = tuple(ExtractedPublicDatasetFile.from_dict(item) for item in files)
    if file_manifest["file_content_manifest_sha256"] != (
        parent_receipt.file_content_manifest_sha256
    ):
        raise ValueError("parent file manifest receipt binding differs")

    nested_source = PublicDatasetSourceContract.from_dict(
        read_strict_json_object(args.nested_source_contract)
    )
    nested_policy = PublicDatasetArchivePolicy.from_dict(
        read_strict_json_object(args.nested_policy)
    )
    receipt = audit_parent_bound_nested_public_zip(
        parent_output_directory=args.parent_output_directory,
        parent_extraction_receipt=parent_receipt,
        parent_files=parent_files,
        parent_member_relative_path=args.parent_member_relative_path,
        terms_snapshot_path=args.terms_snapshot,
        nested_source=nested_source,
        nested_policy=nested_policy,
    )
    tool_provenance = build_offline_tool_provenance(Path(__file__))
    bundle = {
        "schema_version": "cvi.parent_bound_nested_archive_bundle.v1",
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
                "archive_bytes": receipt.parent_member_bytes,
                "regular_files": receipt.nested_archive_receipt.regular_files,
                "receipt_sha256": receipt.receipt_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
