"""Audit one license-bound public dataset ZIP without extracting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.protected_io import read_strict_json_object, write_private_json_bundle
from cvi.provenance import content_sha256
from cvi.public_dataset import (
    PublicDatasetArchivePolicy,
    PublicDatasetSourceContract,
    audit_public_dataset_zip,
)
from cvi.source_provenance import build_offline_tool_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--archive-policy", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--terms-snapshot", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    source = PublicDatasetSourceContract.from_dict(
        read_strict_json_object(args.source_contract)
    )
    policy = PublicDatasetArchivePolicy.from_dict(
        read_strict_json_object(args.archive_policy)
    )
    receipt = audit_public_dataset_zip(
        archive_path=args.archive,
        terms_snapshot_path=args.terms_snapshot,
        source=source,
        policy=policy,
    )
    tool_provenance = build_offline_tool_provenance(Path(__file__))
    bundle = {
        "schema_version": "cvi.public_dataset_archive_bundle.v2",
        "source_contract_sha256": source.contract_sha256,
        "archive_policy_sha256": policy.policy_sha256,
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
                "dataset_id": source.dataset_id,
                "archive_bytes": receipt.archive_bytes,
                "regular_files": receipt.regular_files,
                "receipt_sha256": receipt.receipt_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
