"""Publish the fixed official PDQ source-and-regression byte profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.pdq_regression_source_intake import (
    audit_pdq_regression_source_archive,
    publish_pdq_regression_source_bundle,
)
from cvi.pdq_source_intake import PdqSourceContract
from cvi.protected_io import read_strict_json_object
from cvi.source_provenance import build_offline_tool_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--commit-api-snapshot", required=True, type=Path)
    parser.add_argument("--tree-api-snapshot", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()

    source = PdqSourceContract.from_dict(
        read_strict_json_object(args.source_contract)
    )
    if args.output_directory.exists() or args.output_directory.is_symlink():
        raise FileExistsError(args.output_directory)
    audit = audit_pdq_regression_source_archive(
        archive_path=args.archive,
        commit_api_snapshot_path=args.commit_api_snapshot,
        tree_api_snapshot_path=args.tree_api_snapshot,
        source=source,
    )
    strategy = publish_pdq_regression_source_bundle(
        audit=audit,
        source=source,
        output_directory=args.output_directory,
        tool_provenance=build_offline_tool_provenance(Path(__file__)),
    )
    print(
        json.dumps(
            {
                "status": audit.receipt.decision,
                "commit_sha": audit.receipt.commit_sha,
                "tree_sha": audit.receipt.tree_sha,
                "archive_sha256": audit.receipt.archive_sha256,
                "retained_members": len(audit.receipt.retained_members),
                "retained_source_bytes": audit.receipt.retained_source_bytes,
                "retained_source_aggregate_sha256": (
                    audit.receipt.retained_source_aggregate_sha256
                ),
                "receipt_sha256": audit.receipt.receipt_sha256,
                "publication_strategy": strategy,
                "interpretation": audit.receipt.interpretation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
