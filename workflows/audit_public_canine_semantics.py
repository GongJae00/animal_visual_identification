"""Create a protected semantic receipt for one audited public canine archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foundation.protected_io import write_private_json_bundle
from foundation.provenance import content_sha256
from data_pipeline.public_canine_manifest import (
    ArchiveReceiptBinding,
    DOGFACE_DATASET,
    MPDD_DATASET,
    SIBETAN_DATASET,
    YT_DATASET,
)
from data_pipeline.public_canine_semantic_intake import derive_public_canine_semantics
from data_pipeline.public_dataset_receipt_io import read_public_archive_receipt_bundle
from artifact_contracts.source_provenance import build_offline_tool_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(DOGFACE_DATASET, MPDD_DATASET, SIBETAN_DATASET, YT_DATASET),
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dogface-classes-train", type=Path)
    parser.add_argument("--dogface-classes-test", type=Path)
    args = parser.parse_args()

    archive_receipt = read_public_archive_receipt_bundle(args.archive_receipt)
    binding = ArchiveReceiptBinding(
        dataset_name=args.dataset,
        archive_sha256=archive_receipt.archive_sha256,
        archive_receipt_sha256=archive_receipt.receipt_sha256,
    )
    _, receipt = derive_public_canine_semantics(
        dataset_name=args.dataset,
        archive_path=args.archive,
        binding=binding,
        dogface_classes_train=args.dogface_classes_train,
        dogface_classes_test=args.dogface_classes_test,
    )
    variants = receipt.variants
    tool_provenance = build_offline_tool_provenance(Path(__file__))
    bundle = {
        "schema_version": "cvi.public_canine_semantic_bundle.v1",
        "receipt_sha256": receipt.receipt_sha256,
        "receipt": receipt.to_dict(),
        "tool_provenance": tool_provenance,
        "tool_provenance_sha256": content_sha256(tool_provenance),
    }
    write_private_json_bundle(((args.output, bundle),))
    print(
        json.dumps(
            {
                "status": "PASS_SEMANTIC_INTAKE",
                "dataset": args.dataset,
                "images": sum(item.image_count for item in variants),
                "variants": len(variants),
                "receipt_sha256": receipt.receipt_sha256,
            },
            sort_keys=True,
        )
    )
if __name__ == "__main__":
    main()
