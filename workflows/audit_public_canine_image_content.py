"""Run a receipt-bound decode and pixel-exact duplicate audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts.source_provenance import build_offline_tool_provenance
from data.public.public_canine_manifest import (
    DOGFACE_DATASET,
    MPDD_DATASET,
    SIBETAN_DATASET,
    YT_DATASET,
    ArchiveReceiptBinding,
    PublicCanineManifest,
)
from data.public.public_canine_semantic_intake import derive_public_canine_semantics
from data.public.public_dataset_receipt_io import read_public_archive_receipt_bundle
from data.public.public_image_content_audit import (
    ImageContentAuditPolicy,
    audit_public_canine_image_content,
)
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from foundation.provenance import content_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(DOGFACE_DATASET, MPDD_DATASET, SIBETAN_DATASET, YT_DATASET),
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-receipt", required=True, type=Path)
    parser.add_argument("--semantic-receipt", required=True, type=Path)
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
    manifests, derived_semantic = derive_public_canine_semantics(
        dataset_name=args.dataset,
        archive_path=args.archive,
        binding=binding,
        dogface_classes_train=args.dogface_classes_train,
        dogface_classes_test=args.dogface_classes_test,
    )
    semantic_bundle = _read_semantic_bundle(args.semantic_receipt)
    if semantic_bundle["receipt"] != derived_semantic.to_dict():
        raise ValueError("protected semantic receipt differs from current derivation")
    if semantic_bundle["receipt_sha256"] != derived_semantic.receipt_sha256:
        raise ValueError("protected semantic receipt digest differs")

    records = tuple(record for manifest in manifests for record in manifest.records)
    combined = PublicCanineManifest(
        dataset_name=args.dataset,
        dataset_version=manifests[0].dataset_version,
        source_archive_sha256=binding.archive_sha256,
        source_archive_receipt_sha256=binding.archive_receipt_sha256,
        records=records,
    )
    policy = ImageContentAuditPolicy()
    receipt = audit_public_canine_image_content(
        archive_path=args.archive,
        manifest=combined,
        policy=policy,
    )
    tool_provenance = build_offline_tool_provenance(Path(__file__))
    bundle = {
        "schema_version": "cvi.image_content_audit_bundle.v1",
        "semantic_receipt_sha256": derived_semantic.receipt_sha256,
        "policy": policy.to_dict(),
        "policy_sha256": policy.policy_sha256,
        "receipt": receipt.to_dict(),
        "receipt_sha256": receipt.receipt_sha256,
        "tool_provenance": tool_provenance,
        "tool_provenance_sha256": content_sha256(tool_provenance),
    }
    write_private_json_bundle(((args.output, bundle),))
    print(
        json.dumps(
            {
                "status": receipt.decision,
                "dataset": args.dataset,
                "images": len(receipt.records),
                "exact_duplicate_groups": len(receipt.exact_duplicate_groups),
                "unique_pixel_digests": receipt.unique_pixel_digest_count,
                "receipt_sha256": receipt.receipt_sha256,
            },
            sort_keys=True,
        )
    )


def _read_semantic_bundle(path: Path) -> dict[str, object]:
    bundle = read_strict_json_object(path)
    expected = {
        "schema_version",
        "receipt_sha256",
        "receipt",
        "tool_provenance",
        "tool_provenance_sha256",
    }
    if set(bundle) != expected or bundle["schema_version"] != (
        "cvi.public_canine_semantic_bundle.v1"
    ):
        raise ValueError("public canine semantic bundle fields differ")
    if content_sha256(bundle["tool_provenance"]) != bundle["tool_provenance_sha256"]:
        raise ValueError("public canine semantic provenance differs")
    if content_sha256(bundle["receipt"]) != bundle["receipt_sha256"]:
        raise ValueError("public canine semantic receipt digest differs")
    return bundle


if __name__ == "__main__":
    main()
