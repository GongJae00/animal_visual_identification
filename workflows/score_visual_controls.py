"""Score visual-control requests from a verified label-blind embedding cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.crop_export import CropExportReceipt
from evaluation.controls.control_scoring import (
    ControlScorePolicy,
    EmbeddingCacheManifest,
    EmbeddingCachePolicy,
    build_control_scoring_inventory,
    control_scoring_requests_from_payload,
    score_control_requests_from_cache,
    verify_embedding_cache_files,
)
from evaluation.controls.control_transform import ControlTransformReceipt
from foundation.protected_io import (
    read_strict_json_object,
    write_private_json_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-requests", required=True, type=Path)
    parser.add_argument("--crop-export-receipt", required=True, type=Path)
    parser.add_argument("--base-artifact-directory", required=True, type=Path)
    parser.add_argument(
        "--control-transform-receipt",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--control-artifact-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--embedding-cache-manifest", required=True, type=Path)
    parser.add_argument("--embedding-cache-directory", required=True, type=Path)
    parser.add_argument("--embedding-cache-policy", required=True, type=Path)
    parser.add_argument("--score-policy", required=True, type=Path)
    parser.add_argument("--gallery-sha256", required=True)
    parser.add_argument("--inventory-output", required=True, type=Path)
    parser.add_argument(
        "--cache-verification-output",
        required=True,
        type=Path,
    )
    parser.add_argument("--score-receipt-output", required=True, type=Path)
    args = parser.parse_args()

    plan_sha256, requests = control_scoring_requests_from_payload(
        read_strict_json_object(args.scoring_requests)
    )
    crop_receipt = CropExportReceipt.from_dict(
        read_strict_json_object(args.crop_export_receipt)
    )
    transform_receipt = ControlTransformReceipt.from_dict(
        read_strict_json_object(args.control_transform_receipt)
    )
    inventory = build_control_scoring_inventory(
        plan_sha256=plan_sha256,
        requests=requests,
        base_root=args.base_artifact_directory,
        base_manifest=crop_receipt.artifact_manifest,
        base_verification=crop_receipt.verification,
        control_root=args.control_artifact_directory,
        transform_receipt=transform_receipt,
    )
    cache_manifest = EmbeddingCacheManifest.from_dict(
        read_strict_json_object(args.embedding_cache_manifest)
    )
    cache_policy = EmbeddingCachePolicy.from_dict(
        read_strict_json_object(args.embedding_cache_policy)
    )
    cache_verification = verify_embedding_cache_files(
        root=args.embedding_cache_directory,
        inventory=inventory,
        manifest=cache_manifest,
        policy=cache_policy,
    )
    score_policy = ControlScorePolicy.from_dict(
        read_strict_json_object(args.score_policy)
    )
    score_receipt = score_control_requests_from_cache(
        requests=requests,
        inventory=inventory,
        cache_root=args.embedding_cache_directory,
        cache_manifest=cache_manifest,
        cache_verification=cache_verification,
        cache_policy=cache_policy,
        score_policy=score_policy,
        gallery_sha256=args.gallery_sha256,
    )
    write_private_json_bundle(
        (
            (args.inventory_output, inventory.to_dict()),
            (
                args.cache_verification_output,
                cache_verification.to_dict(),
            ),
            (args.score_receipt_output, score_receipt.to_dict()),
        )
    )
    print(
        json.dumps(
            {
                "status": "CREATED",
                "plan_sha256": plan_sha256,
                "scoring_inventory_sha256": inventory.inventory_sha256,
                "score_receipt_sha256": score_receipt.receipt_sha256,
                "scoring_requests": score_receipt.cost.scoring_requests,
                "unique_artifacts": score_receipt.cost.unique_artifacts,
                "unique_embedding_vectors": (
                    score_receipt.cost.unique_embedding_vectors
                ),
                "neural_embedding_calls_saved": (
                    score_receipt.cost.neural_embedding_calls_saved
                ),
                "total_file_bytes_read": (
                    score_receipt.cost.total_file_bytes_read
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
