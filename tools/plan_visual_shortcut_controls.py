"""Plan matched visual shortcut controls from authenticated protected inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.controls import (
    ControlMaskManifest,
    VisualControlPolicy,
    plan_visual_control_audit,
    verify_control_mask_files,
)
from cvi.crop_export import CropExportReceipt
from cvi.control_transform import (
    ControlTransformConfigManifest,
    validate_control_policy_configs,
)
from cvi.mask_semantics import (
    MaskSemanticPolicy,
    verify_mask_pixel_semantics,
)
from cvi.pairing import pair_construction_from_bundle_payloads
from cvi.protected_io import (
    read_strict_json_object,
    write_private_json_bundle,
)
from cvi.scoring import verify_pair_artifact_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-requests", required=True, type=Path)
    parser.add_argument("--artifact-bindings", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--pair-summary", required=True, type=Path)
    parser.add_argument("--crop-export-receipt", required=True, type=Path)
    parser.add_argument("--base-artifact-directory", required=True, type=Path)
    parser.add_argument("--mask-manifest", required=True, type=Path)
    parser.add_argument("--mask-directory", required=True, type=Path)
    parser.add_argument("--mask-semantic-policy", required=True, type=Path)
    parser.add_argument("--control-policy", required=True, type=Path)
    parser.add_argument(
        "--transform-config-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--scoring-output", required=True, type=Path)
    parser.add_argument("--transform-output", required=True, type=Path)
    parser.add_argument("--evaluation-output", required=True, type=Path)
    parser.add_argument("--mask-verification-output", required=True, type=Path)
    parser.add_argument(
        "--mask-semantic-verification-output",
        required=True,
        type=Path,
    )
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()

    construction = pair_construction_from_bundle_payloads(
        read_strict_json_object(args.scoring_requests),
        read_strict_json_object(args.artifact_bindings),
        read_strict_json_object(args.ground_truth),
        read_strict_json_object(args.pair_summary),
    )
    crop_receipt = CropExportReceipt.from_dict(
        read_strict_json_object(args.crop_export_receipt)
    )
    if crop_receipt.pair_set_sha256 != construction.result_sha256:
        raise ValueError("crop receipt and pair construction mismatch")
    current_base_verification = verify_pair_artifact_files(
        args.base_artifact_directory,
        crop_receipt.artifact_manifest,
    )
    if current_base_verification != crop_receipt.verification:
        raise ValueError("base artifact files changed since crop receipt")
    mask_manifest = ControlMaskManifest.from_dict(
        read_strict_json_object(args.mask_manifest)
    )
    mask_verification = verify_control_mask_files(
        args.mask_directory,
        mask_manifest,
    )
    semantic_verification = verify_mask_pixel_semantics(
        base_root=args.base_artifact_directory,
        base_manifest=crop_receipt.artifact_manifest,
        base_verification=current_base_verification,
        mask_root=args.mask_directory,
        mask_manifest=mask_manifest,
        mask_file_verification=mask_verification,
        policy=MaskSemanticPolicy.from_dict(
            read_strict_json_object(args.mask_semantic_policy)
        ),
    )
    policy = VisualControlPolicy.from_dict(
        read_strict_json_object(args.control_policy)
    )
    config_manifest = ControlTransformConfigManifest.from_dict(
        read_strict_json_object(args.transform_config_manifest)
    )
    validate_control_policy_configs(policy, config_manifest)
    plan = plan_visual_control_audit(
        construction,
        crop_receipt.artifact_manifest,
        current_base_verification,
        mask_manifest,
        mask_verification,
        semantic_verification,
        policy,
    )
    if plan.gate_blockers:
        raise RuntimeError(
            "visual control plan blocked: " + "; ".join(plan.gate_blockers)
        )
    write_private_json_bundle(
        (
            (args.scoring_output, plan.scoring_payload()),
            (args.transform_output, plan.protected_transform_payload()),
            (args.evaluation_output, plan.sealed_evaluation_payload()),
            (
                args.mask_verification_output,
                mask_verification.to_dict(),
            ),
            (
                args.mask_semantic_verification_output,
                semantic_verification.to_dict(),
            ),
            (args.summary_output, plan.summary_payload()),
        )
    )
    print(
        json.dumps(
            {
                "status": "CREATED",
                "plan_sha256": plan.plan_sha256,
                "panel_count": len(plan.panels),
                "scoring_request_count": len(plan.scoring_requests),
                "unique_embedding_artifacts": (
                    plan.cost.unique_embedding_artifacts
                ),
                "reusable_embedding_calls_saved": (
                    plan.cost.reusable_embedding_calls_saved
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
