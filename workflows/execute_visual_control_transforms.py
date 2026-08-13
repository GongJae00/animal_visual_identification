"""Execute protected visual-control tasks with pixel-equation verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.crop_export import CropExportReceipt
from evaluation.controls.control_transform import (
    ControlTransformConfigManifest,
    ControlTransformExecutionPolicy,
    control_transform_tasks_from_payload,
    execute_control_transforms,
)
from evaluation.controls.policy import (
    ControlMaskManifest,
    ControlMaskVerification,
)
from evaluation.controls.mask_semantics import MaskSemanticVerification
from foundation.protected_io import (
    read_strict_json_object,
    write_private_json_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform-tasks", required=True, type=Path)
    parser.add_argument("--crop-export-receipt", required=True, type=Path)
    parser.add_argument("--base-artifact-directory", required=True, type=Path)
    parser.add_argument("--mask-manifest", required=True, type=Path)
    parser.add_argument("--mask-directory", required=True, type=Path)
    parser.add_argument("--mask-verification", required=True, type=Path)
    parser.add_argument(
        "--mask-semantic-verification",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--transform-config-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--execution-policy", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()

    (
        plan_sha256,
        scoring_requests_sha256,
        tasks,
    ) = control_transform_tasks_from_payload(
        read_strict_json_object(args.transform_tasks)
    )
    crop_receipt = CropExportReceipt.from_dict(
        read_strict_json_object(args.crop_export_receipt)
    )
    mask_manifest = ControlMaskManifest.from_dict(
        read_strict_json_object(args.mask_manifest)
    )
    mask_verification = ControlMaskVerification.from_dict(
        read_strict_json_object(args.mask_verification)
    )
    mask_semantic_verification = MaskSemanticVerification.from_dict(
        read_strict_json_object(args.mask_semantic_verification)
    )
    config_manifest = ControlTransformConfigManifest.from_dict(
        read_strict_json_object(args.transform_config_manifest)
    )
    execution_policy = ControlTransformExecutionPolicy.from_dict(
        read_strict_json_object(args.execution_policy)
    )
    receipt = execute_control_transforms(
        plan_sha256=plan_sha256,
        scoring_requests_sha256=scoring_requests_sha256,
        tasks=tasks,
        base_root=args.base_artifact_directory,
        base_manifest=crop_receipt.artifact_manifest,
        base_verification=crop_receipt.verification,
        mask_root=args.mask_directory,
        mask_manifest=mask_manifest,
        mask_verification=mask_verification,
        mask_semantic_verification=mask_semantic_verification,
        config_manifest=config_manifest,
        policy=execution_policy,
        output_directory=args.output_directory,
    )
    try:
        write_private_json_bundle(
            ((args.receipt_output, receipt.to_dict()),)
        )
    except BaseException:
        for entry in receipt.artifact_manifest.entries:
            (args.output_directory / entry.relative_path).unlink(
                missing_ok=True
            )
        raise
    print(
        json.dumps(
            {
                "status": "CREATED",
                "plan_sha256": receipt.plan_sha256,
                "artifact_manifest_sha256": (
                    receipt.artifact_manifest.manifest_sha256
                ),
                "artifact_count": receipt.verification.verified_files,
                "output_bytes": receipt.verification.verified_bytes,
                "subprocess_calls": receipt.cost.subprocess_calls,
                "peak_validation_raw_bytes": (
                    receipt.cost.peak_validation_raw_bytes
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
