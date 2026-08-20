"""Join sealed labels and evaluate pair-matched visual-control scores.

Commands: evaluate (default), plan, execute, score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.crop_export import CropExportReceipt
from evaluation.protected_verification import (
    ClusterBootstrapConfig,
    FrozenVerificationThreshold,
)
from evaluation.controls.control_evaluation import (
    ControlEvaluationPolicy,
    control_evaluation_bindings_from_payload,
    evaluate_sealed_control_scores,
)
from evaluation.controls.control_scoring import (
    ControlBlindScoreReceipt,
    ControlScorePolicy,
    EmbeddingCacheManifest,
    EmbeddingCachePolicy,
    build_control_scoring_inventory,
    control_scoring_requests_from_payload,
    score_control_requests_from_cache,
    verify_embedding_cache_files,
)
from evaluation.controls.control_transform import (
    ControlTransformConfigManifest,
    ControlTransformExecutionPolicy,
    ControlTransformReceipt,
    control_transform_tasks_from_payload,
    execute_control_transforms,
    validate_control_policy_configs,
)
from evaluation.controls.mask_semantics import (
    MaskSemanticPolicy,
    MaskSemanticVerification,
    verify_mask_pixel_semantics,
)
from evaluation.controls.pairing import PairingPolicy, pair_construction_from_bundle_payloads
from evaluation.controls.policy import (
    ControlMaskManifest,
    ControlMaskVerification,
    VisualControlPolicy,
    plan_visual_control_audit,
    verify_control_mask_files,
)
from evaluation.controls.scoring import verify_pair_artifact_files
from shared.foundation.protected_io import (
    read_strict_json_object,
    write_private_json_bundle,
)


def _run_evaluate(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-scoring-requests", required=True, type=Path)
    parser.add_argument("--pair-artifact-bindings", required=True, type=Path)
    parser.add_argument("--pair-ground-truth", required=True, type=Path)
    parser.add_argument("--pair-summary", required=True, type=Path)
    parser.add_argument("--pairing-policy", required=True, type=Path)
    parser.add_argument("--control-evaluation-bindings", required=True, type=Path)
    parser.add_argument("--blind-score-receipt", required=True, type=Path)
    parser.add_argument("--embedding-cache-manifest", required=True, type=Path)
    parser.add_argument("--frozen-threshold", required=True, type=Path)
    parser.add_argument("--bootstrap-config", required=True, type=Path)
    parser.add_argument("--evaluation-policy", required=True, type=Path)
    parser.add_argument("--evaluation-output", required=True, type=Path)
    args = parser.parse_args(argv)

    construction = pair_construction_from_bundle_payloads(
        read_strict_json_object(args.pair_scoring_requests),
        read_strict_json_object(args.pair_artifact_bindings),
        read_strict_json_object(args.pair_ground_truth),
        read_strict_json_object(args.pair_summary),
    )
    plan_sha256, pair_set_sha256, bindings, summaries = (
        control_evaluation_bindings_from_payload(
            read_strict_json_object(args.control_evaluation_bindings)
        )
    )
    result = evaluate_sealed_control_scores(
        construction=construction,
        pairing_policy=PairingPolicy.from_dict(
            read_strict_json_object(args.pairing_policy)
        ),
        plan_sha256=plan_sha256,
        pair_set_sha256=pair_set_sha256,
        bindings=bindings,
        panel_summaries=summaries,
        blind_scores=ControlBlindScoreReceipt.from_dict(
            read_strict_json_object(args.blind_score_receipt)
        ),
        embedding_cache_manifest=EmbeddingCacheManifest.from_dict(
            read_strict_json_object(args.embedding_cache_manifest)
        ),
        threshold=FrozenVerificationThreshold.from_dict(
            read_strict_json_object(args.frozen_threshold)
        ),
        bootstrap=ClusterBootstrapConfig.from_dict(
            read_strict_json_object(args.bootstrap_config)
        ),
        policy=ControlEvaluationPolicy.from_dict(
            read_strict_json_object(args.evaluation_policy)
        ),
    )
    write_private_json_bundle(
        ((args.evaluation_output, result.to_dict()),)
    )
    print(
        json.dumps(
            {
                "status": "CREATED",
                "plan_sha256": result.plan_sha256,
                "pair_set_sha256": result.pair_set_sha256,
                "evaluation_receipt_sha256": result.receipt_sha256,
                "panels": len(result.panels),
                "bindings_joined": result.cost.bindings_joined,
                "auc_sort_items": result.cost.auc_sort_items,
            },
            sort_keys=True,
        )
    )


def _run_plan(argv: list[str]) -> None:
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
    args = parser.parse_args(argv)

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


def _run_execute(argv: list[str]) -> None:
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
    args = parser.parse_args(argv)

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


def _run_score(argv: list[str]) -> None:
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
    args = parser.parse_args(argv)

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


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "evaluate"
    if argv and argv[0] in {"evaluate", "plan", "execute", "score"}:
        command = argv[0]
        argv = argv[1:]
    {
        "evaluate": _run_evaluate,
        "plan": _run_plan,
        "execute": _run_execute,
        "score": _run_score,
    }[command](argv)
