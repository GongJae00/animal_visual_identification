"""Train a receipt-bound ArcFace model on immutable public crop artifacts."""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn

from contracts.pretrained_weight_intake import (
    PretrainedWeightIntakeReceipt,
    PretrainedWeightSourceContract,
    validate_pretrained_weight_receipt_binding,
)
from data.public_crop_manifest import PublicCropManifest
from foundation.protected_io import read_strict_json_object
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file
from identity_governance.role_exposure import RoleExposureLedger, RoleExposureReceipt
from identity_governance.training_admission import (
    TrainingAdmissionManifest,
    TrainingAdmissionReceipt,
)
from representation_learning.trainer import (
    ConvNeXtEmbedding,
    Dinov2Embedding,
    TrainConfig,
    evaluate_pretrained_development,
    train_model,
)

_BACKBONE_DIMENSIONS = {
    "dinov2-small": 384,
    "convnext-base": 768,
}
_BACKBONE_SOURCE_IDS = {
    "dinov2-small": "facebook/dinov2-small",
    "convnext-base": "facebook/convnext-base-224",
}
_MODEL_RECEIPT_BUNDLE_KEYS = {
    "schema_version",
    "source_contract_sha256",
    "source_contract",
    "receipt_sha256",
    "receipt",
    "tool_provenance",
    "tool_provenance_sha256",
}


def _verify_model_artifact(
    artifact_path: Path,
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
    expected_source_model_id: str,
) -> None:
    bundle = read_strict_json_object(receipt_path)
    if set(bundle) != _MODEL_RECEIPT_BUNDLE_KEYS or bundle["schema_version"] != (
        "cvi.pretrained_weight_intake_bundle.v1"
    ):
        raise ValueError("training model receipt bundle schema differs")
    source = PretrainedWeightSourceContract.from_dict(bundle["source_contract"])
    receipt = PretrainedWeightIntakeReceipt.from_dict(bundle["receipt"])
    if bundle["source_contract_sha256"] != source.contract_sha256:
        raise ValueError("training model source contract hash differs")
    if bundle["receipt_sha256"] != receipt.receipt_sha256:
        raise ValueError("training model receipt bundle hash differs")
    if receipt.receipt_sha256 != expected_receipt_sha256:
        raise ValueError("training model receipt hash mismatch")
    provenance = bundle["tool_provenance"]
    if not isinstance(provenance, dict) or content_sha256(provenance) != bundle[
        "tool_provenance_sha256"
    ]:
        raise ValueError("training model receipt provenance hash differs")
    validate_pretrained_weight_receipt_binding(receipt, source)
    if source.source_model_id != expected_source_model_id:
        raise ValueError("training model source ID differs from selected backbone")
    if artifact_path.name != source.weight_filename:
        raise ValueError("training model artifact filename differs from receipt")
    result = read_retained_regular_file(
        artifact_path,
        expected_bytes=source.expected_file_bytes,
        expected_sha256=source.expected_sha256,
        capture_payload=False,
        subject="training model artifact",
    )
    if (
        result.sha256 != receipt.weight_sha256
        or result.byte_count != receipt.weight_bytes
    ):
        raise ValueError("training model artifact bytes differ from receipt")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-manifest", required=True, type=Path)
    parser.add_argument("--admission-receipt", required=True, type=Path)
    parser.add_argument("--crop-manifest", required=True, type=Path)
    parser.add_argument("--crop-root", required=True, type=Path)
    parser.add_argument("--exposure-ledger", required=True, type=Path)
    parser.add_argument("--exposure-receipt", required=True, type=Path)
    parser.add_argument("--model-artifact", required=True, type=Path)
    parser.add_argument("--model-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-admission-manifest-sha256", required=True)
    parser.add_argument("--expected-admission-receipt-sha256", required=True)
    parser.add_argument("--expected-crop-manifest-sha256", required=True)
    parser.add_argument("--expected-split-receipt-sha256", required=True)
    parser.add_argument("--expected-crop-receipt-sha256", required=True)
    parser.add_argument("--expected-exposure-receipt-sha256", required=True)
    parser.add_argument("--expected-model-receipt-sha256", required=True)
    parser.add_argument(
        "--backbone", default="dinov2-small", choices=list(_BACKBONE_DIMENSIONS)
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--architecture",
        choices=("standard_arcface", "appearance_bounded_residual_v4"),
    )
    parser.add_argument("--border-consistency-weight", type=float)
    parser.add_argument("--baseline-anchor-weight", type=float)
    parser.add_argument("--residual-scale", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--evaluate-pretrained-only", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing", default=None, action="store_true", dest="gc"
    )
    parser.add_argument(
        "--no-gradient-checkpointing", default=None, action="store_false", dest="gc"
    )
    parser.add_argument("--compile", default=None, action="store_true", dest="compile")
    parser.add_argument(
        "--no-compile", default=None, action="store_false", dest="compile"
    )
    parser.add_argument("--preload", default=None, action="store_true", dest="preload")
    parser.add_argument(
        "--no-preload", default=None, action="store_false", dest="preload"
    )
    return parser


def _training_config(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> TrainConfig:
    config_dict: dict[str, Any] = {}
    if args.config is not None:
        config_dict = read_strict_json_object(args.config)
    for argument, field in (
        (args.epochs, "epochs"),
        (args.batch_size, "batch_size"),
        (args.lr, "lr"),
        (args.num_workers, "num_workers"),
        (args.architecture, "architecture"),
        (args.border_consistency_weight, "border_consistency_weight"),
        (args.baseline_anchor_weight, "baseline_anchor_weight"),
        (args.residual_scale, "residual_scale"),
        (args.gc, "gradient_checkpointing"),
        (args.compile, "compile_model"),
        (args.preload, "preload_images"),
    ):
        if argument is not None:
            config_dict[field] = argument
    config_dict["checkpoint_dir"] = str(args.output_dir / "checkpoints")
    config_dict["model_name"] = args.backbone
    expected_dim = _BACKBONE_DIMENSIONS[args.backbone]
    configured_dim = config_dict.get("embedding_dim", expected_dim)
    if configured_dim != expected_dim:
        parser.error(
            f"{args.backbone} requires embedding_dim={expected_dim}; got {configured_dim}"
        )
    config_dict["embedding_dim"] = expected_dim
    return TrainConfig.from_dict(config_dict)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.output_dir.is_symlink() or os.path.lexists(args.output_dir):
        parser.error("--output-dir must not exist")

    admission_manifest = TrainingAdmissionManifest.from_dict(
        read_strict_json_object(args.admission_manifest)
    )
    admission_receipt = TrainingAdmissionReceipt.from_dict(
        read_strict_json_object(args.admission_receipt)
    )
    crop_manifest = PublicCropManifest.from_dict(
        read_strict_json_object(args.crop_manifest)
    )
    exposure_ledger = RoleExposureLedger.from_dict(
        read_strict_json_object(args.exposure_ledger)
    )
    exposure_receipt = RoleExposureReceipt.from_dict(
        read_strict_json_object(args.exposure_receipt)
    )
    if crop_manifest.manifest_sha256 != args.expected_crop_manifest_sha256:
        raise ValueError("public crop manifest hash mismatch")

    config = _training_config(args, parser)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(json.dumps({"event": "device", "device": str(device)}), flush=True)

    model_directory = args.model_artifact.parent.resolve(strict=True)
    base_factory: type[nn.Module] = (
        Dinov2Embedding if args.backbone == "dinov2-small" else ConvNeXtEmbedding
    )
    backbone_factory = partial(base_factory, model_directory=model_directory)
    model_verifier = partial(
        _verify_model_artifact,
        args.model_artifact,
        args.model_receipt,
        expected_receipt_sha256=args.expected_model_receipt_sha256,
        expected_source_model_id=_BACKBONE_SOURCE_IDS[args.backbone],
    )

    t0 = time.time()
    common = {
        "config": config,
        "crop_root": args.crop_root,
        "admission_manifest": admission_manifest,
        "crop_manifest": crop_manifest,
        "admission_receipt": admission_receipt,
        "exposure_ledger": exposure_ledger,
        "exposure_receipt": exposure_receipt,
        "expected_admission_manifest_sha256": (
            args.expected_admission_manifest_sha256
        ),
        "expected_admission_receipt_sha256": (
            args.expected_admission_receipt_sha256
        ),
        "expected_split_receipt_sha256": args.expected_split_receipt_sha256,
        "expected_crop_receipt_sha256": args.expected_crop_receipt_sha256,
        "expected_exposure_receipt_sha256": args.expected_exposure_receipt_sha256,
        "expected_model_receipt_sha256": args.expected_model_receipt_sha256,
        "model_artifact_verifier": model_verifier,
        "device": device,
        "backbone_factory": backbone_factory,
    }
    if args.evaluate_pretrained_only:
        summary = evaluate_pretrained_development(**common)
        args.output_dir.mkdir(parents=True, exist_ok=False)
    else:
        summary = train_model(output_directory=args.output_dir, **common)
    elapsed = time.time() - t0

    summary["tool_provenance"] = {"tool": str(Path(__file__).resolve())}
    summary_path = args.output_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "DONE",
                "elapsed_seconds": round(elapsed, 1),
                "best_checkpoint": summary.get("best_checkpoint"),
                "total_steps": summary.get("total_steps", 0),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
