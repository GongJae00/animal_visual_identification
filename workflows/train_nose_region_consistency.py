"""Fine-tune and publish the raw/masked/degraded Nose embedding v3 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts.model_parity import ParityThresholds
from localization.nose_region.embedding_consistency_training import train_and_export


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-lineage", type=Path, required=True)
    parser.add_argument("--parent-lineage-sha256", required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--model-directory-sha256", required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle-sha256", required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle-sha256", required=True)
    parser.add_argument("--old-crop-manifest", type=Path, required=True)
    parser.add_argument("--old-crop-manifest-sha256", required=True)
    parser.add_argument("--old-crop-root", type=Path, required=True)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--native-bundle-sha256", required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--support-manifest-sha256", required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--old-batch-size", type=int, default=32)
    parser.add_argument("--native-pair-batch-size", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=5e-7)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--parity-max-absolute-error", type=float, default=1e-4)
    parser.add_argument("--parity-max-relative-error", type=float, default=2e-2)
    parser.add_argument("--parity-relative-error-floor", type=float, default=1e-4)
    parser.add_argument("--parity-min-cosine", type=float, default=0.99999)
    return parser


def main() -> None:
    args = _parser().parse_args()
    lineage = train_and_export(
        parent_lineage_path=args.parent_lineage,
        parent_lineage_sha256=args.parent_lineage_sha256,
        parent_root=args.parent_root,
        parent_checkpoint_path=args.parent_checkpoint,
        parent_checkpoint_sha256=args.parent_checkpoint_sha256,
        model_directory=args.model_directory,
        model_directory_sha256=args.model_directory_sha256,
        weight_intake_bundle=args.weight_intake_bundle,
        weight_intake_bundle_sha256=args.weight_intake_bundle_sha256,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
        preprocessor_intake_bundle_sha256=args.preprocessor_intake_bundle_sha256,
        old_crop_manifest_path=args.old_crop_manifest,
        old_crop_manifest_sha256=args.old_crop_manifest_sha256,
        old_crop_root=args.old_crop_root,
        native_bundle_path=args.native_bundle,
        native_bundle_sha256=args.native_bundle_sha256,
        native_root=args.native_root,
        support_manifest_path=args.support_manifest,
        support_manifest_sha256=args.support_manifest_sha256,
        support_root=args.support_root,
        output_dir=args.output_dir,
        device_name=args.device,
        epochs=args.epochs,
        old_batch_size=args.old_batch_size,
        native_pair_batch_size=args.native_pair_batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        parity_thresholds=ParityThresholds(
            maximum_absolute_error=args.parity_max_absolute_error,
            maximum_relative_error=args.parity_max_relative_error,
            relative_error_floor=args.parity_relative_error_floor,
            minimum_cosine_similarity=args.parity_min_cosine,
        ),
    )
    print(
        json.dumps(
            {
                "status": "DONE",
                "output_dir": str(args.output_dir),
                "lineage_sha256": lineage["lineage_sha256"],
                "selected_checkpoint": lineage["artifacts"]["selected_checkpoint"][
                    "path"
                ],
                "onnx": lineage["artifacts"]["onnx"]["path"],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
