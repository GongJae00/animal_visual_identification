"""Train and export the receipt-bound DINOv2 nose-region RGB embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.contracts.model_parity import ParityThresholds
from identification.training.nose.embedding_training import train_and_export


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-identity", type=int, default=3)
    parser.add_argument("--backbone-lr", type=float, default=3e-6)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.5)
    parser.add_argument("--embedding-consistency-weight", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--parity-crop-count", type=int, default=3)
    parser.add_argument("--parity-max-absolute-error", type=float, default=1e-4)
    parser.add_argument("--parity-max-relative-error", type=float, default=2e-2)
    parser.add_argument("--parity-relative-error-floor", type=float, default=1e-4)
    parser.add_argument("--parity-min-cosine", type=float, default=0.99999)
    return parser


def main() -> None:
    args = _parser().parse_args()
    lineage = train_and_export(
        manifest_path=args.manifest,
        model_directory=args.model_directory,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
        output_dir=args.output_dir,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_identity=args.samples_per_identity,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        embedding_consistency_weight=args.embedding_consistency_weight,
        label_smoothing=args.label_smoothing,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        mixed_precision=args.mixed_precision,
        parity_crop_count=args.parity_crop_count,
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
                "selected_checkpoint": lineage["artifacts"][
                    "selected_checkpoint"
                ]["path"],
                "onnx": lineage["artifacts"]["onnx"]["path"],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
