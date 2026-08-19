"""Train and export a research-only binary nose segmentation student."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        required=True,
        help="Exact local timm MobileNetV4 Conv Small model.safetensors.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--decoder-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--parity-max-absolute-error", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    # Keep torch, timm, ONNX, and manifest imports out of the CLI help path.
    from parsing.nose_region.segmentation_training import train_and_export

    lineage = train_and_export(
        teacher_manifest_path=args.teacher_manifest,
        source_manifest_path=args.source_manifest,
        native_manifest_path=args.native_manifest,
        backbone_weights_path=args.backbone_weights,
        output_dir=args.output_dir,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        decoder_channels=args.decoder_channels,
        num_workers=args.num_workers,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        threshold=args.threshold,
        parity_max_absolute_error=args.parity_max_absolute_error,
    )
    print(
        json.dumps(
            {
                "status": "CREATED",
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
