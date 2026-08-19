"""Train and DEV-select the bounded residual N4 metric adapter."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--tracks-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-track", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from legacy.version.n4.experiments.n4_metric_adapter import train_metric_adapter

    checkpoint = train_metric_adapter(
        cache_manifest_path=args.cache_manifest,
        cache_manifest_sha256=args.cache_manifest_sha256,
        output_checkpoint_path=args.output_checkpoint,
        epochs=args.epochs,
        bottleneck_dim=args.bottleneck_dim,
        scale=args.scale,
        tracks_per_batch=args.tracks_per_batch,
        samples_per_track=args.samples_per_track,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        margin=args.margin,
        anchor_weight=args.anchor_weight,
        patience=args.patience,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": "TRAINED_DEV_SELECTED_N4_METRIC_ADAPTER",
                "checkpoint_payload_sha256": checkpoint["checkpoint_payload_sha256"],
                "selected_epoch": checkpoint["selected_epoch"],
                "selected_metrics": checkpoint["selection"]["selected_metrics"],
                "output_checkpoint": str(args.output_checkpoint),
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
