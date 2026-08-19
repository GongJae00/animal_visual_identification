"""Evaluate a frozen N4 metric adapter once on fixed-panel publisher EVAL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--topology-manifest", type=Path, required=True)
    parser.add_argument("--topology-sha256", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from legacy.version.n4.experiments.n4_metric_adapter import evaluate_metric_adapter

    bundle = evaluate_metric_adapter(
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        panel_path=args.panel,
        panel_sha256=args.panel_sha256,
        topology_manifest_path=args.topology_manifest,
        topology_sha256=args.topology_sha256,
        output_path=args.output,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence_level=args.bootstrap_confidence_level,
    )
    report = bundle["report"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_sha256": bundle["report_sha256"],
                "baseline_metrics": report["evaluation"]["baseline_N3"]["metrics"],
                "candidate_metrics": report["evaluation"]["candidate_N4"]["metrics"],
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
