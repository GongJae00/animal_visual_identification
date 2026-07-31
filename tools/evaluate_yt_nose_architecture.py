"""Evaluate exact raw, student-mask, and restored Nose K=5 architectures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--native-bundle-sha256", required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest-sha256", required=True)
    parser.add_argument("--embedding-onnx", type=Path, required=True)
    parser.add_argument("--embedding-lineage", type=Path)
    parser.add_argument("--embedding-lineage-sha256")
    parser.add_argument(
        "--population-role", choices=("all", "consistency_eval"), default="all"
    )
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest-sha256", required=True)
    parser.add_argument("--mask-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--calibration-fraction", type=float, default=0.3)
    parser.add_argument("--calibration-seed", type=int, default=73)
    parser.add_argument("--fusion-grid-step", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    # Keep ONNX Runtime and restoration dependencies out of the CLI help path.
    from cvi.nose_id.architecture_evaluation import evaluate_nose_architectures

    bundle = evaluate_nose_architectures(
        native_bundle_path=args.native_bundle,
        native_bundle_sha256=args.native_bundle_sha256,
        native_root=args.native_root,
        embedding_manifest_path=args.embedding_manifest,
        embedding_manifest_sha256=args.embedding_manifest_sha256,
        embedding_onnx_path=args.embedding_onnx,
        embedding_lineage_path=args.embedding_lineage,
        embedding_lineage_sha256=args.embedding_lineage_sha256,
        population_role=args.population_role,
        mask_manifest_path=args.mask_manifest,
        mask_manifest_sha256=args.mask_manifest_sha256,
        mask_onnx_path=args.mask_onnx,
        output_path=args.output,
        use_cuda=args.device == "cuda",
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence_level=args.bootstrap_confidence_level,
        calibration_fraction=args.calibration_fraction,
        calibration_seed=args.calibration_seed,
        fusion_grid_step=args.fusion_grid_step,
    )
    report = bundle["report"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "interpretation": report["interpretation"],
                "report_sha256": bundle["report_sha256"],
                "eligible_identity_count": report["population"][
                    "eligible_identity_count"
                ],
                "metrics": report["metrics"],
                "paired_delta_bootstrap_cis": report[
                    "paired_delta_bootstrap_cis"
                ],
                "calibrated_score_fusion": report["calibrated_score_fusion"],
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
