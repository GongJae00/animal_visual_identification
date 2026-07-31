"""Evaluate K=1/3/5 Nose fusion as a within-YT-track diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.nose_id.fusion_scaling_evaluation import evaluate_fusion_scaling


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--native-bundle-sha256", required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--nose-runtime-manifest", type=Path, required=True)
    parser.add_argument("--nose-runtime-manifest-sha256", required=True)
    parser.add_argument("--nose-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    return parser


def main() -> None:
    args = _parser().parse_args()
    bundle = evaluate_fusion_scaling(
        native_bundle_path=args.native_bundle,
        native_bundle_sha256=args.native_bundle_sha256,
        native_root=args.native_root,
        nose_runtime_manifest_path=args.nose_runtime_manifest,
        nose_runtime_manifest_sha256=args.nose_runtime_manifest_sha256,
        nose_onnx_path=args.nose_onnx,
        output_path=args.output,
        use_cuda=args.device == "cuda",
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence_level=args.bootstrap_confidence_level,
    )
    report = bundle["report"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "interpretation": report["interpretation"],
                "report_sha256": bundle["report_sha256"],
                "eligible_identity_count": report["population"]["eligible_identity_count"],
                "metrics": report["metrics"],
                "paired_delta_bootstrap_cis": report["paired_delta_bootstrap_cis"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
