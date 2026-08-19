"""Evaluate a fixed classical Nose texture branch against raw K5 embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legacy.version.nose.experiments.nose_texture import evaluate_nose_texture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--native-bundle-sha256", required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--nose-runtime-manifest", type=Path, required=True)
    parser.add_argument("--nose-runtime-manifest-sha256", required=True)
    parser.add_argument("--nose-onnx", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = evaluate_nose_texture(
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
    )
    report = bundle["report"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_sha256": bundle["report_sha256"],
                "population": report["population"],
                "selected_texture_weight": report["development_selection"]["selected_texture_weight"],
                "evaluation": report["evaluation"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
