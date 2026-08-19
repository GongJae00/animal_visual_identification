"""Evaluate frozen Appearance, Face, and Nose on one identity-bound YT cohort."""

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
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--roi-manifest-sha256", required=True)
    parser.add_argument("--nose-lineage", type=Path, required=True)
    parser.add_argument("--nose-lineage-sha256", required=True)
    parser.add_argument("--nose-manifest", type=Path, required=True)
    parser.add_argument("--nose-manifest-sha256", required=True)
    parser.add_argument("--nose-onnx", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--frozen-model-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--nose-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fusion-resolution", type=int, default=20)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from legacy.version.afn.experiments.unified_multievidence import evaluate_unified_multievidence

    bundle = evaluate_unified_multievidence(
        native_bundle_path=args.native_bundle,
        native_bundle_sha256=args.native_bundle_sha256,
        native_root=args.native_root,
        source_image_root=args.source_image_root,
        roi_manifest_path=args.roi_manifest,
        roi_manifest_sha256=args.roi_manifest_sha256,
        nose_lineage_path=args.nose_lineage,
        nose_lineage_sha256=args.nose_lineage_sha256,
        nose_manifest_path=args.nose_manifest,
        nose_manifest_sha256=args.nose_manifest_sha256,
        nose_onnx_path=args.nose_onnx,
        model_directory=args.model_dir,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
        frozen_model_sha256=args.frozen_model_sha256,
        output_path=args.output,
        device=args.device,
        nose_use_cuda=args.nose_device == "cuda",
        batch_size=args.batch_size,
        fusion_resolution=args.fusion_resolution,
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
                "population": report["population"],
                "calibration": report["calibration"]["fusions"],
                "metrics": report["evaluation"]["metrics"],
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
