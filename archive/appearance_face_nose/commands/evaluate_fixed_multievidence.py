"""Evaluate the frozen exposed publisher-test A0/F5/N3 fixed panel."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--f5-checkpoint", type=Path, required=True)
    parser.add_argument("--f5-training-roi-manifest", type=Path, required=True)
    parser.add_argument("--n3-lineage", type=Path, required=True)
    parser.add_argument("--n3-runtime-manifest", type=Path, required=True)
    parser.add_argument("--n3-onnx", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--frozen-model-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n3-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fusion-resolution", type=int, default=20)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--topology-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from archive.appearance_face_nose.experiments.fixed_multievidence import evaluate_fixed_panel

    bundle = evaluate_fixed_panel(
        panel_path=args.panel,
        panel_sha256=args.panel_sha256,
        roi_manifest_path=args.roi_manifest,
        source_image_root=args.source_image_root,
        f5_checkpoint_path=args.f5_checkpoint,
        f5_training_roi_manifest_path=args.f5_training_roi_manifest,
        n3_lineage_path=args.n3_lineage,
        n3_runtime_manifest_path=args.n3_runtime_manifest,
        n3_onnx_path=args.n3_onnx,
        model_directory=args.model_dir,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
        frozen_model_sha256=args.frozen_model_sha256,
        output_path=args.output,
        topology_output_path=args.topology_output,
        device=args.device,
        n3_use_cuda=args.n3_device == "cuda",
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
                "metrics": report["evaluation"]["metrics"],
                "rescue_break_vs_A0": report["evaluation"]["rescue_break_vs_A0"],
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
