"""Evaluate paired raw and restored noses as a within-YT-track diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.nose_id.restoration import RestorationConfig
from cvi.nose_id.restoration_evaluation import evaluate_raw_vs_restored


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--native-bundle-sha256", required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest-sha256", required=True)
    parser.add_argument("--embedding-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-size", type=int, default=224)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--mask-mode",
        choices=("full_crop", "manifest_binary"),
        default="full_crop",
    )
    parser.add_argument(
        "--registration-mode",
        choices=("phase_translation", "canonical_crop_identity"),
        default="phase_translation",
    )
    parser.add_argument(
        "--illumination-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    bundle = evaluate_raw_vs_restored(
        native_bundle_path=args.native_bundle,
        native_bundle_sha256=args.native_bundle_sha256,
        native_root=args.native_root,
        embedding_manifest_path=args.embedding_manifest,
        embedding_manifest_sha256=args.embedding_manifest_sha256,
        embedding_onnx_path=args.embedding_onnx,
        output_path=args.output,
        evaluation_size=args.evaluation_size,
        use_cuda=args.device == "cuda",
        mask_mode=args.mask_mode.upper(),
        restoration_config=RestorationConfig(
            registration_mode=args.registration_mode,
            illumination_normalization=args.illumination_normalization,
        ),
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
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
