"""Train and evaluate DINOv2 masked A/F/N residual adapters over K folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity.research.dataset_stratified_kfold import (
    read_dataset_stratified_identity_kfold,
)
from embedding.learning.masked_afn import train_and_evaluate_masked_afn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region-candidates", type=Path, action="append", required=True
    )
    parser.add_argument("--identity-kfold", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--image-content-receipts", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--fusion-resolution", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    kfold = read_dataset_stratified_identity_kfold(args.identity_kfold)
    report = train_and_evaluate_masked_afn(
        candidate_manifest_paths=args.region_candidates,
        kfold=kfold,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        residual_scale=args.residual_scale,
        fusion_resolution=args.fusion_resolution,
        device=args.device,
        seed=args.seed,
        source_bundle_path=args.source_bundle,
        image_content_receipts_path=args.image_content_receipts,
    )
    print(
        json.dumps(
            {
                "status": "CREATED_MASKED_AFN_KFOLD_REPORT",
                "report_sha256": report["report_sha256"],
                "out_of_fold_test": report["out_of_fold_test"],
                "output": str(args.output_dir / "masked_afn_report.json"),
                "interpretation": report["interpretation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
