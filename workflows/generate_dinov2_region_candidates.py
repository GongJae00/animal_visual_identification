"""Generate DINOv2 patch-token A/F/N candidate masks and embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifact_contracts.model_catalog import verify_model_artifact
from artifact_contracts.model_paths import CHECKPOINTS_DIR
from data_pipeline.adapters import ADAPTERS
from data_pipeline.source_lock import get_record
from localization.adapters import UltralyticsDogPoseAdapter
from localization.dinov2_region_segmentation import (
    Dinov2RegionRuntime,
    produce_dataset_region_candidates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--disable-pose", action="store_true")
    args = parser.parse_args()

    source = get_record(args.dataset)
    samples = ADAPTERS[args.dataset](Path(source.data_root))
    runtime = Dinov2RegionRuntime(
        model_directory=args.model_directory,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
        device=args.device,
    )
    pose = None
    try:
        if not args.disable_pose:
            artifact = verify_model_artifact("dog-pose", CHECKPOINTS_DIR)
            pose = UltralyticsDogPoseAdapter(
                artifact,
                "7edc2d96c2ca06942d172527b097d98b12bcf50c42b1c232b3b42ed0f1858760",
                device=args.device,
            )
        bundle = produce_dataset_region_candidates(
            samples,
            data_root=Path(source.data_root),
            output_dir=args.output_dir,
            runtime=runtime,
            pose_adapter=pose,
            batch_size=args.batch_size,
            maximum_samples=args.maximum_samples,
        )
    finally:
        if pose is not None:
            pose.close()
    records = bundle["manifest"]["records"]
    availability = {
        region: sum(
            record["regions"][region]["state"] == "AVAILABLE"
            for record in records
        )
        for region in ("A", "F", "N")
    }
    print(
        json.dumps(
            {
                "status": "CREATED_DINOV2_REGION_CANDIDATES",
                "dataset_name": args.dataset,
                "record_count": len(records),
                "availability": availability,
                "manifest_sha256": bundle["manifest_sha256"],
                "output": str(args.output_dir / "region_candidates.json"),
                "interpretation": bundle["manifest"]["interpretation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
