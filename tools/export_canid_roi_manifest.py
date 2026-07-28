"""Export instance-aware dog crops and source-valid masks from prediction caches."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from cvi.canid_data.adapters import ADAPTERS
from cvi.canid_data.source_lock import get_record
from cvi.localization.prediction_cache import read_prediction_cache
from cvi.localization.roi_manifest import build_roi_manifest
from cvi.protected_io import write_private_json_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--prediction-cache", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=1.15)
    parser.add_argument("--target-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = get_record(args.dataset)
    data_root = Path(record.data_root)
    samples = ADAPTERS[args.dataset](data_root)
    caches = [read_prediction_cache(path) for path in args.prediction_cache]
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite ROI output: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.staging-",
            dir=args.output_dir.parent,
        )
    )
    try:
        bundle = build_roi_manifest(
            samples,
            caches,
            data_root=data_root,
            output_dir=staging,
            margin=args.margin,
            target_size=args.target_size,
        )
        write_private_json_bundle(((staging / "roi_manifest.json", bundle),))
        os.rename(staging, args.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest_path = args.output_dir / "roi_manifest.json"
    summary = {
        "manifest": str(manifest_path),
        "manifest_sha256": bundle["manifest_sha256"],
        "instances": len(bundle["manifest"]["records"]),
        "automatic_accept_review_state": sum(
            record["review_state"] == "ACCEPT"
            for record in bundle["manifest"]["records"]
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
