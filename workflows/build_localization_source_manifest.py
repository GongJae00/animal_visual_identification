"""Build an admission-bindable AP-10K or DogFLW source projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.adapters import ADAPTERS
from data_pipeline.source_lock import get_record
from foundation.protected_io import write_private_json_bundle
from foundation.provenance import content_sha256
from localization.fold_protocol import build_localization_source_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ap10k-dog", "dogflw"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite localization source manifest")
    samples = ADAPTERS[args.dataset](Path(get_record(args.dataset).data_root))
    manifest = build_localization_source_manifest(samples, dataset=args.dataset)
    write_private_json_bundle(((args.output, manifest),))
    print(
        json.dumps(
            {
                "status": "CREATED_LOCALIZATION_SOURCE_MANIFEST",
                "dataset_name": args.dataset,
                "manifest_sha256": content_sha256(manifest),
                "sample_count": len(manifest["records"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
