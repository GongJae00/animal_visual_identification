"""Build the exposed publisher-test A0/F5/N3 fixed diagnostic panel."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--roi-manifest-sha256", required=True)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--f5-checkpoint", type=Path, required=True)
    parser.add_argument("--f5-checkpoint-sha256", required=True)
    parser.add_argument("--f5-training-roi-manifest", type=Path, required=True)
    parser.add_argument("--f5-training-roi-manifest-sha256", required=True)
    parser.add_argument("--n3-lineage", type=Path, required=True)
    parser.add_argument("--n3-lineage-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from legacy.version.afn.experiments.fixed_multievidence import build_fixed_panel
    from foundation.protected_io import read_strict_json_document

    bundle = build_fixed_panel(
        roi_manifest_path=args.roi_manifest,
        roi_manifest_sha256=args.roi_manifest_sha256,
        source_image_root=args.source_image_root,
        f5_checkpoint_path=args.f5_checkpoint,
        f5_checkpoint_sha256=args.f5_checkpoint_sha256,
        f5_training_roi_manifest_path=args.f5_training_roi_manifest,
        f5_training_roi_manifest_sha256=args.f5_training_roi_manifest_sha256,
        n3_lineage_path=args.n3_lineage,
        n3_lineage_sha256=args.n3_lineage_sha256,
        output_path=args.output,
    )
    panel = bundle["panel"]
    document = read_strict_json_document(args.output)
    print(
        json.dumps(
            {
                "status": panel["status"],
                "panel_sha256": bundle["panel_sha256"],
                "panel_bundle_sha256": document.canonical_payload_sha256,
                "dev_identity_count": len(panel["population"]["dev_identity_ids"]),
                "eval_identity_count": len(panel["population"]["eval_identity_ids"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
