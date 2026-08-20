"""Export a fail-closed A/F/N evidence manifest from one validated ROI bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.foundation.protected_io import read_strict_json_document, write_private_json_bundle
from parsing.export.regions.roi_manifest import validate_roi_manifest_bundle
from parsing.export.regions.three_region_manifest import build_three_region_artifact_bundle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def export_three_region_artifacts(
    roi_manifest_path: Path, output_path: Path
) -> dict[str, object]:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite A/F/N output: {output_path}")
    root = roi_manifest_path.parent.resolve(strict=True)
    if output_path.parent.resolve(strict=True) != root:
        raise ValueError("A/F/N output must share the ROI manifest artifact root")
    document = read_strict_json_document(
        roi_manifest_path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    roi_manifest = validate_roi_manifest_bundle(document.payload, root=root)
    bundle = build_three_region_artifact_bundle(
        roi_manifest,
        root=root,
        roi_bundle_raw_sha256=document.raw_sha256,
        roi_manifest_sha256=document.payload["manifest_sha256"],
    )
    write_private_json_bundle(((output_path, bundle),))
    records = bundle["manifest"]["records"]
    return {
        "output": str(output_path),
        "manifest_sha256": bundle["manifest_sha256"],
        "records": len(records),
        "complete_records": sum(
            record["completion"]["state"] == "COMPLETE" for record in records
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            export_three_region_artifacts(args.roi_manifest, args.output), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
