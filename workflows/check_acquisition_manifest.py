"""Validate one G0 acquisition manifest and print its immutable identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.acquisition import AcquisitionManifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    payload = json.loads(
        args.manifest.resolve(strict=True).read_text(encoding="utf-8")
    )
    manifest = AcquisitionManifest.from_dict(payload)
    blockers = manifest.gate_blockers()
    result = {
        "schema_version": manifest.schema_version,
        "manifest_sha256": manifest.manifest_sha256,
        "camera_versions": len(manifest.cameras),
        "source_videos": len(manifest.videos),
        "gate_status": "PASS" if not blockers else "BLOCKED",
        "blockers": list(blockers),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
