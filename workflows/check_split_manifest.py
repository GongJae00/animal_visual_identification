"""Validate one tracklet-level split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.dataset import SplitManifest
from identity_governance.leakage import association_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    payload = json.loads(
        args.manifest.resolve(strict=True).read_text(encoding="utf-8")
    )
    manifest = SplitManifest.from_dict(payload)
    blockers = manifest.gate_blockers()
    result = {
        "schema_version": manifest.schema_version,
        "manifest_sha256": manifest.manifest_sha256,
        "policy": manifest.policy.name,
        "tracklets": len(manifest.records),
        "gate_status": "PASS" if not blockers else "BLOCKED",
        "blockers": list(blockers),
        "association_audits": {
            key: association_audit(manifest.records, key).to_dict()
            for key in ("camera_id", "cage_id", "site_id")
        }
        if manifest.records
        else {},
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
