"""Augment a labels JSON file with deterministic registered_dog_id.

Reads the split labels file (which contains dataset_identity_id per record),
computes registered_dog_id (UUIDv5) from each dataset_identity_id, and
writes an augmented copy.  No SQLite registry required — the mapping is
purely deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity.registry.identity_registry import compute_registered_dog_id
from foundation.provenance import content_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text())
    records = labels.get("records", [])
    n_resolved = 0
    for rec in records:
        did = rec.get("dataset_identity_id", "")
        if did:
            rec["registered_dog_id"] = compute_registered_dog_id(did)
            n_resolved += 1

    labels["registered_id_count"] = n_resolved
    labels["registered_id_manifest_sha256"] = content_sha256(
        {r["sample_token"]: r.get("registered_dog_id", "")
         for r in records}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(labels, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps({
        "status": "DONE",
        "total_records": len(records),
        "resolved": n_resolved,
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
