"""Augment labels with registered UUIDv5 dog_id values.

Registry build/bind and split-manifest check live on evaluation:

``uv run python -m evaluation.commands.evaluate registry-build|registry-bind|split-check``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from enrollment.registry.identity_registry import compute_registered_dog_id
from shared.foundation.protected_io import read_strict_json_object
from shared.foundation.provenance import content_sha256


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "augment-labels":
        argv = argv[1:]
    parser = argparse.ArgumentParser(
        prog="python -m enrollment.commands.enroll",
        description=__doc__,
    )
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    labels = read_strict_json_object(args.labels)
    records = labels.get("records", [])
    n_resolved = 0
    for rec in records:
        did = rec.get("dataset_identity_id", "")
        if did:
            rec["registered_dog_id"] = compute_registered_dog_id(did)
            n_resolved += 1

    labels["registered_id_count"] = n_resolved
    labels["registered_id_manifest_sha256"] = content_sha256(
        {r["sample_token"]: r.get("registered_dog_id", "") for r in records}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(labels, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "DONE",
                "total_records": len(records),
                "resolved": n_resolved,
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
