"""Project known Full128 and Masked-AFN participation into face exposure history."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from foundation.protected_publication import admit_new_external_output
from identity.face.face_exposure_history import build_face_exposure_history

_LIMITS = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-bridge", required=True, type=Path)
    parser.add_argument("--full128-artifact", action="append", default=[], type=Path)
    parser.add_argument(
        "--masked-afn-run",
        action="append",
        default=[],
        nargs=2,
        type=Path,
        metavar=("REPORT", "KFOLD_MANIFEST"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.full128_artifact and not args.masked_afn_run:
        parser.error("at least one --full128-artifact or --masked-afn-run is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = _new_external_output(args.output)
    read = lambda path: read_strict_json_document(path, **_LIMITS).payload
    bundle = build_face_exposure_history(
        read(args.token_bridge),
        full128_artifacts=tuple(read(path) for path in args.full128_artifact),
        masked_afn_runs=tuple(
            (read(report), read(kfold)) for report, kfold in args.masked_afn_run
        ),
    )
    write_private_json_bundle(((output, bundle),))
    history = bundle["history"]
    print(
        json.dumps(
            {
                "status": history["status"],
                "history_sha256": bundle["history_sha256"],
                "resolved_record_count": (
                    0
                    if history["ledger"] is None
                    else len(history["ledger"]["records"])
                ),
                "unresolved_record_count": len(history["unresolved_rows"]),
                "role_allocation_permitted": history["role_allocation_permitted"],
                "clean_role_claims_permitted": False,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if history["role_allocation_permitted"] else 2


def _new_external_output(path: Path) -> Path:
    return admit_new_external_output(
        path,
        repository_root=Path(__file__).resolve().parents[1],
        repository_error="face exposure history output must remain external",
        overwrite_error="refusing to overwrite face exposure history",
    )


if __name__ == "__main__":
    raise SystemExit(main())
