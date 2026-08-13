"""Build one protected metadata-only Full128 experiment inventory."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from embedding.methods.full_segment.preparation.inventory import (
    build_full128_experiment_inventory,
)

REQUEST_SCHEMA = "cvi.full128_experiment_inventory_request.v2"


def run(
    request: object,
    *,
    unified_full_split: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    if (
        not isinstance(request, dict)
        or set(request) != {"schema_version", "rows"}
        or request["schema_version"] != REQUEST_SCHEMA
    ):
        raise ValueError("Full128 experiment inventory request schema differs")
    rows = request["rows"]
    if not isinstance(rows, list):
        raise TypeError("Full128 experiment inventory rows must be an array")
    return build_full128_experiment_inventory(
        unified_full_split=unified_full_split,
        request_rows=rows,
        artifact_root=artifact_root,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--unified-full-split", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    artifact_root = args.artifact_root.absolute()
    requested_output = args.output.absolute()
    output = requested_output.parent.resolve(strict=True) / requested_output.name
    if artifact_root == repository or artifact_root.is_relative_to(repository):
        raise ValueError("Full128 artifact root must remain outside the repository")
    if output == repository or output.is_relative_to(repository):
        raise ValueError("Full128 generated inventory must remain outside the repository")
    if requested_output.is_symlink() or output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Full128 inventory: {output}")
    request = read_strict_json_document(args.request.absolute()).payload
    split = read_strict_json_document(
        args.unified_full_split.absolute(),
        maximum_bytes=2_147_483_648,
        maximum_nodes=25_000_000,
        maximum_keys=10_000_000,
        maximum_array_length=1_000_000,
    ).payload
    bundle = run(request, unified_full_split=split, artifact_root=artifact_root)
    write_private_json_bundle(((output, bundle),))
    print(
        json.dumps(
            {
                "status": "CREATED_FULL128_EXPERIMENT_INVENTORY",
                "inventory_sha256": bundle["inventory_sha256"],
                "split_manifest_sha256": bundle["split_manifest_sha256"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
