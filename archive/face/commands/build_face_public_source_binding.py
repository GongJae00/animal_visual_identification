"""Bind Full128 route and face-overlay samples to canonical public tokens."""

from __future__ import annotations

from archive.root import repository_root as find_repo_root
import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from shared.foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from shared.foundation.protected_publication import admit_new_external_output
from evaluation.splits.face.face_public_source_binding import (
    build_face_public_source_binding,
)

_LIMITS = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-plan", required=True, type=Path)
    parser.add_argument("--face-overlay", required=True, type=Path)
    parser.add_argument("--public-source-bundle", required=True, type=Path)
    parser.add_argument("--image-content-receipts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = _new_external_output(args.output)
    inputs = [
        read_strict_json_document(path, **_LIMITS).payload
        for path in (
            args.route_plan,
            args.face_overlay,
            args.public_source_bundle,
            args.image_content_receipts,
        )
    ]
    bundle = build_face_public_source_binding(*inputs)
    write_private_json_bundle(((output, bundle),))
    print(
        json.dumps(
            {
                "status": "CREATED_FACE_PUBLIC_SOURCE_BINDING",
                "binding_sha256": bundle["binding_sha256"],
                "record_count": len(bundle["binding"]["records"]),
                "score_inputs_used": False,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


def _new_external_output(path: Path) -> Path:
    return admit_new_external_output(
        path,
        repository_root=find_repo_root(__file__),
        repository_error="face public source binding output must remain external",
        overwrite_error="refusing to overwrite face public source binding",
    )


if __name__ == "__main__":
    raise SystemExit(main())
