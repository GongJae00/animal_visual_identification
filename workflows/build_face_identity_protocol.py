"""Build the fixed retrospective B2-FV identity protocol and K census."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from foundation.protected_publication import admit_new_external_output
from identity.face.face_identity_protocol import build_face_identity_protocol

_LARGE_JSON_LIMITS = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--route-plan", required=True, type=Path)
    parser.add_argument("--face-overlay", required=True, type=Path)
    parser.add_argument(
        "--historical-artifact",
        required=True,
        type=Path,
        action="append",
        dest="historical_artifacts",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = _new_external_output(args.output)
    route = read_strict_json_document(args.route_plan, **_LARGE_JSON_LIMITS).payload
    overlay = read_strict_json_document(args.face_overlay, **_LARGE_JSON_LIMITS).payload
    historical_hashes = tuple(
        _file_sha256(_regular_file(path, "historical artifact"))
        for path in args.historical_artifacts
    )
    bundle = build_face_identity_protocol(
        route,
        overlay,
        protocol_name=args.protocol_name,
        historical_artifact_sha256s=historical_hashes,
    )
    write_private_json_bundle(((output, bundle),))
    print(
        json.dumps(
            {
                "status": "CREATED_FACE_IDENTITY_PROTOCOL",
                "protocol_sha256": bundle["protocol_sha256"],
                "identity_count": bundle["census"]["identity_count"],
                "sample_count": bundle["census"]["sample_count"],
                "identity_role_counts": bundle["census"]["identity_role_counts"],
                "final_evaluation_permitted": False,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, subject: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.is_symlink() or not absolute.is_file():
        raise ValueError(f"{subject} must be a regular file")
    return absolute


def _new_external_output(path: Path) -> Path:
    return admit_new_external_output(
        path,
        repository_root=Path(__file__).resolve().parents[1],
        repository_error=(
            "face identity protocol output must remain outside the repository"
        ),
        overwrite_error="refusing to overwrite face identity protocol output",
    )


if __name__ == "__main__":
    raise SystemExit(main())
