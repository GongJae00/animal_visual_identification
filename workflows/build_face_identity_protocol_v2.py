"""Build the public-token-bound governance-v2 face identity protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from identity.face.face_identity_protocol_v2 import (
    build_face_identity_protocol_v2,
)

_LIMITS = {
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
    parser.add_argument("--token-bridge", required=True, type=Path)
    parser.add_argument("--exposure-history", required=True, type=Path)
    parser.add_argument("--joint-filter-evidence-graph", required=True, type=Path)
    parser.add_argument("--joint-filter-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = _new_external_output(args.output)
    read = lambda path: read_strict_json_document(path, **_LIMITS).payload
    bundle = build_face_identity_protocol_v2(
        read(args.route_plan),
        read(args.face_overlay),
        read(args.token_bridge),
        read(args.exposure_history),
        read(args.joint_filter_evidence_graph),
        read(args.joint_filter_receipt),
        protocol_name=args.protocol_name,
    )
    write_private_json_bundle(((output, bundle),))
    print(
        json.dumps(
            {
                "status": "CREATED_FACE_IDENTITY_PROTOCOL_V2",
                "protocol_sha256": bundle["protocol_sha256"],
                "identity_count": bundle["census"]["identity_count"],
                "sample_count": bundle["census"]["sample_count"],
                "cross_identity_unsafe_sample_count": bundle["census"][
                    "cross_identity_unsafe_sample_count"
                ],
                "clean_role_claims_permitted": False,
                "final_evaluation_permitted": False,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


def _new_external_output(path: Path) -> Path:
    repository = Path(__file__).resolve().parents[1]
    requested = path.absolute()
    output = requested.parent.resolve(strict=True) / requested.name
    if output == repository or output.is_relative_to(repository):
        raise ValueError("face identity protocol v2 output must remain external")
    if requested.is_symlink() or output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite face identity protocol v2")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
