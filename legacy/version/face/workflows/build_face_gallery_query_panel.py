"""Build the deterministic common K5-feasible nested face panel."""

from __future__ import annotations

from legacy.version.root import repository_root as find_repo_root
import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from foundation.protected_publication import admit_new_external_output
from identity.face.face_gallery_query_panel import (
    build_face_gallery_query_panel,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-v2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = _new_external_output(args.output)
    protocol = read_strict_json_document(
        args.protocol_v2,
        maximum_bytes=2_147_483_648,
        maximum_nodes=25_000_000,
        maximum_keys=10_000_000,
        maximum_array_length=1_000_000,
    ).payload
    bundle = build_face_gallery_query_panel(protocol)
    write_private_json_bundle(((output, bundle),))
    print(
        json.dumps(
            {
                "status": "CREATED_FACE_GALLERY_QUERY_PANEL",
                "panel_sha256": bundle["panel_sha256"],
                "common_k5_feasible_identity_count": bundle["census"][
                    "common_k5_feasible_identity_count"
                ],
                "dependency_disjoint": True,
                "cross_session_claimed": False,
                "final_evaluation_permitted": False,
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
        repository_error="face gallery/query panel output must remain external",
        overwrite_error="refusing to overwrite face gallery/query panel",
    )


if __name__ == "__main__":
    raise SystemExit(main())
