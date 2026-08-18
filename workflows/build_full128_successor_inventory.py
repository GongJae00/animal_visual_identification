"""Build the fixed panel and metadata-face-eligible Full128 successor inventory."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.protected_publication import admit_new_external_output
from embedding.methods.full_segment.face_visible import (
    build_face_visible_successor_inventory,
    build_face_visible_successor_inventory_v2,
    build_score_blind_face_visible_panel,
)

_LARGE_JSON = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-plan", required=True, type=Path)
    parser.add_argument("--materialization-assembly", required=True, type=Path)
    parser.add_argument("--face-overlay", required=True, type=Path)
    parser.add_argument("--face-protocol", type=Path)
    parser.add_argument("--face-protocol-v2", type=Path)
    parser.add_argument("--gallery-query-panel", type=Path)
    parser.add_argument("--panel-output", type=Path)
    parser.add_argument("--inventory-output", required=True, type=Path)
    parser.add_argument("--validation-workers", type=_positive_int, default=8)
    args = parser.parse_args(argv)

    route = read_strict_json_document(args.route_plan, **_LARGE_JSON).payload
    assembly = read_strict_json_document(
        args.materialization_assembly, **_LARGE_JSON
    ).payload
    overlay = read_strict_json_document(args.face_overlay, **_LARGE_JSON).payload
    inventory_output = _new_external_output(
        args.inventory_output, "successor inventory"
    )
    v2_requested = (
        args.face_protocol_v2 is not None or args.gallery_query_panel is not None
    )
    if v2_requested:
        if args.face_protocol_v2 is None or args.gallery_query_panel is None:
            raise ValueError(
                "--face-protocol-v2 and --gallery-query-panel are required together"
            )
        if args.face_protocol is not None or args.panel_output is not None:
            raise ValueError(
                "governance v2 inventory does not accept v1 protocol/panel output"
            )
        protocol_v2 = read_strict_json_document(
            args.face_protocol_v2, **_LARGE_JSON
        ).payload
        governance_panel = read_strict_json_document(
            args.gallery_query_panel, **_LARGE_JSON
        ).payload
        inventory = build_face_visible_successor_inventory_v2(
            route_plan_bundle=route,
            materialization_assembly=assembly,
            face_overlay_bundle=overlay,
            face_protocol_v2_bundle=protocol_v2,
            gallery_query_panel_bundle=governance_panel,
            validation_workers=args.validation_workers,
        )
        panel_sha256 = inventory["source_binding"]["fixed_panel_sha256"]
        governance_panel_sha256 = governance_panel["panel_sha256"]
        outputs = ((inventory_output, inventory),)
        panel_output_value = None
    else:
        if args.face_protocol is None or args.panel_output is None:
            raise ValueError(
                "legacy mode requires --face-protocol and --panel-output"
            )
        protocol = read_strict_json_document(
            args.face_protocol, **_LARGE_JSON
        ).payload
        panel = build_score_blind_face_visible_panel(route, overlay, protocol)
        inventory = build_face_visible_successor_inventory(
            route_plan_bundle=route,
            materialization_assembly=assembly,
            face_overlay_bundle=overlay,
            face_protocol_bundle=protocol,
            fixed_panel=panel,
            validation_workers=args.validation_workers,
        )
        panel_output = _new_external_output(args.panel_output, "successor panel")
        if panel_output == inventory_output:
            raise ValueError("successor panel and inventory outputs must differ")
        panel_sha256 = panel["panel_sha256"]
        governance_panel_sha256 = None
        outputs = ((panel_output, panel), (inventory_output, inventory))
        panel_output_value = str(panel_output)
    write_private_json_bundle(outputs)
    print(
        json.dumps(
            {
                "status": "CREATED_FULL128_SUCCESSOR_INVENTORY",
                "panel_sha256": panel_sha256,
                "gallery_query_panel_sha256": governance_panel_sha256,
                "inventory_sha256": inventory["inventory_sha256"],
                "bundle_sha256": inventory["bundle_sha256"],
                "route_plan_sample_count": inventory["inventory"]["coverage"][
                    "route_plan_sample_count"
                ],
                "panel_output": panel_output_value,
                "inventory_output": str(inventory_output),
            },
            sort_keys=True,
        )
    )
    return 0


def _new_external_output(path: Path, label: str) -> Path:
    return admit_new_external_output(
        path,
        repository_root=Path(__file__).resolve().parents[1],
        repository_error=f"{label} output must remain outside the repository",
        overwrite_error=f"refusing to overwrite {label} output",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
