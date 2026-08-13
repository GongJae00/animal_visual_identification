"""Evaluate a successor cache family on the fixed face-visible Full128 panel."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.full_segment.full128_successors import (
    build_authoritative_fixed_evaluation_panel,
    build_score_blind_fixed_evaluation_panel,
    evaluate_authoritative_successor_family,
    evaluate_successor_family,
    open_successor_embedding_cache,
)
from foundation.protected_io import json_document_bytes, read_strict_json_document
from foundation.protected_publication import fsync_directory, rename_directory_noreplace

_LARGE_JSON = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successor-inventory", required=True, type=Path)
    parser.add_argument("--fixed-panel", type=Path)
    parser.add_argument("--face-protocol-v2", type=Path)
    parser.add_argument("--gallery-query-panel", type=Path)
    parser.add_argument("--cache-descriptor", required=True, action="append", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=_positive_int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=_nonnegative_int, default=0)
    args = parser.parse_args(argv)

    inventory = read_strict_json_document(
        args.successor_inventory, **_LARGE_JSON
    ).payload
    v2_requested = (
        args.face_protocol_v2 is not None or args.gallery_query_panel is not None
    )
    if v2_requested:
        if args.face_protocol_v2 is None or args.gallery_query_panel is None:
            raise ValueError(
                "--face-protocol-v2 and --gallery-query-panel are required together"
            )
        if args.fixed_panel is not None:
            raise ValueError(
                "governance v2 evaluation does not accept a locally fixed panel"
            )
        protocol_v2 = read_strict_json_document(
            args.face_protocol_v2, **_LARGE_JSON
        ).payload
        governance_panel = read_strict_json_document(
            args.gallery_query_panel, **_LARGE_JSON
        ).payload
        evaluation_panel = build_authoritative_fixed_evaluation_panel(
            inventory, protocol_v2, governance_panel
        )
    else:
        if args.fixed_panel is None:
            raise ValueError("legacy evaluation requires --fixed-panel")
        source_panel = read_strict_json_document(
            args.fixed_panel, **_LARGE_JSON
        ).payload
        evaluation_panel = build_score_blind_fixed_evaluation_panel(
            inventory, source_panel
        )
    descriptors = [
        read_strict_json_document(path, **_LARGE_JSON).payload
        for path in args.cache_descriptor
    ]
    caches = [
        open_successor_embedding_cache(
            descriptor,
            successor_inventory_bundle=inventory,
            evaluation_panel=evaluation_panel,
        )
        for descriptor in descriptors
    ]
    target = Path(os.path.abspath(os.fspath(args.output_directory)))
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite successor evaluation output")
    parent = target.parent.resolve(strict=True)
    with TemporaryDirectory(prefix=f".{target.name}.staging-", dir=parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir(mode=0o700)
        if v2_requested:
            private, public = evaluate_authoritative_successor_family(
                successor_inventory_bundle=inventory,
                face_protocol_v2_bundle=protocol_v2,
                gallery_query_panel_bundle=governance_panel,
                caches=caches,
                gallery_root=staging / "galleries",
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
        else:
            private, public = evaluate_successor_family(
                successor_inventory_bundle=inventory,
                source_panel=source_panel,
                caches=caches,
                gallery_root=staging / "galleries",
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
        _write_new(
            staging / "evaluation-panel.json", json_document_bytes(evaluation_panel)
        )
        _write_new(staging / "private-report.json", json_document_bytes(private))
        _write_new(staging / "public-report.json", json_document_bytes(public))
        fsync_directory(staging)
        strategy = rename_directory_noreplace(staging, target)
    fsync_directory(target)
    fsync_directory(parent)
    print(
        json.dumps(
            {
                "status": "EVALUATED_FULL128_SUCCESSOR_FAMILY",
                "private_report_sha256": private["report_sha256"],
                "public_report_sha256": public["public_report_sha256"],
                "selected_successor_id": private["dev_selection_receipt"][
                    "selected_successor_id"
                ],
                "publication_strategy": strategy,
                "output_directory": str(target),
            },
            sort_keys=True,
        )
    )
    return 0


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("successor publication write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
