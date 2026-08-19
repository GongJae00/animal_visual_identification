"""Evaluate the Full128 B0/B1/B2 family from one packed-cache training run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.full_segment.full128 import (
    build_full128_family_index,
    discover_packed_full128_embedding_cache_adapters,
    evaluate_full128_family,
    full128_master_table_csv,
)
from foundation.protected_io import json_document_bytes, read_strict_json_document
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from embedding.methods.full_segment.preparation.data import load_full128_assembly
from embedding.methods.full_segment.preparation.inventory import (
    validate_full128_experiment_inventory_bundle,
)


def _write_file(path: Path, payload: bytes) -> None:
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
                raise OSError("Full128 publication write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inventory", type=Path)
    source.add_argument("--assembly", type=Path)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--validation-workers", type=_positive_int, default=8)
    args = parser.parse_args(argv)

    if args.inventory is not None:
        inventory_document = read_strict_json_document(args.inventory)
        inventory = validate_full128_experiment_inventory_bundle(
            inventory_document.payload,
            validation_workers=args.validation_workers,
        )
    else:
        _, inventory = load_full128_assembly(
            args.assembly,
            validation_workers=args.validation_workers,
        )
    adapters = discover_packed_full128_embedding_cache_adapters(
        args.training_run,
        inventory_bundle=inventory,
    )

    target = Path(os.path.abspath(os.fspath(args.output_directory)))
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite Full128 evaluation output: {target}"
        )
    parent = target.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)

    with TemporaryDirectory(prefix=f".{target.name}.staging-", dir=parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir(mode=0o700)
        galleries = staging / "galleries"
        evaluated_panel, reports, table = evaluate_full128_family(
            inventory_bundle=inventory,
            adapters=adapters,
            gallery_root=galleries,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        index = build_full128_family_index(reports, table)
        _write_file(
            staging / "evaluation-panel.json", json_document_bytes(evaluated_panel)
        )
        for report in reports:
            variant = report.report["variant_binding"]["variant_id"]
            _write_file(
                staging / f"report-{variant}.json",
                json_document_bytes(report.to_dict()),
            )
        _write_file(staging / "master-table.json", json_document_bytes(table))
        _write_file(
            staging / "master-table.csv",
            full128_master_table_csv(table).encode("utf-8"),
        )
        _write_file(staging / "family-index.json", json_document_bytes(index))
        fsync_directory(staging)
        strategy = rename_directory_noreplace(staging, target)
    fsync_directory(target)
    fsync_directory(parent)
    print(
        json.dumps(
            {
                "event": "full128_family_evaluated",
                "index_sha256": index["index_sha256"],
                "output_directory": str(target),
                "publication_strategy": strategy,
                "variants": ["B0", "B1", "B2"],
            },
            sort_keys=True,
        )
    )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
