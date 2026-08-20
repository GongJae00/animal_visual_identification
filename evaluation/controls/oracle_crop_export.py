"""Export an authenticated protected pair bundle as token-only oracle crops."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from data.crop_export import (
    CropExportPolicy,
    export_oracle_crops,
    oracle_crop_sources_from_payload,
)
from evaluation.controls.pairing import pair_construction_from_bundle_payloads
from shared.foundation.protected_io import read_strict_json_object as _read_object


def _receipt_target(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("receipt output must not be a symlink")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    target = parent / path.name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {target}")
    return target


def _write_private_receipt(path: Path, payload: dict[str, Any]) -> None:
    target = _receipt_target(path)
    with TemporaryDirectory(prefix=".cvi-crop-receipt-", dir=target.parent) as temp:
        staged = Path(temp) / "receipt.json"
        staged.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(staged, 0o600)
        os.link(staged, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-requests", required=True, type=Path)
    parser.add_argument("--artifact-bindings", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--pair-summary", required=True, type=Path)
    parser.add_argument("--crop-sources", required=True, type=Path)
    parser.add_argument("--export-policy", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()

    receipt_target = _receipt_target(args.receipt_output)
    construction = pair_construction_from_bundle_payloads(
        _read_object(args.scoring_requests),
        _read_object(args.artifact_bindings),
        _read_object(args.ground_truth),
        _read_object(args.pair_summary),
    )
    sources = oracle_crop_sources_from_payload(
        _read_object(args.crop_sources)
    )
    policy = CropExportPolicy.from_dict(
        _read_object(args.export_policy)
    )
    receipt = export_oracle_crops(
        construction,
        sources=sources,
        policy=policy,
        output_directory=args.output_directory,
    )
    try:
        _write_private_receipt(receipt_target, receipt.to_dict())
    except BaseException:
        for entry in receipt.artifact_manifest.entries:
            (args.output_directory / entry.relative_path).unlink(
                missing_ok=True
            )
        raise
    print(
        json.dumps(
            {
                "status": "CREATED",
                "pair_set_sha256": receipt.pair_set_sha256,
                "artifact_manifest_sha256": (
                    receipt.artifact_manifest.manifest_sha256
                ),
                "artifact_count": receipt.verification.verified_files,
            },
            sort_keys=True,
        )
    )
