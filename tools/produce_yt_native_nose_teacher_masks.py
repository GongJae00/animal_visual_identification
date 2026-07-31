"""Produce provenance-bound SAM2.1 masks for native YT nose source images."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from cvi.nose_region.sam2_teacher import (
    MaskSelectionPolicy,
    load_local_sam2,
    produce_teacher_manifest,
    validate_source_image_manifest,
    validate_teacher_manifest,
)
from cvi.protected_io import json_document_bytes, read_strict_json_document
from cvi.protected_publication import fsync_directory, rename_directory_noreplace
from cvi.provenance import content_sha256
from cvi.retained_file import read_retained_regular_file
from cvi.source_provenance import build_offline_tool_provenance


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image-manifest", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--expected-source-receipt-sha256", required=True)
    parser.add_argument("--sam2-checkout", required=True, type=Path)
    parser.add_argument("--sam2-checkout-commit", required=True)
    parser.add_argument("--sam2-config", required=True, type=Path)
    parser.add_argument("--sam2-config-sha256", required=True)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint-sha256", required=True)
    parser.add_argument("--sam2-license-snapshot", required=True, type=Path)
    parser.add_argument("--sam2-license-snapshot-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--propagate-tracks", action="store_true")
    parser.add_argument("--minimum-model-score", type=float, default=0.50)
    parser.add_argument("--minimum-anatomical-overlap", type=float, default=0.45)
    parser.add_argument("--minimum-compactness", type=float, default=0.05)
    parser.add_argument("--minimum-area-to-box-ratio", type=float, default=0.05)
    parser.add_argument("--maximum-area-to-box-ratio", type=float, default=1.50)
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    output = args.output_dir
    if not output.is_absolute():
        raise ValueError("output directory must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output_parent = output.parent.resolve(strict=True)
    if output_parent.is_relative_to(repository_root):
        raise ValueError("SAM2 teacher output must be outside the Git worktree")
    for path, name in (
        (args.source_image_manifest, "source image manifest"),
        (args.source_receipt, "source receipt"),
        (args.sam2_checkout, "SAM2 checkout"),
        (args.sam2_config, "SAM2 config"),
        (args.sam2_checkpoint, "SAM2 checkpoint"),
        (args.sam2_license_snapshot, "SAM2 license snapshot"),
    ):
        if not path.is_absolute():
            raise ValueError(f"{name} path must be absolute")

    receipt = read_retained_regular_file(
        args.source_receipt,
        expected_sha256=args.expected_source_receipt_sha256,
        maximum_bytes=536_870_912,
        subject="native YT source receipt",
    )
    source_document = read_strict_json_document(
        args.source_image_manifest,
        maximum_bytes=536_870_912,
        maximum_nodes=20_000_000,
        maximum_keys=10_000_000,
        maximum_array_length=2_000_000,
    )
    sources = validate_source_image_manifest(
        source_document.payload,
        root=args.source_image_manifest.parent,
        source_receipt_file_sha256=receipt.sha256,
    )
    policy = MaskSelectionPolicy(
        minimum_model_score=args.minimum_model_score,
        minimum_anatomical_overlap=args.minimum_anatomical_overlap,
        minimum_compactness=args.minimum_compactness,
        minimum_area_to_box_ratio=args.minimum_area_to_box_ratio,
        maximum_area_to_box_ratio=args.maximum_area_to_box_ratio,
        ambiguity_margin=args.ambiguity_margin,
    )
    predictor, model_provenance = load_local_sam2(
        checkout=args.sam2_checkout,
        expected_checkout_commit=args.sam2_checkout_commit,
        config_path=args.sam2_config,
        expected_config_sha256=args.sam2_config_sha256,
        checkpoint_path=args.sam2_checkpoint,
        expected_checkpoint_sha256=args.sam2_checkpoint_sha256,
        license_snapshot_path=args.sam2_license_snapshot,
        expected_license_snapshot_sha256=args.sam2_license_snapshot_sha256,
        device=args.device,
        enable_video=args.propagate_tracks,
    )
    tool_provenance = build_offline_tool_provenance(
        Path(__file__),
        additional_paths=(repository_root / "src/cvi/nose_region/sam2_teacher.py",),
    )
    producer = {
        **model_provenance,
        "device": args.device,
        "prompt_contract": "NOSE_BOX_AND_POSITIVE_NOSE_KEYPOINTS",
        "output_encoding": "SOURCE_RESOLUTION_BINARY_L_PNG",
        "tool_provenance": tool_provenance,
        "tool_provenance_sha256": content_sha256(tool_provenance),
    }
    source_binding = {
        "source_manifest_schema": source_document.payload["schema_version"],
        "source_manifest_file_sha256": source_document.raw_sha256,
        "source_manifest_payload_sha256": source_document.canonical_payload_sha256,
        "source_receipt_filename": args.source_receipt.name,
        "source_receipt_file_sha256": receipt.sha256,
    }
    manifest, artifacts = produce_teacher_manifest(
        sources,
        predictor,
        source_binding=source_binding,
        producer=producer,
        policy=policy,
        propagate_tracks=args.propagate_tracks,
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output_parent))
    try:
        masks = staging / "masks"
        masks.mkdir(mode=0o700)
        for relative, payload in artifacts.items():
            _write_exclusive(staging / relative, payload)
        validate_teacher_manifest(manifest, root=staging)
        manifest_path = staging / "yt-native-nose-teacher-masks.json"
        _write_exclusive(manifest_path, json_document_bytes(manifest))
        fsync_directory(masks)
        fsync_directory(staging)
        publication = rename_directory_noreplace(staging, output)
        fsync_directory(output_parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    result = {
        "status": "CREATED",
        "output": str(output),
        "manifest": str(output / "yt-native-nose-teacher-masks.json"),
        "manifest_sha256": manifest["manifest_sha256"],
        "record_counts": manifest["record_counts"],
        "publication": publication,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
