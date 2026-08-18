"""Build a unified JSONL manifest across all admitted canid datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.adapters import ADAPTERS
from data.duplicates import (
    find_cross_dataset_duplicates,
    find_exact_duplicates,
    summarize_duplicates,
)
from data.report import compute_dataset_statistics
from data.source_lock import admitted_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override one dataset root; may be repeated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="External directory for private JSONL manifests",
    )
    return parser.parse_args(argv)


def _dataset_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset-root must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in roots:
            raise ValueError(
                "dataset-root names and paths must be non-empty and unique"
            )
        if name not in ADAPTERS:
            raise ValueError(f"unknown dataset root override: {name!r}")
        roots[name] = Path(raw_path).expanduser()
    return roots


def _build_manifest(argv: list[str]) -> None:
    args = parse_args(argv)
    explicit_datasets = _dataset_roots(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in admitted_records():
        name = record.canonical_name
        adapter = ADAPTERS.get(name)
        if adapter is None:
            print(json.dumps({"dataset": name, "status": "NO_ADAPTER"}))
            continue
        root = explicit_datasets.get(name, Path(record.data_root))
        if not root.is_dir():
            print(
                json.dumps({"dataset": name, "root": str(root), "status": "NOT_FOUND"})
            )
            continue
        samples = adapter(root)
        lines = [
            json.dumps(
                {
                    "sample_id": s.sample_id,
                    "dataset_name": s.dataset_name,
                    "dataset_version": s.dataset_version,
                    "image_path": s.image_path,
                    "image_sha256": s.image_sha256,
                    "width": s.width,
                    "height": s.height,
                    "registered_identity_id": s.registered_identity_id,
                    "raw_identity_id": s.raw_identity_id,
                    "capture_group_id": s.capture_group_id,
                    "capture_group_kind": s.capture_group_kind.value,
                    "camera_id": s.camera_id,
                    "split_role": s.split_role,
                    "dog_boxes_xyxy": list(s.dog_boxes_xyxy)
                    if s.dog_boxes_xyxy
                    else None,
                    "face_box_xyxy": list(s.face_box_xyxy) if s.face_box_xyxy else None,
                    "body_keypoints": s.body_keypoints,
                    "face_landmarks": s.face_landmarks,
                    "breed": s.breed,
                    "label_availability": s.label_availability,
                },
                sort_keys=True,
            )
            for s in samples
        ]
        output = args.output_dir / f"{name}.jsonl"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite manifest: {output}")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"dataset": name, "samples": len(samples), "output": str(output)}
            )
        )


def _inspect_datasets() -> None:
    output: dict[str, dict] = {}
    for record in admitted_records():
        name = record.canonical_name
        adapter = ADAPTERS.get(name)
        if adapter is None:
            output[name] = {"error": "no adapter", "admission": record.admission.value}
            continue
        root = Path(record.data_root)
        if not root.is_dir():
            output[name] = {
                "error": "data root not found",
                "root": record.data_root,
                "admission": record.admission.value,
            }
            continue
        samples = adapter(root)
        stats = compute_dataset_statistics(samples)
        dup_summary = summarize_duplicates(samples, root)
        output[name] = {
            "admission": record.admission.value,
            "capture_kind": record.capture_group_kind.value,
            "license": record.license_id,
            "statistics": stats,
            "duplicates": dup_summary,
        }

    print(json.dumps(output, sort_keys=True, indent=2))


def _audit_duplicates() -> None:
    samples_by_dataset: dict[str, tuple] = {}
    roots: dict[str, Path] = {}
    for record in admitted_records():
        adapter = ADAPTERS.get(record.canonical_name)
        if adapter is None:
            continue
        root = Path(record.data_root)
        if not root.is_dir():
            continue
        samples_by_dataset[record.canonical_name] = adapter(root)
        roots[record.canonical_name] = root

    report: dict = {"within_dataset": {}, "cross_dataset": {}}
    for name, samples in samples_by_dataset.items():
        duplicates = find_exact_duplicates(samples, roots[name])
        report["within_dataset"][name] = {
            "groups": len(duplicates),
            "total_duplicate_samples": sum(len(g) for g in duplicates.values()),
        }

    cross = find_cross_dataset_duplicates(samples_by_dataset, roots)
    report["cross_dataset"] = {
        "groups": len(cross),
        "datasets_involved": sorted(
            {entry[0] for entries in cross.values() for entry in entries}
        ),
    }
    print(json.dumps(report, sort_keys=True, indent=2))


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "inspect":
        _inspect_datasets()
    elif arguments and arguments[0] == "duplicates":
        _audit_duplicates()
    else:
        _build_manifest(arguments)


if __name__ == "__main__":
    main()
