"""Build a unified JSONL manifest across all admitted canid datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.adapters import ADAPTERS
from data.source_lock import admitted_records


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
