"""Build a unified JSONL manifest across all admitted canid datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from cvi.canid_data.adapters import ADAPTERS
from cvi.canid_data.source_lock import admitted_records
from cvi.protected_io import write_private_json_bundle


def main() -> None:
    explicit_datasets = {
        arg.split("=", 1)[0]: Path(arg.split("=", 1)[1])
        for arg in sys.argv[1:]
        if "=" in arg
    }
    for record in admitted_records():
        name = record.canonical_name
        adapter = ADAPTERS.get(name)
        if adapter is None:
            print(json.dumps({"dataset": name, "status": "NO_ADAPTER"}))
            continue
        root = explicit_datasets.get(name, Path(record.data_root))
        if not root.is_dir():
            print(json.dumps({"dataset": name, "root": str(root), "status": "NOT_FOUND"}))
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
                    "dog_boxes_xyxy": list(s.dog_boxes_xyxy) if s.dog_boxes_xyxy else None,
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
        output = Path(f"manifests/canid/{name}.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"dataset": name, "samples": len(samples), "output": str(output)}))


if __name__ == "__main__":
    main()
