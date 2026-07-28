"""Materialize AP-10K domestic-dog split 1 for YOLO pose fine-tuning."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath

from cvi.provenance import content_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EOPNOTSUPP}:
            raise
        shutil.copy2(source, target)
        return "copy"


def _source_image(root: Path, relative: object) -> tuple[Path, PurePosixPath]:
    if not isinstance(relative, str):
        raise ValueError("AP-10K image file_name must be text")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or relative != path.as_posix()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("AP-10K image file_name must be a safe relative path")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*path.parts)
    if candidate.is_symlink():
        raise ValueError("AP-10K source image must be a regular file under data root")
    source = candidate.resolve(strict=True)
    if not source.is_relative_to(resolved_root) or not source.is_file():
        raise ValueError("AP-10K source image must be a regular file under data root")
    return source, path


def main() -> None:
    args = parse_args()
    source = args.data_root / "ap-10k"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    materialized_files: list[list[str]] = []
    transfer_modes: set[str] = set()
    for split in ("train", "val", "test"):
        image_output = args.output_dir / "images" / split
        label_output = args.output_dir / "labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        payload = json.loads(
            (source / "annotations" / f"ap10k-{split}-split1.json").read_text()
        )
        images = {int(image["id"]): image for image in payload["images"]}
        grouped: dict[int, list[dict]] = {}
        for annotation in payload["annotations"]:
            if int(annotation["category_id"]) == 8:
                grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
        instance_count = 0
        for image_id, instance_annotations in sorted(grouped.items()):
            image_info = images[image_id]
            source_image, relative_image = _source_image(
                source / "data", image_info["file_name"]
            )
            target_image = image_output.joinpath(*relative_image.parts)
            target_image.parent.mkdir(parents=True, exist_ok=True)
            transfer_modes.add(_link_or_copy(source_image, target_image))
            materialized_files.append(
                [
                    target_image.relative_to(args.output_dir).as_posix(),
                    _file_sha256(target_image),
                ]
            )
            width = float(image_info["width"])
            height = float(image_info["height"])
            lines: list[str] = []
            for annotation in sorted(
                instance_annotations, key=lambda row: int(row["id"])
            ):
                x, y, box_width, box_height = (
                    float(value) for value in annotation["bbox"]
                )
                values = [
                    "0",
                    f"{(x + box_width / 2.0) / width:.8f}",
                    f"{(y + box_height / 2.0) / height:.8f}",
                    f"{box_width / width:.8f}",
                    f"{box_height / height:.8f}",
                ]
                keypoints = annotation["keypoints"]
                for index in range(17):
                    keypoint_x = float(keypoints[index * 3]) / width
                    keypoint_y = float(keypoints[index * 3 + 1]) / height
                    visibility = int(keypoints[index * 3 + 2])
                    values.extend(
                        (
                            f"{keypoint_x:.8f}",
                            f"{keypoint_y:.8f}",
                            str(visibility),
                        )
                    )
                lines.append(" ".join(values))
                instance_count += 1
            label_path = label_output.joinpath(
                *relative_image.with_suffix(".txt").parts
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines) + "\n")
            materialized_files.append(
                [
                    label_path.relative_to(args.output_dir).as_posix(),
                    _file_sha256(label_path),
                ]
            )
        counts[split] = {"images": len(grouped), "instances": instance_count}
    yaml_path = args.output_dir / "ap10k-dog-pose.yaml"
    yaml_path.write_text(
        "\n".join(
            (
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "kpt_shape: [17, 3]",
                "flip_idx: [1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13]",
                "names:",
                "  0: dog",
                "",
            )
        )
    )
    receipt = {
        "schema_version": "cvi.ap10k_dog_yolo_pose_materialization.v1",
        "source_version": "AP-10K official split 1",
        "source_archive_sha256": "420980abb135d6f66bcc8e29f289a46081214016192ae197ad24bc1525c8e62c",
        "counts": counts,
        "transfer_modes": sorted(transfer_modes),
        "materialized_files_sha256": content_sha256(sorted(materialized_files)),
        "keypoint_order": [
            "left_eye",
            "right_eye",
            "nose_center",
            "neck",
            "tail_base",
            "left_shoulder",
            "left_elbow",
            "left_front_paw",
            "right_shoulder",
            "right_elbow",
            "right_front_paw",
            "left_hip",
            "left_knee",
            "left_back_paw",
            "right_hip",
            "right_knee",
            "right_back_paw",
        ],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    (args.output_dir / "materialization.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
