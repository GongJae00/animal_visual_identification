from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from data_pipeline.adapters import (
    _verified_path,
    adapt_ap10k_dog,
    adapt_dogflw,
    adapt_mpdd,
    adapt_sibetan,
)
from data_pipeline.report import compute_dataset_statistics
from data_pipeline.types import CaptureGroupKind


def test_verified_path_rejects_traversal_and_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    image = root / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    assert _verified_path(root, "image.png") == image.resolve()

    with pytest.raises(ValueError, match="unsafe relative path"):
        _verified_path(root, "../image.png")

    link = root / "link.png"
    link.symlink_to(image)
    with pytest.raises(ValueError, match="regular file"):
        _verified_path(root, "link.png")


def test_verified_path_preserves_symlinked_dataset_root(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    Image.new("RGB", (8, 8)).save(protected / "image.png")
    root = tmp_path / "dataset"
    root.symlink_to(protected, target_is_directory=True)

    verified = _verified_path(root, "image.png")

    assert verified == root.absolute() / "image.png"
    assert verified.relative_to(root.absolute()).as_posix() == "image.png"


def test_dogflw_preserves_only_finite_face46_points(tmp_path: Path) -> None:
    image_dir = tmp_path / "DogFLW" / "train" / "images"
    label_dir = tmp_path / "DogFLW" / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "breed_001.png"
    Image.new("RGB", (32, 24)).save(image_path)
    landmarks = [[float(index), float(index + 1)] for index in range(46)]
    landmarks[7] = ["NaN", 8.0]
    (label_dir / "breed_001.json").write_text(
        json.dumps({"landmarks": landmarks, "bounding_boxes": [1, 2, 20, 22]}),
        encoding="utf-8",
    )
    (tmp_path / "DogFLW" / "test" / "images").mkdir(parents=True)
    (tmp_path / "DogFLW" / "test" / "labels").mkdir(parents=True)

    samples = adapt_dogflw(tmp_path)
    assert len(samples) == 1
    assert len(samples[0].face_landmarks or {}) == 45
    assert "face46.7" not in (samples[0].face_landmarks or {})


def test_ap10k_adapter_preserves_instances_and_rejects_unsafe_paths(
    tmp_path: Path,
) -> None:
    annotation_dir = tmp_path / "ap-10k" / "annotations"
    image_dir = tmp_path / "ap-10k" / "data"
    annotation_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (40, 30)).save(image_dir / "dog.jpg")
    keypoints = [10.0, 10.0, 2] * 17
    payload = {
        "images": [{"id": 1, "file_name": "dog.jpg"}],
        "annotations": [
            {
                "id": 11,
                "image_id": 1,
                "category_id": 8,
                "bbox": [1, 2, 20, 15],
                "keypoints": keypoints,
            },
            {
                "id": 12,
                "image_id": 1,
                "category_id": 8,
                "bbox": [3, 4, 10, 8],
                "keypoints": keypoints,
            },
        ],
    }
    for split in ("train", "val", "test"):
        (annotation_dir / f"ap10k-{split}-split1.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    samples = adapt_ap10k_dog(tmp_path)
    assert len(samples) == 6
    assert len({sample.sample_id for sample in samples}) == 6
    assert all(len(sample.body_keypoints or {}) == 17 for sample in samples)

    payload["images"][0]["file_name"] = "../outside.jpg"
    (annotation_dir / "ap10k-train-split1.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsafe relative path"):
        adapt_ap10k_dog(tmp_path)


def test_statistics_support_auxiliary_samples_without_identities(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "ap-10k" / "annotations"
    image_dir = tmp_path / "ap-10k" / "data"
    annotation_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (40, 30)).save(image_dir / "dog.jpg")
    payload = {
        "images": [{"id": 1, "file_name": "dog.jpg"}],
        "annotations": [
            {
                "id": 11,
                "image_id": 1,
                "category_id": 8,
                "bbox": [1, 2, 20, 15],
                "keypoints": [10.0, 10.0, 2] * 17,
            }
        ],
    }
    for split in ("train", "val", "test"):
        (annotation_dir / f"ap10k-{split}-split1.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    statistics = compute_dataset_statistics(adapt_ap10k_dog(tmp_path))

    assert statistics["total_identities"] == 0
    assert statistics["total_samples"] == 3
    assert statistics["total_images"] == 1
    assert statistics["images_per_identity"] == {
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
    }


def test_sibetan_adapter_uses_gt_dog_identity_not_sequence_cluster(
    tmp_path: Path,
) -> None:
    base = tmp_path / "Sibetan"
    for cluster in (0, 1, 2):
        cluster_root = base / str(cluster)
        cluster_root.mkdir(parents=True)
        Image.new("RGB", (32, 24)).save(cluster_root / f"site_C{cluster}_{cluster}.jpg")
    (base / "gt_sibetan.json").write_text(
        json.dumps({"0": [0, 2], "1": [1]}), encoding="utf-8"
    )

    samples = adapt_sibetan(tmp_path)

    assert len(samples) == 3
    assert len({sample.registered_identity_id for sample in samples}) == 2
    assert {sample.raw_identity_id for sample in samples} == {"0", "1"}
    assert len({sample.capture_group_id for sample in samples}) == 3
    assert all(sample.camera_id is None for sample in samples)
    assert all(sample.label_availability["camera"] is False for sample in samples)
    assert samples[0].metadata["unverified_camera_token"] == "C0"


def test_mpdd_adapter_keeps_filename_camera_unverified(tmp_path: Path) -> None:
    image_root = tmp_path / "MPDD" / "pytorch" / "query"
    image_root.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(image_root / "146_c1_s3_1.jpg")

    sample = adapt_mpdd(tmp_path)[0]

    assert sample.capture_group_kind is CaptureGroupKind.POSE_VIEW_CLUSTER
    assert sample.camera_id is None
    assert sample.label_availability["camera"] is False
    assert sample.metadata == {
        "unverified_camera_token": "c1",
        "unverified_sequence_token": "s3",
    }
