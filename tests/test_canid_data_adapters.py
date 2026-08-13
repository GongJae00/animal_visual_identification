from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from data.adapters import (
    ADAPTERS,
    RESEARCH_INTAKE_ADAPTERS,
    _verified_path,
    adapt_ap10k_dog,
    adapt_dogfacenet224,
    adapt_dogflw,
    adapt_mpdd,
    adapt_oxford_pets_dog,
    adapt_petface_dog,
    adapt_sibetan,
    load_petface_dog_split,
)
from data.report import compute_dataset_statistics
from data.types import CaptureGroupKind
from representation_learning.train.dataset import PetFaceDataset

_PETFACE_README = """# PetFace Dataset
This dataset is provided for research purposes only.
You are prohibited from redistributing or sharing this dataset.
"""


def _png_bytes(size: tuple[int, int] = (12, 8)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size).save(stream, format="PNG")
    return stream.getvalue()


def _write_petface_fixture(
    root: Path,
    *,
    train_rows: str = (
        "filename,label\ndog/000007/00.png,7\ndog/000008/00.png,8\n"
    ),
    readme: str = _PETFACE_README,
    tar_members: tuple[str, ...] = (
        "dog/000007/00.png",
        "dog/000008/00.png",
    ),
) -> None:
    root.mkdir()
    with zipfile.ZipFile(root / "metadata.zip", "w") as archive:
        archive.writestr("PetFace/README.md", readme)
        archive.writestr("PetFace/split/dog/train.csv", train_rows)
    payload = _png_bytes()
    with tarfile.open(root / "dog-001.tar.gz", "w:gz") as archive:
        for member_name in tar_members:
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _write_dogface_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_values: tuple[int, ...] = (1, 1),
    test_values: tuple[int, ...] = (2,),
) -> None:
    image_root = root / "after_4_bis"
    for identity, count in Counter(train_values + test_values).items():
        identity_root = image_root / str(identity)
        identity_root.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (12, 8)).save(identity_root / f"{identity}.{index}.jpg")
    for split, values in (("train", train_values), ("test", test_values)):
        payload = "".join(f"{value}\n" for value in values).encode("ascii")
        (root / f"classes_{split}.txt").write_bytes(payload)
        monkeypatch.setattr(
            f"data.adapters.DOGFACE_{split.upper()}_SHA256",
            hashlib.sha256(payload).hexdigest(),
        )
        monkeypatch.setattr(
            f"data.adapters.DOGFACE_{split.upper()}_MD5",
            hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        )


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


def test_dogface_adapter_preserves_authenticated_publisher_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(tmp_path, monkeypatch)

    samples = adapt_dogfacenet224(tmp_path)

    assert [sample.raw_identity_id for sample in samples] == ["1", "1", "2"]
    assert [sample.split_role for sample in samples] == ["train", "train", "test"]


def test_dogface_adapter_accepts_explicit_publisher_class_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    class_root = tmp_path / "publisher"
    class_root.mkdir()
    _write_dogface_fixture(dataset_root, monkeypatch)
    train_path = class_root / "classes_train.txt"
    test_path = class_root / "classes_test.txt"
    (dataset_root / "classes_train.txt").rename(train_path)
    (dataset_root / "classes_test.txt").rename(test_path)

    samples = adapt_dogfacenet224(
        dataset_root,
        classes_train_path=train_path,
        classes_test_path=test_path,
    )

    assert [sample.split_role for sample in samples] == ["train", "train", "test"]


def test_dogface_adapter_rejects_partial_explicit_class_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="must be provided together"):
        adapt_dogfacenet224(
            tmp_path,
            classes_train_path=tmp_path / "classes_train.txt",
        )


@pytest.mark.parametrize("missing_name", ("classes_train.txt", "classes_test.txt"))
def test_dogface_adapter_requires_both_publisher_class_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
) -> None:
    _write_dogface_fixture(tmp_path, monkeypatch)
    (tmp_path / missing_name).unlink()

    with pytest.raises(ValueError, match=missing_name):
        adapt_dogfacenet224(tmp_path)


def test_dogface_adapter_rejects_unauthenticated_class_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(tmp_path, monkeypatch)
    (tmp_path / "classes_train.txt").write_text("1\n", encoding="ascii")

    with pytest.raises(ValueError, match="class-file hash differs"):
        adapt_dogfacenet224(tmp_path)


def test_dogface_adapter_rejects_contradictory_identity_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(
        tmp_path,
        monkeypatch,
        train_values=(1,),
        test_values=(1,),
    )

    with pytest.raises(ValueError, match="identity partition differs"):
        adapt_dogfacenet224(tmp_path)


def test_dogface_adapter_rejects_extracted_image_multiplicity_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(tmp_path, monkeypatch)
    Image.new("RGB", (12, 8)).save(tmp_path / "after_4_bis" / "1" / "1.2.jpg")

    with pytest.raises(ValueError, match="multiplicities differ"):
        adapt_dogfacenet224(tmp_path)


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


def test_mpdd_adapter_parses_compact_camera_sequence_tokens(tmp_path: Path) -> None:
    image_root = tmp_path / "MPDD" / "pytorch" / "query"
    image_root.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(image_root / "100_c1s6_1.jpg")

    sample = adapt_mpdd(tmp_path)[0]

    assert sample.capture_group_kind is CaptureGroupKind.POSE_VIEW_CLUSTER
    assert sample.raw_identity_id == "100"
    assert sample.capture_group_id == "100\0c1\0s6"
    assert sample.camera_id is None
    assert sample.label_availability["camera"] is False
    assert sample.metadata == {
        "unverified_camera_token": "c1",
        "unverified_sequence_token": "s6",
    }


def test_mpdd_adapter_accepts_legacy_separator_but_rejects_unknown_schema(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "MPDD" / "pytorch" / "train"
    image_root.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(image_root / "100_c1_s6_1.jpg")

    legacy_sample = adapt_mpdd(tmp_path)[0]

    assert legacy_sample.metadata["unverified_camera_token"] == "c1"
    assert legacy_sample.metadata["unverified_sequence_token"] == "s6"
    (image_root / "100_c1_s6_1.jpg").rename(image_root / "100_camera1_seq6_1.jpg")

    with pytest.raises(ValueError, match="filename schema differs"):
        adapt_mpdd(tmp_path)


def _write_oxford_sample(
    root: Path,
    stem: str,
    *,
    head_xml: bool = False,
) -> None:
    image_dir = root / "images"
    trimap_dir = root / "annotations" / "trimaps"
    xml_dir = root / "annotations" / "xmls"
    image_dir.mkdir(parents=True, exist_ok=True)
    trimap_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30)).save(image_dir / f"{stem}.jpg")
    Image.new("L", (40, 30), color=1).save(trimap_dir / f"{stem}.png")
    if head_xml:
        (xml_dir / f"{stem}.xml").write_text(
            "<annotation>"
            f"<filename>{stem}.jpg</filename>"
            "<object><name>dog</name><pose>Frontal</pose>"
            "<truncated>0</truncated><occluded>0</occluded><difficult>0</difficult>"
            "<bndbox><xmin>2</xmin><ymin>3</ymin><xmax>20</xmax>"
            "<ymax>22</ymax></bndbox></object></annotation>",
            encoding="utf-8",
        )


def test_oxford_adapter_preserves_dog_splits_trimap_and_head_without_identity(
    tmp_path: Path,
) -> None:
    annotation_root = tmp_path / "annotations"
    annotation_root.mkdir()
    _write_oxford_sample(tmp_path, "american_bulldog_1", head_xml=True)
    _write_oxford_sample(tmp_path, "beagle_201")
    (annotation_root / "trainval.txt").write_text(
        "Abyssinian_1 1 1 1\namerican_bulldog_1 2 2 1\n",
        encoding="utf-8",
    )
    (annotation_root / "test.txt").write_text(
        "beagle_201 3 2 2\n", encoding="utf-8"
    )

    samples = adapt_oxford_pets_dog(tmp_path)

    assert [sample.split_role for sample in samples] == ["trainval", "test"]
    assert [sample.breed for sample in samples] == ["american_bulldog", "beagle"]
    assert samples[0].head_roi_xyxy == (2.0, 3.0, 20.0, 22.0)
    assert samples[0].metadata["head_pose"] == "Frontal"
    assert samples[1].head_roi_xyxy is None
    assert all(sample.foreground_mask_path for sample in samples)
    assert all(sample.raw_identity_id is None for sample in samples)
    assert all(sample.registered_identity_id is None for sample in samples)
    assert all(sample.label_availability["identity"] is False for sample in samples)


def test_oxford_adapter_fails_closed_when_required_trimap_is_missing(
    tmp_path: Path,
) -> None:
    annotation_root = tmp_path / "annotations"
    annotation_root.mkdir()
    _write_oxford_sample(tmp_path, "beagle_1")
    (annotation_root / "trimaps" / "beagle_1.png").unlink()
    (annotation_root / "trainval.txt").write_text(
        "beagle_1 3 2 2\n", encoding="utf-8"
    )
    (annotation_root / "test.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="not a regular file"):
        adapt_oxford_pets_dog(tmp_path)


def test_petface_intake_uses_official_archive_metadata_and_is_bounded(
    tmp_path: Path,
) -> None:
    root = tmp_path / "petface"
    _write_petface_fixture(root)

    rows = load_petface_dog_split(root, "train", maximum_samples=1)
    samples = adapt_petface_dog(root, split_role="train", maximum_samples=1)

    assert len(rows) == len(samples) == 1
    assert rows[0].raw_identity_id == "000007"
    assert samples[0].raw_identity_id == "000007"
    assert samples[0].metadata["source_intake_only"] is True
    assert samples[0].image_path == "dog-001.tar.gz::dog/000007/00.png"
    assert "petface-dog" not in ADAPTERS
    assert RESEARCH_INTAKE_ADAPTERS["petface-dog"] is adapt_petface_dog


def test_petface_training_dataset_fails_closed_while_source_is_blocked(
    tmp_path: Path,
) -> None:
    root = tmp_path / "petface"
    root.mkdir()
    (root / "dog" / "fabricated-id").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(root / "dog" / "fabricated-id" / "image.jpg")

    with pytest.raises(RuntimeError, match="blocked by the source admission registry"):
        PetFaceDataset(root)


def test_petface_training_dataset_cannot_bypass_block_with_valid_archives(
    tmp_path: Path,
) -> None:
    root = tmp_path / "petface"
    _write_petface_fixture(root)
    with pytest.raises(RuntimeError, match="blocked by the source admission registry"):
        PetFaceDataset(
            root,
            transform=lambda image: image.size,
            split="train",
            maximum_samples=2,
        )


def test_petface_intake_rejects_label_and_path_disagreement(tmp_path: Path) -> None:
    root = tmp_path / "petface"
    _write_petface_fixture(root, train_rows="filename,label\ndog/000007/00.png,8\n")

    with pytest.raises(ValueError, match="identity differs"):
        load_petface_dog_split(root, "train")


def test_petface_intake_requires_local_license_metadata(tmp_path: Path) -> None:
    root = tmp_path / "petface"
    _write_petface_fixture(root, readme="# PetFace\nNo terms here.\n")

    with pytest.raises(ValueError, match="license metadata differs"):
        load_petface_dog_split(root, "train")


def test_petface_intake_rejects_non_integer_sample_bound(tmp_path: Path) -> None:
    root = tmp_path / "petface"
    _write_petface_fixture(root)

    with pytest.raises(ValueError, match="positive integer"):
        load_petface_dog_split(root, "train", maximum_samples=1.5)  # type: ignore[arg-type]
