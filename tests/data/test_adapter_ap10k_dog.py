from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from data.adapters.ap10k_dog import adapt_ap10k_dog
from data.adapters.io import file_sha256
from data.types import CaptureGroupKind
from shared.contracts.identity_ids import compute_sample_token

_DOG = 8
_FOX = 9
_WOLF = 10
_ARCTIC_FOX = 7


def _keypoints(visibilities: tuple[int, ...] = (2,) * 17) -> list[float]:
    points: list[float] = []
    for index, visibility in enumerate(visibilities):
        points.extend((float(index), float(index + 1), float(visibility)))
    return points


def _annotation(
    annotation_id: int,
    *,
    image_id: int = 1,
    category_id: int = _DOG,
    bbox: tuple[float, float, float, float] = (1.0, 2.0, 20.0, 15.0),
    keypoints: list[float] | None = None,
) -> dict[str, object]:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "keypoints": _keypoints() if keypoints is None else keypoints,
    }


def _write_jpeg(path: Path, size: tuple[int, int] = (40, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def _write_split(
    root: Path,
    split: str,
    fold: int,
    *,
    images: list[dict[str, object]],
    annotations: list[dict[str, object]],
) -> None:
    annotation_dir = root / "ap-10k" / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / f"ap10k-{split}-split{fold}.json").write_text(
        json.dumps({"images": images, "annotations": annotations}),
        encoding="utf-8",
    )


def _write_split1_tree(
    root: Path,
    *,
    images: list[dict[str, object]] | None = None,
    annotations: list[dict[str, object]] | None = None,
    file_name: str = "000000000001.jpg",
    size: tuple[int, int] = (40, 30),
) -> Path:
    image_path = root / "ap-10k" / "data" / file_name
    _write_jpeg(image_path, size)
    payload_images = images or [{"id": 1, "file_name": file_name, "width": 999, "height": 888}]
    payload_annotations = annotations or [_annotation(11)]
    for split in ("train", "val", "test"):
        _write_split(
            root,
            split,
            1,
            images=payload_images,
            annotations=payload_annotations,
        )
    return image_path


def test_adapter_requires_extracted_ap10k_tree_not_inner_folder_or_zip(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="AP-10K base not found"):
        adapt_ap10k_dog(tmp_path)

    inner = tmp_path / "ap-10k"
    (inner / "annotations").mkdir(parents=True)
    (inner / "data").mkdir()
    (tmp_path / "ap-10k.zip").write_bytes(b"not-used")
    with pytest.raises(FileNotFoundError, match="AP-10K base not found"):
        adapt_ap10k_dog(inner)

    (inner / "data").rmdir()
    with pytest.raises(FileNotFoundError, match="AP-10K base not found"):
        adapt_ap10k_dog(tmp_path)


def test_adapter_requires_official_split1_json(tmp_path: Path) -> None:
    _write_split1_tree(tmp_path)
    (tmp_path / "ap-10k" / "annotations" / "ap10k-val-split1.json").unlink()
    _write_split(
        tmp_path,
        "val",
        2,
        images=[{"id": 1, "file_name": "000000000001.jpg"}],
        annotations=[_annotation(99)],
    )

    with pytest.raises(FileNotFoundError):
        adapt_ap10k_dog(tmp_path)


def test_adapter_reads_split1_only(tmp_path: Path) -> None:
    _write_split1_tree(
        tmp_path,
        annotations=[_annotation(11)],
    )
    for split in ("train", "val", "test"):
        _write_split(
            tmp_path,
            split,
            2,
            images=[{"id": 1, "file_name": "000000000001.jpg"}],
            annotations=[_annotation(901)],
        )
        _write_split(
            tmp_path,
            split,
            3,
            images=[{"id": 1, "file_name": "000000000001.jpg"}],
            annotations=[_annotation(801)],
        )

    samples = adapt_ap10k_dog(tmp_path)

    assert [sample.split_role for sample in samples] == ["train", "val", "test"]
    assert [sample.metadata["annotation_id"] for sample in samples] == [11, 11, 11]
    assert {sample.sample_id for sample in samples} == {
        compute_sample_token(f"ap10k:split1:{split}:11")
        for split in ("train", "val", "test")
    }


def test_adapter_keeps_dog_category_only(tmp_path: Path) -> None:
    _write_split1_tree(
        tmp_path,
        annotations=[
            _annotation(1, category_id=_ARCTIC_FOX),
            _annotation(2, category_id=_FOX),
            _annotation(3, category_id=_WOLF),
            _annotation(4, category_id=1),
            _annotation(5, category_id=_DOG, bbox=(4.0, 5.0, 8.0, 9.0)),
        ],
    )

    samples = adapt_ap10k_dog(tmp_path)

    assert len(samples) == 3
    assert all(sample.metadata["annotation_id"] == 5 for sample in samples)
    assert samples[0].dog_boxes_xyxy == (4.0, 5.0, 12.0, 14.0)


def test_adapter_has_no_identity_and_groups_instances_by_image(
    tmp_path: Path,
) -> None:
    _write_jpeg(tmp_path / "ap-10k" / "data" / "000000000001.jpg")
    _write_jpeg(tmp_path / "ap-10k" / "data" / "000000000002.jpg")
    images = [
        {"id": 1, "file_name": "000000000001.jpg"},
        {"id": 2, "file_name": "000000000002.jpg"},
    ]
    annotations = [
        _annotation(20, image_id=1),
        _annotation(10, image_id=1, bbox=(3.0, 4.0, 5.0, 6.0)),
        _annotation(30, image_id=2),
    ]
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, 1, images=images, annotations=annotations)

    train = [sample for sample in adapt_ap10k_dog(tmp_path) if sample.split_role == "train"]

    assert [sample.metadata["annotation_id"] for sample in train] == [10, 20, 30]
    assert [sample.source_group_id for sample in train] == [
        "ap10k-image:1",
        "ap10k-image:1",
        "ap10k-image:2",
    ]
    for sample in train:
        assert sample.dataset_name == "ap10k-dog"
        assert sample.dataset_version == "official-split1-2021-11-01"
        assert sample.raw_identity_id is None
        assert sample.registered_identity_id is None
        assert sample.generated_identity_id is None
        assert sample.capture_group_id is None
        assert sample.capture_group_kind is CaptureGroupKind.UNKNOWN
        assert sample.label_availability["identity"] is False


def test_adapter_omits_unlabeled_keypoints_and_scales_visibility(
    tmp_path: Path,
) -> None:
    visibilities = (0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    _write_split1_tree(
        tmp_path,
        annotations=[_annotation(11, keypoints=_keypoints(visibilities))],
    )

    sample = adapt_ap10k_dog(tmp_path)[0]

    assert sample.body_keypoints == {
        "right_eye": (1.0, 2.0, 0.5),
        "neck": (3.0, 4.0, 1.0),
    }
    assert "left_eye" not in sample.body_keypoints
    assert "nose_center" not in sample.body_keypoints
    assert "tail_base" not in sample.body_keypoints


def test_adapter_uses_image_bytes_for_dims_and_hash(tmp_path: Path) -> None:
    image_path = _write_split1_tree(tmp_path, size=(40, 30))

    sample = adapt_ap10k_dog(tmp_path)[0]

    assert sample.width == 40
    assert sample.height == 30
    assert sample.image_sha256 == file_sha256(image_path)
    assert sample.image_path == "ap-10k/data/000000000001.jpg"
    assert sample.dog_boxes_xyxy == (1.0, 2.0, 21.0, 17.0)


def test_adapter_rejects_missing_image(tmp_path: Path) -> None:
    _write_split1_tree(tmp_path)
    (tmp_path / "ap-10k" / "data" / "000000000001.jpg").unlink()

    with pytest.raises(ValueError, match="not a regular file"):
        adapt_ap10k_dog(tmp_path)
