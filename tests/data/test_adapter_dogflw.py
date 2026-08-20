from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from data.adapters.dogflw import adapt_dogflw
from data.types import CaptureGroupKind
from shared.contracts.identity_ids import compute_sample_token


def _landmarks() -> list[list[float]]:
    return [[float(index), float(index + 1)] for index in range(46)]


def _write_pair(
    root: Path,
    split: str,
    stem: str,
    *,
    landmarks: object | None = None,
    bbox: object | None = None,
    size: tuple[int, int] = (32, 24),
    write_label: bool = True,
) -> Path:
    image_path = root / "DogFLW" / split / "images" / f"{stem}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(image_path)
    if write_label:
        label_path = root / "DogFLW" / split / "labels" / f"{stem}.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "landmarks": _landmarks() if landmarks is None else landmarks,
            "bounding_boxes": [1, 2, 20, 22] if bbox is None else bbox,
        }
        label_path.write_text(json.dumps(payload), encoding="utf-8")
    return image_path


def test_adapter_requires_publisher_dogflw_base(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="DogFLW base not found"):
        adapt_dogflw(tmp_path)


def test_adapter_fails_closed_when_label_is_missing(tmp_path: Path) -> None:
    _write_pair(tmp_path, "train", "n02085620_1", write_label=False)

    with pytest.raises(FileNotFoundError, match="DogFLW label missing"):
        adapt_dogflw(tmp_path)


@pytest.mark.parametrize(
    ("payload", "match"),
    (
        ({"landmarks": _landmarks()[:45], "bounding_boxes": [1, 2, 20, 22]}, "face46"),
        ({"landmarks": _landmarks(), "bounding_boxes": [1, 2, 20]}, "face bbox"),
        (
            {"landmarks": [[0.0]] + _landmarks()[1:], "bounding_boxes": [1, 2, 20, 22]},
            "landmark point",
        ),
    ),
)
def test_adapter_fails_closed_on_malformed_label(
    tmp_path: Path,
    payload: dict[str, object],
    match: str,
) -> None:
    image_dir = tmp_path / "DogFLW" / "train" / "images"
    label_dir = tmp_path / "DogFLW" / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(image_dir / "n02085620_1.png")
    (label_dir / "n02085620_1.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=match):
        adapt_dogflw(tmp_path)


def test_adapter_preserves_splits_without_dog_identity(tmp_path: Path) -> None:
    train_image = _write_pair(tmp_path, "train", "n02085620_11477")
    _write_pair(tmp_path, "test", "n02088094_7894", bbox=["", "", "", ""])
    (tmp_path / "DogFLW" / "Ear Types.docx").write_bytes(b"not-an-identity")
    Image.new("RGB", (8, 8)).save(
        tmp_path / "DogFLW" / "train" / "images" / "n02085620_ignored.jpg"
    )

    samples = adapt_dogflw(tmp_path)

    assert [sample.split_role for sample in samples] == ["train", "test"]
    assert [sample.breed for sample in samples] == ["n02085620", "n02088094"]
    assert [sample.raw_identity_id for sample in samples] == [None, None]
    assert [sample.registered_identity_id for sample in samples] == [None, None]
    assert [sample.label_availability["identity"] for sample in samples] == [
        False,
        False,
    ]
    assert samples[0].source_group_id == "n02085620_11477"
    assert samples[0].capture_group_id == "n02085620_11477"
    assert samples[0].capture_group_kind is CaptureGroupKind.ALBUM_OR_SOURCE_GROUP
    assert samples[0].dataset_name == "dogflw"
    assert samples[0].dataset_version == "kaggle-2025-07-02"
    assert samples[0].sample_id == compute_sample_token(
        "dogflw:train:n02085620_11477"
    )
    assert samples[0].image_path == "DogFLW/train/images/n02085620_11477.png"
    assert samples[0].width == 32
    assert samples[0].height == 24
    assert samples[0].image_sha256 == hashlib.sha256(train_image.read_bytes()).hexdigest()
    assert samples[0].face_box_xyxy == (1.0, 2.0, 20.0, 22.0)
    assert samples[1].face_box_xyxy is None
    assert len(samples[0].face_landmarks or {}) == 46


def test_adapter_rejects_symlinked_crop(tmp_path: Path) -> None:
    real = tmp_path / "outside.png"
    Image.new("RGB", (8, 8)).save(real)
    image_dir = tmp_path / "DogFLW" / "train" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "n02085620_1.png").symlink_to(real)

    with pytest.raises(ValueError, match="regular file"):
        adapt_dogflw(tmp_path)
