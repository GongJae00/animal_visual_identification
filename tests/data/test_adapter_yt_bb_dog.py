from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from data.adapters.io import _file_sha256
from data.adapters.yt_bb_dog import adapt_yt_bb_dog
from data.types import CaptureGroupKind
from shared.contracts.identity_ids import (
    compute_registered_dog_id,
    compute_sample_token,
)


def _write_image(path: Path, size: tuple[int, int] = (12, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}[path.suffix.lower()]
    Image.new("RGB", size).save(path, format=fmt)


def _publisher_base(root: Path) -> Path:
    return root / "YT-BB-dog" / "YT-BB-Dog"


def test_yt_bb_dog_adapter_uses_official_split_folders_and_video_track_identity(
    tmp_path: Path,
) -> None:
    base = _publisher_base(tmp_path)
    _write_image(base / "train" / "0" / "0_1.jpg", (16, 10))
    _write_image(base / "train" / "0" / "0_0.jpg", (16, 10))
    _write_image(base / "train" / "1" / "1_0.jpg", (8, 6))
    _write_image(base / "test" / "2000" / "2000_0.jpg", (20, 12))

    samples = adapt_yt_bb_dog(tmp_path)

    assert [sample.raw_identity_id for sample in samples] == ["0", "0", "1", "2000"]
    assert [sample.split_role for sample in samples] == [
        "train",
        "train",
        "train",
        "test",
    ]
    assert [sample.image_path for sample in samples] == [
        "YT-BB-dog/YT-BB-Dog/train/0/0_0.jpg",
        "YT-BB-dog/YT-BB-Dog/train/0/0_1.jpg",
        "YT-BB-dog/YT-BB-Dog/train/1/1_0.jpg",
        "YT-BB-dog/YT-BB-Dog/test/2000/2000_0.jpg",
    ]
    assert all(sample.dataset_name == "yt-bb-dog" for sample in samples)
    assert all(
        sample.dataset_version == "publisher-v1-2025-10-27" for sample in samples
    )
    assert all(
        sample.capture_group_kind is CaptureGroupKind.VIDEO_TRACK for sample in samples
    )
    assert {sample.registered_identity_id for sample in samples} == {
        compute_registered_dog_id("yt-bb-dog:v1:video-track:0"),
        compute_registered_dog_id("yt-bb-dog:v1:video-track:1"),
        compute_registered_dog_id("yt-bb-dog:v1:video-track:2000"),
    }
    assert samples[0].sample_id == compute_sample_token("yt-bb-dog:0:train:0_0")
    assert samples[0].source_group_id == "0"
    assert samples[0].capture_group_id == "0"
    assert samples[0].width == 16
    assert samples[0].height == 10
    assert samples[0].image_sha256 == _file_sha256(base / "train" / "0" / "0_0.jpg")
    train_ids = {
        sample.raw_identity_id for sample in samples if sample.split_role == "train"
    }
    test_ids = {
        sample.raw_identity_id for sample in samples if sample.split_role == "test"
    }
    assert train_ids == {"0", "1"}
    assert test_ids == {"2000"}


def test_yt_bb_dog_adapter_requires_publisher_base(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="YT-BB-Dog base not found"):
        adapt_yt_bb_dog(tmp_path)
    (tmp_path / "YT-BB-dog").mkdir()
    with pytest.raises(FileNotFoundError, match="YT-BB-Dog base not found"):
        adapt_yt_bb_dog(tmp_path)


def test_yt_bb_dog_adapter_skips_missing_official_split(tmp_path: Path) -> None:
    base = _publisher_base(tmp_path)
    _write_image(base / "train" / "0" / "0_0.jpg")

    samples = adapt_yt_bb_dog(tmp_path)

    assert len(samples) == 1
    assert samples[0].split_role == "train"


def test_yt_bb_dog_adapter_ignores_non_identities_and_non_official_trees(
    tmp_path: Path,
) -> None:
    base = _publisher_base(tmp_path)
    _write_image(base / "train" / "0" / "0_0.jpg")
    (base / "train" / "0" / "notes.txt").write_text("not an image", encoding="utf-8")
    (base / "train" / "readme.txt").write_text("not an identity", encoding="utf-8")
    _write_image(base / "val" / "9" / "9_0.jpg")
    (tmp_path / "YT-BB-dog" / "YT-BB-Dog.zip").write_bytes(b"archive")
    (tmp_path / "YT-BB-dog" / "YT-BB-Dog_random_bckg.zip").write_bytes(b"archive")
    _write_image(
        tmp_path
        / "YT-BB-dog"
        / "YT-BB-Dog_random_bckg"
        / "YT-BB-Dog"
        / "test"
        / "2000"
        / "2000_0.jpg"
    )
    _write_image(base / "train" / "0" / "0_1.jpeg")
    _write_image(base / "train" / "0" / "0_2.png")

    samples = adapt_yt_bb_dog(tmp_path)

    assert [sample.split_role for sample in samples] == ["train", "train", "train"]
    assert {sample.raw_identity_id for sample in samples} == {"0"}
    assert {Path(sample.image_path).name for sample in samples} == {
        "0_0.jpg",
        "0_1.jpeg",
        "0_2.png",
    }
    assert all("random_bckg" not in sample.image_path for sample in samples)
    assert all("/val/" not in sample.image_path for sample in samples)


def test_yt_bb_dog_adapter_rejects_symlinked_images(tmp_path: Path) -> None:
    base = _publisher_base(tmp_path)
    real = tmp_path / "outside.jpg"
    _write_image(real)
    link = base / "train" / "0" / "0_0.jpg"
    link.parent.mkdir(parents=True)
    link.symlink_to(real)

    with pytest.raises(ValueError, match="regular file"):
        adapt_yt_bb_dog(tmp_path)
