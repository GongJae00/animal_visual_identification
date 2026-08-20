from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from data.adapters.io import file_sha256
from data.adapters.mpdd import adapt_mpdd
from data.types import CaptureGroupKind
from shared.contracts.identity_ids import (
    compute_registered_dog_id,
    compute_sample_token,
)


def _write_jpg(path: Path, size: tuple[int, int] = (32, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def _pytorch(root: Path) -> Path:
    return root / "MPDD" / "pytorch"


def test_adapt_mpdd_missing_base_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MPDD base not found"):
        adapt_mpdd(tmp_path)
    (tmp_path / "MPDD").mkdir()
    with pytest.raises(FileNotFoundError, match="MPDD base not found"):
        adapt_mpdd(tmp_path)


def test_adapt_mpdd_skips_absent_known_splits_and_unknown_folders(
    tmp_path: Path,
) -> None:
    image = _pytorch(tmp_path) / "gallery" / "0_c2s2_6.jpg"
    _write_jpg(image)
    extra = _pytorch(tmp_path) / "extra" / "9_c1s1_1.jpg"
    _write_jpg(extra)
    nested = _pytorch(tmp_path) / "query" / "nested" / "1_c1s1_1.jpg"
    _write_jpg(nested)
    sidecar = _pytorch(tmp_path) / "gallery" / "notes.txt"
    sidecar.write_text("not an image", encoding="utf-8")
    (_pytorch(tmp_path) / "train").mkdir()

    samples = adapt_mpdd(tmp_path)

    assert len(samples) == 1
    assert samples[0].split_role == "gallery"
    assert samples[0].raw_identity_id == "0"
    assert samples[0].image_path == "MPDD/pytorch/gallery/0_c2s2_6.jpg"


def test_adapt_mpdd_walks_publisher_splits_in_declared_order(
    tmp_path: Path,
) -> None:
    _write_jpg(_pytorch(tmp_path) / "gallery" / "2_c1s1_1.jpg")
    _write_jpg(_pytorch(tmp_path) / "query" / "2_c2s1_1.jpg")
    _write_jpg(_pytorch(tmp_path) / "val" / "1_c2s1_1.jpg")
    _write_jpg(_pytorch(tmp_path) / "train" / "1_c1s1_1.jpg")

    samples = adapt_mpdd(tmp_path)

    assert [sample.split_role for sample in samples] == [
        "train",
        "val",
        "query",
        "gallery",
    ]
    assert [sample.raw_identity_id for sample in samples] == ["1", "1", "2", "2"]
    assert {sample.raw_identity_id for sample in samples}.isdisjoint(
        {"train", "val", "query", "gallery", "pytorch", "MPDD"}
    )


def test_adapt_mpdd_hashes_dims_and_identity_tokens(tmp_path: Path) -> None:
    image = _pytorch(tmp_path) / "train" / "12_c3s4_8.jpg"
    _write_jpg(image, size=(40, 30))

    sample = adapt_mpdd(tmp_path)[0]

    assert sample.dataset_name == "mpdd"
    assert sample.dataset_version == "mendeley-v5j6m8dzhv-v1"
    assert sample.source_group_id == "12"
    assert sample.registered_identity_id == compute_registered_dog_id(
        "mpdd:v1:device-capture:12"
    )
    assert sample.sample_id == compute_sample_token("mpdd:12_c3s4_8:train")
    assert sample.capture_group_kind is CaptureGroupKind.POSE_VIEW_CLUSTER
    assert sample.capture_group_id == "12\0c3\0s4"
    assert sample.camera_id is None
    assert sample.label_availability["camera"] is False
    assert sample.metadata == {
        "unverified_camera_token": "c3",
        "unverified_sequence_token": "s4",
    }
    assert sample.width == 40
    assert sample.height == 30
    assert sample.image_sha256 == file_sha256(image)


def test_adapt_mpdd_sample_token_includes_split_role(tmp_path: Path) -> None:
    _write_jpg(_pytorch(tmp_path) / "query" / "7_c1s1_1.jpg")
    _write_jpg(_pytorch(tmp_path) / "gallery" / "7_c1s1_1.jpg")

    samples = adapt_mpdd(tmp_path)

    assert len(samples) == 2
    assert samples[0].sample_id != samples[1].sample_id
    assert samples[0].registered_identity_id == samples[1].registered_identity_id
    assert {sample.split_role for sample in samples} == {"query", "gallery"}


def test_adapt_mpdd_rejects_non_numeric_identity_and_missing_frame(
    tmp_path: Path,
) -> None:
    bad = _pytorch(tmp_path) / "train" / "dog_c1s6_1.jpg"
    _write_jpg(bad)
    with pytest.raises(ValueError, match="filename schema differs"):
        adapt_mpdd(tmp_path)
    bad.rename(_pytorch(tmp_path) / "train" / "100_c1s6.jpg")
    with pytest.raises(ValueError, match="filename schema differs"):
        adapt_mpdd(tmp_path)


def test_adapt_mpdd_rejects_symlink_image(tmp_path: Path) -> None:
    target = tmp_path / "outside.jpg"
    Image.new("RGB", (32, 24)).save(target)
    link = _pytorch(tmp_path) / "val" / "3_c1s1_1.jpg"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)

    with pytest.raises(ValueError, match="not a regular file"):
        adapt_mpdd(tmp_path)
