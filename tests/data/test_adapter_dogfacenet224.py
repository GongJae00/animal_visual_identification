from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from data.adapters.dogfacenet224 import adapt_dogfacenet224
from data.types import CaptureGroupKind
from shared.contracts.identity_ids import (
    compute_registered_dog_id,
    compute_sample_token,
)


def _write_dogface_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_values: tuple[int, ...] = (1, 1),
    test_values: tuple[int, ...] = (2,),
    filenames: dict[str, tuple[str, ...]] | None = None,
) -> None:
    image_root = root / "after_4_bis"
    counts = Counter(train_values + test_values)
    for identity, count in counts.items():
        identity_root = image_root / str(identity)
        identity_root.mkdir(parents=True)
        names = (
            filenames[str(identity)]
            if filenames is not None and str(identity) in filenames
            else tuple(f"{identity}.{index}.jpg" for index in range(count))
        )
        if len(names) != count:
            raise AssertionError("fixture filenames must match class-file multiplicity")
        for name in names:
            Image.new("RGB", (12, 8)).save(identity_root / name)
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


def test_identity_is_web_album_folder_not_filename_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(
        tmp_path,
        monkeypatch,
        filenames={"1": ("1.0.jpg", "480.0.jpg")},
    )

    samples = adapt_dogfacenet224(tmp_path)

    assert [sample.raw_identity_id for sample in samples] == ["1", "1", "2"]
    assert [sample.source_group_id for sample in samples] == ["1", "1", "2"]
    assert [sample.capture_group_id for sample in samples] == ["1", "1", "2"]
    assert {sample.dataset_name for sample in samples} == {"dogfacenet224"}
    assert {sample.dataset_version for sample in samples} == {"zenodo-12578449-v1"}
    assert {sample.capture_group_kind for sample in samples} == {
        CaptureGroupKind.ALBUM_OR_SOURCE_GROUP
    }
    expected_reg = compute_registered_dog_id("dogfacenet224:v1:web-folder:1")
    assert samples[0].registered_identity_id == expected_reg
    assert samples[1].registered_identity_id == expected_reg
    assert samples[0].sample_id == compute_sample_token("dogfacenet224:1:1.0")
    assert samples[1].sample_id == compute_sample_token("dogfacenet224:1:480.0")
    assert samples[1].image_path == str(Path("after_4_bis") / "1" / "480.0.jpg")


def test_reads_dogface_hashes_from_data_adapters_at_call_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "data.public_sources.public_canine_manifest.DOGFACE_TRAIN_SHA256",
        "0" * 64,
    )
    monkeypatch.setattr(
        "data.public_sources.public_canine_manifest.DOGFACE_TRAIN_MD5",
        "0" * 32,
    )

    samples = adapt_dogfacenet224(tmp_path)

    assert len(samples) == 3


def test_missing_after_4_bis_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="DogFaceNet base not found"):
        adapt_dogfacenet224(tmp_path)


def test_non_digit_identity_folder_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(tmp_path, monkeypatch)
    notes = tmp_path / "after_4_bis" / "notes"
    notes.mkdir()
    Image.new("RGB", (12, 8)).save(notes / "1.0.jpg")

    with pytest.raises(ValueError, match="identity folder differs"):
        adapt_dogfacenet224(tmp_path)


def test_identity_folder_absent_from_class_files_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(tmp_path, monkeypatch)
    extra = tmp_path / "after_4_bis" / "3"
    extra.mkdir()
    Image.new("RGB", (12, 8)).save(extra / "3.0.jpg")

    with pytest.raises(ValueError, match="identity partition differs"):
        adapt_dogfacenet224(tmp_path)


def test_class_file_identity_missing_from_disk_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(
        tmp_path,
        monkeypatch,
        train_values=(1, 1),
        test_values=(2, 3),
    )
    missing = tmp_path / "after_4_bis" / "3"
    for image in missing.iterdir():
        image.unlink()
    missing.rmdir()

    with pytest.raises(ValueError, match="identity partition differs"):
        adapt_dogfacenet224(tmp_path)


def test_empty_identity_folder_and_non_image_files_are_not_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(tmp_path, monkeypatch)
    (tmp_path / "after_4_bis" / "99").mkdir()
    (tmp_path / "after_4_bis" / "1" / "readme.txt").write_text("x", encoding="ascii")
    (tmp_path / "after_4_bis" / "ignored.txt").write_text("x", encoding="ascii")

    samples = adapt_dogfacenet224(tmp_path)

    assert [sample.raw_identity_id for sample in samples] == ["1", "1", "2"]


def test_explicit_class_path_must_be_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dogface_fixture(tmp_path, monkeypatch)
    directory = tmp_path / "not-a-file"
    directory.mkdir()

    with pytest.raises(ValueError, match="DogFace train class file"):
        adapt_dogfacenet224(
            tmp_path,
            classes_train_path=directory,
            classes_test_path=tmp_path / "classes_test.txt",
        )
