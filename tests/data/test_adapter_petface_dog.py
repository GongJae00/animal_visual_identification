from __future__ import annotations

import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from data.adapters.petface_dog import (
    adapt_petface_dog,
    load_petface_dog_split,
    read_petface_dog_images,
)
from data.types import CaptureGroupKind

_PETFACE_README = """# PetFace Dataset
This dataset is provided for research purposes only.
You are prohibited from redistributing or sharing this dataset.
"""


def _png_bytes(size: tuple[int, int] = (12, 8)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size).save(stream, format="PNG")
    return stream.getvalue()


def _write_zip(root: Path, name: str, members: dict[str, bytes | str]) -> None:
    with zipfile.ZipFile(root / name, "w") as archive:
        for member_name, payload in members.items():
            data = payload.encode("utf-8") if isinstance(payload, str) else payload
            archive.writestr(member_name, data)


def _write_tar(root: Path, members: tuple[str, ...] = ("dog/000007/00.png",)) -> None:
    payload = _png_bytes()
    with tarfile.open(root / "dog-001.tar.gz", "w:gz") as archive:
        for member_name in members:
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _write_root(
    root: Path,
    *,
    train_rows: str = "filename,label\ndog/000007/00.png,7\n",
    readme: str = _PETFACE_README,
    zip_members: dict[str, bytes | str] | None = None,
    tar_members: tuple[str, ...] | None = ("dog/000007/00.png",),
    extra_zips: dict[str, dict[str, bytes | str]] | None = None,
) -> Path:
    root.mkdir()
    members = zip_members or {
        "PetFace/README.md": readme,
        "PetFace/split/dog/train.csv": train_rows,
    }
    _write_zip(root, "metadata.zip", members)
    if extra_zips:
        for name, extra_members in extra_zips.items():
            _write_zip(root, name, extra_members)
    if tar_members is not None:
        _write_tar(root, tar_members)
    return root


def test_val_and_test_splits_use_filename_only_identity(tmp_path: Path) -> None:
    root = tmp_path / "petface"
    _write_root(
        root,
        zip_members={
            "PetFace/README.md": _PETFACE_README,
            "PetFace/split/dog/val.txt": "dog/000007/00.png\n",
            "PetFace/split/dog/test.txt": "dog/046755/01.png\n",
        },
        tar_members=None,
    )

    val_rows = load_petface_dog_split(root, "val")
    test_rows = load_petface_dog_split(root, "test")

    assert val_rows[0].archive_member == "dog/000007/00.png"
    assert val_rows[0].raw_identity_id == "000007"
    assert test_rows[0].raw_identity_id == "046755"


def test_reidentification_csv_uses_folder_id(tmp_path: Path) -> None:
    root = tmp_path / "petface"
    _write_root(
        root,
        zip_members={
            "PetFace/README.md": _PETFACE_README,
            "PetFace/split/dog/reidentification.csv": (
                "filename,label\ndog/000007/05.png,7\n"
            ),
        },
        tar_members=None,
    )

    rows = load_petface_dog_split(root, "reidentification")

    assert rows[0].archive_member == "dog/000007/05.png"
    assert rows[0].raw_identity_id == "000007"


@pytest.mark.parametrize(
    "split",
    ("verification", "dog", ""),
)
def test_load_rejects_unsupported_split(tmp_path: Path, split: str) -> None:
    root = _write_root(tmp_path / "petface", tar_members=None)

    with pytest.raises(ValueError, match="unsupported PetFace dog split"):
        load_petface_dog_split(root, split)


def test_load_rejects_missing_metadata_archives(tmp_path: Path) -> None:
    root = tmp_path / "petface"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="metadata archives not found"):
        load_petface_dog_split(root, "train")


def test_load_rejects_missing_archive_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="archive root not found"):
        load_petface_dog_split(tmp_path / "missing", "train")


def test_load_rejects_empty_split(tmp_path: Path) -> None:
    root = _write_root(
        tmp_path / "petface",
        train_rows="filename,label\n",
        tar_members=None,
    )

    with pytest.raises(ValueError, match="split is empty"):
        load_petface_dog_split(root, "train")


def test_load_rejects_duplicate_split_member(tmp_path: Path) -> None:
    root = _write_root(
        tmp_path / "petface",
        train_rows=(
            "filename,label\ndog/000007/00.png,7\ndog/000007/00.png,7\n"
        ),
        tar_members=None,
    )

    with pytest.raises(ValueError, match="repeats image member"):
        load_petface_dog_split(root, "train")


@pytest.mark.parametrize(
    "relative",
    (
        "cat/000007/00.png",
        "dog/000007/00.jpg",
        "dog/000007.png",
        "dog/id007/00.png",
        "../dog/000007/00.png",
        "dog/000007/../00.png",
        r"dog\000007\00.png",
        "/dog/000007/00.png",
    ),
)
def test_load_rejects_non_dog_identity_members(
    tmp_path: Path, relative: str
) -> None:
    root = _write_root(
        tmp_path / "petface",
        train_rows=f"filename,label\n{relative},7\n",
        tar_members=None,
    )

    with pytest.raises(ValueError, match="unsafe PetFace dog image member"):
        load_petface_dog_split(root, "train")


def test_load_rejects_split_header_and_column_count(tmp_path: Path) -> None:
    header_root = _write_root(
        tmp_path / "header",
        train_rows="file,id\ndog/000007/00.png,7\n",
        tar_members=None,
    )
    with pytest.raises(ValueError, match="split header differs"):
        load_petface_dog_split(header_root, "train")

    columns_root = _write_root(
        tmp_path / "columns",
        train_rows="filename,label\ndog/000007/00.png,7,extra\n",
        tar_members=None,
    )
    with pytest.raises(ValueError, match="has 3 columns"):
        load_petface_dog_split(columns_root, "train")


def test_load_rejects_nul_bytes_in_split(tmp_path: Path) -> None:
    root = _write_root(
        tmp_path / "petface",
        train_rows="filename,label\ndog/000007/00.png,7\x00\n",
        tar_members=None,
    )

    with pytest.raises(ValueError, match="NUL bytes"):
        load_petface_dog_split(root, "train")


@pytest.mark.parametrize("bound", (True, False, 0, -1))
def test_load_rejects_bool_and_nonpositive_sample_bound(
    tmp_path: Path, bound: object
) -> None:
    root = _write_root(tmp_path / "petface", tar_members=None)

    with pytest.raises(ValueError, match="positive integer"):
        load_petface_dog_split(root, "train", maximum_samples=bound)  # type: ignore[arg-type]


def test_load_rejects_duplicate_zip_member(tmp_path: Path) -> None:
    root = tmp_path / "petface"
    root.mkdir()
    with zipfile.ZipFile(root / "metadata.zip", "w") as archive:
        archive.writestr("PetFace/README.md", _PETFACE_README)
        archive.writestr("PetFace/README.md", _PETFACE_README)
        archive.writestr(
            "PetFace/split/dog/train.csv",
            "filename,label\ndog/000007/00.png,7\n",
        )

    with pytest.raises(ValueError, match="duplicated in metadata.zip"):
        load_petface_dog_split(root, "train")


def test_load_rejects_ambiguous_zip_member(tmp_path: Path) -> None:
    root = _write_root(
        tmp_path / "petface",
        extra_zips={
            "other.zip": {
                "PetFace/split/dog/train.csv": (
                    "filename,label\ndog/000008/00.png,8\n"
                )
            }
        },
        tar_members=None,
    )

    with pytest.raises(ValueError, match="ambiguous across archives"):
        load_petface_dog_split(root, "train")


def test_load_rejects_symlink_metadata_member(tmp_path: Path) -> None:
    root = tmp_path / "petface"
    root.mkdir()
    with zipfile.ZipFile(root / "metadata.zip", "w") as archive:
        info = zipfile.ZipInfo("PetFace/README.md")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, _PETFACE_README)
        archive.writestr(
            "PetFace/split/dog/train.csv",
            "filename,label\ndog/000007/00.png,7\n",
        )

    with pytest.raises(ValueError, match="unsafe PetFace metadata member"):
        load_petface_dog_split(root, "train")


def test_load_rejects_zip_archive_count_limit(tmp_path: Path) -> None:
    root = tmp_path / "petface"
    root.mkdir()
    for index in range(17):
        (root / f"{index:02d}.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    with pytest.raises(ValueError, match="archive count exceeds intake limit"):
        load_petface_dog_split(root, "train")


def test_read_rejects_empty_or_duplicate_request(tmp_path: Path) -> None:
    root = _write_root(tmp_path / "petface")

    with pytest.raises(ValueError, match="non-empty and unique"):
        read_petface_dog_images(root, ())
    with pytest.raises(ValueError, match="non-empty and unique"):
        read_petface_dog_images(
            root, ("dog/000007/00.png", "dog/000007/00.png")
        )


def test_read_rejects_missing_image_archive(tmp_path: Path) -> None:
    root = _write_root(tmp_path / "petface", tar_members=None)

    with pytest.raises(ValueError, match="not a regular file"):
        read_petface_dog_images(root, ("dog/000007/00.png",))


def test_read_rejects_member_missing_from_tar(tmp_path: Path) -> None:
    root = _write_root(tmp_path / "petface", tar_members=("dog/000008/00.png",))

    with pytest.raises(FileNotFoundError, match="missing from dog archive"):
        read_petface_dog_images(root, ("dog/000007/00.png",))


def test_read_rejects_unsafe_tar_member(tmp_path: Path) -> None:
    root = tmp_path / "petface"
    root.mkdir()
    _write_zip(
        root,
        "metadata.zip",
        {
            "PetFace/README.md": _PETFACE_README,
            "PetFace/split/dog/train.csv": (
                "filename,label\ndog/000007/00.png,7\n"
            ),
        },
    )
    payload = _png_bytes()
    with tarfile.open(root / "dog-001.tar.gz", "w:gz") as archive:
        info = tarfile.TarInfo("../dog/000007/00.png")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe PetFace tar member"):
        read_petface_dog_images(root, ("dog/000007/00.png",))


def test_adapt_fails_closed_when_tar_omits_split_image(tmp_path: Path) -> None:
    root = _write_root(tmp_path / "petface", tar_members=("dog/000008/00.png",))

    with pytest.raises(FileNotFoundError, match="missing from dog archive"):
        adapt_petface_dog(root, split_role="train", maximum_samples=1)


def test_adapt_records_official_split_member_and_album_group(
    tmp_path: Path,
) -> None:
    root = _write_root(tmp_path / "petface")

    samples = adapt_petface_dog(root, split_role="train", maximum_samples=1)

    assert samples[0].dataset_name == "petface-dog"
    assert samples[0].dataset_version == "eccv-2024-local-archive-intake-v1"
    assert samples[0].capture_group_kind is CaptureGroupKind.ALBUM_OR_SOURCE_GROUP
    assert samples[0].metadata["official_split_member"] == (
        "PetFace/split/dog/train.csv"
    )
    assert samples[0].metadata["source_intake_only"] is True
