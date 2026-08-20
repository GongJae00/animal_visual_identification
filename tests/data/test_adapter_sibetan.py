from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from data.adapters.io import file_sha256, image_dims
from data.adapters.sibetan import adapt_sibetan
from data.types import CaptureGroupKind
from shared.contracts.identity_ids import (
    compute_registered_dog_id,
    compute_sample_token,
)


def _write_image(path: Path, size: tuple[int, int] = (32, 24)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)
    return path


def _write_gt(base: Path, mapping: dict[str, object]) -> None:
    (base / "gt_sibetan.json").write_text(json.dumps(mapping), encoding="utf-8")


def test_adapt_sibetan_requires_publisher_base(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Sibetan base not found"):
        adapt_sibetan(tmp_path)


def test_adapt_sibetan_requires_identity_gt(tmp_path: Path) -> None:
    (tmp_path / "Sibetan").mkdir()

    with pytest.raises(FileNotFoundError, match="Sibetan identity GT not found"):
        adapt_sibetan(tmp_path)


@pytest.mark.parametrize("payload", ("[]", "{}", '"x"'))
def test_adapt_sibetan_rejects_empty_or_non_object_gt(
    tmp_path: Path, payload: str
) -> None:
    base = tmp_path / "Sibetan"
    base.mkdir()
    (base / "gt_sibetan.json").write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="must be a non-empty object"):
        adapt_sibetan(tmp_path)


@pytest.mark.parametrize(
    "mapping",
    (
        {"dogA": [0]},
        {"0": []},
        {"0": 1},
    ),
)
def test_adapt_sibetan_rejects_identity_gt_schema(
    tmp_path: Path, mapping: dict[str, object]
) -> None:
    base = tmp_path / "Sibetan"
    _write_image(base / "0" / "Indonesia_C01_frame.jpg")
    _write_gt(base, mapping)

    with pytest.raises(ValueError, match="identity GT schema differs"):
        adapt_sibetan(tmp_path)


@pytest.mark.parametrize("clusters", ([True], [-1], ["0"], [0.5]))
def test_adapt_sibetan_rejects_identity_gt_cluster_values(
    tmp_path: Path, clusters: list[object]
) -> None:
    base = tmp_path / "Sibetan"
    _write_image(base / "0" / "Indonesia_C01_frame.jpg")
    _write_gt(base, {"0": clusters})

    with pytest.raises(ValueError, match="identity GT cluster differs"):
        adapt_sibetan(tmp_path)


def test_adapt_sibetan_rejects_repeated_sequence_cluster(tmp_path: Path) -> None:
    base = tmp_path / "Sibetan"
    _write_image(base / "1" / "Indonesia_C01_frame.jpg")
    _write_gt(base, {"0": [1], "1": [1]})

    with pytest.raises(ValueError, match="repeats a sequence cluster"):
        adapt_sibetan(tmp_path)


def test_adapt_sibetan_rejects_image_cluster_absent_from_gt(tmp_path: Path) -> None:
    base = tmp_path / "Sibetan"
    _write_image(base / "0" / "Indonesia_C01_a.jpg")
    _write_image(base / "1" / "Indonesia_C01_b.jpg")
    _write_gt(base, {"0": [0]})

    with pytest.raises(ValueError, match="image cluster is absent from identity GT"):
        adapt_sibetan(tmp_path)


def test_adapt_sibetan_rejects_gt_cluster_without_directory(tmp_path: Path) -> None:
    base = tmp_path / "Sibetan"
    _write_image(base / "0" / "Indonesia_C01_a.jpg")
    _write_gt(base, {"0": [0, 1]})

    with pytest.raises(ValueError, match="GT and sequence directories differ"):
        adapt_sibetan(tmp_path)


def test_adapt_sibetan_skips_non_digit_dirs_and_non_image_files(
    tmp_path: Path,
) -> None:
    base = tmp_path / "Sibetan"
    kept = _write_image(base / "0" / "Indonesia_C01_frame.jpg")
    _write_image(base / "notes" / "Indonesia_C99_ignored.jpg")
    (base / "0" / "readme.txt").write_text("not an image", encoding="utf-8")
    _write_gt(base, {"7": [0]})

    samples = adapt_sibetan(tmp_path)

    assert len(samples) == 1
    assert samples[0].raw_identity_id == "7"
    assert samples[0].capture_group_id == "0"
    assert samples[0].source_group_id == "0"
    assert samples[0].image_path == str(kept.relative_to(tmp_path))


def test_adapt_sibetan_ignores_no_mono_gt_file(tmp_path: Path) -> None:
    base = tmp_path / "Sibetan"
    _write_image(base / "0" / "Indonesia_C01_frame.jpg")
    _write_gt(base, {"4": [0]})
    (base / "gt_sibetan_no_mono_cluster.json").write_text(
        json.dumps({"9": [0]}), encoding="utf-8"
    )

    samples = adapt_sibetan(tmp_path)

    assert [sample.raw_identity_id for sample in samples] == ["4"]


def test_adapt_sibetan_binds_tokens_hashes_and_unassigned_split(
    tmp_path: Path,
) -> None:
    base = tmp_path / "Sibetan"
    jpg = _write_image(base / "2" / "Indonesia_C03_clip.jpg", (16, 12))
    png = _write_image(base / "10" / "site_no_camera.png", (8, 6))
    _write_gt(base, {"1": [2], "8": [10]})

    samples = adapt_sibetan(tmp_path)

    assert [sample.dataset_name for sample in samples] == ["sibetan", "sibetan"]
    assert [sample.dataset_version for sample in samples] == [
        "publisher-v1-2025-10-27",
        "publisher-v1-2025-10-27",
    ]
    assert [sample.split_role for sample in samples] == ["UNASSIGNED", "UNASSIGNED"]
    assert [sample.capture_group_kind for sample in samples] == [
        CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
        CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
    ]
    assert [sample.capture_group_id for sample in samples] == ["10", "2"]
    assert [sample.raw_identity_id for sample in samples] == ["8", "1"]
    first, second = samples
    assert first.sample_id == compute_sample_token("sibetan:10:site_no_camera")
    assert first.registered_identity_id == compute_registered_dog_id(
        "sibetan:v1:gt-json:8"
    )
    assert first.image_sha256 == file_sha256(png)
    assert (first.width, first.height) == image_dims(png)
    assert first.metadata["unverified_camera_token"] is None
    assert second.sample_id == compute_sample_token("sibetan:2:Indonesia_C03_clip")
    assert second.registered_identity_id == compute_registered_dog_id(
        "sibetan:v1:gt-json:1"
    )
    assert second.image_sha256 == file_sha256(jpg)
    assert (second.width, second.height) == image_dims(jpg)
    assert second.metadata["unverified_camera_token"] == "C03"
    assert first.image_path == str(png.relative_to(tmp_path))
    assert second.image_path == str(jpg.relative_to(tmp_path))


def test_adapt_sibetan_rejects_symlinked_image(tmp_path: Path) -> None:
    base = tmp_path / "Sibetan"
    real = _write_image(tmp_path / "outside.jpg")
    cluster = base / "0"
    cluster.mkdir(parents=True)
    (cluster / "Indonesia_C01_frame.jpg").symlink_to(real)
    _write_gt(base, {"0": [0]})

    with pytest.raises(ValueError, match="not a regular file"):
        adapt_sibetan(tmp_path)
