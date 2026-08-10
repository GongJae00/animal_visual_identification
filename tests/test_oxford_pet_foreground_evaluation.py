from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from workflows.evaluate_oxford_pet_foreground import (
    OxfordPetSample,
    _aggregate_evaluations,
    _evaluate_mask,
    _load_split_samples,
    _load_trimap,
    _preflight_samples,
)


def _write_split(root: Path, rows: list[str]) -> None:
    annotations = root / "annotations"
    annotations.mkdir()
    (annotations / "test.txt").write_text("\n".join(rows) + "\n", encoding="ascii")


def test_oxford_selection_is_deterministic_and_species_filtered(
    tmp_path: Path,
) -> None:
    _write_split(
        tmp_path,
        [
            "Abyssinian_1 1 1 1",
            "Abyssinian_2 1 1 1",
            "beagle_1 5 2 1",
            "beagle_2 5 2 1",
            "boxer_1 6 2 2",
        ],
    )
    first = _load_split_samples(
        tmp_path, split="test", species="dog", sample_count=2
    )
    second = _load_split_samples(
        tmp_path, split="test", species="dog", sample_count=2
    )
    assert first == second
    assert len(first) == 2
    assert all(sample.species == "dog" for sample in first)


def test_oxford_selection_rejects_path_like_sample_names(tmp_path: Path) -> None:
    _write_split(tmp_path, ["../beagle_1 5 2 1"])
    with pytest.raises(ValueError, match="row 1 differs"):
        _load_split_samples(
            tmp_path, split="test", species="all", sample_count=None
        )


def test_oxford_trimap_metrics_exclude_not_classified_pixels(
    tmp_path: Path,
) -> None:
    trimap = np.asarray(((1, 1, 3), (2, 2, 3)), dtype=np.uint8)
    path = tmp_path / "trimap.png"
    Image.fromarray(trimap, mode="L").save(path)
    loaded = _load_trimap(path, expected_size=(3, 2))
    mask = np.asarray(((1, 0, 1), (1, 0, 1)), dtype=np.uint8)
    result = _evaluate_mask(mask, loaded)
    assert result["counts"] == {
        "true_positive_pixels": 1,
        "false_positive_pixels": 1,
        "false_negative_pixels": 1,
        "true_negative_pixels": 1,
        "not_classified_pixels": 2,
        "predicted_foreground_not_classified_pixels": 2,
    }
    assert result["metrics"]["classified_pixel_iou"] == pytest.approx(1 / 3)
    assert result["metrics"]["classified_pixel_dice"] == pytest.approx(0.5)
    assert result["metrics"]["correction_rate"] == pytest.approx(0.5)


def test_oxford_aggregate_distinguishes_micro_and_macro() -> None:
    first = _evaluate_mask(
        np.asarray(((1, 0),), dtype=np.uint8),
        np.asarray(((1, 2),), dtype=np.uint8),
    )
    second = _evaluate_mask(
        np.asarray(((1, 1, 1, 1),), dtype=np.uint8),
        np.asarray(((1, 2, 2, 2),), dtype=np.uint8),
    )
    aggregate = _aggregate_evaluations([first, second])
    assert aggregate["record_count"] == 2
    assert aggregate["micro_average"]["background_leakage_rate"] == pytest.approx(
        3 / 4
    )
    assert aggregate["macro_average"]["background_leakage_rate"] == pytest.approx(
        0.5
    )


def test_oxford_preflight_records_empty_ground_truth_exclusion(
    tmp_path: Path,
) -> None:
    (tmp_path / "images").mkdir()
    trimaps = tmp_path / "annotations" / "trimaps"
    trimaps.mkdir(parents=True)
    samples = (
        OxfordPetSample("beagle_1", 5, "dog", 1),
        OxfordPetSample("beagle_2", 5, "dog", 1),
    )
    for sample in samples:
        Image.new("RGB", (3, 2), "white").save(
            tmp_path / "images" / f"{sample.name}.jpg"
        )
    Image.fromarray(np.asarray(((1, 1, 3), (2, 2, 3)), dtype=np.uint8)).save(
        trimaps / "beagle_1.png"
    )
    Image.fromarray(np.full((2, 3), 2, dtype=np.uint8)).save(
        trimaps / "beagle_2.png"
    )
    eligible, exclusions = _preflight_samples(samples, dataset_root=tmp_path)
    assert eligible == (samples[0],)
    assert exclusions[0]["sample_name"] == "beagle_2"
    assert exclusions[0]["reason"] == "GROUND_TRUTH_TRIMAP_HAS_NO_FOREGROUND"
