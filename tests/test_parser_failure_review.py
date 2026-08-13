from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from workflows import render_parser_failure_review as review


def _row(dataset: str, token: str, source: str) -> dict[str, str]:
    return {
        "dataset_name": dataset,
        "sample_token": token,
        "source_sha256": source,
    }


def test_balanced_selection_is_deterministic_and_prefers_unique_sources() -> None:
    rows = [
        _row("b", "b2", "shared"),
        _row("a", "a2", "shared"),
        _row("b", "b1", "b-only"),
        _row("a", "a1", "a-only"),
    ]

    selected = review._balanced_select(rows, 3, "REASON")
    reversed_selected = review._balanced_select(tuple(reversed(rows)), 3, "REASON")

    assert [row["sample_token"] for row in selected] == [
        row["sample_token"] for row in reversed_selected
    ]
    assert len({row["source_sha256"] for row in selected}) == 3


def test_annotated_tile_draws_annotation_and_parser_overlays() -> None:
    source = Image.new("RGB", (20, 20), "black")
    hard_mask = np.zeros((20, 20), dtype=np.uint8)
    hard_mask[8:13, 8:13] = 1
    instance = SimpleNamespace(
        class_name="dog",
        class_score=0.9,
        detector_box_xyxy=(7, 7, 14, 14),
        hard_mask=hard_mask,
    )
    prediction = SimpleNamespace(instances=(instance,))

    tile = review._annotated_tile(
        source,
        prediction,
        {
            "kind": "AP10K_AUTHORITATIVE_BBOX_ASSOCIATION",
            "bbox_xyxy": [1.0, 1.0, 5.0, 5.0],
        },
    )
    colors = set(tile.getdata())

    assert (74, 222, 128) in colors
    assert (248, 113, 113) in colors
    assert (34, 211, 238) in colors


def test_non_dog_annotation_box_is_not_drawn() -> None:
    source = Image.new("RGB", (20, 20), "black")
    prediction = SimpleNamespace(instances=())

    tile = review._annotated_tile(
        source,
        prediction,
        {"kind": "DOGFLW_FACE_BBOX", "bbox_xyxy": [1.0, 1.0, 5.0, 5.0]},
    )

    assert (74, 222, 128) not in set(tile.getdata())


def test_renderer_rejects_non_integer_sample_count(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        review.render_parser_failure_review(
            route_plan=tmp_path / "plan.json",
            materialization_root=tmp_path / "materialization",
            output_dir=tmp_path / "review",
            samples_per_reason=True,
        )


def test_reason_slugs_cover_v6_terminal_taxonomy() -> None:
    assert review._reason_slug("PARSER_DISTINCT_DOG_CARDINALITY_AMBIGUOUS") == (
        "02_distinct_dogs"
    )
    assert review._reason_slug("NO_VALID_PARSED_DOG_INSTANCE") == "03_no_valid_dog"
