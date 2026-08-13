from __future__ import annotations

from dataclasses import replace

import numpy as np

from data.types import UnifiedCanidSample
from parsing.full_segment.animal_parsing import ParsedAnimalInstance, ParsedAnimalQuality
from workflows.run_animal_parsing_panel import (
    _match_predictions_to_annotations,
    _select_ap10k_source_groups,
)


def _sample(
    index: int,
    *,
    group: str,
    annotation_id: int,
    box: tuple[float, float, float, float],
) -> UnifiedCanidSample:
    return UnifiedCanidSample(
        sample_id=f"{index:032x}",
        dataset_name="ap10k-dog",
        dataset_version="fixture",
        source_group_id=group,
        image_path=f"{group}.jpg",
        image_sha256=f"{abs(hash(group)) % (2**256):064x}",
        width=100,
        height=80,
        dog_boxes_xyxy=box,
        split_role="test",
        metadata={"annotation_id": annotation_id, "image_id": group},
    )


def _prediction(
    index: int,
    *,
    box: tuple[int, int, int, int],
    score: float,
) -> ParsedAnimalInstance:
    mask = np.zeros((80, 100), dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1
    probability = mask.astype(np.float32)
    return ParsedAnimalInstance(
        instance_index=index,
        query_index=index,
        class_id=1,
        class_name="dog",
        class_score=score,
        detector_box_xyxy=box,
        refinement_box_xyxy=box,
        mask_box_xyxy=box,
        instance_probability=probability,
        foreground_probability=probability,
        ownership_probability=probability,
        hard_mask=mask,
        quality=ParsedAnimalQuality(
            "USABLE", (), (), 1.0, 1.0, int(mask.sum()), 1, False
        ),
    )


def test_ap10k_source_selection_is_deterministic_and_keeps_multi_groups() -> None:
    samples = [
        _sample(1, group="one", annotation_id=2, box=(1, 1, 20, 20)),
        _sample(2, group="one", annotation_id=1, box=(30, 1, 50, 20)),
        _sample(3, group="two", annotation_id=3, box=(1, 1, 20, 20)),
        _sample(4, group="three", annotation_id=4, box=(1, 1, 20, 20)),
    ]
    first = _select_ap10k_source_groups(
        samples, sample_count=2, multi_source_count=1
    )
    second = _select_ap10k_source_groups(
        reversed(samples), sample_count=2, multi_source_count=1
    )
    assert first == second
    multi = next(group for group in first if group.source_group_id == "one")
    assert [item.metadata["annotation_id"] for item in multi.annotations] == [1, 2]


def test_ap10k_posthoc_matching_is_one_to_one() -> None:
    annotations = (
        _sample(1, group="one", annotation_id=1, box=(0, 0, 40, 40)),
        _sample(2, group="one", annotation_id=2, box=(50, 0, 90, 40)),
    )
    predictions = (
        _prediction(0, box=(1, 1, 39, 39), score=0.8),
        _prediction(1, box=(2, 2, 38, 38), score=0.7),
        _prediction(2, box=(51, 1, 89, 39), score=0.9),
    )
    matches = _match_predictions_to_annotations(
        predictions, annotations, minimum_box_iou=0.5
    )
    assert [(item["instance_index"], item["annotation_id"]) for item in matches] == [
        (0, 1),
        (2, 2),
    ]


def test_ap10k_source_selection_rejects_conflicting_image_contracts() -> None:
    first = _sample(1, group="one", annotation_id=1, box=(0, 0, 40, 40))
    second = replace(
        _sample(2, group="one", annotation_id=2, box=(50, 0, 90, 40)),
        image_sha256="f" * 64,
    )
    try:
        _select_ap10k_source_groups(
            (first, second), sample_count=1, multi_source_count=1
        )
    except ValueError as exc:
        assert "contracts differ" in str(exc)
    else:
        raise AssertionError("conflicting AP-10K source contract was accepted")
