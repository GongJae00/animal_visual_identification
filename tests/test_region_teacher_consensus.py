from __future__ import annotations

import numpy as np
import pytest

from parsing.regions.region_teacher_consensus import (
    ConsensusState,
    RegionTeacherBinding,
    region_teacher_consensus,
)


def _teachers(*families: str) -> tuple[RegionTeacherBinding, ...]:
    return tuple(
        RegionTeacherBinding(f"teacher-{index}", family, f"{index + 1:064x}")
        for index, family in enumerate(families)
    )


def _binary_probabilities(masks: tuple[np.ndarray, ...], confidence: float) -> np.ndarray:
    rows = []
    for mask in masks:
        foreground = np.where(mask, confidence, 1.0 - confidence)
        rows.append(np.stack((1.0 - foreground, foreground)))
    return np.stack(rows)


def test_distinct_agreeing_teachers_produce_hard_candidate() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:7, 2:6] = True
    result = region_teacher_consensus(
        "A",
        _binary_probabilities((mask, mask), 0.95),
        teachers=_teachers("CRADIO_V4", "SAM2_1"),
        source_validity=np.ones((8, 8), dtype=bool),
        geometry_support=np.ones((8, 8), dtype=bool),
    )
    assert result.state is ConsensusState.HARD_CANDIDATE
    assert result.hard_mask is not None
    assert result.soft_probabilities is not None
    assert result.to_record()["qualification"] == "MODEL_GENERATED_CANDIDATE"
    assert "NOT_VERIFIED_SEMANTIC" in result.interpretation


def test_same_family_cannot_produce_hard_candidate() -> None:
    mask = np.ones((6, 6), dtype=bool)
    result = region_teacher_consensus(
        "A",
        _binary_probabilities((mask, mask), 0.95),
        teachers=_teachers("CRADIO_V4", "CRADIO_V4"),
        source_validity=np.ones((6, 6), dtype=bool),
        geometry_support=np.ones((6, 6), dtype=bool),
    )
    assert result.state is ConsensusState.SOFT_CANDIDATE
    assert result.metrics["decision_reasons"] == [
        "INSUFFICIENT_TEACHER_FAMILY_DIVERSITY_FOR_HARD"
    ]


def test_disagreement_abstains_without_candidate_arrays() -> None:
    left = np.zeros((8, 8), dtype=bool)
    right = np.zeros((8, 8), dtype=bool)
    left[:, :3] = True
    right[:, 5:] = True
    result = region_teacher_consensus(
        "A",
        _binary_probabilities((left, right), 0.95),
        teachers=_teachers("CRADIO_V4", "SAM2_1"),
        source_validity=np.ones((8, 8), dtype=bool),
        geometry_support=np.ones((8, 8), dtype=bool),
    )
    assert result.state is ConsensusState.ABSTAIN
    assert result.hard_mask is None
    assert result.soft_probabilities is None
    assert "LOW_PAIRWISE_IOU" in result.metrics["decision_reasons"]


def test_no_geometry_support_fails_closed() -> None:
    mask = np.ones((4, 4), dtype=bool)
    result = region_teacher_consensus(
        "A",
        _binary_probabilities((mask, mask), 0.99),
        teachers=_teachers("CRADIO_V4", "SAM2_1"),
        source_validity=np.ones((4, 4), dtype=bool),
        geometry_support=np.zeros((4, 4), dtype=bool),
    )
    assert result.state is ConsensusState.ABSTAIN
    assert result.metrics["decision_reasons"] == ["NO_VALID_GEOMETRY_SUPPORT"]


def test_malformed_probabilities_are_rejected() -> None:
    values = np.full((2, 2, 4, 4), 0.4)
    with pytest.raises(ValueError, match="sum to one"):
        region_teacher_consensus(
            "A",
            values,
            teachers=_teachers("CRADIO_V4", "SAM2_1"),
            source_validity=np.ones((4, 4), dtype=bool),
            geometry_support=np.ones((4, 4), dtype=bool),
        )
