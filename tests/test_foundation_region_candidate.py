from __future__ import annotations

import numpy as np
import pytest

from localization.foundation_dense_runtime import FoundationImageTransform
from localization.foundation_region_candidate import derive_binary_foundation_candidate


def test_foundation_candidate_maps_back_to_source_and_respects_geometry() -> None:
    yy, xx = np.mgrid[0:8, 0:8]
    features = np.stack((xx, yy, 8 - xx, 8 - yy), axis=2).astype(np.float32)
    features /= np.linalg.norm(features, axis=2, keepdims=True)
    transform = FoundationImageTransform(80, 40, 128, 128, 64, 0, 32)
    candidate = derive_binary_foundation_candidate(
        features,
        transform=transform,
        source_validity=np.pad(np.ones((4, 8), dtype=bool), ((2, 2), (0, 0))),
        box_xyxy=(10.0, 5.0, 70.0, 35.0),
        positive_points_xy=((30.0, 20.0), (50.0, 20.0)),
    )
    assert candidate.source_probability.shape == (40, 80)
    assert candidate.source_hard_mask.shape == (40, 80)
    assert candidate.source_geometry_support.shape == (40, 80)
    assert np.all(candidate.source_probability[~candidate.source_geometry_support] == 0.0)
    assert 0.0 <= candidate.confidence <= 1.0


def test_foundation_candidate_rejects_points_outside_source() -> None:
    features = np.ones((4, 4, 2), dtype=np.float32)
    transform = FoundationImageTransform(40, 40, 64, 64, 64, 0, 0)
    with pytest.raises(ValueError, match="point lies outside"):
        derive_binary_foundation_candidate(
            features,
            transform=transform,
            source_validity=np.ones((4, 4), dtype=bool),
            box_xyxy=(0.0, 0.0, 40.0, 40.0),
            positive_points_xy=((50.0, 20.0),),
        )
