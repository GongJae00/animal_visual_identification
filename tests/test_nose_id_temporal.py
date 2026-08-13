from __future__ import annotations

import numpy as np
import pytest

from embedding.methods.nose.signal.temporal import aggregate_nose_embeddings


def _unit(index: int, dimension: int = 8) -> np.ndarray:
    value = np.zeros(dimension, dtype=np.float32)
    value[index] = 1.0
    return value


def test_quality_consensus_rejects_opposite_outlier_and_normalizes() -> None:
    first = _unit(0)
    second = np.asarray(first + 0.05 * _unit(1), dtype=np.float32)
    outlier = -first
    result = aggregate_nose_embeddings(
        [first, second, outlier], [0.8, 0.7, 0.9], reject_outliers=True
    )
    assert result.accepted_indices == (0, 1)
    assert result.rejected_indices == (2,)
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0, abs=1e-6)
    assert result.embedding[0] > 0.99
    assert result.diagnostics()["aggregation"] == (
        "QUALITY_WEIGHTED_CONSENSUS_L2_MEAN"
    )


def test_equal_quality_matches_normalized_mean_without_outlier() -> None:
    values = [_unit(0), _unit(0), np.asarray(_unit(0) + 0.2 * _unit(1))]
    result = aggregate_nose_embeddings(values)
    expected = np.mean(
        [value / np.linalg.norm(value) for value in values], axis=0
    )
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result.embedding, expected, atol=1e-6)
    assert result.accepted_indices == (0, 1, 2)
    assert result.aggregation == "UNWEIGHTED_L2_MEAN"


@pytest.mark.parametrize(
    "embeddings,qualities",
    [
        ([_unit(0)], None),
        ([_unit(0), np.zeros(8, dtype=np.float32)], None),
        ([_unit(0), _unit(1)], [0.0, 0.0]),
        ([_unit(0), _unit(1)], [1.1, 0.5]),
    ],
)
def test_invalid_temporal_inputs_fail_closed(embeddings, qualities) -> None:
    with pytest.raises(ValueError):
        aggregate_nose_embeddings(embeddings, qualities)
