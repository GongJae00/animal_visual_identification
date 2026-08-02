from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from evidence_fusion import (
    OOF_SIMPLEX_SCHEMA_VERSION,
    OOFSimplexConfig,
    OOFSimplexError,
    fit_oof_simplex,
)


LABELS = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
FOLDS = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])


def _fit(scores: np.ndarray, **kwargs: object):
    return fit_oof_simplex(
        ("appearance", "face"), scores, LABELS, FOLDS, **kwargs
    )


def test_complementary_channels_favor_the_more_correct_channel() -> None:
    appearance = np.asarray([0.05, 0.95, 0.1, 0.9, 0.15, 0.85, 0.8, 0.2])
    face = 1.0 - appearance
    model = _fit(np.column_stack((appearance, face)))

    assert model.weights[0] > model.weights[1]
    assert model.objective == pytest.approx(model.report["objective"]["total"])


def test_missing_availability_renormalizes_and_unusable_rows_fail() -> None:
    scores = np.column_stack((LABELS * 0.8 + 0.1, LABELS * 0.8 + 0.1))
    model = _fit(scores)
    predicted = model.predict_proba(
        np.asarray([[0.2, 0.9], [0.3, 0.8]]),
        availability=np.asarray([[True, False], [False, True]]),
    )
    np.testing.assert_allclose(predicted, [0.2, 0.8])
    with pytest.raises(OOFSimplexError, match="usable channel"):
        model.predict_proba(
            np.asarray([[0.2, 0.9]]),
            availability=np.asarray([[False, False]]),
        )
    fit_availability = np.ones((8, 2), dtype=bool)
    fit_availability[0] = False
    with pytest.raises(OOFSimplexError, match="usable channel"):
        _fit(scores, availability=fit_availability)


def test_quality_gate_changes_prediction_and_floor_prevents_zero_total() -> None:
    scores = np.column_stack((LABELS * 0.8 + 0.1, LABELS * 0.8 + 0.1))
    model = _fit(scores)
    plain = model.predict_proba(np.asarray([[0.1, 0.9]]))[0]
    gated = model.predict_proba(
        np.asarray([[0.1, 0.9]]), quality=np.asarray([[1.0, 0.1]])
    )[0]
    assert gated < plain

    floored = _fit(scores, config=OOFSimplexConfig(quality_floor=0.2))
    assert np.isfinite(
        floored.predict_proba(
            np.asarray([[0.1, 0.9]]), quality=np.asarray([[0.0, 0.0]])
        )[0]
    )


def test_hierarchical_sample_weights_can_change_the_fit() -> None:
    first = np.asarray([0.05, 0.95, 0.05, 0.95, 0.05, 0.95, 0.95, 0.05])
    scores = np.column_stack((first, 1.0 - first))
    unweighted = _fit(scores)
    weighted = _fit(scores, sample_weights=np.asarray([1] * 6 + [10, 10]))
    assert unweighted.weights[0] > unweighted.weights[1]
    assert weighted.weights[1] > weighted.weights[0]


def test_ties_are_deterministic_and_prefer_the_prior() -> None:
    identical = np.column_stack((LABELS * 0.8 + 0.1,) * 2)
    first = _fit(identical, config=OOFSimplexConfig(resolution=10))
    second = _fit(identical.copy(), config=OOFSimplexConfig(resolution=10))
    np.testing.assert_array_equal(first.weights, [0.5, 0.5])
    np.testing.assert_array_equal(first.weights, second.weights)


def test_regularization_shrinks_to_strict_configured_prior() -> None:
    appearance = np.asarray([0.05, 0.95, 0.1, 0.9, 0.15, 0.85, 0.8, 0.2])
    scores = np.column_stack((appearance, 1.0 - appearance))
    model = _fit(
        scores,
        config=OOFSimplexConfig(
            resolution=10, l2_strength=100.0, prior_weights=(0.2, 0.8)
        ),
    )
    np.testing.assert_array_equal(model.weights, [0.2, 0.8])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resolution": 0},
        {"resolution": True},
        {"l2_strength": -0.1},
        {"quality_floor": 1.1},
        {"prior_weights": (0.0, 1.0)},
        {"prior_weights": (0.2, 0.2)},
    ],
)
def test_config_is_strict(kwargs: dict[str, object]) -> None:
    with pytest.raises(OOFSimplexError):
        OOFSimplexConfig(**kwargs)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"calibrated_scores": [[0.1, 1.2]] * 8}, "finite and in"),
        ({"labels": [0.0, 1.0] * 4}, "binary integer"),
        ({"fold_ids": [0, 0, 2, 2, 0, 0, 2, 2]}, "contiguous"),
        ({"fold_ids": [0, 1] * 4}, "each fold"),
        ({"availability": np.ones((8, 2), dtype=np.int64)}, "bool array"),
        ({"quality": np.full((8, 2), np.nan)}, "finite and in"),
        ({"sample_weights": np.asarray([1] * 7 + [0])}, "positive"),
    ],
)
def test_strict_invalid_inputs(replacement: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "channel_names": ("appearance", "face"),
        "calibrated_scores": np.column_stack((LABELS * 0.8 + 0.1,) * 2),
        "labels": LABELS,
        "fold_ids": FOLDS,
    }
    arguments.update(replacement)
    with pytest.raises(OOFSimplexError, match=message):
        fit_oof_simplex(**arguments)


def test_report_is_private_json_safe_and_weights_are_read_only() -> None:
    scores = np.column_stack((LABELS * 0.8 + 0.1,) * 2)
    model = _fit(scores)
    serialized = json.dumps(model.report, allow_nan=False, sort_keys=True)

    assert model.report["schema_version"] == OOF_SIMPLEX_SCHEMA_VERSION
    assert model.report["limitations"]["oof_provenance_is_caller_attested"]
    assert model.report["limitations"]["does_not_admit_final_test_data"]
    for forbidden in ("labels", "calibrated_scores", "sample_ids"):
        assert forbidden not in serialized
    with pytest.raises(ValueError, match="read-only"):
        model.weights[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        model.objective = 0.0


def test_three_channel_scope_and_every_channel_available_validation() -> None:
    scores = np.column_stack((LABELS * 0.8 + 0.1,) * 3)
    availability = np.ones((8, 3), dtype=bool)
    availability[:, 2] = False
    with pytest.raises(OOFSimplexError, match="every channel"):
        fit_oof_simplex(
            ("appearance", "face", "nose"),
            scores,
            LABELS,
            FOLDS,
            availability=availability,
        )
