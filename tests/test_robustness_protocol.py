from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from cvi.evaluation import (
    ROBUSTNESS_PROTOCOL_SCHEMA_VERSION,
    RobustnessProtocolConfig,
    RobustnessProtocolError,
    build_dataset_balanced_oof_protocol,
)
from cvi.role_exposure import ExposureStage


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    dataset_ids: list[str] = []
    identity_ids: list[str] = []
    identity_sample_counts = {
        "dataset-a": (1, 2, 4, 1, 3),
        "dataset-b": (5, 1, 2, 3, 1, 4, 2, 1),
    }
    for dataset, counts in identity_sample_counts.items():
        for identity_index, sample_count in enumerate(counts):
            dataset_ids.extend([dataset] * sample_count)
            identity_ids.extend(
                [f"private-{dataset}-identity-{identity_index}"] * sample_count
            )
    return np.asarray(dataset_ids), np.asarray(identity_ids)


def _config(
    *,
    seed: int = 17,
    target: ExposureStage = ExposureStage.MODEL_SELECTION_SCORED,
) -> RobustnessProtocolConfig:
    return RobustnessProtocolConfig(
        n_splits=3,
        seed=seed,
        target_exposure_stage=target,
    )


def test_assignment_is_deterministic_seeded_grouped_and_dataset_balanced() -> None:
    datasets, identities = _fixture()
    first = build_dataset_balanced_oof_protocol(datasets, identities, config=_config())
    repeated = build_dataset_balanced_oof_protocol(
        datasets.copy(), identities.copy(), config=_config()
    )
    changed_seed = build_dataset_balanced_oof_protocol(
        datasets, identities, config=_config(seed=18)
    )

    np.testing.assert_array_equal(first.fold_ids, repeated.fold_ids)
    assert np.any(first.fold_ids != changed_seed.fold_ids)
    assert first.fold_ids.dtype == np.int64
    assert not first.fold_ids.flags.writeable

    for identity in np.unique(identities):
        assert len(np.unique(first.fold_ids[identities == identity])) == 1
    for dataset in np.unique(datasets):
        identity_folds = [
            int(first.fold_ids[np.flatnonzero(identities == identity)[0]])
            for identity in np.unique(identities[datasets == dataset])
        ]
        fold_counts = np.bincount(identity_folds, minlength=3)
        assert int(fold_counts.max() - fold_counts.min()) <= 1


def test_hierarchical_weights_are_exactly_balanced_at_each_level() -> None:
    datasets, identities = _fixture()
    result = build_dataset_balanced_oof_protocol(datasets, identities, config=_config())
    weights = result.sample_weights

    assert weights.dtype == np.float64
    assert not weights.flags.writeable
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert np.mean(weights) == pytest.approx(1.0)

    expected_dataset_total = len(weights) / len(np.unique(datasets))
    for dataset in np.unique(datasets):
        dataset_mask = datasets == dataset
        assert np.sum(weights[dataset_mask]) == pytest.approx(expected_dataset_total)
        dataset_identities = np.unique(identities[dataset_mask])
        expected_identity_total = expected_dataset_total / len(dataset_identities)
        for identity in dataset_identities:
            identity_weights = weights[identities == identity]
            assert np.sum(identity_weights) == pytest.approx(expected_identity_total)
            np.testing.assert_array_equal(
                identity_weights,
                np.full(len(identity_weights), identity_weights[0]),
            )


def test_result_is_frozen() -> None:
    datasets, identities = _fixture()
    result = build_dataset_balanced_oof_protocol(datasets, identities, config=_config())
    with pytest.raises(FrozenInstanceError):
        result.report = {}
    with pytest.raises(ValueError, match="read-only"):
        result.fold_ids[0] = 1


def test_final_test_and_other_higher_exposure_fail_closed() -> None:
    datasets, identities = _fixture()
    final_test = [ExposureStage.MODEL_SELECTION_SCORED] * len(datasets)
    final_test[0] = ExposureStage.FINAL_TEST_SCORED
    with pytest.raises(RobustnessProtocolError, match="FINAL_TEST_SCORED exceeds"):
        build_dataset_balanced_oof_protocol(
            datasets,
            identities,
            config=_config(),
            historical_exposure_stages=final_test,
        )

    calibration = [ExposureStage.MODEL_SELECTION_SCORED] * len(datasets)
    calibration[0] = ExposureStage.CALIBRATION_SCORED
    with pytest.raises(RobustnessProtocolError, match="CALIBRATION_SCORED exceeds"):
        build_dataset_balanced_oof_protocol(
            datasets,
            identities,
            config=_config(),
            historical_exposure_stages=calibration,
        )


def test_equal_or_lower_historical_exposure_is_allowed() -> None:
    datasets, identities = _fixture()
    allowed = [
        tuple(ExposureStage)[index % 4]
        for index in range(len(datasets))
    ]
    result = build_dataset_balanced_oof_protocol(
        datasets,
        identities,
        config=_config(target=ExposureStage.CALIBRATION_SCORED),
        historical_exposure_stages=allowed,
    )
    assert len(result.fold_ids) == len(datasets)


def test_each_dataset_requires_enough_identities() -> None:
    with pytest.raises(RobustnessProtocolError, match="at least n_splits=3"):
        build_dataset_balanced_oof_protocol(
            ["dataset-a", "dataset-a", "dataset-b", "dataset-b", "dataset-b"],
            ["a-1", "a-2", "b-1", "b-2", "b-3"],
            config=_config(),
        )


def test_identity_must_not_be_shared_across_datasets() -> None:
    with pytest.raises(RobustnessProtocolError, match="shared across datasets"):
        build_dataset_balanced_oof_protocol(
            ["a", "a", "a", "b", "b", "b", "b"],
            ["shared", "a-2", "a-3", "shared", "b-2", "b-3", "b-4"],
            config=_config(),
        )


@pytest.mark.parametrize(
    ("datasets", "identities", "message"),
    [
        ([], [], "must not be empty"),
        (["a", "a", "a"], ["1", "2"], "lengths differ"),
        ([[]], [["1"]], "must be 1-D"),
        (["a", "a", ""], ["1", "2", "3"], "stable non-empty string"),
        (["a", "a", " a"], ["1", "2", "3"], "stable non-empty string"),
        (["a", "a", 3], ["1", "2", "3"], "stable non-empty string"),
    ],
)
def test_ids_are_strict_1d_nonempty_stable_strings(
    datasets: object, identities: object, message: str
) -> None:
    with pytest.raises(RobustnessProtocolError, match=message):
        build_dataset_balanced_oof_protocol(datasets, identities, config=_config())


def test_historical_stage_input_is_strict() -> None:
    datasets, identities = _fixture()
    with pytest.raises(RobustnessProtocolError, match="length differs"):
        build_dataset_balanced_oof_protocol(
            datasets,
            identities,
            config=_config(),
            historical_exposure_stages=[ExposureStage.BYTES_EXPORTED],
        )
    with pytest.raises(RobustnessProtocolError, match="must be 1-D"):
        build_dataset_balanced_oof_protocol(
            datasets,
            identities,
            config=_config(),
            historical_exposure_stages=[
                [ExposureStage.BYTES_EXPORTED] * len(datasets)
            ],
        )
    with pytest.raises(TypeError, match="ExposureStage values"):
        build_dataset_balanced_oof_protocol(
            datasets,
            identities,
            config=_config(),
            historical_exposure_stages=["BYTES_EXPORTED"] * len(datasets),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": 1, "seed": 0, "target_exposure_stage": ExposureStage.MODEL_SELECTION_SCORED},
        {"n_splits": True, "seed": 0, "target_exposure_stage": ExposureStage.MODEL_SELECTION_SCORED},
        {"n_splits": 3, "seed": -1, "target_exposure_stage": ExposureStage.MODEL_SELECTION_SCORED},
        {"n_splits": 3, "seed": True, "target_exposure_stage": ExposureStage.MODEL_SELECTION_SCORED},
        {"n_splits": 3, "seed": 0, "target_exposure_stage": ExposureStage.BYTES_EXPORTED},
        {"n_splits": 3, "seed": 0, "target_exposure_stage": ExposureStage.FINAL_TEST_SCORED},
    ],
)
def test_config_rejects_invalid_ranges_and_target_stages(kwargs: dict) -> None:
    with pytest.raises(RobustnessProtocolError):
        RobustnessProtocolConfig(**kwargs)


def test_config_requires_exposure_stage_enum_and_function_requires_config() -> None:
    with pytest.raises(TypeError, match="must be ExposureStage"):
        RobustnessProtocolConfig(
            n_splits=3,
            seed=0,
            target_exposure_stage="MODEL_SELECTION_SCORED",  # type: ignore[arg-type]
        )
    datasets, identities = _fixture()
    with pytest.raises(TypeError, match="RobustnessProtocolConfig"):
        build_dataset_balanced_oof_protocol(
            datasets, identities, config=None  # type: ignore[arg-type]
        )


def test_report_is_json_safe_aggregate_only_and_states_role_limitations() -> None:
    datasets, identities = _fixture()
    result = build_dataset_balanced_oof_protocol(
        datasets,
        identities,
        config=_config(target=ExposureStage.CALIBRATION_SCORED),
    )
    serialized = json.dumps(result.report, allow_nan=False, sort_keys=True)

    assert result.report["schema_version"] == ROBUSTNESS_PROTOCOL_SCHEMA_VERSION
    assert result.report["target_exposure_stage"] == "CALIBRATION_SCORED"
    assert result.report["sample_count"] == len(datasets)
    assert result.report["dataset_count"] == 2
    assert not result.report["historical_exposure_stages_provided"]
    assert {row["dataset_name"] for row in result.report["datasets"]} == {
        "dataset-a",
        "dataset-b",
    }
    for identity in np.unique(identities):
        assert identity not in serialized
    assert "identity_ids" not in serialized
    assert result.report["limitations"] == {
        "operates_only_inside_already_admitted_role": True,
        "does_not_create_train_calibration_test_split": True,
        "does_not_restore_untouched_status": True,
        "does_not_admit_exposed_data": True,
        "caller_must_supply_authenticated_exposure_history": True,
    }


def test_report_records_when_exposure_history_was_supplied() -> None:
    datasets, identities = _fixture()
    result = build_dataset_balanced_oof_protocol(
        datasets,
        identities,
        config=_config(),
        historical_exposure_stages=[
            ExposureStage.MODEL_TRAINING_USED
        ] * len(datasets),
    )
    assert result.report["historical_exposure_stages_provided"]
