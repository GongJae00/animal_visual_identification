from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from identity.registry.identity_registry import compute_identity_token
from foundation.provenance import content_sha256
from identity.exposure.role_exposure import (
    ExposureDeclarationKind,
    ExposureStage,
    RoleExposureDeclaration,
    RoleExposureDeclarationRecord,
    merge_role_exposure_declarations,
)
from workflows import evaluate_dogface_holdout_fusion as tool


ROOT = Path(__file__).resolve().parents[1]


def _token(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _population_documents(
    *, calibration_samples: int = 4, final_samples: int = 5
) -> tuple[dict, dict]:
    assignment_records = []
    label_records = []
    for role, prefix, sample_count in (
        ("DOGFACE_DEVELOPMENT", "cal", calibration_samples),
        ("DOGFACE_CALIBRATION", "final", final_samples),
    ):
        for identity_index in range(2):
            dataset_identity_id = (
                f"dogfacenet224:v1:web-folder:{prefix}-{identity_index}"
            )
            identity_token = compute_identity_token(dataset_identity_id)
            for sample_index in range(sample_count):
                sample_token = _token(
                    f"sample:{prefix}:{identity_index}:{sample_index}"
                )
                assignment_records.append(
                    {
                        "sample_token": sample_token,
                        "identity_token": identity_token,
                        "identity_role": role,
                        "dataset_name": "dogfacenet224",
                        "source_variant": "original",
                    }
                )
                label_records.append(
                    {
                        "sample_token": sample_token,
                        "identity_token": identity_token,
                        "dataset_identity_id": dataset_identity_id,
                        "original_split": "train",
                        "region": "FACE",
                    }
                )
    return {"records": assignment_records}, {"records": label_records}


def _prior_ledger(
    populations: dict,
    role: str,
    identity_index: int = 0,
    *,
    different_sample: bool = False,
):
    identity = populations[role]["identities"][identity_index]
    declaration = RoleExposureDeclaration(
        source_artifact_sha256=_token("prior-artifact"),
        kind=ExposureDeclarationKind.PRIOR_ASSIGNMENT,
        revoked=False,
        records=(
            RoleExposureDeclarationRecord(
                sample_token=(
                    _token("different-prior-sample")
                    if different_sample
                    else identity["sample_tokens"][0]
                ),
                identity_token=identity["identity_token"],
                public_subject_token=identity["public_subject_token"],
                stage=ExposureStage.BYTES_EXPORTED,
            ),
        ),
    )
    return merge_role_exposure_declarations((declaration,))


def _common_cli() -> list[str]:
    digest = "a" * 64
    return [
        "--assignment",
        "/tmp/assignment.json",
        "--labels",
        "/tmp/labels.json",
        "--source-bundle",
        "/tmp/source.json",
        "--split-receipt",
        "/tmp/split-receipt.json",
        "--split-receipt-sha256",
        digest,
        "--source-spec",
        "/tmp/source-spec.json",
        "--historical-exposure-ledger",
        "/tmp/old-ledger.json",
        "--historical-exposure-receipt",
        "/tmp/old-receipt.json",
        "--historical-exposure-receipt-sha256",
        digest,
        "--appearance-checkpoint",
        "/tmp/appearance.pt",
        "--appearance-checkpoint-sha256",
        digest,
        "--face-checkpoint",
        "/tmp/face.pt",
        "--face-checkpoint-sha256",
        digest,
        "--model-dir",
        "/tmp/model",
        "--weight-intake-bundle",
        "/tmp/weight.json",
        "--preprocessor-intake-bundle",
        "/tmp/preprocessor.json",
        "--frozen-model-sha256",
        digest,
    ]


def test_populations_are_token_sorted_deterministic_and_three_shot_bounded() -> None:
    assignment, labels = _population_documents()
    first = tool._build_holdout_populations(
        assignment, labels, expected_identities=2
    )
    second = tool._build_holdout_populations(
        {"records": list(reversed(assignment["records"]))},
        {"records": list(reversed(labels["records"]))},
        expected_identities=2,
    )

    assert first == second
    assert first["calibration"]["all_sample_tokens"] == sorted(
        first["calibration"]["all_sample_tokens"]
    )
    for shot in ("one_shot", "three_shot"):
        assert first["calibration"][shot]["gallery_sample_tokens"] == sorted(
            first["calibration"][shot]["gallery_sample_tokens"]
        )
        assert first["calibration"][shot]["query_sample_tokens"] == sorted(
            first["calibration"][shot]["query_sample_tokens"]
        )
    for identity in first["calibration"]["identities"]:
        assert identity["sample_tokens"] == sorted(identity["sample_tokens"])
        assert identity["sample_tokens"][0] in first["calibration"]["one_shot"][
            "gallery_sample_tokens"
        ]
        assert set(identity["sample_tokens"][:3]).issubset(
            first["calibration"]["three_shot"]["gallery_sample_tokens"]
        )
    assert tool._population_summary(first["calibration"])[
        "three_shot_identity_count"
    ] == 2


def test_population_rejects_overlap_and_prior_sample_or_identity_exposure() -> None:
    assignment, labels = _population_documents()
    populations = tool._build_holdout_populations(
        assignment, labels, expected_identities=2
    )
    with pytest.raises(ValueError, match="prior role exposure"):
        tool._reject_prior_exposure(
            populations, _prior_ledger(populations, "calibration")
        )
    with pytest.raises(ValueError, match="identities have prior"):
        tool._reject_prior_exposure(
            populations,
            _prior_ledger(
                populations, "calibration", different_sample=True
            ),
        )

    final_identity = populations["final"]["identities"][0]["identity_token"]
    final_label = next(
        value for value in labels["records"] if value["identity_token"] == final_identity
    )
    calibration_identity = populations["calibration"]["identities"][0][
        "identity_token"
    ]
    calibration_label = next(
        value
        for value in labels["records"]
        if value["identity_token"] == calibration_identity
    )
    original_final_dataset_identity = final_label["dataset_identity_id"]
    for record in assignment["records"]:
        if record["identity_token"] == final_identity:
            record["identity_token"] = calibration_identity
    for record in labels["records"]:
        if record["identity_token"] == final_identity:
            record["identity_token"] = calibration_identity
            record["dataset_identity_id"] = calibration_label["dataset_identity_id"]
    assert original_final_dataset_identity != calibration_label["dataset_identity_id"]
    with pytest.raises(ValueError, match="identities overlap"):
        tool._build_holdout_populations(assignment, labels, expected_identities=2)


def test_pair_weights_equalize_identities_and_split_each_class_half() -> None:
    query_ids = ["a", "a", "b"]
    gallery_ids = ["a", "b", "c"]
    labels, weights = tool._build_pair_weights(query_ids, gallery_ids)
    labels = labels.reshape(3, 3)
    weights = weights.reshape(3, 3)

    for identity in ("a", "b"):
        rows = np.asarray(query_ids) == identity
        assert np.sum(weights[rows]) == pytest.approx(1.0)
        assert np.sum(weights[rows][labels[rows] == 1]) == pytest.approx(0.5)
        assert np.sum(weights[rows][labels[rows] == 0]) == pytest.approx(0.5)


def test_oof_isotonic_fitter_never_receives_held_out_fold() -> None:
    scores = np.asarray([0.01, 0.02, 1.01, 1.02, 2.01, 2.02])
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    folds = np.asarray([0, 0, 1, 1, 2, 2])
    fitted_values: list[set[float]] = []

    class FakeIsotonic:
        def __init__(self, values: np.ndarray) -> None:
            self.X_thresholds_ = np.asarray([values.min(), values.max()])
            self.y_thresholds_ = np.asarray([0.0, 1.0])

        def predict(self, values: np.ndarray) -> np.ndarray:
            return np.full(len(values), 0.5)

    def fitter(values: np.ndarray, _: np.ndarray) -> FakeIsotonic:
        fitted_values.append(set(values.tolist()))
        return FakeIsotonic(values)

    probabilities, _, reports = tool._fit_oof_isotonic(
        scores, labels, folds, fitter=fitter
    )

    assert np.array_equal(probabilities, np.full(6, 0.5))
    for report, seen in zip(reports, fitted_values, strict=True):
        held = set(scores[folds == report["held_out_fold"]].tolist())
        assert held.isdisjoint(seen)
        assert report["held_out_fold"] not in report["training_folds"]


def test_aggregation_selection_uses_rank1_mrr_then_lexical_id() -> None:
    assert tool._select_aggregation(
        {
            "max": {"Rank-1": 0.7, "MRR": 0.8},
            "mean": {"Rank-1": 0.8, "MRR": 0.7},
            "median": {"Rank-1": 0.8, "MRR": 0.9},
        }
    ) == "median"
    assert tool._select_aggregation(
        {
            "mean": {"Rank-1": 0.8, "MRR": 0.9},
            "max": {"Rank-1": 0.8, "MRR": 0.9},
        }
    ) == "max"


def test_plan_hash_declaration_and_code_binding_fail_closed(tmp_path: Path) -> None:
    code = tmp_path / "bound.py"
    code.write_text("value = 1\n", encoding="utf-8")
    with patch.object(tool, "_CODE_PATHS", ("bound.py",)):
        plan = {"code_sha256s": tool._code_hashes(tmp_path)}
        tool._verify_code_bindings(plan, tmp_path)
        code.write_text("value = 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="source code differs"):
            tool._verify_code_bindings(plan, tmp_path)

    assignment, labels = _population_documents()
    populations = tool._build_holdout_populations(
        assignment, labels, expected_identities=2
    )
    private_plan = {"schema_version": tool.PLAN_SCHEMA, "populations": populations}
    plan_sha256 = content_sha256(private_plan)
    declaration = tool._declaration_for_plan(plan_sha256, populations)
    assert declaration.source_artifact_sha256 == plan_sha256
    assert declaration.records == tuple(
        sorted(declaration.records, key=lambda value: value.sample_token)
    )


def test_private_report_rejects_rows_ids_vectors_and_scores() -> None:
    tool._validate_private_report(
        {"status": "PASS", "metrics": {"Rank-1": 0.5}, "hash": "a" * 64}
    )
    for forbidden in (
        {"query_rows": []},
        {"nested": {"sample_token": "private"}},
        {"embeddings": [[1.0, 0.0]]},
        {"scores": [0.5]},
    ):
        with pytest.raises(ValueError, match="forbidden"):
            tool._validate_private_report(forbidden)


def test_channel_extraction_roles_are_strict() -> None:
    for roles in ((), ("unknown",), ("calibration", "calibration")):
        with pytest.raises(ValueError, match="roles"):
            tool._extract_holdout_channels(
                args=None,
                populations={},
                source_by_token={},
                sources=(),
                appearance_backbone=None,
                face_model=None,
                roles=roles,
            )


def test_overwrite_refusal_and_parser_subcommands(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        tool._refuse_existing((output,))

    prepare = tool._parse_args(
        ["prepare", *_common_cli(), "--output-dir", "/tmp/prepared"]
    )
    evaluate = tool._parse_args(
        [
            "evaluate",
            *_common_cli(),
            "--plan",
            "/tmp/prepared/plan.json",
            "--plan-sha256",
            "a" * 64,
            "--exposure-ledger",
            "/tmp/prepared/exposure-ledger.json",
            "--exposure-receipt",
            "/tmp/prepared/exposure-receipt.json",
            "--exposure-receipt-sha256",
            "a" * 64,
            "--output",
            "/tmp/report.json",
        ]
    )
    assert prepare.command == "prepare"
    assert evaluate.command == "evaluate"

    completed = subprocess.run(
        [sys.executable, ROOT / "workflows/evaluate_dogface_holdout_fusion.py", "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "{prepare,evaluate}" in completed.stdout
    assert list(tmp_path.iterdir()) == [output]
