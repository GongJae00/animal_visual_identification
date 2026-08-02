from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from foundation.provenance import content_sha256
from identity_governance.research_cycle_admission import (
    IdentityTargetMode,
    ResearchLicenseLane,
    ResearchSourceAdmission,
    ResearchSourceAdmissions,
    ResearchSourceRole,
)
from identity_governance.research_task_plan import (
    ResearchTask,
    ResearchTaskAssignment,
    ResearchTaskPlan,
    ResearchTaskRole,
    build_primary_research_task_plan,
)


DATASETS = (
    "ap10k-dog",
    "dogfacenet224",
    "dogflw",
    "mpdd",
    "sibetan",
    "yt-bb-dog",
)


def _admissions(*, dogflw_lane: ResearchLicenseLane = ResearchLicenseLane.RESEARCH_ONLY):
    return ResearchSourceAdmissions(
        tuple(
            ResearchSourceAdmission(
                dataset_name=dataset,
                source_manifest_sha256=content_sha256({"dataset": dataset}),
                license_id=("CC-BY-NC-4.0" if dataset == "dogflw" else "CC-BY-4.0"),
                license_lane=(dogflw_lane if dataset == "dogflw" else ResearchLicenseLane.RESEARCH_ONLY),
                source_role=(
                    ResearchSourceRole.AUXILIARY_ONLY
                    if dataset in {"ap10k-dog", "dogflw"}
                    else ResearchSourceRole.IDENTITY_RESEARCH
                ),
                identity_target_mode=(
                    IdentityTargetMode.NONE
                    if dataset in {"ap10k-dog", "dogflw"}
                    else IdentityTargetMode.CANONICAL_REGISTERED_UUIDV5
                ),
            )
            for dataset in DATASETS
        )
    )


def test_primary_plan_uses_all_datasets_and_round_trips() -> None:
    plan = build_primary_research_task_plan(_admissions())

    assert {item.dataset_name for item in plan.assignments} == set(DATASETS)
    assert plan.final_evaluation_permitted is False
    assert ResearchTaskPlan.from_dict(plan.to_dict()) == plan
    assert plan.plan_sha256 == content_sha256(plan.to_dict())
    sibetan = [item for item in plan.assignments if item.dataset_name == "sibetan"]
    assert {item.role for item in sibetan} == {ResearchTaskRole.EXPOSED_DIAGNOSTIC}
    assert ResearchTask.SCORE_CALIBRATION not in {item.task for item in sibetan}


@pytest.mark.parametrize("dataset", ("ap10k-dog", "dogflw"))
def test_auxiliary_data_reject_identity_training(dataset: str) -> None:
    with pytest.raises(ValueError, match="not admitted"):
        ResearchTaskAssignment(
            dataset,
            ResearchTask.IDENTITY_TRAINING,
            ResearchTaskRole.FIT,
            "train",
        )


@pytest.mark.parametrize("dataset", ("ap10k-dog", "dogflw"))
def test_auxiliary_data_admits_train_only_ssl_and_provisional_genid(dataset: str) -> None:
    ssl = ResearchTaskAssignment(
        dataset,
        ResearchTask.SELF_SUPERVISION,
        ResearchTaskRole.FIT,
        "publisher-train",
    )
    genid = ResearchTaskAssignment(
        dataset,
        ResearchTask.PROVISIONAL_IDENTITY_MINING,
        ResearchTaskRole.FIT,
        "publisher-train",
    )
    assert ssl.role is ResearchTaskRole.FIT
    assert genid.role is ResearchTaskRole.FIT

    with pytest.raises(ValueError, match="must not use a test partition"):
        ResearchTaskAssignment(
            dataset,
            ResearchTask.PROVISIONAL_IDENTITY_MINING,
            ResearchTaskRole.FIT,
            "publisher-test",
        )


def test_sibetan_rejects_calibration_and_yt_rejects_external_benchmark() -> None:
    with pytest.raises(ValueError, match="not admitted"):
        ResearchTaskAssignment(
            "sibetan",
            ResearchTask.SCORE_CALIBRATION,
            ResearchTaskRole.CAL,
            "all",
        )
    with pytest.raises(ValueError, match="not admitted"):
        ResearchTaskAssignment(
            "yt-bb-dog",
            ResearchTask.EXPOSED_BENCHMARK,
            ResearchTaskRole.EXPOSED_DIAGNOSTIC,
            "official-test",
        )


def test_plan_rejects_missing_dataset_final_claim_and_dogflw_lane() -> None:
    plan = build_primary_research_task_plan(_admissions())
    payload = plan.to_dict()
    payload["assignments"] = [
        item for item in payload["assignments"] if item["dataset_name"] != "mpdd"
    ]
    with pytest.raises(ValueError, match="every admitted dataset"):
        ResearchTaskPlan.from_dict(payload)

    payload = plan.to_dict()
    payload["final_evaluation_permitted"] = True
    with pytest.raises(ValueError, match="cannot permit final evaluation"):
        ResearchTaskPlan.from_dict(payload)

    commercial = _admissions(dogflw_lane=ResearchLicenseLane.COMMERCIAL_ALLOWED)
    with pytest.raises(ValueError, match="DogFLW.*research-only"):
        build_primary_research_task_plan(commercial)


def test_rehashed_tamper_cannot_turn_sibetan_into_calibration() -> None:
    payload = copy.deepcopy(build_primary_research_task_plan(_admissions()).to_dict())
    row = next(item for item in payload["assignments"] if item["dataset_name"] == "sibetan")
    row["task"] = "SCORE_CALIBRATION"
    row["role"] = "CAL"

    with pytest.raises(ValueError, match="not admitted"):
        ResearchTaskPlan.from_dict(payload)


def test_source_checkout_cli_builds_task_plan(tmp_path: Path) -> None:
    admissions_path = tmp_path / "admissions.json"
    output_path = tmp_path / "plan.json"
    admissions_path.write_text(
        json.dumps(_admissions().to_dict()), encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "workflows/build_research_task_plan.py",
            "--source-admissions",
            str(admissions_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = ResearchTaskPlan.from_dict(json.loads(output_path.read_text(encoding="utf-8")))
    summary = json.loads(completed.stdout)
    assert summary["plan_sha256"] == plan.plan_sha256
    assert summary["dataset_count"] == 6
    assert summary["final_evaluation_permitted"] is False
