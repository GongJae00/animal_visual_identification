"""Fail-closed task roles for the six-dataset retrospective research lane."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from foundation.provenance import content_sha256
from identity.research.research_cycle_admission import (
    ResearchLicenseLane,
    ResearchSourceAdmission,
    ResearchSourceAdmissions,
)


_DATASETS = (
    "ap10k-dog",
    "dogfacenet224",
    "dogflw",
    "mpdd",
    "sibetan",
    "yt-bb-dog",
)
_INTERPRETATION = (
    "RETROSPECTIVE_SIX_DATASET_RESEARCH_TASK_PLAN;"
    "EXPOSED_DIAGNOSTICS_ARE_NOT_FINAL_EVALUATION;"
    "NO_UNTOUCHED_OR_CROSS_SESSION_FINAL_CLAIM"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ResearchTask(StrEnum):
    LOCALIZATION = "LOCALIZATION"
    IDENTITY_TRAINING = "IDENTITY_TRAINING"
    SELF_SUPERVISION = "SELF_SUPERVISION"
    PROVISIONAL_IDENTITY_MINING = "PROVISIONAL_IDENTITY_MINING"
    IDENTITY_DEVELOPMENT = "IDENTITY_DEVELOPMENT"
    SCORE_CALIBRATION = "SCORE_CALIBRATION"
    EXPOSED_BENCHMARK = "EXPOSED_BENCHMARK"
    OPEN_SET = "OPEN_SET"
    ROBUSTNESS = "ROBUSTNESS"


class ResearchTaskRole(StrEnum):
    FIT = "FIT"
    DEV = "DEV"
    CAL = "CAL"
    EXPOSED_DIAGNOSTIC = "EXPOSED_DIAGNOSTIC"


_ALLOWED: dict[str, frozenset[tuple[ResearchTask, ResearchTaskRole]]] = {
    "ap10k-dog": frozenset(
        {
            (ResearchTask.LOCALIZATION, ResearchTaskRole.FIT),
            (ResearchTask.LOCALIZATION, ResearchTaskRole.DEV),
            (ResearchTask.LOCALIZATION, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
            (ResearchTask.SELF_SUPERVISION, ResearchTaskRole.FIT),
            (ResearchTask.PROVISIONAL_IDENTITY_MINING, ResearchTaskRole.FIT),
            (ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
        }
    ),
    "dogflw": frozenset(
        {
            (ResearchTask.LOCALIZATION, ResearchTaskRole.FIT),
            (ResearchTask.LOCALIZATION, ResearchTaskRole.DEV),
            (ResearchTask.LOCALIZATION, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
            (ResearchTask.SELF_SUPERVISION, ResearchTaskRole.FIT),
            (ResearchTask.PROVISIONAL_IDENTITY_MINING, ResearchTaskRole.FIT),
            (ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
        }
    ),
    "dogfacenet224": frozenset(
        {
            (ResearchTask.IDENTITY_TRAINING, ResearchTaskRole.FIT),
            (ResearchTask.SELF_SUPERVISION, ResearchTaskRole.FIT),
            (ResearchTask.IDENTITY_DEVELOPMENT, ResearchTaskRole.DEV),
            (ResearchTask.SCORE_CALIBRATION, ResearchTaskRole.CAL),
            (ResearchTask.EXPOSED_BENCHMARK, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
            (ResearchTask.OPEN_SET, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
            (ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
        }
    ),
    "mpdd": frozenset(
        {
            (ResearchTask.IDENTITY_DEVELOPMENT, ResearchTaskRole.DEV),
            (ResearchTask.EXPOSED_BENCHMARK, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
            (ResearchTask.OPEN_SET, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
            (ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
        }
    ),
    "sibetan": frozenset(
        {
            (ResearchTask.EXPOSED_BENCHMARK, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
            (ResearchTask.OPEN_SET, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
            (ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
        }
    ),
    "yt-bb-dog": frozenset(
        {
            (ResearchTask.IDENTITY_TRAINING, ResearchTaskRole.FIT),
            (ResearchTask.SELF_SUPERVISION, ResearchTaskRole.FIT),
            (ResearchTask.IDENTITY_DEVELOPMENT, ResearchTaskRole.DEV),
            (ResearchTask.SCORE_CALIBRATION, ResearchTaskRole.CAL),
            (ResearchTask.OPEN_SET, ResearchTaskRole.CAL),
            (ResearchTask.OPEN_SET, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
            (ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC),
        }
    ),
}


@dataclass(frozen=True, order=True, slots=True)
class ResearchTaskAssignment:
    dataset_name: str
    task: ResearchTask
    role: ResearchTaskRole
    source_partition: str
    schema_version: str = "cvi.research_task_assignment.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.research_task_assignment.v1":
            raise ValueError("unsupported research task assignment schema")
        if self.dataset_name not in _ALLOWED:
            raise ValueError("unsupported research task dataset")
        if not isinstance(self.task, ResearchTask) or not isinstance(
            self.role, ResearchTaskRole
        ):
            raise TypeError("research task and role must use their exact enums")
        if (self.task, self.role) not in _ALLOWED[self.dataset_name]:
            raise ValueError(
                f"{self.dataset_name} is not admitted for {self.task.value}/{self.role.value}"
            )
        if (
            not isinstance(self.source_partition, str)
            or not self.source_partition
            or self.source_partition != self.source_partition.strip()
            or len(self.source_partition.encode("utf-8")) > 256
        ):
            raise ValueError("source_partition must be bounded non-empty text")
        if (
            self.dataset_name in {"ap10k-dog", "dogflw"}
            and self.task
            in {ResearchTask.SELF_SUPERVISION, ResearchTask.PROVISIONAL_IDENTITY_MINING}
            and "test" in self.source_partition.lower()
        ):
            raise ValueError("auxiliary SSL and GenID mining must not use a test partition")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "task": self.task.value,
            "role": self.role.value,
            "source_partition": self.source_partition,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchTaskAssignment":
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("research task assignment fields differ")
        return cls(
            dataset_name=payload["dataset_name"],
            task=ResearchTask(payload["task"]),
            role=ResearchTaskRole(payload["role"]),
            source_partition=payload["source_partition"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ResearchTaskPlan:
    plan_name: str
    source_admissions_sha256: str
    source_admissions: tuple[ResearchSourceAdmission, ...]
    assignments: tuple[ResearchTaskAssignment, ...]
    final_evaluation_permitted: bool = False
    interpretation: str = _INTERPRETATION
    schema_version: str = "cvi.research_task_plan.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.research_task_plan.v1":
            raise ValueError("unsupported research task plan schema")
        if not isinstance(self.plan_name, str) or not self.plan_name.strip():
            raise ValueError("plan_name must be non-empty text")
        if not isinstance(self.source_admissions_sha256, str) or _SHA256.fullmatch(
            self.source_admissions_sha256
        ) is None:
            raise ValueError("source_admissions_sha256 must be lowercase SHA-256")
        admissions = ResearchSourceAdmissions(self.source_admissions)
        if admissions.admissions_sha256 != self.source_admissions_sha256:
            raise ValueError("research task plan source admissions digest differs")
        if self.final_evaluation_permitted is not False:
            raise ValueError("retrospective research task plan cannot permit final evaluation")
        if self.interpretation != _INTERPRETATION:
            raise ValueError("research task plan interpretation differs")
        if (
            not isinstance(self.assignments, tuple)
            or not self.assignments
            or any(not isinstance(item, ResearchTaskAssignment) for item in self.assignments)
            or self.assignments != tuple(sorted(set(self.assignments)))
        ):
            raise ValueError("research task assignments must be sorted and unique")
        datasets = {item.dataset_name for item in self.assignments}
        if datasets != set(_DATASETS):
            raise ValueError("research task plan must use every admitted dataset")
        admissions_by_dataset = {
            item.dataset_name: item for item in self.source_admissions
        }
        if (
            admissions_by_dataset["dogflw"].license_lane
            is not ResearchLicenseLane.RESEARCH_ONLY
        ):
            raise ValueError("DogFLW task lineage must remain research-only")

    @property
    def plan_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_name": self.plan_name,
            "source_admissions_sha256": self.source_admissions_sha256,
            "source_admissions": [item.to_dict() for item in self.source_admissions],
            "assignments": [item.to_dict() for item in self.assignments],
            "final_evaluation_permitted": self.final_evaluation_permitted,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchTaskPlan":
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("research task plan fields differ")
        if not isinstance(payload["source_admissions"], list) or not isinstance(
            payload["assignments"], list
        ):
            raise TypeError("research task plan arrays differ")
        return cls(
            plan_name=payload["plan_name"],
            source_admissions_sha256=payload["source_admissions_sha256"],
            source_admissions=tuple(
                ResearchSourceAdmission.from_dict(item)
                for item in payload["source_admissions"]
            ),
            assignments=tuple(
                ResearchTaskAssignment.from_dict(item)
                for item in payload["assignments"]
            ),
            final_evaluation_permitted=payload["final_evaluation_permitted"],
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


def build_primary_research_task_plan(
    source_admissions: ResearchSourceAdmissions,
    *,
    plan_name: str = "six-dataset-robust-reid-v1",
) -> ResearchTaskPlan:
    """Build the frozen main-line task matrix without target-dataset adaptation."""

    uses = (
        ("ap10k-dog", ResearchTask.LOCALIZATION, ResearchTaskRole.FIT, "official-train"),
        ("ap10k-dog", ResearchTask.LOCALIZATION, ResearchTaskRole.DEV, "official-val"),
        ("ap10k-dog", ResearchTask.LOCALIZATION, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "official-test"),
        ("ap10k-dog", ResearchTask.SELF_SUPERVISION, ResearchTaskRole.FIT, "official-train"),
        ("ap10k-dog", ResearchTask.PROVISIONAL_IDENTITY_MINING, ResearchTaskRole.FIT, "official-train"),
        ("ap10k-dog", ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "official-test"),
        ("dogfacenet224", ResearchTask.IDENTITY_TRAINING, ResearchTaskRole.FIT, "official-train:fit"),
        ("dogfacenet224", ResearchTask.SELF_SUPERVISION, ResearchTaskRole.FIT, "official-train:fit"),
        ("dogfacenet224", ResearchTask.IDENTITY_DEVELOPMENT, ResearchTaskRole.DEV, "official-train:dev"),
        ("dogfacenet224", ResearchTask.SCORE_CALIBRATION, ResearchTaskRole.CAL, "official-train:cal"),
        ("dogfacenet224", ResearchTask.EXPOSED_BENCHMARK, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "official-test"),
        ("dogfacenet224", ResearchTask.OPEN_SET, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "official-test"),
        ("dogfacenet224", ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "official-test"),
        ("dogflw", ResearchTask.LOCALIZATION, ResearchTaskRole.FIT, "publisher-train:fit"),
        ("dogflw", ResearchTask.LOCALIZATION, ResearchTaskRole.DEV, "publisher-train:deterministic-dev"),
        ("dogflw", ResearchTask.LOCALIZATION, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "publisher-test"),
        ("dogflw", ResearchTask.SELF_SUPERVISION, ResearchTaskRole.FIT, "publisher-train:fit"),
        ("dogflw", ResearchTask.PROVISIONAL_IDENTITY_MINING, ResearchTaskRole.FIT, "publisher-train:fit"),
        ("dogflw", ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "publisher-test"),
        ("mpdd", ResearchTask.EXPOSED_BENCHMARK, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "publisher-query-gallery"),
        ("mpdd", ResearchTask.OPEN_SET, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "protected-known64-unknown32"),
        ("mpdd", ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "publisher-query-gallery"),
        ("sibetan", ResearchTask.EXPOSED_BENCHMARK, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "no-mono-cross-sequence"),
        ("sibetan", ResearchTask.OPEN_SET, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "known39-unknown20"),
        ("sibetan", ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "all-sequences"),
        ("yt-bb-dog", ResearchTask.IDENTITY_TRAINING, ResearchTaskRole.FIT, "official-train:fit"),
        ("yt-bb-dog", ResearchTask.SELF_SUPERVISION, ResearchTaskRole.FIT, "official-train:fit"),
        ("yt-bb-dog", ResearchTask.IDENTITY_DEVELOPMENT, ResearchTaskRole.DEV, "official-train:dev"),
        ("yt-bb-dog", ResearchTask.SCORE_CALIBRATION, ResearchTaskRole.CAL, "official-train:cal"),
        ("yt-bb-dog", ResearchTask.OPEN_SET, ResearchTaskRole.CAL, "known300-unknown300"),
        ("yt-bb-dog", ResearchTask.OPEN_SET, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "official-test-known300-unknown423"),
        ("yt-bb-dog", ResearchTask.ROBUSTNESS, ResearchTaskRole.EXPOSED_DIAGNOSTIC, "paired-random-background"),
    )
    assignments = tuple(
        sorted(ResearchTaskAssignment(*values) for values in uses)
    )
    return ResearchTaskPlan(
        plan_name=plan_name,
        source_admissions_sha256=source_admissions.admissions_sha256,
        source_admissions=source_admissions.sources,
        assignments=assignments,
    )


__all__ = [
    "ResearchTask",
    "ResearchTaskAssignment",
    "ResearchTaskPlan",
    "ResearchTaskRole",
    "build_primary_research_task_plan",
]
