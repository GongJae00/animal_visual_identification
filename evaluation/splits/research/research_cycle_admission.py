"""Retrospective research-only admission over the public canine source bundle.

This contract deliberately does not revise or reinterpret the protected split.
It records prior exposure as an immutable audit fact while creating a separate,
non-final research cycle over duplicate-closed allocation blocks.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping

from enrollment.registry.identity_registry import (
    REGISTERED_DOG_NAMESPACE,
    compute_identity_token,
    compute_registered_dog_id,
)
from evaluation.splits.protected_public_split import (
    FrozenPublicSplitEvidenceGraph,
    PublicSplitSourceBundle,
    _build_allocation_blocks,
    _close_components,
    _propagate_component_quarantine,
    _validate_dependency_edges,
    _validate_graph_references,
)
from shared.foundation.provenance import content_sha256
from evaluation.splits.role_exposure import ExposureStage, RoleExposureLedger, RoleExposureReceipt
from evaluation.splits.split_role_exposure import verify_split_role_exposure_inputs


RESEARCH_CYCLE_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_DNS, "evaluation.research_cycle.v1"
)

_IDENTITY_DATASETS = frozenset(
    {"dogfacenet224", "yt-bb-dog", "mpdd", "sibetan"}
)
_AUXILIARY_DATASETS = frozenset({"ap10k-dog", "dogflw"})
_REQUIRED_DATASETS = _IDENTITY_DATASETS | _AUXILIARY_DATASETS
_TARGET_PERCENT = {"RESEARCH_FIT": 70, "RESEARCH_DEV": 15, "RESEARCH_CAL": 15}
_INTERPRETATION = (
    "RETROSPECTIVE_RESEARCH_ONLY;HISTORICAL_MAXIMUM_EXPOSURE_IS_UNCHANGED;"
    "NO_FINAL_EVALUATION_OR_HOLDOUT_INTERPRETATION"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ResearchRole(StrEnum):
    RESEARCH_FIT = "RESEARCH_FIT"
    RESEARCH_DEV = "RESEARCH_DEV"
    RESEARCH_CAL = "RESEARCH_CAL"


class ResearchSourceRole(StrEnum):
    IDENTITY_RESEARCH = "IDENTITY_RESEARCH"
    AUXILIARY_ONLY = "AUXILIARY_ONLY"


class ResearchLicenseLane(StrEnum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    COMMERCIAL_ALLOWED = "COMMERCIAL_ALLOWED"


class IdentityTargetMode(StrEnum):
    CANONICAL_REGISTERED_UUIDV5 = "CANONICAL_REGISTERED_UUIDV5"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class ResearchSourceAdmission:
    dataset_name: str
    source_manifest_sha256: str
    license_id: str
    license_lane: ResearchLicenseLane
    source_role: ResearchSourceRole
    identity_target_mode: IdentityTargetMode
    schema_version: str = "evaluation.research_source_admission.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evaluation.research_source_admission.v1":
            raise ValueError("unsupported research source admission schema")
        if self.dataset_name not in _REQUIRED_DATASETS:
            raise ValueError("unsupported research source dataset")
        _require_sha256(self.source_manifest_sha256, "source_manifest_sha256")
        _require_text(self.license_id, "license_id")
        if not isinstance(self.license_lane, ResearchLicenseLane):
            raise TypeError("license_lane must be ResearchLicenseLane")
        if not isinstance(self.source_role, ResearchSourceRole):
            raise TypeError("source_role must be ResearchSourceRole")
        if not isinstance(self.identity_target_mode, IdentityTargetMode):
            raise TypeError("identity_target_mode must be IdentityTargetMode")
        if self.dataset_name in _IDENTITY_DATASETS:
            if self.source_role is not ResearchSourceRole.IDENTITY_RESEARCH:
                raise ValueError("identity dataset must use the identity research role")
            if (
                self.identity_target_mode
                is not IdentityTargetMode.CANONICAL_REGISTERED_UUIDV5
            ):
                raise ValueError("identity dataset must use canonical UUIDv5 targets")
        else:
            if self.source_role is not ResearchSourceRole.AUXILIARY_ONLY:
                raise ValueError("AP-10K and DogFLW must use auxiliary-only roles")
            if self.identity_target_mode is not IdentityTargetMode.NONE:
                raise ValueError("AP-10K and DogFLW must never receive identity targets")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "source_manifest_sha256": self.source_manifest_sha256,
            "license_id": self.license_id,
            "license_lane": self.license_lane.value,
            "source_role": self.source_role.value,
            "identity_target_mode": self.identity_target_mode.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchSourceAdmission":
        _exact_keys(payload, set(cls.__dataclass_fields__), "research source admission")
        return cls(
            dataset_name=payload["dataset_name"],
            source_manifest_sha256=payload["source_manifest_sha256"],
            license_id=payload["license_id"],
            license_lane=ResearchLicenseLane(payload["license_lane"]),
            source_role=ResearchSourceRole(payload["source_role"]),
            identity_target_mode=IdentityTargetMode(payload["identity_target_mode"]),
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ResearchSourceAdmissions:
    sources: tuple[ResearchSourceAdmission, ...]
    interpretation: str = "EXPLICIT_LICENSED_RESEARCH_SOURCES_ONLY"
    schema_version: str = "evaluation.research_source_admissions.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evaluation.research_source_admissions.v1":
            raise ValueError("unsupported research source admissions schema")
        if self.interpretation != "EXPLICIT_LICENSED_RESEARCH_SOURCES_ONLY":
            raise ValueError("research source admissions interpretation differs")
        if not isinstance(self.sources, tuple) or any(
            not isinstance(item, ResearchSourceAdmission) for item in self.sources
        ):
            raise TypeError("research sources must be ResearchSourceAdmission values")
        if self.sources != tuple(sorted(self.sources, key=lambda item: item.dataset_name)):
            raise ValueError("research sources must be canonically sorted")
        datasets = tuple(item.dataset_name for item in self.sources)
        if len(datasets) != len(set(datasets)) or set(datasets) != _REQUIRED_DATASETS:
            raise ValueError("research sources must contain each required dataset exactly once")

    @property
    def admissions_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sources": [item.to_dict() for item in self.sources],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchSourceAdmissions":
        _exact_keys(payload, set(cls.__dataclass_fields__), "research source admissions")
        if not isinstance(payload["sources"], list):
            raise TypeError("research sources must be a JSON array")
        return cls(
            sources=tuple(ResearchSourceAdmission.from_dict(item) for item in payload["sources"]),
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ResearchIdentityAssignment:
    identity_token: str
    dataset_identity_id: str
    registered_dog_id: str
    dataset_name: str
    role: ResearchRole | None
    allocation_block_token: str
    component_tokens: tuple[str, ...]
    historical_maximum_exposure: ExposureStage | None
    quarantine_reasons: tuple[str, ...]
    schema_version: str = "evaluation.research_identity_assignment.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evaluation.research_identity_assignment.v1":
            raise ValueError("unsupported research identity assignment schema")
        _require_sha256(self.identity_token, "identity_token")
        _require_text(self.dataset_identity_id, "dataset_identity_id")
        if self.identity_token != compute_identity_token(self.dataset_identity_id):
            raise ValueError("research identity token is not deterministic")
        _require_uuid5(self.registered_dog_id, "registered_dog_id")
        if self.registered_dog_id != compute_registered_dog_id(
            self.dataset_identity_id
        ):
            raise ValueError("registered dog UUIDv5 is not deterministic in its namespace")
        if self.dataset_name not in _IDENTITY_DATASETS:
            raise ValueError("research identity target uses an auxiliary dataset")
        if self.dataset_identity_id.split(":", 1)[0] != self.dataset_name:
            raise ValueError("research identity dataset name is not canonical")
        if self.role is not None and not isinstance(self.role, ResearchRole):
            raise TypeError("research identity role must be ResearchRole or null")
        _require_sha256(self.allocation_block_token, "allocation_block_token")
        _require_digest_tuple(self.component_tokens, "component_tokens")
        if self.historical_maximum_exposure is not None and not isinstance(
            self.historical_maximum_exposure, ExposureStage
        ):
            raise TypeError("historical maximum exposure must be ExposureStage or null")
        _require_string_tuple(
            self.quarantine_reasons, "quarantine_reasons", allow_empty=True
        )
        if (self.role is None) != bool(self.quarantine_reasons):
            raise ValueError("quarantined identities must have reasons and no research role")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity_token": self.identity_token,
            "dataset_identity_id": self.dataset_identity_id,
            "registered_dog_id": self.registered_dog_id,
            "dataset_name": self.dataset_name,
            "role": None if self.role is None else self.role.value,
            "allocation_block_token": self.allocation_block_token,
            "component_tokens": list(self.component_tokens),
            "historical_maximum_exposure": (
                None
                if self.historical_maximum_exposure is None
                else self.historical_maximum_exposure.value
            ),
            "quarantine_reasons": list(self.quarantine_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchIdentityAssignment":
        _exact_keys(payload, set(cls.__dataclass_fields__), "research identity assignment")
        return cls(
            identity_token=payload["identity_token"],
            dataset_identity_id=payload["dataset_identity_id"],
            registered_dog_id=payload["registered_dog_id"],
            dataset_name=payload["dataset_name"],
            role=None if payload["role"] is None else ResearchRole(payload["role"]),
            allocation_block_token=payload["allocation_block_token"],
            component_tokens=_string_tuple(payload["component_tokens"], "component_tokens"),
            historical_maximum_exposure=(
                None
                if payload["historical_maximum_exposure"] is None
                else ExposureStage(payload["historical_maximum_exposure"])
            ),
            quarantine_reasons=_string_tuple(
                payload["quarantine_reasons"], "quarantine_reasons"
            ),
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ResearchSampleAssignment:
    sample_token: str
    identity_token: str
    sequence_token: str
    component_token: str
    source_variant: str
    role: ResearchRole | None
    schema_version: str = "evaluation.research_sample_assignment.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evaluation.research_sample_assignment.v1":
            raise ValueError("unsupported research sample assignment schema")
        for value, name in (
            (self.sample_token, "sample_token"),
            (self.identity_token, "identity_token"),
            (self.sequence_token, "sequence_token"),
            (self.component_token, "component_token"),
        ):
            _require_sha256(value, name)
        _require_text(self.source_variant, "source_variant")
        if self.role is not None and not isinstance(self.role, ResearchRole):
            raise TypeError("research sample role must be ResearchRole or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_token": self.sample_token,
            "identity_token": self.identity_token,
            "sequence_token": self.sequence_token,
            "component_token": self.component_token,
            "source_variant": self.source_variant,
            "role": None if self.role is None else self.role.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchSampleAssignment":
        _exact_keys(payload, set(cls.__dataclass_fields__), "research sample assignment")
        values = dict(payload)
        values["role"] = None if values["role"] is None else ResearchRole(values["role"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ResearchCycleManifest:
    cycle_name: str
    cycle_id: str
    cycle_namespace_uuid: str
    registered_identity_namespace_uuid: str
    source_bundle_sha256: str
    dependency_graph_sha256: str
    source_admissions_sha256: str
    source_admissions: tuple[ResearchSourceAdmission, ...]
    role_exposure_ledger_sha256: str
    role_exposure_receipt_sha256: str
    target_percentages: tuple[tuple[str, int], ...]
    dataset_role_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    identity_assignments: tuple[ResearchIdentityAssignment, ...]
    sample_assignments: tuple[ResearchSampleAssignment, ...]
    score_inputs_used: bool = False
    final_evaluation_permitted: bool = False
    interpretation: str = _INTERPRETATION
    schema_version: str = "evaluation.research_cycle_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evaluation.research_cycle_manifest.v1":
            raise ValueError("unsupported research cycle manifest schema")
        _require_text(self.cycle_name, "cycle_name")
        if self.cycle_namespace_uuid != str(RESEARCH_CYCLE_NAMESPACE):
            raise ValueError("research cycle namespace UUID differs")
        if self.cycle_id != compute_research_cycle_id(self.cycle_name):
            raise ValueError("research cycle UUIDv5 is not deterministic")
        if self.registered_identity_namespace_uuid != str(
            REGISTERED_DOG_NAMESPACE
        ):
            raise ValueError("registered identity namespace UUID differs")
        for value, name in (
            (self.source_bundle_sha256, "source_bundle_sha256"),
            (self.dependency_graph_sha256, "dependency_graph_sha256"),
            (self.source_admissions_sha256, "source_admissions_sha256"),
            (self.role_exposure_ledger_sha256, "role_exposure_ledger_sha256"),
            (self.role_exposure_receipt_sha256, "role_exposure_receipt_sha256"),
        ):
            _require_sha256(value, name)
        admitted_sources = ResearchSourceAdmissions(self.source_admissions)
        if admitted_sources.admissions_sha256 != self.source_admissions_sha256:
            raise ValueError("embedded research source admissions hash differs")
        if self.target_percentages != tuple(_TARGET_PERCENT.items()):
            raise ValueError("research target percentages must be 70/15/15")
        if self.score_inputs_used is not False:
            raise ValueError("research-cycle assignment must be score-blind")
        if self.final_evaluation_permitted is not False:
            raise ValueError("research-cycle manifest cannot permit final evaluation")
        if self.interpretation != _INTERPRETATION:
            raise ValueError("research-cycle interpretation differs")
        if not self.identity_assignments or not self.sample_assignments:
            raise ValueError("research-cycle assignments must not be empty")
        if self.identity_assignments != tuple(
            sorted(self.identity_assignments, key=lambda item: item.identity_token)
        ):
            raise ValueError("research identity assignments must be canonically sorted")
        if self.sample_assignments != tuple(
            sorted(self.sample_assignments, key=lambda item: item.sample_token)
        ):
            raise ValueError("research sample assignments must be canonically sorted")
        self._validate_assignment_closure()

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def _validate_assignment_closure(self) -> None:
        identities = {item.identity_token: item for item in self.identity_assignments}
        if len(identities) != len(self.identity_assignments):
            raise ValueError("research identity assignments repeat an identity")
        sample_tokens: set[str] = set()
        role_by_component: dict[str, ResearchRole | None] = {}
        role_by_sequence: dict[str, ResearchRole | None] = {}
        for item in self.identity_assignments:
            for component in item.component_tokens:
                prior = role_by_component.setdefault(component, item.role)
                if prior != item.role:
                    raise ValueError("component crosses research roles")
        for sample in self.sample_assignments:
            if sample.sample_token in sample_tokens:
                raise ValueError("research sample assignments repeat a sample")
            sample_tokens.add(sample.sample_token)
            identity = identities.get(sample.identity_token)
            if identity is None or identity.role != sample.role:
                raise ValueError("sample and identity research roles differ")
            if sample.component_token not in identity.component_tokens:
                raise ValueError("sample component is absent from identity allocation block")
            prior_component = role_by_component.setdefault(sample.component_token, sample.role)
            if prior_component != sample.role:
                raise ValueError("component crosses research roles")
            prior_sequence = role_by_sequence.setdefault(sample.sequence_token, sample.role)
            if prior_sequence != sample.role:
                raise ValueError("sequence crosses research roles")
        expected_counts = _dataset_role_counts(self.identity_assignments)
        if self.dataset_role_counts != expected_counts:
            raise ValueError("research dataset role counts differ from assignments")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cycle_name": self.cycle_name,
            "cycle_id": self.cycle_id,
            "cycle_namespace_uuid": self.cycle_namespace_uuid,
            "registered_identity_namespace_uuid": self.registered_identity_namespace_uuid,
            "source_bundle_sha256": self.source_bundle_sha256,
            "dependency_graph_sha256": self.dependency_graph_sha256,
            "source_admissions_sha256": self.source_admissions_sha256,
            "source_admissions": [item.to_dict() for item in self.source_admissions],
            "role_exposure_ledger_sha256": self.role_exposure_ledger_sha256,
            "role_exposure_receipt_sha256": self.role_exposure_receipt_sha256,
            "target_percentages": dict(self.target_percentages),
            "dataset_role_counts": {
                dataset: dict(counts) for dataset, counts in self.dataset_role_counts
            },
            "identity_assignments": [item.to_dict() for item in self.identity_assignments],
            "sample_assignments": [item.to_dict() for item in self.sample_assignments],
            "score_inputs_used": self.score_inputs_used,
            "final_evaluation_permitted": self.final_evaluation_permitted,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchCycleManifest":
        _exact_keys(payload, set(cls.__dataclass_fields__), "research cycle manifest")
        if not isinstance(payload["target_percentages"], Mapping) or not isinstance(
            payload["dataset_role_counts"], Mapping
        ):
            raise TypeError("research-cycle count fields must be objects")
        if not isinstance(payload["identity_assignments"], list) or not isinstance(
            payload["sample_assignments"], list
        ):
            raise TypeError("research-cycle assignments must be JSON arrays")
        if not isinstance(payload["source_admissions"], list):
            raise TypeError("research source admissions must be a JSON array")
        target_percentages = payload["target_percentages"]
        if set(target_percentages) != set(_TARGET_PERCENT):
            raise ValueError("research target percentage fields differ")
        count_keys = (*tuple(role.value for role in ResearchRole), "QUARANTINED")
        dataset_role_counts: list[tuple[str, tuple[tuple[str, int], ...]]] = []
        for dataset in sorted(payload["dataset_role_counts"]):
            counts = payload["dataset_role_counts"][dataset]
            if not isinstance(counts, Mapping) or set(counts) != set(count_keys):
                raise ValueError("research dataset role count fields differ")
            dataset_role_counts.append(
                (dataset, tuple((key, counts[key]) for key in count_keys))
            )
        return cls(
            cycle_name=payload["cycle_name"],
            cycle_id=payload["cycle_id"],
            cycle_namespace_uuid=payload["cycle_namespace_uuid"],
            registered_identity_namespace_uuid=payload[
                "registered_identity_namespace_uuid"
            ],
            source_bundle_sha256=payload["source_bundle_sha256"],
            dependency_graph_sha256=payload["dependency_graph_sha256"],
            source_admissions_sha256=payload["source_admissions_sha256"],
            source_admissions=tuple(
                ResearchSourceAdmission.from_dict(item)
                for item in payload["source_admissions"]
            ),
            role_exposure_ledger_sha256=payload["role_exposure_ledger_sha256"],
            role_exposure_receipt_sha256=payload["role_exposure_receipt_sha256"],
            target_percentages=tuple(
                (role, target_percentages[role]) for role in _TARGET_PERCENT
            ),
            dataset_role_counts=tuple(dataset_role_counts),
            identity_assignments=tuple(
                ResearchIdentityAssignment.from_dict(item)
                for item in payload["identity_assignments"]
            ),
            sample_assignments=tuple(
                ResearchSampleAssignment.from_dict(item)
                for item in payload["sample_assignments"]
            ),
            score_inputs_used=payload["score_inputs_used"],
            final_evaluation_permitted=payload["final_evaluation_permitted"],
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


def compute_research_cycle_id(cycle_name: str) -> str:
    _require_text(cycle_name, "cycle_name")
    return str(uuid.uuid5(RESEARCH_CYCLE_NAMESPACE, cycle_name))


def build_research_cycle_manifest(
    *,
    cycle_name: str,
    source: PublicSplitSourceBundle,
    graph: FrozenPublicSplitEvidenceGraph,
    source_admissions: ResearchSourceAdmissions,
    role_exposure_ledger: RoleExposureLedger,
    role_exposure_receipt: RoleExposureReceipt,
) -> ResearchCycleManifest:
    """Assign all admissible identity blocks to deterministic research roles."""

    if source.evidence_bindings != graph.evidence_bindings:
        raise ValueError("source bundle and dependency graph evidence bindings differ")
    if {sample.dataset_name for sample in source.samples} != _IDENTITY_DATASETS:
        raise ValueError("research source bundle must cover all four identity datasets")
    history = verify_split_role_exposure_inputs(
        source.samples, role_exposure_ledger, role_exposure_receipt
    )
    samples_by_token = {sample.sample_token: sample for sample in source.samples}
    source_id_to_token = {
        sample.source_sample_id: sample.sample_token for sample in source.samples
    }
    _validate_graph_references(graph, samples_by_token)
    _validate_dependency_edges(source.samples, graph.edges, source_id_to_token)

    components, _, quarantine_reasons = _close_components(source.samples, graph.edges)
    blocks, block_by_identity = _build_allocation_blocks(components)
    _propagate_component_quarantine(blocks, quarantine_reasons)
    component_by_sample = {
        sample.sample_token: component
        for component in components
        for sample in component.samples
    }
    dataset_by_identity: dict[str, str] = {}
    label_by_identity: dict[str, str] = {}
    for sample in source.samples:
        prior_dataset = dataset_by_identity.setdefault(
            sample.identity_token, sample.dataset_name
        )
        prior_label = label_by_identity.setdefault(
            sample.identity_token, sample.dataset_identity_id
        )
        if prior_dataset != sample.dataset_name or prior_label != sample.dataset_identity_id:
            raise ValueError("source identity crosses dataset or canonical label")

    quarantined_blocks = {
        block.token
        for block in blocks
        if any(component in quarantine_reasons for component in block.component_tokens)
    }
    assignable = tuple(block for block in blocks if block.token not in quarantined_blocks)
    role_by_block = _assign_blocks(
        assignable,
        dataset_by_identity,
        cycle_name=cycle_name,
        evidence_root=content_sha256(
            {
                "source_bundle_sha256": source.bundle_sha256,
                "dependency_graph_sha256": graph.graph_sha256,
                "source_admissions_sha256": source_admissions.admissions_sha256,
                "role_exposure_ledger_sha256": role_exposure_ledger.ledger_sha256,
                "role_exposure_receipt_sha256": role_exposure_receipt.receipt_sha256,
            }
        ),
    )

    identity_assignments: list[ResearchIdentityAssignment] = []
    for identity_token in sorted(dataset_by_identity):
        block = block_by_identity[identity_token]
        reasons = tuple(
            sorted(
                {
                    reason
                    for component in block.component_tokens
                    for reason in quarantine_reasons.get(component, ())
                }
            )
        )
        role = role_by_block.get(block.token)
        identity_assignments.append(
            ResearchIdentityAssignment(
                identity_token=identity_token,
                dataset_identity_id=label_by_identity[identity_token],
                registered_dog_id=compute_registered_dog_id(
                    label_by_identity[identity_token]
                ),
                dataset_name=dataset_by_identity[identity_token],
                role=role,
                allocation_block_token=block.token,
                component_tokens=block.component_tokens,
                historical_maximum_exposure=history.get(identity_token),
                quarantine_reasons=reasons,
            )
        )
    role_by_identity = {item.identity_token: item.role for item in identity_assignments}
    sample_assignments = tuple(
        ResearchSampleAssignment(
            sample_token=sample.sample_token,
            identity_token=sample.identity_token,
            sequence_token=sample.sequence_token,
            component_token=component_by_sample[sample.sample_token].token,
            source_variant=sample.source_variant,
            role=role_by_identity[sample.identity_token],
        )
        for sample in sorted(source.samples, key=lambda item: item.sample_token)
    )
    identities = tuple(identity_assignments)
    return ResearchCycleManifest(
        cycle_name=cycle_name,
        cycle_id=compute_research_cycle_id(cycle_name),
        cycle_namespace_uuid=str(RESEARCH_CYCLE_NAMESPACE),
        registered_identity_namespace_uuid=str(REGISTERED_DOG_NAMESPACE),
        source_bundle_sha256=source.bundle_sha256,
        dependency_graph_sha256=graph.graph_sha256,
        source_admissions_sha256=source_admissions.admissions_sha256,
        source_admissions=source_admissions.sources,
        role_exposure_ledger_sha256=role_exposure_ledger.ledger_sha256,
        role_exposure_receipt_sha256=role_exposure_receipt.receipt_sha256,
        target_percentages=tuple(_TARGET_PERCENT.items()),
        dataset_role_counts=_dataset_role_counts(identities),
        identity_assignments=identities,
        sample_assignments=sample_assignments,
    )


def _assign_blocks(
    blocks: Iterable[Any],
    dataset_by_identity: Mapping[str, str],
    *,
    cycle_name: str,
    evidence_root: str,
) -> dict[str, ResearchRole]:
    values = tuple(blocks)
    totals = Counter(
        dataset_by_identity[identity]
        for block in values
        for identity in block.identity_tokens
    )
    targets = {
        dataset: _integer_targets(count) for dataset, count in sorted(totals.items())
    }
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    role_order = tuple(ResearchRole)

    def block_counts(block: Any) -> Counter[str]:
        return Counter(dataset_by_identity[item] for item in block.identity_tokens)

    def tie_token(block: Any) -> str:
        return hashlib.sha256(
            (
                "RESEARCH_BLOCK_ORDER_V1\0"
                + cycle_name
                + "\0"
                + evidence_root
                + "\0"
                + block.token
            ).encode("utf-8")
        ).hexdigest()

    ordered = sorted(
        values,
        key=lambda block: (
            -len(block.identity_tokens),
            tie_token(block),
            block.token,
        ),
    )
    result: dict[str, ResearchRole] = {}
    for block in ordered:
        additions = block_counts(block)

        def objective(role: ResearchRole) -> tuple[int, int, int]:
            absolute_error = 0
            overflow = 0
            for dataset, target in targets.items():
                for candidate in role_order:
                    value = counts[dataset][candidate.value]
                    if candidate is role:
                        value += additions[dataset]
                    delta = value - target[candidate.value]
                    absolute_error += abs(delta)
                    overflow += max(0, delta)
            return absolute_error, overflow, role_order.index(role)

        selected = min(role_order, key=objective)
        result[block.token] = selected
        for dataset, count in additions.items():
            counts[dataset][selected.value] += count
    return result


def _integer_targets(count: int) -> dict[str, int]:
    floors = {
        role: count * percent // 100 for role, percent in _TARGET_PERCENT.items()
    }
    remaining = count - sum(floors.values())
    remainder_order = sorted(
        _TARGET_PERCENT,
        key=lambda role: (-(count * _TARGET_PERCENT[role] % 100), tuple(_TARGET_PERCENT).index(role)),
    )
    for role in remainder_order[:remaining]:
        floors[role] += 1
    return floors


def _dataset_role_counts(
    assignments: Iterable[ResearchIdentityAssignment],
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in assignments:
        key = "QUARANTINED" if item.role is None else item.role.value
        counts[item.dataset_name][key] += 1
    keys = (*tuple(role.value for role in ResearchRole), "QUARANTINED")
    return tuple(
        (
            dataset,
            tuple((key, counts[dataset][key]) for key in keys),
        )
        for dataset in sorted(counts)
    )


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_uuid5(value: object, name: str) -> None:
    try:
        parsed = uuid.UUID(value) if isinstance(value, str) else None
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical UUIDv5") from exc
    if parsed is None or parsed.version != 5 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUIDv5")


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 512:
        raise ValueError(f"{name} must be bounded non-empty text")


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return tuple(value)


def _require_string_tuple(
    values: object, name: str, *, allow_empty: bool = False
) -> None:
    if not isinstance(values, tuple) or (not values and not allow_empty):
        raise ValueError(f"{name} must be a canonical tuple")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} values must be non-empty strings")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be sorted and unique")


def _require_digest_tuple(values: object, name: str) -> None:
    _require_string_tuple(values, name)
    for value in values:
        _require_sha256(value, name)


def _exact_keys(payload: object, expected: set[str], name: str) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be an object")
    if set(payload) != expected:
        raise ValueError(f"{name} fields differ")


__all__ = [
    "RESEARCH_CYCLE_NAMESPACE",
    "IdentityTargetMode",
    "ResearchCycleManifest",
    "ResearchIdentityAssignment",
    "ResearchLicenseLane",
    "ResearchRole",
    "ResearchSampleAssignment",
    "ResearchSourceAdmission",
    "ResearchSourceAdmissions",
    "ResearchSourceRole",
    "build_research_cycle_manifest",
    "compute_research_cycle_id",
]
