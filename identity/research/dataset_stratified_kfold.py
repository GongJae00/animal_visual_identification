"""Dataset-stratified retrospective K-fold protocol over a research cycle."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from foundation.protected_io import read_strict_json_document
from foundation.provenance import content_sha256
from identity.registry.identity_registry import (
    compute_identity_token,
    compute_registered_dog_id,
)
from identity.research.research_cycle_admission import ResearchCycleManifest
from identity.exposure.role_exposure import ExposureStage

_IDENTITY_DATASETS = frozenset(
    {"dogfacenet224", "mpdd", "sibetan", "yt-bb-dog"}
)
_INTERPRETATION = (
    "RETROSPECTIVE_DATASET_STRATIFIED_CROSS_VALIDATION;"
    "IDENTITY_DISJOINT_WITHIN_EACH_FOLD;ALL_FOLDS_ARE_EXPOSED_MODEL_SELECTION_EVIDENCE;"
    "NO_FINAL_EVALUATION_OR_HOLDOUT_INTERPRETATION"
)
_VIEW_INTERPRETATION = (
    "ONE_EXPOSED_CROSS_VALIDATION_VIEW_NOT_AN_INDEPENDENT_FINAL_TEST"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BUNDLE_SCHEMA = "cvi.dataset_stratified_identity_kfold_manifest_bundle.v1"


class FoldStage(StrEnum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    TEST = "TEST"
    QUARANTINED = "QUARANTINED"


class HeldOutSampleRole(StrEnum):
    GALLERY = "GALLERY"
    QUERY = "QUERY"
    EXCLUDED = "EXCLUDED"
    CONTROL_ONLY = "CONTROL_ONLY"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True, slots=True)
class DatasetStratifiedKFoldPolicy:
    fold_count: int = 5
    dev_offset: int = 1
    gallery_fraction_numerator: int = 1
    gallery_fraction_denominator: int = 2
    minimum_gallery_images: int = 1
    minimum_query_images: int = 1
    minimum_identities_per_fold: int = 1
    minimum_retrieval_identities_per_fold: int = 1
    schema_version: str = "cvi.dataset_stratified_identity_kfold_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.dataset_stratified_identity_kfold_policy.v1":
            raise ValueError("unsupported dataset-stratified K-fold policy schema")
        for name in (
            "fold_count",
            "dev_offset",
            "gallery_fraction_numerator",
            "gallery_fraction_denominator",
            "minimum_gallery_images",
            "minimum_query_images",
            "minimum_identities_per_fold",
            "minimum_retrieval_identities_per_fold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.fold_count < 3:
            raise ValueError("fold_count must be at least three")
        if not 1 <= self.dev_offset < self.fold_count:
            raise ValueError("dev_offset must select a distinct fold")
        if not 0 < self.gallery_fraction_numerator < self.gallery_fraction_denominator:
            raise ValueError("gallery fraction must be strictly between zero and one")
        if self.minimum_gallery_images < 1 or self.minimum_query_images < 1:
            raise ValueError("gallery and query image minima must be positive")
        if self.minimum_identities_per_fold < 1:
            raise ValueError("each fold needs at least one identity per dataset")
        if self.minimum_retrieval_identities_per_fold < 1:
            raise ValueError(
                "each scoreable fold needs at least one retrieval identity"
            )

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetStratifiedKFoldPolicy:
        _exact_keys(payload, set(cls.__dataclass_fields__), "K-fold policy")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class IdentityFoldAssignment:
    identity_token: str
    dataset_identity_id: str
    registered_dog_id: str
    dataset_name: str
    home_fold: int | None
    allocation_block_token: str
    component_tokens: tuple[str, ...]
    historical_maximum_exposure: ExposureStage | None
    quarantine_reasons: tuple[str, ...]
    schema_version: str = "cvi.identity_fold_assignment.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.identity_fold_assignment.v1":
            raise ValueError("unsupported identity fold assignment schema")
        _require_sha256(self.identity_token, "identity_token")
        _require_text(self.dataset_identity_id, "dataset_identity_id")
        if self.identity_token != compute_identity_token(self.dataset_identity_id):
            raise ValueError("identity fold token is not deterministic")
        if self.registered_dog_id != compute_registered_dog_id(self.dataset_identity_id):
            raise ValueError("identity fold registered UUIDv5 is not deterministic")
        if self.dataset_name not in _IDENTITY_DATASETS or self.dataset_identity_id.split(
            ":", 1
        )[0] != self.dataset_name:
            raise ValueError("identity fold dataset binding differs")
        if self.home_fold is not None and (
            isinstance(self.home_fold, bool)
            or not isinstance(self.home_fold, int)
            or self.home_fold < 0
        ):
            raise ValueError("home_fold must be a nonnegative integer or null")
        _require_sha256(self.allocation_block_token, "allocation_block_token")
        _require_digest_tuple(self.component_tokens, "component_tokens")
        if self.historical_maximum_exposure is not None and not isinstance(
            self.historical_maximum_exposure, ExposureStage
        ):
            raise TypeError("historical maximum exposure must be ExposureStage or null")
        _require_string_tuple(
            self.quarantine_reasons, "quarantine_reasons", allow_empty=True
        )
        if (self.home_fold is None) != bool(self.quarantine_reasons):
            raise ValueError("quarantined identities require reasons and no home fold")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity_token": self.identity_token,
            "dataset_identity_id": self.dataset_identity_id,
            "registered_dog_id": self.registered_dog_id,
            "dataset_name": self.dataset_name,
            "home_fold": self.home_fold,
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
    def from_dict(cls, payload: Mapping[str, Any]) -> IdentityFoldAssignment:
        _exact_keys(payload, set(cls.__dataclass_fields__), "identity fold assignment")
        return cls(
            identity_token=payload["identity_token"],
            dataset_identity_id=payload["dataset_identity_id"],
            registered_dog_id=payload["registered_dog_id"],
            dataset_name=payload["dataset_name"],
            home_fold=payload["home_fold"],
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
class SampleFoldAssignment:
    sample_token: str
    identity_token: str
    sequence_token: str
    component_token: str
    source_variant: str
    home_fold: int | None
    held_out_role: HeldOutSampleRole
    training_eligible: bool
    schema_version: str = "cvi.sample_fold_assignment.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.sample_fold_assignment.v1":
            raise ValueError("unsupported sample fold assignment schema")
        for value, name in (
            (self.sample_token, "sample_token"),
            (self.identity_token, "identity_token"),
            (self.sequence_token, "sequence_token"),
            (self.component_token, "component_token"),
        ):
            _require_sha256(value, name)
        if self.source_variant not in {"original", "random_background"}:
            raise ValueError("sample fold source variant differs")
        if self.home_fold is not None and (
            isinstance(self.home_fold, bool)
            or not isinstance(self.home_fold, int)
            or self.home_fold < 0
        ):
            raise ValueError("sample home_fold must be nonnegative or null")
        if not isinstance(self.held_out_role, HeldOutSampleRole):
            raise TypeError("held_out_role must be HeldOutSampleRole")
        if not isinstance(self.training_eligible, bool):
            raise TypeError("training_eligible must be boolean")
        expected = (
            HeldOutSampleRole.QUARANTINED
            if self.home_fold is None
            else HeldOutSampleRole.CONTROL_ONLY
            if self.source_variant == "random_background"
            else None
        )
        if expected is not None and self.held_out_role is not expected:
            raise ValueError("sample held-out role differs from fold or source variant")
        if (
            self.home_fold is not None
            and self.source_variant == "original"
            and self.held_out_role
            not in {
                HeldOutSampleRole.GALLERY,
                HeldOutSampleRole.QUERY,
                HeldOutSampleRole.EXCLUDED,
            }
        ):
            raise ValueError("original sample held-out role differs")
        if self.source_variant == "random_background" and self.training_eligible:
            raise ValueError("random-background controls cannot be training inputs")
        if self.home_fold is None and self.training_eligible:
            raise ValueError("quarantined samples cannot be training inputs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_token": self.sample_token,
            "identity_token": self.identity_token,
            "sequence_token": self.sequence_token,
            "component_token": self.component_token,
            "source_variant": self.source_variant,
            "home_fold": self.home_fold,
            "held_out_role": self.held_out_role.value,
            "training_eligible": self.training_eligible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SampleFoldAssignment:
        _exact_keys(payload, set(cls.__dataclass_fields__), "sample fold assignment")
        values = dict(payload)
        values["held_out_role"] = HeldOutSampleRole(values["held_out_role"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class DatasetStratifiedIdentityKFoldManifest:
    protocol_name: str
    policy: DatasetStratifiedKFoldPolicy
    policy_sha256: str
    source_research_cycle_sha256: str
    source_bundle_sha256: str
    dependency_graph_sha256: str
    source_admissions_sha256: str
    role_exposure_ledger_sha256: str
    role_exposure_receipt_sha256: str
    dataset_fold_counts: tuple[tuple[str, tuple[int, ...], int], ...]
    identity_assignments: tuple[IdentityFoldAssignment, ...]
    sample_assignments: tuple[SampleFoldAssignment, ...]
    score_inputs_used: bool = False
    final_evaluation_permitted: bool = False
    interpretation: str = _INTERPRETATION
    schema_version: str = "cvi.dataset_stratified_identity_kfold_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.dataset_stratified_identity_kfold_manifest.v1":
            raise ValueError("unsupported dataset-stratified K-fold manifest schema")
        _require_text(self.protocol_name, "protocol_name")
        if not isinstance(self.policy, DatasetStratifiedKFoldPolicy):
            raise TypeError("policy must be DatasetStratifiedKFoldPolicy")
        if self.policy_sha256 != self.policy.policy_sha256:
            raise ValueError("embedded K-fold policy hash differs")
        for value, name in (
            (self.source_research_cycle_sha256, "source_research_cycle_sha256"),
            (self.source_bundle_sha256, "source_bundle_sha256"),
            (self.dependency_graph_sha256, "dependency_graph_sha256"),
            (self.source_admissions_sha256, "source_admissions_sha256"),
            (self.role_exposure_ledger_sha256, "role_exposure_ledger_sha256"),
            (self.role_exposure_receipt_sha256, "role_exposure_receipt_sha256"),
        ):
            _require_sha256(value, name)
        if self.score_inputs_used is not False:
            raise ValueError("K-fold assignment must be score-blind")
        if self.final_evaluation_permitted is not False:
            raise ValueError("K-fold manifest cannot permit final evaluation")
        if self.interpretation != _INTERPRETATION:
            raise ValueError("K-fold interpretation differs")
        if not self.identity_assignments or not self.sample_assignments:
            raise ValueError("K-fold assignments must not be empty")
        if self.identity_assignments != tuple(
            sorted(self.identity_assignments, key=lambda item: item.identity_token)
        ):
            raise ValueError("identity fold assignments must be canonically sorted")
        if self.sample_assignments != tuple(
            sorted(self.sample_assignments, key=lambda item: item.sample_token)
        ):
            raise ValueError("sample fold assignments must be canonically sorted")
        self._validate_closure()

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def _validate_closure(self) -> None:
        fold_count = self.policy.fold_count
        identities = {item.identity_token: item for item in self.identity_assignments}
        if len(identities) != len(self.identity_assignments):
            raise ValueError("identity fold assignments repeat an identity")
        fold_by_block: dict[str, int | None] = {}
        fold_by_component: dict[str, int | None] = {}
        for identity in self.identity_assignments:
            if identity.home_fold is not None and identity.home_fold >= fold_count:
                raise ValueError("identity home fold exceeds policy")
            prior = fold_by_block.setdefault(
                identity.allocation_block_token, identity.home_fold
            )
            if prior != identity.home_fold:
                raise ValueError("allocation block crosses home folds")
            for component in identity.component_tokens:
                prior = fold_by_component.setdefault(component, identity.home_fold)
                if prior != identity.home_fold:
                    raise ValueError("component crosses home folds")
        seen_samples: set[str] = set()
        roles_by_identity_component: dict[tuple[str, str], set[HeldOutSampleRole]] = defaultdict(set)
        roles_by_identity_sequence: dict[tuple[str, str], set[HeldOutSampleRole]] = defaultdict(set)
        identities_by_component: dict[str, set[str]] = defaultdict(set)
        original_samples_by_identity: dict[str, list[SampleFoldAssignment]] = defaultdict(list)
        for sample in self.sample_assignments:
            if sample.sample_token in seen_samples:
                raise ValueError("sample fold assignments repeat a sample")
            seen_samples.add(sample.sample_token)
            identity = identities.get(sample.identity_token)
            if identity is None or identity.home_fold != sample.home_fold:
                raise ValueError("sample and identity home folds differ")
            if sample.component_token not in identity.component_tokens:
                raise ValueError("sample component is absent from its allocation block")
            if sample.home_fold is not None and sample.home_fold >= fold_count:
                raise ValueError("sample home fold exceeds policy")
            if sample.source_variant == "original":
                original_samples_by_identity[sample.identity_token].append(sample)
                roles_by_identity_component[
                    (sample.identity_token, sample.component_token)
                ].add(sample.held_out_role)
                roles_by_identity_sequence[
                    (sample.identity_token, sample.sequence_token)
                ].add(sample.held_out_role)
                identities_by_component[sample.component_token].add(sample.identity_token)
        if any(len(roles) != 1 for roles in roles_by_identity_component.values()):
            raise ValueError("duplicate component crosses gallery/query roles")
        if any(len(roles) != 1 for roles in roles_by_identity_sequence.values()):
            raise ValueError("sequence crosses gallery/query roles")
        for component, component_identities in identities_by_component.items():
            if len(component_identities) > 1:
                roles = {
                    role
                    for (identity, token), values in roles_by_identity_component.items()
                    if token == component and identity in component_identities
                    for role in values
                }
                if roles != {HeldOutSampleRole.EXCLUDED}:
                    raise ValueError("cross-identity component must be excluded")
                if any(
                    sample.training_eligible
                    for sample in self.sample_assignments
                    if sample.component_token == component
                ):
                    raise ValueError("cross-identity component cannot be used for training")
        unsafe_components = {
            component
            for component, component_identities in identities_by_component.items()
            if len(component_identities) > 1
        }
        for samples in original_samples_by_identity.values():
            unit_by_sample = _close_retrieval_units(samples)
            unsafe_units = {
                unit_by_sample[sample.sample_token]
                for sample in samples
                if sample.component_token in unsafe_components
            }
            for sample in samples:
                if unit_by_sample[sample.sample_token] in unsafe_units and (
                    sample.held_out_role is not HeldOutSampleRole.EXCLUDED
                    or sample.training_eligible
                ):
                    raise ValueError(
                        "unsafe retrieval unit must be excluded from training and scoring"
                    )
        roles_by_identity: dict[str, set[HeldOutSampleRole]] = defaultdict(set)
        for sample in self.sample_assignments:
            if sample.source_variant == "original":
                roles_by_identity[sample.identity_token].add(sample.held_out_role)
        for identity in self.identity_assignments:
            roles = roles_by_identity[identity.identity_token]
            if identity.home_fold is None:
                if roles != {HeldOutSampleRole.QUARANTINED}:
                    raise ValueError("quarantined identity sample roles differ")
            else:
                retrieval_roles = roles & {
                    HeldOutSampleRole.GALLERY,
                    HeldOutSampleRole.QUERY,
                }
                if retrieval_roles and retrieval_roles != {
                    HeldOutSampleRole.GALLERY,
                    HeldOutSampleRole.QUERY,
                }:
                    raise ValueError(
                        "retrieval-eligible identity requires gallery and query"
                    )
                if not retrieval_roles and roles != {HeldOutSampleRole.EXCLUDED}:
                    raise ValueError("ineligible held-out identity must be excluded")
        retrieval_counts: dict[str, Counter[int]] = defaultdict(Counter)
        retrieval_totals: Counter[str] = Counter()
        for identity in self.identity_assignments:
            if identity.home_fold is None:
                continue
            roles = roles_by_identity[identity.identity_token]
            if {
                HeldOutSampleRole.GALLERY,
                HeldOutSampleRole.QUERY,
            } <= roles:
                retrieval_counts[identity.dataset_name][identity.home_fold] += 1
                retrieval_totals[identity.dataset_name] += 1
        retrieval_minimum = self.policy.minimum_retrieval_identities_per_fold
        for dataset, total in retrieval_totals.items():
            if total < fold_count * retrieval_minimum:
                continue
            if any(
                retrieval_counts[dataset][fold] < retrieval_minimum
                for fold in range(fold_count)
            ):
                raise ValueError(
                    "scoreable dataset misses the per-fold retrieval identity minimum"
                )
        expected_counts = _dataset_fold_counts(
            self.identity_assignments, fold_count=fold_count
        )
        for dataset, counts, _ in self.dataset_fold_counts:
            if dataset not in _IDENTITY_DATASETS or len(counts) != fold_count:
                raise ValueError("dataset fold count schema differs")
            if any(
                isinstance(count, bool) or not isinstance(count, int)
                for count in counts
            ):
                raise TypeError("dataset fold counts must be integers")
            if any(
                count < self.policy.minimum_identities_per_fold for count in counts
            ):
                raise ValueError(
                    "every identity dataset must meet the per-fold identity minimum"
                )
        if any(
            isinstance(quarantined, bool)
            or not isinstance(quarantined, int)
            or quarantined < 0
            for _, _, quarantined in self.dataset_fold_counts
        ):
            raise TypeError("quarantined dataset counts must be nonnegative integers")
        if self.dataset_fold_counts != expected_counts:
            raise ValueError("dataset fold counts differ from assignments")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_name": self.protocol_name,
            "policy": self.policy.to_dict(),
            "policy_sha256": self.policy_sha256,
            "source_research_cycle_sha256": self.source_research_cycle_sha256,
            "source_bundle_sha256": self.source_bundle_sha256,
            "dependency_graph_sha256": self.dependency_graph_sha256,
            "source_admissions_sha256": self.source_admissions_sha256,
            "role_exposure_ledger_sha256": self.role_exposure_ledger_sha256,
            "role_exposure_receipt_sha256": self.role_exposure_receipt_sha256,
            "dataset_fold_counts": {
                dataset: {
                    "folds": list(counts),
                    "quarantined": quarantined,
                }
                for dataset, counts, quarantined in self.dataset_fold_counts
            },
            "identity_assignments": [item.to_dict() for item in self.identity_assignments],
            "sample_assignments": [item.to_dict() for item in self.sample_assignments],
            "score_inputs_used": self.score_inputs_used,
            "final_evaluation_permitted": self.final_evaluation_permitted,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> DatasetStratifiedIdentityKFoldManifest:
        _exact_keys(payload, set(cls.__dataclass_fields__), "K-fold manifest")
        policy = DatasetStratifiedKFoldPolicy.from_dict(payload["policy"])
        raw_counts = payload["dataset_fold_counts"]
        if not isinstance(raw_counts, Mapping):
            raise TypeError("dataset_fold_counts must be an object")
        counts: list[tuple[str, tuple[int, ...], int]] = []
        for dataset in sorted(raw_counts):
            value = raw_counts[dataset]
            _exact_keys(value, {"folds", "quarantined"}, "dataset fold count")
            if not isinstance(value["folds"], list):
                raise TypeError("dataset fold counts must be an array")
            counts.append((dataset, tuple(value["folds"]), value["quarantined"]))
        if not isinstance(payload["identity_assignments"], list) or not isinstance(
            payload["sample_assignments"], list
        ):
            raise TypeError("K-fold assignments must be arrays")
        return cls(
            protocol_name=payload["protocol_name"],
            policy=policy,
            policy_sha256=payload["policy_sha256"],
            source_research_cycle_sha256=payload["source_research_cycle_sha256"],
            source_bundle_sha256=payload["source_bundle_sha256"],
            dependency_graph_sha256=payload["dependency_graph_sha256"],
            source_admissions_sha256=payload["source_admissions_sha256"],
            role_exposure_ledger_sha256=payload["role_exposure_ledger_sha256"],
            role_exposure_receipt_sha256=payload["role_exposure_receipt_sha256"],
            dataset_fold_counts=tuple(counts),
            identity_assignments=tuple(
                IdentityFoldAssignment.from_dict(item)
                for item in payload["identity_assignments"]
            ),
            sample_assignments=tuple(
                SampleFoldAssignment.from_dict(item)
                for item in payload["sample_assignments"]
            ),
            score_inputs_used=payload["score_inputs_used"],
            final_evaluation_permitted=payload["final_evaluation_permitted"],
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


def build_dataset_stratified_identity_kfold(
    *,
    protocol_name: str,
    research_cycle: ResearchCycleManifest,
    policy: DatasetStratifiedKFoldPolicy,
) -> DatasetStratifiedIdentityKFoldManifest:
    """Rotate duplicate-closed identity blocks without weakening exposure history."""

    _require_text(protocol_name, "protocol_name")
    if not isinstance(research_cycle, ResearchCycleManifest):
        raise TypeError("research_cycle must be ResearchCycleManifest")
    if not isinstance(policy, DatasetStratifiedKFoldPolicy):
        raise TypeError("policy must be DatasetStratifiedKFoldPolicy")
    cycle_identities = {item.identity_token: item for item in research_cycle.identity_assignments}
    dataset_by_identity = {
        item.identity_token: item.dataset_name for item in research_cycle.identity_assignments
    }
    blocks: dict[str, tuple[str, ...]] = {}
    for item in research_cycle.identity_assignments:
        values = blocks.setdefault(item.allocation_block_token, ())
        blocks[item.allocation_block_token] = tuple(sorted((*values, item.identity_token)))
    assignable_blocks = {
        block: identities
        for block, identities in blocks.items()
        if all(not cycle_identities[identity].quarantine_reasons for identity in identities)
    }
    component_identities: dict[str, set[str]] = defaultdict(set)
    for sample in research_cycle.sample_assignments:
        if sample.source_variant == "original":
            component_identities[sample.component_token].add(sample.identity_token)
    unsafe_components = {
        component
        for component, identities in component_identities.items()
        if len(identities) > 1
    }
    samples_by_identity: dict[str, list[Any]] = defaultdict(list)
    for sample in research_cycle.sample_assignments:
        samples_by_identity[sample.identity_token].append(sample)
    retrieval_eligible_identities = _retrieval_eligible_identities(
        samples_by_identity,
        unsafe_components=unsafe_components,
        policy=policy,
    )
    evidence_root = content_sha256(
        {
            "protocol_name": protocol_name,
            "policy_sha256": policy.policy_sha256,
            "source_research_cycle_sha256": research_cycle.manifest_sha256,
        }
    )
    fold_by_block = _assign_home_folds(
        assignable_blocks,
        dataset_by_identity,
        protocol_name=protocol_name,
        evidence_root=evidence_root,
        fold_count=policy.fold_count,
        minimum_identities_per_fold=policy.minimum_identities_per_fold,
        retrieval_eligible_identities=retrieval_eligible_identities,
        minimum_retrieval_identities_per_fold=(
            policy.minimum_retrieval_identities_per_fold
        ),
    )
    identity_assignments = tuple(
        IdentityFoldAssignment(
            identity_token=item.identity_token,
            dataset_identity_id=item.dataset_identity_id,
            registered_dog_id=item.registered_dog_id,
            dataset_name=item.dataset_name,
            home_fold=fold_by_block.get(item.allocation_block_token),
            allocation_block_token=item.allocation_block_token,
            component_tokens=item.component_tokens,
            historical_maximum_exposure=item.historical_maximum_exposure,
            quarantine_reasons=item.quarantine_reasons,
        )
        for item in research_cycle.identity_assignments
    )
    home_by_identity = {
        item.identity_token: item.home_fold for item in identity_assignments
    }
    held_out_by_sample: dict[str, HeldOutSampleRole] = {}
    training_eligible_by_sample: dict[str, bool] = {}
    for identity_token, samples in sorted(samples_by_identity.items()):
        home_fold = home_by_identity[identity_token]
        if home_fold is None:
            for sample in samples:
                held_out_by_sample[sample.sample_token] = HeldOutSampleRole.QUARANTINED
                training_eligible_by_sample[sample.sample_token] = False
            continue
        for sample in samples:
            if sample.source_variant == "random_background":
                held_out_by_sample[sample.sample_token] = HeldOutSampleRole.CONTROL_ONLY
                training_eligible_by_sample[sample.sample_token] = False
        original = [sample for sample in samples if sample.source_variant == "original"]
        retrieval_unit_by_sample = _close_retrieval_units(original)
        unsafe_units = {
            retrieval_unit_by_sample[sample.sample_token]
            for sample in original
            if sample.component_token in unsafe_components
        }
        safe = [
            sample
            for sample in original
            if retrieval_unit_by_sample[sample.sample_token] not in unsafe_units
        ]
        units: dict[str, int] = Counter(
            retrieval_unit_by_sample[sample.sample_token] for sample in safe
        )
        if (
            len(units) < 2
            or len(safe) < policy.minimum_gallery_images + policy.minimum_query_images
        ):
            gallery_units: set[str] = set()
        else:
            gallery_units = _select_gallery_components(
                units,
                identity_token=identity_token,
                policy=policy,
                evidence_root=evidence_root,
            )
        query_count = sum(
            count for unit, count in units.items() if unit not in gallery_units
        )
        gallery_count = sum(units[unit] for unit in gallery_units)
        eligible = (
            gallery_count >= policy.minimum_gallery_images
            and query_count >= policy.minimum_query_images
        )
        for sample in original:
            unit = retrieval_unit_by_sample[sample.sample_token]
            training_eligible_by_sample[sample.sample_token] = unit not in unsafe_units
            if not eligible or unit in unsafe_units:
                role = HeldOutSampleRole.EXCLUDED
            elif unit in gallery_units:
                role = HeldOutSampleRole.GALLERY
            else:
                role = HeldOutSampleRole.QUERY
            held_out_by_sample[sample.sample_token] = role
    sample_assignments = tuple(
        SampleFoldAssignment(
            sample_token=item.sample_token,
            identity_token=item.identity_token,
            sequence_token=item.sequence_token,
            component_token=item.component_token,
            source_variant=item.source_variant,
            home_fold=home_by_identity[item.identity_token],
            held_out_role=held_out_by_sample[item.sample_token],
            training_eligible=(
                training_eligible_by_sample[item.sample_token]
            ),
        )
        for item in research_cycle.sample_assignments
    )
    return DatasetStratifiedIdentityKFoldManifest(
        protocol_name=protocol_name,
        policy=policy,
        policy_sha256=policy.policy_sha256,
        source_research_cycle_sha256=research_cycle.manifest_sha256,
        source_bundle_sha256=research_cycle.source_bundle_sha256,
        dependency_graph_sha256=research_cycle.dependency_graph_sha256,
        source_admissions_sha256=research_cycle.source_admissions_sha256,
        role_exposure_ledger_sha256=research_cycle.role_exposure_ledger_sha256,
        role_exposure_receipt_sha256=research_cycle.role_exposure_receipt_sha256,
        dataset_fold_counts=_dataset_fold_counts(
            identity_assignments, fold_count=policy.fold_count
        ),
        identity_assignments=identity_assignments,
        sample_assignments=sample_assignments,
    )


def materialize_identity_fold(
    manifest: DatasetStratifiedIdentityKFoldManifest, fold_index: int
) -> dict[str, Any]:
    """Create one content-bound TRAIN/DEV/TEST view of the rotating protocol."""

    if isinstance(fold_index, bool) or not isinstance(fold_index, int) or not 0 <= fold_index < manifest.policy.fold_count:
        raise ValueError("fold_index lies outside the K-fold policy")

    def stage(home_fold: int | None) -> FoldStage:
        if home_fold is None:
            return FoldStage.QUARANTINED
        if home_fold == fold_index:
            return FoldStage.TEST
        if home_fold == (fold_index + manifest.policy.dev_offset) % manifest.policy.fold_count:
            return FoldStage.DEV
        return FoldStage.TRAIN

    identity_rows = [
        {
            "identity_token": item.identity_token,
            "dataset_name": item.dataset_name,
            "stage": stage(item.home_fold).value,
        }
        for item in manifest.identity_assignments
    ]
    sample_rows: list[dict[str, Any]] = []
    for item in manifest.sample_assignments:
        sample_stage = stage(item.home_fold)
        if sample_stage is FoldStage.QUARANTINED:
            role = "QUARANTINED"
        elif item.source_variant == "random_background":
            role = "CONTROL_ONLY"
        elif sample_stage is FoldStage.TRAIN and item.training_eligible:
            role = "TRAIN_INPUT"
        elif sample_stage is FoldStage.TRAIN:
            role = "EXCLUDED"
        else:
            role = item.held_out_role.value
        sample_rows.append(
            {
                "sample_token": item.sample_token,
                "identity_token": item.identity_token,
                "component_token": item.component_token,
                "stage": sample_stage.value,
                "sample_role": role,
            }
        )
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in identity_rows:
        counts[row["dataset_name"]][row["stage"]] += 1
    view = {
        "schema_version": "cvi.dataset_stratified_identity_kfold_view.v1",
        "parent_manifest_sha256": manifest.manifest_sha256,
        "fold_index": fold_index,
        "dataset_stage_counts": {
            dataset: {
                value.value: counts[dataset][value.value] for value in FoldStage
            }
            for dataset in sorted(counts)
        },
        "identity_assignments": identity_rows,
        "sample_assignments": sample_rows,
        "score_inputs_used": False,
        "final_evaluation_permitted": False,
        "interpretation": _VIEW_INTERPRETATION,
    }
    return {
        "schema_version": "cvi.dataset_stratified_identity_kfold_view_bundle.v1",
        "view_sha256": content_sha256(view),
        "view": view,
    }


def dataset_stratified_kfold_bundle(
    manifest: DatasetStratifiedIdentityKFoldManifest,
) -> dict[str, Any]:
    return {
        "schema_version": _BUNDLE_SCHEMA,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest": manifest.to_dict(),
    }


def read_dataset_stratified_identity_kfold(
    path: Any,
) -> DatasetStratifiedIdentityKFoldManifest:
    payload = read_strict_json_document(
        path,
        maximum_bytes=2_147_483_648,
        maximum_nodes=25_000_000,
        maximum_keys=10_000_000,
        maximum_array_length=1_000_000,
    ).payload
    if (
        set(payload) != {"schema_version", "manifest_sha256", "manifest"}
        or payload["schema_version"] != _BUNDLE_SCHEMA
    ):
        raise ValueError("dataset-stratified K-fold bundle schema differs")
    _require_sha256(payload["manifest_sha256"], "K-fold manifest SHA-256")
    if not isinstance(payload["manifest"], Mapping) or content_sha256(
        payload["manifest"]
    ) != payload["manifest_sha256"]:
        raise ValueError("dataset-stratified K-fold bundle digest differs")
    manifest = DatasetStratifiedIdentityKFoldManifest.from_dict(payload["manifest"])
    if manifest.manifest_sha256 != payload["manifest_sha256"]:
        raise ValueError("dataset-stratified K-fold manifest digest differs")
    return manifest


def _assign_home_folds(
    blocks: Mapping[str, tuple[str, ...]],
    dataset_by_identity: Mapping[str, str],
    *,
    protocol_name: str,
    evidence_root: str,
    fold_count: int,
    minimum_identities_per_fold: int,
    retrieval_eligible_identities: set[str],
    minimum_retrieval_identities_per_fold: int,
) -> dict[str, int]:
    totals = Counter(
        dataset_by_identity[identity]
        for identities in blocks.values()
        for identity in identities
    )
    if set(totals) != _IDENTITY_DATASETS or any(
        count < fold_count * minimum_identities_per_fold
        for count in totals.values()
    ):
        raise ValueError("each identity dataset lacks the per-fold identity minimum")
    targets = {
        dataset: tuple(
            count // fold_count + (1 if fold < count % fold_count else 0)
            for fold in range(fold_count)
        )
        for dataset, count in totals.items()
    }
    retrieval_totals = Counter(
        dataset_by_identity[identity]
        for identity in retrieval_eligible_identities
    )
    retrieval_targets = {
        dataset: tuple(
            count // fold_count + (1 if fold < count % fold_count else 0)
            for fold in range(fold_count)
        )
        for dataset, count in retrieval_totals.items()
        if count >= fold_count * minimum_retrieval_identities_per_fold
    }
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    retrieval_counts: dict[str, Counter[int]] = defaultdict(Counter)

    def rank(block: str) -> str:
        return hashlib.sha256(
            (
                "CVI_DATASET_STRATIFIED_KFOLD_BLOCK_ORDER_V1\0"
                + protocol_name
                + "\0"
                + evidence_root
                + "\0"
                + block
            ).encode("utf-8")
        ).hexdigest()

    ordered = sorted(
        blocks,
        key=lambda block: (-len(blocks[block]), rank(block), block),
    )
    result: dict[str, int] = {}
    for block in ordered:
        additions = Counter(dataset_by_identity[item] for item in blocks[block])
        retrieval_additions = Counter(
            dataset_by_identity[item]
            for item in blocks[block]
            if item in retrieval_eligible_identities
        )

        def objective(
            candidate_fold: int,
            additions_for_block: Counter[str] = additions,
            retrieval_additions_for_block: Counter[str] = retrieval_additions,
        ) -> tuple[int, int, int, int, int]:
            identity_deficit = 0
            retrieval_deficit = 0
            absolute_error = 0
            overflow = 0
            for dataset, target in targets.items():
                for fold in range(fold_count):
                    value = counts[dataset][fold]
                    if fold == candidate_fold:
                        value += additions_for_block[dataset]
                    identity_deficit += max(
                        0, minimum_identities_per_fold - value
                    )
                    delta = value - target[fold]
                    absolute_error += abs(delta)
                    overflow += max(0, delta)
            for dataset, target in retrieval_targets.items():
                for fold in range(fold_count):
                    value = retrieval_counts[dataset][fold]
                    if fold == candidate_fold:
                        value += retrieval_additions_for_block[dataset]
                    retrieval_deficit += max(
                        0, minimum_retrieval_identities_per_fold - value
                    )
                    absolute_error += abs(value - target[fold])
            return (
                retrieval_deficit,
                identity_deficit,
                absolute_error,
                overflow,
                candidate_fold,
            )

        selected = min(range(fold_count), key=objective)
        result[block] = selected
        for dataset, count in additions.items():
            counts[dataset][selected] += count
        for dataset, count in retrieval_additions.items():
            retrieval_counts[dataset][selected] += count
    return result


def _retrieval_eligible_identities(
    samples_by_identity: Mapping[str, Sequence[Any]],
    *,
    unsafe_components: set[str],
    policy: DatasetStratifiedKFoldPolicy,
) -> set[str]:
    eligible: set[str] = set()
    for identity_token, samples in samples_by_identity.items():
        original = [sample for sample in samples if sample.source_variant == "original"]
        units = _close_retrieval_units(original)
        unsafe_units = {
            units[sample.sample_token]
            for sample in original
            if sample.component_token in unsafe_components
        }
        safe_units = {
            units[sample.sample_token]
            for sample in original
            if units[sample.sample_token] not in unsafe_units
        }
        safe_count = sum(
            units[sample.sample_token] not in unsafe_units for sample in original
        )
        if (
            len(safe_units) >= 2
            and safe_count
            >= policy.minimum_gallery_images + policy.minimum_query_images
        ):
            eligible.add(identity_token)
    return eligible


def _select_gallery_components(
    components: Mapping[str, int],
    *,
    identity_token: str,
    policy: DatasetStratifiedKFoldPolicy,
    evidence_root: str,
) -> set[str]:
    total = sum(components.values())
    target = math.floor(
        total * policy.gallery_fraction_numerator / policy.gallery_fraction_denominator
    )
    target = max(
        policy.minimum_gallery_images,
        min(target, total - policy.minimum_query_images),
    )

    def rank(component: str) -> str:
        return hashlib.sha256(
            (
                "CVI_KFOLD_GALLERY_COMPONENT_ORDER_V1\0"
                + evidence_root
                + "\0"
                + identity_token
                + "\0"
                + component
            ).encode("utf-8")
        ).hexdigest()

    ordered = sorted(components, key=lambda item: (rank(item), item))
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for component in ordered:
        count = components[component]
        for current, selected in sorted(reachable.items(), reverse=True):
            candidate = current + count
            if candidate <= total - policy.minimum_query_images and candidate not in reachable:
                reachable[candidate] = (*selected, component)
    candidates = [
        value for value in reachable if value >= policy.minimum_gallery_images
    ]
    if not candidates:
        return set()
    selected_count = min(candidates, key=lambda value: (abs(value - target), value > target, value))
    return set(reachable[selected_count])


def _close_retrieval_units(samples: Iterable[Any]) -> dict[str, str]:
    """Close duplicate components and sequences before gallery/query selection."""

    values = tuple(samples)
    if not values:
        return {}
    parent = list(range(len(values)))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            previous = parent[index]
            parent[index] = root
            index = previous
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_by_component: dict[str, int] = {}
    first_by_sequence: dict[str, int] = {}
    for index, sample in enumerate(values):
        for mapping, token in (
            (first_by_component, sample.component_token),
            (first_by_sequence, sample.sequence_token),
        ):
            prior = mapping.setdefault(token, index)
            union(prior, index)
    grouped: dict[int, list[Any]] = defaultdict(list)
    for index, sample in enumerate(values):
        grouped[find(index)].append(sample)
    unit_by_sample: dict[str, str] = {}
    for group in grouped.values():
        unit = content_sha256(
            {
                "sample_tokens": sorted(item.sample_token for item in group),
                "component_tokens": sorted({item.component_token for item in group}),
                "sequence_tokens": sorted({item.sequence_token for item in group}),
            }
        )
        for sample in group:
            unit_by_sample[sample.sample_token] = unit
    return unit_by_sample


def _dataset_fold_counts(
    assignments: Iterable[IdentityFoldAssignment], *, fold_count: int
) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    quarantined: Counter[str] = Counter()
    for item in assignments:
        if item.home_fold is None:
            quarantined[item.dataset_name] += 1
        else:
            counts[item.dataset_name][item.home_fold] += 1
    return tuple(
        (
            dataset,
            tuple(counts[dataset][fold] for fold in range(fold_count)),
            quarantined[dataset],
        )
        for dataset in sorted(set(counts) | set(quarantined))
    )


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


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
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"{name} schema differs")


__all__ = [
    "DatasetStratifiedIdentityKFoldManifest",
    "DatasetStratifiedKFoldPolicy",
    "FoldStage",
    "HeldOutSampleRole",
    "IdentityFoldAssignment",
    "SampleFoldAssignment",
    "build_dataset_stratified_identity_kfold",
    "dataset_stratified_kfold_bundle",
    "materialize_identity_fold",
    "read_dataset_stratified_identity_kfold",
]
