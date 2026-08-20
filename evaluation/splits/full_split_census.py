"""Unified Full observation split allocation and census contracts.

The manifest is observation-complete and allocation-oriented. It does not parse
datasets or produce localization evidence; it binds already observed evidence to
terminal research roles while preserving official assignments.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from shared.foundation.provenance import content_sha256
from enrollment.registry.generated_identity_registry import GENERATED_DOG_NAMESPACE
from enrollment.registry.identity_registry import REGISTERED_DOG_NAMESPACE

MANIFEST_SCHEMA = "cvi.unified_full_split_manifest.v1"
OBSERVATION_SCHEMA = "cvi.unified_full_split_observation.v1"
POLICY_SCHEMA = "cvi.unified_full_split_policy.v1"
CENSUS_SCHEMA = "cvi.unified_full_split_census.v1"
BUNDLE_SCHEMA = "cvi.unified_full_split_bundle.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOCATABLE_ROLES = ("FIT", "DEV", "CAL")
_INTERPRETATION = (
    "OBSERVATION_COMPLETE_FULL_FACE_NOSE_GOVERNANCE;"
    "OFFICIAL_ASSIGNMENTS_PRESERVED;GROUP_IDENTITY_DUPLICATE_DISJOINT;"
    "IDENTITY_FREE_OBSERVATIONS_HAVE_NO_METRIC_IDENTITY_LABEL"
)


class IdentityEvidenceKind(StrEnum):
    REGISTERED = "REGISTERED"
    GENERATED = "GENERATED"
    NONE = "NONE"


class TerminalRole(StrEnum):
    FIT = "FIT"
    DEV = "DEV"
    CAL = "CAL"
    EVAL = "EVAL"
    AUXILIARY = "AUXILIARY"
    BLOCKED = "BLOCKED"


class ViewScope(StrEnum):
    BODY_AVAILABLE = "BODY_AVAILABLE"
    BODY_TRUNCATED = "BODY_TRUNCATED"
    FACE_NATIVE = "FACE_NATIVE"
    HEAD_NATIVE = "HEAD_NATIVE"
    AMBIGUOUS = "AMBIGUOUS"
    UNAVAILABLE = "UNAVAILABLE"


class FullStatus(StrEnum):
    USABLE = "USABLE"
    REVIEW = "REVIEW"
    UNUSABLE = "UNUSABLE"
    AMBIGUOUS = "AMBIGUOUS"


class RegionStatus(StrEnum):
    NOT_RUN = "NOT_RUN"
    NOT_DETECTED = "NOT_DETECTED"
    REVIEW = "REVIEW"
    USABLE = "USABLE"
    NATIVE = "NATIVE"


@dataclass(frozen=True, slots=True)
class UnifiedFullObservation:
    dataset_name: str
    official_split: str
    identity_evidence_kind: IdentityEvidenceKind
    identity_namespace_uuid: str | None
    identity_token: str | None
    sample_token: str
    source_group: str
    capture_group: str
    sequence_group: str
    duplicate_component: str
    gradient_eligible: bool
    validation_only: bool
    full_status: FullStatus
    face_status: RegionStatus
    nose_status: RegionStatus
    view_scope: ViewScope
    source_observation_sha256: str
    terminal_role: TerminalRole | None = None
    schema_version: str = OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != OBSERVATION_SCHEMA:
            raise ValueError("unified Full observation schema differs")
        for value, name in (
            (self.dataset_name, "dataset_name"),
            (self.official_split, "official_split"),
            (self.source_group, "source_group"),
            (self.capture_group, "capture_group"),
            (self.sequence_group, "sequence_group"),
        ):
            _require_text(value, name)
        for value, name in (
            (self.sample_token, "sample_token"),
            (self.duplicate_component, "duplicate_component"),
            (self.source_observation_sha256, "source_observation_sha256"),
        ):
            _require_sha256(value, name)
        if not isinstance(self.identity_evidence_kind, IdentityEvidenceKind):
            raise TypeError("identity_evidence_kind must use its exact enum")
        self._validate_identity_evidence()
        for value, name in (
            (self.gradient_eligible, "gradient_eligible"),
            (self.validation_only, "validation_only"),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        if not isinstance(self.full_status, FullStatus):
            raise TypeError("full_status must use its exact enum")
        if not isinstance(self.face_status, RegionStatus) or not isinstance(
            self.nose_status, RegionStatus
        ):
            raise TypeError("Face and Nose status must use the exact region enum")
        if not isinstance(self.view_scope, ViewScope):
            raise TypeError("view_scope must use its exact enum")
        if self.terminal_role is not None and not isinstance(
            self.terminal_role, TerminalRole
        ):
            raise TypeError("terminal_role must use its exact enum or be null")
        if self.validation_only and self.gradient_eligible:
            raise ValueError("validation-only observations cannot be gradient eligible")
        if self.validation_only and self.terminal_role is TerminalRole.FIT:
            raise ValueError("validation-only observations cannot have the FIT role")
        if self.terminal_role is TerminalRole.FIT and not self.gradient_eligible:
            raise ValueError("FIT observations must be gradient eligible")
        self._validate_observability()

    def _validate_identity_evidence(self) -> None:
        expected_namespace = {
            IdentityEvidenceKind.REGISTERED: str(REGISTERED_DOG_NAMESPACE),
            IdentityEvidenceKind.GENERATED: str(GENERATED_DOG_NAMESPACE),
            IdentityEvidenceKind.NONE: None,
        }[self.identity_evidence_kind]
        if self.identity_namespace_uuid != expected_namespace:
            raise ValueError(
                "identity evidence kind and namespace differ; registered/generated "
                "identities cannot be confused"
            )
        if self.identity_evidence_kind is IdentityEvidenceKind.NONE:
            if self.identity_token is not None:
                raise ValueError("identity-free observation cannot carry an identity token")
            return
        if not _is_uuid5(self.identity_token):
            raise ValueError("identity_token must be a canonical UUIDv5")

    def _validate_observability(self) -> None:
        native = {ViewScope.FACE_NATIVE, ViewScope.HEAD_NATIVE}
        if self.view_scope in native:
            if self.full_status is not FullStatus.USABLE:
                raise ValueError("native Face/Head view requires usable Full appearance")
            if self.face_status is not RegionStatus.NATIVE:
                raise ValueError("native Face/Head view must retain native Face status")
        if self.view_scope in {
            ViewScope.BODY_AVAILABLE,
            ViewScope.BODY_TRUNCATED,
        } and RegionStatus.NATIVE in {self.face_status, self.nose_status}:
            raise ValueError("body view cannot claim native Face/Nose status")
        if (
            self.view_scope is ViewScope.AMBIGUOUS
            and self.full_status is not FullStatus.AMBIGUOUS
        ):
            raise ValueError("ambiguous view must retain ambiguous Full status")
        if (
            self.view_scope is ViewScope.UNAVAILABLE
            and self.full_status is not FullStatus.UNUSABLE
        ):
            raise ValueError("unavailable view must retain unusable Full status")

    @property
    def observability_sha256(self) -> str:
        return content_sha256(
            {
                "source_observation_sha256": self.source_observation_sha256,
                "full_status": self.full_status.value,
                "face_status": self.face_status.value,
                "nose_status": self.nose_status.value,
                "view_scope": self.view_scope.value,
            }
        )

    @property
    def record_sha256(self) -> str:
        return content_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "official_split": self.official_split,
            "identity_evidence_kind": self.identity_evidence_kind.value,
            "identity_namespace_uuid": self.identity_namespace_uuid,
            "identity_token": self.identity_token,
            "sample_token": self.sample_token,
            "source_group": self.source_group,
            "capture_group": self.capture_group,
            "sequence_group": self.sequence_group,
            "duplicate_component": self.duplicate_component,
            "gradient_eligible": self.gradient_eligible,
            "validation_only": self.validation_only,
            "full_status": self.full_status.value,
            "face_status": self.face_status.value,
            "nose_status": self.nose_status.value,
            "view_scope": self.view_scope.value,
            "source_observation_sha256": self.source_observation_sha256,
            "terminal_role": (
                None if self.terminal_role is None else self.terminal_role.value
            ),
            "observability_sha256": self.observability_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "record_sha256": self.record_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UnifiedFullObservation:
        expected = set(cls.__dataclass_fields__) | {
            "observability_sha256",
            "record_sha256",
        }
        _exact_keys(payload, expected, "unified Full observation")
        try:
            record = cls(
                dataset_name=payload["dataset_name"],
                official_split=payload["official_split"],
                identity_evidence_kind=IdentityEvidenceKind(
                    payload["identity_evidence_kind"]
                ),
                identity_namespace_uuid=payload["identity_namespace_uuid"],
                identity_token=payload["identity_token"],
                sample_token=payload["sample_token"],
                source_group=payload["source_group"],
                capture_group=payload["capture_group"],
                sequence_group=payload["sequence_group"],
                duplicate_component=payload["duplicate_component"],
                gradient_eligible=payload["gradient_eligible"],
                validation_only=payload["validation_only"],
                full_status=FullStatus(payload["full_status"]),
                face_status=RegionStatus(payload["face_status"]),
                nose_status=RegionStatus(payload["nose_status"]),
                view_scope=ViewScope(payload["view_scope"]),
                source_observation_sha256=payload["source_observation_sha256"],
                terminal_role=(
                    None
                    if payload["terminal_role"] is None
                    else TerminalRole(payload["terminal_role"])
                ),
                schema_version=payload["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("unified Full observation values differ") from exc
        if payload["observability_sha256"] != record.observability_sha256:
            raise ValueError("unified Full observability digest differs")
        if payload["record_sha256"] != record.record_sha256:
            raise ValueError("unified Full observation record digest differs")
        return record


def _default_percentages() -> tuple[tuple[TerminalRole, int], ...]:
    return (
        (TerminalRole.FIT, 70),
        (TerminalRole.DEV, 15),
        (TerminalRole.CAL, 15),
    )


def _default_minimums() -> tuple[tuple[TerminalRole, int], ...]:
    return tuple((role, 0) for role in TerminalRole)


@dataclass(frozen=True, slots=True)
class FullSplitAllocationPolicy:
    role_target_percentages: tuple[
        tuple[TerminalRole, int], ...
    ] = _default_percentages()
    minimum_role_blocks: tuple[tuple[TerminalRole, int], ...] = _default_minimums()
    schema_version: str = POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_SCHEMA:
            raise ValueError("unified Full split policy schema differs")
        if tuple(role for role, _ in self.role_target_percentages) != tuple(
            TerminalRole(value) for value in _ALLOCATABLE_ROLES
        ):
            raise ValueError("role target percentages must contain FIT, DEV, CAL")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for _, value in self.role_target_percentages
        ) or sum(value for _, value in self.role_target_percentages) != 100:
            raise ValueError("role target percentages must be nonnegative and sum to 100")
        if tuple(role for role, _ in self.minimum_role_blocks) != tuple(TerminalRole):
            raise ValueError("minimum_role_blocks must contain every terminal role")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for _, value in self.minimum_role_blocks
        ):
            raise ValueError("minimum role block counts must be nonnegative integers")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role_target_percentages": {
                role.value: value for role, value in self.role_target_percentages
            },
            "minimum_role_blocks": {
                role.value: value for role, value in self.minimum_role_blocks
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FullSplitAllocationPolicy:
        _exact_keys(
            payload,
            {"schema_version", "role_target_percentages", "minimum_role_blocks"},
            "unified Full split policy",
        )
        percentages = payload["role_target_percentages"]
        minimums = payload["minimum_role_blocks"]
        if not isinstance(percentages, Mapping) or not isinstance(minimums, Mapping):
            raise TypeError("unified Full split policy counts must be objects")
        if set(percentages) != set(_ALLOCATABLE_ROLES) or set(minimums) != {
            role.value for role in TerminalRole
        }:
            raise ValueError("unified Full split policy role fields differ")
        return cls(
            role_target_percentages=tuple(
                (TerminalRole(role), percentages[role]) for role in _ALLOCATABLE_ROLES
            ),
            minimum_role_blocks=tuple(
                (role, minimums[role.value]) for role in TerminalRole
            ),
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class UnifiedFullSplitManifest:
    allocation_name: str
    policy: FullSplitAllocationPolicy
    policy_sha256: str
    observations: tuple[UnifiedFullObservation, ...]
    score_inputs_used: bool = False
    random_frame_splitting_used: bool = False
    interpretation: str = _INTERPRETATION
    schema_version: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA:
            raise ValueError("unified Full split manifest schema differs")
        _require_text(self.allocation_name, "allocation_name")
        if not isinstance(self.policy, FullSplitAllocationPolicy):
            raise TypeError("policy must be FullSplitAllocationPolicy")
        if self.policy_sha256 != self.policy.policy_sha256:
            raise ValueError("embedded unified Full split policy digest differs")
        if self.score_inputs_used is not False:
            raise ValueError("unified Full split allocation must be score blind")
        if self.random_frame_splitting_used is not False:
            raise ValueError("random frame splitting is prohibited")
        if self.interpretation != _INTERPRETATION:
            raise ValueError("unified Full split interpretation differs")
        if not self.observations:
            raise ValueError("unified Full split manifest must not be empty")
        if self.observations != tuple(
            sorted(self.observations, key=lambda item: item.sample_token)
        ):
            raise ValueError("unified Full observations must be canonically sorted")
        if any(item.terminal_role is None for item in self.observations):
            raise ValueError("every unified Full observation requires a terminal role")
        _validate_assignment_closure(self.observations)
        _validate_minimum_role_blocks(self.observations, self.policy)

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "allocation_name": self.allocation_name,
            "policy": self.policy.to_dict(),
            "policy_sha256": self.policy_sha256,
            "observations": [item.to_dict() for item in self.observations],
            "score_inputs_used": self.score_inputs_used,
            "random_frame_splitting_used": self.random_frame_splitting_used,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UnifiedFullSplitManifest:
        _exact_keys(payload, set(cls.__dataclass_fields__), "unified Full split manifest")
        if not isinstance(payload["observations"], list):
            raise TypeError("unified Full observations must be an array")
        return cls(
            allocation_name=payload["allocation_name"],
            policy=FullSplitAllocationPolicy.from_dict(payload["policy"]),
            policy_sha256=payload["policy_sha256"],
            observations=tuple(
                UnifiedFullObservation.from_dict(item)
                for item in payload["observations"]
            ),
            score_inputs_used=payload["score_inputs_used"],
            random_frame_splitting_used=payload["random_frame_splitting_used"],
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class _AllocationBlock:
    sample_tokens: tuple[str, ...]
    fixed_role: TerminalRole | None
    allowed_roles: tuple[TerminalRole, ...]
    stratum: str


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            self.parent[larger] = smaller


def allocate_unified_full_split(
    *,
    allocation_name: str,
    observations: Sequence[UnifiedFullObservation],
    policy: FullSplitAllocationPolicy | None = None,
) -> UnifiedFullSplitManifest:
    """Allocate deterministic atomic blocks without changing supplied roles."""

    _require_text(allocation_name, "allocation_name")
    policy = FullSplitAllocationPolicy() if policy is None else policy
    if not isinstance(policy, FullSplitAllocationPolicy):
        raise TypeError("policy must be FullSplitAllocationPolicy")
    values = tuple(observations)
    if not values:
        raise ValueError("at least one unified Full observation is required")
    if any(not isinstance(item, UnifiedFullObservation) for item in values):
        raise TypeError("observations must contain UnifiedFullObservation values")
    if len({item.sample_token for item in values}) != len(values):
        raise ValueError("unified Full observations repeat a sample token")

    blocks = _allocation_blocks(values)
    _validate_capacity(blocks, policy)
    target_percent = dict(policy.role_target_percentages)
    minimums = dict(policy.minimum_role_blocks)
    role_by_sample: dict[str, TerminalRole] = {}
    block_counts: Counter[TerminalRole] = Counter()
    observation_counts: Counter[TerminalRole] = Counter()
    stratum_counts: dict[str, Counter[TerminalRole]] = defaultdict(Counter)

    for block in blocks:
        if block.fixed_role is not None:
            role = block.fixed_role
            block_counts[role] += 1
            observation_counts[role] += len(block.sample_tokens)
            stratum_counts[block.stratum][role] += len(block.sample_tokens)
            role_by_sample.update((token, role) for token in block.sample_tokens)

    unassigned = [block for block in blocks if block.fixed_role is None]
    unassigned.sort(
        key=lambda block: (
            len(block.allowed_roles),
            -len(block.sample_tokens),
            block.stratum,
            content_sha256(
                {
                    "domain": "CVI_UNIFIED_FULL_BLOCK_ORDER_V1",
                    "allocation_name": allocation_name,
                    "policy_sha256": policy.policy_sha256,
                    "sample_tokens": list(block.sample_tokens),
                }
            ),
        )
    )
    for block in unassigned:
        global_total_after = (
            sum(
                observation_counts[TerminalRole(name)]
                for name in _ALLOCATABLE_ROLES
            )
            + len(block.sample_tokens)
        )
        stratum_total_after = (
            sum(
                stratum_counts[block.stratum][TerminalRole(name)]
                for name in _ALLOCATABLE_ROLES
            )
            + len(block.sample_tokens)
        )

        def objective(
            role: TerminalRole,
            block: _AllocationBlock = block,
            global_total_after: int = global_total_after,
            stratum_total_after: int = stratum_total_after,
        ) -> tuple[Any, ...]:
            projected_blocks = block_counts[role] + 1
            unmet = max(0, minimums[role] - projected_blocks)
            total_minimum_deficit = sum(
                max(
                    0,
                    minimums[candidate]
                    - block_counts[candidate]
                    - (1 if candidate is role else 0),
                )
                for candidate in TerminalRole
            )
            global_error = sum(
                abs(
                    (
                        observation_counts[TerminalRole(name)]
                        + (
                            len(block.sample_tokens)
                            if TerminalRole(name) is role
                            else 0
                        )
                    )
                    * 100
                    - global_total_after * target_percent[TerminalRole(name)]
                )
                for name in _ALLOCATABLE_ROLES
            )
            stratum_error = sum(
                abs(
                    (
                        stratum_counts[block.stratum][TerminalRole(name)]
                        + (
                            len(block.sample_tokens)
                            if TerminalRole(name) is role
                            else 0
                        )
                    )
                    * 100
                    - stratum_total_after * target_percent[TerminalRole(name)]
                )
                for name in _ALLOCATABLE_ROLES
            )
            return (
                total_minimum_deficit,
                unmet,
                global_error,
                stratum_error,
                _ALLOCATABLE_ROLES.index(role.value),
            )

        role = min(block.allowed_roles, key=objective)
        block_counts[role] += 1
        observation_counts[role] += len(block.sample_tokens)
        stratum_counts[block.stratum][role] += len(block.sample_tokens)
        role_by_sample.update((token, role) for token in block.sample_tokens)

    assigned = tuple(
        sorted(
            (
                (
                    item
                    if item.terminal_role is not None
                    else replace(
                        item, terminal_role=role_by_sample[item.sample_token]
                    )
                )
                for item in values
            ),
            key=lambda item: item.sample_token,
        )
    )
    return UnifiedFullSplitManifest(
        allocation_name=allocation_name,
        policy=policy,
        policy_sha256=policy.policy_sha256,
        observations=assigned,
    )


def _allocation_blocks(
    observations: Sequence[UnifiedFullObservation],
) -> tuple[_AllocationBlock, ...]:
    by_sample = {item.sample_token: item for item in observations}
    dsu = _DisjointSet(by_sample)
    first_by_constraint: dict[tuple[str, str], str] = {}
    for item in observations:
        constraints = [
            ("source", item.source_group),
            ("capture", item.capture_group),
            ("sequence", item.sequence_group),
            ("duplicate", item.duplicate_component),
        ]
        if item.identity_token is not None:
            constraints.append(("identity", item.identity_token))
        for constraint in constraints:
            prior = first_by_constraint.setdefault(constraint, item.sample_token)
            dsu.union(prior, item.sample_token)
    grouped: dict[str, list[UnifiedFullObservation]] = defaultdict(list)
    for item in observations:
        grouped[dsu.find(item.sample_token)].append(item)

    identity_sizes = Counter(
        item.identity_token
        for item in observations
        if item.identity_token is not None
    )
    blocks: list[_AllocationBlock] = []
    for members in grouped.values():
        supplied_roles = {item.terminal_role for item in members if item.terminal_role}
        if len(supplied_roles) > 1:
            raise ValueError(
                "existing official assignments conflict inside one identity/group/"
                "duplicate allocation block"
            )
        fixed_role = next(iter(supplied_roles), None)
        if fixed_role is not None:
            _validate_fixed_block_role(members, fixed_role)
            allowed = (fixed_role,)
        elif all(item.identity_token is None for item in members):
            allowed = (TerminalRole.AUXILIARY,)
            fixed_role = TerminalRole.AUXILIARY
        else:
            allowed_values = [TerminalRole.DEV, TerminalRole.CAL]
            if all(item.gradient_eligible and not item.validation_only for item in members):
                allowed_values.insert(0, TerminalRole.FIT)
            allowed = tuple(allowed_values)
        blocks.append(
            _AllocationBlock(
                sample_tokens=tuple(sorted(item.sample_token for item in members)),
                fixed_role=fixed_role,
                allowed_roles=allowed,
                stratum=_stratum(members, identity_sizes),
            )
        )
    return tuple(sorted(blocks, key=lambda block: block.sample_tokens))


def _validate_fixed_block_role(
    members: Sequence[UnifiedFullObservation], role: TerminalRole
) -> None:
    if role is TerminalRole.FIT and any(
        not item.gradient_eligible or item.validation_only for item in members
    ):
        raise ValueError("existing FIT assignment contains ineligible observations")


def _validate_capacity(
    blocks: Sequence[_AllocationBlock], policy: FullSplitAllocationPolicy
) -> None:
    minimums = dict(policy.minimum_role_blocks)
    fixed = Counter(
        block.fixed_role for block in blocks if block.fixed_role is not None
    )
    candidates = [block for block in blocks if block.fixed_role is None]
    possible = Counter(fixed)
    for role in TerminalRole:
        possible[role] += sum(role in block.allowed_roles for block in candidates)
    impossible = [
        role.value for role in TerminalRole if possible[role] < minimums[role]
    ]
    remaining_allocatable_minimum = sum(
        max(0, minimums[TerminalRole(role)] - fixed[TerminalRole(role)])
        for role in _ALLOCATABLE_ROLES
    )
    if remaining_allocatable_minimum > len(candidates):
        impossible.extend(_ALLOCATABLE_ROLES)
    if impossible:
        raise ValueError(
            "unified Full split capacity is impossible for required roles: "
            + ", ".join(sorted(set(impossible)))
        )


def _validate_assignment_closure(
    observations: Sequence[UnifiedFullObservation],
) -> None:
    if len({item.sample_token for item in observations}) != len(observations):
        raise ValueError("unified Full manifest repeats a sample token")
    role_by_constraint: dict[tuple[str, str], TerminalRole] = {}
    for item in observations:
        if item.terminal_role is None:
            raise ValueError("terminal role is required")
        constraints = [
            ("source group", item.source_group),
            ("capture group", item.capture_group),
            ("sequence group", item.sequence_group),
            ("duplicate component", item.duplicate_component),
        ]
        if item.identity_token is not None:
            constraints.append(("identity", item.identity_token))
        for kind, token in constraints:
            prior = role_by_constraint.setdefault((kind, token), item.terminal_role)
            if prior is not item.terminal_role:
                raise ValueError(f"{kind} crosses terminal roles")


def _validate_minimum_role_blocks(
    observations: Sequence[UnifiedFullObservation],
    policy: FullSplitAllocationPolicy,
) -> None:
    blocks = _allocation_blocks(observations)
    counts = Counter(block.fixed_role for block in blocks)
    for role, minimum in policy.minimum_role_blocks:
        if counts[role] < minimum:
            raise ValueError(f"terminal role {role.value} misses its block minimum")


def _stratum(
    members: Sequence[UnifiedFullObservation], identity_sizes: Mapping[str, int]
) -> str:
    datasets = "+".join(sorted({item.dataset_name for item in members}))
    sizes = [
        identity_sizes[item.identity_token]
        for item in members
        if item.identity_token is not None
    ]
    identity_bucket = _size_bucket(max(sizes, default=0))
    full_bucket = _rate_bucket(
        sum(item.full_status is FullStatus.USABLE for item in members), len(members)
    )
    face_bucket = _rate_bucket(
        sum(item.face_status in {RegionStatus.USABLE, RegionStatus.NATIVE} for item in members),
        len(members),
    )
    nose_bucket = _rate_bucket(
        sum(item.nose_status in {RegionStatus.USABLE, RegionStatus.NATIVE} for item in members),
        len(members),
    )
    views: set[str] = set()
    for item in members:
        if item.view_scope is ViewScope.BODY_TRUNCATED:
            views.add("TRUNCATED")
        elif item.view_scope in {ViewScope.FACE_NATIVE, ViewScope.HEAD_NATIVE}:
            views.add("NATIVE_FACE")
        else:
            views.add("OTHER")
    return "|".join(
        (
            datasets,
            identity_bucket,
            full_bucket,
            face_bucket,
            nose_bucket,
            "+".join(sorted(views)),
        )
    )


def _size_bucket(size: int) -> str:
    if size == 0:
        return "IDENTITY_FREE"
    if size == 1:
        return "SINGLETON"
    if size <= 4:
        return "SMALL_2_4"
    if size <= 19:
        return "MEDIUM_5_19"
    return "LARGE_20_PLUS"


def _rate_bucket(usable: int, total: int) -> str:
    if usable == 0:
        return "ZERO"
    if usable == total:
        return "ALL"
    if usable * 4 <= total:
        return "LOW"
    if usable * 2 <= total:
        return "MID"
    return "HIGH"


@dataclass(frozen=True, slots=True)
class UnifiedFullCensus:
    manifest_sha256: str
    observation_count: int
    registered_metric_identity_count: int
    generated_identity_count: int
    identity_free_observation_count: int
    dimension_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    overlap_report: tuple[tuple[str, tuple[str, ...]], ...]
    imbalance_report: tuple[tuple[str, Any], ...]
    schema_version: str = CENSUS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CENSUS_SCHEMA:
            raise ValueError("unified Full census schema differs")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        for value, name in (
            (self.observation_count, "observation_count"),
            (self.registered_metric_identity_count, "registered_metric_identity_count"),
            (self.generated_identity_count, "generated_identity_count"),
            (self.identity_free_observation_count, "identity_free_observation_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.identity_free_observation_count > self.observation_count:
            raise ValueError("identity-free census count exceeds observation count")
        if self.dimension_counts != tuple(sorted(self.dimension_counts)):
            raise ValueError("census dimensions must be canonically sorted")
        if self.overlap_report != tuple(sorted(self.overlap_report)):
            raise ValueError("overlap report must be canonically sorted")
        if any(values for _, values in self.overlap_report):
            raise ValueError("split overlap report contains leakage")

    @property
    def census_sha256(self) -> str:
        return content_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
            "observation_count": self.observation_count,
            "registered_metric_identity_count": self.registered_metric_identity_count,
            "generated_identity_count": self.generated_identity_count,
            "identity_free_observation_count": self.identity_free_observation_count,
            "identity_free_metric_labels": [],
            "dimension_counts": {
                dimension: dict(values) for dimension, values in self.dimension_counts
            },
            "overlap_report": {
                dimension: list(values) for dimension, values in self.overlap_report
            },
            "imbalance_report": dict(self.imbalance_report),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "census_sha256": self.census_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UnifiedFullCensus:
        expected = {
            "schema_version",
            "manifest_sha256",
            "observation_count",
            "registered_metric_identity_count",
            "generated_identity_count",
            "identity_free_observation_count",
            "identity_free_metric_labels",
            "dimension_counts",
            "overlap_report",
            "imbalance_report",
            "census_sha256",
        }
        _exact_keys(payload, expected, "unified Full census")
        if payload["identity_free_metric_labels"] != []:
            raise ValueError("identity-free observations cannot receive metric labels")
        raw_dimensions = payload["dimension_counts"]
        raw_overlap = payload["overlap_report"]
        raw_imbalance = payload["imbalance_report"]
        if not all(
            isinstance(value, Mapping)
            for value in (raw_dimensions, raw_overlap, raw_imbalance)
        ):
            raise TypeError("unified Full census reports must be objects")
        dimensions: list[tuple[str, tuple[tuple[str, int], ...]]] = []
        for name, values in sorted(raw_dimensions.items()):
            if not isinstance(values, Mapping):
                raise TypeError("unified Full census dimension must be an object")
            counts = tuple(sorted(values.items()))
            if any(
                not isinstance(key, str)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for key, count in counts
            ):
                raise ValueError("unified Full census dimension counts differ")
            dimensions.append((name, counts))
        overlap: list[tuple[str, tuple[str, ...]]] = []
        for name, values in sorted(raw_overlap.items()):
            if not isinstance(values, list) or any(
                not isinstance(item, str) for item in values
            ):
                raise TypeError("unified Full overlap values must be string arrays")
            overlap.append((name, tuple(values)))
        census = cls(
            manifest_sha256=payload["manifest_sha256"],
            observation_count=payload["observation_count"],
            registered_metric_identity_count=payload[
                "registered_metric_identity_count"
            ],
            generated_identity_count=payload["generated_identity_count"],
            identity_free_observation_count=payload[
                "identity_free_observation_count"
            ],
            dimension_counts=tuple(dimensions),
            overlap_report=tuple(overlap),
            imbalance_report=tuple(sorted(raw_imbalance.items())),
            schema_version=payload["schema_version"],
        )
        if payload["census_sha256"] != census.census_sha256:
            raise ValueError("unified Full census digest differs")
        return census


def build_unified_full_census(
    manifest: UnifiedFullSplitManifest,
) -> UnifiedFullCensus:
    if not isinstance(manifest, UnifiedFullSplitManifest):
        raise TypeError("manifest must be UnifiedFullSplitManifest")
    observations = manifest.observations
    dimensions: dict[str, Counter[str]] = {
        "dataset": Counter(item.dataset_name for item in observations),
        "official_split": Counter(item.official_split for item in observations),
        "identity_evidence_kind": Counter(
            item.identity_evidence_kind.value for item in observations
        ),
        "terminal_role": Counter(item.terminal_role.value for item in observations),
        "gradient_eligible": Counter(str(item.gradient_eligible).lower() for item in observations),
        "validation_only": Counter(str(item.validation_only).lower() for item in observations),
        "full_status": Counter(item.full_status.value for item in observations),
        "face_status": Counter(item.face_status.value for item in observations),
        "nose_status": Counter(item.nose_status.value for item in observations),
        "view_scope": Counter(item.view_scope.value for item in observations),
        "dataset_role": Counter(
            f"{item.dataset_name}|{item.terminal_role.value}" for item in observations
        ),
        "dataset_official_split": Counter(
            f"{item.dataset_name}|{item.official_split}" for item in observations
        ),
        "dataset_full_status": Counter(
            f"{item.dataset_name}|{item.full_status.value}" for item in observations
        ),
        "dataset_face_status": Counter(
            f"{item.dataset_name}|{item.face_status.value}" for item in observations
        ),
        "dataset_nose_status": Counter(
            f"{item.dataset_name}|{item.nose_status.value}" for item in observations
        ),
        "dataset_view_scope": Counter(
            f"{item.dataset_name}|{item.view_scope.value}" for item in observations
        ),
    }
    overlap = _overlap_report(observations)
    blocks = _allocation_blocks(observations)
    block_roles = Counter(block.fixed_role.value for block in blocks if block.fixed_role)
    observation_roles = Counter(item.terminal_role.value for item in observations)
    target_percentages = dict(manifest.policy.role_target_percentages)
    allocatable_total = sum(observation_roles[role] for role in _ALLOCATABLE_ROLES)
    target_deviation = {
        role: observation_roles[role] * 100
        - allocatable_total * target_percentages[TerminalRole(role)]
        for role in _ALLOCATABLE_ROLES
    }
    stratum_roles: dict[str, Counter[str]] = defaultdict(Counter)
    for block in blocks:
        if block.fixed_role is not None:
            stratum_roles[block.stratum][block.fixed_role.value] += len(
                block.sample_tokens
            )
    imbalance: tuple[tuple[str, Any], ...] = tuple(
        sorted(
            {
                "allocation_block_counts_by_role": dict(sorted(block_roles.items())),
                "observation_counts_by_role": dict(sorted(observation_roles.items())),
                "target_deviation_percentage_points_x_observations": target_deviation,
                "stratum_observation_counts_by_role": {
                    key: dict(sorted(value.items()))
                    for key, value in sorted(stratum_roles.items())
                },
            }.items()
        )
    )
    registered = {
        item.identity_token
        for item in observations
        if item.identity_evidence_kind is IdentityEvidenceKind.REGISTERED
    }
    generated = {
        item.identity_token
        for item in observations
        if item.identity_evidence_kind is IdentityEvidenceKind.GENERATED
    }
    return UnifiedFullCensus(
        manifest_sha256=manifest.manifest_sha256,
        observation_count=len(observations),
        registered_metric_identity_count=len(registered),
        generated_identity_count=len(generated),
        identity_free_observation_count=sum(
            item.identity_evidence_kind is IdentityEvidenceKind.NONE
            for item in observations
        ),
        dimension_counts=tuple(
            sorted(
                (name, tuple(sorted(counts.items())))
                for name, counts in dimensions.items()
            )
        ),
        overlap_report=overlap,
        imbalance_report=imbalance,
    )


def _overlap_report(
    observations: Sequence[UnifiedFullObservation],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    dimensions: dict[str, dict[str, set[TerminalRole]]] = {
        "identity": defaultdict(set),
        "source_group": defaultdict(set),
        "capture_group": defaultdict(set),
        "sequence_group": defaultdict(set),
        "duplicate_component": defaultdict(set),
    }
    for item in observations:
        if item.terminal_role is None:
            continue
        if item.identity_token is not None:
            dimensions["identity"][item.identity_token].add(item.terminal_role)
        for name in (
            "source_group",
            "capture_group",
            "sequence_group",
            "duplicate_component",
        ):
            dimensions[name][getattr(item, name)].add(item.terminal_role)
    return tuple(
        sorted(
            (
                name,
                tuple(sorted(token for token, roles in values.items() if len(roles) > 1)),
            )
            for name, values in dimensions.items()
        )
    )


def unified_full_split_bundle(
    manifest: UnifiedFullSplitManifest,
    census: UnifiedFullCensus | None = None,
) -> dict[str, Any]:
    census = build_unified_full_census(manifest) if census is None else census
    if census != build_unified_full_census(manifest):
        raise ValueError("unified Full census differs from its manifest")
    return {
        "schema_version": BUNDLE_SCHEMA,
        "manifest_sha256": manifest.manifest_sha256,
        "census_sha256": census.census_sha256,
        "manifest": manifest.to_dict(),
        "census": census.to_dict(),
    }


def validate_unified_full_split_bundle(
    payload: Mapping[str, Any],
) -> tuple[UnifiedFullSplitManifest, UnifiedFullCensus]:
    _exact_keys(
        payload,
        {"schema_version", "manifest_sha256", "census_sha256", "manifest", "census"},
        "unified Full split bundle",
    )
    if payload["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError("unified Full split bundle schema differs")
    manifest = UnifiedFullSplitManifest.from_dict(payload["manifest"])
    if payload["manifest_sha256"] != manifest.manifest_sha256:
        raise ValueError("unified Full split manifest bundle digest differs")
    expected = build_unified_full_census(manifest)
    if payload["census"] != expected.to_dict():
        raise ValueError("unified Full split census bundle content differs")
    if payload["census_sha256"] != expected.census_sha256:
        raise ValueError("unified Full split census bundle digest differs")
    return manifest, expected


def _is_uuid5(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return False
    return parsed.version == 5 and str(parsed) == value


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_text(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > 512
    ):
        raise ValueError(f"{name} must be bounded non-empty canonical text")


def _exact_keys(payload: object, expected: set[str], name: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"{name} fields differ")


__all__ = [
    "BUNDLE_SCHEMA",
    "CENSUS_SCHEMA",
    "MANIFEST_SCHEMA",
    "OBSERVATION_SCHEMA",
    "POLICY_SCHEMA",
    "FullSplitAllocationPolicy",
    "FullStatus",
    "IdentityEvidenceKind",
    "RegionStatus",
    "TerminalRole",
    "UnifiedFullCensus",
    "UnifiedFullObservation",
    "UnifiedFullSplitManifest",
    "ViewScope",
    "allocate_unified_full_split",
    "build_unified_full_census",
    "unified_full_split_bundle",
    "validate_unified_full_split_bundle",
]
