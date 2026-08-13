from __future__ import annotations

import copy
import uuid
from collections import Counter
from dataclasses import replace

import pytest

from foundation.provenance import content_sha256
from identity.full.full_split_census import (
    FullSplitAllocationPolicy,
    FullStatus,
    IdentityEvidenceKind,
    RegionStatus,
    TerminalRole,
    UnifiedFullCensus,
    UnifiedFullObservation,
    UnifiedFullSplitManifest,
    ViewScope,
    allocate_unified_full_split,
    build_unified_full_census,
    unified_full_split_bundle,
    validate_unified_full_split_bundle,
)
from identity.registry.generated_identity_registry import (
    GENERATED_DOG_NAMESPACE,
    compute_generated_identity_id,
    compute_source_cluster_token,
)
from identity.registry.identity_registry import (
    REGISTERED_DOG_NAMESPACE,
    compute_registered_dog_id,
)
from workflows.build_unified_full_split import REQUEST_SCHEMA, run


def _sha(value: str) -> str:
    return content_sha256({"fixture": value})


def _registered(dataset: str, identity: int) -> str:
    return compute_registered_dog_id(f"{dataset}:fixture:{identity}")


def _observation(
    index: int,
    *,
    dataset: str = "identity-set",
    identity: int | None = None,
    kind: IdentityEvidenceKind = IdentityEvidenceKind.REGISTERED,
    role: TerminalRole | None = None,
    official_split: str = "official-train",
    gradient_eligible: bool = True,
    validation_only: bool = False,
    source_group: str | None = None,
    capture_group: str | None = None,
    sequence_group: str | None = None,
    duplicate_component: str | None = None,
    view_scope: ViewScope = ViewScope.BODY_AVAILABLE,
) -> UnifiedFullObservation:
    identity = index if identity is None else identity
    if kind is IdentityEvidenceKind.REGISTERED:
        namespace = str(REGISTERED_DOG_NAMESPACE)
        identity_token = _registered(dataset, identity)
    elif kind is IdentityEvidenceKind.GENERATED:
        namespace = str(GENERATED_DOG_NAMESPACE)
        cluster = compute_source_cluster_token(f"{dataset}:cluster:{identity}")
        identity_token = compute_generated_identity_id("fixture-generator:v1", cluster)
    else:
        namespace = None
        identity_token = None
    native = view_scope in {ViewScope.FACE_NATIVE, ViewScope.HEAD_NATIVE}
    return UnifiedFullObservation(
        dataset_name=dataset,
        official_split=official_split,
        identity_evidence_kind=kind,
        identity_namespace_uuid=namespace,
        identity_token=identity_token,
        sample_token=_sha(f"sample:{index}"),
        source_group=source_group or f"{dataset}:source:{index}",
        capture_group=capture_group or f"{dataset}:capture:{identity}",
        sequence_group=sequence_group or f"{dataset}:sequence:{identity}",
        duplicate_component=duplicate_component or _sha(f"duplicate:{index}"),
        gradient_eligible=gradient_eligible,
        validation_only=validation_only,
        full_status=FullStatus.USABLE,
        face_status=RegionStatus.NATIVE if native else RegionStatus.USABLE,
        nose_status=RegionStatus.NOT_DETECTED if native else RegionStatus.USABLE,
        view_scope=view_scope,
        source_observation_sha256=_sha(f"Full-observation:{index}"),
        terminal_role=role,
    )


def _fixture() -> tuple[UnifiedFullObservation, ...]:
    shared_duplicate = _sha("cross-identity-duplicate")
    values = [
        _observation(0, identity=0, duplicate_component=shared_duplicate),
        _observation(
            1,
            dataset="other-set",
            identity=1,
            duplicate_component=shared_duplicate,
            view_scope=ViewScope.BODY_TRUNCATED,
        ),
        _observation(2, identity=2),
        _observation(3, identity=3),
        _observation(4, identity=4),
        _observation(5, identity=5),
        _observation(6, identity=6),
        _observation(7, identity=7),
        _observation(
            8,
            identity=8,
            role=TerminalRole.EVAL,
            official_split="official-test",
            gradient_eligible=False,
        ),
        _observation(
            9,
            identity=8,
            official_split="official-test",
            gradient_eligible=False,
        ),
        _observation(
            10,
            identity=10,
            role=TerminalRole.DEV,
            official_split="official-validation",
            gradient_eligible=False,
            validation_only=True,
        ),
        _observation(
            11,
            dataset="aux-set",
            kind=IdentityEvidenceKind.NONE,
            gradient_eligible=False,
            official_split="official-train",
            view_scope=ViewScope.FACE_NATIVE,
        ),
        _observation(
            12,
            dataset="generated-set",
            kind=IdentityEvidenceKind.GENERATED,
        ),
    ]
    return tuple(values)


def test_allocation_is_deterministic_disjoint_and_preserves_existing_roles() -> None:
    first = allocate_unified_full_split(
        allocation_name="phase-1-2-fixture", observations=_fixture()
    )
    second = allocate_unified_full_split(
        allocation_name="phase-1-2-fixture", observations=tuple(reversed(_fixture()))
    )
    assert first == second
    assert UnifiedFullSplitManifest.from_dict(first.to_dict()) == first
    by_sample = {item.sample_token: item for item in first.observations}
    assert by_sample[_sha("sample:8")].terminal_role is TerminalRole.EVAL
    assert by_sample[_sha("sample:9")].terminal_role is TerminalRole.EVAL
    assert by_sample[_sha("sample:10")].terminal_role is TerminalRole.DEV
    assert by_sample[_sha("sample:11")].terminal_role is TerminalRole.AUXILIARY
    assert by_sample[_sha("sample:11")].identity_token is None
    assert by_sample[_sha("sample:0")].terminal_role == by_sample[
        _sha("sample:1")
    ].terminal_role

    roles_by_constraint: dict[tuple[str, str], set[TerminalRole]] = {}
    for item in first.observations:
        constraints = (
            ("source", item.source_group),
            ("capture", item.capture_group),
            ("sequence", item.sequence_group),
            ("duplicate", item.duplicate_component),
        )
        for key in constraints:
            roles_by_constraint.setdefault(key, set()).add(item.terminal_role)
        if item.identity_token is not None:
            roles_by_constraint.setdefault(("identity", item.identity_token), set()).add(
                item.terminal_role
            )
    assert all(len(roles) == 1 for roles in roles_by_constraint.values())
    assert first.random_frame_splitting_used is False


def test_allocation_tracks_fit_dev_cal_targets_instead_of_starving_fit() -> None:
    observations = tuple(_observation(index + 100) for index in range(100))

    manifest = allocate_unified_full_split(
        allocation_name="target-ratio-regression", observations=observations
    )
    counts = Counter(item.terminal_role for item in manifest.observations)

    assert counts[TerminalRole.FIT] == 70
    assert counts[TerminalRole.DEV] == 15
    assert counts[TerminalRole.CAL] == 15


def test_census_reports_every_dimension_overlap_and_secondary_strata() -> None:
    manifest = allocate_unified_full_split(
        allocation_name="phase-1-2-fixture", observations=_fixture()
    )
    census = build_unified_full_census(manifest)
    payload = census.to_dict()
    assert census.observation_count == len(_fixture())
    assert census.identity_free_observation_count == 1
    assert payload["identity_free_metric_labels"] == []
    assert payload["dimension_counts"]["dataset"]["identity-set"] == 10
    assert payload["dimension_counts"]["official_split"]["official-test"] == 2
    assert all(not values for values in payload["overlap_report"].values())
    strata = payload["imbalance_report"]["stratum_observation_counts_by_role"]
    assert any("TRUNCATED" in key for key in strata)
    assert any("NATIVE_FACE" in key for key in strata)
    assert UnifiedFullCensus.from_dict(payload) == census
    bundle = unified_full_split_bundle(manifest, census)
    assert validate_unified_full_split_bundle(bundle) == (manifest, census)


def test_existing_official_assignment_conflict_fails_as_leakage() -> None:
    left = _observation(20, identity=20, role=TerminalRole.DEV)
    right = _observation(21, identity=20, role=TerminalRole.EVAL)
    with pytest.raises(ValueError, match="official assignments conflict"):
        allocate_unified_full_split(
            allocation_name="leaking-official-assignment",
            observations=(left, right),
        )

    policy = FullSplitAllocationPolicy()
    with pytest.raises(ValueError, match="crosses terminal roles"):
        UnifiedFullSplitManifest(
            allocation_name="forged-leakage",
            policy=policy,
            policy_sha256=policy.policy_sha256,
            observations=tuple(sorted((left, right), key=lambda item: item.sample_token)),
        )


def test_generated_and_registered_identity_namespaces_cannot_be_confused() -> None:
    generated = _observation(
        30, dataset="generated-set", kind=IdentityEvidenceKind.GENERATED
    )
    with pytest.raises(ValueError, match="registered/generated identities cannot be confused"):
        replace(
            generated,
            identity_evidence_kind=IdentityEvidenceKind.REGISTERED,
        )
    with pytest.raises(ValueError, match="registered/generated identities cannot be confused"):
        replace(
            generated,
            identity_evidence_kind=IdentityEvidenceKind.NONE,
            identity_token=None,
        )


def test_observability_and_bundle_tampering_fail_closed() -> None:
    record_payload = _observation(40).to_dict()
    record_payload["face_status"] = RegionStatus.REVIEW.value
    with pytest.raises(ValueError, match="observability digest"):
        UnifiedFullObservation.from_dict(record_payload)

    manifest = allocate_unified_full_split(
        allocation_name="phase-1-2-fixture", observations=_fixture()
    )
    bundle = unified_full_split_bundle(manifest)
    changed = copy.deepcopy(bundle)
    changed["census"]["dimension_counts"]["full_status"]["USABLE"] -= 1
    with pytest.raises(ValueError, match="census bundle content"):
        validate_unified_full_split_bundle(changed)


def test_capacity_impossibility_fails_before_assignment() -> None:
    observations = (
        _observation(
            50,
            identity=50,
            gradient_eligible=False,
            validation_only=True,
        ),
        _observation(
            51,
            identity=51,
            gradient_eligible=False,
            validation_only=True,
        ),
    )
    minimums = tuple(
        (role, 1 if role is TerminalRole.FIT else 0) for role in TerminalRole
    )
    policy = FullSplitAllocationPolicy(minimum_role_blocks=minimums)
    with pytest.raises(ValueError, match="capacity is impossible.*FIT"):
        allocate_unified_full_split(
            allocation_name="impossible-fit-capacity",
            observations=observations,
            policy=policy,
        )


def test_focused_workflow_builds_the_content_bound_manifest_and_census() -> None:
    policy = FullSplitAllocationPolicy()
    payload = {
        "schema_version": REQUEST_SCHEMA,
        "allocation_name": "phase-1-2-fixture",
        "policy": policy.to_dict(),
        "observations": [item.to_dict() for item in _fixture()],
    }
    bundle = run(payload)
    manifest, census = validate_unified_full_split_bundle(bundle)
    assert manifest.allocation_name == "phase-1-2-fixture"
    assert census.observation_count == len(_fixture())


def test_identity_token_must_be_canonical_uuid5() -> None:
    with pytest.raises(ValueError, match="canonical UUIDv5"):
        replace(_observation(60), identity_token=str(uuid.uuid4()))
