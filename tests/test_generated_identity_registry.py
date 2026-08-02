from __future__ import annotations

import uuid
from copy import deepcopy

import pytest

from identity_governance.generated_identity_registry import (
    GENERATED_DOG_NAMESPACE,
    GeneratedIdentityRegistry,
    GeneratedIdentityStatus,
    compute_generated_identity_id,
    compute_source_cluster_token,
    create_provisional_identity,
    transition_generated_identity,
)
from identity_governance.identity_registry import compute_registered_dog_id


def test_generated_namespace_is_stable_and_separate() -> None:
    assert str(GENERATED_DOG_NAMESPACE) == "6c7f371b-120e-530e-b814-a3f24a4d670a"
    record = create_provisional_identity("cvi.track-cluster:v1", "video:1:cluster:2", 8)
    assert record.generated_identity_id != compute_registered_dog_id(
        "video:1:cluster:2"
    )


def test_provisional_identity_is_deterministic_without_storing_source_label() -> None:
    first = create_provisional_identity("cvi.track-cluster:v1", "private label", 4)
    second = create_provisional_identity("cvi.track-cluster:v1", "private label", 4)
    assert first == second
    assert first.status is GeneratedIdentityStatus.PROVISIONAL
    assert first.source_cluster_token == compute_source_cluster_token("private label")
    assert "private label" not in str(first.to_dict())


def test_transition_to_registered_requires_canonical_uuid5() -> None:
    record = create_provisional_identity("cvi.track-cluster:v1", "cluster-a", 3)
    registered_id = compute_registered_dog_id("manual-enrollment:dog:1")
    merged = transition_generated_identity(
        record,
        GeneratedIdentityStatus.MERGED_TO_REGISTERED,
        registered_identity_id=registered_id,
    )
    assert merged.registered_identity_id == registered_id
    with pytest.raises(ValueError, match="only a provisional"):
        transition_generated_identity(merged, GeneratedIdentityStatus.REJECTED)
    with pytest.raises(ValueError, match="canonical UUIDv5"):
        transition_generated_identity(
            record,
            GeneratedIdentityStatus.MERGED_TO_REGISTERED,
            registered_identity_id=str(uuid.uuid4()),
        )


def test_superseded_target_must_exist_in_registry() -> None:
    old = create_provisional_identity("cvi.track-cluster:v1", "cluster-old", 3)
    current = create_provisional_identity("cvi.track-cluster:v1", "cluster-current", 7)
    superseded = transition_generated_identity(
        old,
        GeneratedIdentityStatus.SUPERSEDED,
        superseded_by_generated_identity_id=current.generated_identity_id,
    )
    registry = GeneratedIdentityRegistry(records=(superseded, current))
    assert len(registry.records) == 2
    with pytest.raises(ValueError, match="target is absent"):
        GeneratedIdentityRegistry(records=(superseded,))


def test_registry_rejects_supersession_cycles() -> None:
    first = create_provisional_identity("cvi.track-cluster:v1", "first", 2)
    second = create_provisional_identity("cvi.track-cluster:v1", "second", 2)
    first_to_second = transition_generated_identity(
        first,
        GeneratedIdentityStatus.SUPERSEDED,
        superseded_by_generated_identity_id=second.generated_identity_id,
    )
    second_to_first = transition_generated_identity(
        second,
        GeneratedIdentityStatus.SUPERSEDED,
        superseded_by_generated_identity_id=first.generated_identity_id,
    )
    with pytest.raises(ValueError, match="contains a cycle"):
        GeneratedIdentityRegistry(records=(first_to_second, second_to_first))


def test_registry_rejects_generated_id_as_registered_target() -> None:
    provisional = create_provisional_identity("cvi.track-cluster:v1", "first", 2)
    other = create_provisional_identity("cvi.track-cluster:v1", "second", 2)
    confused = transition_generated_identity(
        provisional,
        GeneratedIdentityStatus.MERGED_TO_REGISTERED,
        registered_identity_id=other.generated_identity_id,
    )
    with pytest.raises(ValueError, match="generated namespace"):
        GeneratedIdentityRegistry(records=(confused, other))


def test_registry_round_trip_is_strict_and_canonical() -> None:
    records = (
        create_provisional_identity("cvi.track-cluster:v1", "cluster-a", 3),
        create_provisional_identity("cvi.track-cluster:v1", "cluster-b", 5),
    )
    registry = GeneratedIdentityRegistry(records=records)
    payload = registry.to_dict()
    assert GeneratedIdentityRegistry.from_dict(payload) == registry

    reversed_payload = deepcopy(payload)
    reversed_payload["generated_identities"].reverse()
    if reversed_payload != payload:
        with pytest.raises(ValueError, match="canonically ordered"):
            GeneratedIdentityRegistry.from_dict(reversed_payload)

    forged = deepcopy(payload)
    forged["generated_identities"][0]["generated_identity_id"] = str(
        uuid.uuid5(uuid.NAMESPACE_DNS, "forged")
    )
    with pytest.raises(ValueError, match="not deterministic"):
        GeneratedIdentityRegistry.from_dict(forged)


def test_generated_identity_id_rejects_unbound_inputs() -> None:
    token = compute_source_cluster_token("cluster")
    assert compute_generated_identity_id("cvi.track-cluster:v1", token)
    with pytest.raises(ValueError, match="generator_id"):
        compute_generated_identity_id("contains spaces", token)
