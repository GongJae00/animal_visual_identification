"""Identity-disjoint B2-FV protocol over the score-blind face overlay."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from foundation.provenance import content_sha256
from identity.face.face_eligibility import (
    validate_face_eligibility_overlay_bundle,
)
from data.full_segment.route_plan import (
    validate_full128_route_plan_bundle,
)

PROTOCOL_SCHEMA = "cvi.face_identity_protocol.v1"
IDENTITY_SCHEMA = "cvi.face_identity_protocol_identity.v1"
SAMPLE_SCHEMA = "cvi.face_identity_protocol_sample.v1"
CENSUS_SCHEMA = "cvi.face_identity_protocol_census.v1"
BUNDLE_SCHEMA = "cvi.face_identity_protocol_bundle.v1"
POLICY_SCHEMA = "cvi.face_identity_protocol_policy.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_DATASETS = ("dogfacenet224", "mpdd")
_ALLOCATABLE_ROLES = ("FIT", "DEV", "CAL")
_TARGET_PERCENTAGES = {"FIT": 70, "DEV": 15, "CAL": 15}
_INTERPRETATION = (
    "FACE_VISIBLE_RETROSPECTIVE_PROTOCOL;DOGFACE_PUBLISHER_TEST_AND_MPDD_ARE_"
    "EXPOSED_DIAGNOSTICS;NO_INDEPENDENT_FINAL_EVALUATION"
)


class FaceIdentityRole(StrEnum):
    FIT = "FIT"
    DEV = "DEV"
    CAL = "CAL"
    EXPOSED_DIAGNOSTIC = "EXPOSED_DIAGNOSTIC"


def build_face_identity_protocol(
    route_plan_bundle: object,
    face_overlay_bundle: object,
    *,
    protocol_name: str,
    historical_artifact_sha256s: Sequence[str],
) -> dict[str, Any]:
    """Build fixed identity roles while closing exact-source duplicate components."""

    if not isinstance(protocol_name, str) or not protocol_name.strip():
        raise ValueError("face identity protocol name must be non-empty text")
    history = tuple(sorted(set(historical_artifact_sha256s)))
    if not history or len(history) != len(historical_artifact_sha256s):
        raise ValueError("historical artifact hashes must be non-empty and unique")
    for value in history:
        _require_sha256(value, "historical artifact SHA-256")

    route = validate_full128_route_plan_bundle(route_plan_bundle, verify_files=False)
    overlay = validate_face_eligibility_overlay_bundle(face_overlay_bundle)
    if (
        overlay["overlay"]["source_route_plan_sha256"] != route["plan_sha256"]
        or overlay["overlay"]["source_route_plan_bundle_sha256"]
        != route["bundle_sha256"]
    ):
        raise ValueError("face overlay and route-plan bindings differ")
    route_by_sample = {row["sample_token"]: row for row in route["plan"]["records"]}
    overlay_records = overlay["overlay"]["records"]
    if set(route_by_sample) != {row["sample_token"] for row in overlay_records}:
        raise ValueError("face overlay is not observation-complete for the route plan")

    selected = [
        row
        for row in overlay_records
        if row["gallery_query_eligible"] and row["dataset_name"] in _IDENTITY_DATASETS
    ]
    samples_by_identity: defaultdict[
        str, list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for row in selected:
        source = route_by_sample[row["sample_token"]]
        if row["source_record_sha256"] != source["record_sha256"]:
            raise ValueError("face overlay source-record binding differs")
        identity = row["registered_identity_id"]
        if not isinstance(identity, str):
            raise TypeError("face identity protocol sample lacks registered identity")
        samples_by_identity[identity].append((row, source))

    block_by_identity = _duplicate_closed_blocks(samples_by_identity)
    exposed = {
        identity
        for identity, rows in samples_by_identity.items()
        if any(_is_exposed(overlay_row) for overlay_row, _ in rows)
    }
    exposed_blocks = {block_by_identity[identity] for identity in exposed}
    role_by_block = _allocate_blocks(
        {
            block
            for identity, block in block_by_identity.items()
            if block not in exposed_blocks
            and _identity_dataset(samples_by_identity[identity]) == "dogfacenet224"
        },
        block_by_identity=block_by_identity,
        protocol_name=protocol_name,
        evidence_root=content_sha256(
            {
                "route_plan_sha256": route["plan_sha256"],
                "face_overlay_sha256": overlay["overlay_sha256"],
                "historical_artifact_sha256s": list(history),
            }
        ),
    )

    role_by_identity: dict[str, FaceIdentityRole] = {}
    for identity in samples_by_identity:
        block = block_by_identity[identity]
        if block in exposed_blocks:
            role_by_identity[identity] = FaceIdentityRole.EXPOSED_DIAGNOSTIC
        else:
            role_by_identity[identity] = role_by_block[block]

    identities = tuple(
        _identity_record(
            identity,
            samples_by_identity[identity],
            role=role_by_identity[identity],
            allocation_block_sha256=block_by_identity[identity],
            dependency_promoted=(
                block_by_identity[identity] in exposed_blocks
                and identity not in exposed
            ),
        )
        for identity in sorted(samples_by_identity)
    )
    samples = tuple(
        _sample_record(
            overlay_row,
            source,
            role=role_by_identity[identity],
            allocation_block_sha256=block_by_identity[identity],
        )
        for identity in sorted(samples_by_identity)
        for overlay_row, source in sorted(
            samples_by_identity[identity], key=lambda pair: pair[0]["sample_token"]
        )
    )
    samples = tuple(sorted(samples, key=lambda row: row["sample_token"]))
    policy = _policy()
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_name": protocol_name,
        "source_route_plan_sha256": route["plan_sha256"],
        "source_route_plan_bundle_sha256": route["bundle_sha256"],
        "source_face_overlay_sha256": overlay["overlay_sha256"],
        "source_face_overlay_bundle_sha256": overlay["bundle_sha256"],
        "historical_artifact_sha256s": list(history),
        "policy": policy,
        "policy_sha256": content_sha256(policy),
        "identity_assignments": list(identities),
        "sample_assignments": list(samples),
        "score_inputs_used": False,
        "final_evaluation_permitted": False,
        "interpretation": _INTERPRETATION,
    }
    protocol = {**protocol, "protocol_sha256": content_sha256(protocol)}
    census = _build_census(identities, samples)
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"],
        "census": census,
        "census_sha256": content_sha256(census),
    }
    return {**payload, "bundle_sha256": content_sha256(payload)}


def validate_face_identity_protocol_bundle(value: object) -> dict[str, Any]:
    """Validate protocol hashes, role closure, and exact feasibility census."""

    expected = {
        "schema_version",
        "protocol",
        "protocol_sha256",
        "census",
        "census_sha256",
        "bundle_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face identity protocol bundle fields differ")
    bundle = dict(value)
    if bundle["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError("face identity protocol bundle schema differs")
    payload = {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    if bundle["bundle_sha256"] != content_sha256(payload):
        raise ValueError("face identity protocol bundle digest differs")
    protocol = bundle["protocol"]
    expected_protocol = {
        "schema_version",
        "protocol_name",
        "source_route_plan_sha256",
        "source_route_plan_bundle_sha256",
        "source_face_overlay_sha256",
        "source_face_overlay_bundle_sha256",
        "historical_artifact_sha256s",
        "policy",
        "policy_sha256",
        "identity_assignments",
        "sample_assignments",
        "score_inputs_used",
        "final_evaluation_permitted",
        "interpretation",
        "protocol_sha256",
    }
    if not isinstance(protocol, dict) or set(protocol) != expected_protocol:
        raise ValueError("face identity protocol fields differ")
    if protocol["schema_version"] != PROTOCOL_SCHEMA:
        raise ValueError("face identity protocol schema differs")
    protocol_payload = {
        key: item for key, item in protocol.items() if key != "protocol_sha256"
    }
    if (
        protocol["protocol_sha256"] != content_sha256(protocol_payload)
        or bundle["protocol_sha256"] != protocol["protocol_sha256"]
    ):
        raise ValueError("face identity protocol digest differs")
    if protocol["policy"] != _policy() or protocol["policy_sha256"] != content_sha256(
        protocol["policy"]
    ):
        raise ValueError("face identity protocol policy differs")
    if (
        protocol["score_inputs_used"] is not False
        or protocol["final_evaluation_permitted"] is not False
        or protocol["interpretation"] != _INTERPRETATION
    ):
        raise ValueError("face identity protocol interpretation differs")
    identities = tuple(
        _validate_identity(row) for row in protocol["identity_assignments"]
    )
    samples = tuple(_validate_sample(row) for row in protocol["sample_assignments"])
    _validate_closure(identities, samples)
    census = _build_census(identities, samples)
    if bundle["census"] != census or bundle["census_sha256"] != content_sha256(census):
        raise ValueError("face identity protocol census differs")
    return bundle


def _identity_record(
    identity: str,
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    role: FaceIdentityRole,
    allocation_block_sha256: str,
    dependency_promoted: bool,
) -> dict[str, Any]:
    dataset = _identity_dataset(rows)
    publisher_splits = sorted({overlay["publisher_split"] for overlay, _ in rows})
    payload = {
        "schema_version": IDENTITY_SCHEMA,
        "registered_identity_id": identity,
        "dataset_name": dataset,
        "role": role.value,
        "allocation_block_sha256": allocation_block_sha256,
        "sample_count": len(rows),
        "publisher_splits": publisher_splits,
        "historically_exposed": role is FaceIdentityRole.EXPOSED_DIAGNOSTIC,
        "dependency_promoted_to_exposed": dependency_promoted,
    }
    return {**payload, "record_sha256": content_sha256(payload)}


def _sample_record(
    overlay: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    role: FaceIdentityRole,
    allocation_block_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": SAMPLE_SCHEMA,
        "sample_token": overlay["sample_token"],
        "registered_identity_id": overlay["registered_identity_id"],
        "dataset_name": overlay["dataset_name"],
        "publisher_split": overlay["publisher_split"],
        "role": role.value,
        "gradient_eligible": role is FaceIdentityRole.FIT,
        "duplicate_component": source["duplicate_component"],
        "allocation_block_sha256": allocation_block_sha256,
        "capture_group_id": source["capture_metadata"]["capture_group_id"],
        "source_record_sha256": source["record_sha256"],
        "face_record_sha256": overlay["record_sha256"],
    }
    return {**payload, "record_sha256": content_sha256(payload)}


def _duplicate_closed_blocks(
    samples_by_identity: Mapping[
        str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ],
) -> dict[str, str]:
    parent = {identity: identity for identity in samples_by_identity}

    def root(identity: str) -> str:
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def union(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    identities_by_component: defaultdict[str, set[str]] = defaultdict(set)
    for identity, rows in samples_by_identity.items():
        for _, source in rows:
            identities_by_component[source["duplicate_component"]].add(identity)
    for identities in identities_by_component.values():
        ordered = sorted(identities)
        for identity in ordered[1:]:
            union(ordered[0], identity)
    members: defaultdict[str, list[str]] = defaultdict(list)
    for identity in sorted(parent):
        members[root(identity)].append(identity)
    return {
        identity: content_sha256(
            {
                "schema_version": "cvi.face_identity_allocation_block.v1",
                "registered_identity_ids": block_members,
            }
        )
        for block_members in members.values()
        for identity in block_members
    }


def _allocate_blocks(
    candidate_blocks: set[str],
    *,
    block_by_identity: Mapping[str, str],
    protocol_name: str,
    evidence_root: str,
) -> dict[str, FaceIdentityRole]:
    sizes = Counter(
        block for block in block_by_identity.values() if block in candidate_blocks
    )
    total = sum(sizes.values())
    if total < len(_ALLOCATABLE_ROLES):
        raise ValueError("face identity protocol lacks allocatable DogFace identities")
    targets = {
        role: total * percentage / 100
        for role, percentage in _TARGET_PERCENTAGES.items()
    }

    def rank(block: str) -> str:
        return hashlib.sha256(
            f"CVI_B2_FV_BLOCK_ORDER_V1\0{protocol_name}\0{evidence_root}\0{block}".encode()
        ).hexdigest()

    counts: Counter[str] = Counter()
    result: dict[str, FaceIdentityRole] = {}
    for block in sorted(sizes, key=lambda item: (-sizes[item], rank(item), item)):
        role = min(
            _ALLOCATABLE_ROLES,
            key=lambda candidate: (
                abs(counts[candidate] + sizes[block] - targets[candidate])
                - abs(counts[candidate] - targets[candidate]),
                counts[candidate] / targets[candidate],
                _ALLOCATABLE_ROLES.index(candidate),
            ),
        )
        result[block] = FaceIdentityRole(role)
        counts[role] += sizes[block]
    if any(counts[role] == 0 for role in _ALLOCATABLE_ROLES):
        raise ValueError("face identity protocol produced an empty allocatable role")
    return result


def _build_census(
    identities: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    role_counts = Counter(row["role"] for row in identities)
    dataset_role_counts: dict[str, dict[str, int]] = {}
    for dataset in _IDENTITY_DATASETS:
        counts = Counter(
            row["role"] for row in identities if row["dataset_name"] == dataset
        )
        dataset_role_counts[dataset] = {
            role.value: counts[role.value] for role in FaceIdentityRole
        }
    samples_by_identity: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_identity[sample["registered_identity_id"]].append(sample)
    feasibility: dict[str, dict[str, int]] = {}
    for k in (1, 3, 5):
        by_role: Counter[str] = Counter()
        for identity in identities:
            rows = samples_by_identity[identity["registered_identity_id"]]
            if identity["dataset_name"] == "mpdd":
                splits = Counter(row["publisher_split"] for row in rows)
                feasible = splits["gallery"] >= k and splits["query"] >= 1
            else:
                feasible = len(rows) >= k + 1
            if feasible:
                by_role[identity["role"]] += 1
        feasibility[f"K{k}"] = {
            role.value: by_role[role.value] for role in FaceIdentityRole
        }
    return {
        "schema_version": CENSUS_SCHEMA,
        "identity_count": len(identities),
        "sample_count": len(samples),
        "identity_role_counts": {
            role.value: role_counts[role.value] for role in FaceIdentityRole
        },
        "dataset_identity_role_counts": dataset_role_counts,
        "k_feasible_identity_counts": feasibility,
        "gradient_sample_count": sum(row["gradient_eligible"] for row in samples),
        "score_inputs_used": False,
        "final_evaluation_permitted": False,
    }


def _validate_closure(
    identities: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]
) -> None:
    if [row["registered_identity_id"] for row in identities] != sorted(
        row["registered_identity_id"] for row in identities
    ):
        raise ValueError("face protocol identities must be canonically sorted")
    if [row["sample_token"] for row in samples] != sorted(
        row["sample_token"] for row in samples
    ):
        raise ValueError("face protocol samples must be canonically sorted")
    identity_by_id = {row["registered_identity_id"]: row for row in identities}
    if len(identity_by_id) != len(identities):
        raise ValueError("face protocol identities repeat")
    role_by_component: dict[str, str] = {}
    for sample in samples:
        identity = identity_by_id.get(sample["registered_identity_id"])
        if (
            identity is None
            or identity["role"] != sample["role"]
            or identity["allocation_block_sha256"] != sample["allocation_block_sha256"]
        ):
            raise ValueError("face protocol sample and identity assignments differ")
        prior = role_by_component.setdefault(
            sample["duplicate_component"], sample["role"]
        )
        if prior != sample["role"]:
            raise ValueError("exact duplicate component crosses face protocol roles")


def _validate_identity(value: object) -> dict[str, Any]:
    expected = {
        "schema_version",
        "registered_identity_id",
        "dataset_name",
        "role",
        "allocation_block_sha256",
        "sample_count",
        "publisher_splits",
        "historically_exposed",
        "dependency_promoted_to_exposed",
        "record_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face protocol identity fields differ")
    row = dict(value)
    if (
        row["schema_version"] != IDENTITY_SCHEMA
        or row["dataset_name"] not in _IDENTITY_DATASETS
    ):
        raise ValueError("face protocol identity schema or dataset differs")
    role = FaceIdentityRole(row["role"])
    if row["historically_exposed"] is not (role is FaceIdentityRole.EXPOSED_DIAGNOSTIC):
        raise ValueError("face protocol historical exposure differs")
    _validate_record_digest(row)
    return row


def _validate_sample(value: object) -> dict[str, Any]:
    expected = {
        "schema_version",
        "sample_token",
        "registered_identity_id",
        "dataset_name",
        "publisher_split",
        "role",
        "gradient_eligible",
        "duplicate_component",
        "allocation_block_sha256",
        "capture_group_id",
        "source_record_sha256",
        "face_record_sha256",
        "record_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face protocol sample fields differ")
    row = dict(value)
    if (
        row["schema_version"] != SAMPLE_SCHEMA
        or row["dataset_name"] not in _IDENTITY_DATASETS
    ):
        raise ValueError("face protocol sample schema or dataset differs")
    role = FaceIdentityRole(row["role"])
    if row["gradient_eligible"] is not (role is FaceIdentityRole.FIT):
        raise ValueError("face protocol gradient eligibility differs")
    for field in (
        "sample_token",
        "duplicate_component",
        "allocation_block_sha256",
        "source_record_sha256",
        "face_record_sha256",
    ):
        _require_sha256(row[field], field)
    _validate_record_digest(row)
    return row


def _validate_record_digest(row: Mapping[str, Any]) -> None:
    _require_sha256(row["record_sha256"], "record_sha256")
    payload = {key: item for key, item in row.items() if key != "record_sha256"}
    if row["record_sha256"] != content_sha256(payload):
        raise ValueError("face protocol record digest differs")


def _identity_dataset(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> str:
    datasets = {overlay["dataset_name"] for overlay, _ in rows}
    if len(datasets) != 1:
        raise ValueError("registered face identity crosses datasets")
    return next(iter(datasets))


def _is_exposed(overlay: Mapping[str, Any]) -> bool:
    return (
        overlay["face_protocol_role"] == "EXPOSED_DIAGNOSTIC"
        or overlay["dataset_name"] == "mpdd"
    )


def _policy() -> dict[str, Any]:
    return {
        "schema_version": POLICY_SCHEMA,
        "allocatable_dataset": "dogfacenet224",
        "allocatable_publisher_split": "train",
        "target_identity_percentages": dict(_TARGET_PERCENTAGES),
        "exposed_diagnostic_datasets": ["dogfacenet224:test", "mpdd:all"],
        "dependency_closure": "EXACT_SOURCE_SHA256_CONNECTED_COMPONENT",
        "k_values": [1, 3, 5],
        "score_inputs_used": False,
        "final_evaluation_permitted": False,
    }


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


__all__ = [
    "FaceIdentityRole",
    "build_face_identity_protocol",
    "validate_face_identity_protocol_bundle",
]
