"""Governance-v2 face protocol over authenticated public tokens and dependencies."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from shared.foundation.provenance import content_sha256
from evaluation.splits.face.face_eligibility import (
    validate_face_eligibility_overlay_bundle,
)
from evaluation.splits.face.face_exposure_history import (
    validate_face_exposure_history_bundle,
)
from evaluation.splits.face.face_public_source_binding import (
    validate_face_public_source_binding_bundle,
)
from evaluation.splits.protected_public_split import (
    EvidenceRelation,
    FrozenPublicSplitEvidenceGraph,
)
from evaluation.splits.role_exposure import ExposureStage, RoleExposureLedger
from data.full_segment.route_plan import (
    validate_full128_route_plan_bundle,
)

PROTOCOL_SCHEMA = "evaluation.face_identity_protocol.v2"
IDENTITY_SCHEMA = "evaluation.face_identity_protocol_identity.v2"
SAMPLE_SCHEMA = "evaluation.face_identity_protocol_sample.v2"
CENSUS_SCHEMA = "evaluation.face_identity_protocol_census.v2"
BUNDLE_SCHEMA = "evaluation.face_identity_protocol_bundle.v2"
POLICY_SCHEMA = "evaluation.face_identity_protocol_policy.v2"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOCATABLE_ROLES = ("FIT", "DEV", "CAL")
_TARGET_PERCENTAGES = {"FIT": 70, "DEV": 15, "CAL": 15}
_STAGE_RANK = {stage: index for index, stage in enumerate(ExposureStage)}
_ROLE_STAGE = {
    "FIT": ExposureStage.MODEL_TRAINING_USED,
    "DEV": ExposureStage.MODEL_SELECTION_SCORED,
    "CAL": ExposureStage.CALIBRATION_SCORED,
    "EXPOSED_DIAGNOSTIC": ExposureStage.FINAL_TEST_SCORED,
}
_CLOSURE_RELATIONS = frozenset(
    {
        EvidenceRelation.EXACT_CONFIRMED,
        EvidenceRelation.GEOMETRIC_CONFIRMED,
        EvidenceRelation.DEPENDENCY,
        EvidenceRelation.REVIEW_CONFIRMED,
    }
)
_INTERPRETATION = (
    "RETROSPECTIVE_PUBLIC_TOKEN_BOUND_FACE_PROTOCOL;NO_DECLARED_EXPOSURE_IS_NOT_"
    "PROOF_OF_CLEAN_HISTORY;NO_INDEPENDENT_FINAL_EVALUATION"
)


class FaceIdentityRoleV2(StrEnum):
    FIT = "FIT"
    DEV = "DEV"
    CAL = "CAL"
    EXPOSED_DIAGNOSTIC = "EXPOSED_DIAGNOSTIC"
    EXCLUDED_UNSAFE_COMPONENT = "EXCLUDED_UNSAFE_COMPONENT"


def build_face_identity_protocol_v2(
    route_plan_bundle: object,
    face_overlay_bundle: object,
    token_bridge_bundle: object,
    exposure_history_bundle: object,
    joint_filter_evidence_graph: object,
    joint_filter_receipt: object,
    *,
    protocol_name: str,
) -> dict[str, Any]:
    """Allocate score-blind roles after exposure and dependency closure."""

    if not isinstance(protocol_name, str) or not protocol_name.strip():
        raise ValueError("face protocol v2 name must be non-empty text")
    route = validate_full128_route_plan_bundle(route_plan_bundle, verify_files=False)
    overlay = validate_face_eligibility_overlay_bundle(face_overlay_bundle)
    bridge = validate_face_public_source_binding_bundle(token_bridge_bundle)
    exposure = validate_face_exposure_history_bundle(exposure_history_bundle)
    graph = _graph(joint_filter_evidence_graph)
    receipt = _validate_joint_filter_receipt(
        joint_filter_receipt,
        graph=graph,
        source_bundle_sha256=bridge["binding"]["public_source_bundle_sha256"],
    )
    _validate_input_bindings(route, overlay, bridge, exposure)
    if exposure["history"]["role_allocation_permitted"] is not True:
        raise ValueError("unresolved face exposure history blocks role allocation")
    if exposure["history"]["ledger"] is None:
        raise ValueError("face exposure history lacks an exact role exposure ledger")

    bridge_by_route = {
        row["route_sample_token"]: row for row in bridge["binding"]["records"]
    }
    overlay_by_route = {
        row["sample_token"]: row
        for row in overlay["overlay"]["records"]
        if row["gallery_query_eligible"]
        and row["dataset_name"] in {"dogfacenet224", "mpdd"}
    }
    if set(bridge_by_route) != set(overlay_by_route):
        raise ValueError("face token bridge is not observation-complete for v2")

    component_by_sample, members_by_component = _graph_components(graph)
    identity_by_public = {
        row["public_sample_token"]: row["public_identity_token"]
        for row in bridge_by_route.values()
    }
    unsafe_components = _unsafe_components(
        members_by_component, identity_by_public=identity_by_public
    )
    exposure_ledger = RoleExposureLedger.from_dict(exposure["history"]["ledger"])
    history_by_identity = _history_by_identity(exposure_ledger)

    rows_by_identity: defaultdict[
        str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = defaultdict(list)
    for route_token, public in bridge_by_route.items():
        face = overlay_by_route[route_token]
        if (
            public["face_record_sha256"] != face["record_sha256"]
            or public["registered_identity_id"] != face["registered_identity_id"]
        ):
            raise ValueError("face protocol v2 bridge and overlay records differ")
        rows_by_identity[public["public_identity_token"]].append((public, face))

    identity_state: dict[str, dict[str, Any]] = {}
    allocatable: set[str] = set()
    for identity, rows in rows_by_identity.items():
        safe = [
            pair
            for pair in rows
            if _component_token(pair[0]["public_sample_token"], component_by_sample)
            not in unsafe_components
        ]
        datasets = {public["dataset_name"] for public, _ in rows}
        if len(datasets) != 1:
            raise ValueError("face protocol v2 public identity crosses datasets")
        dataset = next(iter(datasets))
        history = history_by_identity.get(identity)
        publisher_exposed = dataset == "mpdd" or any(
            public["publisher_split"] == "test" for public, _ in rows
        )
        if not safe:
            fixed_role = FaceIdentityRoleV2.EXCLUDED_UNSAFE_COMPONENT
        elif publisher_exposed or history is ExposureStage.FINAL_TEST_SCORED:
            fixed_role = FaceIdentityRoleV2.EXPOSED_DIAGNOSTIC
        else:
            fixed_role = None
            allocatable.add(identity)
        identity_state[identity] = {
            "rows": rows,
            "safe": safe,
            "dataset": dataset,
            "history": history,
            "fixed_role": fixed_role,
        }
    allocated = _allocate_identities(
        allocatable,
        history_by_identity=history_by_identity,
        protocol_name=protocol_name,
    )

    identity_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    for identity in sorted(rows_by_identity):
        state = identity_state[identity]
        role = state["fixed_role"] or allocated[identity]
        rank_sha256 = _allocation_rank(protocol_name, identity)
        identity_records.append(
            _identity_record(
                identity,
                state,
                role=role,
                allocation_rank_sha256=rank_sha256,
            )
        )
        for public, face in state["rows"]:
            component = _component_token(
                public["public_sample_token"], component_by_sample
            )
            unsafe = component in unsafe_components
            sample_records.append(
                _sample_record(
                    public,
                    face,
                    role=(
                        FaceIdentityRoleV2.EXCLUDED_UNSAFE_COMPONENT if unsafe else role
                    ),
                    identity_role=role,
                    dependency_component=component,
                    cross_identity_unsafe=unsafe,
                )
            )
    sample_records.sort(key=lambda row: row["public_sample_token"])
    policy = _policy()
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_name": protocol_name,
        "source_route_plan_sha256": route["plan_sha256"],
        "source_route_plan_bundle_sha256": route["bundle_sha256"],
        "source_face_overlay_sha256": overlay["overlay_sha256"],
        "source_face_overlay_bundle_sha256": overlay["bundle_sha256"],
        "source_token_bridge_sha256": bridge["binding_sha256"],
        "source_token_bridge_bundle_sha256": bridge["bundle_sha256"],
        "source_exposure_history_sha256": exposure["history_sha256"],
        "source_exposure_history_bundle_sha256": exposure["bundle_sha256"],
        "public_source_bundle_sha256": bridge["binding"]["public_source_bundle_sha256"],
        "joint_filter_graph_sha256": graph.graph_sha256,
        "joint_filter_receipt_sha256": receipt["receipt_sha256"],
        "policy": policy,
        "policy_sha256": content_sha256(policy),
        "identity_assignments": identity_records,
        "sample_assignments": sample_records,
        "score_inputs_used": False,
        "score_bearing_bytes_used_for_role_allocation": False,
        "clean_role_claims_permitted": False,
        "final_evaluation_permitted": False,
        "interpretation": _INTERPRETATION,
    }
    protocol = {**protocol, "protocol_sha256": content_sha256(protocol)}
    census = _build_census(identity_records, sample_records)
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"],
        "census": census,
        "census_sha256": content_sha256(census),
    }
    return {**payload, "bundle_sha256": content_sha256(payload)}


def validate_face_identity_protocol_v2_bundle(value: object) -> dict[str, Any]:
    """Validate all hashes, role constraints, exclusions, and component closure."""

    expected = {
        "schema_version",
        "protocol",
        "protocol_sha256",
        "census",
        "census_sha256",
        "bundle_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face protocol v2 bundle fields differ")
    bundle = dict(value)
    if bundle["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError("face protocol v2 bundle schema differs")
    payload = {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    if bundle["bundle_sha256"] != content_sha256(payload):
        raise ValueError("face protocol v2 bundle digest differs")
    protocol = bundle["protocol"]
    expected_protocol = {
        "schema_version",
        "protocol_name",
        "source_route_plan_sha256",
        "source_route_plan_bundle_sha256",
        "source_face_overlay_sha256",
        "source_face_overlay_bundle_sha256",
        "source_token_bridge_sha256",
        "source_token_bridge_bundle_sha256",
        "source_exposure_history_sha256",
        "source_exposure_history_bundle_sha256",
        "public_source_bundle_sha256",
        "joint_filter_graph_sha256",
        "joint_filter_receipt_sha256",
        "policy",
        "policy_sha256",
        "identity_assignments",
        "sample_assignments",
        "score_inputs_used",
        "score_bearing_bytes_used_for_role_allocation",
        "clean_role_claims_permitted",
        "final_evaluation_permitted",
        "interpretation",
        "protocol_sha256",
    }
    if not isinstance(protocol, dict) or set(protocol) != expected_protocol:
        raise ValueError("face protocol v2 fields differ")
    if (
        protocol["schema_version"] != PROTOCOL_SCHEMA
        or protocol["policy"] != _policy()
        or protocol["policy_sha256"] != content_sha256(protocol["policy"])
        or protocol["score_inputs_used"] is not False
        or protocol["score_bearing_bytes_used_for_role_allocation"] is not False
        or protocol["clean_role_claims_permitted"] is not False
        or protocol["final_evaluation_permitted"] is not False
        or protocol["interpretation"] != _INTERPRETATION
    ):
        raise ValueError("face protocol v2 policy or interpretation differs")
    for field in (
        "source_route_plan_sha256",
        "source_route_plan_bundle_sha256",
        "source_face_overlay_sha256",
        "source_face_overlay_bundle_sha256",
        "source_token_bridge_sha256",
        "source_token_bridge_bundle_sha256",
        "source_exposure_history_sha256",
        "source_exposure_history_bundle_sha256",
        "public_source_bundle_sha256",
        "joint_filter_graph_sha256",
        "joint_filter_receipt_sha256",
        "policy_sha256",
        "protocol_sha256",
    ):
        _require_sha256(protocol[field], field)
    protocol_payload = {
        key: item for key, item in protocol.items() if key != "protocol_sha256"
    }
    if (
        protocol["protocol_sha256"] != content_sha256(protocol_payload)
        or bundle["protocol_sha256"] != protocol["protocol_sha256"]
    ):
        raise ValueError("face protocol v2 digest differs")
    identities = tuple(
        _validate_identity(row, protocol["protocol_name"])
        for row in protocol["identity_assignments"]
    )
    samples = tuple(_validate_sample(row) for row in protocol["sample_assignments"])
    _validate_closure(identities, samples, protocol_name=protocol["protocol_name"])
    census = _build_census(identities, samples)
    if bundle["census"] != census or bundle["census_sha256"] != content_sha256(census):
        raise ValueError("face protocol v2 census differs")
    return bundle


def _validate_input_bindings(
    route: Mapping[str, Any],
    overlay: Mapping[str, Any],
    bridge: Mapping[str, Any],
    exposure: Mapping[str, Any],
) -> None:
    binding = bridge["binding"]
    history = exposure["history"]
    if (
        binding["source_route_plan_sha256"] != route["plan_sha256"]
        or binding["source_route_plan_bundle_sha256"] != route["bundle_sha256"]
        or binding["source_face_overlay_sha256"] != overlay["overlay_sha256"]
        or binding["source_face_overlay_bundle_sha256"] != overlay["bundle_sha256"]
        or history["source_token_bridge_sha256"] != bridge["binding_sha256"]
        or history["source_token_bridge_bundle_sha256"] != bridge["bundle_sha256"]
        or history["public_source_bundle_sha256"]
        != binding["public_source_bundle_sha256"]
    ):
        raise ValueError("face protocol v2 source bindings differ")


def _graph(value: object) -> FrozenPublicSplitEvidenceGraph:
    if not isinstance(value, dict):
        raise TypeError("joint-filter evidence graph must be an object")
    return FrozenPublicSplitEvidenceGraph.from_dict(value)


def _validate_joint_filter_receipt(
    value: object,
    *,
    graph: FrozenPublicSplitEvidenceGraph,
    source_bundle_sha256: str,
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "source_bundle_sha256",
        "adjudication_ledger_sha256",
        "candidate_set_sha256",
        "candidate_count",
        "outcome_counts",
        "unresolved_candidate_count",
        "unbound_candidate_count",
        "adjudication_mode",
        "graph_sha256",
        "edge_count",
        "tool_provenance",
        "tool_provenance_sha256",
        "decision",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("joint-filter admission receipt fields differ")
    receipt = dict(value)
    payload = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if (
        receipt["schema_version"]
        != "evaluation.public_split_evidence_graph_assembly_receipt.v1"
        or receipt["receipt_sha256"] != content_sha256(payload)
        or receipt["source_bundle_sha256"] != source_bundle_sha256
        or receipt["graph_sha256"] != graph.graph_sha256
        or receipt["edge_count"] != len(graph.edges)
        or receipt["unresolved_candidate_count"] != 0
        or receipt["unbound_candidate_count"] != 0
        or receipt["decision"] != "PASS_COMPLETE_ADJUDICATION_GRAPH_PROMOTION"
        or receipt["tool_provenance_sha256"]
        != content_sha256(receipt["tool_provenance"])
    ):
        raise ValueError("joint-filter evidence graph is not admitted")
    counts = (
        "candidate_count",
        "unresolved_candidate_count",
        "unbound_candidate_count",
        "edge_count",
    )
    if any(
        isinstance(receipt[field], bool)
        or not isinstance(receipt[field], int)
        or receipt[field] < 0
        for field in counts
    ):
        raise ValueError("joint-filter admission receipt counts differ")
    outcomes = receipt["outcome_counts"]
    if (
        not isinstance(outcomes, dict)
        or not outcomes
        or any(
            not isinstance(key, str)
            or not key
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for key, count in outcomes.items()
        )
        or sum(outcomes.values()) != receipt["candidate_count"]
        or not isinstance(receipt["adjudication_mode"], str)
        or not receipt["adjudication_mode"]
        or not isinstance(receipt["tool_provenance"], dict)
    ):
        raise ValueError("joint-filter admission receipt census differs")
    for field in (
        "source_bundle_sha256",
        "adjudication_ledger_sha256",
        "candidate_set_sha256",
        "graph_sha256",
        "tool_provenance_sha256",
        "receipt_sha256",
    ):
        _require_sha256(receipt[field], field)
    if any(edge.relation is EvidenceRelation.REVIEW_UNRESOLVED for edge in graph.edges):
        raise ValueError("joint-filter evidence graph contains unresolved edges")
    return receipt


def _graph_components(
    graph: FrozenPublicSplitEvidenceGraph,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    parent: dict[str, str] = {}

    def root(token: str) -> str:
        parent.setdefault(token, token)
        while parent[token] != token:
            parent[token] = parent[parent[token]]
            token = parent[token]
        return token

    def union(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for edge in graph.edges:
        if edge.relation in _CLOSURE_RELATIONS:
            union(edge.left_sample_token, edge.right_sample_token)
    raw_members: defaultdict[str, list[str]] = defaultdict(list)
    for token in sorted(parent):
        raw_members[root(token)].append(token)
    component_by_sample: dict[str, str] = {}
    members_by_component: dict[str, tuple[str, ...]] = {}
    for members in raw_members.values():
        ordered = tuple(sorted(members))
        component = content_sha256(
            {
                "schema_version": "evaluation.face_dependency_component.v2",
                "public_sample_tokens": list(ordered),
            }
        )
        members_by_component[component] = ordered
        component_by_sample.update((token, component) for token in ordered)
    return component_by_sample, members_by_component


def _component_token(sample: str, components: Mapping[str, str]) -> str:
    return components.get(
        sample,
        content_sha256(
            {
                "schema_version": "evaluation.face_dependency_component.v2",
                "public_sample_tokens": [sample],
            }
        ),
    )


def _unsafe_components(
    members_by_component: Mapping[str, Sequence[str]],
    *,
    identity_by_public: Mapping[str, str],
) -> set[str]:
    result: set[str] = set()
    for component, members in members_by_component.items():
        identities = {identity_by_public.get(sample) for sample in members}
        if None in identities or len(identities) > 1:
            result.add(component)
    return result


def _history_by_identity(
    ledger: RoleExposureLedger,
) -> dict[str, ExposureStage]:
    result: dict[str, ExposureStage] = {}
    for row in ledger.records:
        prior = result.get(row.identity_token)
        if (
            prior is None
            or _STAGE_RANK[row.maximum_historical_stage] > _STAGE_RANK[prior]
        ):
            result[row.identity_token] = row.maximum_historical_stage
    return result


def _allocate_identities(
    identities: set[str],
    *,
    history_by_identity: Mapping[str, ExposureStage],
    protocol_name: str,
) -> dict[str, FaceIdentityRoleV2]:
    if len(identities) < len(_ALLOCATABLE_ROLES):
        raise ValueError("face protocol v2 lacks allocatable DogFace identities")
    targets = {
        role: len(identities) * percentage / 100
        for role, percentage in _TARGET_PERCENTAGES.items()
    }
    counts: Counter[str] = Counter()
    result: dict[str, FaceIdentityRoleV2] = {}
    for identity in sorted(
        identities, key=lambda value: (_allocation_rank(protocol_name, value), value)
    ):
        history = history_by_identity.get(identity)
        allowed = [
            role
            for role in _ALLOCATABLE_ROLES
            if history is None or _STAGE_RANK[_ROLE_STAGE[role]] >= _STAGE_RANK[history]
        ]
        if not allowed:
            raise ValueError("face exposure history has no compatible allocatable role")
        role = min(
            allowed,
            key=lambda candidate: (
                abs(counts[candidate] + 1 - targets[candidate])
                - abs(counts[candidate] - targets[candidate]),
                counts[candidate] / targets[candidate],
                _ALLOCATABLE_ROLES.index(candidate),
            ),
        )
        result[identity] = FaceIdentityRoleV2(role)
        counts[role] += 1
    if any(counts[role] == 0 for role in _ALLOCATABLE_ROLES):
        raise ValueError("face protocol v2 produced an empty allocatable role")
    return result


def _allocation_rank(protocol_name: str, public_identity_token: str) -> str:
    return hashlib.sha256(
        (
            "FACE_PROTOCOL_V2_PUBLIC_IDENTITY_ORDER\0"
            f"{protocol_name}\0{public_identity_token}"
        ).encode()
    ).hexdigest()


def _identity_record(
    identity: str,
    state: Mapping[str, Any],
    *,
    role: FaceIdentityRoleV2,
    allocation_rank_sha256: str,
) -> dict[str, Any]:
    rows = state["rows"]
    history = state["history"]
    payload = {
        "schema_version": IDENTITY_SCHEMA,
        "public_identity_token": identity,
        "registered_identity_id": rows[0][0]["registered_identity_id"],
        "dataset_name": state["dataset"],
        "role": role.value,
        "allocation_rank_sha256": allocation_rank_sha256,
        "sample_count": len(rows),
        "included_sample_count": len(state["safe"]),
        "excluded_unsafe_sample_count": len(rows) - len(state["safe"]),
        "historical_maximum_stage": None if history is None else history.value,
        "no_declared_exposure_is_clean_claim": False,
    }
    return {**payload, "record_sha256": content_sha256(payload)}


def _sample_record(
    public: Mapping[str, Any],
    face: Mapping[str, Any],
    *,
    role: FaceIdentityRoleV2,
    identity_role: FaceIdentityRoleV2,
    dependency_component: str,
    cross_identity_unsafe: bool,
) -> dict[str, Any]:
    payload = {
        "schema_version": SAMPLE_SCHEMA,
        "route_sample_token": public["route_sample_token"],
        "public_sample_token": public["public_sample_token"],
        "public_identity_token": public["public_identity_token"],
        "registered_identity_id": public["registered_identity_id"],
        "dataset_name": public["dataset_name"],
        "publisher_split": public["publisher_split"],
        "role": role.value,
        "identity_role": identity_role.value,
        "gradient_eligible": role is FaceIdentityRoleV2.FIT,
        "dependency_component_sha256": dependency_component,
        "cross_identity_unsafe": cross_identity_unsafe,
        "token_bridge_record_sha256": public["record_sha256"],
        "face_record_sha256": face["record_sha256"],
    }
    return {**payload, "record_sha256": content_sha256(payload)}


def _build_census(
    identities: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    identity_roles = Counter(row["role"] for row in identities)
    sample_roles = Counter(row["role"] for row in samples)
    return {
        "schema_version": CENSUS_SCHEMA,
        "identity_count": len(identities),
        "sample_count": len(samples),
        "identity_role_counts": {
            role.value: identity_roles[role.value] for role in FaceIdentityRoleV2
        },
        "sample_role_counts": {
            role.value: sample_roles[role.value] for role in FaceIdentityRoleV2
        },
        "cross_identity_unsafe_sample_count": sum(
            row["cross_identity_unsafe"] for row in samples
        ),
        "gradient_sample_count": sum(row["gradient_eligible"] for row in samples),
        "score_inputs_used": False,
        "clean_role_claims_permitted": False,
        "final_evaluation_permitted": False,
    }


def _validate_identity(value: object, protocol_name: str) -> dict[str, Any]:
    expected = {
        "schema_version",
        "public_identity_token",
        "registered_identity_id",
        "dataset_name",
        "role",
        "allocation_rank_sha256",
        "sample_count",
        "included_sample_count",
        "excluded_unsafe_sample_count",
        "historical_maximum_stage",
        "no_declared_exposure_is_clean_claim",
        "record_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face protocol v2 identity fields differ")
    row = dict(value)
    role = FaceIdentityRoleV2(row["role"])
    if (
        row["schema_version"] != IDENTITY_SCHEMA
        or row["dataset_name"] not in {"dogfacenet224", "mpdd"}
        or row["no_declared_exposure_is_clean_claim"] is not False
        or row["allocation_rank_sha256"]
        != _allocation_rank(protocol_name, row["public_identity_token"])
        or row["sample_count"]
        != row["included_sample_count"] + row["excluded_unsafe_sample_count"]
        or row["sample_count"] <= 0
        or row["included_sample_count"] < 0
        or row["excluded_unsafe_sample_count"] < 0
    ):
        raise ValueError("face protocol v2 identity policy differs")
    if row["historical_maximum_stage"] is not None:
        stage = ExposureStage(row["historical_maximum_stage"])
        if (
            role.value in _ROLE_STAGE
            and _STAGE_RANK[_ROLE_STAGE[role.value]] < (_STAGE_RANK[stage])
        ):
            raise ValueError("face protocol v2 role regresses historical exposure")
    _validate_record_digest(row)
    return row


def _validate_sample(value: object) -> dict[str, Any]:
    expected = {
        "schema_version",
        "route_sample_token",
        "public_sample_token",
        "public_identity_token",
        "registered_identity_id",
        "dataset_name",
        "publisher_split",
        "role",
        "identity_role",
        "gradient_eligible",
        "dependency_component_sha256",
        "cross_identity_unsafe",
        "token_bridge_record_sha256",
        "face_record_sha256",
        "record_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face protocol v2 sample fields differ")
    row = dict(value)
    role = FaceIdentityRoleV2(row["role"])
    FaceIdentityRoleV2(row["identity_role"])
    if (
        row["schema_version"] != SAMPLE_SCHEMA
        or row["dataset_name"] not in {"dogfacenet224", "mpdd"}
        or not isinstance(row["cross_identity_unsafe"], bool)
        or row["gradient_eligible"] is not (role is FaceIdentityRoleV2.FIT)
        or row["cross_identity_unsafe"]
        is not (role is FaceIdentityRoleV2.EXCLUDED_UNSAFE_COMPONENT)
    ):
        raise ValueError("face protocol v2 sample policy differs")
    for field in (
        "route_sample_token",
        "public_sample_token",
        "public_identity_token",
        "dependency_component_sha256",
        "token_bridge_record_sha256",
        "face_record_sha256",
    ):
        _require_sha256(row[field], field)
    _validate_record_digest(row)
    return row


def _validate_closure(
    identities: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    *,
    protocol_name: str,
) -> None:
    identity_tokens = [row["public_identity_token"] for row in identities]
    sample_tokens = [row["public_sample_token"] for row in samples]
    if identity_tokens != sorted(identity_tokens) or len(identity_tokens) != len(
        set(identity_tokens)
    ):
        raise ValueError("face protocol v2 identities must be sorted and unique")
    if sample_tokens != sorted(sample_tokens) or len(sample_tokens) != len(
        set(sample_tokens)
    ):
        raise ValueError("face protocol v2 samples must be sorted and unique")
    identity_by_token = {row["public_identity_token"]: row for row in identities}
    count_by_identity: Counter[str] = Counter()
    included_by_identity: Counter[str] = Counter()
    role_by_component: dict[str, str] = {}
    for sample in samples:
        identity = identity_by_token.get(sample["public_identity_token"])
        if identity is None or sample["identity_role"] != identity["role"]:
            raise ValueError("face protocol v2 sample and identity assignments differ")
        count_by_identity[sample["public_identity_token"]] += 1
        if not sample["cross_identity_unsafe"]:
            included_by_identity[sample["public_identity_token"]] += 1
            prior = role_by_component.setdefault(
                sample["dependency_component_sha256"], sample["role"]
            )
            if prior != sample["role"]:
                raise ValueError("face protocol v2 dependency component crosses roles")
        elif sample["role"] != FaceIdentityRoleV2.EXCLUDED_UNSAFE_COMPONENT.value:
            raise ValueError("face protocol v2 unsafe sample is not excluded")
    for identity in identities:
        token = identity["public_identity_token"]
        if (
            identity["sample_count"] != count_by_identity[token]
            or identity["included_sample_count"] != included_by_identity[token]
        ):
            raise ValueError("face protocol v2 identity sample counts differ")
    samples_by_identity: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_identity[sample["public_identity_token"]].append(sample)
    allocatable: set[str] = set()
    history_by_identity: dict[str, ExposureStage] = {}
    fixed_roles: dict[str, FaceIdentityRoleV2] = {}
    for identity in identities:
        token = identity["public_identity_token"]
        stage = (
            None
            if identity["historical_maximum_stage"] is None
            else ExposureStage(identity["historical_maximum_stage"])
        )
        if stage is not None:
            history_by_identity[token] = stage
        rows = samples_by_identity[token]
        if identity["included_sample_count"] == 0:
            fixed_roles[token] = FaceIdentityRoleV2.EXCLUDED_UNSAFE_COMPONENT
        elif (
            identity["dataset_name"] == "mpdd"
            or any(row["publisher_split"] == "test" for row in rows)
            or stage is ExposureStage.FINAL_TEST_SCORED
        ):
            fixed_roles[token] = FaceIdentityRoleV2.EXPOSED_DIAGNOSTIC
        else:
            allocatable.add(token)
    allocated = _allocate_identities(
        allocatable,
        history_by_identity=history_by_identity,
        protocol_name=protocol_name,
    )
    for identity in identities:
        token = identity["public_identity_token"]
        expected_role = fixed_roles.get(token, allocated.get(token))
        if expected_role is None or identity["role"] != expected_role.value:
            raise ValueError("face protocol v2 deterministic role allocation differs")


def _validate_record_digest(row: Mapping[str, Any]) -> None:
    _require_sha256(row["record_sha256"], "record_sha256")
    payload = {key: item for key, item in row.items() if key != "record_sha256"}
    if row["record_sha256"] != content_sha256(payload):
        raise ValueError("face protocol v2 record digest differs")


def _policy() -> dict[str, Any]:
    return {
        "schema_version": POLICY_SCHEMA,
        "allocatable_dataset": "dogfacenet224",
        "target_identity_percentages": dict(_TARGET_PERCENTAGES),
        "allocation_order_inputs": ["protocol_name", "public_identity_token"],
        "score_bearing_bytes_used_for_role_allocation": False,
        "dependency_closure_relations": sorted(
            relation.value for relation in _CLOSURE_RELATIONS
        ),
        "unknown_component_endpoint_policy": "EXCLUDE_CONNECTED_FACE_SAMPLES",
        "cross_identity_component_policy": "EXCLUDE_CONNECTED_FACE_SAMPLES",
        "unresolved_exposure_policy": "BLOCK_ROLE_ALLOCATION",
        "no_declared_exposure_is_clean_claim": False,
        "final_evaluation_permitted": False,
    }


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


__all__ = [
    "FaceIdentityRoleV2",
    "build_face_identity_protocol_v2",
    "validate_face_identity_protocol_v2_bundle",
]
