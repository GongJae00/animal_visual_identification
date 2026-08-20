"""Deterministic nested K1/K3/K5 face gallery/query panel."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from shared.foundation.provenance import content_sha256
from evaluation.splits.face.face_identity_protocol_v2 import (
    FaceIdentityRoleV2,
    validate_face_identity_protocol_v2_bundle,
)

PANEL_SCHEMA = "cvi.face_gallery_query_panel.v1"
COHORT_SCHEMA = "cvi.face_gallery_query_cohort.v1"
BUNDLE_SCHEMA = "cvi.face_gallery_query_panel_bundle.v1"
POLICY_SCHEMA = "cvi.face_gallery_query_panel_policy.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_K_VALUES = (1, 3, 5)
_INTERPRETATION = (
    "COMMON_K5_FEASIBLE_RETROSPECTIVE_FACE_COHORT;SHARED_QUERY_AND_NESTED_"
    "GALLERIES;PUBLISHER_GROUPS_ARE_NOT_VERIFIED_CROSS_SESSION_EVIDENCE"
)


def build_face_gallery_query_panel(protocol_v2_bundle: object) -> dict[str, Any]:
    """Select one common K5 cohort before any embeddings or scores are read."""

    protocol_bundle = validate_face_identity_protocol_v2_bundle(protocol_v2_bundle)
    protocol = protocol_bundle["protocol"]
    samples_by_identity: dict[str, list[Mapping[str, Any]]] = {}
    for row in protocol["sample_assignments"]:
        if row["role"] == FaceIdentityRoleV2.EXCLUDED_UNSAFE_COMPONENT.value:
            continue
        samples_by_identity.setdefault(row["public_identity_token"], []).append(row)
    identity_by_token = {
        row["public_identity_token"]: row for row in protocol["identity_assignments"]
    }
    cohort: list[dict[str, Any]] = []
    for identity in sorted(samples_by_identity):
        selected = _select_identity_panel(
            identity,
            samples_by_identity[identity],
            protocol_sha256=protocol["protocol_sha256"],
        )
        if selected is None:
            continue
        identity_row = identity_by_token[identity]
        payload = {
            "schema_version": COHORT_SCHEMA,
            "public_identity_token": identity,
            "registered_identity_id": identity_row["registered_identity_id"],
            "dataset_name": identity_row["dataset_name"],
            "protocol_role": identity_row["role"],
            **selected,
            "dependency_disjoint": True,
            "cross_session_verified": False,
        }
        cohort.append({**payload, "record_sha256": content_sha256(payload)})
    if not cohort:
        raise ValueError("face panel has no common K5-feasible identities")
    policy = _policy()
    panel = {
        "schema_version": PANEL_SCHEMA,
        "source_protocol_sha256": protocol["protocol_sha256"],
        "source_protocol_bundle_sha256": protocol_bundle["bundle_sha256"],
        "source_token_bridge_sha256": protocol["source_token_bridge_sha256"],
        "source_exposure_history_sha256": protocol["source_exposure_history_sha256"],
        "source_joint_filter_graph_sha256": protocol["joint_filter_graph_sha256"],
        "policy": policy,
        "policy_sha256": content_sha256(policy),
        "common_k5_feasible_cohort": cohort,
        "common_k5_feasible_identity_count": len(cohort),
        "score_inputs_used": False,
        "cross_session_claimed": False,
        "final_evaluation_permitted": False,
        "interpretation": _INTERPRETATION,
    }
    panel = {**panel, "panel_sha256": content_sha256(panel)}
    census = _census(cohort)
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "panel": panel,
        "panel_sha256": panel["panel_sha256"],
        "census": census,
        "census_sha256": content_sha256(census),
    }
    return {**payload, "bundle_sha256": content_sha256(payload)}


def validate_face_gallery_query_panel_bundle(value: object) -> dict[str, Any]:
    """Validate hashes, common-cohort semantics, nesting, and dependency separation."""

    expected = {
        "schema_version",
        "panel",
        "panel_sha256",
        "census",
        "census_sha256",
        "bundle_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face gallery/query panel bundle fields differ")
    bundle = dict(value)
    if bundle["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError("face gallery/query panel bundle schema differs")
    payload = {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    if bundle["bundle_sha256"] != content_sha256(payload):
        raise ValueError("face gallery/query panel bundle digest differs")
    panel = bundle["panel"]
    expected_panel = {
        "schema_version",
        "source_protocol_sha256",
        "source_protocol_bundle_sha256",
        "source_token_bridge_sha256",
        "source_exposure_history_sha256",
        "source_joint_filter_graph_sha256",
        "policy",
        "policy_sha256",
        "common_k5_feasible_cohort",
        "common_k5_feasible_identity_count",
        "score_inputs_used",
        "cross_session_claimed",
        "final_evaluation_permitted",
        "interpretation",
        "panel_sha256",
    }
    if not isinstance(panel, dict) or set(panel) != expected_panel:
        raise ValueError("face gallery/query panel fields differ")
    if (
        panel["schema_version"] != PANEL_SCHEMA
        or panel["policy"] != _policy()
        or panel["policy_sha256"] != content_sha256(panel["policy"])
        or panel["score_inputs_used"] is not False
        or panel["cross_session_claimed"] is not False
        or panel["final_evaluation_permitted"] is not False
        or panel["interpretation"] != _INTERPRETATION
    ):
        raise ValueError("face gallery/query panel policy differs")
    for field in (
        "source_protocol_sha256",
        "source_protocol_bundle_sha256",
        "source_token_bridge_sha256",
        "source_exposure_history_sha256",
        "source_joint_filter_graph_sha256",
        "policy_sha256",
        "panel_sha256",
    ):
        _require_sha256(panel[field], field)
    panel_payload = {key: item for key, item in panel.items() if key != "panel_sha256"}
    if (
        panel["panel_sha256"] != content_sha256(panel_payload)
        or bundle["panel_sha256"] != panel["panel_sha256"]
    ):
        raise ValueError("face gallery/query panel digest differs")
    raw_cohort = panel["common_k5_feasible_cohort"]
    if not isinstance(raw_cohort, list) or not raw_cohort:
        raise ValueError("face gallery/query cohort must not be empty")
    cohort = tuple(_validate_cohort(row) for row in raw_cohort)
    identities = [row["public_identity_token"] for row in cohort]
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("face gallery/query cohort must be sorted and unique")
    if panel["common_k5_feasible_identity_count"] != len(cohort):
        raise ValueError("face gallery/query cohort count differs")
    census = _census(cohort)
    if bundle["census"] != census or bundle["census_sha256"] != content_sha256(census):
        raise ValueError("face gallery/query panel census differs")
    return bundle


def _select_identity_panel(
    identity: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol_sha256: str,
) -> dict[str, Any] | None:
    dataset = rows[0]["dataset_name"]
    if any(row["dataset_name"] != dataset for row in rows):
        raise ValueError("face panel identity crosses datasets")
    if dataset == "mpdd":
        query_candidates = [row for row in rows if row["publisher_split"] == "query"]
        gallery_pool = [row for row in rows if row["publisher_split"] == "gallery"]
    else:
        query_candidates = list(rows)
        gallery_pool = list(rows)
    query_candidates.sort(
        key=lambda row: (
            _sample_rank(
                protocol_sha256, identity, row["public_sample_token"], "QUERY"
            ),
            row["public_sample_token"],
        )
    )
    for query in query_candidates:
        galleries = [
            row
            for row in gallery_pool
            if row["public_sample_token"] != query["public_sample_token"]
            and row["dependency_component_sha256"]
            != query["dependency_component_sha256"]
        ]
        galleries.sort(
            key=lambda row: (
                _sample_rank(
                    protocol_sha256,
                    identity,
                    row["public_sample_token"],
                    "GALLERY",
                ),
                row["public_sample_token"],
            )
        )
        distinct: list[Mapping[str, Any]] = []
        components: set[str] = set()
        for row in galleries:
            component = row["dependency_component_sha256"]
            if component in components:
                continue
            components.add(component)
            distinct.append(row)
            if len(distinct) == 5:
                break
        if len(distinct) < 5:
            continue
        gallery_tokens = [row["public_sample_token"] for row in distinct]
        gallery_components = [row["dependency_component_sha256"] for row in distinct]
        return {
            "query_sample_token": query["public_sample_token"],
            "query_dependency_component_sha256": query["dependency_component_sha256"],
            "gallery_sample_tokens_by_k": {
                f"K{k}": gallery_tokens[:k] for k in _K_VALUES
            },
            "gallery_dependency_components_by_k": {
                f"K{k}": gallery_components[:k] for k in _K_VALUES
            },
        }
    return None


def _sample_rank(protocol_sha256: str, identity: str, sample: str, purpose: str) -> str:
    return hashlib.sha256(
        (
            f"CVI_FACE_PANEL_V1\0{protocol_sha256}\0{identity}\0{purpose}\0{sample}"
        ).encode("ascii")
    ).hexdigest()


def _validate_cohort(value: object) -> dict[str, Any]:
    expected = {
        "schema_version",
        "public_identity_token",
        "registered_identity_id",
        "dataset_name",
        "protocol_role",
        "query_sample_token",
        "query_dependency_component_sha256",
        "gallery_sample_tokens_by_k",
        "gallery_dependency_components_by_k",
        "dependency_disjoint",
        "cross_session_verified",
        "record_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face gallery/query cohort record fields differ")
    row = dict(value)
    FaceIdentityRoleV2(row["protocol_role"])
    if (
        row["schema_version"] != COHORT_SCHEMA
        or row["dataset_name"] not in {"dogfacenet224", "mpdd"}
        or row["protocol_role"] == FaceIdentityRoleV2.EXCLUDED_UNSAFE_COMPONENT.value
        or row["dependency_disjoint"] is not True
        or row["cross_session_verified"] is not False
    ):
        raise ValueError("face gallery/query cohort policy differs")
    for field in (
        "public_identity_token",
        "query_sample_token",
        "query_dependency_component_sha256",
        "record_sha256",
    ):
        _require_sha256(row[field], field)
    galleries = row["gallery_sample_tokens_by_k"]
    components = row["gallery_dependency_components_by_k"]
    if (
        not isinstance(galleries, dict)
        or not isinstance(components, dict)
        or set(galleries) != {"K1", "K3", "K5"}
        or set(components) != {"K1", "K3", "K5"}
    ):
        raise ValueError("face gallery/query K fields differ")
    for k in _K_VALUES:
        tokens = galleries[f"K{k}"]
        component_values = components[f"K{k}"]
        if (
            not isinstance(tokens, list)
            or not isinstance(component_values, list)
            or len(tokens) != k
            or len(component_values) != k
            or len(tokens) != len(set(tokens))
            or len(component_values) != len(set(component_values))
        ):
            raise ValueError("face gallery/query K population differs")
        for token in (*tokens, *component_values):
            _require_sha256(token, "face gallery/query token")
        if (
            row["query_sample_token"] in tokens
            or row["query_dependency_component_sha256"] in component_values
        ):
            raise ValueError("face gallery/query dependency disjointness differs")
    if galleries["K1"] != galleries["K3"][:1] or galleries["K3"] != galleries["K5"][:3]:
        raise ValueError("face galleries are not nested K1 subset K3 subset K5")
    if (
        components["K1"] != components["K3"][:1]
        or components["K3"] != (components["K5"][:3])
    ):
        raise ValueError("face gallery dependency components are not nested")
    payload = {key: item for key, item in row.items() if key != "record_sha256"}
    if row["record_sha256"] != content_sha256(payload):
        raise ValueError("face gallery/query cohort record digest differs")
    return row


def _census(cohort: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    datasets = Counter(row["dataset_name"] for row in cohort)
    roles = Counter(row["protocol_role"] for row in cohort)
    return {
        "schema_version": "cvi.face_gallery_query_panel_census.v1",
        "common_k5_feasible_identity_count": len(cohort),
        "query_count": len(cohort),
        "gallery_entry_counts_by_k": {f"K{k}": len(cohort) * k for k in _K_VALUES},
        "dataset_identity_counts": {
            dataset: datasets[dataset] for dataset in ("dogfacenet224", "mpdd")
        },
        "role_identity_counts": {
            role.value: roles[role.value]
            for role in FaceIdentityRoleV2
            if role is not FaceIdentityRoleV2.EXCLUDED_UNSAFE_COMPONENT
        },
        "dependency_disjoint": True,
        "cross_session_claimed": False,
        "final_evaluation_permitted": False,
    }


def _policy() -> dict[str, Any]:
    return {
        "schema_version": POLICY_SCHEMA,
        "cohort_basis": "COMMON_K5_FEASIBLE_IDENTITIES",
        "k_values": list(_K_VALUES),
        "shared_query_across_k": True,
        "nested_galleries": "K1_SUBSET_K3_SUBSET_K5",
        "query_gallery_dependency_disjoint": True,
        "gallery_dependency_components_distinct": True,
        "selection_inputs": [
            "protocol_sha256",
            "public_identity_token",
            "public_sample_token",
            "publisher_split",
            "dependency_component_sha256",
        ],
        "score_inputs_used": False,
        "cross_session_claimed": False,
        "final_evaluation_permitted": False,
    }


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


__all__ = [
    "build_face_gallery_query_panel",
    "validate_face_gallery_query_panel_bundle",
]
