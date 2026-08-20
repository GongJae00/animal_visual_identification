"""Metadata-face-eligible successor inventory over materialized Full128 crops.

The contracts in this module join existing authorities.  They never derive a new
crop, reinterpret identity-free rows as identities, or use model scores to select
the fixed evaluation population.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from shared.foundation.provenance import content_sha256
from evaluation.splits.face.face_eligibility import (
    validate_face_eligibility_overlay_bundle,
)
from evaluation.splits.face.face_gallery_query_panel import (
    validate_face_gallery_query_panel_bundle,
)
from evaluation.splits.face.face_identity_protocol import (
    validate_face_identity_protocol_bundle,
)
from evaluation.splits.face.face_identity_protocol_v2 import (
    validate_face_identity_protocol_v2_bundle,
)
from archive.full128.methods.preparation.inventory import (
    validate_full128_experiment_inventory_bundle,
)
from archive.full128.methods.preparation.materialization import ASSEMBLY_SCHEMA
from data.full_segment.route_plan import (
    validate_full128_route_plan_bundle,
)

PANEL_SCHEMA = "cvi.full128_face_visible_fixed_panel.v1"
PANEL_RECORD_SCHEMA = "cvi.full128_face_visible_fixed_panel_record.v1"
AUTHORITATIVE_PANEL_SCHEMA = "cvi.full128_face_visible_authoritative_panel.v2"
AUTHORITATIVE_PANEL_RECORD_SCHEMA = (
    "cvi.full128_face_visible_authoritative_panel_record.v2"
)
AUTHORITATIVE_COHORT_SCHEMA = "cvi.full128_face_visible_authoritative_cohort.v2"
INVENTORY_SCHEMA = "cvi.full128_face_visible_successor_inventory.v1"
INVENTORY_RECORD_SCHEMA = "cvi.full128_face_visible_successor_record.v1"
BUNDLE_SCHEMA = "cvi.full128_face_visible_successor_inventory_bundle.v1"

EVALUATION_SCOPES = ("DEV", "CAL", "EXPOSED_DIAGNOSTIC")
ENROLLMENT_KS = (1, 3, 5)
_ASSEMBLY_FIELDS = {
    "schema_version",
    "plan_sha256",
    "sample_count",
    "allocation_name",
    "topology_report",
    "unified_full_split",
    "inventory_request",
    "inventory_bundle",
    "assembly_sha256",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_ARTIFACT_BYTES = 67_108_864


class FaceVisibleInventoryError(ValueError):
    """Raised when joined successor evidence is incomplete or inconsistent."""


def build_score_blind_face_visible_panel(
    route_plan_bundle: object,
    face_overlay_bundle: object,
    face_protocol_bundle: object,
) -> dict[str, Any]:
    """Freeze deterministic Q/G assignments without loading crops or scores."""

    route, overlay, protocol = _validated_face_sources(
        route_plan_bundle, face_overlay_bundle, face_protocol_bundle
    )
    route_by_sample = {
        row["sample_token"]: row for row in route["plan"]["records"]
    }
    assignments = protocol["protocol"]["sample_assignments"]
    by_identity: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in assignments:
        by_identity[row["registered_identity_id"]].append(row)

    records: list[dict[str, Any]] = []
    for identity in sorted(by_identity):
        identity_rows = sorted(
            by_identity[identity], key=lambda item: item["sample_token"]
        )
        role = identity_rows[0]["role"]
        dataset = identity_rows[0]["dataset_name"]
        uses = _panel_uses(identity_rows, dataset=dataset, role=role)
        for row in identity_rows:
            source = route_by_sample[row["sample_token"]]
            payload = {
                "schema_version": PANEL_RECORD_SCHEMA,
                "sample_token": row["sample_token"],
                "registered_identity_id": identity,
                "dataset_name": dataset,
                "scope": role,
                "publisher_split": row["publisher_split"],
                "duplicate_component": row["duplicate_component"],
                "effective_content_guard": source["source_sha256"],
                "uses_by_enrollment_k": uses[row["sample_token"]],
                "score_inputs_used": False,
            }
            records.append({**payload, "record_sha256": content_sha256(payload)})
    records.sort(key=lambda item: item["sample_token"])
    _reject_panel_leakage(records)
    payload = {
        "schema_version": PANEL_SCHEMA,
        "source_route_plan_sha256": route["plan_sha256"],
        "source_route_plan_bundle_sha256": route["bundle_sha256"],
        "source_face_overlay_sha256": overlay["overlay_sha256"],
        "source_face_overlay_bundle_sha256": overlay["bundle_sha256"],
        "source_face_protocol_sha256": protocol["protocol_sha256"],
        "source_face_protocol_bundle_sha256": protocol["bundle_sha256"],
        "selection_policy": {
            "decision_basis": "PUBLISHER_METADATA_AND_PROTOCOL_ONLY",
            "ordering": "SAMPLE_TOKEN_ASC",
            "enrollment_ks": list(ENROLLMENT_KS),
            "mpdd_roles": "PUBLISHER_QUERY_GALLERY",
            "dogface_roles": "FIRST_K_GALLERY_REMAINDER_QUERY",
            "insufficient_identity_action": "EXPLICIT_TERMINAL_EXCLUSION",
            "random_frame_splitting": False,
            "score_inputs_used": False,
        },
        "scope_labels": list(EVALUATION_SCOPES),
        "records": records,
        "score_inputs_used": False,
    }
    return {**payload, "panel_sha256": content_sha256(payload)}


def build_authoritative_face_visible_panel(
    face_protocol_v2_bundle: object,
    gallery_query_panel_bundle: object,
) -> dict[str, Any]:
    """Map an authoritative public-token panel to route tokens without reselection."""

    protocol_bundle = validate_face_identity_protocol_v2_bundle(
        face_protocol_v2_bundle
    )
    governance_bundle = validate_face_gallery_query_panel_bundle(
        gallery_query_panel_bundle
    )
    protocol = protocol_bundle["protocol"]
    governance = governance_bundle["panel"]
    if (
        governance["source_protocol_sha256"] != protocol["protocol_sha256"]
        or governance["source_protocol_bundle_sha256"]
        != protocol_bundle["bundle_sha256"]
        or governance["source_token_bridge_sha256"]
        != protocol["source_token_bridge_sha256"]
        or governance["source_exposure_history_sha256"]
        != protocol["source_exposure_history_sha256"]
        or governance["source_joint_filter_graph_sha256"]
        != protocol["joint_filter_graph_sha256"]
    ):
        raise FaceVisibleInventoryError(
            "governance gallery/query panel and face protocol v2 bindings differ"
        )
    assignments = protocol["sample_assignments"]
    by_public = {row["public_sample_token"]: row for row in assignments}
    by_route = {row["route_sample_token"]: row for row in assignments}
    if len(by_public) != len(assignments) or len(by_route) != len(assignments):
        raise FaceVisibleInventoryError(
            "face protocol v2 public/route sample mapping is not one-to-one"
        )
    safe_assignments = {
        public: row
        for public, row in by_public.items()
        if not row["cross_identity_unsafe"]
    }
    uses_by_public = {
        public: {
            str(k): {
                "use": (
                    "TRAINING_ONLY"
                    if row["role"] == "FIT"
                    else "TERMINAL_EXCLUSION"
                ),
                "reason": (
                    "FIT_SCOPE_NOT_EVALUATED"
                    if row["role"] == "FIT"
                    else "NOT_IN_AUTHORITATIVE_COMMON_K5_COHORT"
                ),
            }
            for k in ENROLLMENT_KS
        }
        for public, row in safe_assignments.items()
    }
    authority_by_public: dict[str, str] = {}
    selected_public: set[str] = set()
    authoritative_cohorts: list[dict[str, Any]] = []
    for cohort in governance["common_k5_feasible_cohort"]:
        identity = cohort["public_identity_token"]
        query_public = cohort["query_sample_token"]
        gallery_by_k = cohort["gallery_sample_tokens_by_k"]
        expected = {query_public, *gallery_by_k["K5"]}
        for public in expected:
            assignment = safe_assignments.get(public)
            if assignment is None:
                raise FaceVisibleInventoryError(
                    "authoritative panel references an unsafe or unknown public sample"
                )
            if (
                assignment["public_identity_token"] != identity
                or assignment["registered_identity_id"]
                != cohort["registered_identity_id"]
                or assignment["dataset_name"] != cohort["dataset_name"]
                or assignment["role"] != cohort["protocol_role"]
            ):
                raise FaceVisibleInventoryError(
                    "authoritative panel cohort differs from protocol v2 assignment"
                )
            if public in authority_by_public:
                raise FaceVisibleInventoryError(
                    "authoritative panel repeats a public sample across identities"
                )
            authority_by_public[public] = cohort["record_sha256"]
        selected_public.update(expected)
        cohort_payload = {
            "schema_version": AUTHORITATIVE_COHORT_SCHEMA,
            "public_identity_token": identity,
            "registered_identity_id": cohort["registered_identity_id"],
            "dataset_name": cohort["dataset_name"],
            "scope": cohort["protocol_role"],
            "query_public_sample_token": query_public,
            "query_route_sample_token": safe_assignments[query_public][
                "route_sample_token"
            ],
            "gallery_public_sample_tokens_by_k": {
                key: list(cohort["gallery_sample_tokens_by_k"][key])
                for key in ("K1", "K3", "K5")
            },
            "gallery_route_sample_tokens_by_k": {
                key: [
                    safe_assignments[public]["route_sample_token"]
                    for public in cohort["gallery_sample_tokens_by_k"][key]
                ]
                for key in ("K1", "K3", "K5")
            },
            "governance_cohort_record_sha256": cohort["record_sha256"],
        }
        authoritative_cohorts.append(
            {**cohort_payload, "record_sha256": content_sha256(cohort_payload)}
        )
        for k in ENROLLMENT_KS:
            key = f"K{k}"
            uses_by_public[query_public][str(k)] = {
                "use": "QUERY",
                "reason": None,
            }
            gallery_members = set(gallery_by_k[key])
            for public in gallery_by_k["K5"]:
                uses_by_public[public][str(k)] = (
                    {"use": "GALLERY", "reason": None}
                    if public in gallery_members
                    else {
                        "use": "TERMINAL_EXCLUSION",
                        "reason": f"NOT_IN_AUTHORITATIVE_{key}_GALLERY",
                    }
                )
    records: list[dict[str, Any]] = []
    for public in sorted(safe_assignments):
        assignment = safe_assignments[public]
        payload = {
            "schema_version": AUTHORITATIVE_PANEL_RECORD_SCHEMA,
            "sample_token": assignment["route_sample_token"],
            "public_sample_token": public,
            "public_identity_token": assignment["public_identity_token"],
            "registered_identity_id": assignment["registered_identity_id"],
            "dataset_name": assignment["dataset_name"],
            "scope": assignment["role"],
            "publisher_split": assignment["publisher_split"],
            "duplicate_component": assignment["dependency_component_sha256"],
            "effective_content_guard": public,
            "dependency_component_sha256": assignment[
                "dependency_component_sha256"
            ],
            "protocol_sample_record_sha256": assignment["record_sha256"],
            "governance_cohort_record_sha256": authority_by_public.get(public),
            "authoritative_cohort_member": public in selected_public,
            "uses_by_enrollment_k": uses_by_public[public],
            "score_inputs_used": False,
        }
        records.append({**payload, "record_sha256": content_sha256(payload)})
    records.sort(key=lambda item: item["sample_token"])
    if len(records) != len({row["sample_token"] for row in records}):
        raise FaceVisibleInventoryError(
            "authoritative public-to-route token mapping is not one-to-one"
        )
    _reject_panel_leakage(records)
    payload = {
        "schema_version": AUTHORITATIVE_PANEL_SCHEMA,
        "source_route_plan_sha256": protocol["source_route_plan_sha256"],
        "source_route_plan_bundle_sha256": protocol[
            "source_route_plan_bundle_sha256"
        ],
        "source_face_overlay_sha256": protocol["source_face_overlay_sha256"],
        "source_face_overlay_bundle_sha256": protocol[
            "source_face_overlay_bundle_sha256"
        ],
        "source_face_protocol_v2_sha256": protocol["protocol_sha256"],
        "source_face_protocol_v2_bundle_sha256": protocol_bundle["bundle_sha256"],
        "source_gallery_query_panel_sha256": governance["panel_sha256"],
        "source_gallery_query_panel_bundle_sha256": governance_bundle[
            "bundle_sha256"
        ],
        "selection_policy": {
            "authority": "PERSISTED_FACE_GALLERY_QUERY_PANEL",
            "public_to_route_mapping": "FACE_PROTOCOL_V2_SAMPLE_ASSIGNMENTS",
            "shared_query_across_k": True,
            "nested_galleries": "K1_SUBSET_K3_SUBSET_K5",
            "cross_identity_unsafe_policy": "EXCLUDE",
            "random_frame_splitting": False,
            "enrollment_ks": list(ENROLLMENT_KS),
            "score_inputs_used": False,
        },
        "scope_labels": list(EVALUATION_SCOPES),
        "authoritative_cohorts": authoritative_cohorts,
        "records": records,
        "score_inputs_used": False,
    }
    return {**payload, "panel_sha256": content_sha256(payload)}


def validate_score_blind_face_visible_panel(
    value: object,
    *,
    route_plan_bundle: object | None = None,
    face_overlay_bundle: object | None = None,
    face_protocol_bundle: object | None = None,
    face_protocol_v2_bundle: object | None = None,
    gallery_query_panel_bundle: object | None = None,
) -> dict[str, Any]:
    """Validate the fixed panel, optionally rebuilding it from all authorities."""

    if (
        isinstance(value, Mapping)
        and value.get("schema_version") == AUTHORITATIVE_PANEL_SCHEMA
    ):
        return _validate_authoritative_face_visible_panel(
            value,
            face_protocol_v2_bundle=face_protocol_v2_bundle,
            gallery_query_panel_bundle=gallery_query_panel_bundle,
        )
    if face_protocol_v2_bundle is not None or gallery_query_panel_bundle is not None:
        raise FaceVisibleInventoryError(
            "governance v2 authorities require an authoritative v2 panel"
        )

    expected = {
        "schema_version",
        "source_route_plan_sha256",
        "source_route_plan_bundle_sha256",
        "source_face_overlay_sha256",
        "source_face_overlay_bundle_sha256",
        "source_face_protocol_sha256",
        "source_face_protocol_bundle_sha256",
        "selection_policy",
        "scope_labels",
        "records",
        "score_inputs_used",
        "panel_sha256",
    }
    _exact_keys(value, expected, "face-visible fixed panel")
    panel = dict(value)
    if panel["schema_version"] != PANEL_SCHEMA:
        raise FaceVisibleInventoryError("face-visible fixed panel schema differs")
    payload = {key: item for key, item in panel.items() if key != "panel_sha256"}
    if panel["panel_sha256"] != content_sha256(payload):
        raise FaceVisibleInventoryError("face-visible fixed panel digest differs")
    if panel["score_inputs_used"] is not False or panel["scope_labels"] != list(
        EVALUATION_SCOPES
    ):
        raise FaceVisibleInventoryError("face-visible panel must remain score blind")
    policy = panel["selection_policy"]
    if (
        not isinstance(policy, Mapping)
        or policy.get("score_inputs_used") is not False
        or policy.get("random_frame_splitting") is not False
        or policy.get("enrollment_ks") != list(ENROLLMENT_KS)
    ):
        raise FaceVisibleInventoryError("face-visible panel policy differs")
    records = panel["records"]
    if not isinstance(records, list) or not records:
        raise FaceVisibleInventoryError("face-visible panel records must not be empty")
    tokens: list[str] = []
    for record in records:
        _validate_panel_record(record)
        tokens.append(record["sample_token"])
    if tokens != sorted(set(tokens)):
        raise FaceVisibleInventoryError(
            "face-visible panel records must be uniquely sample-token sorted"
        )
    _reject_panel_leakage(records)
    sources = (route_plan_bundle, face_overlay_bundle, face_protocol_bundle)
    if any(item is not None for item in sources):
        if any(item is None for item in sources):
            raise FaceVisibleInventoryError(
                "all face-visible panel authorities must be supplied together"
            )
        rebuilt = build_score_blind_face_visible_panel(*sources)
        if rebuilt != panel:
            raise FaceVisibleInventoryError(
                "face-visible panel differs from its source authorities"
            )
    return panel


def _validate_authoritative_face_visible_panel(
    value: object,
    *,
    face_protocol_v2_bundle: object | None,
    gallery_query_panel_bundle: object | None,
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "source_route_plan_sha256",
        "source_route_plan_bundle_sha256",
        "source_face_overlay_sha256",
        "source_face_overlay_bundle_sha256",
        "source_face_protocol_v2_sha256",
        "source_face_protocol_v2_bundle_sha256",
        "source_gallery_query_panel_sha256",
        "source_gallery_query_panel_bundle_sha256",
        "selection_policy",
        "scope_labels",
        "authoritative_cohorts",
        "records",
        "score_inputs_used",
        "panel_sha256",
    }
    _exact_keys(value, expected, "authoritative face-visible panel")
    panel = dict(value)
    payload = {key: item for key, item in panel.items() if key != "panel_sha256"}
    if (
        panel["panel_sha256"] != content_sha256(payload)
        or panel["score_inputs_used"] is not False
        or panel["scope_labels"] != list(EVALUATION_SCOPES)
        or panel["selection_policy"]
        != {
            "authority": "PERSISTED_FACE_GALLERY_QUERY_PANEL",
            "public_to_route_mapping": "FACE_PROTOCOL_V2_SAMPLE_ASSIGNMENTS",
            "shared_query_across_k": True,
            "nested_galleries": "K1_SUBSET_K3_SUBSET_K5",
            "cross_identity_unsafe_policy": "EXCLUDE",
            "random_frame_splitting": False,
            "enrollment_ks": list(ENROLLMENT_KS),
            "score_inputs_used": False,
        }
    ):
        raise FaceVisibleInventoryError(
            "authoritative face-visible panel contract differs"
        )
    for field in expected - {
        "schema_version",
        "selection_policy",
        "scope_labels",
        "authoritative_cohorts",
        "records",
        "score_inputs_used",
    }:
        _require_sha256(panel[field], field)
    records = panel["records"]
    if not isinstance(records, list) or not records:
        raise FaceVisibleInventoryError(
            "authoritative face-visible panel records must not be empty"
        )
    route_tokens: list[str] = []
    public_tokens: list[str] = []
    for record in records:
        _validate_authoritative_panel_record(record)
        route_tokens.append(record["sample_token"])
        public_tokens.append(record["public_sample_token"])
    if route_tokens != sorted(set(route_tokens)) or len(public_tokens) != len(
        set(public_tokens)
    ):
        raise FaceVisibleInventoryError(
            "authoritative panel route/public token mapping differs"
        )
    _validate_authoritative_nested_memberships(records)
    _validate_authoritative_cohort_mappings(
        panel["authoritative_cohorts"], records
    )
    _reject_panel_leakage(records)
    if (face_protocol_v2_bundle is None) != (gallery_query_panel_bundle is None):
        raise FaceVisibleInventoryError(
            "both governance v2 protocol and gallery/query panel are required"
        )
    if face_protocol_v2_bundle is not None:
        rebuilt = build_authoritative_face_visible_panel(
            face_protocol_v2_bundle, gallery_query_panel_bundle
        )
        if rebuilt != panel:
            raise FaceVisibleInventoryError(
                "authoritative face-visible panel differs from governance inputs"
            )
    return panel


def build_face_visible_successor_inventory(
    *,
    route_plan_bundle: object,
    materialization_assembly: object,
    face_overlay_bundle: object,
    face_protocol_bundle: object | None = None,
    fixed_panel: object | None = None,
    face_protocol_v2_bundle: object | None = None,
    gallery_query_panel_bundle: object | None = None,
    validation_workers: int = 1,
) -> dict[str, Any]:
    """Join route, materialization, face metadata, protocol, and fixed panel."""

    v2_requested = (
        face_protocol_v2_bundle is not None
        or gallery_query_panel_bundle is not None
    )
    if v2_requested:
        if (
            face_protocol_v2_bundle is None
            or gallery_query_panel_bundle is None
            or face_protocol_bundle is not None
            or fixed_panel is not None
        ):
            raise FaceVisibleInventoryError(
                "governance v2 requires protocol v2 and gallery/query panel only"
            )
        return build_face_visible_successor_inventory_v2(
            route_plan_bundle=route_plan_bundle,
            materialization_assembly=materialization_assembly,
            face_overlay_bundle=face_overlay_bundle,
            face_protocol_v2_bundle=face_protocol_v2_bundle,
            gallery_query_panel_bundle=gallery_query_panel_bundle,
            validation_workers=validation_workers,
        )
    if face_protocol_bundle is None or fixed_panel is None:
        raise FaceVisibleInventoryError(
            "legacy v1 inventory requires face protocol and fixed panel"
        )

    route, overlay, protocol = _validated_face_sources(
        route_plan_bundle, face_overlay_bundle, face_protocol_bundle
    )
    panel = validate_score_blind_face_visible_panel(
        fixed_panel,
        route_plan_bundle=route,
        face_overlay_bundle=overlay,
        face_protocol_bundle=protocol,
    )
    assembly, full_inventory = _validate_assembly(
        materialization_assembly, validation_workers=validation_workers
    )
    if assembly["plan_sha256"] != route["plan_sha256"]:
        raise FaceVisibleInventoryError(
            "Full128 materialization assembly and route plan differ"
        )

    route_rows = route["plan"]["records"]
    route_by_sample = {row["sample_token"]: row for row in route_rows}
    overlay_by_sample = {
        row["sample_token"]: row for row in overlay["overlay"]["records"]
    }
    material_by_sample = {
        row["sample_token"]: row
        for row in full_inventory["inventory"]["records"]
    }
    protocol_by_sample = {
        row["sample_token"]: row
        for row in protocol["protocol"]["sample_assignments"]
    }
    panel_by_sample = {row["sample_token"]: row for row in panel["records"]}
    selected = set(route_by_sample)
    for label, population in (
        ("face overlay", overlay_by_sample),
        ("Full128 materialization", material_by_sample),
    ):
        if set(population) != selected:
            missing = sorted(selected - set(population))
            extra = sorted(set(population) - selected)
            raise FaceVisibleInventoryError(
                f"{label} population differs; missing={missing[:3]}, extra={extra[:3]}"
            )
    if set(protocol_by_sample) != set(panel_by_sample):
        raise FaceVisibleInventoryError(
            "face protocol and fixed panel sample populations differ"
        )

    successor: list[dict[str, Any]] = []
    auxiliary: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []
    for sample_token in sorted(selected):
        source = route_by_sample[sample_token]
        face = overlay_by_sample[sample_token]
        material = material_by_sample[sample_token]
        assignment = protocol_by_sample.get(sample_token)
        panel_record = panel_by_sample.get(sample_token)
        record = _joined_record(
            source,
            face,
            material,
            assignment=assignment,
            panel_record=panel_record,
            artifact_root=Path(full_inventory["artifact_root"]),
        )
        if material["identity_evidence_kind"] == "NONE":
            auxiliary.append(record)
        elif assignment is not None:
            successor.append(record)
        else:
            terminal.append(record)

    all_records = successor + auxiliary + terminal
    if len(all_records) != len(route_rows) or len(
        {row["sample_token"] for row in all_records}
    ) != len(route_rows):
        raise FaceVisibleInventoryError(
            "successor inventory did not preserve every route-plan sample"
        )
    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "artifact_root": full_inventory["artifact_root"],
        "crop_policy": {
            "source": "EXISTING_FULL128_MATERIALIZATION_ONLY",
            "recrop_permitted": False,
            "rgb_filename": "full.png",
            "mask_filename": "full-mask.png",
        },
        "successor_population": successor,
        "identity_free_auxiliary_population": auxiliary,
        "terminal_exclusions": terminal,
        "coverage": {
            "route_plan_sample_count": len(route_rows),
            "successor_sample_count": len(successor),
            "identity_free_auxiliary_sample_count": len(auxiliary),
            "terminal_exclusion_count": len(terminal),
            "state_counts": dict(
                sorted(Counter(row["state"] for row in all_records).items())
            ),
        },
    }
    source_binding = {
        "route_plan_sha256": route["plan_sha256"],
        "route_plan_bundle_sha256": route["bundle_sha256"],
        "assembly_sha256": assembly["assembly_sha256"],
        "full128_inventory_bundle_sha256": full_inventory["bundle_sha256"],
        "full128_inventory_sha256": full_inventory["inventory_sha256"],
        "face_overlay_sha256": overlay["overlay_sha256"],
        "face_overlay_bundle_sha256": overlay["bundle_sha256"],
        "face_protocol_sha256": protocol["protocol_sha256"],
        "face_protocol_bundle_sha256": protocol["bundle_sha256"],
        "fixed_panel_sha256": panel["panel_sha256"],
    }
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "content_kind": "METADATA_AND_EXISTING_ARTIFACT_BINDINGS",
        "source_binding": source_binding,
        "inventory": inventory,
        "inventory_sha256": content_sha256(inventory),
    }
    return {**payload, "bundle_sha256": content_sha256(payload)}


def build_face_visible_successor_inventory_v2(
    *,
    route_plan_bundle: object,
    materialization_assembly: object,
    face_overlay_bundle: object,
    face_protocol_v2_bundle: object,
    gallery_query_panel_bundle: object,
    validation_workers: int = 1,
) -> dict[str, Any]:
    """Join Full128 artifacts to governance v2 and its authoritative public panel."""

    route, overlay, protocol, governance = _validated_face_sources_v2(
        route_plan_bundle,
        face_overlay_bundle,
        face_protocol_v2_bundle,
        gallery_query_panel_bundle,
    )
    panel = build_authoritative_face_visible_panel(protocol, governance)
    panel = validate_score_blind_face_visible_panel(
        panel,
        face_protocol_v2_bundle=protocol,
        gallery_query_panel_bundle=governance,
    )
    assembly, full_inventory = _validate_assembly(
        materialization_assembly, validation_workers=validation_workers
    )
    if assembly["plan_sha256"] != route["plan_sha256"]:
        raise FaceVisibleInventoryError(
            "Full128 materialization assembly and route plan differ"
        )
    route_rows = route["plan"]["records"]
    route_by_sample = {row["sample_token"]: row for row in route_rows}
    overlay_by_sample = {
        row["sample_token"]: row for row in overlay["overlay"]["records"]
    }
    material_by_sample = {
        row["sample_token"]: row
        for row in full_inventory["inventory"]["records"]
    }
    protocol_by_sample = {
        row["route_sample_token"]: row
        for row in protocol["protocol"]["sample_assignments"]
    }
    safe_protocol_by_sample = {
        token: row
        for token, row in protocol_by_sample.items()
        if not row["cross_identity_unsafe"]
    }
    panel_by_sample = {row["sample_token"]: row for row in panel["records"]}
    selected = set(route_by_sample)
    for label, population in (
        ("face overlay", overlay_by_sample),
        ("Full128 materialization", material_by_sample),
    ):
        if set(population) != selected:
            missing = sorted(selected - set(population))
            extra = sorted(set(population) - selected)
            raise FaceVisibleInventoryError(
                f"{label} population differs; missing={missing[:3]}, extra={extra[:3]}"
            )
    if set(safe_protocol_by_sample) != set(panel_by_sample):
        raise FaceVisibleInventoryError(
            "safe protocol v2 population and authoritative panel mapping differ"
        )
    successor: list[dict[str, Any]] = []
    auxiliary: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []
    for sample_token in sorted(selected):
        source = route_by_sample[sample_token]
        face = overlay_by_sample[sample_token]
        material = material_by_sample[sample_token]
        assignment = protocol_by_sample.get(sample_token)
        unsafe = bool(assignment is not None and assignment["cross_identity_unsafe"])
        record = _joined_record(
            source,
            face,
            material,
            assignment=assignment,
            panel_record=panel_by_sample.get(sample_token),
            artifact_root=Path(full_inventory["artifact_root"]),
        )
        if material["identity_evidence_kind"] == "NONE":
            auxiliary.append(record)
        elif assignment is not None and not unsafe:
            successor.append(record)
        else:
            terminal.append(record)
    all_records = successor + auxiliary + terminal
    if len(all_records) != len(route_rows) or len(
        {row["sample_token"] for row in all_records}
    ) != len(route_rows):
        raise FaceVisibleInventoryError(
            "successor inventory did not preserve every route-plan sample"
        )
    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "artifact_root": full_inventory["artifact_root"],
        "crop_policy": {
            "source": "EXISTING_FULL128_MATERIALIZATION_ONLY",
            "recrop_permitted": False,
            "rgb_filename": "full.png",
            "mask_filename": "full-mask.png",
        },
        "successor_population": successor,
        "identity_free_auxiliary_population": auxiliary,
        "terminal_exclusions": terminal,
        "coverage": {
            "route_plan_sample_count": len(route_rows),
            "successor_sample_count": len(successor),
            "identity_free_auxiliary_sample_count": len(auxiliary),
            "terminal_exclusion_count": len(terminal),
            "state_counts": dict(
                sorted(Counter(row["state"] for row in all_records).items())
            ),
        },
    }
    source_binding = {
        "route_plan_sha256": route["plan_sha256"],
        "route_plan_bundle_sha256": route["bundle_sha256"],
        "assembly_sha256": assembly["assembly_sha256"],
        "full128_inventory_bundle_sha256": full_inventory["bundle_sha256"],
        "full128_inventory_sha256": full_inventory["inventory_sha256"],
        "face_overlay_sha256": overlay["overlay_sha256"],
        "face_overlay_bundle_sha256": overlay["bundle_sha256"],
        "face_protocol_v2_sha256": protocol["protocol_sha256"],
        "face_protocol_v2_bundle_sha256": protocol["bundle_sha256"],
        "gallery_query_panel_sha256": governance["panel_sha256"],
        "gallery_query_panel_bundle_sha256": governance["bundle_sha256"],
        "fixed_panel_sha256": panel["panel_sha256"],
    }
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "content_kind": "METADATA_AND_EXISTING_ARTIFACT_BINDINGS",
        "source_binding": source_binding,
        "inventory": inventory,
        "inventory_sha256": content_sha256(inventory),
    }
    return {**payload, "bundle_sha256": content_sha256(payload)}


def validate_face_visible_successor_inventory_bundle(
    value: object,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate a successor bundle and optionally rehash every usable RGB/mask."""

    _exact_keys(
        value,
        {
            "schema_version",
            "content_kind",
            "source_binding",
            "inventory",
            "inventory_sha256",
            "bundle_sha256",
        },
        "face-visible successor inventory bundle",
    )
    bundle = dict(value)
    if bundle["schema_version"] != BUNDLE_SCHEMA:
        raise FaceVisibleInventoryError("successor inventory bundle schema differs")
    payload = {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    if bundle["bundle_sha256"] != content_sha256(payload):
        raise FaceVisibleInventoryError("successor inventory bundle digest differs")
    inventory = bundle["inventory"]
    _exact_keys(
        inventory,
        {
            "schema_version",
            "artifact_root",
            "crop_policy",
            "successor_population",
            "identity_free_auxiliary_population",
            "terminal_exclusions",
            "coverage",
        },
        "face-visible successor inventory",
    )
    if (
        inventory["schema_version"] != INVENTORY_SCHEMA
        or bundle["inventory_sha256"] != content_sha256(inventory)
        or inventory["crop_policy"].get("recrop_permitted") is not False
    ):
        raise FaceVisibleInventoryError("successor inventory digest or crop policy differs")
    for source_digest in bundle["source_binding"].values():
        _require_sha256(source_digest, "successor source binding")
    populations = (
        inventory["successor_population"],
        inventory["identity_free_auxiliary_population"],
        inventory["terminal_exclusions"],
    )
    if any(not isinstance(rows, list) for rows in populations):
        raise FaceVisibleInventoryError("successor populations must be arrays")
    artifact_root = Path(inventory["artifact_root"])
    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or not artifact_root.is_dir()
        or artifact_root.resolve(strict=True) != artifact_root
    ):
        raise FaceVisibleInventoryError(
            "successor artifact root must be a canonical non-symlink directory"
        )
    all_records: list[dict[str, Any]] = []
    for population_name, records in zip(
        ("successor", "auxiliary", "terminal"), populations, strict=True
    ):
        tokens = []
        for record in records:
            _validate_inventory_record(
                record,
                population_name=population_name,
                verify_artifacts=verify_artifacts,
                artifact_root=artifact_root,
            )
            tokens.append(record["sample_token"])
            all_records.append(record)
        if tokens != sorted(tokens):
            raise FaceVisibleInventoryError(
                f"{population_name} population must be sample-token sorted"
            )
    tokens = [row["sample_token"] for row in all_records]
    if len(tokens) != len(set(tokens)):
        raise FaceVisibleInventoryError("successor populations overlap")
    coverage = inventory["coverage"]
    if (
        coverage.get("route_plan_sample_count") != len(tokens)
        or coverage.get("successor_sample_count") != len(populations[0])
        or coverage.get("identity_free_auxiliary_sample_count")
        != len(populations[1])
        or coverage.get("terminal_exclusion_count") != len(populations[2])
        or coverage.get("state_counts")
        != dict(sorted(Counter(row["state"] for row in all_records).items()))
    ):
        raise FaceVisibleInventoryError("successor inventory coverage differs")
    return bundle


def _validated_face_sources(
    route_plan_bundle: object,
    face_overlay_bundle: object,
    face_protocol_bundle: object,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    route = validate_full128_route_plan_bundle(route_plan_bundle, verify_files=False)
    overlay = validate_face_eligibility_overlay_bundle(face_overlay_bundle)
    protocol = validate_face_identity_protocol_bundle(face_protocol_bundle)
    if (
        overlay["overlay"]["source_route_plan_sha256"] != route["plan_sha256"]
        or overlay["overlay"]["source_route_plan_bundle_sha256"]
        != route["bundle_sha256"]
        or protocol["protocol"]["source_route_plan_sha256"] != route["plan_sha256"]
        or protocol["protocol"]["source_route_plan_bundle_sha256"]
        != route["bundle_sha256"]
        or protocol["protocol"]["source_face_overlay_sha256"]
        != overlay["overlay_sha256"]
        or protocol["protocol"]["source_face_overlay_bundle_sha256"]
        != overlay["bundle_sha256"]
    ):
        raise FaceVisibleInventoryError("face successor source bindings differ")
    route_tokens = {row["sample_token"] for row in route["plan"]["records"]}
    overlay_tokens = {
        row["sample_token"] for row in overlay["overlay"]["records"]
    }
    if route_tokens != overlay_tokens:
        raise FaceVisibleInventoryError(
            "face overlay is not complete for the selected route plan"
        )
    expected_protocol_tokens = {
        row["sample_token"]
        for row in overlay["overlay"]["records"]
        if row["gallery_query_eligible"]
        and row["dataset_name"] in {"dogfacenet224", "mpdd"}
    }
    protocol_tokens = {
        row["sample_token"]
        for row in protocol["protocol"]["sample_assignments"]
    }
    if protocol_tokens != expected_protocol_tokens:
        missing = sorted(expected_protocol_tokens - protocol_tokens)
        extra = sorted(protocol_tokens - expected_protocol_tokens)
        raise FaceVisibleInventoryError(
            "face protocol population differs from metadata eligibility; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return route, overlay, protocol


def _validated_face_sources_v2(
    route_plan_bundle: object,
    face_overlay_bundle: object,
    face_protocol_v2_bundle: object,
    gallery_query_panel_bundle: object,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    route = validate_full128_route_plan_bundle(route_plan_bundle, verify_files=False)
    overlay = validate_face_eligibility_overlay_bundle(face_overlay_bundle)
    protocol = validate_face_identity_protocol_v2_bundle(face_protocol_v2_bundle)
    governance = validate_face_gallery_query_panel_bundle(
        gallery_query_panel_bundle
    )
    protocol_value = protocol["protocol"]
    governance_value = governance["panel"]
    if (
        overlay["overlay"]["source_route_plan_sha256"] != route["plan_sha256"]
        or overlay["overlay"]["source_route_plan_bundle_sha256"]
        != route["bundle_sha256"]
        or protocol_value["source_route_plan_sha256"] != route["plan_sha256"]
        or protocol_value["source_route_plan_bundle_sha256"]
        != route["bundle_sha256"]
        or protocol_value["source_face_overlay_sha256"]
        != overlay["overlay_sha256"]
        or protocol_value["source_face_overlay_bundle_sha256"]
        != overlay["bundle_sha256"]
        or governance_value["source_protocol_sha256"]
        != protocol["protocol_sha256"]
        or governance_value["source_protocol_bundle_sha256"]
        != protocol["bundle_sha256"]
    ):
        raise FaceVisibleInventoryError(
            "governance v2 successor source bindings differ"
        )
    route_tokens = {row["sample_token"] for row in route["plan"]["records"]}
    overlay_rows = overlay["overlay"]["records"]
    if route_tokens != {row["sample_token"] for row in overlay_rows}:
        raise FaceVisibleInventoryError(
            "face overlay is not complete for the selected route plan"
        )
    expected_protocol_tokens = {
        row["sample_token"]
        for row in overlay_rows
        if row["gallery_query_eligible"]
        and row["dataset_name"] in {"dogfacenet224", "mpdd"}
    }
    protocol_rows = protocol_value["sample_assignments"]
    protocol_route_tokens = {row["route_sample_token"] for row in protocol_rows}
    protocol_public_tokens = {row["public_sample_token"] for row in protocol_rows}
    if (
        len(protocol_route_tokens) != len(protocol_rows)
        or len(protocol_public_tokens) != len(protocol_rows)
        or protocol_route_tokens != expected_protocol_tokens
    ):
        missing = sorted(expected_protocol_tokens - protocol_route_tokens)
        extra = sorted(protocol_route_tokens - expected_protocol_tokens)
        raise FaceVisibleInventoryError(
            "face protocol v2 route/public mapping differs from metadata eligibility; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return route, overlay, protocol, governance


def _validate_assembly(
    value: object, *, validation_workers: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    _exact_keys(value, _ASSEMBLY_FIELDS, "Full128 materialization assembly")
    assembly = dict(value)
    if assembly["schema_version"] != ASSEMBLY_SCHEMA:
        raise FaceVisibleInventoryError("Full128 materialization assembly schema differs")
    payload = {key: item for key, item in assembly.items() if key != "assembly_sha256"}
    if assembly["assembly_sha256"] != content_sha256(payload):
        raise FaceVisibleInventoryError("Full128 materialization assembly digest differs")
    inventory = validate_full128_experiment_inventory_bundle(
        assembly["inventory_bundle"], validation_workers=validation_workers
    )
    if (
        assembly["sample_count"] != len(inventory["inventory"]["records"])
        or assembly["unified_full_split"] != inventory["split_bundle"]
    ):
        raise FaceVisibleInventoryError(
            "Full128 materialization assembly inventory differs"
        )
    return assembly, inventory


def _panel_uses(
    rows: Sequence[Mapping[str, Any]], *, dataset: str, role: str
) -> dict[str, dict[str, dict[str, str | None]]]:
    result = {row["sample_token"]: {} for row in rows}
    if role not in EVALUATION_SCOPES:
        for row in rows:
            result[row["sample_token"]] = {
                str(k): {"use": "TRAINING_ONLY", "reason": "FIT_SCOPE_NOT_EVALUATED"}
                for k in ENROLLMENT_KS
            }
        return result
    for k in ENROLLMENT_KS:
        if dataset == "mpdd":
            gallery = [row for row in rows if row["publisher_split"] == "gallery"]
            query = [row for row in rows if row["publisher_split"] == "query"]
            selected_gallery = gallery[:k] if len(gallery) >= k and query else []
        else:
            selected_gallery = list(rows[:k]) if len(rows) > k else []
            query = list(rows[k:]) if selected_gallery else []
        gallery_tokens = {row["sample_token"] for row in selected_gallery}
        query_tokens = {row["sample_token"] for row in query}
        for row in rows:
            token = row["sample_token"]
            if token in gallery_tokens:
                use, reason = "GALLERY", None
            elif token in query_tokens:
                use, reason = "QUERY", None
            else:
                use, reason = "TERMINAL_EXCLUSION", "IDENTITY_NOT_K_FEASIBLE"
            result[token][str(k)] = {"use": use, "reason": reason}
    return result


def _reject_panel_leakage(records: Sequence[Mapping[str, Any]]) -> None:
    for scope in EVALUATION_SCOPES:
        scoped = [row for row in records if row["scope"] == scope]
        for k in ENROLLMENT_KS:
            query = [
                row
                for row in scoped
                if row["uses_by_enrollment_k"][str(k)]["use"] == "QUERY"
            ]
            gallery = [
                row
                for row in scoped
                if row["uses_by_enrollment_k"][str(k)]["use"] == "GALLERY"
            ]
            for field in (
                "sample_token",
                "duplicate_component",
                "effective_content_guard",
            ):
                if {row[field] for row in query} & {row[field] for row in gallery}:
                    raise FaceVisibleInventoryError(
                        f"face-visible panel query/gallery leakage: {scope}/K{k}/{field}"
                    )


def _joined_record(
    source: Mapping[str, Any],
    face: Mapping[str, Any],
    material: Mapping[str, Any],
    *,
    assignment: Mapping[str, Any] | None,
    panel_record: Mapping[str, Any] | None,
    artifact_root: Path,
) -> dict[str, Any]:
    token = source["sample_token"]
    if (
        face["source_record_sha256"] != source["record_sha256"]
        or material["dataset_name"] != source["dataset_name"]
        or material["original_source_sha256"] != source["source_sha256"]
    ):
        raise FaceVisibleInventoryError(
            f"joined source binding differs for sample {token}"
        )
    governance_v2 = assignment is not None and "route_sample_token" in assignment
    if assignment is not None:
        if governance_v2:
            assignment_differs = (
                assignment["route_sample_token"] != token
                or assignment["face_record_sha256"] != face["record_sha256"]
                or assignment["registered_identity_id"]
                != face["registered_identity_id"]
                or assignment["dataset_name"] != source["dataset_name"]
            )
        else:
            assignment_differs = (
                assignment["source_record_sha256"] != source["record_sha256"]
                or assignment["face_record_sha256"] != face["record_sha256"]
                or assignment["duplicate_component"]
                != source["duplicate_component"]
            )
        if assignment_differs:
            raise FaceVisibleInventoryError(
                f"face protocol sample binding differs for sample {token}"
            )
    if panel_record is not None and governance_v2 and (
        panel_record["sample_token"] != assignment["route_sample_token"]
        or panel_record["public_sample_token"]
        != assignment["public_sample_token"]
        or panel_record["public_identity_token"]
        != assignment["public_identity_token"]
        or panel_record["protocol_sample_record_sha256"]
        != assignment["record_sha256"]
    ):
        raise FaceVisibleInventoryError(
            f"authoritative panel mapping differs for sample {token}"
        )
    material_usable = material["full_status"] in {"USABLE", "REVIEW"}
    face_eligible = face["status"] == "ELIGIBLE"
    if material_usable and not material["crop_artifacts_present"]:
        raise FaceVisibleInventoryError("usable Full128 row is missing materialized crops")
    if governance_v2 and assignment["cross_identity_unsafe"]:
        state = "TERMINAL_EXCLUSION"
        reason = "CROSS_IDENTITY_UNSAFE_COMPONENT"
    elif not material_usable:
        state = "TERMINAL_EXCLUSION"
        reason = f"FULL128_{material['full_status']}"
    elif not face_eligible:
        state = "TERMINAL_EXCLUSION"
        reason = f"FACE_{face['status']}"
    elif assignment is None and material["identity_evidence_kind"] != "NONE":
        state = "TERMINAL_EXCLUSION"
        reason = "OUTSIDE_REGISTERED_FACE_PROTOCOL"
    else:
        state = "USABLE"
        reason = None
    artifact = None
    if material_usable:
        rgb = artifact_root / material["full_rgb_path"]
        mask = artifact_root / material["full_mask_path"]
        artifact = {
            "full_rgb_path": os.fspath(rgb),
            "full_rgb_sha256": material["full_rgb_sha256"],
            "full_mask_path": os.fspath(mask),
            "full_mask_sha256": material["full_mask_sha256"],
            "crop_record_sha256": material["crop_record_sha256"],
            "full_segment_cache_sha256": material["full_segment_cache_sha256"],
        }
    payload = {
        "schema_version": INVENTORY_RECORD_SCHEMA,
        "sample_token": token,
        "dataset_name": source["dataset_name"],
        "registered_identity_id": face["registered_identity_id"],
        "identity_evidence_kind": material["identity_evidence_kind"],
        "face_status": face["status"],
        "face_evidence_kind": face["evidence_kind"],
        "protocol_scope": None if assignment is None else assignment["role"],
        "gradient_eligible": bool(
            assignment is not None and assignment["gradient_eligible"]
        ),
        "panel_record_sha256": (
            None if panel_record is None else panel_record["record_sha256"]
        ),
        "state": state,
        "terminal_reason": reason,
        "source_record_sha256": source["record_sha256"],
        "face_record_sha256": face["record_sha256"],
        "materialization_record_sha256": material["source_observation_sha256"],
        "artifact": artifact,
        "recropped": False,
    }
    return {**payload, "record_sha256": content_sha256(payload)}


def _validate_panel_record(value: object) -> None:
    expected = {
        "schema_version",
        "sample_token",
        "registered_identity_id",
        "dataset_name",
        "scope",
        "publisher_split",
        "duplicate_component",
        "effective_content_guard",
        "uses_by_enrollment_k",
        "score_inputs_used",
        "record_sha256",
    }
    _exact_keys(value, expected, "face-visible panel record")
    row = value
    if row["schema_version"] != PANEL_RECORD_SCHEMA or row["score_inputs_used"] is not False:
        raise FaceVisibleInventoryError("face-visible panel record contract differs")
    for field in (
        "sample_token",
        "duplicate_component",
        "effective_content_guard",
        "record_sha256",
    ):
        _require_sha256(row[field], field)
    if row["scope"] not in (*EVALUATION_SCOPES, "FIT"):
        raise FaceVisibleInventoryError("face-visible panel scope differs")
    uses = row["uses_by_enrollment_k"]
    if not isinstance(uses, Mapping) or set(uses) != {str(k) for k in ENROLLMENT_KS}:
        raise FaceVisibleInventoryError("face-visible panel enrollment-K uses differ")
    for use in uses.values():
        _exact_keys(use, {"use", "reason"}, "face-visible panel use")
        if use["use"] not in {
            "GALLERY",
            "QUERY",
            "TERMINAL_EXCLUSION",
            "TRAINING_ONLY",
        }:
            raise FaceVisibleInventoryError("face-visible panel use differs")
        if (use["use"] in {"GALLERY", "QUERY"}) != (use["reason"] is None):
            raise FaceVisibleInventoryError("face-visible panel use reason differs")
    _validate_record_digest(row, "face-visible panel record")


def _validate_authoritative_panel_record(value: object) -> None:
    expected = {
        "schema_version",
        "sample_token",
        "public_sample_token",
        "public_identity_token",
        "registered_identity_id",
        "dataset_name",
        "scope",
        "publisher_split",
        "duplicate_component",
        "effective_content_guard",
        "dependency_component_sha256",
        "protocol_sample_record_sha256",
        "governance_cohort_record_sha256",
        "authoritative_cohort_member",
        "uses_by_enrollment_k",
        "score_inputs_used",
        "record_sha256",
    }
    _exact_keys(value, expected, "authoritative face-visible panel record")
    row = value
    if (
        row["schema_version"] != AUTHORITATIVE_PANEL_RECORD_SCHEMA
        or row["score_inputs_used"] is not False
        or not isinstance(row["authoritative_cohort_member"], bool)
        or row["duplicate_component"] != row["dependency_component_sha256"]
        or row["effective_content_guard"] != row["public_sample_token"]
    ):
        raise FaceVisibleInventoryError(
            "authoritative face-visible panel record contract differs"
        )
    for field in (
        "sample_token",
        "public_sample_token",
        "public_identity_token",
        "duplicate_component",
        "dependency_component_sha256",
        "protocol_sample_record_sha256",
        "record_sha256",
    ):
        _require_sha256(row[field], field)
    if row["governance_cohort_record_sha256"] is not None:
        _require_sha256(
            row["governance_cohort_record_sha256"],
            "governance cohort record",
        )
    if row["authoritative_cohort_member"] is not (
        row["governance_cohort_record_sha256"] is not None
    ):
        raise FaceVisibleInventoryError(
            "authoritative panel cohort membership binding differs"
        )
    if row["scope"] not in (*EVALUATION_SCOPES, "FIT"):
        raise FaceVisibleInventoryError("authoritative panel scope differs")
    uses = row["uses_by_enrollment_k"]
    if not isinstance(uses, Mapping) or set(uses) != {str(k) for k in ENROLLMENT_KS}:
        raise FaceVisibleInventoryError("authoritative panel K uses differ")
    for use in uses.values():
        _exact_keys(use, {"use", "reason"}, "authoritative panel use")
        if use["use"] not in {
            "GALLERY",
            "QUERY",
            "TERMINAL_EXCLUSION",
            "TRAINING_ONLY",
        }:
            raise FaceVisibleInventoryError("authoritative panel use differs")
        if (use["use"] in {"GALLERY", "QUERY"}) != (use["reason"] is None):
            raise FaceVisibleInventoryError("authoritative panel use reason differs")
    _validate_record_digest(row, "authoritative face-visible panel record")


def _validate_authoritative_nested_memberships(
    records: Sequence[Mapping[str, Any]],
) -> None:
    by_identity: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_identity[row["public_identity_token"]].append(row)
    for identity_rows in by_identity.values():
        cohort_rows = [
            row for row in identity_rows if row["authoritative_cohort_member"]
        ]
        if not cohort_rows:
            if any(
                use["use"] in {"QUERY", "GALLERY"}
                for row in identity_rows
                for use in row["uses_by_enrollment_k"].values()
            ):
                raise FaceVisibleInventoryError(
                    "non-cohort authoritative panel identity carries Q/G membership"
                )
            continue
        cohort_hashes = {
            row["governance_cohort_record_sha256"] for row in cohort_rows
        }
        if len(cohort_hashes) != 1:
            raise FaceVisibleInventoryError(
                "authoritative panel identity mixes governance cohort records"
            )
        queries_by_k = {
            str(k): [
                row["public_sample_token"]
                for row in cohort_rows
                if row["uses_by_enrollment_k"][str(k)]["use"] == "QUERY"
            ]
            for k in ENROLLMENT_KS
        }
        galleries_by_k = {
            str(k): [
                row["public_sample_token"]
                for row in cohort_rows
                if row["uses_by_enrollment_k"][str(k)]["use"] == "GALLERY"
            ]
            for k in ENROLLMENT_KS
        }
        if not (
            len(queries_by_k["1"]) == 1
            and queries_by_k["1"] == queries_by_k["3"] == queries_by_k["5"]
            and len(galleries_by_k["1"]) == 1
            and len(galleries_by_k["3"]) == 3
            and len(galleries_by_k["5"]) == 5
            and set(galleries_by_k["1"]) < set(galleries_by_k["3"])
            and set(galleries_by_k["3"]) < set(galleries_by_k["5"])
        ):
            raise FaceVisibleInventoryError(
                "authoritative shared-query or nested K memberships differ"
            )


def _validate_authoritative_cohort_mappings(
    value: object, records: Sequence[Mapping[str, Any]]
) -> None:
    if not isinstance(value, list) or not value:
        raise FaceVisibleInventoryError(
            "authoritative mapped cohorts must be a non-empty array"
        )
    by_public = {row["public_sample_token"]: row for row in records}
    identities: list[str] = []
    for cohort in value:
        expected = {
            "schema_version",
            "public_identity_token",
            "registered_identity_id",
            "dataset_name",
            "scope",
            "query_public_sample_token",
            "query_route_sample_token",
            "gallery_public_sample_tokens_by_k",
            "gallery_route_sample_tokens_by_k",
            "governance_cohort_record_sha256",
            "record_sha256",
        }
        _exact_keys(cohort, expected, "authoritative mapped cohort")
        if cohort["schema_version"] != AUTHORITATIVE_COHORT_SCHEMA:
            raise FaceVisibleInventoryError(
                "authoritative mapped cohort schema differs"
            )
        for field in (
            "public_identity_token",
            "query_public_sample_token",
            "query_route_sample_token",
            "governance_cohort_record_sha256",
            "record_sha256",
        ):
            _require_sha256(cohort[field], field)
        query = by_public.get(cohort["query_public_sample_token"])
        if (
            query is None
            or query["sample_token"] != cohort["query_route_sample_token"]
            or query["public_identity_token"] != cohort["public_identity_token"]
            or query["registered_identity_id"]
            != cohort["registered_identity_id"]
            or query["dataset_name"] != cohort["dataset_name"]
            or query["scope"] != cohort["scope"]
            or query["governance_cohort_record_sha256"]
            != cohort["governance_cohort_record_sha256"]
        ):
            raise FaceVisibleInventoryError(
                "authoritative mapped cohort query differs"
            )
        public_by_k = cohort["gallery_public_sample_tokens_by_k"]
        route_by_k = cohort["gallery_route_sample_tokens_by_k"]
        if (
            not isinstance(public_by_k, Mapping)
            or not isinstance(route_by_k, Mapping)
            or set(public_by_k) != {"K1", "K3", "K5"}
            or set(route_by_k) != {"K1", "K3", "K5"}
        ):
            raise FaceVisibleInventoryError(
                "authoritative mapped cohort K fields differ"
            )
        for k in ENROLLMENT_KS:
            key = f"K{k}"
            public_tokens = public_by_k[key]
            route_tokens = route_by_k[key]
            if (
                not isinstance(public_tokens, list)
                or not isinstance(route_tokens, list)
                or len(public_tokens) != k
                or len(route_tokens) != k
                or len(public_tokens) != len(set(public_tokens))
                or len(route_tokens) != len(set(route_tokens))
            ):
                raise FaceVisibleInventoryError(
                    "authoritative mapped cohort K membership differs"
                )
            for public, route in zip(public_tokens, route_tokens, strict=True):
                row = by_public.get(public)
                if (
                    row is None
                    or row["sample_token"] != route
                    or row["public_identity_token"]
                    != cohort["public_identity_token"]
                    or row["uses_by_enrollment_k"][str(k)]["use"] != "GALLERY"
                ):
                    raise FaceVisibleInventoryError(
                        "authoritative public-to-route gallery mapping differs"
                    )
        if (
            public_by_k["K1"] != public_by_k["K3"][:1]
            or public_by_k["K3"] != public_by_k["K5"][:3]
            or route_by_k["K1"] != route_by_k["K3"][:1]
            or route_by_k["K3"] != route_by_k["K5"][:3]
        ):
            raise FaceVisibleInventoryError(
                "authoritative mapped cohort galleries are not exactly nested"
            )
        _validate_record_digest(cohort, "authoritative mapped cohort")
        identities.append(cohort["public_identity_token"])
    if identities != sorted(set(identities)):
        raise FaceVisibleInventoryError(
            "authoritative mapped cohorts must be identity sorted and unique"
        )


def _validate_inventory_record(
    value: object,
    *,
    population_name: str,
    verify_artifacts: bool,
    artifact_root: Path,
) -> None:
    expected = {
        "schema_version",
        "sample_token",
        "dataset_name",
        "registered_identity_id",
        "identity_evidence_kind",
        "face_status",
        "face_evidence_kind",
        "protocol_scope",
        "gradient_eligible",
        "panel_record_sha256",
        "state",
        "terminal_reason",
        "source_record_sha256",
        "face_record_sha256",
        "materialization_record_sha256",
        "artifact",
        "recropped",
        "record_sha256",
    }
    _exact_keys(value, expected, "face-visible successor record")
    row = value
    if row["schema_version"] != INVENTORY_RECORD_SCHEMA or row["recropped"] is not False:
        raise FaceVisibleInventoryError("face-visible successor record contract differs")
    for field in (
        "sample_token",
        "source_record_sha256",
        "face_record_sha256",
        "materialization_record_sha256",
        "record_sha256",
    ):
        _require_sha256(row[field], field)
    if population_name == "successor" and (
        row["identity_evidence_kind"] != "REGISTERED"
        or row["protocol_scope"] is None
        or row["panel_record_sha256"] is None
    ):
        raise FaceVisibleInventoryError("successor population identity contract differs")
    if population_name == "auxiliary" and (
        row["identity_evidence_kind"] != "NONE"
        or row["registered_identity_id"] is not None
        or row["protocol_scope"] is not None
        or row["panel_record_sha256"] is not None
    ):
        raise FaceVisibleInventoryError(
            "identity-free auxiliary population carries identity protocol data"
        )
    if population_name == "terminal" and (
        row["identity_evidence_kind"] == "NONE"
        or row["protocol_scope"]
        not in {None, "EXCLUDED_UNSAFE_COMPONENT"}
        or row["panel_record_sha256"] is not None
    ):
        raise FaceVisibleInventoryError("terminal population lane differs")
    if row["protocol_scope"] == "EXCLUDED_UNSAFE_COMPONENT" and (
        row["state"] != "TERMINAL_EXCLUSION"
        or row["terminal_reason"] != "CROSS_IDENTITY_UNSAFE_COMPONENT"
    ):
        raise FaceVisibleInventoryError(
            "cross-identity-unsafe sample was not terminally excluded"
        )
    if row["panel_record_sha256"] is not None:
        _require_sha256(row["panel_record_sha256"], "panel record")
    if row["gradient_eligible"] is not (row["protocol_scope"] == "FIT"):
        raise FaceVisibleInventoryError("successor gradient eligibility differs")
    if row["state"] not in {"USABLE", "TERMINAL_EXCLUSION"}:
        raise FaceVisibleInventoryError("successor record state differs")
    if (row["state"] == "USABLE") != (row["terminal_reason"] is None):
        raise FaceVisibleInventoryError("successor terminal reason differs")
    artifact = row["artifact"]
    if row["state"] == "USABLE" and artifact is None:
        raise FaceVisibleInventoryError("usable successor record requires Full128 artifacts")
    if artifact is not None:
        _exact_keys(
            artifact,
            {
                "full_rgb_path",
                "full_rgb_sha256",
                "full_mask_path",
                "full_mask_sha256",
                "crop_record_sha256",
                "full_segment_cache_sha256",
            },
            "successor Full128 artifact binding",
        )
        for field in (
            "full_rgb_sha256",
            "full_mask_sha256",
            "crop_record_sha256",
            "full_segment_cache_sha256",
        ):
            _require_sha256(artifact[field], field)
        for field in ("full_rgb_path", "full_mask_path"):
            path = Path(artifact[field])
            if not path.is_absolute() or not path.is_relative_to(artifact_root):
                raise FaceVisibleInventoryError(
                    "successor Full128 artifact must remain under artifact root"
                )
        if verify_artifacts:
            _verify_artifact(Path(artifact["full_rgb_path"]), artifact["full_rgb_sha256"])
            _verify_artifact(Path(artifact["full_mask_path"]), artifact["full_mask_sha256"])
    _validate_record_digest(row, "face-visible successor record")


def _verify_artifact(path: Path, expected_sha256: str) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise FaceVisibleInventoryError("Full128 artifact path must be absolute and direct")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FaceVisibleInventoryError(f"Full128 artifact cannot be opened: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_ARTIFACT_BYTES:
            raise FaceVisibleInventoryError("Full128 artifact size or type differs")
        observed = 0
        while chunk := os.read(descriptor, min(1_048_576, _MAX_ARTIFACT_BYTES + 1 - observed)):
            observed += len(chunk)
            if observed > _MAX_ARTIFACT_BYTES:
                raise FaceVisibleInventoryError("Full128 artifact exceeds byte limit")
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or digest.hexdigest() != expected_sha256:
        raise FaceVisibleInventoryError("Full128 artifact content binding differs")


def _validate_record_digest(row: Mapping[str, Any], label: str) -> None:
    payload = {key: item for key, item in row.items() if key != "record_sha256"}
    if row["record_sha256"] != content_sha256(payload):
        raise FaceVisibleInventoryError(f"{label} digest differs")


def _exact_keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FaceVisibleInventoryError(f"{label} fields differ")


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FaceVisibleInventoryError(f"{label} must be lowercase SHA-256")


# Explicit long-form alias used by workflow callers and artifact documentation.
build_metadata_face_eligible_successor_inventory = build_face_visible_successor_inventory
validate_metadata_face_eligible_successor_inventory_bundle = (
    validate_face_visible_successor_inventory_bundle
)

__all__ = [
    "AUTHORITATIVE_PANEL_SCHEMA",
    "BUNDLE_SCHEMA",
    "ENROLLMENT_KS",
    "EVALUATION_SCOPES",
    "INVENTORY_SCHEMA",
    "PANEL_SCHEMA",
    "FaceVisibleInventoryError",
    "build_authoritative_face_visible_panel",
    "build_face_visible_successor_inventory",
    "build_face_visible_successor_inventory_v2",
    "build_metadata_face_eligible_successor_inventory",
    "build_score_blind_face_visible_panel",
    "validate_face_visible_successor_inventory_bundle",
    "validate_metadata_face_eligible_successor_inventory_bundle",
    "validate_score_blind_face_visible_panel",
]
