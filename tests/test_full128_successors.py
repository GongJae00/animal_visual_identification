from __future__ import annotations

import hashlib
import json
import mmap
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest
import torch
from torch import nn

from evaluation.full_segment.full128_analysis import (
    RepresentationTraceError,
    build_executed_representation_trace_manifest,
    build_public_representation_analysis,
    build_representation_trace_manifest,
    sanitize_representation_trace_manifest,
    validate_executed_representation_trace_manifest,
    validate_public_representation_analysis,
    validate_public_representation_trace_manifest,
    validate_representation_trace_manifest,
)
from evaluation.full_segment.full128_successors import (
    Full128SuccessorEvaluationError,
    build_authoritative_fixed_evaluation_panel,
    build_score_blind_fixed_evaluation_panel,
    build_successor_embedding_cache_descriptor,
    evaluate_successor_family,
    open_successor_embedding_cache,
    paired_identity_cluster_bootstrap,
    sanitize_successor_evaluation_report,
)
from foundation.provenance import content_sha256
from embedding.methods.full_segment import face_visible
from embedding.methods.full_segment.face_visible import (
    FaceVisibleInventoryError,
    build_authoritative_face_visible_panel,
    build_face_visible_successor_inventory,
    build_face_visible_successor_inventory_v2,
    build_score_blind_face_visible_panel,
    validate_face_visible_successor_inventory_bundle,
)
from embedding.methods.full_segment.preparation.materialization import ASSEMBLY_SCHEMA
from embedding.methods.full_segment.models.successor_models import (
    Dinov2OccupancyProbe128,
    SpatialScorer128,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _identity(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"full128-successor-test:{label}"))


def _artifact(tmp_path: Path, label: str) -> tuple[str, str]:
    path = tmp_path / f"{label}.bin"
    path.write_bytes(f"artifact:{label}".encode("ascii"))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(tmp_path: Path) -> tuple[dict[str, Any], ...]:
    route_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    material_rows: list[dict[str, Any]] = []

    def add(
        label: str,
        *,
        dataset: str,
        identity: str | None,
        role: str | None,
        split: str,
        kind: str = "REGISTERED",
        face_status: str = "ELIGIBLE",
        full_status: str = "USABLE",
    ) -> None:
        token = _sha(f"sample:{label}")
        source_record = _sha(f"route-record:{label}")
        duplicate = _sha(f"duplicate:{label}")
        route_rows.append(
            {
                "sample_token": token,
                "dataset_name": dataset,
                "source_sha256": _sha(f"source:{label}"),
                "duplicate_component": duplicate,
                "record_sha256": source_record,
            }
        )
        face_payload = {
            "sample_token": token,
            "dataset_name": dataset,
            "source_record_sha256": source_record,
            "registered_identity_id": identity,
            "status": face_status,
            "evidence_kind": "PUBLISHER_NATIVE_FACE_CROP"
            if face_status == "ELIGIBLE"
            else "NONE",
            "publisher_split": split,
        }
        overlay_rows.append(
            {
                **face_payload,
                "gallery_query_eligible": identity is not None
                and dataset in {"dogfacenet224", "mpdd"}
                and face_status == "ELIGIBLE",
                "record_sha256": content_sha256(face_payload),
            }
        )
        if role is not None:
            assignment_payload = {
                "sample_token": token,
                "registered_identity_id": identity,
                "dataset_name": dataset,
                "publisher_split": split,
                "role": role,
                "gradient_eligible": role == "FIT",
                "duplicate_component": duplicate,
                "source_record_sha256": source_record,
                "face_record_sha256": content_sha256(face_payload),
            }
            assignments.append(
                {
                    **assignment_payload,
                    "record_sha256": content_sha256(assignment_payload),
                }
            )
        rgb_path, rgb_sha = _artifact(tmp_path, f"{label}-rgb")
        mask_path, mask_sha = _artifact(tmp_path, f"{label}-mask")
        material_rows.append(
            {
                "sample_token": token,
                "dataset_name": dataset,
                "identity_evidence_kind": kind,
                "original_source_sha256": _sha(f"source:{label}"),
                "source_observation_sha256": _sha(f"observation:{label}"),
                "full_status": full_status,
                "crop_artifacts_present": full_status in {"USABLE", "REVIEW"},
                "full_rgb_path": Path(rgb_path).name,
                "full_rgb_sha256": rgb_sha,
                "full_mask_path": Path(mask_path).name,
                "full_mask_sha256": mask_sha,
                "crop_record_sha256": _sha(f"crop:{label}"),
                "full_segment_cache_sha256": _sha(f"cache:{label}"),
            }
        )

    for scope in ("DEV", "CAL", "EXPOSED_DIAGNOSTIC"):
        for identity_index in range(2):
            identity = _identity(f"{scope}:{identity_index}")
            dataset = "mpdd" if scope == "EXPOSED_DIAGNOSTIC" else "dogfacenet224"
            for sample_index in range(6):
                split = (
                    ("query" if sample_index == 5 else "gallery")
                    if dataset == "mpdd"
                    else "train"
                )
                add(
                    f"{scope}-{identity_index}-{sample_index}",
                    dataset=dataset,
                    identity=identity,
                    role=scope,
                    split=split,
                )
    add(
        "auxiliary",
        dataset="dogflw",
        identity=None,
        role=None,
        split="train",
        kind="NONE",
    )
    add(
        "terminal",
        dataset="sibetan",
        identity=_identity("terminal"),
        role=None,
        split="unassigned",
        face_status="UNAVAILABLE",
    )
    route_rows.sort(key=lambda row: row["sample_token"])
    overlay_rows.sort(key=lambda row: row["sample_token"])
    assignments.sort(key=lambda row: row["sample_token"])
    material_rows.sort(key=lambda row: row["sample_token"])
    route = {
        "plan_sha256": _sha("route-plan"),
        "bundle_sha256": _sha("route-bundle"),
        "plan": {"records": route_rows},
    }
    overlay = {
        "overlay_sha256": _sha("overlay"),
        "bundle_sha256": _sha("overlay-bundle"),
        "overlay": {
            "source_route_plan_sha256": route["plan_sha256"],
            "source_route_plan_bundle_sha256": route["bundle_sha256"],
            "records": overlay_rows,
        },
    }
    protocol = {
        "protocol_sha256": _sha("protocol"),
        "bundle_sha256": _sha("protocol-bundle"),
        "protocol": {
            "source_route_plan_sha256": route["plan_sha256"],
            "source_route_plan_bundle_sha256": route["bundle_sha256"],
            "source_face_overlay_sha256": overlay["overlay_sha256"],
            "source_face_overlay_bundle_sha256": overlay["bundle_sha256"],
            "sample_assignments": assignments,
        },
    }
    full_inventory = {
        "artifact_root": str(tmp_path),
        "bundle_sha256": _sha("full-inventory-bundle"),
        "inventory_sha256": _sha("full-inventory"),
        "split_bundle": {"fixture": True},
        "inventory": {"records": material_rows},
    }
    assembly_payload = {
        "schema_version": ASSEMBLY_SCHEMA,
        "plan_sha256": route["plan_sha256"],
        "sample_count": len(material_rows),
        "allocation_name": "fixture",
        "topology_report": {},
        "unified_full_split": full_inventory["split_bundle"],
        "inventory_request": {},
        "inventory_bundle": full_inventory,
    }
    assembly = {
        **assembly_payload,
        "assembly_sha256": content_sha256(assembly_payload),
    }
    return route, overlay, protocol, assembly, full_inventory


@pytest.fixture
def successor_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], ...]:
    route, overlay, protocol, assembly, full_inventory = _sources(tmp_path)
    monkeypatch.setattr(
        face_visible,
        "validate_full128_route_plan_bundle",
        lambda value, *, verify_files: value,
    )
    monkeypatch.setattr(
        face_visible, "validate_face_eligibility_overlay_bundle", lambda value: value
    )
    monkeypatch.setattr(
        face_visible, "validate_face_identity_protocol_bundle", lambda value: value
    )
    monkeypatch.setattr(
        face_visible,
        "validate_full128_experiment_inventory_bundle",
        lambda value, *, validation_workers: value,
    )
    panel = build_score_blind_face_visible_panel(route, overlay, protocol)
    inventory = build_face_visible_successor_inventory(
        route_plan_bundle=route,
        materialization_assembly=assembly,
        face_overlay_bundle=overlay,
        face_protocol_bundle=protocol,
        fixed_panel=panel,
    )
    return route, overlay, protocol, assembly, full_inventory, panel, inventory


def test_inventory_is_complete_deterministic_and_separates_auxiliary(
    successor_sources: tuple[dict[str, Any], ...],
) -> None:
    route, overlay, protocol, assembly, _, panel, inventory = successor_sources
    assert validate_face_visible_successor_inventory_bundle(inventory) == inventory
    populations = inventory["inventory"]
    assert len(populations["identity_free_auxiliary_population"]) == 1
    auxiliary = populations["identity_free_auxiliary_population"][0]
    assert auxiliary["identity_evidence_kind"] == "NONE"
    assert auxiliary["registered_identity_id"] is None
    assert auxiliary["protocol_scope"] is None
    assert all(
        row["recropped"] is False
        for rows in (
            populations["successor_population"],
            populations["identity_free_auxiliary_population"],
            populations["terminal_exclusions"],
        )
        for row in rows
    )
    assert populations["coverage"]["route_plan_sample_count"] == len(
        route["plan"]["records"]
    )

    reversed_route = deepcopy(route)
    reversed_route["plan"]["records"].reverse()
    reversed_overlay = deepcopy(overlay)
    reversed_overlay["overlay"]["records"].reverse()
    reversed_protocol = deepcopy(protocol)
    reversed_protocol["protocol"]["sample_assignments"].reverse()
    assert (
        build_score_blind_face_visible_panel(
            reversed_route, reversed_overlay, reversed_protocol
        )
        == panel
    )
    assert (
        build_face_visible_successor_inventory(
            route_plan_bundle=reversed_route,
            materialization_assembly=assembly,
            face_overlay_bundle=reversed_overlay,
            face_protocol_bundle=reversed_protocol,
            fixed_panel=panel,
        )
        == inventory
    )


def test_missing_mpdd_materialization_row_and_panel_leakage_fail_closed(
    successor_sources: tuple[dict[str, Any], ...],
) -> None:
    route, overlay, protocol, assembly, _, _, _ = successor_sources
    broken = deepcopy(assembly)
    rows = broken["inventory_bundle"]["inventory"]["records"]
    missing = next(row for row in rows if row["dataset_name"] == "mpdd")
    rows.remove(missing)
    broken["sample_count"] -= 1
    payload = {key: item for key, item in broken.items() if key != "assembly_sha256"}
    broken["assembly_sha256"] = content_sha256(payload)
    panel = build_score_blind_face_visible_panel(route, overlay, protocol)
    with pytest.raises(FaceVisibleInventoryError, match="materialization population"):
        build_face_visible_successor_inventory(
            route_plan_bundle=route,
            materialization_assembly=broken,
            face_overlay_bundle=overlay,
            face_protocol_bundle=protocol,
            fixed_panel=panel,
        )

    missing_mpdd_protocol = deepcopy(protocol)
    missing_mpdd_protocol["protocol"]["sample_assignments"] = [
        row
        for row in missing_mpdd_protocol["protocol"]["sample_assignments"]
        if row["dataset_name"] != "mpdd"
    ]
    with pytest.raises(FaceVisibleInventoryError, match="protocol population"):
        build_score_blind_face_visible_panel(route, overlay, missing_mpdd_protocol)

    leaked_protocol = deepcopy(protocol)
    same_identity = next(
        row["registered_identity_id"]
        for row in leaked_protocol["protocol"]["sample_assignments"]
        if row["role"] == "DEV"
    )
    identity_rows = [
        row
        for row in leaked_protocol["protocol"]["sample_assignments"]
        if row["registered_identity_id"] == same_identity
    ]
    identity_rows[1]["duplicate_component"] = identity_rows[0]["duplicate_component"]
    with pytest.raises(FaceVisibleInventoryError, match="leakage"):
        build_score_blind_face_visible_panel(route, overlay, leaked_protocol)


def _v2_governance(
    route: dict[str, Any],
    overlay: dict[str, Any],
    protocol_v1: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    face_by_route = {row["sample_token"]: row for row in overlay["overlay"]["records"]}
    v1_rows = protocol_v1["protocol"]["sample_assignments"]
    unsafe_route = next(row["sample_token"] for row in v1_rows if row["role"] == "DEV")
    assignments = []
    for row in v1_rows:
        public = _sha(f"public:{row['sample_token']}")
        payload = {
            "route_sample_token": row["sample_token"],
            "public_sample_token": public,
            "public_identity_token": _sha(
                f"public-identity:{row['registered_identity_id']}"
            ),
            "registered_identity_id": row["registered_identity_id"],
            "dataset_name": row["dataset_name"],
            "publisher_split": row["publisher_split"],
            "role": (
                "EXCLUDED_UNSAFE_COMPONENT"
                if row["sample_token"] == unsafe_route
                else row["role"]
            ),
            "identity_role": row["role"],
            "gradient_eligible": row["role"] == "FIT"
            and row["sample_token"] != unsafe_route,
            "dependency_component_sha256": row["duplicate_component"],
            "cross_identity_unsafe": row["sample_token"] == unsafe_route,
            "token_bridge_record_sha256": _sha(f"bridge:{public}"),
            "face_record_sha256": face_by_route[row["sample_token"]]["record_sha256"],
        }
        assignments.append({**payload, "record_sha256": content_sha256(payload)})
    assignments.sort(key=lambda row: row["public_sample_token"])
    protocol_payload = {
        "source_route_plan_sha256": route["plan_sha256"],
        "source_route_plan_bundle_sha256": route["bundle_sha256"],
        "source_face_overlay_sha256": overlay["overlay_sha256"],
        "source_face_overlay_bundle_sha256": overlay["bundle_sha256"],
        "source_token_bridge_sha256": _sha("v2-token-bridge"),
        "source_exposure_history_sha256": _sha("v2-exposure"),
        "joint_filter_graph_sha256": _sha("v2-graph"),
        "sample_assignments": assignments,
    }
    protocol_sha = content_sha256(protocol_payload)
    protocol = {
        "protocol_sha256": protocol_sha,
        "bundle_sha256": _sha("v2-protocol-bundle"),
        "protocol": {**protocol_payload, "protocol_sha256": protocol_sha},
    }
    by_identity: dict[str, list[dict[str, Any]]] = {}
    for row in assignments:
        if not row["cross_identity_unsafe"]:
            by_identity.setdefault(row["public_identity_token"], []).append(row)
    cohorts = []
    for identity, rows in sorted(by_identity.items()):
        if len(rows) < 6:
            continue
        if rows[0]["dataset_name"] == "mpdd":
            query = next(row for row in rows if row["publisher_split"] == "query")
            galleries = sorted(
                (row for row in rows if row["publisher_split"] == "gallery"),
                key=lambda row: row["public_sample_token"],
            )
        else:
            ordered = sorted(rows, key=lambda row: row["public_sample_token"])
            query, galleries = ordered[0], ordered[1:]
        galleries = galleries[:5]
        cohort_payload = {
            "public_identity_token": identity,
            "registered_identity_id": query["registered_identity_id"],
            "dataset_name": query["dataset_name"],
            "protocol_role": query["identity_role"],
            "query_sample_token": query["public_sample_token"],
            "gallery_sample_tokens_by_k": {
                "K1": [row["public_sample_token"] for row in galleries[:1]],
                "K3": [row["public_sample_token"] for row in galleries[:3]],
                "K5": [row["public_sample_token"] for row in galleries],
            },
        }
        cohorts.append(
            {**cohort_payload, "record_sha256": content_sha256(cohort_payload)}
        )
    governance_payload = {
        "source_protocol_sha256": protocol_sha,
        "source_protocol_bundle_sha256": protocol["bundle_sha256"],
        "source_token_bridge_sha256": protocol_payload["source_token_bridge_sha256"],
        "source_exposure_history_sha256": protocol_payload[
            "source_exposure_history_sha256"
        ],
        "source_joint_filter_graph_sha256": protocol_payload[
            "joint_filter_graph_sha256"
        ],
        "common_k5_feasible_cohort": cohorts,
    }
    governance_sha = content_sha256(governance_payload)
    governance = {
        "panel_sha256": governance_sha,
        "bundle_sha256": _sha("governance-panel-bundle"),
        "panel": {**governance_payload, "panel_sha256": governance_sha},
    }
    return protocol, governance, unsafe_route


def test_governance_v2_is_authoritative_maps_tokens_and_excludes_unsafe(
    successor_sources: tuple[dict[str, Any], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route, overlay, protocol_v1, assembly, _, _, _ = successor_sources
    protocol_v2, governance, unsafe_route = _v2_governance(route, overlay, protocol_v1)
    monkeypatch.setattr(
        face_visible,
        "validate_face_identity_protocol_v2_bundle",
        lambda value: value,
    )
    monkeypatch.setattr(
        face_visible,
        "validate_face_gallery_query_panel_bundle",
        lambda value: value,
    )
    normalized = build_authoritative_face_visible_panel(protocol_v2, governance)
    assert normalized["selection_policy"]["authority"] == (
        "PERSISTED_FACE_GALLERY_QUERY_PANEL"
    )
    assert normalized["source_gallery_query_panel_sha256"] == governance["panel_sha256"]
    public_to_route = {
        row["public_sample_token"]: row["route_sample_token"]
        for row in protocol_v2["protocol"]["sample_assignments"]
    }
    normalized_by_public = {
        row["public_sample_token"]: row for row in normalized["records"]
    }
    unsafe_public = next(
        row["public_sample_token"]
        for row in protocol_v2["protocol"]["sample_assignments"]
        if row["route_sample_token"] == unsafe_route
    )
    assert unsafe_public not in normalized_by_public
    cohort = next(
        row
        for row in governance["panel"]["common_k5_feasible_cohort"]
        if row["protocol_role"] == "CAL"
    )
    query = normalized_by_public[cohort["query_sample_token"]]
    assert query["sample_token"] == public_to_route[cohort["query_sample_token"]]
    assert {query["uses_by_enrollment_k"][str(k)]["use"] for k in (1, 3, 5)} == {
        "QUERY"
    }
    mapped_cohort = next(
        row
        for row in normalized["authoritative_cohorts"]
        if row["public_identity_token"] == cohort["public_identity_token"]
    )
    assert (
        mapped_cohort["query_route_sample_token"]
        == public_to_route[cohort["query_sample_token"]]
    )
    for k in (1, 3, 5):
        observed = {
            public
            for public, row in normalized_by_public.items()
            if row["public_identity_token"] == cohort["public_identity_token"]
            and row["uses_by_enrollment_k"][str(k)]["use"] == "GALLERY"
        }
        assert observed == set(cohort["gallery_sample_tokens_by_k"][f"K{k}"])
        assert (
            mapped_cohort["gallery_public_sample_tokens_by_k"][f"K{k}"]
            == (cohort["gallery_sample_tokens_by_k"][f"K{k}"])
        )
        assert mapped_cohort["gallery_route_sample_tokens_by_k"][f"K{k}"] == [
            public_to_route[public]
            for public in cohort["gallery_sample_tokens_by_k"][f"K{k}"]
        ]

    inventory = build_face_visible_successor_inventory_v2(
        route_plan_bundle=route,
        materialization_assembly=assembly,
        face_overlay_bundle=overlay,
        face_protocol_v2_bundle=protocol_v2,
        gallery_query_panel_bundle=governance,
    )
    assert (
        build_face_visible_successor_inventory(
            route_plan_bundle=route,
            materialization_assembly=assembly,
            face_overlay_bundle=overlay,
            face_protocol_v2_bundle=protocol_v2,
            gallery_query_panel_bundle=governance,
        )
        == inventory
    )
    assert validate_face_visible_successor_inventory_bundle(inventory) == inventory
    unsafe = next(
        row
        for row in inventory["inventory"]["terminal_exclusions"]
        if row["sample_token"] == unsafe_route
    )
    assert unsafe["protocol_scope"] == "EXCLUDED_UNSAFE_COMPONENT"
    assert unsafe["terminal_reason"] == "CROSS_IDENTITY_UNSAFE_COMPONENT"
    assert unsafe["panel_record_sha256"] is None
    effective = build_authoritative_fixed_evaluation_panel(
        inventory, protocol_v2, governance
    )
    assert (
        build_score_blind_fixed_evaluation_panel(
            inventory,
            face_protocol_v2_bundle=protocol_v2,
            gallery_query_panel_bundle=governance,
        )
        == effective
    )
    cal_queries = [
        tuple(row["query_sample_tokens"])
        for row in effective["cohorts"]
        if row["scope"] == "CAL" and row["status"] == "AVAILABLE"
    ]
    assert len(set(cal_queries)) == 1

    material_terminal = deepcopy(inventory)
    cal_gallery_route = public_to_route[cohort["gallery_sample_tokens_by_k"]["K5"][-1]]
    material_row = next(
        row
        for row in material_terminal["inventory"]["successor_population"]
        if row["sample_token"] == cal_gallery_route
    )
    material_row["state"] = "TERMINAL_EXCLUSION"
    material_row["terminal_reason"] = "FULL128_UNUSABLE"
    material_row["record_sha256"] = content_sha256(
        {key: value for key, value in material_row.items() if key != "record_sha256"}
    )
    material_terminal["inventory"]["coverage"]["state_counts"] = {
        "TERMINAL_EXCLUSION": 3,
        "USABLE": material_terminal["inventory"]["coverage"]["route_plan_sample_count"]
        - 3,
    }
    material_terminal["inventory_sha256"] = content_sha256(
        material_terminal["inventory"]
    )
    material_terminal["bundle_sha256"] = content_sha256(
        {
            key: value
            for key, value in material_terminal.items()
            if key != "bundle_sha256"
        }
    )
    unavailable = build_authoritative_fixed_evaluation_panel(
        material_terminal, protocol_v2, governance
    )
    cal_status_by_k = {
        row["enrollment_k"]: row["status"]
        for row in unavailable["cohorts"]
        if row["scope"] == "CAL"
    }
    assert cal_status_by_k == {1: "AVAILABLE", 3: "AVAILABLE", 5: "NOT_AVAILABLE"}

    changed = deepcopy(governance)
    changed["panel"]["common_k5_feasible_cohort"][0]["query_sample_token"] = (
        unsafe_public
    )
    with pytest.raises(FaceVisibleInventoryError, match="unsafe or unknown"):
        build_authoritative_face_visible_panel(protocol_v2, changed)


def test_successor_workflow_help_exposes_v2_governance_inputs() -> None:
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "legacy/version/full128/workflows/evaluate_full128_successors.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--face-protocol-v2" in completed.stdout
    assert "--gallery-query-panel" in completed.stdout


def _cache(
    tmp_path: Path,
    inventory: dict[str, Any],
    panel: dict[str, Any],
    successor_id: str,
) -> Any:
    evaluation_panel = build_score_blind_fixed_evaluation_panel(inventory, panel)
    identities = {
        row["registered_identity_id"]: index
        for index, row in enumerate(evaluation_panel["records"])
    }
    by_token = {row["sample_token"]: row for row in evaluation_panel["records"]}
    matrix = np.zeros(
        (len(evaluation_panel["required_sample_tokens"]), 128), dtype=np.float32
    )
    for index, token in enumerate(evaluation_panel["required_sample_tokens"]):
        matrix[index, identities[by_token[token]["registered_identity_id"]]] = 1.0
    pack = tmp_path / f"{successor_id}.f32le"
    pack.write_bytes(matrix.astype("<f4").tobytes())
    descriptor = build_successor_embedding_cache_descriptor(
        successor_id=successor_id,
        pack_path=pack,
        sample_tokens=evaluation_panel["required_sample_tokens"],
        successor_inventory_bundle_sha256=inventory["bundle_sha256"],
        successor_inventory_sha256=inventory["inventory_sha256"],
        evaluation_panel_sha256=evaluation_panel["panel_sha256"],
        model_manifest_sha256=_sha(f"model:{successor_id}"),
        checkpoint_sha256=_sha(f"checkpoint:{successor_id}"),
        preprocessing_manifest_sha256=_sha(f"preprocessing:{successor_id}"),
        embedding_manifest_sha256=_sha(f"embedding:{successor_id}"),
    )
    return (
        evaluation_panel,
        descriptor,
        open_successor_embedding_cache(
            descriptor,
            successor_inventory_bundle=inventory,
            evaluation_panel=evaluation_panel,
        ),
    )


def test_cache_exactness_gallery_reopen_private_public_and_tamper(
    successor_sources: tuple[dict[str, Any], ...], tmp_path: Path
) -> None:
    *_, panel, inventory = successor_sources
    evaluation_panel, first_descriptor, first = _cache(tmp_path, inventory, panel, "S0")
    _, _, second = _cache(tmp_path, inventory, panel, "S1")
    private, public = evaluate_successor_family(
        successor_inventory_bundle=inventory,
        source_panel=panel,
        caches=(first, second),
        gallery_root=tmp_path / "galleries",
        bootstrap_resamples=30,
        bootstrap_seed=11,
    )
    assert private["dev_selection_receipt"]["selection_scope"] == "DEV_ONLY"
    assert private["scope_interpretation"]["CAL"].endswith("NOT_SELECTION")
    assert private["scope_interpretation"]["EXPOSED_DIAGNOSTIC"].startswith(
        "RETROSPECTIVE_EXPOSED"
    )
    assert private["candidates"][0]["ranked_private_qkv_traces"]
    serialized_private = json.dumps(private, sort_keys=True)
    assert private["schema_version"] == "cvi.full128_successor_private_evaluation.v2"
    assert '"embedding"' not in serialized_private
    assert all(
        binding["reopened_read_only"] and binding["exact_cosine"]
        for candidate in private["candidates"]
        for binding in candidate["gallery_bindings"]
    )
    serialized_public = json.dumps(public, sort_keys=True)
    assert "ranked_private_qkv_traces" not in serialized_public
    assert '"embedding"' not in serialized_public
    assert '"sample_token"' not in serialized_public
    assert sanitize_successor_evaluation_report(private) == public

    arbitrary_public = deepcopy(private)
    candidate = arbitrary_public["candidates"][0]
    candidate["scope_aggregates"][0]["tensor"] = [[1.0, 2.0]]
    candidate_payload = {
        key: item for key, item in candidate.items() if key != "candidate_report_sha256"
    }
    candidate["candidate_report_sha256"] = content_sha256(candidate_payload)
    report_payload = {
        key: item for key, item in arbitrary_public.items() if key != "report_sha256"
    }
    arbitrary_public["report_sha256"] = content_sha256(report_payload)
    with pytest.raises(Full128SuccessorEvaluationError, match="fields differ"):
        sanitize_successor_evaluation_report(arbitrary_public)

    pack = Path(first_descriptor["pack_path"])
    payload = bytearray(pack.read_bytes())
    payload[-1] ^= 1
    pack.write_bytes(payload)
    with pytest.raises(Full128SuccessorEvaluationError, match="tampered"):
        open_successor_embedding_cache(
            first_descriptor,
            successor_inventory_bundle=inventory,
            evaluation_panel=evaluation_panel,
        )


def test_cache_validation_uses_owned_read_only_mapping_and_closes(
    successor_sources: tuple[dict[str, Any], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evaluation.full_segment.full128_successors as successors

    *_, panel, inventory = successor_sources

    def reject_full_pack_read(*_: object, **__: object) -> bytes:
        raise AssertionError("packed cache must not use whole-file byte reads")

    monkeypatch.setattr(successors, "_read_regular", reject_full_pack_read)
    evaluation_panel, descriptor, cache = _cache(tmp_path, inventory, panel, "MAPPED")

    assert isinstance(cache._mapping, mmap.mmap)
    assert cache._vectors.base is cache._mapping
    assert cache._vectors.flags.writeable is False
    loaded = cache.load_embeddings(evaluation_panel["required_sample_tokens"][:2])
    assert loaded.flags.owndata is True
    assert np.isfinite(loaded).all()

    mapping = cache._mapping
    cache.close()
    assert mapping.closed
    cache.close()
    with pytest.raises(Full128SuccessorEvaluationError, match="closed"):
        cache.load_embeddings(descriptor["sample_tokens"][:1])


def test_cache_mapping_preserves_per_vector_digest_validation(
    successor_sources: tuple[dict[str, Any], ...], tmp_path: Path
) -> None:
    *_, panel, inventory = successor_sources
    evaluation_panel, descriptor, cache = _cache(
        tmp_path, inventory, panel, "VECTOR-TAMPER"
    )
    cache.close()

    pack = Path(descriptor["pack_path"])
    payload = bytearray(pack.read_bytes())
    first = np.frombuffer(payload, dtype="<f4", count=128)
    first *= -1.0
    pack.write_bytes(payload)

    tampered = deepcopy(descriptor)
    tampered["pack_sha256"] = hashlib.sha256(payload).hexdigest()
    tampered["cache_descriptor_sha256"] = content_sha256(
        {
            key: value
            for key, value in tampered.items()
            if key != "cache_descriptor_sha256"
        }
    )
    with pytest.raises(Full128SuccessorEvaluationError, match="vector digest differs"):
        open_successor_embedding_cache(
            tampered,
            successor_inventory_bundle=inventory,
            evaluation_panel=evaluation_panel,
        )


def test_paired_identity_bootstrap_is_deterministic_and_rejects_population_change() -> (
    None
):
    left = [
        {
            "sample_token": f"q-{identity}-{sample}",
            "bootstrap_cluster_id": f"identity-{identity}",
            "Rank-1": float(identity == 0 or sample == 0),
        }
        for identity in range(3)
        for sample in range(2)
    ]
    right = [{**row, "Rank-1": row["Rank-1"] * 0.5} for row in left]
    first = paired_identity_cluster_bootstrap(
        left, right, metric="Rank-1", resamples=100, seed=19
    )
    second = paired_identity_cluster_bootstrap(
        list(reversed(left)),
        list(reversed(right)),
        metric="Rank-1",
        resamples=100,
        seed=19,
    )
    assert first == second
    with pytest.raises(Full128SuccessorEvaluationError, match="populations differ"):
        paired_identity_cluster_bootstrap(
            left, right[:-1], metric="Rank-1", resamples=10
        )


def test_representation_trace_contract_sanitizes_tensors_and_detects_tamper() -> None:
    embedding = np.zeros(128, dtype=np.float32)
    embedding[3] = 1.0
    trace = build_representation_trace_manifest(
        successor_id="S0",
        sample_token=_sha("trace-sample"),
        model_binding_sha256=_sha("trace-model"),
        model_input_transform={
            "source_size": [224, 224],
            "model_input_size": [224, 224],
            "color_mode": "RGB",
            "resize_interpolation": "BILINEAR",
            "mask_application": "NEUTRAL_BACKGROUND",
            "channel_mean": [0.5, 0.5, 0.5],
            "channel_std": [0.25, 0.25, 0.25],
        },
        layers=[{"name": "patch_embed", "shape": [1, 196, 384]}],
        patch_geometry={
            "input_height": 224,
            "input_width": 224,
            "patch_height": 16,
            "patch_width": 16,
            "grid_height": 14,
            "grid_width": 14,
        },
        mask_occupancy=np.ones((14, 14), dtype=np.float32).tolist(),
        embedding=embedding,
        embedding_cache_descriptor_sha256=_sha("trace-cache"),
        pair={
            "query_sample_token": "query-private",
            "key_sample_token": "key-private",
            "winning_template_id": "template-private",
            "score": 0.75,
            "rank": 1,
            "exact_cosine": True,
        },
        spatial_maps={"pair_similarity": np.eye(14, dtype=np.float32).tolist()},
    )
    assert validate_representation_trace_manifest(trace) == trace
    public = sanitize_representation_trace_manifest(trace)
    serialized = json.dumps(public, sort_keys=True)
    assert public["contains_raw_tensor_values"] is False
    assert public["visibility"] == "PUBLIC_SAFE_TRACE_SUMMARY"
    assert "template-private" not in serialized
    assert '"values"' not in serialized
    assert public["pair_evidence"] == "AVAILABLE_IN_PRIVATE_TRACE"
    assert trace["pair"]["score"] == 0.75
    assert trace["embedding_binding"]["vector_sha256"]
    _assert_public_trace_has_no_sample_detail(public)
    assert validate_public_representation_trace_manifest(public) == public

    tampered = deepcopy(trace)
    tampered["spatial_maps"]["pair_similarity"]["values"][0][0] = 0.5
    with pytest.raises(RepresentationTraceError, match="digest"):
        validate_representation_trace_manifest(tampered)


def _executed_trace_inputs() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(71)
    query_tokens = torch.randn(1, 256, 384, generator=generator)
    key_tokens = torch.randn(1, 256, 384, generator=generator)
    query_occupancy = torch.rand(1, 256, generator=generator).clamp_min_(0.01)
    key_occupancy = torch.rand(1, 256, generator=generator).clamp_min_(0.01)
    model = Dinov2OccupancyProbe128(nn.Identity()).eval()
    with torch.inference_mode():
        query_embedding = model.forward_from_tokens(query_tokens, query_occupancy)[0]
        key_embedding = model.forward_from_tokens(key_tokens, key_occupancy)[0]
    binding_names = {
        "run_manifest_sha256",
        "candidate_run_sha256",
        "model_manifest_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_state_sha256",
        "preprocessing_manifest_sha256",
        "embedding_manifest_sha256",
        "token_cache_manifest_sha256",
        "token_cache_tokens_sha256",
        "token_cache_occupancy_sha256",
        "evaluation_cache_descriptor_sha256",
        "evaluation_pack_sha256",
        "dinov2_model_sha256",
        "dinov2_config_sha256",
        "dinov2_preprocessor_sha256",
    }
    return {
        "successor_id": "B3",
        "model": model,
        "query_sample_token": _sha("executed-query"),
        "key_sample_token": _sha("executed-key"),
        "cached_query_tokens": query_tokens,
        "cached_key_tokens": key_tokens,
        "cached_query_occupancy": query_occupancy,
        "cached_key_occupancy": key_occupancy,
        "live_query_tokens": query_tokens.clone(),
        "live_key_tokens": key_tokens.clone(),
        "live_query_occupancy": query_occupancy.clone(),
        "live_key_occupancy": key_occupancy.clone(),
        "cached_query_embedding": query_embedding.numpy(),
        "cached_key_embedding": key_embedding.numpy(),
        "model_input_transform": {
            "source_size": [224, 224],
            "model_input_size": [224, 224],
            "color_mode": "RGB",
            "resize_interpolation": "NONE_ALREADY_224X224",
            "mask_application": "IMAGENET_MEAN_NEUTRAL_BEFORE_NORMALIZATION",
            "channel_mean": [0.485, 0.456, 0.406],
            "channel_std": [0.229, 0.224, 0.225],
        },
        "artifact_bindings": {name: _sha(name) for name in binding_names},
        "query_input_binding": {
            "rgb_sha256": _sha("query-rgb"),
            "mask_sha256": _sha("query-mask"),
            "crop_record_sha256": _sha("query-crop"),
        },
        "key_input_binding": {
            "rgb_sha256": _sha("key-rgb"),
            "mask_sha256": _sha("key-mask"),
            "crop_record_sha256": _sha("key-crop"),
        },
        "rank": 1,
    }


def test_executed_trace_proves_cache_matches_and_sanitizes_private_tokens() -> None:
    inputs = _executed_trace_inputs()
    trace = build_executed_representation_trace_manifest(**inputs)

    assert validate_executed_representation_trace_manifest(trace) == trace
    assert trace["execution_verification"]["query_embedding_exact_cache_match"] is True
    assert trace["pair"]["exact_cosine"] is True
    assert "pair_patch_contribution" in trace["available_maps"]
    assert any(
        row["name"] == "transformer_attention_query_key"
        for row in trace["unavailable_evidence"]
    )
    public = sanitize_representation_trace_manifest(trace)
    serialized = json.dumps(public, sort_keys=True)
    assert inputs["query_sample_token"] not in serialized
    assert inputs["key_sample_token"] not in serialized
    assert '"values"' not in serialized
    assert public["visibility"] == "PUBLIC_SAFE_TRACE_SUMMARY"
    assert public["execution_evidence"] == (
        "ACTUAL_EXECUTION_WITH_EXACT_PRIVATE_CACHE_BINDINGS"
    )
    _assert_public_trace_has_no_sample_detail(public)
    assert validate_public_representation_trace_manifest(public) == public

    analysis = build_public_representation_analysis([public])
    assert analysis["visibility"] == "PUBLIC_SAFE_TRACE_SUMMARY"
    assert analysis["aggregation"] == "NONE_SELECTED_TRACE_AVAILABILITY_ONLY"
    assert validate_public_representation_analysis(analysis) == analysis


def test_executed_trace_occupancy_maps_select_nonzero_decomposition_rows() -> None:
    inputs = _executed_trace_inputs()
    generator = torch.Generator().manual_seed(79)
    query_tokens = torch.randn(3, 256, 384, generator=generator)
    key_tokens = torch.randn(3, 256, 384, generator=generator)
    query_occupancy = torch.stack(
        [torch.full((256,), value) for value in (0.1, 0.4, 0.8)]
    )
    key_occupancy = torch.stack(
        [torch.full((256,), value) for value in (0.2, 0.5, 0.9)]
    )
    model = inputs["model"]
    with torch.inference_mode():
        query_embeddings = model.forward_from_tokens(query_tokens, query_occupancy)
        key_embeddings = model.forward_from_tokens(key_tokens, key_occupancy)
    inputs.update(
        {
            "cached_query_tokens": query_tokens,
            "cached_key_tokens": key_tokens,
            "cached_query_occupancy": query_occupancy,
            "cached_key_occupancy": key_occupancy,
            "live_query_tokens": query_tokens.clone(),
            "live_key_tokens": key_tokens.clone(),
            "live_query_occupancy": query_occupancy.clone(),
            "live_key_occupancy": key_occupancy.clone(),
            "cached_query_embedding": query_embeddings[1].numpy(),
            "cached_key_embedding": key_embeddings[2].numpy(),
            "query_index": 1,
            "key_index": 2,
        }
    )

    trace = build_executed_representation_trace_manifest(**inputs)

    assert np.array_equal(
        np.asarray(trace["available_maps"]["query_mask_occupancy"]["values"]),
        query_occupancy[1].reshape(16, 16).numpy(),
    )
    assert np.array_equal(
        np.asarray(trace["available_maps"]["key_mask_occupancy"]["values"]),
        key_occupancy[2].reshape(16, 16).numpy(),
    )
    assert validate_executed_representation_trace_manifest(trace) == trace


def test_executed_b5_trace_contains_actual_spatial_scorer_maps() -> None:
    inputs = _executed_trace_inputs()
    model = SpatialScorer128(nn.Identity(), nn.Linear(384, 128)).eval()
    with torch.inference_mode():
        query = model.forward_from_tokens(
            inputs["cached_query_tokens"], inputs["cached_query_occupancy"]
        )[0]
        key = model.forward_from_tokens(
            inputs["cached_key_tokens"], inputs["cached_key_occupancy"]
        )[0]
    inputs.update(
        {
            "successor_id": "B5-SPATIAL",
            "model": model,
            "cached_query_embedding": query.numpy(),
            "cached_key_embedding": key.numpy(),
        }
    )

    trace = build_executed_representation_trace_manifest(**inputs)

    assert "query_spatial_scorer_logit" in trace["available_maps"]
    assert "key_spatial_scorer_logit" in trace["available_maps"]
    assert all(
        row["name"] != "spatial_scorer_logits" for row in trace["unavailable_evidence"]
    )


def test_executed_trace_fails_closed_on_live_or_embedding_cache_divergence() -> None:
    live_mismatch = _executed_trace_inputs()
    live_mismatch["live_query_tokens"][0, 0, 0] += 1
    with pytest.raises(RepresentationTraceError, match="patch tokens"):
        build_executed_representation_trace_manifest(**live_mismatch)

    embedding_mismatch = _executed_trace_inputs()
    embedding_mismatch["cached_query_embedding"] = embedding_mismatch[
        "cached_query_embedding"
    ].copy()
    embedding_mismatch["cached_query_embedding"][0] += np.float32(1e-4)
    embedding_mismatch["cached_query_embedding"] /= np.linalg.norm(
        embedding_mismatch["cached_query_embedding"]
    )
    with pytest.raises(
        RepresentationTraceError, match="embedding does not exactly match"
    ):
        build_executed_representation_trace_manifest(**embedding_mismatch)


def test_executed_trace_rejects_rehashed_non_cosine_pair_evidence() -> None:
    trace = build_executed_representation_trace_manifest(**_executed_trace_inputs())
    tampered = deepcopy(trace)
    tampered["pair"]["exact_cosine"] = False
    payload = {key: value for key, value in tampered.items() if key != "trace_sha256"}
    tampered["trace_sha256"] = content_sha256(payload)

    with pytest.raises(RepresentationTraceError, match="cosine evidence"):
        validate_executed_representation_trace_manifest(tampered)


def _assert_public_trace_has_no_sample_detail(value: Mapping[str, Any]) -> None:
    forbidden = {
        "source_trace_sha256",
        "vector_sha256",
        "query_vector_sha256",
        "key_vector_sha256",
        "values_sha256",
        "score",
        "rank",
        "winning_template_binding_sha256",
        "mean",
        "minimum",
        "maximum",
        "sum",
    }

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            assert forbidden.isdisjoint(item)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
