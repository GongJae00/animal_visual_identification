from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from shared.foundation.provenance import content_sha256
from evaluation.splits.face import (
    face_exposure_history,
    face_identity_protocol_v2,
    face_public_source_binding,
)
from evaluation.splits.face.face_exposure_history import (
    build_face_exposure_history,
    validate_face_exposure_history_bundle,
)
from evaluation.splits.face.face_gallery_query_panel import (
    build_face_gallery_query_panel,
    validate_face_gallery_query_panel_bundle,
)
from evaluation.splits.face.face_identity_protocol_v2 import (
    build_face_identity_protocol_v2,
    validate_face_identity_protocol_v2_bundle,
)
from evaluation.splits.face.face_public_source_binding import (
    build_face_public_source_binding,
    validate_face_public_source_binding_bundle,
)
from enrollment.registry.identity_registry import (
    compute_identity_token,
    compute_registered_dog_id,
    compute_sample_token,
    compute_sequence_token,
)
from evaluation.splits.protected_public_split import (
    EvidenceRelation,
    FrozenPublicSplitEvidenceGraph,
    PublicSplitEvidenceEdge,
    PublicSplitSample,
    PublicSplitSourceBundle,
)
from evaluation.splits.role_exposure import ExposureStage

from tests.repo_root import REPO_ROOT as ROOT

def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()

def _bindings(receipts_sha256: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "exact_duplicate_graph_sha256": _sha("exact"),
                "geometric_verifier_sha256": _sha("geometry"),
                "image_content_receipts_sha256": receipts_sha256,
                "pdq_candidates_sha256": _sha("pdq"),
                "phash_candidates_sha256": _sha("phash"),
                "review_adjudication_sha256": _sha("review"),
                "semantic_receipts_sha256": _sha("semantic"),
            }.items()
        )
    )

def _base_inputs(
    *, byte_salt: str = "a"
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    route_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    public_samples: list[PublicSplitSample] = []
    receipt_rows: list[dict[str, Any]] = []
    for identity_index in range(6):
        dataset_identity = f"dogfacenet224:v1:web-folder:{identity_index}"
        identity_token = compute_identity_token(dataset_identity)
        registered = compute_registered_dog_id(dataset_identity)
        for sample_index in range(7):
            source_id = (
                f"dogfacenet224:v1:web-folder:{identity_index}:"
                f"image:image{sample_index}.0"
            )
            public_token = compute_sample_token(source_id)
            route_token = _sha(f"route:{identity_index}:{sample_index}")
            encoded = _sha(f"encoded:{byte_salt}:{identity_index}:{sample_index}")
            member_path = f"after_4_bis/{identity_index}/image{sample_index}.png"
            public_samples.append(
                PublicSplitSample(
                    sample_token=public_token,
                    identity_token=identity_token,
                    sequence_token=compute_sequence_token(None, identity_token),
                    source_sample_id=source_id,
                    dataset_identity_id=dataset_identity,
                    dataset_name="dogfacenet224",
                    source_variant="original",
                    original_split="train",
                    raw_frame_index=sample_index,
                    paired_source_sample_id=None,
                    in_no_mono_subset=None,
                    region="FACE",
                )
            )
            receipt_rows.append(
                {
                    "source_sample_id": source_id,
                    "dataset_name": "dogfacenet224",
                    "source_variant": "original",
                    "encoded_sha256": encoded,
                    "member_path": member_path,
                }
            )
            source_record = _sha(
                f"route-record:{byte_salt}:{identity_index}:{sample_index}"
            )
            face_record = _sha(
                f"face-record:{byte_salt}:{identity_index}:{sample_index}"
            )
            route_rows.append(
                {
                    "sample_token": route_token,
                    "record_sha256": source_record,
                    "dataset_name": "dogfacenet224",
                    "source_sha256": encoded,
                    "source_path": member_path,
                    "identity_metadata": {
                        "registered_identity_id": registered,
                    },
                }
            )
            overlay_rows.append(
                {
                    "sample_token": route_token,
                    "record_sha256": face_record,
                    "source_record_sha256": source_record,
                    "registered_identity_id": registered,
                    "dataset_name": "dogfacenet224",
                    "publisher_split": "train",
                    "gallery_query_eligible": True,
                }
            )
    route_rows.sort(key=lambda row: row["sample_token"])
    overlay_rows.sort(key=lambda row: row["sample_token"])
    receipts = {
        "dogfacenet224": {
            "receipt": {
                "decision": "PASS_IMAGE_CONTENT_AUDIT",
                "records": receipt_rows,
            }
        }
    }
    source = PublicSplitSourceBundle(
        _bindings(content_sha256(receipts)), tuple(public_samples)
    )
    route = {
        "plan_sha256": _sha(f"plan:{byte_salt}"),
        "bundle_sha256": _sha(f"route-bundle:{byte_salt}"),
        "plan": {"records": route_rows},
    }
    overlay = {
        "overlay_sha256": _sha(f"overlay:{byte_salt}"),
        "bundle_sha256": _sha(f"overlay-bundle:{byte_salt}"),
        "overlay": {
            "source_route_plan_sha256": route["plan_sha256"],
            "source_route_plan_bundle_sha256": route["bundle_sha256"],
            "records": overlay_rows,
        },
    }
    return route, overlay, source.to_dict(), receipts

def _patch_route_overlay_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        face_public_source_binding,
        "validate_full128_route_plan_bundle",
        lambda value, *, verify_files: value,
    )
    monkeypatch.setattr(
        face_public_source_binding,
        "validate_face_eligibility_overlay_bundle",
        lambda value: value,
    )
    monkeypatch.setattr(
        face_identity_protocol_v2,
        "validate_full128_route_plan_bundle",
        lambda value, *, verify_files: value,
    )
    monkeypatch.setattr(
        face_identity_protocol_v2,
        "validate_face_eligibility_overlay_bundle",
        lambda value: value,
    )

def _variant(bridge: dict[str, Any], *, variant: str = "B2") -> dict[str, Any]:
    samples = [
        {
            "sample_id": row["route_sample_token"],
            "identity_id": row["registered_identity_id"],
            "dataset_name": row["dataset_name"],
            "view": "face",
            "crop_record_sha256": _sha(f"crop:{variant}:{row['route_sample_token']}"),
        }
        for row in bridge["binding"]["records"]
    ]
    fit_payload = {
        "partition": "FIT",
        "sample_count": len(samples),
        "identity_count": len({row["identity_id"] for row in samples}),
        "samples": samples,
    }
    fit = {**fit_payload, "fit_population_sha256": content_sha256(fit_payload)}
    payload = {
        "schema_version": "cvi.full128_variant_run.v1",
        "variant_id": variant,
        "method": "fixture",
        "initialization": "fixture",
        "bindings": {},
        "fit_population": fit,
        "training": {"executed": True},
        "artifacts": {},
    }
    return {**payload, "variant_run_sha256": content_sha256(payload)}

def _graph_and_receipt(
    bridge: dict[str, Any], source: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = bridge["binding"]["records"]
    by_identity: dict[str, list[str]] = {}
    for row in rows:
        by_identity.setdefault(row["public_identity_token"], []).append(
            row["public_sample_token"]
        )
    identities = sorted(by_identity)
    left, right = sorted((by_identity[identities[0]][0], by_identity[identities[1]][0]))
    edge = PublicSplitEvidenceEdge(
        left_sample_token=left,
        right_sample_token=right,
        relation=EvidenceRelation.EXACT_CONFIRMED,
        evidence_token=_sha("cross-identity-evidence"),
    )
    graph = FrozenPublicSplitEvidenceGraph(
        evidence_bindings=tuple(tuple(item) for item in source["evidence_bindings"]),
        edges=(edge,),
    )
    provenance = {"tool": "fixture", "version": 1}
    receipt_payload = {
        "schema_version": ("cvi.public_split_evidence_graph_assembly_receipt.v1"),
        "source_bundle_sha256": PublicSplitSourceBundle.from_dict(source).bundle_sha256,
        "adjudication_ledger_sha256": _sha("ledger"),
        "candidate_set_sha256": _sha("candidates"),
        "candidate_count": 1,
        "outcome_counts": {"EXACT_CONFIRMED": 1},
        "unresolved_candidate_count": 0,
        "unbound_candidate_count": 0,
        "adjudication_mode": "STANDARD",
        "graph_sha256": graph.graph_sha256,
        "edge_count": 1,
        "tool_provenance": provenance,
        "tool_provenance_sha256": content_sha256(provenance),
        "decision": "PASS_COMPLETE_ADJUDICATION_GRAPH_PROMOTION",
    }
    receipt = {
        **receipt_payload,
        "receipt_sha256": content_sha256(receipt_payload),
    }
    return graph.to_dict(), receipt

def _pipeline(
    monkeypatch: pytest.MonkeyPatch, *, byte_salt: str = "a"
) -> dict[str, Any]:
    _patch_route_overlay_validators(monkeypatch)
    route, overlay, source, receipts = _base_inputs(byte_salt=byte_salt)
    bridge = build_face_public_source_binding(route, overlay, source, receipts)
    exposure = build_face_exposure_history(
        bridge, full128_artifacts=(_variant(bridge),)
    )
    graph, receipt = _graph_and_receipt(bridge, source)
    protocol = build_face_identity_protocol_v2(
        route,
        overlay,
        bridge,
        exposure,
        graph,
        receipt,
        protocol_name="phase-1-governance-v2-fixture",
    )
    panel = build_face_gallery_query_panel(protocol)
    return {
        "route": route,
        "overlay": overlay,
        "source": source,
        "receipts": receipts,
        "bridge": bridge,
        "exposure": exposure,
        "graph": graph,
        "graph_receipt": receipt,
        "protocol": protocol,
        "panel": panel,
    }

def test_governance_v2_pipeline_is_deterministic_closed_and_nested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _pipeline(monkeypatch)
    second = _pipeline(monkeypatch)

    assert first == second
    assert (
        validate_face_public_source_binding_bundle(first["bridge"]) == first["bridge"]
    )
    assert validate_face_exposure_history_bundle(first["exposure"]) == first["exposure"]
    assert (
        validate_face_identity_protocol_v2_bundle(first["protocol"])
        == first["protocol"]
    )
    assert validate_face_gallery_query_panel_bundle(first["panel"]) == first["panel"]
    protocol = first["protocol"]["protocol"]
    assert protocol["final_evaluation_permitted"] is False
    assert protocol["score_bearing_bytes_used_for_role_allocation"] is False
    assert first["protocol"]["census"]["cross_identity_unsafe_sample_count"] == 2
    assert set(first["protocol"]["census"]["identity_role_counts"]) == {
        "FIT",
        "DEV",
        "CAL",
        "EXPOSED_DIAGNOSTIC",
        "EXCLUDED_UNSAFE_COMPONENT",
    }
    for row in first["panel"]["panel"]["common_k5_feasible_cohort"]:
        galleries = row["gallery_sample_tokens_by_k"]
        assert galleries["K1"] == galleries["K3"][:1]
        assert galleries["K3"] == galleries["K5"][:3]
        assert row["query_sample_token"] not in galleries["K5"]
        assert row["cross_session_verified"] is False

def test_source_binding_rejects_ambiguity_and_identity_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_route_overlay_validators(monkeypatch)
    route, overlay, source, receipts = _base_inputs()
    first, second = receipts["dogfacenet224"]["receipt"]["records"][:2]
    second["encoded_sha256"] = first["encoded_sha256"]
    source["evidence_bindings"] = [
        [
            key,
            content_sha256(receipts)
            if key == "image_content_receipts_sha256"
            else value,
        ]
        for key, value in source["evidence_bindings"]
    ]
    route["plan"]["records"][0]["source_sha256"] = first["encoded_sha256"]
    route["plan"]["records"][0]["source_path"] = "not/a/member.png"
    with pytest.raises(ValueError, match="exactly one audited public source"):
        build_face_public_source_binding(route, overlay, source, receipts)

    route, overlay, source, receipts = _base_inputs()
    overlay["overlay"]["records"][0]["registered_identity_id"] = (
        compute_registered_dog_id("dogfacenet224:v1:web-folder:999")
    )
    with pytest.raises(ValueError, match="identity conflict"):
        build_face_public_source_binding(route, overlay, source, receipts)

def test_unresolved_exposure_is_auditable_and_blocks_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_route_overlay_validators(monkeypatch)
    route, overlay, source, receipts = _base_inputs()
    bridge = build_face_public_source_binding(route, overlay, source, receipts)
    unsupported = {
        "schema_version": "cvi.face_identity_protocol_bundle.v1",
        "protocol": {
            "sample_assignments": [
                {"sample_token": bridge["binding"]["records"][0]["route_sample_token"]}
            ]
        },
    }
    exposure = build_face_exposure_history(bridge, full128_artifacts=(unsupported,))
    assert exposure["history"]["status"] == "BLOCKED_UNRESOLVED_PROJECTIONS"
    assert exposure["history"]["role_allocation_permitted"] is False
    assert exposure["history"]["clean_role_claims_permitted"] is False
    assert len(exposure["history"]["unresolved_rows"]) == 1
    assert validate_face_exposure_history_bundle(exposure) == exposure
    graph, receipt = _graph_and_receipt(bridge, source)
    with pytest.raises(ValueError, match="unresolved face exposure"):
        build_face_identity_protocol_v2(
            route,
            overlay,
            bridge,
            exposure,
            graph,
            receipt,
            protocol_name="blocked-fixture",
        )

def test_full128_nonface_fit_rows_are_outside_face_projection_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_route_overlay_validators(monkeypatch)
    route, overlay, source, receipts = _base_inputs()
    bridge = build_face_public_source_binding(route, overlay, source, receipts)
    variant = _variant(bridge)
    fit = variant["fit_population"]
    fit["samples"].append(
        {
            "sample_id": _sha("yt-route-sample"),
            "identity_id": "generated-body-identity",
            "dataset_name": "yt-bb-dog",
            "view": "body",
            "crop_record_sha256": _sha("yt-crop"),
        }
    )
    fit["sample_count"] = len(fit["samples"])
    fit["identity_count"] += 1
    fit["fit_population_sha256"] = content_sha256(
        {key: value for key, value in fit.items() if key != "fit_population_sha256"}
    )
    variant["variant_run_sha256"] = content_sha256(
        {key: value for key, value in variant.items() if key != "variant_run_sha256"}
    )
    exposure = build_face_exposure_history(bridge, full128_artifacts=(variant,))
    assert exposure["history"]["status"] == "COMPLETE_EXACT_PROJECTIONS"
    assert len(exposure["history"]["ledger"]["records"]) == len(
        bridge["binding"]["records"]
    )

def test_masked_afn_projection_uses_report_bound_kfold_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_route_overlay_validators(monkeypatch)
    route, overlay, source, receipts = _base_inputs()
    bridge = build_face_public_source_binding(route, overlay, source, receipts)
    bridge_rows = bridge["binding"]["records"]
    identities = {
        row["public_identity_token"]: SimpleNamespace(
            identity_token=row["public_identity_token"],
            registered_dog_id=row["registered_identity_id"],
        )
        for row in bridge_rows
    }
    samples = tuple(
        SimpleNamespace(
            sample_token=row["public_sample_token"],
            identity_token=row["public_identity_token"],
            source_variant="original",
            home_fold=0,
            held_out_role=(
                face_exposure_history.HeldOutSampleRole.QUERY
                if index == 0
                else face_exposure_history.HeldOutSampleRole.EXCLUDED
            ),
            training_eligible=True,
        )
        for index, row in enumerate(bridge_rows)
    )
    kfold = SimpleNamespace(
        manifest_sha256=_sha("masked-kfold"),
        policy=SimpleNamespace(fold_count=3),
        identity_assignments=tuple(identities.values()),
        sample_assignments=samples,
    )
    monkeypatch.setattr(
        face_exposure_history,
        "DatasetStratifiedIdentityKFoldManifest",
        SimpleNamespace(from_dict=lambda payload: kfold),
    )
    report_body = {
        "schema_version": "cvi.masked_afn_kfold_report.v1",
        "kfold_manifest_sha256": kfold.manifest_sha256,
        "candidate_manifest_sha256s": {},
        "candidate_source_token_binding": {
            "source_bundle_sha256": bridge["binding"]["public_source_bundle_sha256"]
        },
        "config": {},
        "checkpoints": [],
        "folds": [{"fold_index": index} for index in range(3)],
        "retrieval_eligibility": {},
        "out_of_fold_test": {},
        "interpretation": "fixture",
    }
    report = {**report_body, "report_sha256": content_sha256(report_body)}
    exposure = build_face_exposure_history(
        bridge, masked_afn_runs=((report, {"fixture": True}),)
    )
    stages = {
        row["maximum_historical_stage"]
        for row in exposure["history"]["ledger"]["records"]
    }
    assert stages == {
        ExposureStage.MODEL_TRAINING_USED.value,
        ExposureStage.MODEL_SELECTION_SCORED.value,
    }
    assert exposure["history"]["role_allocation_permitted"] is True

    excluded_samples = tuple(
        SimpleNamespace(
            **{
                **sample.__dict__,
                "training_eligible": False,
                "held_out_role": face_exposure_history.HeldOutSampleRole.EXCLUDED,
            }
        )
        if index == 1
        else sample
        for index, sample in enumerate(samples)
    )
    excluded_kfold = SimpleNamespace(
        **{**kfold.__dict__, "sample_assignments": excluded_samples}
    )
    monkeypatch.setattr(
        face_exposure_history,
        "DatasetStratifiedIdentityKFoldManifest",
        SimpleNamespace(from_dict=lambda payload: excluded_kfold),
    )
    explicit_nonparticipation = build_face_exposure_history(
        bridge, masked_afn_runs=((report, {"fixture": True}),)
    )
    source = explicit_nonparticipation["history"]["source_artifacts"][0]
    assert source["nonparticipating_record_count"] == 1
    assert explicit_nonparticipation["history"]["role_allocation_permitted"] is True

    unbound_body = {**report_body, "candidate_source_token_binding": None}
    unbound = {
        **unbound_body,
        "report_sha256": content_sha256(unbound_body),
    }
    blocked = build_face_exposure_history(
        bridge, masked_afn_runs=((unbound, {"fixture": True}),)
    )
    assert blocked["history"]["role_allocation_permitted"] is False
    assert len(blocked["history"]["unresolved_rows"]) == len(bridge_rows)

def test_masked_afn_projection_accepts_persisted_kfold_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_route_overlay_validators(monkeypatch)
    route, overlay, source, receipts = _base_inputs()
    bridge = build_face_public_source_binding(route, overlay, source, receipts)
    kfold = SimpleNamespace(
        manifest_sha256=_sha("masked-kfold-envelope"),
        policy=SimpleNamespace(fold_count=1),
        identity_assignments=(),
        sample_assignments=(),
    )
    observed: list[object] = []
    monkeypatch.setattr(
        face_exposure_history,
        "DatasetStratifiedIdentityKFoldManifest",
        SimpleNamespace(
            from_dict=lambda payload: (observed.append(payload), kfold)[1]
        ),
    )
    manifest = {"fixture": True}
    envelope = {
        "schema_version": "cvi.dataset_stratified_identity_kfold_manifest_bundle.v1",
        "manifest_sha256": content_sha256(manifest),
        "manifest": manifest,
    }
    report_body = {
        "schema_version": "cvi.masked_afn_kfold_report.v1",
        "kfold_manifest_sha256": kfold.manifest_sha256,
        "candidate_manifest_sha256s": {},
        "candidate_source_token_binding": {
            "source_bundle_sha256": bridge["binding"]["public_source_bundle_sha256"]
        },
        "config": {},
        "checkpoints": [],
        "folds": [{"fold_index": 0}],
        "retrieval_eligibility": {},
        "out_of_fold_test": {},
        "interpretation": "fixture",
    }
    report = {**report_body, "report_sha256": content_sha256(report_body)}

    build_face_exposure_history(
        bridge, masked_afn_runs=((report, envelope),)
    )

    assert observed == [manifest]

def test_role_allocation_does_not_depend_on_score_bearing_source_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _pipeline(monkeypatch, byte_salt="first")
    second = _pipeline(monkeypatch, byte_salt="second")
    first_roles = {
        row["public_identity_token"]: row["role"]
        for row in first["protocol"]["protocol"]["identity_assignments"]
    }
    second_roles = {
        row["public_identity_token"]: row["role"]
        for row in second["protocol"]["protocol"]["identity_assignments"]
    }
    assert first_roles == second_roles
    assert (
        first["bridge"]["binding"]["image_content_receipts_sha256"]
        != second["bridge"]["binding"]["image_content_receipts_sha256"]
    )

@pytest.mark.parametrize(
    ("artifact", "validator", "path"),
    [
        (
            "bridge",
            validate_face_public_source_binding_bundle,
            ("binding", "records", 0, "encoded_sha256"),
        ),
        (
            "exposure",
            validate_face_exposure_history_bundle,
            ("history", "status"),
        ),
        (
            "protocol",
            validate_face_identity_protocol_v2_bundle,
            ("protocol", "final_evaluation_permitted"),
        ),
        (
            "panel",
            validate_face_gallery_query_panel_bundle,
            ("panel", "cross_session_claimed"),
        ),
    ],
)
def test_every_governance_bundle_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    validator: Any,
    path: tuple[Any, ...],
) -> None:
    pipeline = _pipeline(monkeypatch)
    changed = copy.deepcopy(pipeline[artifact])
    target: Any = changed
    for key in path[:-1]:
        target = target[key]
    final = path[-1]
    target[final] = not target[final] if isinstance(target[final], bool) else "0" * 64
    with pytest.raises(ValueError, match="digest"):
        validator(changed)

def test_protocol_rejects_rehashed_nonadmitted_graph_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline(monkeypatch)
    receipt = copy.deepcopy(pipeline["graph_receipt"])
    receipt["unresolved_candidate_count"] = 1
    receipt["receipt_sha256"] = content_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="not admitted"):
        build_face_identity_protocol_v2(
            pipeline["route"],
            pipeline["overlay"],
            pipeline["bridge"],
            pipeline["exposure"],
            pipeline["graph"],
            receipt,
            protocol_name="phase-1-governance-v2-fixture",
        )

def test_protocol_rejects_rehashed_role_reallocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_bundle = copy.deepcopy(_pipeline(monkeypatch)["protocol"])
    protocol = protocol_bundle["protocol"]
    identity = next(
        row for row in protocol["identity_assignments"] if row["role"] == "FIT"
    )
    identity["role"] = "DEV"
    identity["record_sha256"] = content_sha256(
        {key: value for key, value in identity.items() if key != "record_sha256"}
    )
    for sample in protocol["sample_assignments"]:
        if sample["public_identity_token"] == identity["public_identity_token"]:
            sample["identity_role"] = "DEV"
            if not sample["cross_identity_unsafe"]:
                sample["role"] = "DEV"
                sample["gradient_eligible"] = False
            sample["record_sha256"] = content_sha256(
                {key: value for key, value in sample.items() if key != "record_sha256"}
            )
    protocol["protocol_sha256"] = content_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    )
    protocol_bundle["protocol_sha256"] = protocol["protocol_sha256"]
    census = face_identity_protocol_v2._build_census(
        protocol["identity_assignments"], protocol["sample_assignments"]
    )
    protocol_bundle["census"] = census
    protocol_bundle["census_sha256"] = content_sha256(census)
    protocol_bundle["bundle_sha256"] = content_sha256(
        {key: value for key, value in protocol_bundle.items() if key != "bundle_sha256"}
    )
    with pytest.raises(ValueError, match="deterministic role allocation"):
        validate_face_identity_protocol_v2_bundle(protocol_bundle)

def test_panel_rejects_rehashed_dependency_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = copy.deepcopy(_pipeline(monkeypatch)["panel"])
    row = panel["panel"]["common_k5_feasible_cohort"][0]
    row["gallery_dependency_components_by_k"]["K5"][0] = row[
        "query_dependency_component_sha256"
    ]
    row["gallery_dependency_components_by_k"]["K3"][0] = row[
        "query_dependency_component_sha256"
    ]
    row["gallery_dependency_components_by_k"]["K1"][0] = row[
        "query_dependency_component_sha256"
    ]
    row["record_sha256"] = content_sha256(
        {key: value for key, value in row.items() if key != "record_sha256"}
    )
    panel_body = {
        key: value for key, value in panel["panel"].items() if key != "panel_sha256"
    }
    panel["panel"]["panel_sha256"] = content_sha256(panel_body)
    panel["panel_sha256"] = panel["panel"]["panel_sha256"]
    bundle_body = {key: value for key, value in panel.items() if key != "bundle_sha256"}
    panel["bundle_sha256"] = content_sha256(bundle_body)
    with pytest.raises(ValueError, match="dependency disjointness"):
        validate_face_gallery_query_panel_bundle(panel)

def test_phase_1_workflow_help_exposes_exact_required_inputs() -> None:
    expected = {
        "build_face_public_source_binding.py": "--image-content-receipts",
        "build_face_exposure_history.py": "--masked-afn-run",
        "build_face_identity_protocol_v2.py": "--joint-filter-receipt",
        "build_face_gallery_query_panel.py": "--protocol-v2",
    }
    for workflow, option in expected.items():
        completed = subprocess.run(
            [sys.executable, str(ROOT / "archive/face/commands" / workflow), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert option in completed.stdout
