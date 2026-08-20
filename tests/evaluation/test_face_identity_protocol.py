from __future__ import annotations

import hashlib
from typing import Any

import pytest

from evaluation.splits.face import face_identity_protocol
from evaluation.splits.face.face_identity_protocol import (
    build_face_identity_protocol,
    validate_face_identity_protocol_bundle,
)

def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()

def _fixtures() -> tuple[dict[str, Any], dict[str, Any]]:
    route_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []

    def add(
        label: str,
        dataset: str,
        identity: str,
        split: str,
        protocol_role: str,
        *,
        duplicate: str | None = None,
    ) -> None:
        sample_token = _sha(f"sample:{label}")
        source_record = _sha(f"source-record:{label}")
        route_rows.append(
            {
                "sample_token": sample_token,
                "record_sha256": source_record,
                "duplicate_component": _sha(duplicate or f"duplicate:{label}"),
                "capture_metadata": {"capture_group_id": f"capture:{label}"},
            }
        )
        overlay_rows.append(
            {
                "sample_token": sample_token,
                "record_sha256": _sha(f"face-record:{label}"),
                "source_record_sha256": source_record,
                "registered_identity_id": identity,
                "dataset_name": dataset,
                "publisher_split": split,
                "face_protocol_role": protocol_role,
                "gallery_query_eligible": True,
            }
        )

    for identity_index in range(6):
        identity = f"dogface-{identity_index}"
        for sample_index in range(2):
            add(
                f"fit-{identity_index}-{sample_index}",
                "dogfacenet224",
                identity,
                "train",
                "FIT",
                duplicate=(
                    "cross-exposure"
                    if identity_index == 0 and sample_index == 0
                    else None
                ),
            )
    for sample_index in range(2):
        add(
            f"test-{sample_index}",
            "dogfacenet224",
            "dogface-test",
            "test",
            "EXPOSED_DIAGNOSTIC",
            duplicate="cross-exposure" if sample_index == 0 else None,
        )
    add(
        "mpdd-gallery",
        "mpdd",
        "mpdd-1",
        "gallery",
        "EXPOSED_DIAGNOSTIC",
    )
    add(
        "mpdd-query",
        "mpdd",
        "mpdd-1",
        "query",
        "EXPOSED_DIAGNOSTIC",
    )
    route_rows.sort(key=lambda row: row["sample_token"])
    overlay_rows.sort(key=lambda row: row["sample_token"])
    route = {
        "plan_sha256": _sha("plan"),
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
    return route, overlay

def _build(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    route, overlay = _fixtures()
    monkeypatch.setattr(
        face_identity_protocol,
        "validate_full128_route_plan_bundle",
        lambda value, *, verify_files: value,
    )
    monkeypatch.setattr(
        face_identity_protocol,
        "validate_face_eligibility_overlay_bundle",
        lambda value: value,
    )
    return build_face_identity_protocol(
        route,
        overlay,
        protocol_name="b2-fv-fixture-v1",
        historical_artifact_sha256s=(_sha("B0"), _sha("B1"), _sha("B2")),
    )

def test_protocol_is_identity_disjoint_and_exposure_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build(monkeypatch)

    assert validate_face_identity_protocol_bundle(bundle) == bundle
    identities = bundle["protocol"]["identity_assignments"]
    role_by_identity = {
        row["registered_identity_id"]: row["role"] for row in identities
    }
    assert set(role_by_identity.values()) == {
        "FIT",
        "DEV",
        "CAL",
        "EXPOSED_DIAGNOSTIC",
    }
    assert role_by_identity["dogface-test"] == "EXPOSED_DIAGNOSTIC"
    assert role_by_identity["mpdd-1"] == "EXPOSED_DIAGNOSTIC"
    assert role_by_identity["dogface-0"] == "EXPOSED_DIAGNOSTIC"
    promoted = next(
        row for row in identities if row["registered_identity_id"] == "dogface-0"
    )
    assert promoted["dependency_promoted_to_exposed"] is True
    samples = bundle["protocol"]["sample_assignments"]
    assert all(
        sample["gradient_eligible"] == (sample["role"] == "FIT") for sample in samples
    )
    component_roles: dict[str, set[str]] = {}
    for sample in samples:
        component_roles.setdefault(sample["duplicate_component"], set()).add(
            sample["role"]
        )
    assert all(len(roles) == 1 for roles in component_roles.values())

def test_protocol_rejects_overlay_bound_to_another_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route, overlay = _fixtures()
    overlay["overlay"]["source_route_plan_sha256"] = _sha("other-plan")
    monkeypatch.setattr(
        face_identity_protocol,
        "validate_full128_route_plan_bundle",
        lambda value, *, verify_files: value,
    )
    monkeypatch.setattr(
        face_identity_protocol,
        "validate_face_eligibility_overlay_bundle",
        lambda value: value,
    )

    with pytest.raises(ValueError, match="bindings differ"):
        build_face_identity_protocol(
            route,
            overlay,
            protocol_name="b2-fv-fixture-v1",
            historical_artifact_sha256s=(_sha("B0"),),
        )

def test_protocol_rejects_duplicate_history_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route, overlay = _fixtures()
    monkeypatch.setattr(
        face_identity_protocol,
        "validate_full128_route_plan_bundle",
        lambda value, *, verify_files: value,
    )
    monkeypatch.setattr(
        face_identity_protocol,
        "validate_face_eligibility_overlay_bundle",
        lambda value: value,
    )

    with pytest.raises(ValueError, match="non-empty and unique"):
        build_face_identity_protocol(
            route,
            overlay,
            protocol_name="b2-fv-fixture-v1",
            historical_artifact_sha256s=(_sha("B0"), _sha("B0")),
        )
