from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

import pytest

from foundation.provenance import content_sha256
from identity.face import face_eligibility
from identity.face.face_eligibility import (
    DogFaceSplitEvidence,
    build_face_eligibility_overlay,
    validate_face_eligibility_overlay_bundle,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _row(
    label: str,
    dataset: str,
    *,
    split: str = "train",
    raw_identity: str | None = None,
    registered_identity: str | None = None,
    adapter_metadata: dict[str, Any] | None = None,
    route_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sample_token": _sha(f"sample:{label}"),
        "dataset_name": dataset,
        "record_sha256": _sha(f"record:{label}"),
        "split": split,
        "identity_metadata": {
            "raw_identity_id": raw_identity,
            "registered_identity_id": registered_identity,
            "generated_identity_id": None,
        },
        "source_metadata": {"adapter_metadata": adapter_metadata or {}},
        "route_evidence": route_evidence or {},
    }


def _fixture() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    annotation_sha = _sha("ap10k-annotation")
    rows = [
        _row(
            "ap-visible",
            "ap10k-dog",
            route_evidence={
                "annotation_id": 1,
                "image_id": 10,
                "annotation_artifact": {"sha256": annotation_sha},
            },
        ),
        _row(
            "ap-no-eye",
            "ap10k-dog",
            route_evidence={
                "annotation_id": 2,
                "image_id": 11,
                "annotation_artifact": {"sha256": annotation_sha},
            },
        ),
        _row(
            "dogface-train",
            "dogfacenet224",
            raw_identity="1",
            registered_identity="11111111-1111-5111-8111-111111111111",
        ),
        _row(
            "dogface-test",
            "dogfacenet224",
            raw_identity="2",
            registered_identity="22222222-2222-5222-8222-222222222222",
        ),
        _row("dogflw", "dogflw"),
        _row(
            "mpdd",
            "mpdd",
            split="query",
            raw_identity="4",
            registered_identity="44444444-4444-5444-8444-444444444444",
        ),
        _row(
            "oxford",
            "oxford-pets-dog",
            adapter_metadata={"head_pose": "Frontal"},
        ),
        _row("sibetan", "sibetan"),
        _row("yt", "yt-bb-dog"),
    ]
    rows.sort(key=lambda row: row["sample_token"])
    route = {
        "plan_sha256": _sha("plan"),
        "bundle_sha256": _sha("route-bundle"),
        "plan": {"records": rows},
    }
    visible = [0] * 51
    visible[2] = 2
    visible[8] = 2
    no_eye = [0] * 51
    no_eye[8] = 2
    annotations = {
        annotation_sha: {
            "annotations": [
                {"id": 1, "image_id": 10, "category_id": 8, "keypoints": visible},
                {"id": 2, "image_id": 11, "category_id": 8, "keypoints": no_eye},
            ]
        }
    }
    return route, annotations


def _build(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    route, annotations = _fixture()
    monkeypatch.setattr(
        face_eligibility,
        "validate_full128_route_plan_bundle",
        lambda value, *, verify_files: value,
    )
    return build_face_eligibility_overlay(
        route,
        dogface_split=DogFaceSplitEvidence(
            train_values=(1,),
            test_values=(2,),
            train_sha256=_sha("train-classes"),
            test_sha256=_sha("test-classes"),
        ),
        ap10k_annotations_by_sha256=annotations,
    )


def test_overlay_is_complete_score_blind_and_protocol_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build(monkeypatch)

    assert validate_face_eligibility_overlay_bundle(bundle) == bundle
    census = bundle["census"]
    assert census["observation_count"] == 9
    assert census["eligible_count"] == 6
    assert census["score_inputs_used"] is False
    assert census["learned_candidate_used"] is False
    records = {
        record["dataset_name"] + ":" + record["reason"]: record
        for record in bundle["overlay"]["records"]
    }
    assert records["ap10k-dog:NOSE_AND_AT_LEAST_ONE_EYE_VISIBLE"]["status"] == (
        "ELIGIBLE"
    )
    assert records["ap10k-dog:NO_PUBLISHER_FACE_GEOMETRY"]["status"] == ("UNAVAILABLE")
    assert records["dogfacenet224:PUBLISHER_NATIVE_FACE_CROP"]["publisher_split"] in {
        "train",
        "test",
    }
    dogface_roles = {
        record["publisher_split"]: record["face_protocol_role"]
        for record in bundle["overlay"]["records"]
        if record["dataset_name"] == "dogfacenet224"
    }
    assert dogface_roles == {"train": "FIT", "test": "EXPOSED_DIAGNOSTIC"}


def test_overlay_fails_closed_on_dogface_class_multiplicity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route, annotations = _fixture()
    monkeypatch.setattr(
        face_eligibility,
        "validate_full128_route_plan_bundle",
        lambda value, *, verify_files: value,
    )

    with pytest.raises(ValueError, match="multiplicities differ"):
        build_face_eligibility_overlay(
            route,
            dogface_split=DogFaceSplitEvidence(
                train_values=(1, 1),
                test_values=(2,),
                train_sha256=_sha("train-classes"),
                test_sha256=_sha("test-classes"),
            ),
            ap10k_annotations_by_sha256=annotations,
        )


def test_overlay_tampering_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = deepcopy(_build(monkeypatch))
    record = bundle["overlay"]["records"][0]
    record["learned_candidate_used"] = True
    record_payload = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    record["record_sha256"] = content_sha256(record_payload)
    overlay = bundle["overlay"]
    overlay_payload = {
        key: value for key, value in overlay.items() if key != "overlay_sha256"
    }
    overlay["overlay_sha256"] = content_sha256(overlay_payload)
    bundle["overlay_sha256"] = overlay["overlay_sha256"]
    payload = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    bundle["bundle_sha256"] = content_sha256(payload)

    with pytest.raises(ValueError, match="score- and model-blind"):
        validate_face_eligibility_overlay_bundle(bundle)
