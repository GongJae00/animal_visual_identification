from __future__ import annotations

import copy

import pytest

from legacy.version.common.experiments.sibetan_evidence import (
    build_evidence_bundle,
    build_evidence_bundle_v2,
    validate_evidence_bundle,
    validate_evidence_bundle_v2,
)


POLICY = {
    "minimum_detector_confidence": 0.5,
    "minimum_frontality": 0.5,
    "minimum_native_short_side": 32,
    "face_margin": 1.15,
    "face_size": 224,
    "target_association": "EXACTLY_ONE_POSE_DOG_INSTANCE",
}


def _outcome(available: bool) -> dict[str, object]:
    return {
        "state": "AVAILABLE" if available else "UNAVAILABLE",
        "reasons": [] if available else ["NO_TARGET_DOG_INSTANCE"],
        "crop_path": "crops/a.png" if available else None,
        "crop_sha256": "1" * 64 if available else None,
        "crop_width": 32 if available else None,
        "crop_height": 40 if available else None,
    }


def _record(sample_id: str, *, face: bool, nose: bool) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "registered_identity_id": "dog-id",
        "source_group_id": "sequence",
        "image_path": f"Sibetan/{sample_id}.jpg",
        "image_sha256": "0" * 64,
        "source_width": 100,
        "source_height": 80,
        "face": _outcome(face),
        "nose": _outcome(nose),
    }


def test_bundle_requires_explicit_sorted_outcomes_and_counts() -> None:
    bundle = build_evidence_bundle(
        records=[
            _record("a", face=True, nose=False),
            _record("b", face=False, nose=True),
        ],
        input_bindings={"fixture": "value"},
        policy=POLICY,
    )

    manifest = validate_evidence_bundle(bundle)
    assert manifest["record_count"] == 2
    assert manifest["state_counts"] == {
        "face": {"AVAILABLE": 1, "UNAVAILABLE": 1},
        "nose": {"AVAILABLE": 1, "UNAVAILABLE": 1},
    }


def test_bundle_rejects_rehashed_missing_reason_and_artifact_tamper() -> None:
    bundle = build_evidence_bundle(
        records=[_record("a", face=True, nose=False)],
        input_bindings={"fixture": "value"},
        policy=POLICY,
    )
    changed = copy.deepcopy(bundle)
    changed["manifest"]["records"][0]["nose"]["reasons"] = []
    from foundation.provenance import content_sha256

    changed["manifest_sha256"] = content_sha256(changed["manifest"])
    with pytest.raises(ValueError, match="availability differs"):
        validate_evidence_bundle(changed)

    changed = copy.deepcopy(bundle)
    changed["manifest"]["records"][0]["face"]["crop_path"] = "../escape.png"
    changed["manifest_sha256"] = content_sha256(changed["manifest"])
    with pytest.raises(ValueError, match="unsafe"):
        validate_evidence_bundle(changed)


def test_bundle_rejects_unfrozen_policy() -> None:
    policy = {**POLICY, "minimum_frontality": 0.7}
    with pytest.raises(ValueError, match="policy differs"):
        build_evidence_bundle(
            records=[_record("a", face=True, nose=True)],
            input_bindings={"fixture": "value"},
            policy=policy,
        )


def test_v2_keeps_tiny_profile_nose_as_available_continuous_quality() -> None:
    policy = {
        "target_association": "EXACTLY_ONE_POSE_DOG_INSTANCE",
        "head_geometry": "ROI_MANIFEST_FACE_CROP_RECT_XYXY",
        "localizer_input": "RAW_SOURCE_RGB_HEAD_RECT",
        "localizer_resize": "BILINEAR_STRETCH_224X224",
        "nose_margin": 0.08,
        "minimum_localizer_confidence": 0.5,
        "frontality_admission": "NONE_CONTINUOUS_QUALITY",
        "native_short_side_admission": "NONE_CONTINUOUS_QUALITY",
        "crop_encoding": "PNG_RGB_LOSSLESS_FROM_DECODED_SOURCE",
    }
    face = {
        "state": "AVAILABLE", "reasons": [],
        "proposal_box_xyxy": [5.2, 6.1, 30.8, 31.9],
        "source_box_xyxy": [5, 6, 31, 32], "upstream_quality": {"overall": 0.4},
        "quality": {"native_short_side": 26}, "crop_path": "face_crops/a.png",
        "crop_sha256": "1" * 64, "crop_width": 26, "crop_height": 26,
    }
    nose = {
        "state": "AVAILABLE", "reasons": [], "head_box_xyxy": [5, 6, 31, 32],
        "head_relative_box_xyxy": [10, 11, 16, 16],
        "source_box_xyxy": [15, 17, 21, 22],
        "keypoints": {
            "normalized_head": [[0.5, 0.5, 0.7] for _ in range(8)],
            "source_xyc": [[18.0, 19.0, 0.7] for _ in range(8)],
        },
        "localizer_confidence": 0.7, "frontality": 0.0, "native_short_side": 5,
        "quality": {"native_short_side": 5, "frontality": 0.0},
        "crop_path": "nose_crops/a.png", "crop_sha256": "2" * 64,
        "crop_width": 6, "crop_height": 5,
    }
    bundle = build_evidence_bundle_v2(
        records=[{
            "sample_id": "a", "registered_identity_id": "dog-id",
            "source_group_id": "sequence", "image_path": "Sibetan/1/a.jpg",
            "image_sha256": "0" * 64, "source_width": 100, "source_height": 80,
            "face": face, "nose": nose,
        }],
        input_bindings={"fixture": "v2"}, policy=policy,
    )
    manifest = validate_evidence_bundle_v2(bundle)
    assert manifest["state_counts"]["nose"] == {"AVAILABLE": 1}
    assert manifest["records"][0]["nose"]["frontality"] == 0.0
    assert manifest["records"][0]["nose"]["native_short_side"] == 5
