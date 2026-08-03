from __future__ import annotations

import argparse
import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from experiments.nose_observability import (
    REPORT_BUNDLE_SCHEMA,
    audit_nose_observability,
)
from experiments.sibetan_evidence import (
    build_evidence_bundle,
    build_evidence_bundle_v2,
)
from foundation.protected_io import json_document_bytes
from foundation.provenance import content_sha256
from identity_governance.identity_registry import compute_registered_dog_id
from localization.nose_region.native_yt import (
    NativeYtSample,
    build_manifest_bundle,
    process_native_sample,
)
from workflows.audit_nose_observability import run


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


SIBETAN_V1_POLICY = {
    "minimum_detector_confidence": 0.5,
    "minimum_frontality": 0.5,
    "minimum_native_short_side": 32,
    "face_margin": 1.15,
    "face_size": 224,
    "target_association": "EXACTLY_ONE_POSE_DOG_INSTANCE",
}

SIBETAN_V2_POLICY = {
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


def _v1_outcome(token: str, *, available: bool, nose: bool) -> dict[str, object]:
    return {
        "state": "AVAILABLE" if available else "UNAVAILABLE",
        "reasons": [] if available else ["NO_TARGET_DOG_INSTANCE"],
        "crop_path": f"{'nose' if nose else 'face'}_crops/{token}.png" if available else None,
        "crop_sha256": _sha(f"crop:{nose}:{token}") if available else None,
        "crop_width": 32 if available else None,
        "crop_height": 40 if available else None,
    }


def _sibetan_v1_bundle() -> dict:
    records = []
    for token, available in (("a", True), ("b", False)):
        records.append(
            {
                "sample_id": token,
                "registered_identity_id": "dog-id",
                "source_group_id": "sequence",
                "image_path": f"Sibetan/{token}.jpg",
                "image_sha256": _sha(f"image:{token}"),
                "source_width": 100,
                "source_height": 80,
                "face": _v1_outcome(token, available=available, nose=False),
                "nose": _v1_outcome(token, available=available, nose=True),
            }
        )
    return build_evidence_bundle(
        records=records,
        input_bindings={"source_manifest_sha256": _sha("source-manifest")},
        policy=SIBETAN_V1_POLICY,
    )


def _v2_face(token: str) -> dict[str, object]:
    return {
        "state": "AVAILABLE",
        "reasons": [],
        "proposal_box_xyxy": [5.2, 6.1, 44.8, 45.9],
        "source_box_xyxy": [5, 6, 45, 46],
        "upstream_quality": {"overall": 0.8},
        "quality": {"native_short_side": 40},
        "crop_path": f"face_crops/{token}.png",
        "crop_sha256": _sha(f"face:{token}"),
        "crop_width": 40,
        "crop_height": 40,
    }


def _v2_nose(token: str, *, available: bool) -> dict[str, object]:
    local_box = [10, 11, 16, 16] if available else [8, 8, 16, 16]
    source_box = [5 + local_box[0], 6 + local_box[1], 5 + local_box[2], 6 + local_box[3]]
    width = source_box[2] - source_box[0]
    height = source_box[3] - source_box[1]
    confidence = 0.8 if available else 0.4
    return {
        "state": "AVAILABLE" if available else "UNAVAILABLE",
        "reasons": [] if available else ["LOW_LOCALIZATION_CONFIDENCE"],
        "head_box_xyxy": [5, 6, 45, 46],
        "head_relative_box_xyxy": local_box,
        "source_box_xyxy": source_box,
        "keypoints": {
            "normalized_head": [[0.5, 0.5, confidence] for _ in range(8)],
            "source_xyc": [[18.0, 19.0, confidence] for _ in range(8)],
        },
        "localizer_confidence": confidence,
        "frontality": 0.75 if available else 0.2,
        "native_short_side": min(width, height),
        "quality": {
            "blur_score": 0.6 if available else 0.2,
            "contrast_score": 0.7 if available else 0.3,
            "native_short_side": min(width, height),
        },
        "crop_path": f"nose_crops/{token}.png" if available else None,
        "crop_sha256": _sha(f"nose:{token}") if available else None,
        "crop_width": width if available else None,
        "crop_height": height if available else None,
    }


def _sibetan_v2_bundle() -> dict:
    records = [
        {
            "sample_id": token,
            "registered_identity_id": "dog-id",
            "source_group_id": "sequence",
            "image_path": f"Sibetan/1/{token}.jpg",
            "image_sha256": _sha(f"image:{token}"),
            "source_width": 100,
            "source_height": 80,
            "face": _v2_face(token),
            "nose": _v2_nose(token, available=available),
        }
        for token, available in (("a", True), ("b", False))
    ]
    return build_evidence_bundle_v2(
        records=records,
        input_bindings={
            "source_manifest_path": "/evidence/source.json",
            "source_manifest_sha256": _sha("source-manifest"),
        },
        policy=SIBETAN_V2_POLICY,
    )


def _source_jpeg() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (128, 96), (31, 63, 127)).save(stream, format="JPEG", quality=95)
    return stream.getvalue()


def _yt_bundle() -> dict:
    source = _source_jpeg()
    policy = {
        "minimum_detector_confidence": 0.1,
        "minimum_frontality": 0.0,
        "minimum_native_short_side": 1,
        "maximum_mask_uncertainty": 1.0,
    }
    base_prediction = [
        [0.28, 0.25, 0.9],
        [0.72, 0.25, 0.9],
        [0.50, 0.34, 0.9],
        [0.50, 0.76, 0.9],
        [0.43, 0.64, 0.9],
        [0.57, 0.64, 0.9],
        [0.34, 0.60, 0.9],
        [0.66, 0.60, 0.9],
    ]
    records = []
    for index in range(2):
        sample = NativeYtSample(
            sample_token=_sha(f"yt-sample:{index}"),
            identity_token=_sha("yt-identity"),
            registered_dog_id=compute_registered_dog_id("yt-bb-dog:v1:video-track:7"),
            source_sample_id=f"yt-bb-dog:v1:original:video-track:7:frame:{index}",
            sequence_token=_sha("yt-sequence"),
            track_token=_sha("yt-track"),
            frame_index=index,
            source_role="YT_FIT",
            member_path=f"YT-BB-Dog/train/7/7_{index}.jpg",
            member_crc32=1,
            member_uncompressed_bytes=len(source),
            container_member_path="YT-BB-dog/YT-BB-Dog.zip",
            container_member_crc32=2,
            container_member_uncompressed_bytes=len(source) * 2,
            expected_source_sha256=hashlib.sha256(source).hexdigest(),
            roi_metadata_available=True,
        )
        prediction = deepcopy(base_prediction)
        if index:
            for point in prediction[2:]:
                point[0] += 0.03
        record, _ = process_native_sample(sample, source, prediction, policy=policy)
        records.append(record)
    records.sort(key=lambda row: row["sample_token"])
    return build_manifest_bundle(
        records=records,
        input_sha256s={"source_manifest_sha256": _sha("yt-source-manifest")},
        policy=policy,
        tool_provenance={"schema_version": "synthetic.fixture.v1"},
    )


def _rehash(bundle: dict) -> None:
    bundle["manifest_sha256"] = content_sha256(bundle["manifest"])


def test_sibetan_v2_reports_sizes_scales_instability_and_explicit_proxy_boundary() -> None:
    report = audit_nose_observability(_sibetan_v2_bundle())

    assert report["input_contract"]["input_format"] == "SIBETAN_MULTIEVIDENCE_V2"
    assert report["availability_and_rejections"]["nose"] == {
        "state_counts": {"AVAILABLE": 1, "UNAVAILABLE": 1},
        "rejection_reason_counts": {"LOW_LOCALIZATION_CONFIDENCE": 1},
    }
    sizes = report["native_materialized_crop_size"]
    assert sizes["materialized_crop_count"] == 1
    assert sizes["width_pixels"]["median"] == 6.0
    assert sizes["height_pixels"]["median"] == 5.0
    scale = report["reference_resize_scale_proxy"]
    assert scale["short_side_upsampling_factor_floor_one"]["median"] == pytest.approx(44.8)
    assert scale["square_resize_anisotropic_scale_ratio"]["median"] == pytest.approx(1.2)
    correspondence = report["crop_correspondence_proxy"]
    assert correspondence["available"] is True
    assert correspondence["repeated_source_group_count"] == 1
    assert correspondence["records_in_repeated_source_groups"] == 2
    assert correspondence["metrics_are_crop_instability_proxies_not_anatomical_correspondence"] is True
    topology = report["topology_observability_proxy"]
    assert topology["category_counts"][
        "TOPOLOGY_OBSERVABILITY_PROXY_NATIVE_SHORT_SIDE_LT_32_PX"
    ] == 2
    assert topology["pixels_never_equated_with_visible_ridge_topology"] is True
    assert topology["manual_annotation_required_for_any_topology_claim"] is True


def test_sibetan_v1_reports_unavailable_coordinate_and_quality_strata() -> None:
    report = audit_nose_observability(_sibetan_v1_bundle())

    assert report["input_contract"]["input_format"] == "SIBETAN_MULTIEVIDENCE_V1"
    assert report["native_materialized_crop_size"]["short_side_fixed_strata"][
        "GE_32_LT_64_PX"
    ] == 1
    assert report["crop_correspondence_proxy"]["available"] is False
    assert report["crop_correspondence_proxy"]["coordinate_record_count"] == 0
    assert report["quality_frontality_confidence_strata"]["confidence"]["summary"] == {
        "available": False,
        "count": 0,
        "reason": "NO_VALUES",
    }


def test_yt_native_manifest_uses_track_coordinate_correspondence_proxy() -> None:
    report = audit_nose_observability(_yt_bundle())

    assert report["input_contract"]["input_format"] == "YT_NATIVE_NOSE_V1"
    assert report["input_contract"]["sample_token_field"] == "sample_token"
    assert report["availability_and_rejections"]["nose"]["state_counts"] == {
        "AVAILABLE": 2
    }
    assert report["crop_correspondence_proxy"]["grouping_field"] == "track_token"
    assert report["crop_correspondence_proxy"]["coordinate_record_count"] == 2
    assert report["quality_frontality_confidence_strata"]["confidence"]["summary"][
        "count"
    ] == 2


def test_audit_is_deterministic_for_same_manifest() -> None:
    bundle = _sibetan_v2_bundle()

    first = audit_nose_observability(bundle)
    second = audit_nose_observability(deepcopy(bundle))

    assert first == second
    assert content_sha256(first) == content_sha256(second)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda bundle: bundle.__setitem__("schema_version", "cvi.unknown.v1"),
            "unsupported Nose evidence bundle schema",
        ),
        (
            lambda bundle: bundle["manifest"].__setitem__("input_bindings", {}),
            "required lineage is missing",
        ),
        (
            lambda bundle: bundle["manifest"]["records"][0]["nose"].__setitem__(
                "crop_width", 0
            ),
            "crop dimensions differ",
        ),
        (
            lambda bundle: bundle["manifest"]["records"][1].__setitem__(
                "sample_id", "a"
            ),
            "repeats a sample token",
        ),
    ),
)
def test_invalid_schema_lineage_dimensions_and_duplicate_tokens_fail_closed(
    mutation, message: str
) -> None:
    bundle = _sibetan_v1_bundle()
    mutation(bundle)
    if bundle["schema_version"] != "cvi.unknown.v1":
        _rehash(bundle)

    with pytest.raises(ValueError, match=message):
        audit_nose_observability(bundle)


def test_quality_values_outside_declared_strata_fail_closed() -> None:
    bundle = _sibetan_v2_bundle()
    bundle["manifest"]["records"][0]["nose"]["quality"]["blur_score"] = 1.1
    _rehash(bundle)

    with pytest.raises(ValueError, match=r"quality.blur_score must be in \[0, 1\]"):
        audit_nose_observability(bundle)


def test_cli_binds_file_and_code_writes_canonical_json_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "evidence.json"
    source.write_bytes(json_document_bytes(_sibetan_v1_bundle()))
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "audit.json"
    args = argparse.Namespace(
        manifest=source,
        manifest_file_sha256=expected_sha,
        output=output,
    )

    result = run(args)

    assert result["schema_version"] == REPORT_BUNDLE_SCHEMA
    assert result["report_sha256"] == content_sha256(result["report"])
    assert result["report"]["source_binding"]["file_sha256"] == expected_sha
    assert result["report"]["source_binding"]["canonical_payload_sha256"] == content_sha256(
        json.loads(source.read_text(encoding="utf-8"))
    )
    assert output.read_bytes() == json_document_bytes(result)
    assert {
        "experiments/nose_observability.py",
        "experiments/sibetan_evidence.py",
        "localization/nose_region/native_yt.py",
        "workflows/audit_nose_observability.py",
    }.issubset(result["report"]["tool_provenance"]["code_sha256s"])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(args)


def test_cli_rejects_unpinned_manifest_bytes(tmp_path: Path) -> None:
    source = tmp_path / "evidence.json"
    source.write_bytes(json_document_bytes(_sibetan_v1_bundle()))

    with pytest.raises(ValueError, match="differs from external pin"):
        run(
            argparse.Namespace(
                manifest=source,
                manifest_file_sha256="0" * 64,
                output=tmp_path / "audit.json",
            )
        )
