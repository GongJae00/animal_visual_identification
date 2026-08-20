from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from enrollment.registry.identity_registry import (
    compute_identity_token,
    compute_registered_dog_id,
    compute_sample_token,
    compute_sequence_token,
)
from parsing.export.regions.native_yt import (
    BUNDLE_SCHEMA,
    MANIFEST_SCHEMA,
    NativeYtSample,
    build_manifest_bundle,
    process_native_sample,
    read_nested_member_bytes,
    validate_manifest_bundle,
)
from shared.foundation.provenance import content_sha256
from evaluation.splits.protected_public_split import PublicSplitSample, PublicSplitSourceBundle
from archive.nose.commands.extract_yt_native_nose_regions import (
    _teacher_source_record,
    _validate_split_inputs,
)

def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()

def _source_jpeg(size: tuple[int, int] = (640, 640)) -> bytes:
    y, x = np.indices((size[1], size[0]))
    rgb = np.stack(
        (
            (x * 3 + y) % 256,
            (x + y * 2) % 256,
            ((x // 8) * 17 + (y // 8) * 11) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(stream, format="JPEG", quality=91)
    return stream.getvalue()

def _nested_archive(tmp_path: Path) -> tuple[Path, NativeYtSample, bytes]:
    source = _source_jpeg()
    member_path = "YT-BB-Dog/train/7/7_11.jpg"
    nested_stream = io.BytesIO()
    with zipfile.ZipFile(nested_stream, "w", compression=zipfile.ZIP_STORED) as nested:
        nested.writestr(member_path, source)
        member = nested.getinfo(member_path)
    container_path = "YT-BB-dog/YT-BB-Dog.zip"
    archive_path = (tmp_path / "YT-BB-dog.zip").resolve()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as outer:
        outer.writestr(container_path, nested_stream.getvalue())
        container = outer.getinfo(container_path)
    sample = NativeYtSample(
        sample_token=_sha("sample"),
        identity_token=_sha("identity"),
        registered_dog_id=compute_registered_dog_id("yt-bb-dog:v1:video-track:7"),
        source_sample_id="yt-bb-dog:v1:original:video-track:7:frame:11",
        sequence_token=_sha("sequence"),
        track_token=_sha("track"),
        frame_index=11,
        source_role="YT_FIT",
        member_path=member_path,
        member_crc32=member.CRC,
        member_uncompressed_bytes=member.file_size,
        container_member_path=container_path,
        container_member_crc32=container.CRC,
        container_member_uncompressed_bytes=container.file_size,
        expected_source_sha256=hashlib.sha256(source).hexdigest(),
        roi_metadata_available=True,
    )
    return archive_path, sample, source

def _prediction(confidence: float = 0.92) -> list[list[float]]:
    return [
        [0.28, 0.25, confidence],
        [0.72, 0.25, confidence],
        [0.50, 0.34, confidence],
        [0.50, 0.76, confidence],
        [0.43, 0.64, confidence],
        [0.57, 0.64, confidence],
        [0.34, 0.60, confidence],
        [0.66, 0.60, confidence],
    ]

def _policy(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "minimum_detector_confidence": 0.5,
        "minimum_frontality": 0.5,
        "minimum_native_short_side": 32,
        "maximum_mask_uncertainty": 1.0,
    }
    result.update(changes)
    return result

def test_reads_original_nested_bytes_and_preserves_native_resolution_and_hashes(
    tmp_path: Path,
) -> None:
    archive_path, sample, source = _nested_archive(tmp_path)

    observed = read_nested_member_bytes(archive_path, sample)
    record, artifacts = process_native_sample(
        sample, observed, _prediction(), policy=_policy()
    )

    assert observed == source
    assert record["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert (record["source_width"], record["source_height"]) == (640, 640)
    assert (record["crop_width"], record["crop_height"]) != (224, 224)
    assert record["source_bytes_role"] == "ORIGINAL_PUBLISHER_DOG_CROP"
    assert record["intermediaries_used"] == []
    assert record["source_archive_member"] == sample.member_path
    for path_field, hash_field in (
        ("crop_path", "crop_sha256"),
        ("soft_mask_path", "soft_mask_sha256"),
        ("binary_mask_path", "binary_mask_sha256"),
    ):
        assert record[hash_field] == hashlib.sha256(artifacts[record[path_field]]).hexdigest()

def test_builds_exact_sam2_teacher_source_record(tmp_path: Path) -> None:
    archive_path, sample, source = _nested_archive(tmp_path)
    record, _ = process_native_sample(
        sample,
        read_nested_member_bytes(archive_path, sample),
        _prediction(),
        policy=_policy(),
    )
    source_path = f"teacher_source_images/{sample.sample_token}.jpg"
    result = _teacher_source_record(record, source_path)

    assert result == {
        "sample_token": sample.sample_token,
        "sequence_token": sample.sequence_token,
        "track_token": sample.track_token,
        "frame_index": sample.frame_index,
        "source_image_path": source_path,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_width": 640,
        "source_height": 640,
        "nose_box_xyxy": record["nose_box_xyxy"],
        "keypoints": record["keypoints"],
    }

def test_automatic_mask_geometry_and_quality_outputs(tmp_path: Path) -> None:
    archive_path, sample, _ = _nested_archive(tmp_path)
    record, artifacts = process_native_sample(
        sample,
        read_nested_member_bytes(archive_path, sample),
        _prediction(),
        policy=_policy(),
    )
    with Image.open(io.BytesIO(artifacts[record["soft_mask_path"]])) as soft:
        soft_values = np.asarray(soft)
    with Image.open(io.BytesIO(artifacts[record["binary_mask_path"]])) as binary:
        binary_values = np.asarray(binary)

    assert record["mask_method"] in {
        "KEYPOINT_GEOMETRY",
        "KEYPOINT_GEOMETRY_GRABCUT",
    }
    assert soft_values.shape == (record["crop_height"], record["crop_width"])
    assert 0 < np.count_nonzero(binary_values) < binary_values.size
    assert binary_values[binary_values.shape[0] // 2, binary_values.shape[1] // 2] == 255
    assert binary_values[0, 0] == 0
    assert len(np.unique(soft_values)) > 2
    expected_quality = {
        "blur_laplacian_variance",
        "blur_score",
        "saturation_mean",
        "clipped_pixel_fraction",
        "specular_fraction",
        "contrast_rms",
        "contrast_score",
        "jpeg_blocking_score",
        "noise_score",
        "native_short_side",
        "mask_uncertainty",
        "mask_available",
        "detector_confidence",
        "frontality",
    }
    assert set(record["quality"]) == expected_quality
    assert record["quality"]["mask_available"] is True
    assert record["quality"]["native_short_side"] == min(
        record["crop_width"], record["crop_height"]
    )
    assert all(
        np.isfinite(value)
        for name, value in record["quality"].items()
        if name != "mask_available" and value is not None
    )

def test_rejects_unsafe_nested_member_and_source_hash_substitution(tmp_path: Path) -> None:
    archive_path, sample, source = _nested_archive(tmp_path)
    values = {
        field: getattr(sample, field)
        for field in sample.__dataclass_fields__
    }
    values["member_path"] = "../7_11.jpg"
    with pytest.raises(ValueError, match="ZIP member path is unsafe"):
        NativeYtSample(**values)

    values["member_path"] = sample.member_path
    values["expected_source_sha256"] = "0" * 64
    substituted = NativeYtSample(**values)
    with pytest.raises(ValueError, match="source member SHA-256 differs"):
        read_nested_member_bytes(archive_path, substituted)
    assert hashlib.sha256(source).hexdigest() != "0" * 64

def test_low_quality_and_no_roi_rows_are_retained_for_quality_ssl(tmp_path: Path) -> None:
    archive_path, sample, source = _nested_archive(tmp_path)
    low_record, low_artifacts = process_native_sample(
        sample,
        read_nested_member_bytes(archive_path, sample),
        _prediction(confidence=0.55),
        policy=_policy(
            minimum_detector_confidence=0.9,
            minimum_frontality=0.9,
            minimum_native_short_side=500,
        ),
    )
    no_roi_record, no_roi_artifacts = process_native_sample(
        sample, source, None, policy=_policy()
    )

    assert low_record["record_state"] == "LOW_QUALITY"
    assert low_record["usage"] == "QUALITY_SSL_ONLY"
    assert low_artifacts
    assert "LOW_DETECTOR_CONFIDENCE" in low_record["quality_flags"]
    assert "LOW_NATIVE_RESOLUTION" in low_record["quality_flags"]
    assert no_roi_record["record_state"] == "NO_ROI"
    assert no_roi_record["usage"] == "QUALITY_SSL_ONLY"
    assert no_roi_record["quality_flags"] == ["NO_ROI"]
    assert no_roi_record["crop_path"] is None
    assert no_roi_record["quality"]["mask_available"] is False
    assert no_roi_artifacts == {}

def test_exact_teacher_mask_hook_does_not_claim_sam(tmp_path: Path) -> None:
    archive_path, sample, source = _nested_archive(tmp_path)
    teacher_values = np.zeros((640, 640), dtype=np.uint8)
    teacher_values[250:430, 220:420] = 173
    teacher = Image.fromarray(teacher_values, mode="L")

    record, artifacts = process_native_sample(
        sample,
        read_nested_member_bytes(archive_path, sample),
        _prediction(),
        policy=_policy(),
        teacher_mask=teacher,
        teacher_mask_uncertainty=0.2,
    )
    box = record["nose_box_xyxy"]
    expected = teacher_values[box[1] : box[3], box[0] : box[2]]
    with Image.open(io.BytesIO(artifacts[record["soft_mask_path"]])) as soft:
        assert np.array_equal(np.asarray(soft), expected)
    assert record["mask_method"] == "EXTERNAL_EXACT_TEACHER"
    assert record["quality"]["mask_uncertainty"] == 0.2
    assert b"SAM" not in b"".join(artifacts.values())
    assert hashlib.sha256(source).hexdigest() == record["source_sha256"]

def test_manifest_binds_states_policy_inputs_and_provenance(tmp_path: Path) -> None:
    archive_path, sample, source = _nested_archive(tmp_path)
    available, artifacts = process_native_sample(
        sample,
        read_nested_member_bytes(archive_path, sample),
        _prediction(),
        policy=_policy(),
    )
    values = {
        field: getattr(sample, field)
        for field in sample.__dataclass_fields__
    }
    values["sample_token"] = _sha("second-sample")
    second = NativeYtSample(**values)
    no_roi, _ = process_native_sample(second, source, None, policy=_policy())
    bundle = build_manifest_bundle(
        records=sorted((available, no_roi), key=lambda row: row["sample_token"]),
        input_sha256s={"archive": _sha("archive")},
        policy=_policy(),
        tool_provenance={"schema_version": "fixture"},
    )
    for relative, payload in artifacts.items():
        target = tmp_path / relative
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(payload)

    assert bundle["schema_version"] == BUNDLE_SCHEMA
    assert bundle["manifest"]["schema_version"] == MANIFEST_SCHEMA
    assert bundle["manifest_sha256"] == content_sha256(bundle["manifest"])
    assert bundle["manifest"]["record_counts"] == {
        "AVAILABLE": 1,
        "LOW_QUALITY": 0,
        "NO_ROI": 1,
    }
    assert "ROI_FACE_CROP_JPEG" in bundle["manifest"]["intermediaries_prohibited"]
    assert validate_manifest_bundle(bundle, root=tmp_path)["records"] == bundle["manifest"]["records"]

def test_assignment_source_identity_and_yt_fit_role_are_content_bound() -> None:
    source_sample_id = "yt-bb-dog:v1:original:video-track:7:frame:11"
    dataset_identity_id = "yt-bb-dog:v1:video-track:7"
    identity_token = compute_identity_token(dataset_identity_id)
    sample_token = compute_sample_token(source_sample_id)
    source = PublicSplitSourceBundle(
        evidence_bindings=tuple(
            (name, _sha(name))
            for name in (
                "exact_duplicate_graph_sha256",
                "geometric_verifier_sha256",
                "image_content_receipts_sha256",
                "pdq_candidates_sha256",
                "phash_candidates_sha256",
                "review_adjudication_sha256",
                "semantic_receipts_sha256",
            )
        ),
        samples=(PublicSplitSample(
            sample_token=sample_token,
            identity_token=identity_token,
            sequence_token=compute_sequence_token(
                "yt-bb-dog:v1:video-track:7", identity_token
            ),
            source_sample_id=source_sample_id,
            dataset_identity_id=dataset_identity_id,
            dataset_name="yt-bb-dog",
            source_variant="original",
            original_split="train",
            raw_frame_index=11,
            paired_source_sample_id=None,
            in_no_mono_subset=None,
            region="DOG_CROP",
        ),),
    )
    assignment = {
        "schema_version": "evaluation.protected_public_split_assignment.v1",
        "status": "PASS_PROTECTED_SPLIT_CONSTRUCTION",
        "records": [{
            "sample_token": sample_token,
            "identity_token": identity_token,
            "dataset_name": "yt-bb-dog",
            "source_variant": "original",
            "identity_role": "YT_FIT",
            "model_access": "MODEL_TRAINING",
            "sample_disposition": "PRIMARY_ORACLE_CROP",
        }],
    }
    receipt = {
        "schema_version": "evaluation.protected_public_split_receipt.v3",
        "status": "PASS_PROTECTED_SPLIT_CONSTRUCTION",
        "assignment_sha256": content_sha256(assignment),
        "source_bundle_sha256": source.bundle_sha256,
    }
    receipt["receipt_sha256"] = content_sha256(receipt)

    _, selected = _validate_split_inputs(
        assignment, source.to_dict(), receipt, receipt["receipt_sha256"]
    )
    assert selected[0]["assignment"]["identity_role"] == "YT_FIT"

    changed = {**assignment, "records": [dict(assignment["records"][0])]}
    changed["records"][0]["identity_token"] = "0" * 64
    changed_receipt = dict(receipt)
    changed_receipt["assignment_sha256"] = content_sha256(changed)
    changed_receipt["receipt_sha256"] = content_sha256(
        {key: value for key, value in changed_receipt.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="identity/source binding differs"):
        _validate_split_inputs(
            changed,
            source.to_dict(),
            changed_receipt,
            changed_receipt["receipt_sha256"],
        )
