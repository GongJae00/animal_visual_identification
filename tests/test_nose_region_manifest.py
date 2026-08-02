from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from identity_governance.identity_registry import compute_registered_dog_id
from localization.nose_region.manifest import (
    BUNDLE_SCHEMA,
    LICENSING_LANES,
    admitted_split_for_role,
    build_nose_region_manifest,
    build_protocol_plan,
    build_summary,
    encode_png_crop,
    frontality_from_keypoints,
    normalized_box_to_pixel_box,
    read_nose_region_manifest,
    validate_nose_region_manifest_bundle,
)
from foundation.provenance import content_sha256


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _policy() -> dict:
    return {
        "minimum_detector_confidence": 0.8,
        "minimum_frontality": 0.75,
        "minimum_native_short_side": 4,
        "frontality_metric": (
            "EYE_MIDLINE_NOSE_OFFSET_ROLL_WITH_KEYPOINT_CONFIDENCE_V1"
        ),
        "crop_encoding": "PNG_RGB_LOSSLESS",
        "path_layout": "FLAT_SAMPLE_TOKEN_HASH",
    }


def _input_hashes() -> dict[str, str]:
    return {
        "assignment_payload_sha256": _sha("assignment"),
        "labels_payload_sha256": _sha("labels"),
        "localizer_checkpoint_file_sha256": _sha("localizer"),
        "registry_binding_payload_sha256": _sha("registry"),
        "source_bundle_payload_sha256": _sha("source"),
        "split_receipt_payload_sha256": _sha("receipt"),
        "yt_roi_manifest_payload_sha256": _sha("roi"),
    }


def _counts() -> dict[str, dict]:
    return {
        dataset: {
            "candidates": 1,
            "admitted": 1,
            "rejected": 0,
            "rejection_reasons": {},
        }
        for dataset in ("dogfacenet224", "mpdd", "yt-bb-dog")
    }


def _plan_counts() -> dict[str, dict]:
    return {
        dataset: {"candidates": 1, "rejected": 0, "rejection_reasons": {}}
        for dataset in ("dogfacenet224", "mpdd", "yt-bb-dog")
    }


def _records(root: Path) -> list[dict]:
    definitions = (
        (
            "dogfacenet224",
            "dogfacenet224:v1:web-folder:1",
            "DOGFACE_FIT",
            "TRAIN",
        ),
        (
            "mpdd",
            "mpdd:v1:device-capture:1",
            "MPDD_EXTERNAL_UNUSED_TRAIN_DOMAIN",
            "DEV",
        ),
        (
            "yt-bb-dog",
            "yt-bb-dog:v1:video-track:1",
            "YT_FIT",
            "TRAIN",
        ),
    )
    crop_root = root / "crops"
    crop_root.mkdir()
    records: list[dict] = []
    for dataset, dataset_identity, role, split in definitions:
        sample_token = _sha(f"sample:{dataset}")
        identity_token = _sha(f"identity:{dataset}")
        image = Image.new("RGB", (12, 10), (31, 63, 127))
        box = (2, 2, 10, 8)
        payload, dimensions = encode_png_crop(image, box)
        (crop_root / f"{sample_token}.png").write_bytes(payload)
        records.append(
            {
                "dataset_name": dataset,
                "sample_token": sample_token,
                "identity_token": identity_token,
                "registered_dog_id": compute_registered_dog_id(dataset_identity),
                "capture_session_token": _sha(f"session:{dataset}"),
                "source_sha256": _sha(f"source:{dataset}"),
                "source_width": image.width,
                "source_height": image.height,
                "crop_path": f"crops/{sample_token}.png",
                "crop_sha256": hashlib.sha256(payload).hexdigest(),
                "crop_width": dimensions[0],
                "crop_height": dimensions[1],
                "detector_confidence": 0.9,
                "frontality": 0.85,
                "nose_box_xyxy": list(box),
                "source_role": role,
                "split_role": split,
                "licensing_lane": LICENSING_LANES[dataset],
            }
        )
    return sorted(records, key=lambda record: (record["dataset_name"], record["sample_token"]))


def _bundle(root: Path) -> dict:
    hashes = _input_hashes()
    plan = build_protocol_plan(
        input_sha256s=hashes,
        policy=_policy(),
        dataset_counts=_plan_counts(),
    )
    summary = build_summary(
        input_sha256s=hashes,
        dataset_counts=_counts(),
        protocol_plan_sha256=plan["plan_sha256"],
    )
    return build_nose_region_manifest(
        records=_records(root),
        input_sha256s=hashes,
        policy=_policy(),
        summary=summary,
    )


def _rehash(bundle: dict) -> None:
    bundle["manifest_sha256"] = content_sha256(bundle["manifest"])


def test_manifest_round_trip_verifies_content_bound_pngs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    path = tmp_path / "nose-region-manifest.json"
    path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")

    manifest = read_nose_region_manifest(path)

    assert bundle["schema_version"] == BUNDLE_SCHEMA
    assert [record["dataset_name"] for record in manifest["records"]] == [
        "dogfacenet224",
        "mpdd",
        "yt-bb-dog",
    ]
    assert {record["split_role"] for record in manifest["records"]} == {
        "TRAIN",
        "DEV",
    }


def test_manifest_rejects_schema_and_content_hash_changes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    changed = deepcopy(bundle)
    changed["manifest"]["unknown"] = True
    _rehash(changed)
    with pytest.raises(ValueError, match="schema differs"):
        validate_nose_region_manifest_bundle(changed)

    changed = deepcopy(bundle)
    changed["manifest"]["records"][0]["frontality"] = 0.7
    _rehash(changed)
    with pytest.raises(ValueError, match="frontality threshold"):
        validate_nose_region_manifest_bundle(changed)

    changed = deepcopy(bundle)
    changed["manifest"]["summary"]["summary_sha256"] = "0" * 64
    _rehash(changed)
    with pytest.raises(ValueError, match="summary digest"):
        validate_nose_region_manifest_bundle(changed)


def test_only_three_protected_roles_are_admitted() -> None:
    assert admitted_split_for_role("DOGFACE_FIT", "dogfacenet224") == "TRAIN"
    assert admitted_split_for_role("YT_FIT", "yt-bb-dog") == "TRAIN"
    assert (
        admitted_split_for_role("MPDD_EXTERNAL_UNUSED_TRAIN_DOMAIN", "mpdd")
        == "DEV"
    )
    for role in (
        "DOGFACE_DEVELOPMENT",
        "DOGFACE_CALIBRATION",
        "DOGFACE_TEST",
        "YT_DEVELOPMENT",
        "YT_CALIBRATION_KNOWN",
        "YT_CALIBRATION_UNKNOWN",
        "YT_TEST_KNOWN",
        "YT_TEST_UNKNOWN",
        "MPDD_EXTERNAL_KNOWN",
        "MPDD_EXTERNAL_UNKNOWN",
        "SIBETAN_EXTERNAL_KNOWN",
        "SIBETAN_EXTERNAL_UNKNOWN",
    ):
        assert admitted_split_for_role(role, "fixture") is None
    with pytest.raises(ValueError, match="role and dataset"):
        admitted_split_for_role("YT_FIT", "dogfacenet224")


def test_manifest_rejects_protected_role_and_required_dataset_absence(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    changed = deepcopy(bundle)
    changed["manifest"]["records"][0]["source_role"] = "DOGFACE_DEVELOPMENT"
    _rehash(changed)
    with pytest.raises(ValueError, match="protected rejected role"):
        validate_nose_region_manifest_bundle(changed)

    changed = deepcopy(bundle)
    changed["manifest"]["records"] = [
        record
        for record in changed["manifest"]["records"]
        if record["dataset_name"] != "mpdd"
    ]
    _rehash(changed)
    with pytest.raises(ValueError, match="required materialized dataset is absent: mpdd"):
        validate_nose_region_manifest_bundle(changed)


def test_manifest_rejects_token_collisions_and_split_identity_overlap(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    changed = deepcopy(bundle)
    dogface = changed["manifest"]["records"][0]
    yt = changed["manifest"]["records"][2]
    yt["sample_token"] = dogface["identity_token"]
    yt["crop_path"] = f"crops/{yt['sample_token']}.png"
    _rehash(changed)
    with pytest.raises(ValueError, match="token domains collide"):
        validate_nose_region_manifest_bundle(changed)

    changed = deepcopy(bundle)
    dogface = changed["manifest"]["records"][0]
    mpdd = changed["manifest"]["records"][1]
    mpdd["identity_token"] = dogface["identity_token"]
    mpdd["registered_dog_id"] = dogface["registered_dog_id"]
    _rehash(changed)
    with pytest.raises(ValueError, match="conflicting manifest contracts"):
        validate_nose_region_manifest_bundle(changed)


def test_manifest_paths_are_relative_flat_hashes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    for unsafe in (
        "/tmp/crop.png",
        "crops/../crop.png",
        "nested/crops/crop.png",
        "crops\\crop.png",
    ):
        changed = deepcopy(bundle)
        changed["manifest"]["records"][0]["crop_path"] = unsafe
        _rehash(changed)
        with pytest.raises(ValueError, match="flat sample-token hash"):
            validate_nose_region_manifest_bundle(changed)


def test_reader_rejects_crop_substitution_and_symlink(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    path = tmp_path / "nose-region-manifest.json"
    path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    crop = tmp_path / bundle["manifest"]["records"][0]["crop_path"]
    crop.write_bytes(b"substituted")
    with pytest.raises(ValueError, match="crop hash differs"):
        read_nose_region_manifest(path)

    other_root = tmp_path / "other"
    other_root.mkdir()
    other_bundle = _bundle(other_root)
    other_path = other_root / "nose-region-manifest.json"
    other_path.write_text(json.dumps(other_bundle, sort_keys=True), encoding="utf-8")
    other_crop = other_root / other_bundle["manifest"]["records"][0]["crop_path"]
    target = other_root / "target.png"
    other_crop.rename(target)
    other_crop.symlink_to(target)
    with pytest.raises(ValueError, match="must not be a symlink"):
        read_nose_region_manifest(other_path)

    parent_root = tmp_path / "parent-link"
    parent_root.mkdir()
    parent_bundle = _bundle(parent_root)
    parent_path = parent_root / "nose-region-manifest.json"
    parent_path.write_text(json.dumps(parent_bundle, sort_keys=True), encoding="utf-8")
    (parent_root / "crops").rename(parent_root / "real-crops")
    (parent_root / "crops").symlink_to(parent_root / "real-crops", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes its manifest root"):
        read_nose_region_manifest(parent_path)


def test_pure_crop_geometry_clips_and_materializes_exact_box() -> None:
    image = Image.new("RGB", (10, 8), (10, 20, 30))
    box = normalized_box_to_pixel_box((-0.2, 0.24, 1.2, 0.76), 10, 8)
    assert box == (0, 1, 10, 7)
    payload, dimensions = encode_png_crop(image, box)
    assert dimensions == (10, 6)
    with Image.open(io.BytesIO(payload)) as crop:
        assert crop.mode == "RGB"
        assert crop.size == dimensions
    with pytest.raises(ValueError, match="non-empty"):
        normalized_box_to_pixel_box((0.5, 0.1, 0.5, 0.9), 10, 8)


def test_frontality_is_pure_normalized_keypoint_geometry() -> None:
    frontal = [
        [0.3, 0.3, 0.9],
        [0.7, 0.3, 0.9],
        [0.5, 0.5, 0.9],
        [0.5, 0.7, 0.9],
        [0.4, 0.6, 0.9],
        [0.6, 0.6, 0.9],
        [0.35, 0.6, 0.9],
        [0.65, 0.6, 0.9],
    ]
    assert frontality_from_keypoints(frontal) == pytest.approx(0.9)
    yawed = deepcopy(frontal)
    yawed[2][0] = yawed[3][0] = 0.7
    assert frontality_from_keypoints(yawed) == 0.0
    with pytest.raises(ValueError, match="eight normalized"):
        frontality_from_keypoints(frontal[:4])
