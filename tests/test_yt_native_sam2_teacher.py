from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from cvi.identity_registry import compute_registered_dog_id
from cvi.nose_region.localizer import KEYPOINT_ORDER
from cvi.nose_region.native_yt import (
    NativeYtSample,
    build_manifest_bundle,
    process_native_sample,
)
from cvi.nose_region.sam2_teacher import (
    MaskSelectionPolicy,
    SOURCE_IMAGE_MANIFEST_SCHEMA,
    TeacherSource,
    produce_teacher_manifest,
    sources_from_native_manifest,
    validate_sam2_artifacts,
    validate_source_image_manifest,
    validate_teacher_manifest,
)
from cvi.nose_region.sam2_teacher import _sam2_runtime_config_name
from cvi.protected_io import json_document_bytes
from cvi.provenance import content_sha256
from tools.extract_yt_native_nose_regions import (
    _load_teacher,
    _teacher_records,
    _teacher_uncertainty,
)
from tools.produce_yt_native_nose_teacher_masks import run as run_tool


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _source_bytes(color: int = 80) -> bytes:
    y, x = np.indices((80, 96))
    rgb = np.stack(
        ((x + color) % 256, (y * 2 + color) % 256, (x + y + color) % 256),
        axis=2,
    ).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


def _source(token_name: str = "sample", *, frame_index: int = 3, color: int = 80) -> TeacherSource:
    payload = _source_bytes(color)
    return TeacherSource(
        sample_token=_sha(token_name),
        sequence_token=_sha("sequence"),
        track_token=_sha("track"),
        frame_index=frame_index,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_width=96,
        source_height=80,
        nose_box_xyxy=(24, 24, 72, 64),
        positive_keypoints_xy=(
            (48.0, 30.0),
            (48.0, 54.0),
            (42.0, 49.0),
            (54.0, 49.0),
            (36.0, 46.0),
            (60.0, 46.0),
        ),
        source_bytes=payload,
    )


def _good_mask(offset: int = 0) -> np.ndarray:
    mask = np.zeros((80, 96), dtype=np.uint8)
    cv2.ellipse(mask, (48 + offset, 46), (18, 16), 0, 0, 360, 1, -1)
    return mask


def _producer() -> dict[str, object]:
    tool = {"schema_version": "fixture.tool.v1"}
    return {
        "model_name": "sam2.1",
        "sam2_checkout_commit": "1" * 40,
        "sam2_python_sources_sha256": _sha("sources"),
        "sam2_config_relative_path": "configs/sam2.1/fixture.yaml",
        "sam2_config_sha256": _sha("config"),
        "sam2_checkpoint_filename": "sam2.1_fixture.pt",
        "sam2_checkpoint_sha256": _sha("checkpoint"),
        "license_id": "Apache-2.0",
        "license_snapshot_sha256": _sha("license"),
        "device": "cpu",
        "prompt_contract": "NOSE_BOX_AND_POSITIVE_NOSE_KEYPOINTS",
        "output_encoding": "SOURCE_RESOLUTION_BINARY_L_PNG",
        "tool_provenance": tool,
        "tool_provenance_sha256": content_sha256(tool),
    }


def _source_binding() -> dict[str, str]:
    return {
        "source_manifest_schema": SOURCE_IMAGE_MANIFEST_SCHEMA,
        "source_manifest_file_sha256": _sha("manifest-file"),
        "source_manifest_payload_sha256": _sha("manifest-payload"),
        "source_receipt_filename": "receipt.json",
        "source_receipt_file_sha256": _sha("receipt"),
    }


class _StrictPredictor:
    def __init__(self, outputs: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.outputs = list(outputs)
        self.images: list[np.ndarray] = []

    def set_image(self, image: np.ndarray) -> None:
        assert image.shape == (80, 96, 3)
        assert image.dtype == np.uint8
        self.images.append(image.copy())

    def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, None]:
        assert set(kwargs) == {
            "point_coords",
            "point_labels",
            "box",
            "multimask_output",
        }
        assert np.array_equal(kwargs["box"], np.asarray((24, 24, 72, 64), dtype=np.float32))
        assert np.array_equal(kwargs["point_labels"], np.ones(6, dtype=np.int32))
        assert np.asarray(kwargs["point_coords"]).shape == (6, 2)
        assert kwargs["multimask_output"] is True
        masks, scores = self.outputs.pop(0)
        return masks, scores, None


def test_selects_by_score_anatomy_and_compactness_and_emits_exact_png() -> None:
    source = _source()
    outside = np.zeros((80, 96), dtype=np.uint8)
    outside[0:35, 0:35] = 1
    predictor = _StrictPredictor(
        [(np.stack((outside, _good_mask())), np.asarray((0.99, 0.86)))]
    )

    manifest, artifacts = produce_teacher_manifest(
        (source,),
        predictor,
        source_binding=_source_binding(),
        producer=_producer(),
    )

    record = manifest["records"][0]
    assert record["status"] == "ACCEPTED"
    assert record["selection"]["selected_candidate_index"] == 1
    assert record["source_sha256"] == hashlib.sha256(source.source_bytes).hexdigest()
    assert manifest["producer"]["sam2_checkpoint_sha256"] == _sha("checkpoint")
    payload = artifacts[record["mask_path"]]
    assert record["mask_sha256"] == hashlib.sha256(payload).hexdigest()
    with Image.open(io.BytesIO(payload)) as image:
        assert (image.format, image.mode, image.size) == ("PNG", "L", (96, 80))
        assert set(np.unique(np.asarray(image))) == {0, 255}
    assert validate_teacher_manifest(manifest, artifacts=artifacts) is manifest


def test_ambiguous_masks_are_rejected_without_artifact(tmp_path: Path) -> None:
    source = _source()
    predictor = _StrictPredictor(
        [
            (
                np.stack((_good_mask(-1), _good_mask(1))),
                np.asarray((0.90, 0.89)),
            )
        ]
    )

    manifest, artifacts = produce_teacher_manifest(
        (source,),
        predictor,
        source_binding=_source_binding(),
        producer=_producer(),
        policy=MaskSelectionPolicy(ambiguity_margin=0.10),
    )

    record = manifest["records"][0]
    assert record["status"] == "REJECTED"
    assert record["rejection_reasons"] == ["AMBIGUOUS_MASKS"]
    assert record["mask_path"] is None
    assert record["mask_sha256"] is None
    assert artifacts == {}
    manifest_path = tmp_path / "teacher.json"
    manifest_path.write_bytes(json_document_bytes(manifest))
    hook_records, _ = _teacher_records(manifest_path.resolve())
    assert hook_records[source.sample_token]["status"] == "REJECTED"
    assert "bytes" not in hook_records[source.sample_token]


class _PropagatingPredictor(_StrictPredictor):
    video_api_available = True

    def __init__(self) -> None:
        empty = np.zeros((80, 96), dtype=np.uint8)
        super().__init__(
            [
                (np.stack((_good_mask(),)), np.asarray((0.91,))),
                (np.stack((empty,)), np.asarray((0.99,))),
            ]
        )
        self.propagated = False

    def propagate_track(self, **kwargs: object) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        self.propagated = True
        assert kwargs["seed_index"] == 0
        assert len(kwargs["images"]) == 2
        assert np.array_equal(kwargs["seed_mask"], _good_mask().astype(bool))
        return {1: (np.stack((_good_mask(),)), np.asarray((0.88,)))}


def test_optional_video_api_uses_ordered_track_frames() -> None:
    first = _source("b", frame_index=3, color=80)
    second = _source("a", frame_index=7, color=90)
    predictor = _PropagatingPredictor()

    manifest, artifacts = produce_teacher_manifest(
        tuple(sorted((first, second), key=lambda item: item.sample_token)),
        predictor,
        source_binding=_source_binding(),
        producer=_producer(),
        propagate_tracks=True,
    )

    assert predictor.propagated is True
    assert manifest["propagation"] == {
        "requested": True,
        "api_available": True,
        "frame_runs_attempted": 1,
        "frame_runs_propagated": 1,
        "score_semantics": "VIDEO_LOGIT_FOREGROUND_MEAN",
    }
    second_record = next(row for row in manifest["records"] if row["frame_index"] == 7)
    assert second_record["status"] == "ACCEPTED"
    assert second_record["selection"]["candidates"][1]["origin"] == "VIDEO_PROPAGATION"
    assert len(artifacts) == 2


def test_source_manifest_validates_receipt_geometry_and_source_hash(tmp_path: Path) -> None:
    source = _source()
    image_path = tmp_path / "images" / "source.png"
    image_path.parent.mkdir()
    image_path.write_bytes(source.source_bytes)
    all_points = (
        (34.0, 18.0),
        (62.0, 18.0),
        *source.positive_keypoints_xy,
    )
    keypoints = [
        {
            "name": name,
            "normalized_x": x / 96,
            "normalized_y": y / 80,
            "source_x": x,
            "source_y": y,
            "confidence": 0.9,
        }
        for name, (x, y) in zip(KEYPOINT_ORDER, all_points, strict=True)
    ]
    row = {
        "sample_token": source.sample_token,
        "sequence_token": source.sequence_token,
        "track_token": source.track_token,
        "frame_index": source.frame_index,
        "source_image_path": "images/source.png",
        "source_sha256": source.source_sha256,
        "source_width": source.source_width,
        "source_height": source.source_height,
        "nose_box_xyxy": list(source.nose_box_xyxy),
        "keypoints": keypoints,
    }
    payload = {
        "schema_version": SOURCE_IMAGE_MANIFEST_SCHEMA,
        "source_receipt_file_sha256": _sha("receipt"),
        "records": [row],
    }

    assert validate_source_image_manifest(
        payload, root=tmp_path, source_receipt_file_sha256=_sha("receipt")
    ) == (source,)
    with pytest.raises(ValueError, match="receipt differs"):
        validate_source_image_manifest(
            payload, root=tmp_path, source_receipt_file_sha256=_sha("other")
        )
    changed = json.loads(json.dumps(payload))
    changed["records"][0]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="image SHA-256"):
        validate_source_image_manifest(
            changed, root=tmp_path, source_receipt_file_sha256=_sha("receipt")
        )
    changed = json.loads(json.dumps(payload))
    changed["records"][0]["keypoints"][2]["source_x"] += 1
    with pytest.raises(ValueError, match="coordinate spaces"):
        validate_source_image_manifest(
            changed, root=tmp_path, source_receipt_file_sha256=_sha("receipt")
        )


def test_validated_native_manifest_can_supply_exact_source_bytes() -> None:
    source = _source()
    sample = NativeYtSample(
        sample_token=source.sample_token,
        identity_token=_sha("identity"),
        registered_dog_id=compute_registered_dog_id("yt-bb-dog:v1:video-track:7"),
        source_sample_id="yt-bb-dog:v1:original:video-track:7:frame:3",
        sequence_token=source.sequence_token,
        track_token=source.track_token,
        frame_index=source.frame_index,
        source_role="YT_FIT",
        member_path="YT-BB-Dog/train/7/7_3.png",
        member_crc32=1,
        member_uncompressed_bytes=len(source.source_bytes),
        container_member_path="YT-BB-dog/YT-BB-Dog.zip",
        container_member_crc32=2,
        container_member_uncompressed_bytes=123,
        expected_source_sha256=source.source_sha256,
        roi_metadata_available=True,
    )
    normalized = (
        (34.0 / 96, 18.0 / 80),
        (62.0 / 96, 18.0 / 80),
        *((x / 96, y / 80) for x, y in source.positive_keypoints_xy),
    )
    prediction = [[x, y, 0.9] for x, y in normalized]
    policy = {
        "minimum_detector_confidence": 0.5,
        "minimum_frontality": 0.5,
        "minimum_native_short_side": 8,
        "maximum_mask_uncertainty": 1.0,
    }
    record, _ = process_native_sample(
        sample, source.source_bytes, prediction, policy=policy
    )
    bundle = build_manifest_bundle(
        records=[record],
        input_sha256s={"source_receipt": _sha("receipt")},
        policy=policy,
        tool_provenance={"schema_version": "fixture"},
    )

    adapted = sources_from_native_manifest(
        bundle, source_bytes_by_token={source.sample_token: source.source_bytes}
    )
    assert adapted[0].source_bytes == source.source_bytes
    assert adapted[0].source_sha256 == source.source_sha256
    assert adapted[0].nose_box_xyxy == tuple(record["nose_box_xyxy"])


def test_rich_manifest_is_compatible_with_native_extraction_teacher_hook(
    tmp_path: Path,
) -> None:
    source = _source()
    manifest, artifacts = produce_teacher_manifest(
        (source,),
        _StrictPredictor([(np.stack((_good_mask(),)), np.asarray((0.91,)))]),
        source_binding=_source_binding(),
        producer=_producer(),
    )
    for relative, payload in artifacts.items():
        target = tmp_path / relative
        target.parent.mkdir()
        target.write_bytes(payload)
    manifest_path = tmp_path / "teacher.json"
    manifest_path.write_bytes(json_document_bytes(manifest))

    records, file_sha256 = _teacher_records(manifest_path.resolve())
    assert file_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert set(records) == {source.sample_token}
    teacher = _load_teacher(
        records[source.sample_token], source.source_sha256, (96, 80)
    )
    selected_index = manifest["records"][0]["selection"]["selected_candidate_index"]
    selected = manifest["records"][0]["selection"]["candidates"][selected_index]
    assert _teacher_uncertainty(records[source.sample_token]) == pytest.approx(
        1.0 - selected["combined_score"]
    )
    assert teacher.mode == "L"
    assert np.array_equal(np.asarray(teacher) > 0, _good_mask() > 0)


def test_artifact_validation_binds_checkout_config_checkpoint_and_apache_license(
    tmp_path: Path,
) -> None:
    checkout = (tmp_path / "sam2").resolve()
    checkout.mkdir()
    (checkout / "sam2.py").write_text("MODEL = 'fixture'\n", encoding="utf-8")
    config = checkout / "fixture.yaml"
    config.write_text("model: fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(checkout), "-c", "user.name=Fixture",
            "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkpoint = (tmp_path / "sam2.1.pt").resolve()
    checkpoint.write_bytes(b"synthetic non-model checkpoint bytes")
    license_path = (tmp_path / "LICENSE.snapshot").resolve()
    license_path.write_text(
        "Apache License\nVersion 2.0, January 2004\nfixture snapshot\n",
        encoding="utf-8",
    )
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

    provenance = validate_sam2_artifacts(
        checkout=checkout,
        expected_checkout_commit=commit,
        config_path=config,
        expected_config_sha256=digest(config),
        checkpoint_path=checkpoint,
        expected_checkpoint_sha256=digest(checkpoint),
        license_snapshot_path=license_path,
        expected_license_snapshot_sha256=digest(license_path),
    )
    assert provenance["model_name"] == "sam2.1"
    assert provenance["sam2_checkpoint_sha256"] == digest(checkpoint)
    assert provenance["license_id"] == "Apache-2.0"

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        validate_sam2_artifacts(
            checkout=checkout,
            expected_checkout_commit=commit,
            config_path=config,
            expected_config_sha256=digest(config),
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256="0" * 64,
            license_snapshot_path=license_path,
            expected_license_snapshot_sha256=digest(license_path),
        )
    non_apache = (tmp_path / "OTHER.snapshot").resolve()
    non_apache.write_text("not this license", encoding="utf-8")
    with pytest.raises(ValueError, match="not an Apache-2.0"):
        validate_sam2_artifacts(
            checkout=checkout,
            expected_checkout_commit=commit,
            config_path=config,
            expected_config_sha256=digest(config),
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256=digest(checkpoint),
            license_snapshot_path=non_apache,
            expected_license_snapshot_sha256=digest(non_apache),
        )


def test_runtime_config_name_is_relative_to_official_sam2_package(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    config = checkout / "sam2" / "configs" / "sam2.1" / "small.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model: fixture\n", encoding="utf-8")

    assert _sam2_runtime_config_name(checkout, config) == (
        "configs/sam2.1/small.yaml"
    )
    outside = checkout / "small.yaml"
    outside.write_text("model: fixture\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the sam2 package"):
        _sam2_runtime_config_name(checkout, outside)


def test_tool_help_does_not_import_sam2() -> None:
    tool = Path(__file__).parents[1] / "tools" / "produce_yt_native_nose_teacher_masks.py"
    command = (
        "import runpy,sys; runpy.run_path(sys.argv[1], run_name='not_main'); "
        "assert not any(name == 'sam2' or name.startswith('sam2.') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command, str(tool)], check=True)


def test_tool_refuses_overwrite_and_output_inside_worktree(tmp_path: Path) -> None:
    existing = (tmp_path / "existing").resolve()
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_tool(Namespace(output_dir=existing))

    repository_root = Path(__file__).parents[1].resolve()
    prohibited = repository_root / ".sam2-teacher-test-output"
    assert not prohibited.exists()
    with pytest.raises(ValueError, match="outside the Git worktree"):
        run_tool(Namespace(output_dir=prohibited))
    assert not prohibited.exists()
