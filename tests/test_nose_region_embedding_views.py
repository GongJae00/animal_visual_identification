from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from cvi.evidence.artifact_manifest import (
    ArtifactContractError,
    ArtifactLicense,
    ExactOnnxRuntime,
    ImagePreprocessing,
    NoseMaskManifest,
    UsageLane,
)
from cvi.identity_registry import compute_registered_dog_id
from cvi.nose_region.embedding_views import (
    MANIFEST_FILENAME,
    load_embedding_views_manifest,
    prepare_embedding_views,
    reconstruct_student_masked_rgb,
    student_masked_rgb,
)
from cvi.nose_region.native_yt import (
    NativeYtSample,
    build_manifest_bundle,
    process_native_sample,
)
from cvi.provenance import content_sha256


pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _png_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(values, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


def _write_mask_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weights = numpy_helper.from_array(
        np.asarray([[[[1.0]], [[-0.5]], [[0.25]]]], dtype=np.float32), "weights"
    )
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["images", "weights"], ["logits"]),
            helper.make_node("Sigmoid", ["logits"], ["mask"]),
        ],
        "tiny-embedding-view-mask",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, 224, 224]
            )
        ],
        [
            helper.make_tensor_value_info(
                "mask", TensorProto.FLOAT, [1, 1, 224, 224]
            )
        ],
        [weights],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path
    )


def _prediction() -> list[list[float]]:
    return [
        [0.26, 0.24, 0.96],
        [0.74, 0.24, 0.96],
        [0.50, 0.34, 0.96],
        [0.50, 0.78, 0.96],
        [0.41, 0.63, 0.96],
        [0.59, 0.63, 0.96],
        [0.32, 0.60, 0.96],
        [0.68, 0.60, 0.96],
    ]


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    native_root = tmp_path / "native"
    native_root.mkdir(parents=True)
    y, x = np.indices((48, 52))
    source = np.stack(
        ((x * 5 + 7) % 256, (y * 7 + 13) % 256, (x * 3 + y * 2) % 256),
        axis=-1,
    ).astype(np.uint8)
    source_bytes = _png_bytes(source)
    sample = NativeYtSample(
        sample_token=_sha("embedding-view-sample"),
        identity_token=_sha("embedding-view-identity"),
        registered_dog_id=compute_registered_dog_id("fixture:embedding-view"),
        source_sample_id="fixture:embedding-view:frame:4",
        sequence_token=_sha("embedding-view-sequence"),
        track_token=_sha("embedding-view-track"),
        frame_index=4,
        source_role="YT_FIT",
        member_path="track/4.png",
        member_crc32=0,
        member_uncompressed_bytes=len(source_bytes),
        container_member_path="YT-BB-Dog.zip",
        container_member_crc32=0,
        container_member_uncompressed_bytes=1,
        expected_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )
    policy = {
        "minimum_detector_confidence": 0.5,
        "minimum_frontality": 0.0,
        "minimum_native_short_side": 8,
        "maximum_mask_uncertainty": 1.0,
    }
    record, artifacts = process_native_sample(
        sample, source_bytes, _prediction(), policy=policy
    )
    for relative, payload in artifacts.items():
        target = native_root / relative
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(payload)
    bundle = build_manifest_bundle(
        records=[record],
        input_sha256s={"fixture": _sha("native-input")},
        policy=policy,
        tool_provenance={"schema_version": "embedding-view-fixture.v1"},
    )
    native_bundle = native_root / "bundle.json"
    native_bundle.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")

    student_root = tmp_path / "student"
    student_root.mkdir()
    mask_onnx = student_root / "nose_mask.onnx"
    _write_mask_onnx(mask_onnx)
    mask_manifest = NoseMaskManifest(
        artifact_id="tiny-research-mask",
        artifact_sha256=hashlib.sha256(mask_onnx.read_bytes()).hexdigest(),
        input_name="images",
        input_shape=(1, 3, 224, 224),
        output_name="mask",
        output_shape=(1, 1, 224, 224),
        license=ArtifactLicense("LicenseRef-Research-Fixture", UsageLane.RESEARCH_ONLY),
        preprocessing=ImagePreprocessing(
            color_mode="RGB",
            layout="NCHW",
            dtype="float32",
            resize="bilinear",
            scale=1.0 / 255.0,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            clahe=None,
        ),
        threshold=0.55,
    ).to_dict()
    mask_manifest_path = student_root / "nose_mask.runtime.json"
    mask_manifest_path.write_text(
        json.dumps(mask_manifest, sort_keys=True), encoding="utf-8"
    )
    lineage = {
        "artifacts": {
            "runtime_manifest": {
                "path": mask_manifest_path.name,
                "sha256": hashlib.sha256(mask_manifest_path.read_bytes()).hexdigest(),
                "bytes": mask_manifest_path.stat().st_size,
            },
            "onnx": {
                "path": mask_onnx.name,
                "sha256": hashlib.sha256(mask_onnx.read_bytes()).hexdigest(),
                "bytes": mask_onnx.stat().st_size,
            },
        },
        "lineage_sha256": _sha("validated-segmentation-lineage"),
    }
    lineage_path = student_root / "artifact_lineage.json"
    lineage_path.write_text(json.dumps(lineage, sort_keys=True), encoding="utf-8")

    import cvi.nose_region.segmentation_training as segmentation_training

    validated: list[tuple[object, Path]] = []

    def validate_lineage(payload: object, root: Path) -> None:
        validated.append((payload, root))

    monkeypatch.setattr(
        segmentation_training, "validate_lineage_manifest", validate_lineage
    )
    return {
        "native_root": native_root,
        "native_bundle": native_bundle,
        "native_bundle_sha256": content_sha256(bundle),
        "record": record,
        "student_root": student_root,
        "lineage": lineage,
        "lineage_path": lineage_path,
        "lineage_sha256": content_sha256(lineage),
        "mask_onnx": mask_onnx,
        "mask_manifest": mask_manifest,
        "mask_manifest_path": mask_manifest_path,
        "mask_manifest_sha256": content_sha256(mask_manifest),
        "validated": validated,
    }


def _prepare(fixture: dict[str, object], output: Path) -> dict[str, object]:
    return prepare_embedding_views(
        native_bundle_path=fixture["native_bundle"],
        native_bundle_sha256=fixture["native_bundle_sha256"],
        native_root=fixture["native_root"],
        student_lineage_path=fixture["lineage_path"],
        student_lineage_sha256=fixture["lineage_sha256"],
        student_root=fixture["student_root"],
        mask_manifest_path=fixture["mask_manifest_path"],
        mask_manifest_sha256=fixture["mask_manifest_sha256"],
        mask_onnx_path=fixture["mask_onnx"],
        output_dir=output,
    )


def test_student_masked_rgb_matches_architecture_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    rng = np.random.default_rng(17)
    crop = rng.integers(0, 256, size=(31, 27, 3), dtype=np.uint8)
    support = (np.indices((224, 224)).sum(axis=0) % 3) == 0
    resized = np.asarray(
        Image.fromarray(crop, mode="RGB").resize(
            (224, 224), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )
    median = np.median(resized, axis=(0, 1)).astype(np.float32)
    expected = np.where(
        support[..., None],
        resized,
        np.rint(0.25 * resized.astype(np.float32) + 0.75 * median),
    ).astype(np.uint8)
    assert np.array_equal(student_masked_rgb(crop, support), expected)

    import cvi.nose_id.architecture_evaluation as architecture

    called = 0
    shared_helper = student_masked_rgb

    def recording_helper(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal called
        called += 1
        return shared_helper(*args, **kwargs)

    monkeypatch.setattr(architecture, "student_masked_rgb", recording_helper)
    manifest = NoseMaskManifest.from_dict(fixture["mask_manifest"])
    runtime = ExactOnnxRuntime(fixture["mask_onnx"], manifest)
    masked, architecture_support, _ = architecture._student_mask(
        runtime, manifest, resized
    )
    architecture_median = np.median(resized, axis=(0, 1)).astype(np.float32)
    architecture_expected = np.where(
        architecture_support[..., None],
        resized,
        np.rint(0.25 * resized.astype(np.float32) + 0.75 * architecture_median),
    ).astype(np.uint8)
    assert called == 1
    assert np.array_equal(masked, architecture_expected)
    assert np.array_equal(masked, shared_helper(resized, architecture_support))


def test_prepare_actual_onnx_hashes_reconstruction_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "embedding-views"
    manifest = _prepare(fixture, output)

    assert fixture["validated"] == [(fixture["lineage"], fixture["student_root"])]
    assert manifest["record_count"] == 1
    assert manifest["student_binding"]["usage_lane"] == "RESEARCH_ONLY"
    assert manifest["transform"]["outside_support_original_weight"] == 0.25
    assert manifest["student_binding"]["mask_onnx"]["sha256"] == fixture[
        "mask_manifest"
    ]["artifact_sha256"]
    manifest_path = output / MANIFEST_FILENAME
    loaded = load_embedding_views_manifest(
        manifest_path,
        expected_payload_sha256=content_sha256(manifest),
        root=output,
    )
    reconstructed = reconstruct_student_masked_rgb(
        native_root=fixture["native_root"],
        view_root=output,
        native_record=fixture["record"],
        view_record=loaded["records"][0],
    )
    with Image.open(fixture["native_root"] / fixture["record"]["crop_path"]) as opened:
        crop = np.asarray(opened, dtype=np.uint8)
    with Image.open(output / loaded["records"][0]["support_path"]) as opened:
        support = np.asarray(opened, dtype=np.uint8)
    assert np.array_equal(reconstructed, student_masked_rgb(crop, support))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _prepare(fixture, output)


def test_cache_and_source_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "embedding-views"
    manifest = _prepare(fixture, output)
    support = output / manifest["records"][0]["support_path"]
    support.write_bytes(support.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        load_embedding_views_manifest(
            output / MANIFEST_FILENAME,
            expected_payload_sha256=content_sha256(manifest),
            root=output,
        )
    with pytest.raises(ValueError, match="external pin"):
        load_embedding_views_manifest(
            output / MANIFEST_FILENAME,
            expected_payload_sha256="0" * 64,
            root=output,
        )

    second = _fixture(tmp_path / "second", monkeypatch)
    crop = second["native_root"] / second["record"]["crop_path"]
    crop.write_bytes(crop.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="artifact SHA-256"):
        _prepare(second, tmp_path / "must-not-publish")
    assert not (tmp_path / "must-not-publish").exists()

    third = _fixture(tmp_path / "third", monkeypatch)
    onnx = third["mask_onnx"]
    onnx.write_bytes(onnx.read_bytes() + b"tamper")
    with pytest.raises(ArtifactContractError, match="ONNX hash"):
        _prepare(third, tmp_path / "onnx-must-not-publish")
    assert not (tmp_path / "onnx-must-not-publish").exists()


def test_prepare_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/prepare_nose_embedding_views.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--student-lineage-sha256" in completed.stdout
    assert "--mask-manifest-sha256" in completed.stdout
    assert "--output-dir" in completed.stdout
