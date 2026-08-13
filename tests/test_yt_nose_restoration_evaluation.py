from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from contracts.artifact_manifest import (
    ArtifactContractError,
    ArtifactLicense,
    ImagePreprocessing,
    NoseEmbeddingManifest,
    UsageLane,
)
from experiments.nose_restoration import (
    INTERPRETATION,
    METHODS,
    evaluate_raw_vs_restored,
    validate_report_bundle,
)
from foundation.provenance import content_sha256
from identity_governance.identity_registry import compute_registered_dog_id
from localization.nose_region.native_yt import (
    NativeYtSample,
    build_manifest_bundle,
    process_native_sample,
)

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_embedding_onnx(path: Path, *, diagonal: float = 1.0) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weights = numpy_helper.from_array(
        np.eye(3, dtype=np.float32) * diagonal, "weights"
    )
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node(
                "Gemm", ["flat", "weights"], ["output"], transB=1
            ),
        ],
        "generated-real-nose-embedding",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        [weights],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path
    )


def _png_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(values, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


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


def _make_fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "native"
    root.mkdir()
    policy = {
        "minimum_detector_confidence": 0.5,
        "minimum_frontality": 0.0,
        "minimum_native_short_side": 8,
        "maximum_mask_uncertainty": 1.0,
    }
    colors = (
        np.asarray((185, 42, 35), dtype=np.int16),
        np.asarray((38, 184, 48), dtype=np.int16),
        np.asarray((35, 48, 185), dtype=np.int16),
    )
    records: list[dict[str, object]] = []
    expected_frames: dict[str, tuple[list[int], list[int]]] = {}
    for identity_index, color in enumerate(colors):
        identity_source = f"yt-bb-dog:v1:video-track:{identity_index}"
        registered = compute_registered_dog_id(identity_source)
        identity_token = _sha(f"identity-{identity_index}")
        track_token = _sha(f"track-{identity_index}")
        sequence_token = _sha(f"sequence-{identity_index}")
        frame_indices = [70, 10, 50, 30, 80, 20, 60, 40]
        expected_frames[registered] = ([10, 20, 30], [60, 70, 80])
        for frame_index in frame_indices:
            y, x = np.indices((64, 64))
            texture = ((x // 4 + y // 4 + identity_index) % 2)[..., None] * 18
            base = np.broadcast_to(color, (64, 64, 3)) + texture
            shifted = np.roll(
                base,
                shift=((frame_index // 10) % 3 - 1, (frame_index // 20) % 3 - 1),
                axis=(0, 1),
            )
            noise = np.random.default_rng(identity_index * 100 + frame_index).integers(
                -7, 8, size=shifted.shape
            )
            source = np.clip(shifted + noise, 4, 235).astype(np.uint8)
            payload = _png_bytes(source)
            sample_token = _sha(f"sample-{identity_index}-{frame_index}")
            sample = NativeYtSample(
                sample_token=sample_token,
                identity_token=identity_token,
                registered_dog_id=registered,
                source_sample_id=f"{identity_source}:frame:{frame_index}",
                sequence_token=sequence_token,
                track_token=track_token,
                frame_index=frame_index,
                source_role="YT_FIT",
                member_path=f"track-{identity_index}/{frame_index}.png",
                member_crc32=0,
                member_uncompressed_bytes=len(payload),
                container_member_path="YT-BB-Dog.zip",
                container_member_crc32=0,
                container_member_uncompressed_bytes=1,
                expected_source_sha256=hashlib.sha256(payload).hexdigest(),
            )
            record, artifacts = process_native_sample(
                sample, payload, _prediction(), policy=policy
            )
            records.append(record)
            for relative, artifact in artifacts.items():
                target = root / relative
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(artifact)

    bundle = build_manifest_bundle(
        records=sorted(records, key=lambda row: row["sample_token"]),
        input_sha256s={"synthetic_source": _sha("synthetic-source")},
        policy=policy,
        tool_provenance={"schema_version": "generated-test-fixture.v1"},
    )
    bundle_path = root / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")

    onnx_path = tmp_path / "embedding.onnx"
    _write_embedding_onnx(onnx_path)
    embedding_manifest = NoseEmbeddingManifest(
        artifact_id="generated-test-nose-embedding",
        artifact_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        input_name="input",
        input_shape=(1, 3, 16, 16),
        output_name="output",
        output_shape=(1, 3),
        license=ArtifactLicense("LicenseRef-Test", UsageLane.TEST_FIXTURE),
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
    ).to_dict()
    embedding_manifest_path = tmp_path / "embedding-manifest.json"
    embedding_manifest_path.write_text(
        json.dumps(embedding_manifest, sort_keys=True), encoding="utf-8"
    )
    return {
        "root": root,
        "bundle": bundle,
        "bundle_path": bundle_path,
        "bundle_sha256": content_sha256(bundle),
        "onnx_path": onnx_path,
        "embedding_manifest": embedding_manifest,
        "embedding_manifest_path": embedding_manifest_path,
        "embedding_manifest_sha256": content_sha256(embedding_manifest),
        "expected_frames": expected_frames,
    }


def _evaluate(
    fixture: dict[str, object],
    output: Path,
    *,
    mask_mode: str = "FULL_CROP",
) -> dict[str, object]:
    return evaluate_raw_vs_restored(
        native_bundle_path=fixture["bundle_path"],
        native_bundle_sha256=fixture["bundle_sha256"],
        native_root=fixture["root"],
        embedding_manifest_path=fixture["embedding_manifest_path"],
        embedding_manifest_sha256=fixture["embedding_manifest_sha256"],
        embedding_onnx_path=fixture["onnx_path"],
        output_path=output,
        evaluation_size=16,
        mask_mode=mask_mode,
    )


def test_real_onnx_paired_evaluation_has_exact_temporal_pairing_and_metrics(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "report.json"

    bundle = _evaluate(fixture, output)
    report = validate_report_bundle(bundle)

    assert output.is_file()
    assert report["interpretation"] == INTERPRETATION
    assert report["population"]["eligible_identity_count"] == 3
    assert report["protocol"]["evaluation_size"] == [16, 16]
    assert report["protocol"]["frames_per_vector"] == 3
    assert report["protocol"]["mask_mode"] == "FULL_CROP"
    expected_frames = fixture["expected_frames"]
    for identity in report["paired_per_identity"]:
        gallery_expected, query_expected = expected_frames[identity["registered_dog_id"]]
        assert identity["gallery"]["frame_indices"] == gallery_expected
        assert identity["query"]["frame_indices"] == query_expected
        assert not set(identity["gallery"]["sample_tokens"]) & set(
            identity["query"]["sample_tokens"]
        )
        assert identity["gallery"]["single_best_sample_token"] in identity["gallery"][
            "sample_tokens"
        ]
        assert identity["query"]["single_best_sample_token"] in identity["query"][
            "sample_tokens"
        ]
        assert len(identity["gallery"]["restoration_diagnostics"]["frames"]) == 3
        assert len(identity["query"]["restoration_diagnostics"]["frames"]) == 3
        for role in ("gallery", "query"):
            diagnostics = identity[role]["restoration_diagnostics"]["frames"]
            assert [frame["index"] for frame in diagnostics] == [0, 1, 2]
            assert len({frame["input_sha256"] for frame in diagnostics}) == 3

    for method in METHODS:
        metrics = report["metrics"][method]
        assert metrics["Rank-1"] == pytest.approx(1.0)
        assert metrics["Rank-5"] == pytest.approx(1.0)
        assert metrics["mAP"] == pytest.approx(1.0)
        assert metrics["MRR"] == pytest.approx(1.0)
        assert metrics["genuine_scores"]["count"] == 3
        assert metrics["impostor_scores"]["count"] == 6
        outcomes = [
            identity["method_outcomes"][method]
            for identity in report["paired_per_identity"]
        ]
        assert metrics["MRR"] == pytest.approx(
            np.mean([outcome["reciprocal_rank"] for outcome in outcomes])
        )
        assert metrics["mAP"] == pytest.approx(
            np.mean([outcome["average_precision"] for outcome in outcomes])
        )
    assert report["restoration"]["restoration_count"] == 6
    assert set(report["paired_delta_summaries"]) == {
        "B_minus_A",
        "C_minus_A",
        "C_minus_B",
        "D_minus_A",
        "D_minus_B",
        "E_minus_B",
    }

    masked = validate_report_bundle(
        _evaluate(
            fixture,
            tmp_path / "masked-report.json",
            mask_mode="MANIFEST_BINARY",
        )
    )
    assert masked["protocol"]["mask_mode"] == "MANIFEST_BINARY"
    assert masked["protocol"]["segmentation_mask_use"] == (
        "manifest_binary_mask_defines_observed_source_support"
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _evaluate(fixture, output)
    repository_output = Path(__file__).resolve().parents[1] / "forbidden-report.json"
    with pytest.raises(ValueError, match="outside the Git repository"):
        _evaluate(fixture, repository_output)


def test_rejects_manifest_artifact_and_report_tampering(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    bundle_path = fixture["bundle_path"]
    original_bundle_bytes = bundle_path.read_bytes()
    tampered_bundle = json.loads(original_bundle_bytes)
    tampered_bundle["manifest"]["records"][0]["frame_index"] += 1
    bundle_path.write_text(json.dumps(tampered_bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="external pin"):
        _evaluate(fixture, tmp_path / "manifest-tamper-report.json")

    bundle_path.write_bytes(original_bundle_bytes)
    embedding_manifest_path = fixture["embedding_manifest_path"]
    original_embedding_manifest_bytes = embedding_manifest_path.read_bytes()
    tampered_embedding_manifest = json.loads(original_embedding_manifest_bytes)
    tampered_embedding_manifest["artifact_id"] = "substituted"
    embedding_manifest_path.write_text(
        json.dumps(tampered_embedding_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="embedding manifest content SHA-256"):
        _evaluate(fixture, tmp_path / "embedding-manifest-tamper-report.json")

    embedding_manifest_path.write_bytes(original_embedding_manifest_bytes)
    first_record = fixture["bundle"]["manifest"]["records"][0]
    crop = fixture["root"] / first_record["crop_path"]
    original_crop = crop.read_bytes()
    crop.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact SHA-256 differs"):
        _evaluate(fixture, tmp_path / "crop-tamper-report.json")

    crop.write_bytes(original_crop)
    _write_embedding_onnx(fixture["onnx_path"], diagonal=2.0)
    with pytest.raises(ArtifactContractError, match="artifact SHA256"):
        _evaluate(fixture, tmp_path / "onnx-tamper-report.json")

    _write_embedding_onnx(fixture["onnx_path"])
    report_bundle = _evaluate(fixture, tmp_path / "valid-report.json")
    report_bundle["report"]["status"] = "TAMPERED"
    with pytest.raises(ValueError, match="report digest differs"):
        validate_report_bundle(report_bundle)
