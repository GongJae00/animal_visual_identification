from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from shared.contracts.artifact_manifest import (
    ArtifactContractError,
    ArtifactLicense,
    ImagePreprocessing,
    NoseEmbeddingManifest,
    UsageLane,
)

from tests.repo_root import REPO_ROOT
from archive.nose.experiments.nose_fusion_scaling import (
    INTERPRETATION,
    METHODS,
    evaluate_fusion_scaling,
    validate_report_bundle,
)
from shared.foundation.provenance import content_sha256
from enrollment.registry.identity_registry import compute_registered_dog_id
from parsing.export.regions.native_yt import (
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
            helper.make_node("Gemm", ["flat", "weights"], ["output"], transB=1),
        ],
        "generated-real-fusion-scaling-embedding",
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
    root.mkdir(parents=True)
    policy = {
        "minimum_detector_confidence": 0.5,
        "minimum_frontality": 0.0,
        "minimum_native_short_side": 8,
        "maximum_mask_uncertainty": 1.0,
    }
    colors = (
        np.asarray((190, 35, 30), dtype=np.int16),
        np.asarray((30, 190, 40), dtype=np.int16),
        np.asarray((35, 45, 190), dtype=np.int16),
        np.asarray((145, 105, 35), dtype=np.int16),
    )
    frame_counts = (10, 11, 12, 9)
    shuffled_frames = [90, 10, 70, 30, 110, 50, 20, 100, 40, 80, 60, 120]
    records: list[dict[str, object]] = []
    expected_frames: dict[str, list[int]] = {}
    for identity_index, (color, frame_count) in enumerate(
        zip(colors, frame_counts, strict=True)
    ):
        identity_source = f"yt-bb-dog:v1:video-track:{identity_index}"
        registered = compute_registered_dog_id(identity_source)
        identity_token = _sha(f"identity-{identity_index}")
        track_token = _sha(f"track-{identity_index}")
        sequence_token = _sha(f"sequence-{identity_index}")
        frame_indices = shuffled_frames[:frame_count]
        expected_frames[registered] = sorted(frame_indices)
        for frame_index in frame_indices:
            y, x = np.indices((48, 48))
            texture = ((x // 4 + y // 4 + identity_index) % 2)[..., None] * 14
            noise = np.random.default_rng(identity_index * 1000 + frame_index).integers(
                -5, 6, size=(48, 48, 3)
            )
            source = np.clip(color + texture + noise, 4, 235).astype(np.uint8)
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
    runtime_manifest = NoseEmbeddingManifest(
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
    runtime_manifest_path = tmp_path / "runtime-manifest.json"
    runtime_manifest_path.write_text(
        json.dumps(runtime_manifest, sort_keys=True), encoding="utf-8"
    )
    return {
        "root": root,
        "policy": policy,
        "records": records,
        "bundle": bundle,
        "bundle_path": bundle_path,
        "bundle_sha256": content_sha256(bundle),
        "onnx_path": onnx_path,
        "runtime_manifest": runtime_manifest,
        "runtime_manifest_path": runtime_manifest_path,
        "runtime_manifest_sha256": content_sha256(runtime_manifest),
        "expected_frames": expected_frames,
    }

def _evaluate(fixture: dict[str, object], output: Path) -> dict[str, object]:
    return evaluate_fusion_scaling(
        native_bundle_path=fixture["bundle_path"],
        native_bundle_sha256=fixture["bundle_sha256"],
        native_root=fixture["root"],
        nose_runtime_manifest_path=fixture["runtime_manifest_path"],
        nose_runtime_manifest_sha256=fixture["runtime_manifest_sha256"],
        nose_onnx_path=fixture["onnx_path"],
        output_path=output,
        bootstrap_resamples=200,
        bootstrap_seed=17,
    )

def _pairing_contract(report: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "registered_dog_id": row["registered_dog_id"],
            "identity_token": row["identity_token"],
            "track_token": row["track_token"],
            "sequence_token": row["sequence_token"],
            "gallery": {
                method: row["gallery"][method]["sample_tokens"] for method in METHODS
            },
            "query": {
                method: row["query"][method]["sample_tokens"] for method in METHODS
            },
        }
        for row in report["paired_per_identity"]
    ]

def test_real_onnx_fixed_population_exact_fusion_pairing_metrics_and_bootstrap(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "fusion-scaling.json"

    bundle = _evaluate(fixture, output)
    report = validate_report_bundle(bundle)

    assert json.loads(output.read_text(encoding="utf-8")) == bundle
    assert report["interpretation"] == INTERPRETATION
    assert report["population"] == {
        "localized_identity_count": 4,
        "eligible_identity_count": 3,
        "excluded_identity_counts": {"fewer_than_ten_localized_frames": 1},
        "gallery_vector_count_per_method": 3,
        "query_vector_count_per_method": 3,
    }
    assert report["protocol"]["fixed_common_population_across_scales"] is True
    assert report["protocol"]["scales"] == [1, 3, 5]
    assert report["protocol"]["bootstrap"]["resamples"] == 200
    assert report["protocol"]["bootstrap"]["seed"] == 17
    assert report["protocol"]["pairing_sha256"] == content_sha256(
        _pairing_contract(report)
    )

    expected_frames = fixture["expected_frames"]
    for identity in report["paired_per_identity"]:
        ordered = expected_frames[identity["registered_dog_id"]]
        for scale in (1, 3, 5):
            method = f"K{scale}"
            gallery = identity["gallery"][method]
            query = identity["query"][method]
            assert gallery["frame_indices"] == ordered[:scale]
            assert query["frame_indices"] == ordered[-scale:]
            assert not set(gallery["sample_tokens"]) & set(query["sample_tokens"])
            diagnostics = gallery["fusion_diagnostics"]
            if scale == 1:
                assert diagnostics == {
                    "aggregation": "SINGLE_L2_NORMALIZED_EMBEDDING",
                    "temporal_api_invoked": False,
                }
            else:
                assert diagnostics["aggregation"] == "UNWEIGHTED_L2_MEAN"
                assert diagnostics["temporal_api_invoked"] is True
                assert diagnostics["accepted_indices"] == list(range(scale))
                assert diagnostics["rejected_indices"] == []
                assert diagnostics["normalized_qualities"] == pytest.approx(
                    [1.0 / scale] * scale
                )

    for method in METHODS:
        metrics = report["metrics"][method]
        assert metrics["query_count"] == metrics["gallery_count"] == 3
        outcomes = [
            identity["method_outcomes"][method]
            for identity in report["paired_per_identity"]
        ]
        for metric in ("Rank-1", "Rank-5", "MRR", "mAP"):
            assert metrics[metric] == pytest.approx(
                np.mean([outcome[metric] for outcome in outcomes])
            )

    for contrast, intervals in report["paired_delta_bootstrap_cis"].items():
        for metric, interval in intervals.items():
            expected = np.mean(
                [
                    identity["paired_deltas"][contrast][metric]
                    for identity in report["paired_per_identity"]
                ]
            )
            assert interval["estimate"] == pytest.approx(expected)
            assert interval["cluster_count"] == 3
            assert interval["cluster_unit"] == "query_identity"
            assert interval["resamples"] == 200
            assert interval["seed"] == 17

    assert report["input_sha256s"]["native_manifest_bundle_content"] == fixture[
        "bundle_sha256"
    ]
    assert report["input_sha256s"]["nose_runtime_manifest_content"] == fixture[
        "runtime_manifest_sha256"
    ]
    assert report["input_sha256s"]["nose_onnx"] == hashlib.sha256(
        fixture["onnx_path"].read_bytes()
    ).hexdigest()
    assert set(report["code_sha256s"]) >= {
        "archive/nose/experiments/nose_fusion_scaling.py",
        "identification/export/nose/signal/temporal.py",
        "archive/nose/commands/evaluate_yt_nose_fusion_scaling.py",
    }

    second = _evaluate(fixture, tmp_path / "fusion-scaling-second.json")
    assert second == bundle
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _evaluate(fixture, output)
    repository_output = REPO_ROOT / "forbidden-report.json"
    with pytest.raises(ValueError, match="outside the Git repository"):
        _evaluate(fixture, repository_output)

def test_rejects_input_report_tampering_and_noncanonical_track_mapping(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)

    fixture["bundle_path"].write_text(
        json.dumps({**fixture["bundle"], "schema_version": "tampered"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="external pin"):
        _evaluate(fixture, tmp_path / "native-tamper.json")
    fixture["bundle_path"].write_text(
        json.dumps(fixture["bundle"], sort_keys=True), encoding="utf-8"
    )

    fixture["runtime_manifest_path"].write_text(
        json.dumps({**fixture["runtime_manifest"], "artifact_id": "substituted"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime manifest content SHA-256"):
        _evaluate(fixture, tmp_path / "runtime-tamper.json")
    fixture["runtime_manifest_path"].write_text(
        json.dumps(fixture["runtime_manifest"], sort_keys=True), encoding="utf-8"
    )

    _write_embedding_onnx(fixture["onnx_path"], diagonal=2.0)
    with pytest.raises(ArtifactContractError, match="artifact SHA256"):
        _evaluate(fixture, tmp_path / "onnx-tamper.json")
    _write_embedding_onnx(fixture["onnx_path"])

    records = copy.deepcopy(fixture["records"])
    identities = sorted({row["registered_dog_id"] for row in records})
    shared_track = next(
        row["track_token"] for row in records if row["registered_dog_id"] == identities[0]
    )
    for row in records:
        if row["registered_dog_id"] == identities[1]:
            row["track_token"] = shared_track
    noncanonical = build_manifest_bundle(
        records=sorted(records, key=lambda row: row["sample_token"]),
        input_sha256s={"synthetic_source": _sha("synthetic-source")},
        policy=fixture["policy"],
        tool_provenance={"schema_version": "generated-test-fixture.v1"},
    )
    fixture["bundle_path"].write_text(
        json.dumps(noncanonical, sort_keys=True), encoding="utf-8"
    )
    fixture["bundle_sha256"] = content_sha256(noncanonical)
    with pytest.raises(ValueError, match="track_token maps to multiple"):
        _evaluate(fixture, tmp_path / "noncanonical-track.json")

    fixture = _make_fixture(tmp_path / "report-tamper-fixture")
    report_bundle = _evaluate(fixture, tmp_path / "valid-report.json")
    report_bundle["report"]["status"] = "TAMPERED"
    with pytest.raises(ValueError, match="report digest differs"):
        validate_report_bundle(report_bundle)
