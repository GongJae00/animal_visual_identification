from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from artifact_contracts.artifact_manifest import (
    ArtifactContractError,
    ArtifactLicense,
    ImagePreprocessing,
    NoseEmbeddingManifest,
    NoseMaskManifest,
    UsageLane,
)
from identity_governance.identity_registry import compute_registered_dog_id
from experiments.nose_architecture import (
    INTERPRETATION,
    METHODS,
    _filter_consistency_eval_population,
    _validate_population_role_inputs,
    evaluate_calibrated_score_fusion,
    evaluate_nose_architectures,
    validate_report_bundle,
)
from localization.nose_region.native_yt import (
    NativeYtSample,
    build_manifest_bundle,
    process_native_sample,
)
from foundation.provenance import content_sha256


pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_embedding_onnx(path: Path, *, scale: float = 1.0) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weights = numpy_helper.from_array(
        np.eye(3, dtype=np.float32) * scale, "weights"
    )
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "weights"], ["output"], transB=1),
        ],
        "tiny-static-nose-embedding",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, 3, 224, 224]
            )
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        [weights],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path
    )


def _write_mask_onnx(path: Path, *, weight: float = 1.0 / 3.0) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weights = numpy_helper.from_array(
        np.full((1, 3, 1, 1), weight, dtype=np.float32), "weights"
    )
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["images", "weights"], ["logits"]),
            helper.make_node("Sigmoid", ["logits"], ["mask"]),
        ],
        "tiny-static-nose-mask",
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


def _manifest_preprocessing() -> ImagePreprocessing:
    return ImagePreprocessing(
        color_mode="RGB",
        layout="NCHW",
        dtype="float32",
        resize="bilinear",
        scale=1.0 / 255.0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        clahe=None,
    )


def _make_fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "native"
    root.mkdir(parents=True)
    policy = {
        "minimum_detector_confidence": 0.5,
        "minimum_frontality": 0.0,
        "minimum_native_short_side": 8,
        "maximum_mask_uncertainty": 1.0,
    }
    frame_order = [90, 10, 70, 30, 110, 50, 20, 100, 40, 80, 60, 120]
    calibration_sources: list[int] = []
    evaluation_sources: list[int] = []
    candidate = 0
    while len(calibration_sources) < 10 or len(evaluation_sources) < 10:
        identity_source = f"yt-bb-dog:v1:video-track:{candidate}"
        registered = compute_registered_dog_id(identity_source)
        digest = hashlib.sha256(f"73:{registered}".encode("utf-8")).digest()
        target = (
            calibration_sources
            if int.from_bytes(digest, byteorder="big") < int(0.3 * (1 << 256))
            else evaluation_sources
        )
        if len(target) < 10:
            target.append(candidate)
        candidate += 1
    identity_sources = calibration_sources + evaluation_sources + [candidate]
    records: list[dict[str, object]] = []
    expected_frames: dict[str, list[int]] = {}
    for fixture_index, identity_index in enumerate(identity_sources):
        color = np.random.default_rng(identity_index).integers(
            28, 221, size=3, dtype=np.int16
        )
        frame_count = 9 if fixture_index == len(identity_sources) - 1 else 10
        identity_source = f"yt-bb-dog:v1:video-track:{identity_index}"
        registered = compute_registered_dog_id(identity_source)
        identity_token = _sha(f"identity-{identity_index}")
        track_token = _sha(f"track-{identity_index}")
        sequence_token = _sha(f"sequence-{identity_index}")
        frame_indices = frame_order[:frame_count]
        expected_frames[registered] = sorted(frame_indices)
        for frame_index in frame_indices:
            y, x = np.indices((64, 64))
            checker = ((x // 4 + y // 4 + identity_index) % 2)[..., None] * 12
            source = np.broadcast_to(color, (64, 64, 3)) + checker
            dark_border = (x < 13) | (x >= 51) | (y < 13) | (y >= 51)
            source = np.where(dark_border[..., None], 8, source)
            noise = np.random.default_rng(identity_index * 1000 + frame_index).integers(
                -4, 5, size=source.shape
            )
            source = np.clip(source + noise, 3, 235).astype(np.uint8)
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

    embedding_onnx = tmp_path / "embedding.onnx"
    _write_embedding_onnx(embedding_onnx)
    embedding_manifest = NoseEmbeddingManifest(
        artifact_id="tiny-test-nose-embedding",
        artifact_sha256=hashlib.sha256(embedding_onnx.read_bytes()).hexdigest(),
        input_name="input",
        input_shape=(1, 3, 224, 224),
        output_name="output",
        output_shape=(1, 3),
        license=ArtifactLicense("LicenseRef-Test", UsageLane.TEST_FIXTURE),
        preprocessing=_manifest_preprocessing(),
    ).to_dict()
    embedding_manifest_path = tmp_path / "embedding.runtime.json"
    embedding_manifest_path.write_text(
        json.dumps(embedding_manifest, sort_keys=True), encoding="utf-8"
    )

    mask_onnx = tmp_path / "mask.onnx"
    _write_mask_onnx(mask_onnx)
    mask_manifest = NoseMaskManifest(
        artifact_id="tiny-test-nose-mask",
        artifact_sha256=hashlib.sha256(mask_onnx.read_bytes()).hexdigest(),
        input_name="images",
        input_shape=(1, 3, 224, 224),
        output_name="mask",
        output_shape=(1, 1, 224, 224),
        license=ArtifactLicense("LicenseRef-Test", UsageLane.TEST_FIXTURE),
        preprocessing=_manifest_preprocessing(),
        threshold=0.55,
    ).to_dict()
    mask_manifest_path = tmp_path / "mask.runtime.json"
    mask_manifest_path.write_text(
        json.dumps(mask_manifest, sort_keys=True), encoding="utf-8"
    )
    return {
        "root": root,
        "bundle": bundle,
        "bundle_path": bundle_path,
        "bundle_sha256": content_sha256(bundle),
        "embedding_onnx": embedding_onnx,
        "embedding_manifest": embedding_manifest,
        "embedding_manifest_path": embedding_manifest_path,
        "embedding_manifest_sha256": content_sha256(embedding_manifest),
        "mask_onnx": mask_onnx,
        "mask_manifest": mask_manifest,
        "mask_manifest_path": mask_manifest_path,
        "mask_manifest_sha256": content_sha256(mask_manifest),
        "expected_frames": expected_frames,
    }


def _evaluate(fixture: dict[str, object], output: Path) -> dict[str, object]:
    return evaluate_nose_architectures(
        native_bundle_path=fixture["bundle_path"],
        native_bundle_sha256=fixture["bundle_sha256"],
        native_root=fixture["root"],
        embedding_manifest_path=fixture["embedding_manifest_path"],
        embedding_manifest_sha256=fixture["embedding_manifest_sha256"],
        embedding_onnx_path=fixture["embedding_onnx"],
        mask_manifest_path=fixture["mask_manifest_path"],
        mask_manifest_sha256=fixture["mask_manifest_sha256"],
        mask_onnx_path=fixture["mask_onnx"],
        output_path=output,
        bootstrap_resamples=100,
        bootstrap_seed=17,
    )


def test_real_static_onnx_all_architectures_pairing_metrics_and_bootstrap(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "architecture.json"

    bundle = _evaluate(fixture, output)
    report = validate_report_bundle(bundle)

    assert json.loads(output.read_text(encoding="utf-8")) == bundle
    assert report["interpretation"] == INTERPRETATION
    assert report["population"] == {
        "selected_population_role": "all",
        "selected_identity_count": 20,
        "selected_registered_dog_ids_sha256": content_sha256(
            {
                "registered_dog_ids": [
                    row["registered_dog_id"] for row in report["paired_per_identity"]
                ]
            }
        ),
        "localized_identity_count": 21,
        "eligible_identity_count": 20,
        "excluded_identity_counts": {"fewer_than_ten_localized_frames": 1},
        "gallery_vector_count_per_method": 20,
        "query_vector_count_per_method": 20,
    }
    assert set(report["metrics"]) == set(METHODS)
    assert report["protocol"]["population_role"] == "all"
    assert "embedding_lineage" not in report["input_bindings"]
    assert "embedding_lineage" not in report["input_sha256s"]
    assert report["protocol"]["fixed_common_population_across_methods"] is True
    assert report["protocol"]["frames_per_role"] == 5
    assert report["protocol"]["image_size"] == [224, 224]
    assert report["protocol"]["mask_threshold"] == 0.55
    assert report["protocol"]["mask_background_transform"][
        "outside_support_original_weight"
    ] == 0.25
    assert report["protocol"]["restoration_config"]["registration_mode"] == (
        "canonical_crop_identity"
    )
    assert report["protocol"]["restoration_config"][
        "illumination_normalization"
    ] is False

    expected_frames = fixture["expected_frames"]
    for identity in report["paired_per_identity"]:
        ordered = expected_frames[identity["registered_dog_id"]]
        assert identity["gallery"]["frame_indices"] == ordered[:5]
        assert identity["query"]["frame_indices"] == ordered[-5:]
        assert not set(identity["gallery"]["sample_tokens"]) & set(
            identity["query"]["sample_tokens"]
        )
        assert set(identity["method_outcomes"]) == set(METHODS)
        assert len(identity["paired_deltas_against_A"]) == len(METHODS) - 1
        for role in ("gallery", "query"):
            assert len(identity[role]["mask_diagnostics"]) == 5
            assert identity[role]["raw_K5_diagnostics"]["aggregation"] == (
                "UNWEIGHTED_L2_MEAN"
            )
            assert identity[role]["student_masked_K5_diagnostics"][
                "aggregation"
            ] == "UNWEIGHTED_L2_MEAN"
            restoration = identity[role]["restoration_diagnostics"]
            assert len(restoration["frames"]) == 5
            assert restoration["config"]["registration_mode"] == (
                "canonical_crop_identity"
            )

    for method in METHODS:
        outcomes = [
            identity["method_outcomes"][method]
            for identity in report["paired_per_identity"]
        ]
        for metric in ("Rank-1", "Rank-5", "MRR", "mAP"):
            assert report["metrics"][method][metric] == pytest.approx(
                np.mean([outcome[metric] for outcome in outcomes])
            )
    assert report["mask_diagnostics"]["frame_count"] == 200
    assert report["mask_diagnostics"]["support_fraction"]["minimum"] >= 0.0
    assert report["mask_diagnostics"]["support_fraction"]["maximum"] <= 1.0
    assert len(report["paired_delta_bootstrap_cis"]) == len(METHODS) - 1
    for contrast, intervals in report["paired_delta_bootstrap_cis"].items():
        assert set(intervals) == {"Rank-1", "Rank-5", "MRR", "mAP"}
        for metric, interval in intervals.items():
            expected = np.mean(
                [
                    identity["paired_deltas_against_A"][contrast][metric]
                    for identity in report["paired_per_identity"]
                ]
            )
            assert interval["estimate"] == pytest.approx(expected)
            assert interval["resamples"] == 100
            assert interval["seed"] == 17

    second = _evaluate(fixture, tmp_path / "architecture-second.json")
    assert second == bundle


def _complementary_score_matrices(
    identities: list[str], *, calibration_fraction: float = 0.3, calibration_seed: int = 73
) -> dict[str, np.ndarray]:
    size = len(identities)
    raw = np.zeros((size, size), dtype=np.float64)
    masked = np.zeros((size, size), dtype=np.float64)
    threshold = int(calibration_fraction * (1 << 256))
    calibration = []
    evaluation = []
    for index, identity in enumerate(identities):
        digest = hashlib.sha256(f"{calibration_seed}:{identity}".encode("utf-8")).digest()
        (calibration if int.from_bytes(digest, "big") < threshold else evaluation).append(
            index
        )

    def assign(matrix: np.ndarray, indices: list[int], local_index: int, good: bool) -> None:
        row = indices[local_index]
        competitor = indices[(local_index + 1) % len(indices)]
        values = list(np.linspace(-1.1, 0.2, len(indices) - 2)) + [1.0, 2.0]
        matrix[row, row] = 2.0 if good else 1.0
        matrix[row, competitor] = -1.1 if good else 2.0
        used_values = {2.0 if good else 1.0, -1.1 if good else 2.0}
        remaining_values = list(values)
        for used in used_values:
            remaining_values.remove(used)
        remaining_columns = [
            column for column in indices if column not in {row, competitor}
        ]
        for column, value in zip(remaining_columns, remaining_values, strict=True):
            matrix[row, column] = value

    for indices in (calibration, evaluation):
        for local_index in range(len(indices)):
            assign(raw, indices, local_index, good=local_index % 2 == 1)
            assign(masked, indices, local_index, good=local_index % 2 == 0)
    return {
        "A_raw_K5": raw,
        "B_student_masked_K5": masked,
        "D_restored_early_fusion": masked.copy(),
    }


def test_calibrated_score_fusion_is_deterministic_and_improves_held_out_eval() -> None:
    identities = [f"dog-{index:03d}" for index in range(80)]
    scores = _complementary_score_matrices(identities)

    first = evaluate_calibrated_score_fusion(
        identities,
        scores,
        bootstrap_resamples=100,
        bootstrap_seed=19,
    )
    second = evaluate_calibrated_score_fusion(
        identities,
        scores,
        bootstrap_resamples=100,
        bootstrap_seed=19,
    )

    assert second == first
    assert first["calibration"]["selected_weights"] == {
        "A_raw_K5": 0.75,
        "B_student_masked_K5": 0.25,
        "D_restored_early_fusion": 0.0,
    }
    assert first["calibration"]["objective"]["selected_values"]["Rank-1"] == 1.0
    evaluation = first["evaluation"]
    assert evaluation["fused_metrics"]["Rank-1"] == 1.0
    assert evaluation["fused_metrics"]["Rank-1"] > evaluation["baseline_A_metrics"][
        "Rank-1"
    ]
    assert len(evaluation["per_identity"]) == first["population"][
        "evaluation_identity_count"
    ]
    assert all(
        interval["seed"] == 19 and interval["resamples"] == 100
        for interval in evaluation["paired_delta_bootstrap_cis"].values()
    )

    different_split = evaluate_calibrated_score_fusion(
        identities,
        _complementary_score_matrices(identities, calibration_seed=74),
        calibration_seed=74,
        bootstrap_resamples=100,
        bootstrap_seed=19,
    )
    assert different_split["population"]["partition_assignment_sha256"] != first[
        "population"
    ]["partition_assignment_sha256"]


def test_calibrated_score_fusion_rejects_small_split_and_near_zero_rows() -> None:
    identities = [f"dog-{index:03d}" for index in range(80)]
    scores = _complementary_score_matrices(identities)
    scores["D_restored_early_fusion"] = np.ones((80, 80), dtype=np.float64)
    with pytest.raises(ValueError, match="near-zero"):
        evaluate_calibrated_score_fusion(identities, scores, bootstrap_resamples=10)

    small_identities = [f"dog-{index:03d}" for index in range(19)]
    small_scores = {
        branch: np.eye(19, dtype=np.float64)
        for branch in ("A_raw_K5", "B_student_masked_K5", "D_restored_early_fusion")
    }
    with pytest.raises(ValueError, match="at least 10 calibration and 10 evaluation"):
        evaluate_calibrated_score_fusion(
            small_identities, small_scores, bootstrap_resamples=10
        )


def test_strict_hash_tamper_and_output_refusals(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "existing.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _evaluate(fixture, output)

    repository_output = Path(__file__).resolve().parents[1] / "forbidden-report.json"
    with pytest.raises(ValueError, match="outside the Git repository"):
        _evaluate(fixture, repository_output)

    bundle_path = fixture["bundle_path"]
    original = bundle_path.read_bytes()
    bundle_path.write_text(
        json.dumps({**fixture["bundle"], "schema_version": "tampered"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="external pin"):
        _evaluate(fixture, tmp_path / "native-tamper.json")
    bundle_path.write_bytes(original)

    embedding_manifest_path = fixture["embedding_manifest_path"]
    original = embedding_manifest_path.read_bytes()
    embedding_manifest_path.write_text(
        json.dumps({**fixture["embedding_manifest"], "artifact_id": "tampered"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="embedding manifest content SHA-256"):
        _evaluate(fixture, tmp_path / "embedding-manifest-tamper.json")
    embedding_manifest_path.write_bytes(original)

    mask_manifest_path = fixture["mask_manifest_path"]
    original = mask_manifest_path.read_bytes()
    mask_manifest_path.write_text(
        json.dumps({**fixture["mask_manifest"], "threshold": 0.9}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mask manifest content SHA-256"):
        _evaluate(fixture, tmp_path / "mask-manifest-tamper.json")
    mask_manifest_path.write_bytes(original)

    first_record = fixture["bundle"]["manifest"]["records"][0]
    crop_path = fixture["root"] / first_record["crop_path"]
    original = crop_path.read_bytes()
    crop_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact SHA-256 differs"):
        _evaluate(fixture, tmp_path / "crop-tamper.json")
    crop_path.write_bytes(original)

    _write_embedding_onnx(fixture["embedding_onnx"], scale=2.0)
    with pytest.raises(ArtifactContractError, match="artifact SHA256"):
        _evaluate(fixture, tmp_path / "embedding-onnx-tamper.json")
    _write_embedding_onnx(fixture["embedding_onnx"])

    _write_mask_onnx(fixture["mask_onnx"], weight=0.5)
    with pytest.raises(ArtifactContractError, match="artifact SHA256"):
        _evaluate(fixture, tmp_path / "mask-onnx-tamper.json")
    _write_mask_onnx(fixture["mask_onnx"])

    valid = _evaluate(fixture, tmp_path / "valid.json")
    config_tamper = copy.deepcopy(valid)
    config_tamper["report"]["calibrated_score_fusion"]["config"]["identity_split"][
        "calibration_fraction"
    ] = 0.4
    config_tamper["report_sha256"] = content_sha256(config_tamper["report"])
    with pytest.raises(ValueError, match="config digest differs"):
        validate_report_bundle(config_tamper)

    population_tamper = copy.deepcopy(valid)
    population_tamper["report"]["calibrated_score_fusion"]["population"][
        "calibration_registered_dog_ids_sha256"
    ] = _sha("tampered-calibration-population")
    population_tamper["report_sha256"] = content_sha256(population_tamper["report"])
    with pytest.raises(ValueError, match="calibration population digest differs"):
        validate_report_bundle(population_tamper)

    consistency_report = copy.deepcopy(valid)
    selected_sha256 = consistency_report["report"]["population"][
        "selected_registered_dog_ids_sha256"
    ]
    consistency_report["report"]["protocol"]["population_role"] = "consistency_eval"
    consistency_report["report"]["population"]["selected_population_role"] = (
        "consistency_eval"
    )
    consistency_report["report"]["input_bindings"]["population_selection"][
        "role"
    ] = "consistency_eval"
    consistency_report["report"]["input_bindings"]["embedding_lineage"] = {
        "path": "/artifact/artifact_lineage.json",
        "parent_root": "/artifact",
        "raw_sha256": "0" * 64,
        "content_sha256": "1" * 64,
        "lineage_sha256": "2" * 64,
        "byte_size": 1,
        "eval_identity_count": 20,
        "eval_registered_dog_ids_sha256": selected_sha256,
    }
    consistency_report["report"]["input_sha256s"].update(
        {
            "embedding_lineage_raw": "0" * 64,
            "embedding_lineage_content": "1" * 64,
            "embedding_lineage": "2" * 64,
        }
    )
    consistency_report["report_sha256"] = content_sha256(consistency_report["report"])
    validate_report_bundle(consistency_report)

    consistency_report["report"]["input_bindings"]["embedding_lineage"][
        "eval_identity_count"
    ] = 19
    consistency_report["report_sha256"] = content_sha256(consistency_report["report"])
    with pytest.raises(ValueError, match="embedding lineage binding differs"):
        validate_report_bundle(consistency_report)

    valid["report"]["status"] = "TAMPERED"
    with pytest.raises(ValueError, match="report digest differs"):
        validate_report_bundle(valid)


def test_population_role_inputs_reject_missing_or_ambiguous_lineage() -> None:
    digest = "0" * 64
    lineage = Path("artifact_lineage.json")

    assert _validate_population_role_inputs("all", None, None) is None
    assert (
        _validate_population_role_inputs("consistency_eval", lineage, digest)
        == digest
    )
    with pytest.raises(ValueError, match="requires both"):
        _validate_population_role_inputs("consistency_eval", None, None)
    with pytest.raises(ValueError, match="requires both"):
        _validate_population_role_inputs("consistency_eval", lineage, None)
    with pytest.raises(ValueError, match="rejects embedding lineage"):
        _validate_population_role_inputs("all", lineage, digest)
    with pytest.raises(ValueError, match="population_role"):
        _validate_population_role_inputs("eval", None, None)


def test_consistency_eval_population_filter_is_canonical_exact_and_at_least_20() -> None:
    identities = sorted(
        compute_registered_dog_id(f"yt-bb-dog:v1:video-track:{index}")
        for index in range(20)
    )
    population = [{"registered_dog_id": identity} for identity in identities]

    assert _filter_consistency_eval_population(population, identities) == population

    with pytest.raises(ValueError, match="sorted unique list"):
        _filter_consistency_eval_population(population, identities[:-1])
    noncanonical = sorted([*identities[1:], identities[0].upper()])
    with pytest.raises(ValueError, match="canonical UUIDv5"):
        _filter_consistency_eval_population(population, noncanonical)
    missing = sorted(
        [
            *identities[1:],
            compute_registered_dog_id("yt-bb-dog:v1:video-track:missing"),
        ]
    )
    with pytest.raises(ValueError, match="does not exactly match"):
        _filter_consistency_eval_population(population, missing)


def test_cli_help() -> None:
    tool = Path(__file__).resolve().parents[1] / "workflows" / "evaluate_yt_nose_architecture.py"
    completed = subprocess.run(
        [sys.executable, str(tool), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--mask-manifest-sha256" in completed.stdout
    assert "--bootstrap-resamples" in completed.stdout
    assert "--calibration-fraction" in completed.stdout
    assert "--calibration-seed" in completed.stdout
    assert "--fusion-grid-step" in completed.stdout
    assert "--embedding-lineage" in completed.stdout
    assert "--embedding-lineage-sha256" in completed.stdout
    assert "--population-role" in completed.stdout
