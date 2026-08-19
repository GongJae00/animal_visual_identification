from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from legacy.version.afn.experiments.fixed_multievidence import (
    DEV_FRACTION,
    FRAMES_PER_WINDOW,
    MINIMUM_DEV_IDENTITIES,
    MINIMUM_EVAL_IDENTITIES,
    PANEL_BUNDLE_SCHEMA_VERSION,
    PANEL_SCHEMA_VERSION,
    SPLIT_COMMITMENT,
    partition_identities,
)
from legacy.version.common.experiments.identity_topology import FIXED_PANEL_TOPOLOGY_MANIFEST_SCHEMA_VERSION
from legacy.version.n4.experiments.n4_metric_adapter import (
    CACHE_BUNDLE_SCHEMA_VERSION,
    CACHE_SCHEMA_VERSION,
    N3_BRANCH,
    REPORT_BUNDLE_SCHEMA_VERSION,
    DeterministicPKBatchSampler,
    ResidualMetricAdapter,
    apply_adapter,
    batch_hard_metric_loss,
    build_adapter_checkpoint,
    evaluate_k5,
    evaluate_metric_adapter,
    load_adapter_checkpoint,
    materialize_embedding_cache,
    select_dev_epoch,
    train_metric_adapter,
    validate_adapter_checkpoint,
    validate_cache_manifest,
    validate_evaluation_bundle,
)
from foundation.provenance import content_sha256

_PANEL_LIMITATIONS = [
    "SAME_VIDEO_TRACK_GALLERY_AND_QUERY",
    "PUBLISHER_TEST_EXPOSED_DIAGNOSTIC",
    "PRIOR_PUBLISHER_TEST_EXPOSURE",
    "CLOSED_SET_ONLY_NO_UNKNOWN_REJECTION",
    "TRACK_IDENTITIES_NOT_LIFELONG_DOG_IDENTITIES",
    "WEAK_NOSE_ROI_INPUT_DIFFERS_FROM_NATIVE_NOSE_TRAINING_INPUT",
    "NOT_FINAL_EVALUATION",
    "NO_BIOMETRIC_OR_OPEN_SET_CLAIM",
]
_PANEL_CODE_PATHS = (
    "legacy/version/afn/experiments/fixed_multievidence.py",
    "embedding/methods/face/checkpoint.py",
    "parsing/roi_manifest.py",
    "embedding/methods/nose/training/embedding_consistency_training.py",
    "legacy/version/face/workflows/train_roi_face_reid.py",
    "legacy/version/afn/workflows/build_fixed_multievidence_panel.py",
)
_HASH = "0" * 64


def _unit(index: int, dimension: int = 4) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index % dimension] = 1.0
    return vector


def _cache_row(index: int, role: str, identity: str, track: str, frame: int) -> dict:
    return {
        "row_index": index,
        "sample_token": f"sample-{role.lower()}-{track}-{frame:02d}",
        "registered_identity_id": identity,
        "identity_token": hashlib.sha256(identity.encode()).hexdigest(),
        "track_token": track,
        "sequence_token": f"sequence-{track}",
        "frame_index": frame,
        "record_state": "AVAILABLE",
        "role": role,
        "quality": {"blur_score": 0.8},
        "quality_weight": 0.75,
        "source_hashes": {
            "source_sha256": hashlib.sha256(
                f"source-{role}-{track}-{frame}".encode()
            ).hexdigest(),
            "crop_sha256": hashlib.sha256(
                f"crop-{role}-{track}-{frame}".encode()
            ).hexdigest(),
            "soft_mask_sha256": hashlib.sha256(
                f"soft-{role}-{track}-{frame}".encode()
            ).hexdigest(),
            "binary_mask_sha256": hashlib.sha256(
                f"binary-{role}-{track}-{frame}".encode()
            ).hexdigest(),
        },
    }


def _write_cache(tmp_path: Path) -> tuple[Path, dict, np.ndarray]:
    rows = []
    vectors = []
    for track_index, track in enumerate(("train-track-a", "train-track-b")):
        for frame in range(2):
            rows.append(
                _cache_row(
                    len(rows), "TRAIN", f"train-identity-{track_index}", track, frame
                )
            )
            vectors.append(_unit(track_index))
    for track_index, track in enumerate(("dev-track-a", "dev-track-b")):
        for frame in range(10):
            rows.append(
                _cache_row(
                    len(rows), "DEV", f"dev-identity-{track_index}", track, frame
                )
            )
            vectors.append(_unit(track_index + 2))
    matrix = np.stack(vectors).astype(np.float32)
    matrix_path = tmp_path / "embeddings.npy"
    with matrix_path.open("wb") as stream:
        np.save(stream, matrix, allow_pickle=False)
    identity_lists = {
        "parent_seen_yt": [],
        "parent_seen_native_ssl_train": [],
        "ssl_train": ["train-identity-0", "train-identity-1"],
        "dev": ["dev-identity-0", "dev-identity-1"],
        "eval": ["lineage-eval-only"],
    }
    input_bindings = {
        "n3_lineage": {
            "content_sha256": "1" * 64,
            "lineage_sha256": "2" * 64,
        },
        "n3_runtime_manifest": {"content_sha256": "3" * 64},
        "n3_onnx": {"sha256": "4" * 64},
        "lineage_identity_lists": identity_lists,
    }
    cache = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "FROZEN_N3_EMBEDDINGS_FOR_N4_TRAIN_AND_DEV_ONLY",
        "interpretation": (
            "TRACK_PROXY_METRIC_LEARNING_NOT_LIFELONG_IDENTITY_OR_PHYSICAL_NOSE_TOPOLOGY"
        ),
        "protocol": {
            "source_roles": ["ssl_train", "dev"],
            "emitted_roles": ["TRAIN", "DEV"],
            "lineage_eval_included": False,
            "publisher_test_included": False,
            "train_record_states": ["AVAILABLE", "LOW_QUALITY"],
            "dev_record_states": ["AVAILABLE", "LOW_QUALITY"],
            "train_label": "track_token proxy, not lifelong identity",
            "quality_weight": (
                "0.5 + 0.5 * mean(blur_score, contrast_score, "
                "detector_confidence, frontality, 1-mask_uncertainty)"
            ),
        },
        "input_bindings": input_bindings,
        "input_bindings_sha256": content_sha256(input_bindings),
        "matrix": {
            "path": "embeddings.npy",
            "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            "shape": list(matrix.shape),
            "dtype": "float32",
            "format": "NUMPY_NPY_ALLOW_PICKLE_FALSE",
        },
        "rows": rows,
        "exclusions": {
            "train_identity_reason": "FEWER_THAN_TWO_AVAILABLE_FRAMES",
            "train_registered_identity_ids": [],
            "low_quality_train_sample_tokens": [],
            "no_roi_sample_tokens_not_cached": [],
        },
        "code_sha256s": {"synthetic.py": _HASH},
    }
    bundle = {
        "schema_version": CACHE_BUNDLE_SCHEMA_VERSION,
        "cache_sha256": content_sha256(cache),
        "cache": cache,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    return path, bundle, matrix


def _history(rank1: float = 1.0, mrr: float = 1.0, rank5: float = 1.0) -> list[dict]:
    return [
        {
            "epoch": 0,
            "train": None,
            "adapter_change": 0.0,
            "dev": {"metrics": {"Rank-1": rank1, "MRR": mrr, "Rank-5": rank5}},
        }
    ]


def _checkpoint(*, exposed_identity: str | None = None) -> dict:
    torch.manual_seed(3)
    model = ResidualMetricAdapter(4, 2, scale=0.1)
    history = _history()
    selection = select_dev_epoch(history)
    identity_lists = {
        "parent_seen_yt": [],
        "parent_seen_native_ssl_train": [],
        "ssl_train": ["train-identity"],
        "dev": ["dev-identity"],
        "eval": ([] if exposed_identity is None else [exposed_identity]),
    }
    config = {
        "architecture": "Linear-ReLU-Linear residual adapter",
        "input_dimension": 4,
        "bottleneck_dimension": 2,
        "scale": 0.1,
        "scale_maximum": 0.1,
        "epochs_requested": 1,
        "epochs_completed": 0,
        "early_stop_patience": 1,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "sampler": {
            "name": "DETERMINISTIC_P_K_TRACK_PROXY",
            "tracks_per_batch": 2,
            "samples_per_track": 2,
            "seed": 3,
        },
        "loss": {
            "metric": "BATCH_HARD_COSINE_TRIPLET_TRAIN_ONLY_HARD_NEGATIVES",
            "margin": 0.2,
            "parent_anchor": "BOUNDED_ONE_MINUS_COSINE_IN_[0,2]",
            "parent_anchor_weight": 0.25,
            "sample_quality_weight_range": [0.5, 1.0],
        },
        "labels": "track_token proxy labels are not lifelong identities",
        "selection": "LINEAGE_DEV_EXACT_EARLIEST_LATEST_K5_ONLY",
        "lineage_eval_used": False,
        "publisher_test_used": False,
        "device": "cpu",
    }
    bindings = {
        "cache_manifest": {
            "path": "/tmp/cache.json",
            "content_sha256": "5" * 64,
            "cache_sha256": "6" * 64,
            "matrix_sha256": "7" * 64,
        },
        "n3": {
            "lineage_payload_sha256": "1" * 64,
            "lineage_sha256": "2" * 64,
            "runtime_manifest_payload_sha256": "3" * 64,
            "onnx_sha256": "4" * 64,
            "identity_lists": identity_lists,
        },
        "train_registered_identity_ids": ["train-identity"],
        "dev_registered_identity_ids": ["dev-identity"],
        "train_track_tokens_sha256": "8" * 64,
        "dev_track_tokens_sha256": "9" * 64,
    }
    return build_adapter_checkpoint(
        state_dict=model.state_dict(),
        config=config,
        bindings=bindings,
        selected_epoch=0,
        history=history,
        selection=selection,
        worktree_provenance={
            "git_head": "a" * 40,
            "worktree_dirty": True,
            "git_status_porcelain_sha256": "b" * 64,
        },
        code_sha256s={"synthetic.py": _HASH},
    )


def _write_checkpoint(
    tmp_path: Path, checkpoint: dict, name: str = "adapter.pt"
) -> Path:
    path = tmp_path / name
    torch.save(checkpoint, path)
    return path


def _panel_population() -> tuple[list[str], list[str], list[str]]:
    identities: list[str] = []
    for index in range(10_000):
        identities.append(
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"panel-identity-{index:05d}"))
        )
        ordered = sorted(identities)
        dev, evaluation = partition_identities(ordered)
        if (
            len(dev) >= MINIMUM_DEV_IDENTITIES
            and len(evaluation) >= MINIMUM_EVAL_IDENTITIES
        ):
            return ordered, dev, evaluation
    raise AssertionError("could not form fixed-panel minimum population")


def _fixed_panel() -> tuple[dict, dict]:
    identities, dev_ids, eval_ids = _panel_population()
    partition = {
        **{identity: "DEV" for identity in dev_ids},
        **{identity: "EVAL" for identity in eval_ids},
    }
    records = []
    topology = []
    for identity_index, identity in enumerate(identities, start=1):
        angle = 2.0 * np.pi * identity_index / len(identities)
        vector = [float(np.cos(angle)), float(np.sin(angle)), 0.0, 0.0]
        for role, frames in (("gallery", range(5)), ("query", range(10, 15))):
            for frame in frames:
                sample = f"sample-{identity_index}-{frame}"
                quality = {"overall": 0.8}
                records.append(
                    {
                        "sample_id": sample,
                        "instance_id": f"instance-{sample}",
                        "registered_identity_id": identity,
                        "partition": partition[identity],
                        "window_role": role,
                        "publisher_frame_index": frame,
                        "split_role": "test",
                        "capture_group_id": f"track-{identity_index:05d}",
                        "capture_group_kind": "VIDEO_TRACK",
                        "source": {
                            "path": (
                                f"YT-BB-Dog/test/{identity_index}/"
                                f"{identity_index}_{frame}.jpg"
                            ),
                            "sha256": hashlib.sha256(
                                f"source-{sample}".encode()
                            ).hexdigest(),
                            "quality": quality,
                        },
                        "face": {
                            "path": f"faces/{sample}.jpg",
                            "sha256": hashlib.sha256(
                                f"face-{sample}".encode()
                            ).hexdigest(),
                            "quality": {"overall": 0.7},
                        },
                        "weak_nose": {
                            "path": f"noses/{sample}.jpg",
                            "sha256": hashlib.sha256(
                                f"nose-{sample}".encode()
                            ).hexdigest(),
                            "quality": quality,
                            "quality_semantics": (
                                "DOG_ROI_QUALITY_PROXY_NO_NOSE_SPECIFIC_SCORE"
                            ),
                        },
                    }
                )
                topology.append(
                    {
                        "sample_token": sample,
                        "identity_token": identity,
                        "session_token": f"track-{identity_index:05d}",
                        "branch": N3_BRANCH,
                        "quality": 0.8,
                        "available": True,
                        "embedding": vector,
                    }
                )
    exposure_lists = {
        "f5_train": [],
        "f5_model_selection": [],
        "n3_parent_seen_yt": [],
        "n3_parent_seen_native_ssl_train": [],
        "n3_ssl_train": [],
        "n3_dev": [],
        "n3_eval": [],
    }
    exposure = {
        "lists": exposure_lists,
        "sha256s": {
            name: content_sha256({"identity_ids": values})
            for name, values in exposure_lists.items()
        },
    }
    panel = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "status": "FROZEN_EXPOSED_PUBLISHER_TEST_DIAGNOSTIC_PANEL",
        "interpretation": "SYNTHETIC_FIXED_PANEL",
        "protocol": {
            "population_source": "PUBLISHER_TEST_ROI_MANIFEST_ONLY",
            "selection_uses_model_outputs": False,
            "split_commitment": SPLIT_COMMITMENT,
            "dev_fraction": DEV_FRACTION,
            "gallery_selection": "EARLIEST_FIVE_BY_PARSED_PUBLISHER_FRAME_INDEX",
            "query_selection": "LATEST_FIVE_BY_PARSED_PUBLISHER_FRAME_INDEX",
            "frames_per_window": FRAMES_PER_WINDOW,
            "same_track_only": True,
            "limitations": _PANEL_LIMITATIONS,
        },
        "input_bindings": {
            "roi_manifest_bundle": {
                "path": "/tmp/roi.json", "raw_sha256": "5" * 64,
                "content_sha256": "6" * 64, "byte_size": 1,
            },
            "roi_manifest_sha256": "7" * 64,
            "source_image_root": "/tmp/source",
            "f5_checkpoint": {
                "path": "/tmp/f5.pt", "sha256": "0" * 64,
                "training_split_sha256": "8" * 64,
            },
            "f5_training_roi_manifest_bundle": {
                "path": "/tmp/train-roi.json", "raw_sha256": "9" * 64,
                "content_sha256": "a" * 64, "byte_size": 1,
            },
            "f5_training_roi_manifest_sha256": "b" * 64,
            "n3_lineage": {
                "path": "/tmp/lineage.json", "raw_sha256": "c" * 64,
                "content_sha256": "1" * 64, "byte_size": 1,
                "lineage_sha256": "d" * 64,
            },
        },
        "exposure_identity_lists": exposure,
        "population": {
            "observed_publisher_test_identity_ids": identities,
            "eligible_identity_ids": identities,
            "dev_identity_ids": dev_ids,
            "eval_identity_ids": eval_ids,
            "dev_identity_ids_sha256": content_sha256({"identity_ids": dev_ids}),
            "eval_identity_ids_sha256": content_sha256({"identity_ids": eval_ids}),
            "minimum_dev_identities": MINIMUM_DEV_IDENTITIES,
            "minimum_eval_identities": MINIMUM_EVAL_IDENTITIES,
        },
        "exclusions": {},
        "records": records,
        "code_sha256s": {path: _HASH for path in _PANEL_CODE_PATHS},
    }
    bundle = {
        "schema_version": PANEL_BUNDLE_SCHEMA_VERSION,
        "panel_sha256": content_sha256(panel),
        "panel": panel,
    }
    topology_bindings = {
        "panel_bundle_content_sha256": content_sha256(bundle),
        "panel_sha256": bundle["panel_sha256"],
        "frozen_dinov2_sha256": "a" * 64,
        "f5_checkpoint_sha256": "0" * 64,
        "n3_lineage_content_sha256": "1" * 64,
        "n3_runtime_manifest_content_sha256": "3" * 64,
        "n3_runtime_manifest_raw_sha256": "b" * 64,
        "n3_onnx_sha256": "4" * 64,
        "execution": {"device": "cpu", "n3_device": "cpu", "batch_size": 2},
    }
    topology_manifest = {
        "schema_version": FIXED_PANEL_TOPOLOGY_MANIFEST_SCHEMA_VERSION,
        "input_bindings": topology_bindings,
        "input_bindings_sha256": content_sha256(topology_bindings),
        "records": topology,
    }
    return bundle, topology_manifest


def test_cache_row_matrix_integrity_roles_and_no_pickle(tmp_path: Path) -> None:
    path, bundle, matrix = _write_cache(tmp_path)
    cache = validate_cache_manifest(bundle, root=tmp_path)

    loaded = np.load(tmp_path / "embeddings.npy", allow_pickle=False)
    assert loaded.dtype == np.float32
    assert loaded == pytest.approx(matrix)
    assert {row["role"] for row in cache["rows"]} == {"TRAIN", "DEV"}
    assert not (
        {
            row["registered_identity_id"]
            for row in cache["rows"]
            if row["role"] == "TRAIN"
        }
        & {
            row["registered_identity_id"]
            for row in cache["rows"]
            if row["role"] == "DEV"
        }
    )
    assert path.is_file()

    tampered = copy.deepcopy(bundle)
    tampered["cache"]["rows"][0]["row_index"] = 7
    tampered["cache_sha256"] = content_sha256(tampered["cache"])
    with pytest.raises(ValueError, match="row indices"):
        validate_cache_manifest(tampered)


def test_cache_rejects_object_npy_even_when_file_hash_is_rebound(
    tmp_path: Path,
) -> None:
    _, bundle, matrix = _write_cache(tmp_path)
    with (tmp_path / "embeddings.npy").open("wb") as stream:
        objects = np.empty(matrix.shape, dtype=object)
        objects.fill({"unsafe": True})
        np.save(stream, objects, allow_pickle=True)
    tampered = copy.deepcopy(bundle)
    tampered["cache"]["matrix"]["sha256"] = hashlib.sha256(
        (tmp_path / "embeddings.npy").read_bytes()
    ).hexdigest()
    tampered["cache_sha256"] = content_sha256(tampered["cache"])
    with pytest.raises(ValueError, match="Object arrays"):
        validate_cache_manifest(tampered, root=tmp_path)


def test_cache_materialization_refuses_overwrite_before_reading_inputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing-cache"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_embedding_cache(
            native_bundle_path=tmp_path / "missing-native.json",
            native_bundle_sha256=_HASH,
            native_root=tmp_path,
            n3_lineage_path=tmp_path / "missing-lineage.json",
            n3_lineage_sha256=_HASH,
            n3_runtime_manifest_path=tmp_path / "missing-runtime.json",
            n3_runtime_manifest_sha256=_HASH,
            n3_onnx_path=tmp_path / "missing.onnx",
            n3_onnx_sha256=_HASH,
            output_dir=output,
        )


def test_sampler_and_loss_are_deterministic_and_have_train_hard_negatives() -> None:
    labels = ["a", "a", "b", "b", "c", "c"]
    sampler = DeterministicPKBatchSampler(
        labels, tracks_per_batch=2, samples_per_track=2, seed=11
    )
    assert sampler.batches(3) == sampler.batches(3)
    assert sampler.batches(2) != sampler.batches(3)

    parent = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=torch.float32,
    )
    parent = torch.nn.functional.normalize(parent, dim=1)
    quality = torch.tensor([0.5, 0.75, 1.0, 0.8])
    first, components = batch_hard_metric_loss(
        parent, parent, ["a", "a", "b", "b"], quality, margin=0.2, anchor_weight=0.25
    )
    second, _ = batch_hard_metric_loss(
        parent, parent, ["a", "a", "b", "b"], quality, margin=0.2, anchor_weight=0.25
    )
    assert first == second
    assert components["parent_anchor"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="positive and hard negative"):
        batch_hard_metric_loss(
            parent[:2],
            parent[:2],
            ["a", "a"],
            quality[:2],
            margin=0.2,
            anchor_weight=0.25,
        )


def test_identity_adapter_outputs_are_normalized() -> None:
    checkpoint = _checkpoint()
    inputs = np.stack((_unit(0), _unit(1))).astype(np.float32)
    outputs = apply_adapter(checkpoint, inputs)
    assert outputs == pytest.approx(inputs)
    assert np.linalg.norm(outputs, axis=1) == pytest.approx([1.0, 1.0])


def test_epoch_zero_tie_and_non_regression_gate() -> None:
    history = _history(rank1=0.5, mrr=0.7, rank5=0.9)
    history.append(
        {
            "epoch": 1,
            "train": {"loss": 1.0},
            "adapter_change": 0.01,
            "dev": {"metrics": {"Rank-1": 0.5, "MRR": 0.7, "Rank-5": 0.9}},
        }
    )
    assert select_dev_epoch(history)["selected_epoch"] == 0

    regression = copy.deepcopy(history)
    regression[1]["dev"]["metrics"] = {"Rank-1": 0.6, "MRR": 0.69, "Rank-5": 1.0}
    selection = select_dev_epoch(regression)
    assert selection["selected_epoch"] == 0
    assert selection["decisions"][1]["admissible"] is False


def test_checkpoint_tampering_fails_closed(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    path = _write_checkpoint(tmp_path, checkpoint)
    assert (
        load_adapter_checkpoint(path)["checkpoint_payload_sha256"]
        == checkpoint["checkpoint_payload_sha256"]
    )

    tampered = copy.deepcopy(checkpoint)
    tampered["state_dict"]["up.bias"][0] = 0.5
    with pytest.raises(ValueError, match="state digest"):
        validate_adapter_checkpoint(tampered)

    with pytest.raises(ValueError, match="external pin"):
        load_adapter_checkpoint(path, expected_file_sha256=_HASH)


def test_k5_paired_metrics_and_failure_paths() -> None:
    rows = []
    vectors = []
    for index, track in enumerate(("a", "b")):
        for frame in range(10):
            rows.append(
                {
                    "sample_token": f"{track}-{frame}",
                    "registered_identity_id": f"identity-{track}",
                    "track_token": track,
                    "frame_index": frame,
                }
            )
            vectors.append(_unit(index))
    result = evaluate_k5(np.stack(vectors), rows)
    assert result["metrics"]["Rank-1"] == 1.0
    assert result["metrics"]["MRR"] == 1.0
    assert len(result["outcomes"]) == 2
    with pytest.raises(ValueError, match="fewer than ten"):
        evaluate_k5(np.stack(vectors[:-1]), rows[:-1])


def test_gpu_free_training_selects_identity_baseline_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    manifest_path, bundle, _ = _write_cache(tmp_path)
    output = tmp_path / "trained.pt"
    checkpoint = train_metric_adapter(
        cache_manifest_path=manifest_path,
        cache_manifest_sha256=content_sha256(bundle),
        output_checkpoint_path=output,
        epochs=1,
        bottleneck_dim=2,
        tracks_per_batch=2,
        samples_per_track=2,
        patience=1,
        seed=5,
    )
    assert checkpoint["selected_epoch"] == 0
    assert checkpoint["config"]["device"] == "cpu"
    assert output.stat().st_mode & 0o077 == 0

    with pytest.raises(FileExistsError, match="overwrite"):
        train_metric_adapter(
            cache_manifest_path=tmp_path / "missing.json",
            cache_manifest_sha256=_HASH,
            output_checkpoint_path=output,
        )


@pytest.fixture(scope="module")
def fixed_panel() -> tuple[dict, dict]:
    return _fixed_panel()


def test_publisher_evaluation_reports_paired_metrics_and_dispersion(
    tmp_path: Path, fixed_panel: tuple[dict, dict]
) -> None:
    panel, topology = fixed_panel
    checkpoint_path = _write_checkpoint(tmp_path, _checkpoint())
    panel_path = tmp_path / "panel.json"
    topology_path = tmp_path / "topology.json"
    output = tmp_path / "evaluation.json"
    panel_path.write_text(json.dumps(panel, sort_keys=True), encoding="utf-8")
    topology_path.write_text(json.dumps(topology, sort_keys=True), encoding="utf-8")
    bundle = evaluate_metric_adapter(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        panel_path=panel_path,
        panel_sha256=content_sha256(panel),
        topology_manifest_path=topology_path,
        topology_sha256=content_sha256(topology),
        output_path=output,
        bootstrap_resamples=5,
        bootstrap_seed=7,
    )
    report = validate_evaluation_bundle(bundle)
    assert bundle["schema_version"] == REPORT_BUNDLE_SCHEMA_VERSION
    assert (
        report["evaluation"]["baseline_N3"]["metrics"]
        == report["evaluation"]["candidate_N4"]["metrics"]
    )
    assert set(report["evaluation"]["paired_N4_minus_N3_identity_bootstrap_cis"]) == {
        "Rank-1",
        "MRR",
        "Rank-5",
    }
    assert report["evaluation"]["rescue_break"]["rescue_count"] == 0
    assert set(report["evaluation"]["embedding_topology_dispersion"]) == {
        "before_N3",
        "after_N4",
    }
    assert report["protocol"]["publisher_dev_used_for_selection"] is False
    assert report["protocol"]["physical_nose_topology_claim"] is False

    with pytest.raises(FileExistsError, match="overwrite"):
        evaluate_metric_adapter(
            checkpoint_path=tmp_path / "missing.pt",
            checkpoint_sha256=_HASH,
            panel_path=tmp_path / "missing-panel.json",
            panel_sha256=_HASH,
            topology_manifest_path=tmp_path / "missing-topology.json",
            topology_sha256=_HASH,
            output_path=output,
        )


def test_publisher_panel_overlap_fails_closed(
    tmp_path: Path, fixed_panel: tuple[dict, dict]
) -> None:
    panel, topology = fixed_panel
    exposed = panel["panel"]["population"]["eval_identity_ids"][0]
    checkpoint_path = _write_checkpoint(tmp_path, _checkpoint(exposed_identity=exposed))
    panel_path = tmp_path / "panel.json"
    topology_path = tmp_path / "topology.json"
    panel_path.write_text(json.dumps(panel, sort_keys=True), encoding="utf-8")
    topology_path.write_text(json.dumps(topology, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_metric_adapter(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            panel_path=panel_path,
            panel_sha256=content_sha256(panel),
            topology_manifest_path=topology_path,
            topology_sha256=content_sha256(topology),
            output_path=tmp_path / "output.json",
            bootstrap_resamples=2,
        )


@pytest.mark.parametrize(
    "workflow",
    (
        "materialize_n4_embedding_cache.py",
        "train_n4_metric_adapter.py",
        "evaluate_n4_metric_adapter.py",
    ),
)
def test_workflow_help_exposes_external_pins(workflow: str) -> None:
    tool = Path(__file__).resolve().parents[1] / "legacy/version/n4/workflows" / workflow
    completed = subprocess.run(
        [sys.executable, str(tool), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "sha256" in completed.stdout
