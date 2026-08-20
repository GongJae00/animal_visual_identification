from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from shared.contracts.artifact_manifest import (
    ExactOnnxRuntime,
    NoseEmbeddingManifest,
    UsageLane,
    preprocess_image,
)
from shared.contracts.model_parity import ParityFixtureKind, ParityThresholds
from identification.training.nose import embedding_consistency_training as consistency
from identification.training.nose.embedding_consistency_training import (
    build_consistency_checkpoint,
    build_identity_partitions,
    deterministic_mild_degradation,
    initialize_from_parent,
    load_consistency_checkpoint,
    native_consistency_loss,
    select_epoch,
)
from identification.training.nose.embedding_training import (
    DEV_INTERPRETATION,
    EMBEDDING_DIM,
    LICENSE_ID,
    ArcFaceClassificationHead,
    IdentityBalancedBatchSampler,
    NoseEmbeddingModel,
    build_dev_protocol,
    build_embedding_checkpoint,
    build_runtime_manifest,
    evaluate_dev_leave_one_out,
    export_static_onnx,
    load_embedding_checkpoint,
    produce_parity_receipt,
    replace_embedding_checkpoint,
)
from identification.export.nose.data.embedding_views import student_masked_rgb

def _dog(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"test:nose-region:{name}"))

def _record(identity: str, sample: str, session: str = "a") -> dict[str, str]:
    token = hashlib.sha256(sample.encode("ascii")).hexdigest()
    return {
        "split_role": "DEV",
        "registered_dog_id": identity,
        "sample_token": token,
        "capture_session_token": hashlib.sha256(session.encode("ascii")).hexdigest(),
    }

def _unit_vector(index: int) -> np.ndarray:
    result = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    result[index] = 1.0
    return result

def _checkpoint_bindings() -> dict:
    train_id = _dog("checkpoint-train")
    eligible_id = _dog("checkpoint-eligible")
    singleton_id = _dog("checkpoint-singleton")
    eligible_samples = ["1" * 64, "2" * 64]
    return {
        "training_code": {
            "embedding_training_sha256": "0" * 64,
            "tool_sha256": "f" * 64,
        },
        "crop_manifest": {
            "payload_sha256": "a" * 64,
            "file_sha256": "b" * 64,
            "protocol_plan_sha256": "c" * 64,
            "summary_sha256": "d" * 64,
        },
        "identity_populations": {
            "train_registered_dog_ids": [train_id],
            "dev_registered_dog_ids": sorted((eligible_id, singleton_id)),
            "evaluation_eligible_registered_dog_ids": [eligible_id],
            "excluded_singleton_registered_dog_ids": [singleton_id],
        },
        "development_protocol": {
            "interpretation": DEV_INTERPRETATION,
            "identity_disjoint_from_train": True,
            "same_session_within_each_dev_identity": True,
            "selection_metric_order": ["DEV_mAP", "DEV_Rank-1"],
            "dev_registered_dog_ids": sorted((eligible_id, singleton_id)),
            "eligible_registered_dog_ids": [eligible_id],
            "excluded_singleton_registered_dog_ids": [singleton_id],
            "eligible_sample_tokens": eligible_samples,
            "eligible_image_count": 2,
        },
        "dinov2": {
            "source_model_id": "facebook/dinov2-small",
            "source_revision": "frozen-test-revision",
            "weight_sha256": "3" * 64,
            "preprocessor_sha256": "4" * 64,
            "config_sha256": "5" * 64,
            "weight_intake_receipt_sha256": "6" * 64,
            "preprocessor_intake_receipt_sha256": "7" * 64,
            "weight_intake_bundle_file_sha256": "8" * 64,
            "preprocessor_intake_bundle_file_sha256": "9" * 64,
        },
        "license": {
            "license_id": LICENSE_ID,
            "usage_lane": "RESEARCH_ONLY",
            "inherited_crop_licensing_lanes": [
                "RESEARCH_ONLY_CC_BY_NC_4_0_DERIVED_LOCALIZER"
            ],
        },
    }

def _checkpoint_config() -> dict:
    return {
        "schema_version": "cvi.nose_region_rgb_embedding_training_config.v1",
        "model": "DINOv2-small CLS L2-normalized 384D",
        "input": {
            "shape": [3, 224, 224],
            "resize": "DIRECT_BICUBIC_STRETCH",
            "scale": 1.0 / 255.0,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "augmentation": {
            "interpretation": "MILD_REGION_SAFE_NO_GENERIC_RANDAUGMENT"
        },
        "optimizer": {"type": "AdamW"},
        "arcface": {"scale": 30.0, "margin": 0.5},
        "embedding_consistency_weight": 5.0,
        "label_smoothing": 0.1,
        "freeze_backbone_epochs": 1,
        "epochs": 1,
        "batch_size": 2,
        "samples_per_identity_per_epoch": 1,
        "num_workers": 0,
        "seed": 3,
        "gradient_checkpointing": False,
        "mixed_precision": False,
        "device": "cpu",
        "parity": {
            "receipt_bound_crop_count": 1,
            "thresholds": ParityThresholds(1e-4, 2e-2, 1e-4, 0.99999).to_dict(),
        },
        "selection_metric_order": ["DEV_mAP", "DEV_Rank-1"],
        "selection_interpretation": DEV_INTERPRETATION,
    }

class _FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("offset", torch.linspace(0.1, 0.9, EMBEDDING_DIM))

    def forward(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = pixel_values.mean(dim=(2, 3)).repeat(1, EMBEDDING_DIM // 3)
        return pooled + self.offset[None]

def test_dev_leave_one_out_excludes_singletons_and_is_deterministic() -> None:
    dog_a, dog_b, singleton = _dog("a"), _dog("b"), _dog("singleton")
    records = (
        _record(dog_a, "a1"),
        _record(dog_a, "a2"),
        _record(dog_b, "b1"),
        _record(dog_b, "b2"),
        _record(singleton, "single"),
    )
    embeddings = np.stack(
        (
            _unit_vector(0),
            _unit_vector(0),
            _unit_vector(1),
            _unit_vector(1),
            _unit_vector(2),
        )
    )
    protocol = build_dev_protocol(records)
    report = evaluate_dev_leave_one_out(embeddings, records, protocol)

    assert protocol["eligible_registered_dog_ids"] == sorted((dog_a, dog_b))
    assert protocol["excluded_singleton_registered_dog_ids"] == [singleton]
    assert protocol["eligible_image_count"] == 4
    assert report["mAP"] == 1.0
    assert report["Rank-1"] == 1.0
    assert report["query_count"] == 4
    assert report["interpretation"] == DEV_INTERPRETATION

def test_dev_protocol_rejects_cross_session_interpretation() -> None:
    identity = _dog("multi-session")
    records = (
        _record(identity, "one", "session-one"),
        _record(identity, "two", "session-two"),
    )
    with pytest.raises(ValueError, match="same-session DEV diagnostic"):
        build_dev_protocol(records)

def test_dev_protocol_requires_two_retrieval_identities() -> None:
    eligible = _dog("only-eligible")
    records = (
        _record(eligible, "one"),
        _record(eligible, "two"),
        _record(_dog("single"), "single"),
    )
    with pytest.raises(ValueError, match="at least two identities"):
        build_dev_protocol(records)

def test_identity_balanced_sampler_includes_one_shot_without_dominance() -> None:
    labels = [0, 1, 1, 1, 1, 2, 2]
    sampler = IdentityBalancedBatchSampler(
        labels, batch_size=2, samples_per_identity=2, seed=7
    )
    batches = list(sampler)
    selected_labels = [labels[index] for batch in batches for index in batch]

    assert {label: selected_labels.count(label) for label in set(labels)} == {
        0: 2,
        1: 2,
        2: 2,
    }
    assert all(len({labels[index] for index in batch}) == len(batch) for batch in batches)

def test_checkpoint_is_weights_only_safe_and_rejects_metadata_and_tensor_tamper() -> None:
    model = NoseEmbeddingModel(_FakeBackbone())
    arcface = ArcFaceClassificationHead(2)
    bindings = _checkpoint_bindings()
    config = _checkpoint_config()
    selection = {
        "metric_order": ["DEV_mAP", "DEV_Rank-1"],
        "mAP": 0.75,
        "Rank-1": 0.5,
        "interpretation": DEV_INTERPRETATION,
    }
    payload = build_embedding_checkpoint(
        model=model,
        arcface=arcface,
        epoch=0,
        global_step=0,
        selection=selection,
        bindings=bindings,
        training_config=config,
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "best.pt"
        replace_embedding_checkpoint(path, payload)
        loaded = load_embedding_checkpoint(
            path, expected_bindings=bindings, expected_training_config=config
        )
        assert loaded["schema_version"].endswith(".v1")
        assert torch.load(path, weights_only=True)["bindings"] == bindings

        metadata_tamper = torch.load(path, weights_only=True)
        metadata_tamper["bindings"]["crop_manifest"]["payload_sha256"] = "b" * 64
        metadata_path = root / "metadata-tamper.pt"
        torch.save(metadata_tamper, metadata_path)
        with pytest.raises(ValueError, match="bindings digest"):
            load_embedding_checkpoint(metadata_path)

        tensor_tamper = torch.load(path, weights_only=True)
        tensor_tamper["model_state_dict"]["backbone.offset"][0] += 1.0
        tensor_path = root / "tensor-tamper.pt"
        torch.save(tensor_tamper, tensor_path)
        with pytest.raises(ValueError, match="model state digest"):
            load_embedding_checkpoint(tensor_path)

def test_runtime_manifest_is_exact_static_research_contract() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.onnx"
        path.write_bytes(b"fixture-onnx")
        manifest = build_runtime_manifest(path)
        loaded = NoseEmbeddingManifest.from_dict(manifest.to_dict())

    assert loaded.input_name == "images"
    assert loaded.input_shape == (1, 3, 224, 224)
    assert loaded.output_name == "embedding"
    assert loaded.output_shape == (1, 384)
    assert loaded.preprocessing.resize == "bicubic"
    assert loaded.preprocessing.scale == 1.0 / 255.0
    assert loaded.license.license_id == LICENSE_ID
    assert loaded.license.usage_lane is UsageLane.RESEARCH_ONLY

def test_fake_backbone_static_onnx_and_actual_crop_parity() -> None:
    model = NoseEmbeddingModel(_FakeBackbone()).eval()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        onnx_path = root / "nose.onnx"
        export_static_onnx(model, onnx_path)
        manifest = build_runtime_manifest(onnx_path)
        runtime = ExactOnnxRuntime(onnx_path, manifest)
        image = Image.new("RGB", (31, 47), color=(80, 120, 160))
        output = runtime.run(preprocess_image(image, manifest))
        assert output.shape == (1, EMBEDDING_DIM)
        assert np.linalg.norm(output[0]) == pytest.approx(1.0, abs=1e-4)

        crop_path = root / "crop.png"
        image.save(crop_path, format="PNG")
        crop_sha256 = hashlib.sha256(crop_path.read_bytes()).hexdigest()
        receipt = produce_parity_receipt(
            model=model,
            onnx_path=onnx_path,
            runtime_manifest=manifest,
            crop_root=root,
            crop_records=(
                {
                    "sample_token": "1" * 64,
                    "crop_path": "crop.png",
                    "crop_sha256": crop_sha256,
                },
            ),
            crop_manifest_file_sha256="2" * 64,
            source_weights_sha256="3" * 64,
            weight_intake_receipt_sha256="4" * 64,
            preprocessor_intake_receipt_sha256="5" * 64,
            thresholds=ParityThresholds(1e-4, 2e-2, 1e-4, 0.99999),
        )

    assert receipt.decision == "PASS"
    assert any(
        fixture.fixture_kind is ParityFixtureKind.RECEIPT_BOUND_CROP
        for fixture in receipt.fixtures
    )
    assert sum(
        fixture.fixture_kind is ParityFixtureKind.SYNTHETIC
        for fixture in receipt.fixtures
    ) == 2

def _partition_identity(prefix: str, role: str) -> str:
    for index in range(10_000):
        identity = _dog(f"{prefix}-{index}")
        digest = hashlib.sha256(f"73:{identity}".encode("ascii")).digest()
        is_dev = int.from_bytes(digest, "big") < int(0.30 * (1 << 256))
        if (role == "dev") == is_dev:
            return identity
    raise AssertionError("could not construct partition fixture")

def _native_rows(identity: str, prefix: str, count: int, state: str = "AVAILABLE") -> list[dict]:
    return [
        {
            "registered_dog_id": identity,
            "sample_token": hashlib.sha256(f"{prefix}-{index}".encode()).hexdigest(),
            "frame_index": index,
            "record_state": state,
        }
        for index in range(count)
    ]

def test_consistency_splits_are_disjoint_and_exclude_parent_seen_native() -> None:
    parent_seen = _dog("parent-seen-native")
    ssl_identity = _dog("short-native")
    dev_identity = _partition_identity("partition-dev", "dev")
    eval_identity = _partition_identity("partition-eval", "eval")
    old = [
        {
            "registered_dog_id": parent_seen,
            "dataset_name": "yt-bb-dog",
            "split_role": "TRAIN",
        }
    ]
    native = [
        *_native_rows(parent_seen, "seen", 12),
        *_native_rows(ssl_identity, "ssl", 3),
        *_native_rows(dev_identity, "dev", 2),
        *_native_rows(dev_identity, "dev-low", 8, "LOW_QUALITY"),
        *_native_rows(eval_identity, "eval", 2),
        *_native_rows(eval_identity, "eval-low", 8, "LOW_QUALITY"),
    ]

    result = build_identity_partitions(old, native)
    identities = result["identity_lists"]
    samples = result["sample_token_lists"]

    assert identities["parent_seen_native_ssl_train"] == [parent_seen]
    assert identities["ssl_train"] == sorted((parent_seen, ssl_identity))
    assert identities["dev"] == [dev_identity]
    assert identities["eval"] == [eval_identity]
    assert not (set(identities["ssl_train"]) & set(identities["dev"]))
    assert len(samples["ssl_train"]) == 15
    assert len(samples["dev"]) == 10
    assert len(samples["eval"]) == 10
    assert hashlib.sha256(b"dev-low-0").hexdigest() in samples["dev"]

def test_mask_pixels_and_degradation_are_exact_and_deterministic() -> None:
    crop = np.arange(9 * 7 * 3, dtype=np.uint8).reshape(9, 7, 3)
    support = np.zeros((224, 224), dtype=bool)
    support[20:80, 30:90] = True
    masked = student_masked_rgb(crop, support)
    resized = np.asarray(
        Image.fromarray(crop).resize((224, 224), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    median = np.median(resized, axis=(0, 1)).astype(np.float32)
    expected = np.where(
        support[..., None],
        resized,
        np.rint(0.25 * resized.astype(np.float32) + 0.75 * median),
    ).astype(np.uint8)
    assert np.array_equal(masked, expected)

    token = hashlib.sha256(b"degradation").hexdigest()
    first, report = deterministic_mild_degradation(
        crop, seed=19, epoch=2, sample_token=token
    )
    second, repeated = deterministic_mild_degradation(
        crop, seed=19, epoch=2, sample_token=token
    )
    assert np.array_equal(first, second)
    assert report == repeated
    assert report["kind"] in {"downsample", "blur", "JPEG", "noise"}
    assert report["loss_weight"] == consistency.DEGRADATION_WEIGHTS[report["kind"]]
    assert first.shape == crop.shape

def test_consistency_loss_is_finite_and_has_student_gradients() -> None:
    generator = torch.Generator().manual_seed(7)
    base = torch.randn((2, 2, EMBEDDING_DIM), generator=generator, requires_grad=True)
    raw = torch.nn.functional.normalize(base, dim=-1)
    masked = torch.nn.functional.normalize(base + 0.1, dim=-1)
    degraded = torch.nn.functional.normalize(base - 0.1, dim=-1)
    parent = torch.nn.functional.normalize(
        torch.randn((2, 2, EMBEDDING_DIM), generator=generator), dim=-1
    )
    total, parts = native_consistency_loss(
        raw,
        masked,
        degraded,
        parent,
        torch.tensor([[1.0, 0.5], [0.75, 0.25]]),
        torch.tensor([[1.0, 0.9], [0.8, 0.7]]),
    )
    total.backward()

    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in parts.values())
    assert base.grad is not None
    assert torch.isfinite(base.grad).all()
    assert float(base.grad.abs().sum()) > 0.0

def test_parent_model_and_arcface_are_initialized_exactly_and_frozen() -> None:
    parent_model = NoseEmbeddingModel(_FakeBackbone())
    parent_head = ArcFaceClassificationHead(2)
    checkpoint = build_embedding_checkpoint(
        model=parent_model,
        arcface=parent_head,
        epoch=0,
        global_step=0,
        selection={
            "metric_order": ["DEV_mAP", "DEV_Rank-1"],
            "mAP": 0.5,
            "Rank-1": 0.5,
            "interpretation": DEV_INTERPRETATION,
        },
        bindings=_checkpoint_bindings(),
        training_config=_checkpoint_config(),
    )
    student = NoseEmbeddingModel(_FakeBackbone())
    head = ArcFaceClassificationHead(2)
    with torch.no_grad():
        student.backbone.offset.zero_()
        head.weight.zero_()

    frozen = initialize_from_parent(student, head, checkpoint)

    for key, value in checkpoint["model_state_dict"].items():
        assert torch.equal(student.state_dict()[key], value)
        assert torch.equal(frozen.state_dict()[key], value)
    for key, value in checkpoint["arcface_state_dict"].items():
        assert torch.equal(head.state_dict()[key], value)
    assert not any(parameter.requires_grad for parameter in frozen.parameters())

def _selection_epoch(epoch: int, old: float, raw: float, masked: float, rank: float) -> dict:
    return {
        "epoch": epoch,
        "dev": {
            "old_mpdd_raw": {"mAP": old},
            "native_raw_k5": {"mAP": raw},
            "native_masked_k5": {"mAP": masked, "Rank-1": rank},
        },
    }

def test_epoch_zero_is_selected_when_all_trained_epochs_are_inadmissible() -> None:
    selected = select_epoch(
        [
            _selection_epoch(0, 0.8, 0.6, 0.55, 0.5),
            _selection_epoch(1, 0.78, 0.7, 0.7, 1.0),
            _selection_epoch(2, 0.9, 0.589, 0.8, 1.0),
        ]
    )
    assert selected["selected_epoch"] == 0
    assert selected["selected_objective"][-1] == 0

def test_selection_accepts_bounded_raw_tradeoff_for_masked_improvement() -> None:
    selected = select_epoch(
        [
            _selection_epoch(0, 0.8, 0.6, 0.55, 0.5),
            _selection_epoch(1, 0.795, 0.595, 0.60, 0.6),
        ]
    )
    assert selected["selected_epoch"] == 1

def _consistency_bindings() -> dict:
    ssl = _dog("binding-ssl")
    split_body = {
        "schema_version": "cvi.nose_region_embedding_consistency_splits.v1",
        "rule": {
            "parent_seen": "old TRAIN yt-bb-dog registered_dog_id",
            "minimum_localized_frames_for_dev_eval": 10,
            "hash": "SHA256('73:'+canonical_UUIDv5)",
            "dev_fraction": 0.30,
            "remaining_fraction": 0.70,
            "low_quality_usage": (
                "SSL_TRAIN_OR_FIXED_DIAGNOSTIC_ONLY_NEVER_IDENTITY_SUPERVISION"
            ),
        },
        "identity_lists": {
            "parent_seen_yt": [],
            "parent_seen_native_ssl_train": [],
            "ssl_train": [ssl],
            "dev": [],
            "eval": [],
        },
        "sample_token_lists": {
            "ssl_train": ["1" * 64],
            "dev": [],
            "eval": [],
            "excluded_no_roi": [],
        },
    }
    splits = {**split_body, "splits_sha256": consistency.content_sha256(split_body)}
    return {
        "parent_embedding": {
            "lineage_file_sha256": "0" * 64,
            "lineage_payload_sha256": "1" * 64,
            "lineage_sha256": "2" * 64,
            "selected_checkpoint_file_sha256": "3" * 64,
            "selected_checkpoint_payload_sha256": "4" * 64,
            "model_state_sha256": "5" * 64,
            "arcface_state_sha256": "6" * 64,
            "selected_epoch": 0,
        },
        "dinov2": {
            "model_directory_sha256": "7" * 64,
            "weight_intake_bundle_file_sha256": "8" * 64,
            "preprocessor_intake_bundle_file_sha256": "9" * 64,
            "parent_dinov2_binding": {},
        },
        "old_crop_manifest": {
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "summary_sha256": "d" * 64,
        },
        "native_v4_bundle": {
            "file_sha256": "e" * 64,
            "payload_sha256": "f" * 64,
            "manifest_sha256": "0" * 64,
            "input_sha256s": {"fixture": "1" * 64},
        },
        "support_cache": {
            "manifest_file_sha256": "2" * 64,
            "manifest_payload_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "transform_sha256": "5" * 64,
            "student_binding": {},
        },
        "splits": splits,
        "old_dev_protocol": {"fixture": True},
        "code_sha256s": {
            path: str(index) * 64
            for index, path in enumerate(consistency._CODE_PATHS)
        },
        "config_sha256": "6" * 64,
        "license": {"license_id": consistency.LICENSE_ID, "usage_lane": "RESEARCH_ONLY"},
    }

def _consistency_config() -> dict:
    return consistency._training_config(
        epochs=3,
        old_batch_size=32,
        native_pair_batch_size=8,
        backbone_lr=5e-7,
        head_lr=1e-4,
        weight_decay=1e-4,
        num_workers=0,
        seed=42,
        mixed_precision=False,
        device_name="cpu",
        parity_thresholds=ParityThresholds(1e-4, 2e-2, 1e-4, 0.99999),
    )

def test_consistency_bindings_accept_only_complete_code_path_generations() -> None:
    current = _consistency_bindings()
    consistency._validate_bindings(current)

    for paths in (
        consistency._PRE_EMBEDDING_CODE_PATHS,
        consistency._LEGACY_CODE_PATHS,
    ):
        historical = copy.deepcopy(current)
        historical["code_sha256s"] = {
            path: str(index) * 64 for index, path in enumerate(paths)
        }
        consistency._validate_bindings(historical)

    mixed = copy.deepcopy(current)
    digest = mixed["code_sha256s"].pop(consistency._CODE_PATHS[0])
    mixed["code_sha256s"][consistency._PRE_EMBEDDING_CODE_PATHS[0]] = digest
    with pytest.raises(ValueError, match="code binding"):
        consistency._validate_bindings(mixed)

def test_consistency_checkpoint_rejects_metadata_and_tensor_tamper(tmp_path: Path) -> None:
    model = NoseEmbeddingModel(_FakeBackbone())
    head = ArcFaceClassificationHead(1)
    bindings = _consistency_bindings()
    config = _consistency_config()
    bindings["config_sha256"] = consistency.content_sha256(config)
    checkpoint = build_consistency_checkpoint(
        model=model,
        arcface=head,
        epoch=0,
        global_step=0,
        selection={"selected_epoch": 0},
        bindings=bindings,
        training_config=config,
        parent_model_state_sha256="a" * 64,
        parent_arcface_state_sha256="b" * 64,
    )
    path = tmp_path / "selected.pt"
    consistency.replace_consistency_checkpoint(path, checkpoint)
    assert load_consistency_checkpoint(path)["epoch"] == 0

    metadata = torch.load(path, weights_only=True)
    metadata["bindings"]["config_sha256"] = "0" * 64
    metadata_path = tmp_path / "metadata.pt"
    torch.save(metadata, metadata_path)
    with pytest.raises(ValueError, match="bindings digest"):
        load_consistency_checkpoint(metadata_path)

    tensors = torch.load(path, weights_only=True)
    tensors["model_state_dict"]["backbone.offset"][0] += 1
    tensor_path = tmp_path / "tensor.pt"
    torch.save(tensors, tensor_path)
    with pytest.raises(ValueError, match="model state digest"):
        load_consistency_checkpoint(tensor_path)

def test_consistency_tiny_static_onnx_ort_parity_and_runtime_id(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = NoseEmbeddingModel(_FakeBackbone()).eval()
    onnx_path = tmp_path / "nose-v2.onnx"
    export_static_onnx(model, onnx_path)
    manifest = consistency.build_runtime_manifest(onnx_path)
    assert manifest.artifact_id == "cvi.nose_region_rgb_embedding.dinov2-small-cls.v3"

    crop = tmp_path / "crop.png"
    Image.new("RGB", (27, 33), color=(60, 100, 150)).save(crop)
    record = {
        "sample_token": "c" * 64,
        "crop_path": "crop.png",
        "crop_sha256": hashlib.sha256(crop.read_bytes()).hexdigest(),
    }
    receipt = consistency.produce_parity_receipt(
        model=model,
        onnx_path=onnx_path,
        runtime_manifest=manifest,
        crop_root=tmp_path,
        crop_record=record,
        crop_manifest_file_sha256="d" * 64,
        source_weights_sha256="e" * 64,
        weight_intake_receipt_sha256="f" * 64,
        preprocessor_intake_receipt_sha256="1" * 64,
        thresholds=ParityThresholds(1e-4, 2e-2, 1e-4, 0.99999),
    )
    assert receipt.model_id == manifest.artifact_id
    assert receipt.decision == "PASS"
    assert receipt.fixtures[0].fixture_kind is ParityFixtureKind.RECEIPT_BOUND_CROP

def test_consistency_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "archive/nose/commands/train_nose_region_consistency.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--parent-lineage-sha256" in completed.stdout
    assert "--native-bundle-sha256" in completed.stdout
    assert "--support-manifest-sha256" in completed.stdout
    assert "--native-pair-batch-size" in completed.stdout
