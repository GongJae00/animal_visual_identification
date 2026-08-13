from __future__ import annotations

import hashlib
import json
import os
import uuid
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import resnet18

import embedding.methods.full_segment.training.artifacts as full128_artifacts
from foundation.protected_io import json_document_bytes
from foundation.provenance import content_sha256
from embedding.methods.full_segment.training.artifacts import (
    default_full128_run_config,
    file_binding,
    validate_embedding_cache,
    validate_full128_run_config,
    validate_variant_run,
    write_embedding_cache,
)
from embedding.methods.full_segment.models.classical import Classical128
from embedding.methods.full_segment.preparation.data import (
    Full128Inventory,
    Full128Sample,
    Full128TorchDataset,
    read_full128_crop,
    read_full128_mask,
)
from embedding.methods.full_segment.training.losses import batch_hard_triplet_loss
from embedding.methods.full_segment.models.model import MaskedGAP128
from embedding.methods.full_segment.training.sampler import DatasetViewBalancedPKSampler
from embedding.learning.full_segment.full128 import (
    _extract_raw_descriptors,
    _fit_population_for_variant,
    _initialize_or_validate_run,
    _prepare_training_runtime,
    _reset_training_seed,
    _validate_variant_training_contract,
)
from workflows.run_full128_training import main as training_workflow


def _sha(character: str) -> str:
    return character * 64


def _sample(index: int, *, role: str = "FIT") -> Full128Sample:
    return Full128Sample(
        sample_id=hashlib.sha256(f"sample-{index}".encode()).hexdigest(),
        identity_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"identity-{index // 2}")),
        dataset_name="dogfacenet224",
        view="body",
        role=role,
        rgb_path=Path(f"/external/rgb-{index}.png"),
        rgb_sha256=_sha("a"),
        mask_path=Path(f"/external/mask-{index}.png"),
        mask_sha256=_sha("b"),
        crop_record_sha256=_sha("c"),
    )


def _materialized_sample(root: Path, index: int) -> Full128Sample:
    rng = np.random.default_rng(index)
    rgb = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[16 + index : 208, 24:200] = 255
    rgb_path = root / f"rgb-{index}.png"
    mask_path = root / f"mask-{index}.png"
    Image.fromarray(rgb, mode="RGB").save(rgb_path, format="PNG")
    Image.fromarray(mask, mode="L").save(mask_path, format="PNG")
    return Full128Sample(
        sample_id=hashlib.sha256(f"sample-{index:04d}".encode()).hexdigest(),
        identity_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"identity-{index // 2}")),
        dataset_name="dogfacenet224",
        view="face",
        role="FIT",
        rgb_path=rgb_path,
        rgb_sha256=hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
        mask_path=mask_path,
        mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        crop_record_sha256=hashlib.sha256(f"crop-{index}".encode()).hexdigest(),
    )


def test_mask_only_reader_matches_full_crop_reader(tmp_path: Path) -> None:
    sample = _materialized_sample(tmp_path, 0)
    _, full_mask = read_full128_crop(sample)

    np.testing.assert_array_equal(read_full128_mask(sample), full_mask)


def test_run_config_fixes_current_admitted_quotas_and_protocol() -> None:
    config = default_full128_run_config()
    assert validate_full128_run_config(config) == config
    assert config["workers"] == 8
    assert config["sampler"]["logical_batch_size"] == 54
    assert config["sampler"]["group_quotas"] == [
        {"dataset_name": "dogfacenet224", "view": "body", "identities": 9},
        {"dataset_name": "yt-bb-dog", "view": "body", "identities": 18},
    ]
    changed = deepcopy(config)
    changed["sampler"]["group_quotas"].insert(
        1, {"dataset_name": "petface-dog", "view": "face", "identities": 9}
    )
    changed["sampler"]["logical_batch_size"] = 72
    with pytest.raises(ValueError, match="DogFaceNet=9 and YT-BB=18"):
        validate_full128_run_config(changed)


def test_real_masked_gap_optimizer_step_uses_exact_triplet_objective() -> None:
    torch.manual_seed(20260811)
    model = MaskedGAP128().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rgb = torch.rand(4, 3, 32, 32)
    mask = torch.ones(4, 1, 32, 32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    before = model.projection.weight.detach().clone()

    loss = batch_hard_triplet_loss(model(rgb, mask), labels, margin=0.2)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(before, model.projection.weight)


def test_deterministic_one_step_has_exact_cpu_parity() -> None:
    def one_step() -> tuple[torch.Tensor, torch.Tensor]:
        _reset_training_seed(20260811)
        model = MaskedGAP128().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        rgb = torch.rand(4, 3, 32, 32)
        mask = torch.ones(4, 1, 32, 32)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        loss = batch_hard_triplet_loss(model(rgb, mask), labels, margin=0.2)
        loss.backward()
        optimizer.step()
        return loss.detach(), model.projection.weight.detach().clone()

    first_loss, first_weights = one_step()
    second_loss, second_weights = one_step()

    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    torch.testing.assert_close(first_weights, second_weights, rtol=0, atol=0)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_compact_dataset_payload_is_exactly_equivalent(tmp_path: Path) -> None:
    sample = _materialized_sample(tmp_path, 0)
    labels = {sample.identity_id: 7}
    public = Full128TorchDataset((sample,), identity_to_label=labels)[0]
    compact = Full128TorchDataset(
        (sample,), identity_to_label=labels, payload_mode="compact"
    )[0]

    assert set(compact) == {"rgb", "mask", "label"}
    assert compact["rgb"].dtype == torch.uint8
    assert compact["mask"].dtype == torch.bool
    assert compact["label"] == public["label"] == 7
    torch.testing.assert_close(
        compact["rgb"].float().div_(255.0), public["rgb"], rtol=0, atol=0
    )
    torch.testing.assert_close(compact["mask"].float(), public["mask"], rtol=0, atol=0)


def test_b0_worker_descriptors_match_previous_tensor_path_exactly(
    tmp_path: Path,
) -> None:
    samples = tuple(_materialized_sample(tmp_path, index) for index in range(4))
    model = Classical128()
    public = Full128TorchDataset(samples)
    expected = np.stack(
        [
            model.raw_descriptor(
                public[index]["rgb"].numpy().transpose(1, 2, 0),
                public[index]["mask"].numpy()[0],
            )
            for index in range(len(samples))
        ]
    )

    actual = _extract_raw_descriptors(model, samples, workers=2)

    np.testing.assert_array_equal(actual, expected)


def test_seed_reset_matches_projection_and_sampler_order() -> None:
    pretrained = resnet18(weights=None).state_dict()
    torch.manual_seed(41)
    scratch = MaskedGAP128()
    torch.manual_seed(41)
    initialized = MaskedGAP128.from_backbone_state_dict_for_testing(pretrained)
    torch.testing.assert_close(
        scratch.projection.weight, initialized.projection.weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        scratch.projection.bias, initialized.projection.bias, rtol=0, atol=0
    )

    identities: list[str] = []
    observations: list[str] = []
    datasets: list[str] = []
    views: list[str] = []
    for dataset, view, count in (
        ("dogfacenet224", "body", 9),
        ("yt-bb-dog", "body", 18),
    ):
        for identity_index in range(count):
            identity = f"{dataset}-{identity_index}"
            for observation_index in range(2):
                identities.append(identity)
                observations.append(f"{identity}-{observation_index}")
                datasets.append(dataset)
                views.append(view)
    first = DatasetViewBalancedPKSampler(
        identities, observations, datasets, views, seed=41
    )
    second = DatasetViewBalancedPKSampler(
        identities, observations, datasets, views, seed=41
    )
    assert list(first) == list(second)


def test_blocked_petface_quota_fails_closed() -> None:
    with pytest.raises(ValueError, match="PetFace is blocked"):
        DatasetViewBalancedPKSampler(
            ["dog", "dog", "other", "other"],
            ["a", "b", "c", "d"],
            ["petface-dog"] * 4,
            ["face"] * 4,
            group_quotas={("petface-dog", "face"): 2},
        )


def test_embedding_pack_is_little_endian_bound_and_tamper_evident(
    tmp_path: Path,
) -> None:
    samples = tuple(
        sorted((_sample(0), _sample(1, role="DEV")), key=lambda item: item.sample_id)
    )
    embeddings = np.zeros((2, 128), dtype=np.float32)
    embeddings[0, 0] = 1
    embeddings[1, 1] = 1
    pack = tmp_path / "embeddings.f32le"
    manifest = write_embedding_cache(pack, samples, embeddings)

    restored = validate_embedding_cache(tmp_path, manifest)
    np.testing.assert_array_equal(restored, embeddings)
    assert pack.stat().st_size == 1024
    assert all(row["byte_size"] == 512 for row in manifest["vectors"])

    payload = bytearray(pack.read_bytes())
    payload[513] ^= 1
    pack.write_bytes(payload)
    with pytest.raises(ValueError, match="pack digest or length"):
        validate_embedding_cache(tmp_path, manifest)


def test_embedding_pack_stream_validation_can_discard_vectors(tmp_path: Path) -> None:
    samples = tuple(
        sorted((_sample(0), _sample(1, role="DEV")), key=lambda item: item.sample_id)
    )
    embeddings = np.zeros((2, 128), dtype=np.float32)
    embeddings[0, 0] = 1
    embeddings[1, 1] = 1
    manifest = write_embedding_cache(tmp_path / "embeddings.f32le", samples, embeddings)

    assert (
        full128_artifacts._stream_validate_embedding_cache(
            tmp_path, manifest, retain_embeddings=False
        )
        is None
    )


def test_variant_validation_binds_cache_manifest_file_content(tmp_path: Path) -> None:
    samples = tuple(
        sorted((_sample(0), _sample(1, role="DEV")), key=lambda item: item.sample_id)
    )
    embeddings = np.zeros((2, 128), dtype=np.float32)
    embeddings[0, 0] = 1
    embeddings[1, 1] = 1
    cache = write_embedding_cache(tmp_path / "embeddings.f32le", samples, embeddings)
    cache_path = tmp_path / "embedding-cache-manifest.json"
    cache_path.write_bytes(json_document_bytes(cache))
    artifacts: dict[str, Any] = {}
    for name in (
        "state",
        "model_manifest",
        "preprocessing_manifest",
        "embedding_manifest",
        "checkpoint_manifest",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        artifacts[name] = {"relative_path": path.name, **file_binding(path)}
    artifacts["embedding_cache_manifest"] = {
        "relative_path": cache_path.name,
        **file_binding(cache_path),
        "manifest": cache,
    }
    payload = {
        "schema_version": "cvi.full128_variant_run.v1",
        "variant_id": "B0",
        "method": "CLASSICAL128",
        "initialization": "FIT_ONLY",
        "bindings": {},
        "fit_population": {},
        "training": {},
        "artifacts": artifacts,
    }
    manifest = {**payload, "variant_run_sha256": content_sha256(payload)}
    assert validate_variant_run(tmp_path, manifest) == manifest

    cache_path.write_bytes(json_document_bytes({"different": True}))
    artifacts["embedding_cache_manifest"].update(file_binding(cache_path))
    changed = {
        key: value for key, value in manifest.items() if key != "variant_run_sha256"
    }
    manifest["variant_run_sha256"] = content_sha256(changed)
    with pytest.raises(ValueError, match="cache manifest binding differs"):
        validate_variant_run(tmp_path, manifest)


def test_embedding_cache_rejects_noncanonical_typed_row_metadata(
    tmp_path: Path,
) -> None:
    samples = tuple(
        sorted((_sample(0), _sample(1, role="DEV")), key=lambda item: item.sample_id)
    )
    embeddings = np.zeros((2, 128), dtype=np.float32)
    embeddings[0, 0] = 1
    embeddings[1, 1] = 1
    manifest = write_embedding_cache(tmp_path / "embeddings.f32le", samples, embeddings)

    invalid_offset = deepcopy(manifest)
    invalid_offset["vectors"][0]["offset_bytes"] = False
    payload = {
        key: value
        for key, value in invalid_offset.items()
        if key != "cache_manifest_sha256"
    }
    invalid_offset["cache_manifest_sha256"] = content_sha256(payload)
    with pytest.raises(TypeError, match="offset must be an integer"):
        validate_embedding_cache(tmp_path, invalid_offset)

    invalid_metadata = deepcopy(manifest)
    invalid_metadata["vectors"][0]["dataset_name"] = " dogfacenet224"
    payload = {
        key: value
        for key, value in invalid_metadata.items()
        if key != "cache_manifest_sha256"
    }
    invalid_metadata["cache_manifest_sha256"] = content_sha256(payload)
    with pytest.raises(ValueError, match="canonical non-empty text"):
        validate_embedding_cache(tmp_path, invalid_metadata)


def test_fit_population_excludes_non_fit_roles() -> None:
    inventory = Full128Inventory(
        assembly_sha256=_sha("1"),
        inventory_bundle_sha256=_sha("2"),
        inventory_sha256=_sha("3"),
        split_manifest_sha256=_sha("4"),
        split_census_sha256=_sha("5"),
        baseline_family_sha256=_sha("6"),
        artifact_root=Path("/external"),
        samples=(_sample(0), _sample(1, role="DEV"), _sample(2, role="CAL")),
    )
    assert inventory.fit_samples == (_sample(0),)


def test_variant_fit_population_filters_singletons_only_for_neural_variants() -> None:
    singleton = replace(
        _sample(4),
        identity_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "singleton")),
    )
    inventory = Full128Inventory(
        assembly_sha256=_sha("1"),
        inventory_bundle_sha256=_sha("2"),
        inventory_sha256=_sha("3"),
        split_manifest_sha256=_sha("4"),
        split_census_sha256=_sha("5"),
        baseline_family_sha256=_sha("6"),
        artifact_root=Path("/external"),
        samples=(_sample(0), _sample(1), singleton, _sample(5, role="DEV")),
    )

    assert _fit_population_for_variant(inventory, "B0") == (
        _sample(0),
        _sample(1),
        singleton,
    )
    for variant in ("B1", "B2"):
        assert _fit_population_for_variant(inventory, variant) == (
            _sample(0),
            _sample(1),
        )


def test_resume_training_contract_rejects_neural_summary_tampering() -> None:
    config = default_full128_run_config()
    config["precision"] = {"device": "cpu", "amp": False, "amp_dtype": "float32"}
    config["epochs"] = 2
    face = tuple(_sample(index) for index in range(18))
    body = tuple(
        replace(
            _sample(index + 100),
            dataset_name="yt-bb-dog",
            view="body",
        )
        for index in range(36)
    )
    fit = face + body
    training = {
        "kind": "BATCH_HARD_EUCLIDEAN_TRIPLET",
        "margin": 0.2,
        "fit_sample_count": 54,
        "fit_identity_count": 27,
        "logical_batch_size": 54,
        "epoch_summaries": [
            {
                "epoch": epoch,
                "batch_count": 1,
                "sample_count": 54,
                "mean_batch_hard_triplet_loss": 0.25,
            }
            for epoch in (1, 2)
        ],
        "selection": "FIXED_LAST_EPOCH",
        "selected_epoch": 2,
        "determinism": {
            "deterministic_algorithms": True,
            "cublas_workspace_config": None,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    _validate_variant_training_contract(
        training, variant="B1", config=config, fit_samples=fit
    )

    tampered = deepcopy(training)
    tampered["selected_epoch"] = 1
    with pytest.raises(ValueError, match="neural training contract differs"):
        _validate_variant_training_contract(
            tampered, variant="B1", config=config, fit_samples=fit
        )


def test_run_root_is_no_overwrite_but_exactly_resumable(tmp_path: Path) -> None:
    manifest_payload = {
        "schema_version": "cvi.full128_training_run.v1",
        "run_config": {},
        "bindings": {},
        "source_closure": {},
        "runtime_versions": {},
    }
    manifest = {
        **manifest_payload,
        "run_manifest_sha256": content_sha256(manifest_payload),
    }
    output = tmp_path / "run"
    assert _initialize_or_validate_run(output, manifest) == output
    assert _initialize_or_validate_run(output, manifest) == output

    changed = deepcopy(manifest)
    changed["run_manifest_sha256"] = _sha("0")
    with pytest.raises(FileExistsError, match="different immutable bindings"):
        _initialize_or_validate_run(output, changed)


@pytest.mark.parametrize(("workers", "validation_workers"), [(8, 8), (0, 1)])
def test_training_workflow_uses_run_config_workers_for_assembly_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    validation_workers: int,
) -> None:
    config = default_full128_run_config()
    config["workers"] = workers
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assembly_path = tmp_path / "assembly.json"
    assembly_path.write_text("{}", encoding="utf-8")
    loaded = Full128Inventory(
        assembly_sha256=_sha("1"),
        inventory_bundle_sha256=_sha("2"),
        inventory_sha256=_sha("3"),
        split_manifest_sha256=_sha("4"),
        split_census_sha256=_sha("5"),
        baseline_family_sha256=_sha("6"),
        artifact_root=tmp_path,
        samples=(_sample(0),),
    )
    observed_workers: list[int] = []

    def fake_load(path: Path, *, validation_workers: int) -> tuple[Any, dict[str, Any]]:
        assert path == assembly_path
        observed_workers.append(validation_workers)
        return loaded, {}

    def fake_train(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["inventory"] is loaded
        assert kwargs["run_config"] == config
        return {"family_complete": False}

    monkeypatch.setattr(
        "workflows.run_full128_training.load_full128_assembly", fake_load
    )
    monkeypatch.setattr(
        "workflows.run_full128_training.run_full128_training", fake_train
    )

    assert (
        training_workflow(
            [
                "--assembly",
                str(assembly_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "output"),
                "--variants",
                "B0",
            ]
        )
        == 0
    )
    assert observed_workers == [validation_workers]


def test_cuda_runtime_is_prepared_once_for_multiple_neural_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = default_full128_run_config()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    initialized_calls = 0

    def is_initialized() -> bool:
        nonlocal initialized_calls
        initialized_calls += 1
        return False

    monkeypatch.setattr(torch.cuda, "is_initialized", is_initialized)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    _prepare_training_runtime(config, ("B1", "B2"))

    assert initialized_calls == 1
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
