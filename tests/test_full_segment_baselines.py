from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision.models import resnet18

from contracts.intake.pretrained_weight_intake import (
    PretrainedWeightIntakeReceipt,
    PretrainedWeightSourceContract,
)
from foundation.provenance import content_sha256
from embedding.methods.full_segment import (
    Classical128,
    ClassicalFitInput,
    DatasetViewBalancedPKSampler,
    FeatureOutputStatisticsHooks,
    MaskedGAP128,
    batch_hard_triplet_loss,
    build_baseline_family_manifest,
    build_checkpoint_manifest,
    build_embedding_manifest,
    build_preprocessing_manifest,
    manifest_sha256,
)

_B2_SOURCE_CONTRACT_PATH = (
    Path(__file__).parents[1]
    / "contracts"
    / "configs"
    / "pretrained-weights"
    / "torchvision-resnet18-imagenet1k-v1-336d36e8.json"
)


def _b2_source() -> PretrainedWeightSourceContract:
    return PretrainedWeightSourceContract.from_dict(
        json.loads(_B2_SOURCE_CONTRACT_PATH.read_text(encoding="utf-8"))
    )


def _b2_receipt(source: PretrainedWeightSourceContract) -> PretrainedWeightIntakeReceipt:
    return PretrainedWeightIntakeReceipt(
        source_contract_sha256=source.contract_sha256,
        weight_sha256=source.expected_sha256,
        weight_bytes=source.expected_file_bytes,
        license_snapshot_sha256=source.license_snapshot_sha256,
        training_description_snapshot_sha256=(
            source.training_description_snapshot_sha256
        ),
        checksum_authority=source.checksum_authority,
        admitted_lane=source.target_lane,
        file_format=source.file_format,
        decision="PASS_UNVERIFIED_SHA256_RESEARCH_ONLY",
    )


def _write_b2_bundle(
    path: Path,
    source: PretrainedWeightSourceContract,
    receipt: PretrainedWeightIntakeReceipt,
    *,
    receipt_sha256: str | None = None,
) -> None:
    tool_provenance = {
        "schema_version": "canine_identity.source_provenance.v2",
        "logical_component": "workflows.audit_pretrained_weight",
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "cvi.pretrained_weight_intake_bundle.v1",
                "source_contract_sha256": source.contract_sha256,
                "source_contract": source.to_dict(),
                "receipt_sha256": receipt_sha256 or receipt.receipt_sha256,
                "receipt": receipt.to_dict(),
                "tool_provenance": tool_provenance,
                "tool_provenance_sha256": content_sha256(tool_provenance),
            }
        ),
        encoding="utf-8",
    )


def _classical_input(seed: int, *, partition: str = "FIT") -> ClassicalFitInput:
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(40, 48, 3), dtype=np.uint8)
    y, x = np.ogrid[:40, :48]
    mask = (x - (23 + seed % 3)) ** 2 + (y - (19 + seed % 5)) ** 2 <= (13 + seed % 4) ** 2
    return ClassicalFitInput(rgb=rgb, mask=mask, partition=partition, sample_id=f"fit-{seed}")


def test_classical128_fit_output_determinism_and_state_roundtrip(tmp_path: Path) -> None:
    fit_inputs = [_classical_input(seed) for seed in range(128)]
    descriptor = Classical128().fit(fit_inputs)
    independently_fitted = Classical128().fit(fit_inputs)
    query = _classical_input(1000)

    first = descriptor.transform(query.rgb, query.mask)
    second = descriptor.transform(query.rgb, query.mask)
    independent = independently_fitted.transform(query.rgb, query.mask)

    assert first.shape == (128,)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, independent)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-6)
    assert descriptor.ablation_metadata["enabled_groups"] == [
        "hog",
        "hsv_histogram",
        "uniform_lbp",
    ]
    assert descriptor.ablation_metadata["output_dimension_semantics"] == (
        "UNINTERPRETED_PCA_COORDINATES"
    )
    state = tmp_path / "classical128-state.npz"
    descriptor.save_state(state)
    second_state = tmp_path / "classical128-state-second.npz"
    descriptor.save_state(second_state)
    assert state.read_bytes() == second_state.read_bytes()
    restored = Classical128.load_state(state)
    np.testing.assert_array_equal(restored.transform(query.rgb, query.mask), first)
    assert restored.fit_sample_ids == descriptor.fit_sample_ids
    with pytest.raises(FileExistsError, match="overwrite"):
        descriptor.save_state(state)


def test_classical128_is_foreground_only_and_fails_closed() -> None:
    sample = _classical_input(7)
    changed = sample.rgb.copy()
    changed[~sample.mask] = 255 - changed[~sample.mask]
    descriptor = Classical128()

    np.testing.assert_array_equal(
        descriptor.raw_descriptor(sample.rgb, sample.mask),
        descriptor.raw_descriptor(changed, sample.mask),
    )
    with pytest.raises(ValueError, match="only FIT"):
        _classical_input(0, partition="DEVELOPMENT")
    with pytest.raises(ValueError, match="at least 128"):
        descriptor.fit([sample])
    with pytest.raises(RuntimeError, match="fitted"):
        descriptor.transform(sample.rgb, sample.mask)
    with pytest.raises(ValueError, match="binary"):
        descriptor.raw_descriptor(sample.rgb, sample.mask.astype(np.float32) * 0.5)


def test_masked_gap_background_invariance_and_statistics() -> None:
    torch.manual_seed(4)
    model = MaskedGAP128().eval()
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 12:53, 14:49] = 1
    foreground = torch.rand(2, 3, 64, 64)
    first_rgb = foreground * mask + torch.rand_like(foreground) * (1 - mask)
    second_rgb = foreground * mask + torch.rand_like(foreground) * (1 - mask)

    with FeatureOutputStatisticsHooks(model) as hooks, torch.inference_mode():
        first = model(first_rgb, mask)
        second = model(second_rgb, mask)
        statistics = hooks.snapshot()

    assert first.shape == (2, 128)
    torch.testing.assert_close(first, second, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(torch.linalg.vector_norm(first, dim=1), torch.ones(2))
    assert len(statistics["feature_channels"]["mean"]) == 512
    assert len(statistics["output_dimensions"]["mean"]) == 128
    assert statistics["output_dimensions"]["axis_interpretation"] == (
        "INDEX_ONLY_NO_SEMANTIC_DIMENSION_CLAIM"
    )
    assert statistics["output_dimensions"]["observation_count_per_index"] == 4


def test_masked_gap_scratch_and_structural_state_have_architecture_parity() -> None:
    scratch = MaskedGAP128()
    structural = MaskedGAP128.from_backbone_state_dict_for_testing(
        resnet18(weights=None).state_dict()
    )

    assert scratch.initialization == "RANDOM_SCRATCH"
    assert structural.initialization == "STRUCTURAL_TEST_ONLY"
    assert structural.initialization_sha256 is None
    assert structural.initialization_source_contract_sha256 is None
    assert structural.initialization_intake_receipt_sha256 is None
    assert {
        name: tuple(value.shape) for name, value in scratch.state_dict().items()
    } == {name: tuple(value.shape) for name, value in structural.state_dict().items()}


def test_random_resnet_state_cannot_claim_production_b2_provenance(tmp_path) -> None:
    source = _b2_source()
    receipt = _b2_receipt(source)
    intake_bundle = tmp_path / "intake.json"
    _write_b2_bundle(intake_bundle, source, receipt)
    random_weights = tmp_path / source.weight_filename
    torch.save(resnet18(weights=None).state_dict(), random_weights)

    with pytest.raises(ValueError, match="byte size|SHA-256"):
        MaskedGAP128.from_supervised_imagenet(
            random_weights,
            intake_bundle_path=intake_bundle,
        )


def test_b2_receipt_content_and_bundle_hash_mismatches_fail(tmp_path) -> None:
    source = _b2_source()
    receipt = _b2_receipt(source)
    checkpoint = tmp_path / source.weight_filename
    checkpoint.write_bytes(b"not reached")

    mismatched_receipt = replace(receipt, weight_sha256="0" * 64)
    content_mismatch = tmp_path / "receipt-content-mismatch.json"
    _write_b2_bundle(content_mismatch, source, mismatched_receipt)
    with pytest.raises(ValueError, match="receipt weight_sha256 differs"):
        MaskedGAP128.from_supervised_imagenet(
            checkpoint,
            intake_bundle_path=content_mismatch,
        )

    hash_mismatch = tmp_path / "receipt-hash-mismatch.json"
    _write_b2_bundle(hash_mismatch, source, receipt, receipt_sha256="0" * 64)
    with pytest.raises(ValueError, match="receipt hash differs"):
        MaskedGAP128.from_supervised_imagenet(
            checkpoint,
            intake_bundle_path=hash_mismatch,
        )


def test_b2_checkpoint_manifest_requires_source_and_receipt_binding() -> None:
    source = _b2_source()
    receipt = _b2_receipt(source)

    preprocessing = build_preprocessing_manifest(
        method="SEGMENT_FULL_MASKED_GAP128_RESNET18"
    )
    embedding = build_embedding_manifest(
        method="SEGMENT_FULL_MASKED_GAP128_RESNET18"
    )
    manifest = build_checkpoint_manifest(
        method="SEGMENT_FULL_MASKED_GAP128_RESNET18",
        checkpoint_sha256="b" * 64,
        preprocessing_manifest=preprocessing,
        embedding_manifest=embedding,
        initialization="SUPERVISED_IMAGENET",
        initialization_sha256=source.expected_sha256,
        initialization_source_contract_sha256=source.contract_sha256,
        initialization_intake_receipt_sha256=receipt.receipt_sha256,
        initialization_usage_lane=receipt.admitted_lane.value,
    )
    assert manifest["schema_version"] == "cvi.full_segment_checkpoint_manifest.v2"
    assert manifest["initialization_sha256"] == source.expected_sha256
    assert manifest["initialization_source_contract_sha256"] == (
        source.contract_sha256
    )
    assert manifest["initialization_intake_receipt_sha256"] == receipt.receipt_sha256
    assert manifest["initialization_usage_lane"] == "RESEARCH_ONLY"

    with pytest.raises(ValueError, match="initialization_source_contract_sha256"):
        build_checkpoint_manifest(
            method="SEGMENT_FULL_MASKED_GAP128_RESNET18",
            checkpoint_sha256="b" * 64,
            preprocessing_manifest=preprocessing,
            embedding_manifest=embedding,
            initialization="SUPERVISED_IMAGENET",
            initialization_sha256=source.expected_sha256,
        )


def _sampler_arrays() -> tuple[list[str], list[str], list[str], list[str]]:
    identities: list[str] = []
    observations: list[str] = []
    datasets: list[str] = []
    views: list[str] = []
    for dataset, view, count in (
        ("yt-bb-dog", "body", 18),
        ("dogfacenet224", "body", 9),
    ):
        for identity_index in range(count):
            identity = f"{dataset}-{identity_index}"
            for observation_index in range(2):
                identities.append(identity)
                observations.append(f"{identity}-observation-{observation_index}")
                datasets.append(dataset)
                views.append(view)
    return identities, observations, datasets, views


def test_pk_sampler_is_deterministic_balanced_and_provenance_preserving() -> None:
    arrays = _sampler_arrays()
    first_sampler = DatasetViewBalancedPKSampler(*arrays, seed=19)
    second_sampler = DatasetViewBalancedPKSampler(*arrays, seed=19)
    first_batch = next(iter(first_sampler))

    assert first_batch == next(iter(second_sampler))
    assert len(first_batch) == 54
    provenance = first_sampler.provenance_for_batch(first_batch)
    group_identity_counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    observations_by_identity: dict[str, set[str]] = defaultdict(set)
    for record in provenance:
        group_identity_counts[(record.dataset_name, record.view)].add(record.identity_id)
        observations_by_identity[record.identity_id].add(record.observation_id)
    assert {group: len(values) for group, values in group_identity_counts.items()} == {
        ("yt-bb-dog", "body"): 18,
        ("dogfacenet224", "body"): 9,
    }
    assert Counter(len(values) for values in observations_by_identity.values()) == {2: 27}


def test_pk_sampler_and_triplet_loss_fail_closed_on_missing_positives() -> None:
    identities, observations, datasets, views = _sampler_arrays()
    del identities[0], observations[0], datasets[0], views[0]
    with pytest.raises(ValueError, match="requires K distinct"):
        DatasetViewBalancedPKSampler(identities, observations, datasets, views)

    embeddings = torch.eye(3, dtype=torch.float32)
    with pytest.raises(ValueError, match="distinct positive"):
        batch_hard_triplet_loss(embeddings, torch.tensor([0, 1, 1]))


def test_batch_hard_triplet_and_manifests_are_fixed_and_content_bound() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels)
    loss.backward()
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="fixed at 0.2"):
        batch_hard_triplet_loss(embeddings.detach(), labels, margin=0.3)

    classical = Classical128(enabled_groups=("hog", "uniform_lbp"))
    preprocessing = build_preprocessing_manifest(method="CLASSICAL128")
    embedding = build_embedding_manifest(
        method="CLASSICAL128", component_metadata=classical.ablation_metadata
    )
    checkpoint = build_checkpoint_manifest(
        method="CLASSICAL128",
        checkpoint_sha256="a" * 64,
        preprocessing_manifest=preprocessing,
        embedding_manifest=embedding,
        initialization="FIT_SCALER_PCA",
        initialization_sha256=None,
        fit_partition="FIT",
    )
    assert checkpoint["preprocessing_manifest_sha256"] == manifest_sha256(preprocessing)
    assert checkpoint["embedding_manifest_sha256"] == manifest_sha256(embedding)
    assert embedding["output_dimension_semantics"] == "UNINTERPRETED_COORDINATE_INDEX"
    with pytest.raises(ValueError, match="bind estimator fitting to FIT"):
        build_checkpoint_manifest(
            method="CLASSICAL128",
            checkpoint_sha256="a" * 64,
            preprocessing_manifest=preprocessing,
            embedding_manifest=embedding,
            initialization="FIT_SCALER_PCA",
            initialization_sha256=None,
            fit_partition="DEVELOPMENT",
        )


def test_b0_b1_b2_family_contract_is_fixed_and_content_bound() -> None:
    family = build_baseline_family_manifest()

    assert family["family_id"] == "FULL128_B0_B1_B2"
    assert family["embedding_dimension"] == 128
    assert family["variants"] == [
        {
            "variant_id": "B0",
            "method": "CLASSICAL128",
            "initialization": "FIT_SCALER_PCA",
        },
        {
            "variant_id": "B1",
            "method": "SEGMENT_FULL_MASKED_GAP128_RESNET18",
            "initialization": "RANDOM_SCRATCH",
        },
        {
            "variant_id": "B2",
            "method": "SEGMENT_FULL_MASKED_GAP128_RESNET18",
            "initialization": "SUPERVISED_IMAGENET",
        },
    ]
    assert manifest_sha256(family) == manifest_sha256(build_baseline_family_manifest())
