"""Executable B0/B1/B2 training and artifact production for Full128."""

from __future__ import annotations

import math
import os
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from artifact_contracts.source_provenance import build_offline_tool_provenance
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from identity_methods.full_segment.artifacts import (
    FAMILY_RUN_SCHEMA,
    VARIANT_RUN_SCHEMA,
    file_binding,
    group_quotas_from_config,
    runtime_versions,
    validate_full128_run_config,
    validate_variant_run,
    write_embedding_cache,
)
from identity_methods.full_segment.classical import (
    Classical128,
    ClassicalDescriptorDataset,
    collate_classical_descriptors,
    initialize_classical_worker,
)
from identity_methods.full_segment.data import (
    Full128Inventory,
    Full128Sample,
    Full128TorchDataset,
)
from identity_methods.full_segment.losses import batch_hard_triplet_loss
from identity_methods.full_segment.manifests import (
    BASELINE_VARIANTS,
    build_baseline_family_manifest,
    build_checkpoint_manifest,
    build_embedding_manifest,
    build_model_manifest,
    build_preprocessing_manifest,
)
from identity_methods.full_segment.model import MaskedGAP128
from identity_methods.full_segment.sampler import DatasetViewBalancedPKSampler

RUN_MANIFEST_SCHEMA = "cvi.full128_training_run.v1"
_VARIANT_IDS = {item[0] for item in BASELINE_VARIANTS}
_PREFETCH_FACTOR = 2


@dataclass(frozen=True, slots=True)
class _ValidatedVariant:
    root: Path
    variant_id: str
    manifest: dict[str, Any]


def run_full128_training(
    *,
    inventory: Full128Inventory,
    run_config: Mapping[str, Any],
    output_dir: Path,
    variants: Sequence[str],
    b2_checkpoint_path: Path | None = None,
    b2_intake_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Produce selected variants resumably without performing evaluation/gallery work."""

    config = validate_full128_run_config(run_config)
    requested = tuple(dict.fromkeys(variants))
    if not requested or any(variant not in _VARIANT_IDS for variant in requested):
        raise ValueError("Full128 variants must be a non-empty subset of B0/B1/B2")
    if "B2" in requested and (
        b2_checkpoint_path is None or b2_intake_bundle_path is None
    ):
        raise ValueError("B2 requires its canonical checkpoint and intake bundle")
    repository = Path(__file__).resolve().parents[1]
    workflow = repository / "workflows" / "run_full128_training.py"
    source_closure = build_offline_tool_provenance(
        workflow, logical_component="workflows.run_full128_training"
    )
    run_manifest = _build_run_manifest(
        inventory=inventory,
        config=config,
        source_closure=source_closure,
        repository=repository,
    )
    root = _initialize_or_validate_run(output_dir, run_manifest)
    neural_variants_to_create = tuple(
        variant
        for variant in requested
        if variant in {"B1", "B2"}
        and not ((root / variant).exists() or (root / variant).is_symlink())
    )
    _prepare_training_runtime(config, neural_variants_to_create)

    statuses: dict[str, str] = {}
    validated_variants: dict[str, _ValidatedVariant] = {}
    for variant in requested:
        target = root / variant
        if target.exists() or target.is_symlink():
            validated_variants[variant] = _validate_existing_variant(
                target, variant, run_manifest, inventory
            )
            statuses[variant] = "VALIDATED_EXISTING"
            continue
        validated_variants[variant] = _produce_variant(
            variant,
            root=root,
            inventory=inventory,
            config=config,
            run_manifest=run_manifest,
            b2_checkpoint_path=b2_checkpoint_path,
            b2_intake_bundle_path=b2_intake_bundle_path,
        )
        statuses[variant] = "CREATED"

    complete = _publish_or_validate_family(
        root, run_manifest, inventory, validated_variants=validated_variants
    )
    return {
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "variant_statuses": statuses,
        "family_complete": complete,
        "output_dir": os.fspath(root),
    }


def _build_run_manifest(
    *,
    inventory: Full128Inventory,
    config: Mapping[str, Any],
    source_closure: Mapping[str, Any],
    repository: Path,
) -> dict[str, Any]:
    family = build_baseline_family_manifest()
    bindings = {
        "assembly_sha256": inventory.assembly_sha256,
        "inventory_bundle_sha256": inventory.inventory_bundle_sha256,
        "inventory_sha256": inventory.inventory_sha256,
        "split_manifest_sha256": inventory.split_manifest_sha256,
        "split_census_sha256": inventory.split_census_sha256,
        "baseline_family_sha256": inventory.baseline_family_sha256,
        "family_manifest_sha256": content_sha256(family),
        "run_config_sha256": content_sha256(config),
        "source_closure_sha256": content_sha256(source_closure),
        "uv_lock": file_binding(repository / "uv.lock"),
    }
    if bindings["baseline_family_sha256"] != bindings["family_manifest_sha256"]:
        raise ValueError("Full128 inventory baseline family differs from current family")
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_config": dict(config),
        "bindings": bindings,
        "source_closure": dict(source_closure),
        "runtime_versions": runtime_versions(),
    }
    return {**payload, "run_manifest_sha256": content_sha256(payload)}


def _initialize_or_validate_run(
    output_dir: Path, run_manifest: Mapping[str, Any]
) -> Path:
    requested = output_dir.absolute()
    parent = requested.parent.resolve(strict=True)
    root = parent / requested.name
    repository = Path(__file__).resolve().parents[1]
    if root == repository or root.is_relative_to(repository):
        raise ValueError("Full128 training output must remain outside the repository")
    if requested.is_symlink():
        raise ValueError("Full128 training output must not be a symlink")
    if root.exists() or root.is_symlink():
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Full128 run output must be a regular directory")
        existing = _read_json(root / "run-manifest.json")
        if existing != run_manifest:
            raise FileExistsError("existing Full128 run has different immutable bindings")
        return root
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=parent))
    os.chmod(staging, 0o700)
    try:
        write_private_json_bundle(((staging / "run-manifest.json", dict(run_manifest)),))
        fsync_directory(staging)
        rename_directory_noreplace(staging, root)
        fsync_directory(parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root


def _produce_variant(
    variant: str,
    *,
    root: Path,
    inventory: Full128Inventory,
    config: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    b2_checkpoint_path: Path | None,
    b2_intake_bundle_path: Path | None,
) -> _ValidatedVariant:
    staging = Path(tempfile.mkdtemp(prefix=f".{variant}.staging-", dir=root))
    os.chmod(staging, 0o700)
    try:
        if variant == "B0":
            result = _produce_b0(staging, inventory, config)
        else:
            result = _produce_neural(
                staging,
                inventory,
                config,
                variant=variant,
                b2_checkpoint_path=b2_checkpoint_path,
                b2_intake_bundle_path=b2_intake_bundle_path,
            )
        variant_manifest = _write_variant_manifests(
            staging,
            inventory=inventory,
            run_manifest=run_manifest,
            variant=variant,
            **result,
        )
        write_private_json_bundle(
            ((staging / "variant-run.json", variant_manifest),)
        )
        validated = _validate_existing_variant(
            staging, variant, run_manifest, inventory
        )
        fsync_directory(staging)
        target = root / variant
        rename_directory_noreplace(staging, target)
        fsync_directory(root)
        return _ValidatedVariant(target, variant, validated.manifest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _produce_b0(
    staging: Path,
    inventory: Full128Inventory,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fit = _fit_population_for_variant(inventory, "B0")
    if len(fit) < 128:
        raise ValueError("B0 requires at least 128 materialized FIT samples")
    model = Classical128()
    raw = _extract_raw_descriptors(model, inventory.samples, workers=config["workers"])
    fit_ids = {sample.sample_id for sample in fit}
    fit_indices = np.asarray(
        [
            index
            for index, sample in enumerate(inventory.samples)
            if sample.sample_id in fit_ids
        ],
        dtype=np.int64,
    )
    non_fit_indices = np.asarray(
        [
            index
            for index, sample in enumerate(inventory.samples)
            if sample.sample_id not in fit_ids
        ],
        dtype=np.int64,
    )
    fit_raw = raw[fit_indices]
    fit_embeddings = model.fit_descriptors(
        fit_raw, sample_ids=[sample.sample_id for sample in fit]
    )
    embeddings = np.empty((len(inventory.samples), 128), dtype=np.float32)
    embeddings[fit_indices] = fit_embeddings
    if non_fit_indices.size:
        embeddings[non_fit_indices] = model.transform_descriptors(raw[non_fit_indices])
    state_path = staging / "classical128-state.npz"
    model.save_state(state_path)
    restored = Classical128.load_state(state_path)
    np.testing.assert_array_equal(restored.transform_descriptors(fit_raw), fit_embeddings)
    return {
        "model": model,
        "state_path": state_path,
        "embeddings": embeddings,
        "initialization_artifacts": None,
        "training": {
            "kind": "FIT_STANDARD_SCALER_AND_FULL_SVD_PCA128",
            "fit_sample_count": len(fit),
            "epoch_summaries": [],
            "selection": "FITTED_ESTIMATOR_STATE",
        },
    }


def _extract_raw_descriptors(
    model: Classical128,
    samples: Sequence[Full128Sample],
    *,
    workers: int,
) -> np.ndarray:
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ClassicalDescriptorDataset(samples, enabled_groups=model.enabled_groups),
        batch_size=64,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
        collate_fn=collate_classical_descriptors,
        worker_init_fn=initialize_classical_worker if workers > 0 else None,
        **({"prefetch_factor": _PREFETCH_FACTOR} if workers > 0 else {}),
    )
    matrix = np.empty((len(samples), model.raw_dimension), dtype=np.float32)
    populated = np.zeros(len(samples), dtype=bool)
    for indices, descriptors in loader:
        if (
            descriptors.shape != (len(indices), model.raw_dimension)
            or np.any(indices < 0)
            or np.any(indices >= len(samples))
            or np.any(populated[indices])
        ):
            raise RuntimeError("B0 descriptor extraction batch differs")
        matrix[indices] = descriptors
        populated[indices] = True
    if not populated.all():
        raise RuntimeError("B0 descriptor extraction coverage differs")
    return matrix


def _produce_neural(
    staging: Path,
    inventory: Full128Inventory,
    config: Mapping[str, Any],
    *,
    variant: str,
    b2_checkpoint_path: Path | None,
    b2_intake_bundle_path: Path | None,
) -> dict[str, Any]:
    precision = config["precision"]
    use_cuda = precision["device"] == "cuda"
    import torch
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader

    _reset_training_seed(config["seed"], use_cuda=use_cuda)
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("Full128 CUDA was requested but is unavailable")
    device = torch.device(precision["device"])
    if variant == "B1":
        model = MaskedGAP128()
    else:
        assert b2_checkpoint_path is not None and b2_intake_bundle_path is not None
        if b2_checkpoint_path.is_symlink() or b2_intake_bundle_path.is_symlink():
            raise ValueError("B2 source artifacts must not be symlinks")
        b2_checkpoint_path = b2_checkpoint_path.resolve(strict=True)
        b2_intake_bundle_path = b2_intake_bundle_path.resolve(strict=True)
        model = MaskedGAP128.from_supervised_imagenet(
            b2_checkpoint_path, intake_bundle_path=b2_intake_bundle_path
        )
    model.to(device)

    fit = _fit_population_for_variant(inventory, variant)
    group_quotas = group_quotas_from_config(config)
    allowed_groups = set(group_quotas)
    observed_groups = {(sample.dataset_name, sample.view) for sample in fit}
    if observed_groups != allowed_groups:
        raise ValueError(
            "Full128 FIT population must exactly match admitted DogFaceNet face and "
            "YT-BB body groups"
        )
    identities = sorted({sample.identity_id for sample in fit})
    identity_to_label = {identity: index for index, identity in enumerate(identities)}
    dataset = Full128TorchDataset(
        fit,
        identity_to_label=identity_to_label,
        payload_mode="compact",
    )
    sampler = DatasetViewBalancedPKSampler(
        [sample.identity_id for sample in fit],
        [sample.sample_id for sample in fit],
        [sample.dataset_name for sample in fit],
        [sample.view for sample in fit],
        group_quotas=group_quotas,
        samples_per_identity=2,
        seed=config["seed"],
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["workers"] > 0,
        generator=torch.Generator().manual_seed(config["seed"]),
        **(
            {"prefetch_factor": _PREFETCH_FACTOR}
            if config["workers"] > 0
            else {}
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    amp_enabled = bool(precision["amp"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    summaries: list[dict[str, Any]] = []
    for epoch in range(config["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        batch_count = 0
        sample_count = 0
        loss_values = torch.empty(len(loader), dtype=torch.float32, device=device)
        batch_sizes = np.empty(len(loader), dtype=np.int64)
        for batch in loader:
            rgb, mask = _float_model_inputs(batch, device=device)
            labels = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                embeddings = model(rgb, mask)
                loss = batch_hard_triplet_loss(embeddings, labels, margin=0.2)
            try:
                scaler.scale(loss).backward()
            except RuntimeError as exc:
                message = str(exc).lower()
                if use_cuda and ("determin" in message or "cdist" in message):
                    raise RuntimeError(
                        "Full128 deterministic CUDA cdist backward is unsupported by "
                        "the active torch/CUDA runtime"
                    ) from exc
                raise
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(labels.shape[0])
            loss_values[batch_count] = loss.detach().float()
            batch_sizes[batch_count] = batch_size
            sample_count += batch_size
            batch_count += 1
        if batch_count == 0:
            raise RuntimeError("Full128 training epoch produced no batches")
        epoch_losses = loss_values[:batch_count].cpu().tolist()
        loss_sum = sum(
            value * int(batch_size)
            for value, batch_size in zip(
                epoch_losses, batch_sizes[:batch_count], strict=True
            )
        )
        summaries.append(
            {
                "epoch": epoch + 1,
                "batch_count": batch_count,
                "sample_count": sample_count,
                "mean_batch_hard_triplet_loss": loss_sum / sample_count,
            }
        )

    state_path = staging / "model.safetensors"
    cpu_state = {
        key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()
    }
    save_file(
        cpu_state,
        str(state_path),
        metadata={
            "schema_version": "cvi.full128_masked_gap_state.v1",
            "variant_id": variant,
            "run_config_sha256": content_sha256(config),
            "selection": "FIXED_LAST_EPOCH",
        },
    )
    restored = MaskedGAP128()
    restored.load_state_dict(load_file(str(state_path), device="cpu"), strict=True)
    for key, expected in cpu_state.items():
        torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)
    embeddings = _extract_neural_embeddings(
        model, inventory.samples, device=device, config=config
    )
    return {
        "model": model,
        "state_path": state_path,
        "embeddings": embeddings,
        "initialization_artifacts": (
            None
            if variant == "B1"
            else {
                "checkpoint": {
                    "absolute_path": os.fspath(b2_checkpoint_path),
                    **file_binding(b2_checkpoint_path),
                },
                "intake_bundle": {
                    "absolute_path": os.fspath(b2_intake_bundle_path),
                    **file_binding(b2_intake_bundle_path),
                },
            }
        ),
        "training": {
            "kind": "BATCH_HARD_EUCLIDEAN_TRIPLET",
            "margin": 0.2,
            "fit_sample_count": len(fit),
            "fit_identity_count": len(identities),
            "logical_batch_size": sampler.identities_per_batch * 2,
            "epoch_summaries": summaries,
            "selection": "FIXED_LAST_EPOCH",
            "selected_epoch": config["epochs"],
            "determinism": {
                "deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8" if use_cuda else None,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
            },
        },
    }


def _extract_neural_embeddings(
    model: Any,
    samples: Sequence[Full128Sample],
    *,
    device: Any,
    config: Mapping[str, Any],
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        Full128TorchDataset(samples, payload_mode="compact"),
        batch_size=config["sampler"]["logical_batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["workers"] > 0,
        **(
            {"prefetch_factor": _PREFETCH_FACTOR}
            if config["workers"] > 0
            else {}
        ),
    )
    model.eval()
    matrix = np.empty((len(samples), 128), dtype=np.float32)
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            rgb, mask = _float_model_inputs(batch, device=device)
            embedding = model(rgb, mask).float().cpu().numpy()
            stop = offset + len(embedding)
            if embedding.shape[1:] != (128,) or stop > len(samples):
                raise RuntimeError("Full128 embedding extraction batch differs")
            matrix[offset:stop] = embedding
            offset = stop
    if offset != len(samples):
        raise RuntimeError("Full128 embedding extraction coverage differs")
    return matrix


def _write_variant_manifests(
    staging: Path,
    *,
    inventory: Full128Inventory,
    run_manifest: Mapping[str, Any],
    variant: str,
    model: Any,
    state_path: Path,
    embeddings: np.ndarray,
    initialization_artifacts: Mapping[str, Any] | None,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    variant_contract = {item[0]: item for item in BASELINE_VARIANTS}[variant]
    _, method, initialization = variant_contract
    model_manifest = build_model_manifest(method=method)
    preprocessing = build_preprocessing_manifest(method=method)
    embedding = build_embedding_manifest(
        method=method,
        component_metadata=model.ablation_metadata if variant == "B0" else None,
    )
    state = file_binding(state_path)
    checkpoint = build_checkpoint_manifest(
        method=method,
        checkpoint_sha256=state["sha256"],
        preprocessing_manifest=preprocessing,
        embedding_manifest=embedding,
        initialization=initialization,
        initialization_sha256=getattr(model, "initialization_sha256", None),
        initialization_source_contract_sha256=getattr(
            model, "initialization_source_contract_sha256", None
        ),
        initialization_intake_receipt_sha256=getattr(
            model, "initialization_intake_receipt_sha256", None
        ),
        initialization_usage_lane=getattr(model, "initialization_usage_lane", None),
        fit_partition="FIT" if variant == "B0" else None,
    )
    json_outputs = (
        (staging / "model-manifest.json", model_manifest),
        (staging / "preprocessing-manifest.json", preprocessing),
        (staging / "embedding-manifest.json", embedding),
        (staging / "checkpoint-manifest.json", checkpoint),
    )
    write_private_json_bundle(json_outputs)
    cache = write_embedding_cache(
        staging / "embeddings.f32le", inventory.samples, embeddings
    )
    write_private_json_bundle(((staging / "embedding-cache-manifest.json", cache),))
    fit_samples = _fit_population_for_variant(inventory, variant)
    fit_population_payload = {
        "partition": "FIT",
        "sample_count": len(fit_samples),
        "identity_count": len({sample.identity_id for sample in fit_samples}),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "identity_id": sample.identity_id,
                "dataset_name": sample.dataset_name,
                "view": sample.view,
                "crop_record_sha256": sample.crop_record_sha256,
            }
            for sample in fit_samples
        ],
    }
    fit_population = {
        **fit_population_payload,
        "fit_population_sha256": content_sha256(fit_population_payload),
    }
    artifacts = {
        "state": {"relative_path": state_path.name, **state},
        "model_manifest": _artifact_reference(staging / "model-manifest.json"),
        "preprocessing_manifest": _artifact_reference(
            staging / "preprocessing-manifest.json"
        ),
        "embedding_manifest": _artifact_reference(staging / "embedding-manifest.json"),
        "checkpoint_manifest": _artifact_reference(
            staging / "checkpoint-manifest.json"
        ),
        "embedding_cache_manifest": {
            **_artifact_reference(staging / "embedding-cache-manifest.json"),
            "manifest": cache,
        },
    }
    bindings: dict[str, Any] = {
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        **run_manifest["bindings"],
    }
    if variant == "B2":
        bindings["b2_initialization"] = {
            "weight_sha256": model.initialization_sha256,
            "source_contract_sha256": model.initialization_source_contract_sha256,
            "intake_receipt_sha256": model.initialization_intake_receipt_sha256,
            "usage_lane": model.initialization_usage_lane,
            "artifacts": dict(initialization_artifacts or {}),
        }
    payload = {
        "schema_version": VARIANT_RUN_SCHEMA,
        "variant_id": variant,
        "method": method,
        "initialization": initialization,
        "bindings": bindings,
        "fit_population": fit_population,
        "training": dict(training),
        "artifacts": artifacts,
    }
    return {**payload, "variant_run_sha256": content_sha256(payload)}


def _validate_existing_variant(
    root: Path,
    variant: str,
    run_manifest: Mapping[str, Any],
    inventory: Full128Inventory,
) -> _ValidatedVariant:
    import torch
    from safetensors.torch import load_file

    manifest = validate_variant_run(root, _read_json(root / "variant-run.json"))
    variant_contract = {item[0]: item for item in BASELINE_VARIANTS}[variant]
    _, method, initialization = variant_contract
    expected_bindings: dict[str, Any] = {
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        **run_manifest["bindings"],
    }
    observed_bindings = manifest["bindings"]
    if (
        manifest["variant_id"] != variant
        or manifest["method"] != method
        or manifest["initialization"] != initialization
    ):
        raise ValueError("Full128 resumed variant binding differs")
    if variant == "B2":
        if not isinstance(observed_bindings, Mapping) or set(observed_bindings) != {
            *expected_bindings,
            "b2_initialization",
        }:
            raise ValueError("Full128 resumed B2 binding fields differ")
    elif observed_bindings != expected_bindings:
        raise ValueError("Full128 resumed variant bindings differ")
    for name, expected_value in expected_bindings.items():
        if observed_bindings[name] != expected_value:
            raise ValueError(f"Full128 resumed variant {name} differs")
    expected_cache_rows = [
        {
            "sample_id": sample.sample_id,
            "identity_id": sample.identity_id,
            "dataset_name": sample.dataset_name,
            "view": sample.view,
            "role": sample.role,
            "crop_record_sha256": sample.crop_record_sha256,
        }
        for sample in inventory.samples
    ]
    cache_rows = [
        {
            key: row[key]
            for key in (
                "sample_id",
                "identity_id",
                "dataset_name",
                "view",
                "role",
                "crop_record_sha256",
            )
        }
        for row in manifest["artifacts"]["embedding_cache_manifest"]["manifest"][
            "vectors"
        ]
    ]
    if cache_rows != expected_cache_rows:
        raise ValueError("Full128 resumed embedding cache population differs")
    state_path = root / manifest["artifacts"]["state"]["relative_path"]
    if variant == "B0":
        restored = Classical128.load_state(state_path)
        expected_fit_ids = [
            sample.sample_id
            for sample in _fit_population_for_variant(inventory, variant)
        ]
        if list(restored.fit_sample_ids) != expected_fit_ids:
            raise ValueError("Full128 resumed B0 FIT population differs")
        initialization_values = {
            "initialization_sha256": None,
            "initialization_source_contract_sha256": None,
            "initialization_intake_receipt_sha256": None,
            "initialization_usage_lane": None,
        }
    else:
        state = load_file(str(state_path), device="cpu")
        restored = MaskedGAP128()
        restored.load_state_dict(state, strict=True)
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Full128 resumed checkpoint contains non-finite tensors")
        if variant == "B2":
            b2 = manifest["bindings"].get("b2_initialization")
            if not isinstance(b2, Mapping) or set(b2) != {
                "weight_sha256",
                "source_contract_sha256",
                "intake_receipt_sha256",
                "usage_lane",
                "artifacts",
            }:
                raise ValueError("Full128 resumed B2 initialization binding differs")
            source_artifacts = b2["artifacts"]
            if not isinstance(source_artifacts, Mapping) or set(source_artifacts) != {
                "checkpoint",
                "intake_bundle",
            }:
                raise ValueError("Full128 resumed B2 source artifacts differ")
            for artifact in source_artifacts.values():
                if not isinstance(artifact, Mapping) or set(artifact) != {
                    "absolute_path",
                    "sha256",
                    "byte_size",
                }:
                    raise ValueError("Full128 resumed B2 source artifact fields differ")
                path = Path(artifact["absolute_path"])
                if file_binding(path) != {
                    "sha256": artifact["sha256"],
                    "byte_size": artifact["byte_size"],
                }:
                    raise ValueError("Full128 resumed B2 source artifact differs")
            source_model = MaskedGAP128.from_supervised_imagenet(
                Path(source_artifacts["checkpoint"]["absolute_path"]),
                intake_bundle_path=Path(
                    source_artifacts["intake_bundle"]["absolute_path"]
                ),
            )
            if (
                source_model.initialization_sha256 != b2["weight_sha256"]
                or source_model.initialization_source_contract_sha256
                != b2["source_contract_sha256"]
                or source_model.initialization_intake_receipt_sha256
                != b2["intake_receipt_sha256"]
                or source_model.initialization_usage_lane != b2["usage_lane"]
            ):
                raise ValueError("Full128 resumed B2 provenance differs")
            initialization_values = {
                "initialization_sha256": b2["weight_sha256"],
                "initialization_source_contract_sha256": b2[
                    "source_contract_sha256"
                ],
                "initialization_intake_receipt_sha256": b2[
                    "intake_receipt_sha256"
                ],
                "initialization_usage_lane": b2["usage_lane"],
            }
        else:
            initialization_values = {
                "initialization_sha256": None,
                "initialization_source_contract_sha256": None,
                "initialization_intake_receipt_sha256": None,
                "initialization_usage_lane": None,
            }
    fit_samples = _fit_population_for_variant(inventory, variant)
    expected_fit_payload = {
        "partition": "FIT",
        "sample_count": len(fit_samples),
        "identity_count": len({sample.identity_id for sample in fit_samples}),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "identity_id": sample.identity_id,
                "dataset_name": sample.dataset_name,
                "view": sample.view,
                "crop_record_sha256": sample.crop_record_sha256,
            }
            for sample in fit_samples
        ],
    }
    expected_fit = {
        **expected_fit_payload,
        "fit_population_sha256": content_sha256(expected_fit_payload),
    }
    if manifest["fit_population"] != expected_fit:
        raise ValueError("Full128 resumed FIT population manifest differs")
    _validate_variant_training_contract(
        manifest["training"],
        variant=variant,
        config=run_manifest["run_config"],
        fit_samples=fit_samples,
    )

    expected_model = build_model_manifest(method=method)
    expected_preprocessing = build_preprocessing_manifest(method=method)
    expected_embedding = build_embedding_manifest(
        method=method,
        component_metadata=restored.ablation_metadata if variant == "B0" else None,
    )
    artifact_paths = manifest["artifacts"]
    if _read_json(root / artifact_paths["model_manifest"]["relative_path"]) != expected_model:
        raise ValueError("Full128 resumed model manifest differs")
    if _read_json(
        root / artifact_paths["preprocessing_manifest"]["relative_path"]
    ) != expected_preprocessing:
        raise ValueError("Full128 resumed preprocessing manifest differs")
    if _read_json(
        root / artifact_paths["embedding_manifest"]["relative_path"]
    ) != expected_embedding:
        raise ValueError("Full128 resumed embedding manifest differs")
    expected_checkpoint = build_checkpoint_manifest(
        method=method,
        checkpoint_sha256=artifact_paths["state"]["sha256"],
        preprocessing_manifest=expected_preprocessing,
        embedding_manifest=expected_embedding,
        initialization=initialization,
        fit_partition="FIT" if variant == "B0" else None,
        **initialization_values,
    )
    if _read_json(
        root / artifact_paths["checkpoint_manifest"]["relative_path"]
    ) != expected_checkpoint:
        raise ValueError("Full128 resumed checkpoint manifest differs")
    return _ValidatedVariant(root, variant, manifest)


def _publish_or_validate_family(
    root: Path,
    run_manifest: Mapping[str, Any],
    inventory: Full128Inventory,
    *,
    validated_variants: Mapping[str, _ValidatedVariant],
) -> bool:
    variants: list[dict[str, Any]] = []
    for variant in ("B0", "B1", "B2"):
        target = root / variant
        if not target.is_dir() or target.is_symlink():
            return False
        validated = validated_variants.get(variant)
        if validated is None:
            validated = _validate_existing_variant(
                target, variant, run_manifest, inventory
            )
        elif validated.root != target or validated.variant_id != variant:
            raise RuntimeError("Full128 retained variant validation evidence differs")
        manifest = validated.manifest
        variants.append(
            {
                "variant_id": variant,
                "variant_run_sha256": manifest["variant_run_sha256"],
            }
        )
    payload = {
        "schema_version": FAMILY_RUN_SCHEMA,
        "family_id": "FULL128_B0_B1_B2",
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "run_config_sha256": run_manifest["bindings"]["run_config_sha256"],
        "family_manifest_sha256": run_manifest["bindings"]["family_manifest_sha256"],
        "variants": variants,
        "status": "COMPLETE_EXACT_THREE_VARIANT_FAMILY",
    }
    family = {**payload, "family_run_sha256": content_sha256(payload)}
    path = root / "family-run.json"
    if path.exists() or path.is_symlink():
        if _read_json(path) != family:
            raise ValueError("Full128 family completion manifest differs")
    else:
        write_private_json_bundle(((path, family),))
        fsync_directory(root)
    return True


def _artifact_reference(path: Path) -> dict[str, Any]:
    return {"relative_path": path.name, **file_binding(path)}


def _read_json(path: Path) -> dict[str, Any]:
    return read_strict_json_document(path, maximum_bytes=1_073_741_824).payload


def _fit_population_for_variant(
    inventory: Full128Inventory, variant: str
) -> tuple[Full128Sample, ...]:
    """Return the exact estimator-fitting population for one baseline variant."""

    if variant not in _VARIANT_IDS:
        raise ValueError("Full128 FIT population variant differs")
    fit = inventory.fit_samples
    if variant == "B0":
        return fit
    counts = Counter(sample.identity_id for sample in fit)
    return tuple(sample for sample in fit if counts[sample.identity_id] >= 2)


def _validate_variant_training_contract(
    value: object,
    *,
    variant: str,
    config: Mapping[str, Any],
    fit_samples: Sequence[Full128Sample],
) -> None:
    if variant == "B0":
        expected = {
            "kind": "FIT_STANDARD_SCALER_AND_FULL_SVD_PCA128",
            "fit_sample_count": len(fit_samples),
            "epoch_summaries": [],
            "selection": "FITTED_ESTIMATOR_STATE",
        }
        if value != expected:
            raise ValueError("Full128 resumed B0 training contract differs")
        return
    if not isinstance(value, Mapping) or set(value) != {
        "kind",
        "margin",
        "fit_sample_count",
        "fit_identity_count",
        "logical_batch_size",
        "epoch_summaries",
        "selection",
        "selected_epoch",
        "determinism",
    }:
        raise ValueError("Full128 resumed neural training fields differ")
    quotas = group_quotas_from_config(config)
    identities_by_group: dict[tuple[str, str], set[str]] = {
        group: set() for group in quotas
    }
    for sample in fit_samples:
        try:
            identities_by_group[(sample.dataset_name, sample.view)].add(
                sample.identity_id
            )
        except KeyError:
            raise ValueError("Full128 resumed neural FIT group differs") from None
    batch_count = max(
        math.ceil(len(identities_by_group[group]) / quota)
        for group, quota in quotas.items()
    )
    logical_batch_size = config["sampler"]["logical_batch_size"]
    use_cuda = config["precision"]["device"] == "cuda"
    expected_static = {
        "kind": "BATCH_HARD_EUCLIDEAN_TRIPLET",
        "margin": 0.2,
        "fit_sample_count": len(fit_samples),
        "fit_identity_count": len({sample.identity_id for sample in fit_samples}),
        "logical_batch_size": logical_batch_size,
        "selection": "FIXED_LAST_EPOCH",
        "selected_epoch": config["epochs"],
        "determinism": {
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8" if use_cuda else None,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    if any(value[name] != expected for name, expected in expected_static.items()):
        raise ValueError("Full128 resumed neural training contract differs")
    summaries = value["epoch_summaries"]
    if not isinstance(summaries, list) or len(summaries) != config["epochs"]:
        raise ValueError("Full128 resumed epoch summary count differs")
    for epoch, summary in enumerate(summaries, start=1):
        if not isinstance(summary, Mapping) or set(summary) != {
            "epoch",
            "batch_count",
            "sample_count",
            "mean_batch_hard_triplet_loss",
        }:
            raise ValueError("Full128 resumed epoch summary fields differ")
        loss = summary["mean_batch_hard_triplet_loss"]
        if (
            summary["epoch"] != epoch
            or summary["batch_count"] != batch_count
            or summary["sample_count"] != batch_count * logical_batch_size
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not np.isfinite(loss)
            or loss < 0
        ):
            raise ValueError("Full128 resumed epoch summary content differs")


def _float_model_inputs(
    batch: Mapping[str, Any], *, device: Any
) -> tuple[Any, Any]:
    import torch

    rgb = batch["rgb"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    rgb.div_(255.0)
    mask = batch["mask"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    return rgb, mask


def _reset_training_seed(seed: int, *, use_cuda: bool = False) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def _prepare_training_runtime(
    config: Mapping[str, Any], neural_variants_to_create: Sequence[str]
) -> None:
    """Configure deterministic CUDA once before any neural variant creates a context."""

    if not neural_variants_to_create or config["precision"]["device"] != "cuda":
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "Full128 deterministic CUDA must be configured before CUDA context creation"
        )
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Full128 CUDA was requested but is unavailable")


__all__ = ["RUN_MANIFEST_SCHEMA", "run_full128_training"]
