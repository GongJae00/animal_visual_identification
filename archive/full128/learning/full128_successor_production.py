"""Real-data production orchestration for the Full128 successor family."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from shared.foundation.protected_io import (
    json_document_bytes,
    read_strict_json_document,
    write_private_json_bundle,
)
from shared.foundation.protected_publication import fsync_directory, rename_directory_noreplace
from shared.foundation.provenance import content_sha256
from archive.full128.methods.training.artifacts import file_binding
from archive.full128.methods.models.classical import Classical128
from archive.full128.methods.preparation.data import Full128Sample, Full128TorchDataset
from archive.full128.methods.face_visible import (
    validate_face_visible_successor_inventory_bundle,
)
from archive.full128.methods.models.successor_models import (
    ClassicalFV128,
    Dinov2OccupancyProbe128,
    IdentityBlindResidualTokenAdapter128,
    SpatialScorer128,
    build_b1_fv,
    build_b2_fv,
    dinov2_contract_bindings,
)
from archive.full128.learning.full128 import _extract_raw_descriptors
from archive.full128.learning.full128_successors import (
    SUCCESSOR_SSL_OBJECTIVE,
    load_successor_checkpoint,
    make_identity_blind_views,
    reset_successor_seed,
    train_identity_blind_fixed_steps,
    train_supervised_fixed_steps,
    train_supervised_no_update_fixed_steps,
    write_successor_checkpoint,
)

PRODUCTION_CONFIG_SCHEMA = "cvi.full128_successor_production_config.v1"
PRODUCTION_RUN_SCHEMA = "cvi.full128_successor_production_run.v1"
PRODUCTION_CANDIDATE_SCHEMA = "cvi.full128_successor_candidate_run.v1"
PRODUCTION_FAMILY_SCHEMA = "cvi.full128_successor_production_family.v1"
PRODUCTION_TOKEN_CACHE_SCHEMA = "cvi.full128_successor_dinov2_cache.v2"
PRODUCTION_CANDIDATES = (
    "B0-FV",
    "B1-FV",
    "B2-FV",
    "B3",
    "B4-U0",
    "B4-U1",
    "B5-UNIFORM",
    "B5-CHANNEL",
    "B5-SPATIAL",
)
B5_PARENTS = ("B3", "B4-U0", "B4-U1")
_DINO_CANDIDATES = set(PRODUCTION_CANDIDATES[3:])
_B5_CANDIDATES = set(PRODUCTION_CANDIDATES[6:])
_CONFIG_FIELDS = {
    "schema_version",
    "seed",
    "supervised_steps",
    "ssl_steps",
    "optimizer",
    "precision",
    "workers",
    "triplet_margin",
    "ssl_objective",
    "identities_per_batch",
    "samples_per_identity",
    "extraction_batch_size",
}


def default_production_config() -> dict[str, Any]:
    """Return fixed-step real-run defaults; tests must supply bounded configs."""

    return {
        "schema_version": PRODUCTION_CONFIG_SCHEMA,
        "seed": 20260811,
        "supervised_steps": 2_000,
        "ssl_steps": 2_000,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
        "precision": {"device": "cuda", "amp": True, "amp_dtype": "float16"},
        "workers": 8,
        "triplet_margin": 0.2,
        "ssl_objective": SUCCESSOR_SSL_OBJECTIVE,
        "identities_per_batch": 16,
        "samples_per_identity": 2,
        "extraction_batch_size": 64,
    }


def validate_production_config(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CONFIG_FIELDS:
        raise ValueError("successor production config fields differ")
    config = dict(value)
    if config["schema_version"] != PRODUCTION_CONFIG_SCHEMA:
        raise ValueError("successor production config schema differs")
    for name in (
        "seed",
        "supervised_steps",
        "ssl_steps",
        "workers",
        "identities_per_batch",
        "samples_per_identity",
        "extraction_batch_size",
    ):
        item = config[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"successor production {name} must be an integer")
    if (
        config["seed"] < 0
        or config["supervised_steps"] <= 0
        or config["ssl_steps"] <= 0
        or config["workers"] < 0
        or config["identities_per_batch"] < 2
        or config["samples_per_identity"] != 2
        or config["extraction_batch_size"] <= 0
    ):
        raise ValueError("successor production integer range differs")
    optimizer = config["optimizer"]
    if not isinstance(optimizer, Mapping) or set(optimizer) != {
        "name",
        "learning_rate",
        "weight_decay",
    }:
        raise ValueError("successor production optimizer fields differ")
    if optimizer["name"] != "AdamW":
        raise ValueError("successor production optimizer must be AdamW")
    if (
        any(
            isinstance(optimizer[name], bool)
            or not isinstance(optimizer[name], (int, float))
            or not np.isfinite(optimizer[name])
            for name in ("learning_rate", "weight_decay")
        )
        or optimizer["learning_rate"] <= 0
        or optimizer["weight_decay"] < 0
    ):
        raise ValueError("successor production optimizer values differ")
    precision = config["precision"]
    if not isinstance(precision, Mapping) or set(precision) != {
        "device",
        "amp",
        "amp_dtype",
    }:
        raise ValueError("successor production precision fields differ")
    if precision["device"] not in {"cpu", "cuda"} or not isinstance(
        precision["amp"], bool
    ):
        raise ValueError("successor production precision values differ")
    expected_dtype = "float16" if precision["amp"] else "float32"
    if precision["amp_dtype"] != expected_dtype or (
        precision["device"] == "cpu" and precision["amp"]
    ):
        raise ValueError("successor production AMP contract differs")
    if config["triplet_margin"] != 0.2:
        raise ValueError("successor production triplet margin is fixed at 0.2")
    if config["ssl_objective"] != SUCCESSOR_SSL_OBJECTIVE:
        raise ValueError("successor production SSL objective differs")
    return config


def prepare_production_runtime(config: Mapping[str, Any]) -> None:
    """Configure deterministic CUDA before any production model creates a context."""

    validated = validate_production_config(config)
    if validated["precision"]["device"] != "cuda":
        return
    expected = ":4096:8"
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in {None, expected}:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with successor determinism"
        )
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "successor CUDA runtime must be configured before initialization"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("successor production CUDA was requested but unavailable")


def restore_successor_trace_context(
    *,
    run_directory: Path,
    dinov2_backbone: nn.Module,
    dinov2_contract: Any,
) -> dict[str, Any]:
    """Restore the production B3/B5-SPATIAL path and its bound token cache."""

    root = run_directory.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("successor trace run must be a regular directory")
    run_manifest = _read_json(root / "run-manifest.json")
    run_payload = {
        key: value
        for key, value in run_manifest.items()
        if key != "run_manifest_sha256"
    }
    if run_manifest.get("schema_version") != PRODUCTION_RUN_SCHEMA or run_manifest.get(
        "run_manifest_sha256"
    ) != content_sha256(run_payload):
        raise ValueError("successor trace run manifest binding differs")
    config = validate_production_config(run_manifest.get("config"))
    dino_bindings = dinov2_contract_bindings(dinov2_contract)
    if run_manifest.get("dinov2") != dino_bindings:
        raise ValueError("successor trace DINOv2 contract differs from production")
    if run_manifest.get("b5_parent_id") != "B3" or not {
        "B3",
        "B5-SPATIAL",
    }.issubset(run_manifest.get("candidates", ())):
        raise ValueError("successor trace requires the B3-parent B5-SPATIAL run")

    family = _read_json(root / "family-run.json")
    family_payload = {
        key: value for key, value in family.items() if key != "family_run_sha256"
    }
    if (
        family.get("schema_version") != PRODUCTION_FAMILY_SCHEMA
        or family.get("run_manifest_sha256") != run_manifest["run_manifest_sha256"]
        or family.get("family_run_sha256") != content_sha256(family_payload)
    ):
        raise ValueError("successor trace family manifest binding differs")
    family_candidates = {
        row["candidate_id"]: row
        for row in family.get("candidates", ())
        if isinstance(row, Mapping)
    }
    if not {"B3", "B5-SPATIAL"}.issubset(family_candidates):
        raise ValueError("successor trace family omits B3 or B5-SPATIAL")

    cache_manifest = _read_json(root / "dinov2-token-cache" / "cache-manifest.json")
    population_tokens = cache_manifest.get("sample_tokens")
    expected_cache_bindings = {
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "population_tokens_sha256": content_sha256(population_tokens),
        **dino_bindings,
    }
    token_cache = _open_token_cache(
        root / "dinov2-token-cache", bindings=expected_cache_bindings
    )

    models: dict[str, Any] = {}
    candidate_runs: dict[str, dict[str, Any]] = {}
    cache_descriptors: dict[str, dict[str, Any]] = {}
    b2_sources = run_manifest["b2_sources"]
    for candidate_id in ("B3", "B5-SPATIAL"):
        target = root / candidate_id
        model = _restore_candidate_model(
            candidate_id,
            target=target,
            config=config,
            bindings=_candidate_bindings(root, run_manifest, candidate_id, "B3"),
            b2_checkpoint_path=Path(b2_sources["checkpoint"]["absolute_path"]),
            b2_intake_bundle_path=Path(b2_sources["intake_bundle"]["absolute_path"]),
            dinov2_backbone=dinov2_backbone,
            models=models,
            b5_parent_id="B3",
        )
        models[candidate_id] = model
        candidate_run = _read_json(target / "candidate-run.json")
        descriptor = _read_json(target / "evaluation-cache-descriptor.json")
        descriptor_payload = {
            key: value
            for key, value in descriptor.items()
            if key != "cache_descriptor_sha256"
        }
        family_row = family_candidates[candidate_id]
        if (
            family_row.get("candidate_run_sha256")
            != candidate_run["candidate_run_sha256"]
            or family_row.get("cache_descriptor_sha256")
            != descriptor.get("cache_descriptor_sha256")
            or descriptor.get("cache_descriptor_sha256")
            != content_sha256(descriptor_payload)
            or descriptor.get("checkpoint_sha256") != candidate_run["checkpoint_sha256"]
        ):
            raise ValueError(
                f"successor trace {candidate_id} descriptor binding differs"
            )
        candidate_runs[candidate_id] = candidate_run
        cache_descriptors[candidate_id] = descriptor
    return {
        "root": root,
        "run_manifest": run_manifest,
        "family": family,
        "config": config,
        "token_cache": token_cache,
        "models": models,
        "candidate_runs": candidate_runs,
        "cache_descriptors": cache_descriptors,
    }


def run_successor_production(
    *,
    successor_inventory_bundle: Mapping[str, Any],
    required_evaluation_tokens: Sequence[str],
    evaluation_panel_sha256: str,
    output_dir: Path,
    candidates: Sequence[str],
    b5_parent_id: str,
    config: Mapping[str, Any],
    b2_checkpoint_path: Path,
    b2_intake_bundle_path: Path,
    dinov2_backbone: nn.Module,
    dinov2_contract: Any,
    descriptor_builder: Callable[..., dict[str, Any]],
    real_smoke_fit_limit: int | None = None,
) -> dict[str, Any]:
    """Train requested candidates and publish evaluation-compatible artifacts."""

    validated_config = validate_production_config(config)
    requested = tuple(dict.fromkeys(candidates))
    if not requested or any(item not in PRODUCTION_CANDIDATES for item in requested):
        raise ValueError("production candidates must use canonical successor IDs")
    if b5_parent_id not in B5_PARENTS:
        raise ValueError("B5 parent must be B3, B4-U0, or B4-U1")
    if set(requested) & _B5_CANDIDATES and b5_parent_id not in requested:
        raise ValueError("the precommitted B5 parent must be included in candidates")
    if b5_parent_id.startswith("B4") and "B3" not in requested:
        raise ValueError("B4 parent production requires B3")
    if any(item.startswith("B4") for item in requested) and "B3" not in requested:
        raise ValueError("B4 production requires B3")
    if ("B4-U0" in requested) != ("B4-U1" in requested):
        raise ValueError("B4-U0 and B4-U1 must run together for equal-step semantics")
    if real_smoke_fit_limit is not None and (
        isinstance(real_smoke_fit_limit, bool)
        or not isinstance(real_smoke_fit_limit, int)
        or real_smoke_fit_limit <= 0
    ):
        raise ValueError("real smoke FIT limit must be a positive integer")
    if (
        real_smoke_fit_limit is not None
        and "B0-FV" in requested
        and real_smoke_fit_limit < 128
    ):
        raise ValueError("B0-FV real smoke requires a FIT limit of at least 128")

    inventory = validate_face_visible_successor_inventory_bundle(
        successor_inventory_bundle, verify_artifacts=True
    )
    evaluation_tokens = tuple(required_evaluation_tokens)
    if (
        not evaluation_tokens
        or evaluation_tokens != tuple(sorted(set(evaluation_tokens)))
        or any(not _is_sha256(item) for item in evaluation_tokens)
        or not _is_sha256(evaluation_panel_sha256)
    ):
        raise ValueError("required evaluation population or panel digest differs")
    rows_by_token = {
        row["sample_token"]: row
        for row in inventory["inventory"]["successor_population"]
    }
    if missing := set(evaluation_tokens) - set(rows_by_token):
        raise ValueError(
            f"evaluation population is absent from inventory: {sorted(missing)[:3]}"
        )
    evaluation_rows = tuple(rows_by_token[token] for token in evaluation_tokens)
    if any(row["state"] != "USABLE" for row in evaluation_rows):
        raise ValueError("required evaluation population must be usable")
    fit_rows = tuple(
        row
        for row in inventory["inventory"]["successor_population"]
        if row["state"] == "USABLE" and row["gradient_eligible"]
    )
    fit_rows = _eligible_fit_rows(fit_rows)
    if real_smoke_fit_limit is not None:
        fit_rows = _bounded_balanced_fit_rows(fit_rows, real_smoke_fit_limit)
    samples_by_token = {
        row["sample_token"]: _sample_from_row(row)
        for row in (*fit_rows, *evaluation_rows)
    }
    population_tokens = tuple(sorted(samples_by_token))
    population_samples = tuple(samples_by_token[token] for token in population_tokens)
    fit_samples = tuple(samples_by_token[row["sample_token"]] for row in fit_rows)
    fit_population_sha256 = content_sha256(
        [
            {
                "sample_token": row["sample_token"],
                "registered_identity_id": row["registered_identity_id"],
                "record_sha256": row["record_sha256"],
            }
            for row in fit_rows
        ]
    )
    dino_bindings = dinov2_contract_bindings(dinov2_contract)
    run_payload = {
        "schema_version": PRODUCTION_RUN_SCHEMA,
        "successor_inventory_bundle_sha256": inventory["bundle_sha256"],
        "successor_inventory_sha256": inventory["inventory_sha256"],
        "evaluation_panel_sha256": evaluation_panel_sha256,
        "required_evaluation_tokens_sha256": content_sha256(list(evaluation_tokens)),
        "fit_population_sha256": fit_population_sha256,
        "fit_sample_count": len(fit_rows),
        "population_sample_count": len(population_tokens),
        "candidates": list(requested),
        "b5_parent_id": b5_parent_id,
        "config": validated_config,
        "config_sha256": content_sha256(validated_config),
        "real_smoke_fit_limit": real_smoke_fit_limit,
        "b2_sources": {
            "checkpoint": {
                "absolute_path": os.fspath(b2_checkpoint_path.resolve(strict=True)),
                **file_binding(b2_checkpoint_path),
            },
            "intake_bundle": {
                "absolute_path": os.fspath(b2_intake_bundle_path.resolve(strict=True)),
                **file_binding(b2_intake_bundle_path),
            },
        },
        "dinov2": dino_bindings,
    }
    run_manifest = {
        **run_payload,
        "run_manifest_sha256": content_sha256(run_payload),
    }
    root = _initialize_run(output_dir, run_manifest)
    started = time.monotonic()
    _progress("run_initialized", started)
    device = _training_device(validated_config)
    reset_successor_seed(validated_config["seed"], use_cuda=device.type == "cuda")
    token_cache = None
    if set(requested) & _DINO_CANDIDATES:
        _progress("dinov2_token_cache_start", started)
        token_cache = _materialize_or_open_token_cache(
            root / "dinov2-token-cache",
            model=Dinov2OccupancyProbe128(dinov2_backbone),
            samples=population_samples,
            config=validated_config,
            bindings={
                "run_manifest_sha256": run_manifest["run_manifest_sha256"],
                "population_tokens_sha256": content_sha256(list(population_tokens)),
                **dino_bindings,
            },
        )
        _progress("dinov2_token_cache_ready", started)

    schedule = build_balanced_pk_schedule(
        fit_samples,
        identities_per_batch=validated_config["identities_per_batch"],
        samples_per_identity=validated_config["samples_per_identity"],
        seed=validated_config["seed"],
    )
    models: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    for candidate_id in PRODUCTION_CANDIDATES:
        if candidate_id not in requested:
            continue
        target = root / candidate_id
        _progress(f"candidate_start:{candidate_id}", started)
        if target.is_dir() and not target.is_symlink():
            models[candidate_id] = _restore_candidate_model(
                candidate_id,
                target=target,
                config=validated_config,
                bindings=_candidate_bindings(
                    root, run_manifest, candidate_id, b5_parent_id
                ),
                b2_checkpoint_path=b2_checkpoint_path,
                b2_intake_bundle_path=b2_intake_bundle_path,
                dinov2_backbone=dinov2_backbone,
                models=models,
                b5_parent_id=b5_parent_id,
            )
            _complete_descriptor_if_needed(
                target,
                candidate_id=candidate_id,
                inventory=inventory,
                evaluation_tokens=evaluation_tokens,
                evaluation_panel_sha256=evaluation_panel_sha256,
                descriptor_builder=descriptor_builder,
            )
            statuses[candidate_id] = "VALIDATED_EXISTING"
            _progress(f"candidate_validated:{candidate_id}", started)
            continue
        model, training = _train_candidate(
            candidate_id,
            config=validated_config,
            fit_samples=fit_samples,
            population_samples=population_samples,
            schedule=schedule,
            token_cache=token_cache,
            b2_checkpoint_path=b2_checkpoint_path,
            b2_intake_bundle_path=b2_intake_bundle_path,
            dinov2_backbone=dinov2_backbone,
            models=models,
            b5_parent_id=b5_parent_id,
        )
        models[candidate_id] = model
        embeddings = _extract_candidate_embeddings(
            candidate_id,
            model,
            population_samples=population_samples,
            token_cache=token_cache,
            config=validated_config,
        )
        _publish_candidate(
            target,
            candidate_id=candidate_id,
            model=model,
            training=training,
            embeddings=embeddings,
            population_tokens=population_tokens,
            evaluation_tokens=evaluation_tokens,
            config=validated_config,
            bindings=_candidate_bindings(
                root, run_manifest, candidate_id, b5_parent_id
            ),
            parent_id=b5_parent_id if candidate_id in _B5_CANDIDATES else None,
        )
        _complete_descriptor_if_needed(
            target,
            candidate_id=candidate_id,
            inventory=inventory,
            evaluation_tokens=evaluation_tokens,
            evaluation_panel_sha256=evaluation_panel_sha256,
            descriptor_builder=descriptor_builder,
        )
        statuses[candidate_id] = "CREATED"
        _progress(f"candidate_created:{candidate_id}", started)

    family_payload = {
        "schema_version": PRODUCTION_FAMILY_SCHEMA,
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "b5_parent_id": b5_parent_id,
        "candidates": [
            {
                "candidate_id": candidate_id,
                "candidate_run_sha256": _read_json(
                    root / candidate_id / "candidate-run.json"
                )["candidate_run_sha256"],
                "cache_descriptor_sha256": _read_json(
                    root / candidate_id / "evaluation-cache-descriptor.json"
                )["cache_descriptor_sha256"],
            }
            for candidate_id in requested
        ],
        "status": "COMPLETE_REQUESTED_SUCCESSOR_FAMILY",
    }
    family = {**family_payload, "family_run_sha256": content_sha256(family_payload)}
    family_path = root / "family-run.json"
    if family_path.exists():
        if _read_json(family_path) != family:
            raise ValueError("successor family resume binding differs")
    else:
        write_private_json_bundle(((family_path, family),))
        fsync_directory(root)
    return {
        "output_dir": os.fspath(root),
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "family_run_sha256": family["family_run_sha256"],
        "candidate_statuses": statuses,
        "cache_descriptors": [
            os.fspath(root / candidate_id / "evaluation-cache-descriptor.json")
            for candidate_id in requested
        ],
    }


def _progress(stage: str, started: float) -> None:
    print(
        f"[full128-successors] {stage} elapsed_seconds={time.monotonic() - started:.1f}",
        flush=True,
    )


def build_balanced_pk_schedule(
    samples: Sequence[Full128Sample],
    *,
    identities_per_batch: int,
    samples_per_identity: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Build deterministic dataset-balanced P x 2 batches."""

    rows = tuple(samples)
    if identities_per_batch < 2 or samples_per_identity != 2 or not rows:
        raise ValueError("balanced successor schedule requires P>=2, K=2, and samples")
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, sample in enumerate(rows):
        grouped[sample.dataset_name][sample.identity_id].append(index)
    if any(
        len(indices) < 2
        for identities in grouped.values()
        for indices in identities.values()
    ):
        raise ValueError(
            "balanced successor schedule requires two samples per identity"
        )
    dataset_names = sorted(grouped)
    if len(dataset_names) > identities_per_batch:
        raise ValueError(
            "balanced successor schedule cannot represent every dataset within P"
        )
    identity_total = sum(len(grouped[name]) for name in dataset_names)
    p = min(identities_per_batch, identity_total)
    if p < 2:
        raise ValueError("balanced successor schedule requires two identities")
    quotas = {name: p // len(dataset_names) for name in dataset_names}
    for name in dataset_names[: p % len(dataset_names)]:
        quotas[name] += 1
    for name in dataset_names:
        if quotas[name] == 0:
            quotas[name] = 1
    while sum(quotas.values()) > p:
        name = max(dataset_names, key=lambda item: quotas[item])
        if quotas[name] <= 1:
            break
        quotas[name] -= 1
    generator = torch.Generator().manual_seed(seed)
    orders: dict[str, list[str]] = {}
    for name in dataset_names:
        identities = sorted(grouped[name])
        permutation = torch.randperm(len(identities), generator=generator).tolist()
        orders[name] = [identities[index] for index in permutation]
    batch_count = max(
        math.ceil(len(orders[name]) / quotas[name]) for name in dataset_names
    )
    batches: list[tuple[int, ...]] = []
    for batch_index in range(batch_count):
        batch: list[int] = []
        for name in dataset_names:
            identities = orders[name]
            for offset in range(quotas[name]):
                identity = identities[
                    (batch_index * quotas[name] + offset) % len(identities)
                ]
                candidates = grouped[name][identity]
                order = torch.randperm(len(candidates), generator=generator).tolist()
                batch.extend(candidates[index] for index in order[:2])
        if len({rows[index].identity_id for index in batch}) < 2:
            raise ValueError("balanced successor batch lacks a negative identity")
        batches.append(tuple(batch))
    return tuple(batches)


class _RawBatchSequence(Sequence[Mapping[str, torch.Tensor]]):
    def __init__(
        self, samples: Sequence[Full128Sample], schedule: Sequence[Sequence[int]]
    ) -> None:
        self.dataset = Full128TorchDataset(
            samples,
            identity_to_label={
                identity: index
                for index, identity in enumerate(
                    sorted({row.identity_id for row in samples})
                )
            },
            payload_mode="compact",
        )
        self.schedule = tuple(tuple(batch) for batch in schedule)

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        items = [self.dataset[item] for item in self.schedule[index]]
        return {
            "rgb": torch.stack([item["rgb"] for item in items]).float().div_(255.0),
            "mask": torch.stack([item["mask"] for item in items]).float(),
            "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
        }


class _TokenBatchSequence(Sequence[Mapping[str, torch.Tensor]]):
    def __init__(
        self,
        samples: Sequence[Full128Sample],
        schedule: Sequence[Sequence[int]],
        cache: Mapping[str, Any],
    ) -> None:
        index = {token: row for row, token in enumerate(cache["sample_tokens"])}
        self.indices = tuple(
            tuple(index[samples[item].sample_id] for item in batch)
            for batch in schedule
        )
        identities = sorted({sample.identity_id for sample in samples})
        labels = {identity: index for index, identity in enumerate(identities)}
        self.batch_labels = tuple(
            torch.tensor([labels[samples[item].identity_id] for item in batch])
            for batch in schedule
        )
        self.tokens = cache["tokens"]
        self.occupancy = cache["occupancy"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        rows = self.indices[index]
        return {
            "tokens": torch.from_numpy(np.asarray(self.tokens[list(rows)]).copy()),
            "occupancy": torch.from_numpy(
                np.asarray(self.occupancy[list(rows)]).copy()
            ),
            "label": self.batch_labels[index],
        }


class _IdentityBlindBatchSequence(Sequence[Mapping[str, torch.Tensor]]):
    def __init__(self, cache: Mapping[str, Any], batch_size: int) -> None:
        self.tokens = cache["tokens"]
        self.occupancy = cache["occupancy"]
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.tokens) / self.batch_size)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.tokens))
        tokens = torch.from_numpy(np.asarray(self.tokens[start:stop]).copy())
        occupancy = torch.from_numpy(np.asarray(self.occupancy[start:stop]).copy())
        return make_identity_blind_views(tokens, occupancy, phase=index)


def _train_candidate(
    candidate_id: str,
    *,
    config: Mapping[str, Any],
    fit_samples: Sequence[Full128Sample],
    population_samples: Sequence[Full128Sample],
    schedule: Sequence[Sequence[int]],
    token_cache: Mapping[str, Any] | None,
    b2_checkpoint_path: Path,
    b2_intake_bundle_path: Path,
    dinov2_backbone: nn.Module,
    models: Mapping[str, Any],
    b5_parent_id: str,
) -> tuple[Any, dict[str, Any]]:
    reset_successor_seed(
        config["seed"], use_cuda=config["precision"]["device"] == "cuda"
    )
    if candidate_id == "B0-FV":
        if len(fit_samples) < 128:
            raise ValueError("B0-FV requires at least 128 FIT samples")
        model = Classical128()
        raw = _extract_raw_descriptors(
            model, population_samples, workers=config["workers"]
        )
        by_token = {
            sample.sample_id: index for index, sample in enumerate(population_samples)
        }
        fit_indices = np.asarray([by_token[sample.sample_id] for sample in fit_samples])
        model.fit_descriptors(
            raw[fit_indices], sample_ids=[row.sample_id for row in fit_samples]
        )
        return ClassicalFV128(model), {
            "kind": "FIT_STANDARD_SCALER_AND_FULL_SVD_PCA128",
            "fit_sample_count": len(fit_samples),
            "attempted_steps": 1,
            "update_steps": 1,
        }
    raw_batches = _RawBatchSequence(fit_samples, schedule)
    if candidate_id == "B1-FV":
        model = build_b1_fv()
        return model, train_supervised_fixed_steps(
            model, raw_batches, _legacy_config(config)
        )
    if candidate_id == "B2-FV":
        model = build_b2_fv(
            b2_checkpoint_path, intake_bundle_path=b2_intake_bundle_path
        )
        return model, train_supervised_fixed_steps(
            model, raw_batches, _legacy_config(config)
        )
    assert token_cache is not None
    token_batches = _TokenBatchSequence(fit_samples, schedule, token_cache)
    if candidate_id == "B3":
        model = Dinov2OccupancyProbe128(dinov2_backbone)
        return model, train_supervised_fixed_steps(
            model, token_batches, _legacy_config(config)
        )
    b3 = models.get("B3")
    if not isinstance(b3, Dinov2OccupancyProbe128):
        raise TypeError("B4/B5 production requires a restored B3 model")
    if candidate_id.startswith("B4"):
        model = IdentityBlindResidualTokenAdapter128(
            dinov2_backbone, deepcopy(b3.projection).cpu()
        )
        if candidate_id == "B4-U1" and isinstance(
            models.get("B4-U0"), IdentityBlindResidualTokenAdapter128
        ):
            model.adapter.load_state_dict(
                models["B4-U0"].adapter.state_dict(), strict=True
            )
        ssl_batches = _IdentityBlindBatchSequence(
            token_cache, config["extraction_batch_size"]
        )
        return model, train_identity_blind_fixed_steps(
            model,
            ssl_batches,
            _legacy_config(config),
            update_enabled=candidate_id == "B4-U1",
        )
    parent_adapter: nn.Module | None = None
    if b5_parent_id.startswith("B4"):
        parent = models.get(b5_parent_id)
        if not isinstance(parent, IdentityBlindResidualTokenAdapter128):
            raise RuntimeError("B5 precommitted B4 parent is unavailable")
        parent_adapter = deepcopy(parent.adapter).cpu()
    model = SpatialScorer128(
        dinov2_backbone,
        deepcopy(b3.projection).cpu(),
        token_adapter=parent_adapter,
        uniform_spatial=candidate_id != "B5-SPATIAL",
        channel_gate=candidate_id == "B5-CHANNEL",
    )
    if candidate_id == "B5-UNIFORM":
        return model, train_supervised_no_update_fixed_steps(
            model, token_batches, _legacy_config(config)
        )
    return model, train_supervised_fixed_steps(
        model, token_batches, _legacy_config(config)
    )


def _extract_candidate_embeddings(
    candidate_id: str,
    model: Any,
    *,
    population_samples: Sequence[Full128Sample],
    token_cache: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> np.ndarray:
    if candidate_id == "B0-FV":
        descriptor = model.model
        raw = _extract_raw_descriptors(
            descriptor, population_samples, workers=config["workers"]
        )
        return descriptor.transform_descriptors(raw)
    if candidate_id in {"B1-FV", "B2-FV"}:
        loader = DataLoader(
            Full128TorchDataset(population_samples, payload_mode="compact"),
            batch_size=config["extraction_batch_size"],
            shuffle=False,
            num_workers=config["workers"],
        )
        device = _training_device(config)
        model.to(device).eval()
        blocks = []
        with torch.inference_mode():
            for batch in loader:
                rgb = batch["rgb"].to(device=device, dtype=torch.float32).div_(255.0)
                mask = batch["mask"].to(device=device, dtype=torch.float32)
                blocks.append(model(rgb, mask).float().cpu().numpy())
        return np.concatenate(blocks).astype(np.float32, copy=False)
    assert token_cache is not None
    device = _training_device(config)
    model.to(device).eval()
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(population_samples), config["extraction_batch_size"]):
            stop = min(start + config["extraction_batch_size"], len(population_samples))
            tokens = torch.from_numpy(
                np.asarray(token_cache["tokens"][start:stop]).copy()
            ).to(device)
            occupancy = torch.from_numpy(
                np.asarray(token_cache["occupancy"][start:stop]).copy()
            ).to(device)
            blocks.append(
                model.forward_from_tokens(tokens, occupancy).float().cpu().numpy()
            )
    return np.concatenate(blocks).astype(np.float32, copy=False)


def _publish_candidate(
    target: Path,
    *,
    candidate_id: str,
    model: Any,
    training: Mapping[str, Any],
    embeddings: np.ndarray,
    population_tokens: Sequence[str],
    evaluation_tokens: Sequence[str],
    config: Mapping[str, Any],
    bindings: Mapping[str, Any],
    parent_id: str | None,
) -> None:
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    os.chmod(staging, 0o700)
    try:
        if candidate_id == "B0-FV":
            checkpoint_dir = staging / "checkpoint"
            checkpoint_dir.mkdir(mode=0o700)
            state_path = checkpoint_dir / "classical-state.npz"
            model.model.save_state(state_path)
            checkpoint = _simple_checkpoint_manifest(
                candidate_id, state_path, config, bindings, training
            )
            (checkpoint_dir / "checkpoint-manifest.json").write_bytes(
                json_document_bytes(checkpoint)
            )
        else:
            checkpoint = write_successor_checkpoint(
                staging / "checkpoint",
                candidate_id=_checkpoint_candidate_id(candidate_id),
                model=model,
                config=_legacy_config(config),
                bindings=bindings,
                training_receipt=training,
            )
        manifests = _candidate_manifests(candidate_id, parent_id=parent_id)
        write_private_json_bundle(
            tuple(
                (staging / f"{name}-manifest.json", value)
                for name, value in manifests.items()
            )
        )
        matrix = _validate_embeddings(embeddings, len(population_tokens))
        population_path = staging / "population-embeddings.f32le"
        population_path.write_bytes(np.ascontiguousarray(matrix, dtype="<f4").tobytes())
        index = {token: row for row, token in enumerate(population_tokens)}
        evaluation_matrix = matrix[[index[token] for token in evaluation_tokens]]
        evaluation_path = staging / "evaluation-embeddings.f32le"
        evaluation_path.write_bytes(
            np.ascontiguousarray(evaluation_matrix, dtype="<f4").tobytes()
        )
        population_manifest_payload = {
            "schema_version": "cvi.full128_successor_population_embeddings.v1",
            "candidate_id": candidate_id,
            "sample_tokens": list(population_tokens),
            "sample_tokens_sha256": content_sha256(list(population_tokens)),
            "pack": {
                "relative_path": population_path.name,
                **file_binding(population_path),
            },
            "dimension": 128,
            "dtype": "float32_little_endian",
            "normalization": "L2",
        }
        population_manifest = {
            **population_manifest_payload,
            "manifest_sha256": content_sha256(population_manifest_payload),
        }
        (staging / "population-embedding-manifest.json").write_bytes(
            json_document_bytes(population_manifest)
        )
        run_payload = {
            "schema_version": PRODUCTION_CANDIDATE_SCHEMA,
            "candidate_id": candidate_id,
            "parent_id": parent_id,
            "bindings": dict(bindings),
            "bindings_sha256": content_sha256(dict(bindings)),
            "training": dict(training),
            "checkpoint_manifest_sha256": checkpoint["checkpoint_manifest_sha256"],
            "checkpoint_sha256": checkpoint["state"]["sha256"],
            "model_manifest_sha256": content_sha256(manifests["model"]),
            "preprocessing_manifest_sha256": content_sha256(manifests["preprocessing"]),
            "embedding_manifest_sha256": content_sha256(manifests["embedding"]),
            "population_embedding_manifest_sha256": population_manifest[
                "manifest_sha256"
            ],
            "evaluation_pack": {
                "relative_path": evaluation_path.name,
                **file_binding(evaluation_path),
            },
        }
        candidate_run = {
            **run_payload,
            "candidate_run_sha256": content_sha256(run_payload),
        }
        (staging / "candidate-run.json").write_bytes(json_document_bytes(candidate_run))
        fsync_directory(staging)
        rename_directory_noreplace(staging, target)
        fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _complete_descriptor_if_needed(
    target: Path,
    *,
    candidate_id: str,
    inventory: Mapping[str, Any],
    evaluation_tokens: Sequence[str],
    evaluation_panel_sha256: str,
    descriptor_builder: Callable[..., dict[str, Any]],
) -> None:
    descriptor_path = target / "evaluation-cache-descriptor.json"
    run = _read_json(target / "candidate-run.json")
    if run["candidate_id"] != candidate_id:
        raise ValueError("candidate resume ID differs")
    descriptor = descriptor_builder(
        successor_id=candidate_id,
        pack_path=target / run["evaluation_pack"]["relative_path"],
        sample_tokens=evaluation_tokens,
        successor_inventory_bundle_sha256=inventory["bundle_sha256"],
        successor_inventory_sha256=inventory["inventory_sha256"],
        evaluation_panel_sha256=evaluation_panel_sha256,
        model_manifest_sha256=run["model_manifest_sha256"],
        checkpoint_sha256=run["checkpoint_sha256"],
        preprocessing_manifest_sha256=run["preprocessing_manifest_sha256"],
        embedding_manifest_sha256=run["embedding_manifest_sha256"],
    )
    if descriptor_path.exists():
        if _read_json(descriptor_path) != descriptor:
            raise ValueError("candidate evaluation descriptor resume differs")
    else:
        write_private_json_bundle(((descriptor_path, descriptor),))
        fsync_directory(target)


def _restore_candidate_model(
    candidate_id: str,
    *,
    target: Path,
    config: Mapping[str, Any],
    bindings: Mapping[str, Any],
    b2_checkpoint_path: Path,
    b2_intake_bundle_path: Path,
    dinov2_backbone: nn.Module,
    models: Mapping[str, Any],
    b5_parent_id: str,
) -> Any:
    run = _validate_candidate_artifacts(target)
    if run["candidate_id"] != candidate_id or run["bindings"] != dict(bindings):
        raise ValueError("candidate resume immutable binding differs")
    if candidate_id == "B0-FV":
        return ClassicalFV128(
            Classical128.load_state(target / "checkpoint" / "classical-state.npz")
        )
    if candidate_id == "B1-FV":
        model: Any = build_b1_fv()
    elif candidate_id == "B2-FV":
        model = build_b2_fv(
            b2_checkpoint_path, intake_bundle_path=b2_intake_bundle_path
        )
    elif candidate_id == "B3":
        model = Dinov2OccupancyProbe128(dinov2_backbone)
    elif candidate_id.startswith("B4"):
        b3 = models.get("B3")
        if not isinstance(b3, Dinov2OccupancyProbe128):
            raise RuntimeError("resumed B4 requires B3")
        model = IdentityBlindResidualTokenAdapter128(
            dinov2_backbone, deepcopy(b3.projection)
        )
    else:
        b3 = models.get("B3")
        if not isinstance(b3, Dinov2OccupancyProbe128):
            raise RuntimeError("resumed B5 requires B3")
        adapter = None
        if b5_parent_id.startswith("B4"):
            parent = models.get(b5_parent_id)
            if not isinstance(parent, IdentityBlindResidualTokenAdapter128):
                raise RuntimeError("resumed B5 parent is unavailable")
            adapter = deepcopy(parent.adapter)
        model = SpatialScorer128(
            dinov2_backbone,
            deepcopy(b3.projection),
            token_adapter=adapter,
            uniform_spatial=candidate_id != "B5-SPATIAL",
            channel_gate=candidate_id == "B5-CHANNEL",
        )
    load_successor_checkpoint(
        target / "checkpoint",
        candidate_id=_checkpoint_candidate_id(candidate_id),
        model=model,
        config=_legacy_config(config),
        bindings=bindings,
    )
    return model


def _validate_candidate_artifacts(target: Path) -> dict[str, Any]:
    run = _read_json(target / "candidate-run.json")
    payload = {
        key: value for key, value in run.items() if key != "candidate_run_sha256"
    }
    if run.get("schema_version") != PRODUCTION_CANDIDATE_SCHEMA or run.get(
        "candidate_run_sha256"
    ) != content_sha256(payload):
        raise ValueError("candidate run manifest digest differs")
    evaluation = run.get("evaluation_pack")
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "relative_path",
        "sha256",
        "byte_size",
    }:
        raise ValueError("candidate evaluation pack binding differs")
    evaluation_path = target / evaluation["relative_path"]
    if file_binding(evaluation_path) != {
        "sha256": evaluation["sha256"],
        "byte_size": evaluation["byte_size"],
    }:
        raise ValueError("candidate evaluation pack digest differs")
    for name in ("model", "preprocessing", "embedding"):
        manifest = _read_json(target / f"{name}-manifest.json")
        if content_sha256(manifest) != run[f"{name}_manifest_sha256"]:
            raise ValueError(f"candidate {name} manifest digest differs")
    checkpoint = _read_json(target / "checkpoint" / "checkpoint-manifest.json")
    checkpoint_payload = {
        key: value
        for key, value in checkpoint.items()
        if key != "checkpoint_manifest_sha256"
    }
    if (
        checkpoint.get("checkpoint_manifest_sha256")
        != run["checkpoint_manifest_sha256"]
        or checkpoint.get("checkpoint_manifest_sha256")
        != content_sha256(checkpoint_payload)
        or checkpoint.get("state", {}).get("sha256") != run["checkpoint_sha256"]
    ):
        raise ValueError("candidate checkpoint manifest binding differs")
    state = checkpoint["state"]
    state_path = target / "checkpoint" / state["relative_path"]
    if file_binding(state_path) != {
        "sha256": state["sha256"],
        "byte_size": state["byte_size"],
    }:
        raise ValueError("candidate checkpoint state digest differs")
    population = _read_json(target / "population-embedding-manifest.json")
    population_payload = {
        key: value for key, value in population.items() if key != "manifest_sha256"
    }
    if (
        population.get("manifest_sha256") != content_sha256(population_payload)
        or population.get("manifest_sha256")
        != run["population_embedding_manifest_sha256"]
    ):
        raise ValueError("candidate population embedding manifest differs")
    pack = population["pack"]
    if file_binding(target / pack["relative_path"]) != {
        "sha256": pack["sha256"],
        "byte_size": pack["byte_size"],
    }:
        raise ValueError("candidate population embedding pack differs")
    return run


def _materialize_or_open_token_cache(
    target: Path,
    *,
    model: Dinov2OccupancyProbe128,
    samples: Sequence[Full128Sample],
    config: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if target.is_dir() and not target.is_symlink():
        return _open_token_cache(target, bindings=bindings)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    os.chmod(staging, 0o700)
    try:
        token_count = 256
        token_path = staging / "tokens.f32le"
        occupancy_path = staging / "occupancy.f32le"
        tokens = np.memmap(
            token_path,
            dtype="<f4",
            mode="w+",
            shape=(len(samples), token_count, 384),
        )
        occupancy = np.memmap(
            occupancy_path,
            dtype="<f4",
            mode="w+",
            shape=(len(samples), token_count),
        )
        loader = DataLoader(
            Full128TorchDataset(samples, payload_mode="compact"),
            batch_size=config["extraction_batch_size"],
            shuffle=False,
            num_workers=config["workers"],
        )
        device = _training_device(config)
        model.to(device).eval()
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                rgb = batch["rgb"].to(device=device, dtype=torch.float32).div_(255.0)
                mask = batch["mask"].to(device=device, dtype=torch.float32)
                patch, area = model.extract_tokens(rgb, mask)
                stop = offset + len(patch)
                tokens[offset:stop] = patch.float().cpu().numpy()
                occupancy[offset:stop] = area.float().cpu().numpy()
                offset = stop
        if offset != len(samples):
            raise RuntimeError("DINOv2 token extraction coverage differs")
        tokens.flush()
        occupancy.flush()
        del tokens, occupancy
        payload = {
            "schema_version": PRODUCTION_TOKEN_CACHE_SCHEMA,
            "sample_tokens": [sample.sample_id for sample in samples],
            "sample_tokens_sha256": content_sha256(
                [sample.sample_id for sample in samples]
            ),
            "sample_count": len(samples),
            "token_count": token_count,
            "token_dimension": 384,
            "dtype": "float32_little_endian",
            "tokens": {"relative_path": token_path.name, **file_binding(token_path)},
            "occupancy": {
                "relative_path": occupancy_path.name,
                **file_binding(occupancy_path),
            },
            "bindings": dict(bindings),
            "bindings_sha256": content_sha256(dict(bindings)),
        }
        manifest = {**payload, "cache_manifest_sha256": content_sha256(payload)}
        (staging / "cache-manifest.json").write_bytes(json_document_bytes(manifest))
        fsync_directory(staging)
        rename_directory_noreplace(staging, target)
        fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _open_token_cache(target, bindings=bindings)


def _open_token_cache(target: Path, *, bindings: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_json(target / "cache-manifest.json")
    payload = {
        key: value for key, value in manifest.items() if key != "cache_manifest_sha256"
    }
    if (
        manifest.get("schema_version") != PRODUCTION_TOKEN_CACHE_SCHEMA
        or manifest.get("cache_manifest_sha256") != content_sha256(payload)
        or manifest.get("bindings") != dict(bindings)
        or manifest.get("bindings_sha256") != content_sha256(dict(bindings))
        or manifest.get("token_count") != 256
        or manifest.get("token_dimension") != 384
    ):
        raise ValueError("production DINOv2 token cache binding differs")
    shape = (manifest["sample_count"], 256, 384)
    occupancy_shape = (manifest["sample_count"], 256)
    for name, expected_bytes in (
        ("tokens", int(np.prod(shape)) * 4),
        ("occupancy", int(np.prod(occupancy_shape)) * 4),
    ):
        path = target / manifest[name]["relative_path"]
        if (
            file_binding(path)
            != {
                "sha256": manifest[name]["sha256"],
                "byte_size": manifest[name]["byte_size"],
            }
            or manifest[name]["byte_size"] != expected_bytes
        ):
            raise ValueError("production DINOv2 token cache file differs")
    return {
        "manifest": manifest,
        "sample_tokens": tuple(manifest["sample_tokens"]),
        "tokens": np.memmap(
            target / manifest["tokens"]["relative_path"],
            dtype="<f4",
            mode="r",
            shape=shape,
        ),
        "occupancy": np.memmap(
            target / manifest["occupancy"]["relative_path"],
            dtype="<f4",
            mode="r",
            shape=occupancy_shape,
        ),
    }


def _candidate_manifests(
    candidate_id: str, *, parent_id: str | None
) -> dict[str, dict[str, Any]]:
    model = {
        "schema_version": "cvi.full128_successor_model.v1",
        "candidate_id": candidate_id,
        "architecture": _architecture(candidate_id),
        "parent_id": parent_id,
    }
    preprocessing = {
        "schema_version": "cvi.full128_successor_preprocessing.v1",
        "candidate_id": candidate_id,
        "input": "EXISTING_224X224_FULL128_RGB_AND_BINARY_MASK",
        "recrop_permitted": False,
        "background": "IMAGENET_MEAN_NEUTRAL",
        "patch_occupancy": "AREA" if candidate_id in _DINO_CANDIDATES else None,
    }
    embedding = {
        "schema_version": "cvi.full128_successor_embedding.v1",
        "candidate_id": candidate_id,
        "dimension": 128,
        "dtype": "float32",
        "normalization": "L2",
    }
    return {"model": model, "preprocessing": preprocessing, "embedding": embedding}


def _architecture(candidate_id: str) -> str:
    if candidate_id == "B0-FV":
        return "CLASSICAL128_REUSE"
    if candidate_id in {"B1-FV", "B2-FV"}:
        return "RESNET18_AREA_MASKED_GAP_LINEAR128"
    if candidate_id == "B3":
        return "FROZEN_DINOV2_SMALL_PATCH_OCCUPANCY_LINEAR128"
    if candidate_id.startswith("B4"):
        return "FROZEN_DINOV2_AND_PROJECTION_ZERO_INIT_TOKEN_ADAPTER"
    return "FROZEN_PRECOMMITTED_PARENT_ZERO_INIT_SPATIAL_SCORER"


def _simple_checkpoint_manifest(
    candidate_id: str,
    state_path: Path,
    config: Mapping[str, Any],
    bindings: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "cvi.full128_successor_checkpoint.v1",
        "candidate_id": candidate_id,
        "state": {"relative_path": state_path.name, **file_binding(state_path)},
        "parameter_partition": {"trainable": [], "frozen": []},
        "config_sha256": content_sha256(_legacy_config(config)),
        "bindings": dict(bindings),
        "bindings_sha256": content_sha256(dict(bindings)),
        "training_receipt": dict(training),
        "training_receipt_sha256": content_sha256(dict(training)),
    }
    return {**payload, "checkpoint_manifest_sha256": content_sha256(payload)}


def _candidate_bindings(
    root: Path,
    run_manifest: Mapping[str, Any],
    candidate_id: str,
    parent_id: str,
) -> dict[str, Any]:
    bindings: dict[str, Any] = {
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "candidate_id": candidate_id,
    }
    if candidate_id in _B5_CANDIDATES:
        parent_run = _read_json(root / parent_id / "candidate-run.json")
        bindings["precommitted_parent_id"] = parent_id
        bindings["precommitted_parent_candidate_run_sha256"] = parent_run[
            "candidate_run_sha256"
        ]
        bindings["precommitted_parent_checkpoint_sha256"] = parent_run[
            "checkpoint_sha256"
        ]
    return bindings


def _checkpoint_candidate_id(candidate_id: str) -> str:
    aliases = {
        "B3": "B3-FV",
        "B4-U0": "B4-U0-FV",
        "B4-U1": "B4-U1-FV",
        "B5-UNIFORM": "B5-UNIFORM-FV",
        "B5-CHANNEL": "B5-CHANNEL-GATE-FV",
        "B5-SPATIAL": "B5-FV",
    }
    return aliases.get(candidate_id, candidate_id)


def _legacy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "cvi.full128_successor_training_config.v1",
        "seed": config["seed"],
        "supervised_steps": config["supervised_steps"],
        "ssl_steps": config["ssl_steps"],
        "optimizer": dict(config["optimizer"]),
        "precision": dict(config["precision"]),
        "workers": config["workers"],
        "triplet_margin": config["triplet_margin"],
        "ssl_objective": config["ssl_objective"],
    }


def _eligible_fit_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    counts = Counter(row["registered_identity_id"] for row in rows)
    eligible = tuple(row for row in rows if counts[row["registered_identity_id"]] >= 2)
    if len({row["registered_identity_id"] for row in eligible}) < 2:
        raise ValueError("successor FIT population needs at least two K=2 identities")
    return eligible


def _bounded_balanced_fit_rows(
    rows: Sequence[Mapping[str, Any]], limit: int
) -> tuple[Mapping[str, Any], ...]:
    by_dataset: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_dataset[row["dataset_name"]][row["registered_identity_id"]].append(row)
    selected: list[Mapping[str, Any]] = []
    identities = {
        dataset: sorted(values.items())
        for dataset, values in sorted(by_dataset.items())
    }
    offsets = {dataset: 0 for dataset in identities}
    while len(selected) + 2 <= limit:
        progressed = False
        for dataset, dataset_identities in identities.items():
            offset = offsets[dataset]
            if offset >= len(dataset_identities) or len(selected) + 2 > limit:
                continue
            _, values = dataset_identities[offset]
            selected.extend(sorted(values, key=lambda item: item["sample_token"])[:2])
            offsets[dataset] += 1
            progressed = True
        if not progressed:
            break
    if len({row["registered_identity_id"] for row in selected}) < 2:
        raise ValueError("real smoke limit cannot retain two K=2 identities")
    return tuple(sorted(selected, key=lambda item: item["sample_token"]))


def _sample_from_row(row: Mapping[str, Any]) -> Full128Sample:
    artifact = row["artifact"]
    if row["state"] != "USABLE" or not isinstance(artifact, Mapping):
        raise ValueError("production sample must have usable artifact bindings")
    return Full128Sample(
        sample_id=row["sample_token"],
        identity_id=row["registered_identity_id"],
        dataset_name=row["dataset_name"],
        view="face",
        role=row["protocol_scope"],
        rgb_path=Path(artifact["full_rgb_path"]),
        rgb_sha256=artifact["full_rgb_sha256"],
        mask_path=Path(artifact["full_mask_path"]),
        mask_sha256=artifact["full_mask_sha256"],
        crop_record_sha256=artifact["crop_record_sha256"],
    )


def _validate_embeddings(values: np.ndarray, expected: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (expected, 128) or not np.isfinite(matrix).all():
        raise ValueError("successor population embeddings must be finite [N,128]")
    if not np.allclose(
        np.linalg.norm(matrix.astype(np.float64), axis=1), 1.0, atol=1e-5, rtol=1e-5
    ):
        raise ValueError("successor population embeddings must be L2 normalized")
    return matrix


def _training_device(config: Mapping[str, Any]) -> torch.device:
    device = torch.device(config["precision"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("successor production CUDA was requested but unavailable")
    return device


def _initialize_run(output_dir: Path, manifest: Mapping[str, Any]) -> Path:
    requested = output_dir.absolute()
    parent = requested.parent.resolve(strict=True)
    root = parent / requested.name
    repository = Path(__file__).resolve().parents[3]
    if root == repository or root.is_relative_to(repository):
        raise ValueError("successor production output must remain outside repository")
    if requested.is_symlink():
        raise ValueError("successor production output must not be a symlink")
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ValueError("successor production output must be a regular directory")
        if _read_json(root / "run-manifest.json") != dict(manifest):
            raise FileExistsError(
                "existing successor run has different immutable bindings"
            )
        return root
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=parent))
    os.chmod(staging, 0o700)
    try:
        write_private_json_bundle(((staging / "run-manifest.json", dict(manifest)),))
        fsync_directory(staging)
        rename_directory_noreplace(staging, root)
        fsync_directory(parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root


def _read_json(path: Path) -> dict[str, Any]:
    return read_strict_json_document(path, maximum_bytes=1_073_741_824).payload


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "B5_PARENTS",
    "PRODUCTION_CANDIDATES",
    "PRODUCTION_CONFIG_SCHEMA",
    "build_balanced_pk_schedule",
    "default_production_config",
    "prepare_production_runtime",
    "restore_successor_trace_context",
    "run_successor_production",
    "validate_production_config",
]
