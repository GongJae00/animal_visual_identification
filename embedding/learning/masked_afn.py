"""K-fold residual training and availability-aware fusion for masked A/F/N."""

from __future__ import annotations

import hashlib
import math
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from identity.research.dataset_stratified_kfold import (
    DatasetStratifiedIdentityKFoldManifest,
    materialize_identity_fold,
)
from identity.registry.identity_registry import (
    compute_registered_dog_id,
    compute_sample_token,
)
from identity.splits.protected_public_split import PublicSplitSourceBundle
from parsing.regions.dinov2_region_segmentation import read_region_candidates

REPORT_SCHEMA = "cvi.masked_afn_kfold_report.v1"
INTERPRETATION = (
    "RETROSPECTIVE_EXPOSED_CROSS_VALIDATION_WITH_MODEL_GENERATED_REGION_CANDIDATES_"
    "NOT_VERIFIED_SEGMENTATION_OR_FINAL_BIOMETRIC_EVALUATION"
)
REGIONS = ("A", "F", "N")


def train_and_evaluate_masked_afn(
    *,
    candidate_manifest_paths: Sequence[Path],
    kfold: DatasetStratifiedIdentityKFoldManifest,
    output_dir: Path,
    epochs: int = 8,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    residual_scale: float = 0.25,
    fusion_resolution: int = 10,
    device: str = "cuda",
    seed: int = 20260803,
    source_bundle_path: Path | None = None,
    image_content_receipts_path: Path | None = None,
) -> dict[str, Any]:
    """Train 3xK adapters and evaluate sparse A/F/N retrieval."""

    _validate_training_arguments(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        residual_scale=residual_scale,
        fusion_resolution=fusion_resolution,
        device=device,
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite masked AFN output: {output_dir}")
    if (source_bundle_path is None) != (image_content_receipts_path is None):
        raise ValueError(
            "source bundle and image-content receipts must be provided together"
        )
    source_binding = None
    source_resolver = None
    if source_bundle_path is not None and image_content_receipts_path is not None:
        source_resolver, source_binding = _load_source_token_resolver(
            source_bundle_path=source_bundle_path,
            image_content_receipts_path=image_content_receipts_path,
        )
    rows, manifest_hashes = _load_candidate_rows(
        candidate_manifest_paths,
        source_resolver=source_resolver,
    )
    identity_by_token = {
        item.identity_token: item for item in kfold.identity_assignments
    }
    fold_sample_by_token = {
        item.sample_token: item for item in kfold.sample_assignments
    }
    joined: dict[str, dict[str, Any]] = {}
    for sample_token, row in rows.items():
        fold_sample = fold_sample_by_token.get(sample_token)
        if fold_sample is None or fold_sample.source_variant != "original":
            continue
        identity = identity_by_token[fold_sample.identity_token]
        if row["registered_dog_id"] != identity.registered_dog_id:
            raise ValueError("candidate and K-fold registered identity differ")
        joined[sample_token] = {
            **row,
            "identity_token": fold_sample.identity_token,
            "dataset_name": identity.dataset_name,
        }
    missing = {
        item.sample_token
        for item in kfold.sample_assignments
        if item.source_variant == "original" and item.home_fold is not None
    } - set(joined)
    if missing:
        raise ValueError(
            f"region candidates do not cover {len(missing)} K-fold original samples"
        )

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    torch_device = torch.device(device)
    _set_seed(seed)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    )
    try:
        fold_reports: list[dict[str, Any]] = []
        checkpoint_bindings: list[dict[str, Any]] = []
        for fold_index in range(kfold.policy.fold_count):
            view = materialize_identity_fold(kfold, fold_index)["view"]
            sample_roles = {
                item["sample_token"]: item for item in view["sample_assignments"]
            }
            adapters: dict[str, Any] = {}
            region_training: dict[str, Any] = {}
            for region in REGIONS:
                train_rows = [
                    row
                    for token, row in joined.items()
                    if sample_roles[token]["sample_role"] == "TRAIN_INPUT"
                    and row["embeddings"].get(region) is not None
                ]
                if not train_rows:
                    raise ValueError(
                        f"fold {fold_index} region {region} has no training candidates"
                    )
                adapter, training_summary = _train_region_adapter(
                    train_rows,
                    region=region,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    residual_scale=residual_scale,
                    device=torch_device,
                    seed=seed + fold_index * 10 + REGIONS.index(region),
                )
                adapters[region] = adapter
                checkpoint_path = staging / f"fold-{fold_index}-{region}.safetensors"
                from safetensors.torch import save_file

                save_file(
                    {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
                    str(checkpoint_path),
                    metadata={
                        "schema_version": "cvi.masked_afn_residual_adapter.v1",
                        "fold_index": str(fold_index),
                        "region": region,
                        "kfold_manifest_sha256": kfold.manifest_sha256,
                    },
                )
                checkpoint_bindings.append(
                    {
                        "fold_index": fold_index,
                        "region": region,
                        "relative_path": checkpoint_path.name,
                        **_file_binding(checkpoint_path),
                    }
                )
                region_training[region] = training_summary
            encoded = _encode_joined(joined, adapters, device=torch_device)
            dev = _partition_rows(joined, encoded, sample_roles, stage="DEV")
            test = _partition_rows(joined, encoded, sample_roles, stage="TEST")
            selected_weights, dev_metrics = _fit_fusion_weights(
                dev,
                resolution=fusion_resolution,
            )
            test_metrics = _evaluate_partition(test, selected_weights)
            fold_reports.append(
                {
                    "fold_index": fold_index,
                    "training": region_training,
                    "fusion": {
                        "selection_partition": "DEV",
                        "resolution": fusion_resolution,
                        "selected_weights": selected_weights,
                        "dev": dev_metrics,
                        "test": test_metrics,
                    },
                }
            )
        body = {
            "schema_version": REPORT_SCHEMA,
            "kfold_manifest_sha256": kfold.manifest_sha256,
            "candidate_manifest_sha256s": dict(sorted(manifest_hashes.items())),
            "candidate_source_token_binding": source_binding,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "residual_scale": residual_scale,
                "fusion_resolution": fusion_resolution,
                "device": device,
                "seed": seed,
                "regions": list(REGIONS),
                "embedding": "DINOV2_FOREGROUND_PATCH_TOKEN_MEAN_384D",
                "adapter": "BOUNDED_RESIDUAL_ARCFACE",
            },
            "checkpoints": checkpoint_bindings,
            "folds": fold_reports,
            "retrieval_eligibility": _retrieval_eligibility(kfold),
            "out_of_fold_test": _aggregate_fold_metrics(fold_reports),
            "interpretation": INTERPRETATION,
        }
        report = {**body, "report_sha256": content_sha256(body)}
        write_private_json_bundle(((staging / "masked_afn_report.json", report),))
        fsync_directory(staging)
        rename_directory_noreplace(staging, output_dir)
        fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def _load_candidate_rows(
    paths: Sequence[Path],
    *,
    source_resolver: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    rows: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for path in paths:
        manifest, arrays = read_region_candidates(path)
        dataset = manifest["dataset_name"]
        if dataset in hashes:
            raise ValueError("candidate manifests repeat a dataset")
        hashes[dataset] = content_sha256(manifest)
        for record in manifest["records"]:
            token = record["sample_id"]
            registered_dog_id = record["registered_identity_id"]
            if source_resolver is not None:
                source = _resolve_candidate_source(record, source_resolver)
                token = source["sample_token"]
                registered_dog_id = compute_registered_dog_id(
                    source["dataset_identity_id"]
                )
                if record["registered_identity_id"] != registered_dog_id:
                    raise ValueError(
                        "candidate and receipt-bound source registered identity differ"
                    )
            if token in rows:
                raise ValueError("candidate manifests repeat a sample")
            index = record["array_index"]
            embeddings: dict[str, np.ndarray | None] = {}
            for region in REGIONS:
                value = arrays[f"{region}_embeddings"][index].astype(np.float32)
                embeddings[region] = value if np.isfinite(value).all() else None
            rows[token] = {
                "sample_token": token,
                "registered_dog_id": registered_dog_id,
                "embeddings": embeddings,
                "availability": "".join(
                    region
                    for region in REGIONS
                    if embeddings[region] is not None
                )
                or "NONE",
            }
    return rows, hashes


def _load_source_token_resolver(
    *,
    source_bundle_path: Path,
    image_content_receipts_path: Path,
) -> tuple[
    dict[tuple[str, str], tuple[dict[str, Any], ...]],
    dict[str, Any],
]:
    limits = {
        "maximum_bytes": 536_870_912,
        "maximum_nodes": 10_000_000,
        "maximum_keys": 5_000_000,
        "maximum_array_length": 1_000_000,
    }
    source_document = read_strict_json_document(source_bundle_path, **limits)
    receipts_document = read_strict_json_document(
        image_content_receipts_path, **limits
    )
    source_bundle = PublicSplitSourceBundle.from_dict(source_document.payload)
    evidence_bindings = dict(source_bundle.evidence_bindings)
    if (
        evidence_bindings.get("image_content_receipts_sha256")
        != receipts_document.canonical_payload_sha256
    ):
        raise ValueError(
            "image-content receipts are not bound by the public source bundle"
        )
    source_by_id = {
        item.source_sample_id: item
        for item in source_bundle.samples
        if item.source_variant == "original"
    }
    resolver: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_source_ids: set[str] = set()
    for value in receipts_document.payload.values():
        if not isinstance(value, Mapping) or not isinstance(
            value.get("receipt"), Mapping
        ):
            raise TypeError("merged image-content receipt schema differs")
        receipt = value["receipt"]
        if receipt.get("decision") != "PASS_IMAGE_CONTENT_AUDIT" or not isinstance(
            receipt.get("records"), list
        ):
            raise ValueError("image-content receipt is not an audited record set")
        for row in receipt["records"]:
            if not isinstance(row, Mapping) or row.get("source_variant") != "original":
                continue
            source_id = row.get("source_sample_id")
            encoded_sha256 = row.get("encoded_sha256")
            dataset_name = row.get("dataset_name")
            source = source_by_id.get(source_id)
            if source is None:
                continue
            if (
                dataset_name != source.dataset_name
                or not isinstance(encoded_sha256, str)
                or len(encoded_sha256) != 64
                or compute_sample_token(source.source_sample_id) != source.sample_token
                or source_id in seen_source_ids
            ):
                raise ValueError("image-content receipt source binding differs")
            seen_source_ids.add(source_id)
            resolver[(dataset_name, encoded_sha256)].append(
                {
                    "sample_token": source.sample_token,
                    "dataset_identity_id": source.dataset_identity_id,
                    "member_path": row.get("member_path"),
                }
            )
    if seen_source_ids != set(source_by_id):
        raise ValueError("image-content receipts do not cover source-bundle originals")
    return (
        {key: tuple(value) for key, value in resolver.items()},
        {
            "source_bundle_sha256": source_bundle.bundle_sha256,
            "source_bundle_document_sha256": source_document.canonical_payload_sha256,
            "image_content_receipts_sha256": (
                receipts_document.canonical_payload_sha256
            ),
            "method": "DATASET_AND_ENCODED_SHA256_WITH_MEMBER_PATH_DISAMBIGUATION",
        },
    )


def _resolve_candidate_source(
    record: Mapping[str, Any],
    resolver: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    candidates = tuple(
        resolver.get((record.get("dataset_name"), record.get("image_sha256")), ())
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidates = tuple(
            item
            for item in candidates
            if _paths_share_suffix(record.get("image_path"), item.get("member_path"))
        )
    if len(candidates) != 1:
        raise ValueError("candidate image does not resolve to one audited source sample")
    return candidates[0]


def _paths_share_suffix(left: object, right: object) -> bool:
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        return False
    left_parts = PurePosixPath(left.replace("\\", "/")).parts
    right_parts = PurePosixPath(right.replace("\\", "/")).parts
    width = min(len(left_parts), len(right_parts))
    return left_parts[-width:] == right_parts[-width:]


def _train_region_adapter(
    rows: Sequence[Mapping[str, Any]],
    *,
    region: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    residual_scale: float,
    device: Any,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    from embedding.learning.train.trainer import ArcFaceHead

    identities = sorted({row["identity_token"] for row in rows})
    label_by_identity = {identity: index for index, identity in enumerate(identities)}
    features = torch.from_numpy(
        np.stack([row["embeddings"][region] for row in rows]).astype(np.float32)
    )
    labels = torch.tensor(
        [label_by_identity[row["identity_token"]] for row in rows],
        dtype=torch.long,
    )
    adapter = _ResidualAdapter(residual_scale=residual_scale).to(device)
    head = ArcFaceHead(384, len(identities), 32.0, 0.25).to(device)
    optimizer = torch.optim.AdamW(
        [*adapter.parameters(), *head.parameters()],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    adapter.train()
    head.train()
    for _ in range(epochs):
        order = torch.randperm(len(rows), generator=generator)
        epoch_loss = 0.0
        batches = 0
        for offset in range(0, len(rows), batch_size):
            indices = order[offset : offset + batch_size]
            batch_features = features[indices].to(device)
            batch_labels = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = adapter(batch_features)
            logits = head(embeddings, batch_labels)
            loss = torch.nn.functional.cross_entropy(logits, batch_labels)
            if not torch.isfinite(loss):
                raise RuntimeError("masked AFN training produced non-finite loss")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        losses.append(epoch_loss / max(batches, 1))
    adapter.eval()
    return adapter, {
        "sample_count": len(rows),
        "identity_count": len(identities),
        "epoch_losses": losses,
    }


def _encode_joined(
    rows: Mapping[str, Mapping[str, Any]],
    adapters: Mapping[str, Any],
    *,
    device: Any,
) -> dict[str, dict[str, np.ndarray]]:
    import torch

    result: dict[str, dict[str, np.ndarray]] = {token: {} for token in rows}
    for region in REGIONS:
        tokens = [
            token for token, row in rows.items() if row["embeddings"][region] is not None
        ]
        if not tokens:
            continue
        values = np.stack([rows[token]["embeddings"][region] for token in tokens])
        with torch.inference_mode():
            encoded = adapters[region](
                torch.from_numpy(values.astype(np.float32)).to(device)
            ).cpu().numpy()
        for token, embedding in zip(tokens, encoded, strict=True):
            result[token][region] = embedding.astype(np.float32, copy=False)
    return result


class _ResidualAdapter:
    def __new__(cls, *, residual_scale: float):
        import torch

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.adapter = torch.nn.Sequential(
                    torch.nn.LayerNorm(384),
                    torch.nn.Linear(384, 384),
                    torch.nn.GELU(),
                    torch.nn.Linear(384, 384),
                )
                torch.nn.init.zeros_(self.adapter[-1].weight)
                torch.nn.init.zeros_(self.adapter[-1].bias)

            def forward(self, base):
                base = torch.nn.functional.normalize(base, dim=1)
                residual = self.adapter(base)
                residual = residual / torch.linalg.vector_norm(
                    residual, dim=1, keepdim=True
                ).clamp_min(1.0)
                return torch.nn.functional.normalize(
                    base + residual_scale * residual, dim=1
                )

        return Module()


def _partition_rows(
    joined: Mapping[str, Mapping[str, Any]],
    encoded: Mapping[str, Mapping[str, np.ndarray]],
    roles: Mapping[str, Mapping[str, Any]],
    *,
    stage: str,
) -> dict[str, Any]:
    gallery: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    queries: list[dict[str, Any]] = []
    for token, row in joined.items():
        role = roles[token]
        if role["stage"] != stage:
            continue
        if role["sample_role"] == "GALLERY":
            for region, embedding in encoded[token].items():
                gallery[row["identity_token"]][region].append(embedding)
        elif role["sample_role"] == "QUERY":
            queries.append(
                {
                    "sample_token": token,
                    "identity_token": row["identity_token"],
                    "dataset_name": row["dataset_name"],
                    "availability": "".join(encoded[token]) or "NONE",
                    "embeddings": encoded[token],
                }
            )
    gallery_embeddings: dict[str, dict[str, np.ndarray]] = {}
    for identity, regions in gallery.items():
        gallery_embeddings[identity] = {
            region: _normalize(np.mean(values, axis=0))
            for region, values in regions.items()
        }
    return {
        "gallery": gallery_embeddings,
        "queries": sorted(queries, key=lambda item: item["sample_token"]),
    }


def _fit_fusion_weights(
    partition: Mapping[str, Any], *, resolution: int
) -> tuple[dict[str, float], dict[str, Any]]:
    prepared = _prepare_partition_scores(partition)
    probe = _evaluate_prepared_partition(
        prepared, {region: 1.0 / len(REGIONS) for region in REGIONS}
    )
    if probe["overall"]["query_count"] == 0:
        raise ValueError("DEV partition has no scoreable retrieval queries")
    best: tuple[float, float, tuple[int, int, int]] | None = None
    best_weights: dict[str, float] | None = None
    best_metrics: dict[str, Any] | None = None
    for first in range(resolution + 1):
        for second in range(resolution - first + 1):
            values = (first, second, resolution - first - second)
            if not any(values):
                continue
            weights = {
                region: value / resolution
                for region, value in zip(REGIONS, values, strict=True)
            }
            metrics = _evaluate_prepared_partition(prepared, weights)
            key = (metrics["overall"]["Rank-1"], metrics["overall"]["MRR"], values)
            if best is None or key > best:
                best, best_weights, best_metrics = key, weights, metrics
    if best_weights is None or best_metrics is None:
        raise RuntimeError("fusion search produced no candidate")
    return best_weights, best_metrics


def _evaluate_partition(
    partition: Mapping[str, Any], weights: Mapping[str, float]
) -> dict[str, Any]:
    return _evaluate_prepared_partition(_prepare_partition_scores(partition), weights)


def _prepare_partition_scores(partition: Mapping[str, Any]) -> dict[str, Any]:
    gallery = partition["gallery"]
    queries = partition["queries"]
    identities = tuple(sorted(gallery))
    identity_index = {identity: index for index, identity in enumerate(identities)}
    branch_scores: dict[str, np.ndarray] = {}
    for region in REGIONS:
        gallery_valid = np.asarray(
            [region in gallery[identity] for identity in identities], dtype=bool
        )
        query_valid = np.asarray(
            [region in query["embeddings"] for query in queries], dtype=bool
        )
        if gallery_valid.sum() < 2 or not query_valid.any():
            continue
        gallery_values = np.zeros((len(identities), 384), dtype=np.float32)
        query_values = np.zeros((len(queries), 384), dtype=np.float32)
        for index, identity in enumerate(identities):
            if gallery_valid[index]:
                gallery_values[index] = gallery[identity][region]
        for index, query in enumerate(queries):
            if query_valid[index]:
                query_values[index] = query["embeddings"][region]
        scores = query_values @ gallery_values.T
        valid = query_valid[:, None] & gallery_valid[None, :]
        valid_counts = valid.sum(axis=1)
        eligible = valid_counts >= 2
        sums = np.where(valid, scores, 0.0).sum(axis=1)
        means = np.divide(
            sums,
            valid_counts,
            out=np.zeros(len(queries), dtype=np.float32),
            where=valid_counts > 0,
        )
        centered = scores - means[:, None]
        variances = np.divide(
            np.where(valid, centered * centered, 0.0).sum(axis=1),
            valid_counts,
            out=np.zeros(len(queries), dtype=np.float32),
            where=valid_counts > 0,
        )
        normalized = centered / np.sqrt(variances).clip(min=1e-8)[:, None]
        branch_scores[region] = np.where(
            valid & eligible[:, None], normalized, np.nan
        ).astype(np.float32, copy=False)
    return {
        "identities": identities,
        "truth_indices": np.asarray(
            [identity_index.get(query["identity_token"], -1) for query in queries],
            dtype=np.int64,
        ),
        "datasets": tuple(query["dataset_name"] for query in queries),
        "availability": tuple(query["availability"] for query in queries),
        "branch_scores": branch_scores,
    }


def _retrieval_eligibility(
    kfold: DatasetStratifiedIdentityKFoldManifest,
) -> dict[str, Any]:
    dataset_by_identity = {
        item.identity_token: item.dataset_name for item in kfold.identity_assignments
    }
    roles_by_identity: dict[str, Counter[str]] = defaultdict(Counter)
    samples_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in kfold.sample_assignments:
        if sample.source_variant != "original" or sample.home_fold is None:
            continue
        dataset = dataset_by_identity[sample.identity_token]
        roles_by_identity[sample.identity_token][sample.held_out_role.value] += 1
        samples_by_dataset[dataset][sample.held_out_role.value] += 1
    return {
        dataset: {
            "identity_count": sum(
                dataset_by_identity[identity] == dataset
                for identity in roles_by_identity
            ),
            "retrieval_eligible_identity_count": sum(
                dataset_by_identity[identity] == dataset
                and roles["GALLERY"] > 0
                and roles["QUERY"] > 0
                for identity, roles in roles_by_identity.items()
            ),
            "gallery_sample_count": samples_by_dataset[dataset]["GALLERY"],
            "query_sample_count": samples_by_dataset[dataset]["QUERY"],
            "excluded_sample_count": samples_by_dataset[dataset]["EXCLUDED"],
        }
        for dataset in sorted(set(dataset_by_identity.values()))
    }


def _evaluate_prepared_partition(
    prepared: Mapping[str, Any], weights: Mapping[str, float]
) -> dict[str, Any]:
    identities = prepared["identities"]
    truth_indices = prepared["truth_indices"]
    shape = (len(truth_indices), len(identities))
    numerator = np.zeros(shape, dtype=np.float32)
    denominator = np.zeros(shape, dtype=np.float32)
    for region, scores in prepared["branch_scores"].items():
        weight = weights[region]
        if weight <= 0.0:
            continue
        valid = np.isfinite(scores)
        numerator += np.where(valid, scores * weight, 0.0)
        denominator += valid * weight
    fused = np.divide(
        numerator,
        denominator,
        out=np.full(shape, -np.inf, dtype=np.float32),
        where=denominator > 0.0,
    )
    rows: list[dict[str, Any]] = []
    abstained = 0
    candidate_indices = np.arange(len(identities))
    for query_index, truth in enumerate(truth_indices):
        if truth < 0 or not np.isfinite(fused[query_index, truth]):
            abstained += 1
            continue
        truth_score = fused[query_index, truth]
        rank = 1 + int(np.count_nonzero(fused[query_index] > truth_score))
        rank += int(
            np.count_nonzero(
                (fused[query_index] == truth_score) & (candidate_indices < truth)
            )
        )
        rows.append(
            {
                "dataset_name": prepared["datasets"][query_index],
                "availability": prepared["availability"][query_index],
                "rank": rank,
                "Rank-1": float(rank == 1),
                "Rank-5": float(rank <= 5),
                "MRR": 1.0 / rank,
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["availability"]].append(row)
    return {
        "overall": _metrics(rows, abstained=abstained),
        "by_availability": {
            key: _metrics(values, abstained=0) for key, values in sorted(grouped.items())
        },
        "by_dataset": {
            dataset: _metrics(
                [row for row in rows if row["dataset_name"] == dataset],
                abstained=0,
            )
            for dataset in sorted({row["dataset_name"] for row in rows})
        },
    }


def _metrics(rows: Sequence[Mapping[str, Any]], *, abstained: int) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "abstained_count": abstained,
        "Rank-1": float(np.mean([row["Rank-1"] for row in rows])) if rows else 0.0,
        "Rank-5": float(np.mean([row["Rank-5"] for row in rows])) if rows else 0.0,
        "MRR": float(np.mean([row["MRR"] for row in rows])) if rows else 0.0,
    }


def _aggregate_fold_metrics(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    weighted = Counter()
    for fold in folds:
        metrics = fold["fusion"]["test"]["overall"]
        count = metrics["query_count"]
        totals["query_count"] += count
        totals["abstained_count"] += metrics["abstained_count"]
        for name in ("Rank-1", "Rank-5", "MRR"):
            weighted[name] += metrics[name] * count
    count = totals["query_count"]
    return {
        "query_count": totals["query_count"],
        "abstained_count": totals["abstained_count"],
        **{
            name: float(weighted[name] / count) if count else 0.0
            for name in ("Rank-1", "Rank-5", "MRR")
        },
    }


def _normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(array).all() or norm <= 1e-8:
        raise ValueError("masked AFN embedding is non-finite or zero")
    return array / norm


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_training_arguments(**values: Any) -> None:
    for name in ("epochs", "batch_size", "fusion_resolution"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("learning_rate", "residual_scale"):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if values["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")


def _file_binding(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "byte_size": path.stat().st_size}


__all__ = ["train_and_evaluate_masked_afn"]
