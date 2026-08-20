"""Score-blind Full128 panels and variant-bound cached evaluation.

This module consumes metadata and externally produced embedding packs.  It does
not import or depend on Full128 training implementations.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from shared.foundation.provenance import canonical_json_bytes, content_sha256
from evaluation.splits.full_split_census import validate_unified_full_split_bundle
from archive.full128.methods.preparation.inventory import (
    BUNDLE_SCHEMA as INVENTORY_BUNDLE_SCHEMA,
)
from archive.full128.methods.preparation.inventory import (
    INVENTORY_SCHEMA,
)
from archive.full128.methods.training.manifests import build_baseline_family_manifest
from search.scoring.roles import FULL128_CHANNEL, SCORER_ALGORITHM

PANEL_SCHEMA = "archive.full128.evaluation_panel.v1"
CACHE_DESCRIPTOR_SCHEMA = "archive.full128.packed_embedding_cache_descriptor.v1"
REPORT_SCHEMA = "archive.full128.evaluation_report.v1"
FAMILY_INDEX_SCHEMA = "archive.full128.evaluation_family_index.v1"
MASTER_TABLE_SCHEMA = "archive.full128.master_table.v1"
EMBEDDING_DIMENSION = 128
VARIANT_IDS = ("B0", "B1", "B2")
ENROLLMENT_KS = (1, 3, 5)
RANK_KS = (1, 5, 10)
IDENTITY_NONE_DATASETS = ("ap10k-dog", "dogflw", "oxford-pets-dog")
_METRIC_NAMES = ("Rank-1", "Rank-5", "Rank-10", "MRR", "mAP", "mINP")
_MAXIMUM_PACK_BYTES = 64 * 1024 * 1024 * 1024
_VECTOR_BYTES = EMBEDDING_DIMENSION * np.dtype("<f4").itemsize
_DIAGNOSTIC_QUERY_BLOCK_ROWS = 4_096
_DIAGNOSTIC_MAX_SCORE_ENTRIES = 4_194_304
_BOOTSTRAP_BLOCK_RESAMPLES = 4_096


class Full128EvaluationError(ValueError):
    """Raised when Full128 evaluation evidence violates its contract."""


@runtime_checkable
class Full128EmbeddingCacheAdapter(Protocol):
    """Generic strict adapter for an external, non-inline embedding pack."""

    @property
    def descriptor(self) -> Mapping[str, Any]: ...

    def load_embeddings(self, sample_tokens: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class _Full128FamilyEvidence:
    """Private typed proof that family metadata and its panel were validated once."""

    bundle: dict[str, Any]
    panel: dict[str, Any]
    inventory_sample_tokens: frozenset[str]
    variant_manifests: dict[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class PackedFull128EmbeddingCacheAdapter:
    """Validated immutable view of one training variant's packed cache."""

    _descriptor: dict[str, Any]
    pack_path: Path
    _panel_vectors: np.ndarray
    _family_evidence: _Full128FamilyEvidence | None = None

    @property
    def descriptor(self) -> Mapping[str, Any]:
        return self._descriptor

    @classmethod
    def from_training_variant_directory(
        cls,
        variant_directory: Path,
        *,
        training_run_manifest: Mapping[str, Any],
        inventory_bundle: Mapping[str, Any],
        panel: Mapping[str, Any],
    ) -> PackedFull128EmbeddingCacheAdapter:
        evidence = _build_family_evidence(inventory_bundle, panel=panel)
        return cls._from_training_variant_directory(
            variant_directory,
            training_run_manifest=training_run_manifest,
            evidence=evidence,
        )

    @classmethod
    def _from_training_variant_directory(
        cls,
        variant_directory: Path,
        *,
        training_run_manifest: Mapping[str, Any],
        evidence: _Full128FamilyEvidence,
        training_run_validated: bool = False,
    ) -> PackedFull128EmbeddingCacheAdapter:
        from shared.foundation.protected_io import read_strict_json_document

        bundle = evidence.bundle
        validated_panel = evidence.panel
        run_manifest = (
            dict(training_run_manifest)
            if training_run_validated
            else _validate_training_run_manifest(
                training_run_manifest, inventory_bundle=bundle
            )
        )
        root = _resolve_regular_directory(variant_directory, "training variant")
        try:
            variant_document = read_strict_json_document(
                root / "variant-run.json", maximum_bytes=1_073_741_824
            )
            variant_run = _validate_variant_run_for_evaluation(
                root, variant_document.payload
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise Full128EvaluationError("training variant validation failed") from exc
        variant_id = variant_run["variant_id"]
        if root.name != variant_id or variant_id not in VARIANT_IDS:
            raise Full128EvaluationError("training variant directory identity differs")
        _validate_variant_training_bindings(variant_run, run_manifest, bundle)

        family_variant = evidence.variant_manifests[variant_id]
        if (
            variant_run["method"] != family_variant["method"]
            or variant_run["initialization"] != family_variant["initialization"]
        ):
            raise Full128EvaluationError("training variant family contract differs")

        artifacts = variant_run["artifacts"]
        manifests = {
            name: _read_bound_json_artifact(root, artifacts[f"{name}_manifest"])
            for name in ("checkpoint", "preprocessing", "embedding")
        }
        if (
            manifests["checkpoint"].get("preprocessing_manifest_sha256")
            != content_sha256(manifests["preprocessing"])
            or manifests["checkpoint"].get("embedding_manifest_sha256")
            != content_sha256(manifests["embedding"])
            or manifests["checkpoint"].get("checkpoint_sha256")
            != artifacts["state"]["sha256"]
        ):
            raise Full128EvaluationError("training artifact manifest content differs")

        cache_binding = artifacts["embedding_cache_manifest"]
        cache_manifest = cache_binding["manifest"]
        _validate_cache_inventory_rows(cache_manifest, bundle)
        by_sample = {row["sample_id"]: row for row in cache_manifest["vectors"]}
        requested = validated_panel["required_sample_tokens"]
        if missing := set(requested) - set(by_sample):
            raise Full128EvaluationError(
                f"training cache omits evaluation panel samples: {sorted(missing)[:3]}"
            )
        selected_vectors = [
            {
                "sample_token": token,
                "offset_bytes": by_sample[token]["offset_bytes"],
                "byte_size": by_sample[token]["byte_size"],
                "sha256": by_sample[token]["sha256"],
            }
            for token in requested
        ]
        storage = {
            "format": "PACKED_FLOAT32_LITTLE_ENDIAN",
            "relative_path": cache_manifest["relative_path"],
            "pack_sha256": cache_manifest["pack_sha256"],
            "pack_byte_size": cache_manifest["pack_byte_size"],
            "source_vector_count": cache_manifest["vector_count"],
            "vectors": selected_vectors,
        }
        descriptor: dict[str, Any] = {
            "schema_version": CACHE_DESCRIPTOR_SCHEMA,
            "variant_id": variant_id,
            "baseline_family_sha256": bundle["baseline_family_sha256"],
            "variant_manifest_sha256": content_sha256(family_variant),
            "variant_artifact_sha256": variant_run["variant_run_sha256"],
            "training_run_sha256": run_manifest["run_manifest_sha256"],
            "checkpoint_manifest_file_sha256": artifacts["checkpoint_manifest"][
                "sha256"
            ],
            "checkpoint_manifest_sha256": content_sha256(manifests["checkpoint"]),
            "preprocessing_manifest_file_sha256": artifacts["preprocessing_manifest"][
                "sha256"
            ],
            "preprocessing_manifest_sha256": content_sha256(manifests["preprocessing"]),
            "embedding_manifest_file_sha256": artifacts["embedding_manifest"]["sha256"],
            "embedding_manifest_sha256": content_sha256(manifests["embedding"]),
            "embedding_cache_manifest_file_sha256": cache_binding["sha256"],
            "embedding_cache_manifest_sha256": cache_manifest["cache_manifest_sha256"],
            "inventory_bundle_sha256": bundle["bundle_sha256"],
            "inventory_sha256": bundle["inventory_sha256"],
            "split_manifest_sha256": bundle["split_manifest_sha256"],
            "split_census_sha256": bundle["split_census_sha256"],
            "panel_sha256": validated_panel["panel_sha256"],
            "embedding_dimension": EMBEDDING_DIMENSION,
            "dtype": "float32",
            "normalization": "L2",
            "sample_tokens": list(requested),
            "sample_tokens_sha256": content_sha256(requested),
            "storage": storage,
        }
        descriptor["cache_descriptor_sha256"] = content_sha256(descriptor)
        validated_descriptor = _validate_cache_descriptor_shape(descriptor)
        pack_path = root / cache_manifest["relative_path"]
        panel_vectors = _stream_validate_embedding_pack(
            pack_path,
            cache_manifest=cache_manifest,
            selected_tokens=requested,
        )
        panel_vectors.setflags(write=False)
        return cls(validated_descriptor, pack_path, panel_vectors, evidence)

    def load_embeddings(self, sample_tokens: Sequence[str]) -> np.ndarray:
        descriptor = _validate_cache_descriptor_shape(self._descriptor)
        requested = tuple(sample_tokens)
        if len(requested) != len(set(requested)):
            raise Full128EvaluationError("embedding request repeats a sample token")
        row_by_token = {
            row["sample_token"]: row for row in descriptor["storage"]["vectors"]
        }
        if unknown := set(requested) - set(row_by_token):
            raise Full128EvaluationError(
                f"embedding cache omits requested samples: {sorted(unknown)[:3]}"
            )
        if not requested:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        source_index = {
            token: index for index, token in enumerate(descriptor["sample_tokens"])
        }
        matrix = self._panel_vectors[
            np.fromiter((source_index[token] for token in requested), dtype=np.int64)
        ].copy()
        _validate_loaded_matrix(matrix, len(requested))
        return matrix


def discover_packed_full128_embedding_cache_adapters(
    training_run: Path,
    *,
    inventory_bundle: Mapping[str, Any],
    panel: Mapping[str, Any] | None = None,
) -> tuple[PackedFull128EmbeddingCacheAdapter, ...]:
    """Discover and validate exactly B0/B1/B2 under one completed training run."""

    from shared.foundation.protected_io import read_strict_json_document

    root = _resolve_regular_directory(training_run, "training run")
    directories = {
        entry.name for entry in root.iterdir() if entry.is_dir() or entry.is_symlink()
    }
    if directories != set(VARIANT_IDS):
        raise Full128EvaluationError(
            "training run must contain exactly B0, B1, and B2 variant directories"
        )
    try:
        run_manifest = read_strict_json_document(
            root / "run-manifest.json", maximum_bytes=1_073_741_824
        ).payload
        family_run = read_strict_json_document(
            root / "family-run.json", maximum_bytes=64 * 1024 * 1024
        ).payload
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise Full128EvaluationError("training family manifests are invalid") from exc
    evidence = _build_family_evidence(inventory_bundle, panel=panel)
    validated_run = _validate_training_run_manifest(
        run_manifest, inventory_bundle=evidence.bundle
    )
    _validate_training_family_run(family_run, validated_run)
    adapters = tuple(
        PackedFull128EmbeddingCacheAdapter._from_training_variant_directory(
            root / variant,
            training_run_manifest=validated_run,
            evidence=evidence,
            training_run_validated=True,
        )
        for variant in VARIANT_IDS
    )
    expected = {
        adapter.descriptor["variant_id"]: adapter.descriptor["variant_artifact_sha256"]
        for adapter in adapters
    }
    observed = {
        row["variant_id"]: row["variant_run_sha256"] for row in family_run["variants"]
    }
    if observed != expected:
        raise Full128EvaluationError("training family variant bindings differ")
    return adapters


@dataclass(frozen=True, slots=True)
class ImmutableFull128EvaluationReport:
    """Canonical immutable Full128 variant report."""

    _canonical_report: bytes
    report_sha256: str

    @property
    def report(self) -> dict[str, Any]:
        value = json.loads(self._canonical_report)
        assert isinstance(value, dict)
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA,
            "report": self.report,
            "report_sha256": self.report_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImmutableFull128EvaluationReport:
        _exact_keys(payload, {"schema_version", "report", "report_sha256"}, "report")
        if payload["schema_version"] != REPORT_SCHEMA:
            raise Full128EvaluationError("Full128 report schema differs")
        report = payload["report"]
        if not isinstance(report, Mapping):
            raise Full128EvaluationError("Full128 report payload must be an object")
        _require_sha256(payload["report_sha256"], "report_sha256")
        if content_sha256(report) != payload["report_sha256"]:
            raise Full128EvaluationError("Full128 report was tampered with")
        _validate_report_content(report)
        canonical = canonical_json_bytes(report)
        return cls(canonical, hashlib.sha256(canonical).hexdigest())


def build_full128_evaluation_panel(
    inventory_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Build all evaluation cohorts from metadata before any vectors are loaded.

    The caller must first run the artifact-level inventory-v2 validator.  This
    function independently revalidates all metadata hashes and the split/census.
    """

    return _build_family_evidence(inventory_bundle).panel


def _build_panel_from_validated_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    records = tuple(
        sorted(bundle["inventory"]["records"], key=lambda row: row["sample_token"])
    )
    usable = tuple(
        row
        for row in records
        if row["crop_artifacts_present"] and row["full_status"] in {"USABLE", "REVIEW"}
    )
    datasets = [
        _build_mpdd_panel(usable),
        _build_sibetan_panel(usable),
        _build_generated_panel(usable),
    ]
    datasets.extend(
        _build_identity_none_panel(usable, name) for name in IDENTITY_NONE_DATASETS
    )
    required = sorted(
        {token for dataset in datasets for token in _dataset_sample_tokens(dataset)}
    )
    payload: dict[str, Any] = {
        "schema_version": PANEL_SCHEMA,
        "score_inputs_used": False,
        "selection_policy": {
            "ordering": "lexicographic_sample_token",
            "random_frame_splitting": False,
            "mpdd": "publisher_query_gallery_eval",
            "sibetan": "eval_only_deterministic_cross_sequence",
            "yt_bb": "generated_test_distinct_frame_research_diagnostic",
            "identity_none": "not_available_cache_v1_excludes_auxiliary_none",
        },
        "source_binding": {
            "inventory_bundle_sha256": bundle["bundle_sha256"],
            "inventory_sha256": bundle["inventory_sha256"],
            "split_manifest_sha256": bundle["split_manifest_sha256"],
            "split_census_sha256": bundle["split_census_sha256"],
            "baseline_family_sha256": bundle["baseline_family_sha256"],
        },
        "required_sample_tokens": required,
        "required_sample_tokens_sha256": content_sha256(required),
        "datasets": datasets,
    }
    return {**payload, "panel_sha256": content_sha256(payload)}


def validate_full128_evaluation_panel(
    panel: Mapping[str, Any], inventory_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    return _build_family_evidence(inventory_bundle, panel=panel).panel


def validate_full128_embedding_cache_descriptor(
    descriptor: Mapping[str, Any],
    *,
    variant_id: str,
    panel: Mapping[str, Any],
    inventory_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one descriptor without loading its external vector pack."""

    evidence = _build_family_evidence(inventory_bundle, panel=panel)
    return _validate_cache_descriptor_with_evidence(
        descriptor, variant_id=variant_id, evidence=evidence
    )


def _validate_cache_descriptor_with_evidence(
    descriptor: Mapping[str, Any],
    *,
    variant_id: str,
    evidence: _Full128FamilyEvidence,
) -> dict[str, Any]:
    value = _validate_cache_descriptor_shape(descriptor)
    if variant_id not in VARIANT_IDS or value["variant_id"] != variant_id:
        raise Full128EvaluationError("cache descriptor variant differs")
    bundle = evidence.bundle
    validated_panel = evidence.panel
    expected = {
        "baseline_family_sha256": bundle["baseline_family_sha256"],
        "inventory_bundle_sha256": bundle["bundle_sha256"],
        "inventory_sha256": bundle["inventory_sha256"],
        "split_manifest_sha256": bundle["split_manifest_sha256"],
        "split_census_sha256": bundle["split_census_sha256"],
        "panel_sha256": validated_panel["panel_sha256"],
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise Full128EvaluationError(f"cache descriptor {field} differs")
    variant = evidence.variant_manifests[variant_id]
    if value["variant_manifest_sha256"] != content_sha256(variant):
        raise Full128EvaluationError("cache descriptor variant manifest differs")
    required = validated_panel["required_sample_tokens"]
    samples = value["sample_tokens"]
    if required != samples:
        raise Full128EvaluationError("cache descriptor sample panel differs")
    if not set(samples) <= evidence.inventory_sample_tokens:
        raise Full128EvaluationError(
            "cache descriptor contains samples outside inventory v2"
        )
    return value


def build_full128_gallery_embedding_contract(
    descriptor: Mapping[str, Any], panel: Mapping[str, Any]
) -> dict[str, Any]:
    """Return one exact, single-channel, variant-bound gallery contract."""

    value = _validate_cache_descriptor_shape(descriptor)
    if panel.get("panel_sha256") != value["panel_sha256"]:
        raise Full128EvaluationError("gallery panel and cache descriptor differ")
    binding_fields = (
        "variant_id",
        "baseline_family_sha256",
        "variant_manifest_sha256",
        "variant_artifact_sha256",
        "training_run_sha256",
        "checkpoint_manifest_file_sha256",
        "checkpoint_manifest_sha256",
        "preprocessing_manifest_file_sha256",
        "preprocessing_manifest_sha256",
        "embedding_manifest_file_sha256",
        "embedding_manifest_sha256",
        "embedding_cache_manifest_file_sha256",
        "embedding_cache_manifest_sha256",
        "inventory_bundle_sha256",
        "inventory_sha256",
        "split_manifest_sha256",
        "split_census_sha256",
        "panel_sha256",
        "cache_descriptor_sha256",
    )
    return {
        "schema_version": "gallery.embedding_contract.v1",
        "kind": "FULL128_VARIANT_BOUND",
        "dimension": EMBEDDING_DIMENSION,
        "dtype": "float32",
        "normalization": "L2",
        "channels": [
            {
                "name": FULL128_CHANNEL,
                "dimension": EMBEDDING_DIMENSION,
                "optional": False,
            }
        ],
        "fusion": {"type": SCORER_ALGORITHM, "weights": [1.0], "exact": True},
        "full128_binding": {field: value[field] for field in binding_fields},
    }


def evaluate_full128_variant(
    *,
    inventory_bundle: Mapping[str, Any],
    panel: Mapping[str, Any],
    adapter: Full128EmbeddingCacheAdapter,
    gallery_root: Path,
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 0,
) -> ImmutableFull128EvaluationReport:
    """Evaluate one variant, persisting and reopening every canonical gallery."""

    _validate_bootstrap(bootstrap_resamples, bootstrap_seed)
    if not isinstance(adapter, Full128EmbeddingCacheAdapter):
        raise TypeError("adapter must implement Full128EmbeddingCacheAdapter")
    evidence = _reusable_adapter_evidence(
        adapter, inventory_bundle=inventory_bundle, panel=panel
    )
    if evidence is None:
        evidence = _build_family_evidence(inventory_bundle, panel=panel)
    return _evaluate_full128_variant_with_evidence(
        evidence=evidence,
        adapter=adapter,
        gallery_root=gallery_root,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def _evaluate_full128_variant_with_evidence(
    *,
    evidence: _Full128FamilyEvidence,
    adapter: Full128EmbeddingCacheAdapter,
    gallery_root: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> ImmutableFull128EvaluationReport:
    descriptor_shape = _validate_cache_descriptor_shape(adapter.descriptor)
    variant_id = descriptor_shape["variant_id"]
    descriptor = _validate_cache_descriptor_with_evidence(
        descriptor_shape,
        variant_id=variant_id,
        evidence=evidence,
    )
    validated_panel = evidence.panel
    root = _new_gallery_root(gallery_root)

    sample_tokens = validated_panel["required_sample_tokens"]
    packed_matrix = adapter.load_embeddings(sample_tokens)
    _validate_loaded_matrix(packed_matrix, len(sample_tokens))
    norms = np.linalg.norm(packed_matrix, axis=1, keepdims=True)
    matrix = np.asarray(packed_matrix / norms, dtype=np.float32)
    _validate_loaded_matrix(matrix, len(sample_tokens))
    embeddings = {token: matrix[index] for index, token in enumerate(sample_tokens)}
    contract = build_full128_gallery_embedding_contract(descriptor, validated_panel)

    dataset_results: list[dict[str, Any]] = []
    gallery_artifacts: list[dict[str, Any]] = []
    for dataset in validated_panel["datasets"]:
        result, artifacts = _evaluate_dataset(
            dataset=dataset,
            embeddings=embeddings,
            descriptor=descriptor,
            contract=contract,
            gallery_root=root,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        dataset_results.append(result)
        gallery_artifacts.extend(artifacts)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "variant_binding": {
            field: descriptor[field]
            for field in (
                "variant_id",
                "baseline_family_sha256",
                "variant_manifest_sha256",
                "variant_artifact_sha256",
                "training_run_sha256",
                "checkpoint_manifest_file_sha256",
                "checkpoint_manifest_sha256",
                "preprocessing_manifest_file_sha256",
                "preprocessing_manifest_sha256",
                "embedding_manifest_file_sha256",
                "embedding_manifest_sha256",
                "embedding_cache_manifest_file_sha256",
                "embedding_cache_manifest_sha256",
                "cache_descriptor_sha256",
            )
        },
        "source_binding": dict(validated_panel["source_binding"]),
        "panel_sha256": validated_panel["panel_sha256"],
        "scorer": {
            "algorithm": SCORER_ALGORITHM,
            "exact": True,
            "ann": False,
            "open_set": False,
            "channel": FULL128_CHANNEL,
            "dimension": EMBEDDING_DIMENSION,
            "dtype": "float32",
            "normalization": "L2",
        },
        "datasets": dataset_results,
        "pooled": {
            "status": "NOT_AVAILABLE",
            "reason": (
                "MPDD publisher retrieval and constructed Sibetan cross-sequence "
                "retrieval are heterogeneous protocols; generated and identity-free "
                "diagnostics cannot be pooled with canonical registered identity metrics"
            ),
            "metrics": None,
        },
        "gallery_artifacts": sorted(
            gallery_artifacts, key=lambda item: (item["dataset"], item["enrollment_k"])
        ),
        "limitations": [
            "YT-BB test labels are GENERATED video-track diagnostics, not canonical biometric identities.",
            "Identity-NONE datasets do not support biometric identity metrics.",
            "No open-set decision, ANN search, calibration, or deployment claim is evaluated.",
            "MPDD filename-derived capture/sequence tokens are unverified pose/view groups; publisher query/gallery evaluation does not claim cross-session independence.",
        ],
    }
    _validate_report_content(report)
    canonical = canonical_json_bytes(report)
    return ImmutableFull128EvaluationReport(
        canonical, hashlib.sha256(canonical).hexdigest()
    )


def evaluate_full128_family(
    *,
    inventory_bundle: Mapping[str, Any],
    adapters: Sequence[Full128EmbeddingCacheAdapter],
    gallery_root: Path,
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 0,
) -> tuple[
    dict[str, Any], tuple[ImmutableFull128EvaluationReport, ...], dict[str, Any]
]:
    """Build one score-blind panel and evaluate exactly B0, B1, and B2."""

    by_variant: dict[str, Full128EmbeddingCacheAdapter] = {}
    for adapter in adapters:
        if not isinstance(adapter, Full128EmbeddingCacheAdapter):
            raise TypeError("every cache adapter must implement the strict interface")
        variant_id = adapter.descriptor.get("variant_id")
        if variant_id in by_variant:
            raise Full128EvaluationError("Full128 family repeats a variant cache")
        if isinstance(variant_id, str):
            by_variant[variant_id] = adapter
    if tuple(sorted(by_variant)) != VARIANT_IDS:
        raise Full128EvaluationError(
            "Full128 family requires exactly B0, B1, and B2 caches"
        )
    packed_evidence = {
        id(adapter._family_evidence): adapter._family_evidence
        for adapter in by_variant.values()
        if isinstance(adapter, PackedFull128EmbeddingCacheAdapter)
        and adapter._family_evidence is not None
    }
    evidence = (
        next(iter(packed_evidence.values())) if len(packed_evidence) == 1 else None
    )
    if evidence is None or content_sha256(inventory_bundle) != content_sha256(
        evidence.bundle
    ):
        evidence = _build_family_evidence(inventory_bundle)
    panel = evidence.panel
    root = _new_gallery_root(gallery_root)
    reports = tuple(
        _evaluate_full128_variant_with_evidence(
            evidence=evidence,
            adapter=by_variant[variant],
            gallery_root=root / variant,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        for variant in VARIANT_IDS
    )
    _validate_common_denominators(reports)
    return panel, reports, build_full128_master_table(reports)


def build_full128_master_table(
    reports: Sequence[ImmutableFull128EvaluationReport | Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a compact deterministic JSON table from three sealed reports."""

    sealed = tuple(
        item
        if isinstance(item, ImmutableFull128EvaluationReport)
        else ImmutableFull128EvaluationReport.from_dict(item)
        for item in reports
    )
    if (
        tuple(sorted(item.report["variant_binding"]["variant_id"] for item in sealed))
        != VARIANT_IDS
    ):
        raise Full128EvaluationError("master table requires exactly B0, B1, and B2")
    _validate_common_denominators(sealed)
    rows: list[dict[str, Any]] = []
    for sealed_report in sealed:
        report = sealed_report.report
        variant = report["variant_binding"]["variant_id"]
        for dataset in report["datasets"]:
            for lane in ("identity_metrics", "diagnostic"):
                section = dataset[lane]
                if section["status"] != "AVAILABLE":
                    rows.append(
                        {
                            "variant_id": variant,
                            "dataset": dataset["dataset"],
                            "lane": lane,
                            "protocol_label": dataset["protocol_label"],
                            "enrollment_k": None,
                            "metric": None,
                            "status": section["status"],
                            "value": None,
                            "denominator": 0,
                            "lower_bound": None,
                            "upper_bound": None,
                            "reason": section["reason"],
                        }
                    )
                    continue
                result_sets = section.get("by_enrollment_k") or [section["result"]]
                for result in result_sets:
                    for metric_name in _METRIC_NAMES:
                        metric = result["metrics"][metric_name]
                        ci = metric["confidence_interval"]
                        rows.append(
                            {
                                "variant_id": variant,
                                "dataset": dataset["dataset"],
                                "lane": lane,
                                "protocol_label": dataset["protocol_label"],
                                "enrollment_k": result.get("enrollment_k"),
                                "metric": metric_name,
                                "status": metric["status"],
                                "value": metric["value"],
                                "denominator": metric["denominator"],
                                "lower_bound": ci.get("lower_bound"),
                                "upper_bound": ci.get("upper_bound"),
                                "reason": metric["reason"],
                            }
                        )
    rows.sort(
        key=lambda row: (
            row["variant_id"],
            row["dataset"],
            row["lane"],
            -1 if row["enrollment_k"] is None else row["enrollment_k"],
            str(row["metric"]),
        )
    )
    payload = {
        "schema_version": MASTER_TABLE_SCHEMA,
        "panel_sha256": sealed[0].report["panel_sha256"],
        "source_report_sha256s": sorted(item.report_sha256 for item in sealed),
        "columns": [
            "variant_id",
            "dataset",
            "lane",
            "protocol_label",
            "enrollment_k",
            "metric",
            "status",
            "value",
            "denominator",
            "lower_bound",
            "upper_bound",
            "reason",
        ],
        "rows": rows,
    }
    return {**payload, "table_sha256": content_sha256(payload)}


def full128_master_table_csv(table: Mapping[str, Any]) -> str:
    """Serialize the compact master table as deterministic RFC-4180 CSV."""

    if table.get("schema_version") != MASTER_TABLE_SCHEMA:
        raise Full128EvaluationError("Full128 master table schema differs")
    payload = {key: value for key, value in table.items() if key != "table_sha256"}
    if table.get("table_sha256") != content_sha256(payload):
        raise Full128EvaluationError("Full128 master table digest differs")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=table["columns"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(table["rows"])
    return output.getvalue()


def build_full128_family_index(
    reports: Sequence[ImmutableFull128EvaluationReport], table: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_common_denominators(reports)
    csv_sha256 = hashlib.sha256(
        full128_master_table_csv(table).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": FAMILY_INDEX_SCHEMA,
        "panel_sha256": reports[0].report["panel_sha256"],
        "inventory_sha256": reports[0].report["source_binding"]["inventory_sha256"],
        "split_manifest_sha256": reports[0].report["source_binding"][
            "split_manifest_sha256"
        ],
        "split_census_sha256": reports[0].report["source_binding"][
            "split_census_sha256"
        ],
        "exact_scorer_algorithm": SCORER_ALGORITHM,
        "reports": [
            {
                "variant_id": report.report["variant_binding"]["variant_id"],
                "relative_path": f"report-{report.report['variant_binding']['variant_id']}.json",
                "report_sha256": report.report_sha256,
            }
            for report in reports
        ],
        "master_table_json": "master-table.json",
        "master_table_csv": "master-table.csv",
        "master_table_sha256": table["table_sha256"],
        "master_table_csv_sha256": csv_sha256,
    }
    return {**payload, "index_sha256": content_sha256(payload)}


def _build_mpdd_panel(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in records if row["dataset_name"] == "mpdd"]
    protocol_rows = [
        row
        for row in rows
        if _official_role(row["official_split"]) in {"query", "gallery"}
    ]
    if any(row["terminal_role"] != "EVAL" for row in protocol_rows):
        raise Full128EvaluationError(
            "MPDD publisher query/gallery rows must remain EVAL"
        )
    query = [
        row for row in protocol_rows if _official_role(row["official_split"]) == "query"
    ]
    gallery = [
        row
        for row in protocol_rows
        if _official_role(row["official_split"]) == "gallery"
    ]
    if not query or not gallery:
        return _unavailable_dataset(
            "mpdd",
            "REGISTERED",
            "MPDD_OFFICIAL_QUERY_GALLERY_EVAL",
            "publisher query/gallery EVAL is unavailable",
        )
    _require_identity_kind(query + gallery, "REGISTERED", "MPDD")
    _reject_leakage(
        query,
        gallery,
        "MPDD",
        allow_capture_sequence_overlap=True,
    )
    gallery_by_identity = _group_by_identity(gallery)
    missing = sorted(
        {row["identity_token"] for row in query} - set(gallery_by_identity)
    )
    if missing:
        raise Full128EvaluationError(
            "MPDD closed-set query identity lacks publisher gallery"
        )
    selected = {
        str(k): [
            row
            for identity in sorted(gallery_by_identity)
            for row in gallery_by_identity[identity][
                : min(k, len(gallery_by_identity[identity]))
            ]
        ]
        for k in ENROLLMENT_KS
    }
    result = _canonical_dataset_panel(
        dataset="mpdd",
        protocol_label="MPDD_OFFICIAL_QUERY_GALLERY_EVAL",
        query=query,
        gallery_by_k=selected,
        candidate_count=len(rows),
    )
    result["independence_policy"] = (
        "publisher query/gallery roles; distinct sample, exact-source duplicate "
        "component, and effective content; unverified filename-derived "
        "capture/sequence groups may overlap"
    )
    return result


def _build_sibetan_panel(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for row in records
        if row["dataset_name"] == "sibetan" and row["terminal_role"] == "EVAL"
    ]
    if not rows:
        return _unavailable_dataset(
            "sibetan",
            "REGISTERED",
            "SIBETAN_EVAL_CROSS_SEQUENCE",
            "Sibetan EVAL does not exist; DEV/CAL were not repurposed",
        )
    _require_identity_kind(rows, "REGISTERED", "Sibetan")
    query: list[Mapping[str, Any]] = []
    gallery_by_identity: dict[str, list[Mapping[str, Any]]] = {}
    excluded_identities = 0
    for identity, identity_rows in sorted(_group_by_identity(rows).items()):
        by_sequence: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in identity_rows:
            by_sequence[row["sequence_group"]].append(row)
        if len(by_sequence) < 2:
            excluded_identities += 1
            continue
        sequence_order = sorted(
            by_sequence,
            key=lambda value: (
                min(row["sample_token"] for row in by_sequence[value]),
                value,
            ),
        )
        gallery_sequence = sequence_order[0]
        gallery_by_identity[identity] = sorted(
            by_sequence[gallery_sequence], key=lambda row: row["sample_token"]
        )
        query.extend(
            row
            for sequence in sequence_order[1:]
            for row in sorted(
                by_sequence[sequence], key=lambda item: item["sample_token"]
            )
        )
    if not query:
        return _unavailable_dataset(
            "sibetan",
            "REGISTERED",
            "SIBETAN_EVAL_CROSS_SEQUENCE",
            "Sibetan EVAL has no identity with two independent sequences",
        )
    gallery = [row for values in gallery_by_identity.values() for row in values]
    _reject_leakage(query, gallery, "Sibetan")
    selected = {
        str(k): [
            row
            for identity in sorted(gallery_by_identity)
            for row in gallery_by_identity[identity][
                : min(k, len(gallery_by_identity[identity]))
            ]
        ]
        for k in ENROLLMENT_KS
    }
    result = _canonical_dataset_panel(
        dataset="sibetan",
        protocol_label="SIBETAN_EVAL_CROSS_SEQUENCE",
        query=query,
        gallery_by_k=selected,
        candidate_count=len(rows),
    )
    result["coverage"]["excluded_identity_count"] = excluded_identities
    return result


def _build_generated_panel(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for row in records
        if row["dataset_name"] == "yt-bb-dog"
        and row["terminal_role"] == "EVAL"
        and _official_role(row["official_split"]) == "test"
    ]
    if not rows:
        return _unavailable_dataset(
            "yt-bb-dog",
            "GENERATED",
            "GENERATED_IDENTITY_DIAGNOSTIC",
            "YT-BB GENERATED test EVAL is unavailable",
        )
    _require_identity_kind(rows, "GENERATED", "YT-BB")
    gallery: list[Mapping[str, Any]] = []
    query: list[Mapping[str, Any]] = []
    excluded = 0
    for identity_rows in _group_by_identity(rows).values():
        ordered = sorted(identity_rows, key=lambda row: row["sample_token"])
        if len(ordered) < 2:
            excluded += 1
            continue
        gallery.append(ordered[0])
        query.extend(
            row
            for row in ordered[1:]
            if row["duplicate_component"] != ordered[0]["duplicate_component"]
            and row["effective_source_sha256"] != ordered[0]["effective_source_sha256"]
        )
    if not query:
        return _unavailable_dataset(
            "yt-bb-dog",
            "GENERATED",
            "GENERATED_IDENTITY_DIAGNOSTIC",
            "no independent test frames remain after leakage exclusions",
        )
    _reject_leakage(query, gallery, "YT-BB", allow_capture_sequence_overlap=True)
    samples = _sample_snapshots(query + gallery)
    return {
        "dataset": "yt-bb-dog",
        "identity_evidence_kind": "GENERATED",
        "protocol_label": "GENERATED_IDENTITY_DIAGNOSTIC",
        "canonical_biometric_claim": False,
        "status": "AVAILABLE",
        "reason": None,
        "query_sample_tokens": sorted(row["sample_token"] for row in query),
        "gallery_sample_tokens_by_k": {},
        "diagnostic_gallery_sample_tokens": sorted(
            row["sample_token"] for row in gallery
        ),
        "samples": samples,
        "independence_policy": "distinct sample/content/duplicate; track overlap explicitly allowed",
        "coverage": {
            "candidate_count": len(rows),
            "query_count": len(query),
            "excluded_count": len(rows) - len(query) - len(gallery),
            "excluded_identity_count": excluded,
        },
    }


def _build_identity_none_panel(
    records: Sequence[Mapping[str, Any]], dataset: str
) -> dict[str, Any]:
    rows = [row for row in records if row["dataset_name"] == dataset]
    _require_identity_kind(rows, "NONE", dataset)
    result = _unavailable_dataset(
        dataset,
        "NONE",
        "INSTANCE_INVARIANCE_RETRIEVAL",
        (
            "identity metrics are NOT_APPLICABLE and the Full128 cache v1 schema "
            "excludes AUXILIARY identity-NONE samples"
        ),
    )
    result["coverage"]["candidate_count"] = len(rows)
    return result


def _canonical_dataset_panel(
    *,
    dataset: str,
    protocol_label: str,
    query: Sequence[Mapping[str, Any]],
    gallery_by_k: Mapping[str, list[Mapping[str, Any]]],
    candidate_count: int,
) -> dict[str, Any]:
    gallery_rows = [row for values in gallery_by_k.values() for row in values]
    gallery_tokens = {row["sample_token"] for row in gallery_rows}
    return {
        "dataset": dataset,
        "identity_evidence_kind": "REGISTERED",
        "protocol_label": protocol_label,
        "canonical_biometric_claim": True,
        "status": "AVAILABLE",
        "reason": None,
        "query_sample_tokens": sorted(row["sample_token"] for row in query),
        "gallery_sample_tokens_by_k": {
            key: sorted(row["sample_token"] for row in value)
            for key, value in gallery_by_k.items()
        },
        "diagnostic_gallery_sample_tokens": [],
        "samples": _sample_snapshots(list(query) + gallery_rows),
        "independence_policy": "query/gallery duplicate, capture, sequence, and content disjoint",
        "coverage": {
            "candidate_count": candidate_count,
            "query_count": len(query),
            "excluded_count": candidate_count - len(query) - len(gallery_tokens),
            "excluded_identity_count": 0,
        },
    }


def _evaluate_dataset(
    *,
    dataset: Mapping[str, Any],
    embeddings: Mapping[str, np.ndarray],
    descriptor: Mapping[str, Any],
    contract: Mapping[str, Any],
    gallery_root: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity_kind = dataset["identity_evidence_kind"]
    base = {
        "dataset": dataset["dataset"],
        "identity_evidence_kind": identity_kind,
        "protocol_label": dataset["protocol_label"],
        "canonical_biometric_claim": dataset["canonical_biometric_claim"],
        "coverage": dataset["coverage"],
    }
    if dataset["status"] != "AVAILABLE":
        return (
            {
                **base,
                "identity_metrics": {
                    "status": "NOT_APPLICABLE"
                    if identity_kind == "NONE"
                    else "NOT_AVAILABLE",
                    "reason": dataset["reason"],
                    "by_enrollment_k": [],
                },
                "diagnostic": {
                    "status": "NOT_AVAILABLE",
                    "reason": dataset["reason"],
                    "result": None,
                },
            },
            [],
        )
    sample_map = {item["sample_token"]: item for item in dataset["samples"]}
    if identity_kind == "REGISTERED":
        results: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        for enrollment_k in ENROLLMENT_KS:
            result, artifact = _evaluate_canonical_gallery(
                dataset=dataset,
                sample_map=sample_map,
                embeddings=embeddings,
                descriptor=descriptor,
                contract=contract,
                gallery_directory=gallery_root
                / f"{dataset['dataset']}-K{enrollment_k}",
                enrollment_k=enrollment_k,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            )
            results.append(result)
            artifacts.append(artifact)
        return (
            {
                **base,
                "identity_metrics": {
                    "status": "AVAILABLE",
                    "reason": None,
                    "by_enrollment_k": results,
                },
                "diagnostic": {
                    "status": "NOT_APPLICABLE",
                    "reason": "canonical registered protocol uses IdentityGallery",
                    "result": None,
                },
            },
            artifacts,
        )
    diagnostic = _evaluate_matrix_diagnostic(
        query_tokens=dataset["query_sample_tokens"],
        gallery_tokens=dataset["diagnostic_gallery_sample_tokens"],
        sample_map=sample_map,
        embeddings=embeddings,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_valid=identity_kind == "GENERATED",
    )
    return (
        {
            **base,
            "identity_metrics": {
                "status": "NOT_APPLICABLE",
                "reason": (
                    "GENERATED track labels are not canonical registered identities"
                    if identity_kind == "GENERATED"
                    else "dataset identity evidence kind is NONE"
                ),
                "by_enrollment_k": [],
            },
            "diagnostic": {"status": "AVAILABLE", "reason": None, "result": diagnostic},
        },
        [],
    )


def _evaluate_canonical_gallery(
    *,
    dataset: Mapping[str, Any],
    sample_map: Mapping[str, Mapping[str, Any]],
    embeddings: Mapping[str, np.ndarray],
    descriptor: Mapping[str, Any],
    contract: Mapping[str, Any],
    gallery_directory: Path,
    enrollment_k: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from enrollment.binding.policy import IdentityRegistryPolicy
    from gallery.store.gallery import (
        GalleryEnrollment,
        IdentityGallery,
    )
    from search.scoring.roles import EnrollmentRank

    gallery_tokens = dataset["gallery_sample_tokens_by_k"][str(enrollment_k)]
    identities = frozenset(
        sample_map[token]["relevance_token"] for token in gallery_tokens
    )
    rank = EnrollmentRank(f"K{enrollment_k}")
    enrollments = []
    for token in gallery_tokens:
        sample = sample_map[token]
        vector = embeddings[token]
        digest = hashlib.sha256()
        digest.update(descriptor["cache_descriptor_sha256"].encode("ascii"))
        digest.update(token.encode("utf-8"))
        digest.update(vector.astype("<f4", copy=False).tobytes())
        enrollments.append(
            GalleryEnrollment(
                embedding=vector,
                registered_identity_id=sample["relevance_token"],
                metadata={"sample_token": token, "dataset": dataset["dataset"]},
                idempotency_key=f"{dataset['dataset']}:K{enrollment_k}:{token}",
                content_sha256=digest.hexdigest(),
                enrollment_rank=rank,
                enrollment_view=token,
                duplicate_group_ids=(sample["duplicate_component"],),
            )
        )
    policy = IdentityRegistryPolicy(registered_identity_ids=identities)
    gallery = IdentityGallery.build(
        gallery_directory,
        enrollments,
        dim=EMBEDDING_DIMENSION,
        embedding_contract=dict(contract),
        registry_policy=policy,
    )
    try:
        query_rows: list[dict[str, Any]] = []
        for token in dataset["query_sample_tokens"]:
            expected = sample_map[token]["relevance_token"]
            relevant_rank = gallery.rank_of_identity(embeddings[token], expected)
            if relevant_rank is None:
                raise Full128EvaluationError(
                    "canonical query has no relevant identity in reopened gallery"
                )
            query_rows.append(_query_metric_row(token, expected, relevant_rank))
        scorer_hash = gallery.scorer_hash
    finally:
        gallery.close()
    metrics = _summarize_query_rows(
        query_rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_valid=True,
    )
    artifact = _gallery_artifact_binding(
        gallery_directory,
        dataset=dataset["dataset"],
        enrollment_k=enrollment_k,
        scorer_hash=scorer_hash,
    )
    return (
        {
            "enrollment_k": enrollment_k,
            "query_panel_sha256": content_sha256(dataset["query_sample_tokens"]),
            "query_count": len(query_rows),
            "gallery_template_count": len(gallery_tokens),
            "gallery_identity_count": len(identities),
            "ranking_unit": "registered_identity",
            "aggregation": "max_exact_template_cosine",
            "metrics": metrics,
            "query_rows": query_rows,
            "gallery_bytes_sha256": artifact["gallery_bytes_sha256"],
            "scorer_hash": scorer_hash,
        },
        artifact,
    )


def _evaluate_matrix_diagnostic(
    *,
    query_tokens: Sequence[str],
    gallery_tokens: Sequence[str],
    sample_map: Mapping[str, Mapping[str, Any]],
    embeddings: Mapping[str, np.ndarray],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    bootstrap_valid: bool,
) -> dict[str, Any]:
    query = np.stack([embeddings[token] for token in query_tokens]).astype(np.float32)
    gallery = np.stack([embeddings[token] for token in gallery_tokens]).astype(
        np.float32
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(gallery_tokens):
        grouped[sample_map[token]["relevance_token"]].append(index)
    identity_order = sorted(grouped)
    identity_index = {identity: index for index, identity in enumerate(identity_order)}
    expected_identities = [
        sample_map[token]["relevance_token"] for token in query_tokens
    ]
    if any(expected not in identity_index for expected in expected_identities):
        raise Full128EvaluationError("diagnostic query has no relevant gallery unit")
    query_rows: list[dict[str, Any]] = []
    query_block_rows = min(
        _DIAGNOSTIC_QUERY_BLOCK_ROWS,
        max(1, _DIAGNOSTIC_MAX_SCORE_ENTRIES // len(identity_order)),
    )
    for query_start in range(0, len(query_tokens), query_block_rows):
        query_stop = min(len(query_tokens), query_start + query_block_rows)
        query_block = query[query_start:query_stop]
        identity_scores = np.full(
            (len(query_block), len(identity_order)),
            -np.inf,
            dtype=np.float32,
        )
        template_block_rows = max(1, _DIAGNOSTIC_MAX_SCORE_ENTRIES // len(query_block))
        for ordinal, identity in enumerate(identity_order):
            template_indices = grouped[identity]
            for template_start in range(0, len(template_indices), template_block_rows):
                selected = template_indices[
                    template_start : template_start + template_block_rows
                ]
                block_maximum = np.max(query_block @ gallery[selected].T, axis=1)
                np.maximum(
                    identity_scores[:, ordinal],
                    block_maximum,
                    out=identity_scores[:, ordinal],
                )
        for local_index, row_index in enumerate(range(query_start, query_stop)):
            token = query_tokens[row_index]
            expected = expected_identities[row_index]
            expected_ordinal = identity_index[expected]
            relevant_score = identity_scores[local_index, expected_ordinal]
            if not np.isfinite(relevant_score):
                raise Full128EvaluationError(
                    "diagnostic query has no relevant gallery unit"
                )
            relevant_rank = 1 + int(
                sum(
                    score > relevant_score
                    or (score == relevant_score and identity < expected)
                    for identity, score in zip(
                        identity_order, identity_scores[local_index], strict=True
                    )
                )
            )
            query_rows.append(_query_metric_row(token, expected, relevant_rank))
    return {
        "enrollment_k": None,
        "query_count": len(query_rows),
        "gallery_template_count": len(gallery_tokens),
        "gallery_identity_count": len(identity_order),
        "ranking_unit": "generated_track" if bootstrap_valid else "source_instance",
        "aggregation": "max_exact_cosine_research_matrix",
        "metrics": _summarize_query_rows(
            query_rows,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            bootstrap_valid=bootstrap_valid,
        ),
        "query_rows": query_rows,
        "gallery_bytes_sha256": None,
        "scorer_hash": content_sha256(
            {"algorithm": "exact_float32_cosine_matrix.v1", "aggregation": "max"}
        ),
    }


def _query_metric_row(sample_token: str, relevance: str, rank: int) -> dict[str, Any]:
    reciprocal = 1.0 / rank
    return {
        "sample_token": sample_token,
        "bootstrap_cluster_id": relevance,
        "relevant_rank": rank,
        "Rank-1": float(rank <= 1),
        "Rank-5": float(rank <= 5),
        "Rank-10": float(rank <= 10),
        "MRR": reciprocal,
        "mAP": reciprocal,
        "mINP": reciprocal,
    }


def _summarize_query_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    bootstrap_valid: bool,
) -> dict[str, Any]:
    cluster_count = len({row["bootstrap_cluster_id"] for row in rows})
    bootstrap_cis = (
        _clustered_bootstrap_cis(
            rows,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        if bootstrap_valid and cluster_count >= 2
        else None
    )
    output: dict[str, Any] = {}
    for metric_name in _METRIC_NAMES:
        values = [float(row[metric_name]) for row in rows]
        ci: dict[str, Any]
        if bootstrap_cis is not None:
            ci = {
                "status": "AVAILABLE",
                **bootstrap_cis[metric_name],
                "reason": None,
            }
        else:
            ci = {
                "status": "NOT_AVAILABLE",
                "lower_bound": None,
                "upper_bound": None,
                "cluster_count": cluster_count,
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
                "reason": (
                    "identity-clustered bootstrap is invalid for identity-NONE diagnostics"
                    if not bootstrap_valid
                    else "at least two relevance identities are required"
                ),
            }
        output[metric_name] = {
            "status": "AVAILABLE",
            "value": float(math.fsum(values) / len(values)),
            "numerator": float(math.fsum(values)),
            "denominator": len(values),
            "reason": None,
            "confidence_interval": ci,
        }
    return output


def _clustered_bootstrap_cis(
    rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["bootstrap_cluster_id"], []).append(row)
    cluster_rows = tuple(groups.values())
    cluster_sums = np.asarray(
        [
            [sum(float(row[metric]) for row in grouped) for metric in _METRIC_NAMES]
            for grouped in cluster_rows
        ],
        dtype=np.float64,
    )
    cluster_counts = np.asarray(
        [len(grouped) for grouped in cluster_rows], dtype=np.int64
    )
    sampled_estimates = np.empty((len(_METRIC_NAMES), resamples), dtype=np.float64)
    rng = np.random.default_rng(seed)
    cluster_count = len(cluster_rows)
    for start in range(0, resamples, _BOOTSTRAP_BLOCK_RESAMPLES):
        length = min(_BOOTSTRAP_BLOCK_RESAMPLES, resamples - start)
        sampled = rng.integers(0, cluster_count, size=(length, cluster_count))
        sampled_counts = cluster_counts[sampled].sum(axis=1)
        sampled_sums = cluster_sums.T[:, sampled].sum(axis=2)
        sampled_estimates[:, start : start + length] = (
            sampled_sums / sampled_counts[np.newaxis, :]
        )
    sampled_estimates.sort(axis=1)
    alpha = 0.025
    lower_index = max(0, math.ceil(alpha * resamples) - 1)
    upper_index = min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1)
    all_values = np.asarray(
        [[float(row[metric]) for metric in _METRIC_NAMES] for row in rows],
        dtype=np.float64,
    )
    return {
        metric: {
            "metric": metric,
            "estimate": float(np.mean(all_values[:, index])),
            "lower_bound": float(sampled_estimates[index, lower_index]),
            "upper_bound": float(sampled_estimates[index, upper_index]),
            "confidence_level": 0.95,
            "cluster_unit": "query_identity",
            "cluster_count": cluster_count,
            "query_row_count": len(rows),
            "resamples": resamples,
            "seed": seed,
            "interval_method": "whole_identity_percentile_bootstrap",
        }
        for index, metric in enumerate(_METRIC_NAMES)
    }


def _sample_snapshots(
    rows: Sequence[Mapping[str, Any]], *, relevance_field: str = "identity_token"
) -> list[dict[str, Any]]:
    by_token = {row["sample_token"]: row for row in rows}
    return [
        {
            "sample_token": token,
            "relevance_token": by_token[token][relevance_field],
            "source_group": by_token[token]["source_group"],
            "capture_group": by_token[token]["capture_group"],
            "sequence_group": by_token[token]["sequence_group"],
            "duplicate_component": by_token[token]["duplicate_component"],
            "content_sha256": by_token[token]["effective_source_sha256"],
        }
        for token in sorted(by_token)
    ]


def _unavailable_dataset(
    dataset: str, identity_kind: str, protocol_label: str, reason: str
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "identity_evidence_kind": identity_kind,
        "protocol_label": protocol_label,
        "canonical_biometric_claim": identity_kind == "REGISTERED",
        "status": "NOT_AVAILABLE",
        "reason": reason,
        "query_sample_tokens": [],
        "gallery_sample_tokens_by_k": {},
        "diagnostic_gallery_sample_tokens": [],
        "samples": [],
        "independence_policy": None,
        "coverage": {
            "candidate_count": 0,
            "query_count": 0,
            "excluded_count": 0,
            "excluded_identity_count": 0,
        },
    }


def _dataset_sample_tokens(dataset: Mapping[str, Any]) -> set[str]:
    result = set(dataset["query_sample_tokens"])
    result.update(dataset["diagnostic_gallery_sample_tokens"])
    for values in dataset["gallery_sample_tokens_by_k"].values():
        result.update(values)
    return result


def _reject_leakage(
    query: Sequence[Mapping[str, Any]],
    gallery: Sequence[Mapping[str, Any]],
    label: str,
    *,
    allow_capture_sequence_overlap: bool = False,
) -> None:
    fields = ["sample_token", "duplicate_component", "effective_source_sha256"]
    if not allow_capture_sequence_overlap:
        fields.extend(("capture_group", "sequence_group"))
    for field in fields:
        overlap = {row[field] for row in query} & {row[field] for row in gallery}
        if overlap:
            raise Full128EvaluationError(
                f"{label} query/gallery {field} leakage detected"
            )


def _require_identity_kind(
    rows: Sequence[Mapping[str, Any]], expected: str, label: str
) -> None:
    if any(row["identity_evidence_kind"] != expected for row in rows):
        raise Full128EvaluationError(f"{label} identity evidence kind differs")
    if expected == "NONE" and any(row["identity_token"] is not None for row in rows):
        raise Full128EvaluationError(f"{label} identity-NONE row carries an identity")
    if expected != "NONE" and any(
        not isinstance(row["identity_token"], str) for row in rows
    ):
        raise Full128EvaluationError(f"{label} identity row lacks a canonical token")


def _group_by_identity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped = _group_rows(rows, "identity_token")
    return {
        str(key): sorted(values, key=lambda row: row["sample_token"])
        for key, values in grouped.items()
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[Any, list[Mapping[str, Any]]]:
    result: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[row[field]].append(row)
    return dict(result)


def _official_role(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("_", "-")
    for role in ("query", "gallery", "test"):
        if normalized in {role, f"official-{role}", f"publisher-{role}"}:
            return role
    return None


def _validate_inventory_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "artifact_root",
        "content_kind",
        "source_registry_admission_state",
        "source_registry_admission_sha256",
        "baseline_family_manifest",
        "baseline_family_sha256",
        "split_manifest_sha256",
        "split_census_sha256",
        "split_bundle",
        "inventory_sha256",
        "inventory",
        "bundle_sha256",
    }
    _exact_keys(value, expected, "Full128 inventory bundle")
    bundle = dict(value)
    if (
        bundle["schema_version"] != INVENTORY_BUNDLE_SCHEMA
        or bundle["content_kind"] != "METADATA_ONLY"
    ):
        raise Full128EvaluationError(
            "Full128 inventory bundle schema or content kind differs"
        )
    payload = {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    if content_sha256(payload) != bundle["bundle_sha256"]:
        raise Full128EvaluationError("Full128 inventory bundle digest differs")
    inventory = bundle["inventory"]
    if (
        not isinstance(inventory, Mapping)
        or set(inventory) != {"schema_version", "records"}
        or inventory["schema_version"] != INVENTORY_SCHEMA
        or not isinstance(inventory["records"], list)
        or not inventory["records"]
    ):
        raise Full128EvaluationError("Full128 inventory v2 differs")
    if content_sha256(inventory) != bundle["inventory_sha256"]:
        raise Full128EvaluationError("Full128 inventory digest differs")
    samples = [
        row.get("sample_token")
        for row in inventory["records"]
        if isinstance(row, Mapping)
    ]
    if len(samples) != len(inventory["records"]) or len(samples) != len(set(samples)):
        raise Full128EvaluationError("Full128 inventory sample tokens differ")
    split_manifest, split_census = validate_unified_full_split_bundle(
        bundle["split_bundle"]
    )
    if (
        split_manifest.manifest_sha256 != bundle["split_manifest_sha256"]
        or split_census.census_sha256 != bundle["split_census_sha256"]
    ):
        raise Full128EvaluationError("Full128 split/census binding differs")
    family = build_baseline_family_manifest()
    if bundle["baseline_family_manifest"] != family or bundle[
        "baseline_family_sha256"
    ] != content_sha256(family):
        raise Full128EvaluationError("Full128 baseline family binding differs")
    return bundle


def _build_family_evidence(
    inventory_bundle: Mapping[str, Any],
    *,
    panel: Mapping[str, Any] | None = None,
) -> _Full128FamilyEvidence:
    bundle = _validate_inventory_metadata(inventory_bundle)
    rebuilt_panel = _build_panel_from_validated_bundle(bundle)
    if panel is not None and dict(panel) != rebuilt_panel:
        raise Full128EvaluationError("Full128 evaluation panel content differs")
    variant_manifests = {
        item["variant_id"]: item
        for item in bundle["baseline_family_manifest"]["variants"]
    }
    if tuple(sorted(variant_manifests)) != VARIANT_IDS:
        raise Full128EvaluationError("Full128 baseline family variants differ")
    return _Full128FamilyEvidence(
        bundle=bundle,
        panel=rebuilt_panel,
        inventory_sample_tokens=frozenset(
            row["sample_token"] for row in bundle["inventory"]["records"]
        ),
        variant_manifests=variant_manifests,
    )


def _reusable_adapter_evidence(
    adapter: Full128EmbeddingCacheAdapter,
    *,
    inventory_bundle: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> _Full128FamilyEvidence | None:
    if not isinstance(adapter, PackedFull128EmbeddingCacheAdapter):
        return None
    evidence = adapter._family_evidence
    if evidence is None:
        return None
    if (
        content_sha256(inventory_bundle) != content_sha256(evidence.bundle)
        or panel.get("panel_sha256") != evidence.panel["panel_sha256"]
        or dict(panel) != evidence.panel
    ):
        return None
    return evidence


def _validate_training_run_manifest(
    value: object, *, inventory_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "run_config",
        "bindings",
        "source_closure",
        "runtime_versions",
        "run_manifest_sha256",
    }
    _exact_keys(value, fields, "Full128 training run manifest")
    manifest = dict(value)
    payload = {
        key: item for key, item in manifest.items() if key != "run_manifest_sha256"
    }
    if manifest["schema_version"] != "archive.full128.training_run.v1" or manifest[
        "run_manifest_sha256"
    ] != content_sha256(payload):
        raise Full128EvaluationError("Full128 training run manifest differs")
    bindings = manifest["bindings"]
    expected_fields = {
        "assembly_sha256",
        "inventory_bundle_sha256",
        "inventory_sha256",
        "split_manifest_sha256",
        "split_census_sha256",
        "baseline_family_sha256",
        "family_manifest_sha256",
        "run_config_sha256",
        "source_closure_sha256",
        "uv_lock",
    }
    _exact_keys(bindings, expected_fields, "Full128 training run bindings")
    for field in expected_fields - {"uv_lock"}:
        _require_sha256(bindings[field], field)
    _exact_keys(bindings["uv_lock"], {"sha256", "byte_size"}, "uv.lock binding")
    _require_sha256(bindings["uv_lock"]["sha256"], "uv.lock sha256")
    if (
        isinstance(bindings["uv_lock"]["byte_size"], bool)
        or not isinstance(bindings["uv_lock"]["byte_size"], int)
        or bindings["uv_lock"]["byte_size"] <= 0
        or bindings["run_config_sha256"] != content_sha256(manifest["run_config"])
        or bindings["source_closure_sha256"]
        != content_sha256(manifest["source_closure"])
    ):
        raise Full128EvaluationError("Full128 training run content binding differs")
    bundle = inventory_bundle
    expected = {
        "inventory_bundle_sha256": bundle["bundle_sha256"],
        "inventory_sha256": bundle["inventory_sha256"],
        "split_manifest_sha256": bundle["split_manifest_sha256"],
        "split_census_sha256": bundle["split_census_sha256"],
        "baseline_family_sha256": bundle["baseline_family_sha256"],
        "family_manifest_sha256": bundle["baseline_family_sha256"],
    }
    for field, expected_value in expected.items():
        if bindings[field] != expected_value:
            raise Full128EvaluationError(f"training run {field} differs")
    return manifest


def _validate_variant_training_bindings(
    variant_run: Mapping[str, Any],
    training_run: Mapping[str, Any],
    inventory_bundle: Mapping[str, Any],
) -> None:
    bindings = variant_run["bindings"]
    expected = {
        "run_manifest_sha256": training_run["run_manifest_sha256"],
        **training_run["bindings"],
    }
    if variant_run["variant_id"] == "B2":
        if set(bindings) != {*expected, "b2_initialization"}:
            raise Full128EvaluationError("B2 training variant bindings differ")
    elif set(bindings) != set(expected):
        raise Full128EvaluationError("training variant bindings differ")
    for field, expected_value in expected.items():
        if bindings[field] != expected_value:
            raise Full128EvaluationError(f"training variant {field} differs")
    if bindings["inventory_bundle_sha256"] != inventory_bundle["bundle_sha256"]:
        raise Full128EvaluationError("training variant inventory binding differs")


def _validate_training_family_run(
    value: object, training_run: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "family_id",
        "run_manifest_sha256",
        "run_config_sha256",
        "family_manifest_sha256",
        "variants",
        "status",
        "family_run_sha256",
    }
    _exact_keys(value, fields, "Full128 family run")
    family = dict(value)
    payload = {key: item for key, item in family.items() if key != "family_run_sha256"}
    variants = family["variants"]
    if (
        family["schema_version"] != "archive.full128.family_run.v1"
        or family["family_id"] != "FULL128_B0_B1_B2"
        or family["status"] != "COMPLETE_EXACT_THREE_VARIANT_FAMILY"
        or family["family_run_sha256"] != content_sha256(payload)
        or not isinstance(variants, list)
        or [row.get("variant_id") for row in variants] != list(VARIANT_IDS)
    ):
        raise Full128EvaluationError("Full128 family run content differs")
    for row in variants:
        _exact_keys(row, {"variant_id", "variant_run_sha256"}, "family variant")
        _require_sha256(row["variant_run_sha256"], "variant_run_sha256")
    expected = {
        "run_manifest_sha256": training_run["run_manifest_sha256"],
        "run_config_sha256": training_run["bindings"]["run_config_sha256"],
        "family_manifest_sha256": training_run["bindings"]["family_manifest_sha256"],
    }
    for field, expected_value in expected.items():
        if family[field] != expected_value:
            raise Full128EvaluationError(f"family run {field} differs")
    return family


def _read_bound_json_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    from shared.foundation.protected_io import read_strict_json_document

    try:
        payload = read_strict_json_document(
            root / binding["relative_path"], maximum_bytes=1_073_741_824
        ).payload
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise Full128EvaluationError("training JSON artifact is invalid") from exc
    if not isinstance(payload, dict):
        raise Full128EvaluationError("training JSON artifact must be an object")
    return payload


def _validate_variant_run_for_evaluation(root: Path, value: object) -> dict[str, Any]:
    """Validate a variant while deferring its pack to the streaming reader."""

    fields = {
        "schema_version",
        "variant_id",
        "method",
        "initialization",
        "bindings",
        "fit_population",
        "training",
        "artifacts",
        "variant_run_sha256",
    }
    _exact_keys(value, fields, "Full128 variant run")
    manifest = dict(value)
    payload = {
        key: item for key, item in manifest.items() if key != "variant_run_sha256"
    }
    if manifest["schema_version"] != "archive.full128.variant_run.v1" or manifest[
        "variant_run_sha256"
    ] != content_sha256(payload):
        raise Full128EvaluationError("Full128 variant run manifest differs")
    artifacts = manifest["artifacts"]
    _exact_keys(
        artifacts,
        {
            "state",
            "model_manifest",
            "preprocessing_manifest",
            "embedding_manifest",
            "checkpoint_manifest",
            "embedding_cache_manifest",
        },
        "Full128 variant artifacts",
    )
    for name in (
        "state",
        "model_manifest",
        "preprocessing_manifest",
        "embedding_manifest",
        "checkpoint_manifest",
    ):
        binding = artifacts[name]
        _validate_file_binding(binding, f"Full128 {name}")
        if _hash_regular_file_binding(root / binding["relative_path"]) != {
            "sha256": binding["sha256"],
            "byte_size": binding["byte_size"],
        }:
            raise Full128EvaluationError(f"Full128 {name} binding differs")

    cache_binding = artifacts["embedding_cache_manifest"]
    _exact_keys(
        cache_binding,
        {"relative_path", "sha256", "byte_size", "manifest"},
        "Full128 embedding cache manifest binding",
    )
    _validate_file_binding(
        {key: cache_binding[key] for key in ("relative_path", "sha256", "byte_size")},
        "Full128 embedding cache manifest",
    )
    from shared.foundation.protected_io import read_strict_json_document

    document = read_strict_json_document(
        root / cache_binding["relative_path"], maximum_bytes=1_073_741_824
    )
    if (
        document.raw_sha256 != cache_binding["sha256"]
        or document.byte_size != cache_binding["byte_size"]
        or document.payload != cache_binding["manifest"]
    ):
        raise Full128EvaluationError("Full128 embedding cache manifest binding differs")
    _validate_embedding_cache_manifest_shape(cache_binding["manifest"])
    return manifest


def _validate_file_binding(value: object, label: str) -> None:
    _exact_keys(value, {"relative_path", "sha256", "byte_size"}, label)
    relative_path = value["relative_path"]
    if (
        not isinstance(relative_path, str)
        or Path(relative_path).name != relative_path
        or relative_path in {"", ".", ".."}
        or isinstance(value["byte_size"], bool)
        or not isinstance(value["byte_size"], int)
        or value["byte_size"] <= 0
    ):
        raise Full128EvaluationError(f"{label} contract differs")
    _require_sha256(value["sha256"], f"{label} sha256")


def _hash_regular_file_binding(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Full128EvaluationError(
            "training artifact cannot be opened safely"
        ) from exc
    digest = hashlib.sha256()
    observed = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise Full128EvaluationError("training artifact type or size differs")
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after) or observed != before.st_size:
        raise Full128EvaluationError("training artifact changed while reading")
    return {"sha256": digest.hexdigest(), "byte_size": observed}


def _validate_embedding_cache_manifest_shape(value: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "relative_path",
        "dtype",
        "dimension",
        "bytes_per_vector",
        "vector_count",
        "pack_byte_size",
        "pack_sha256",
        "vectors",
        "cache_manifest_sha256",
    }
    _exact_keys(value, fields, "Full128 embedding cache manifest")
    manifest = dict(value)
    payload = {
        key: item for key, item in manifest.items() if key != "cache_manifest_sha256"
    }
    relative_path = manifest["relative_path"]
    vector_count = manifest["vector_count"]
    pack_byte_size = manifest["pack_byte_size"]
    if (
        manifest["schema_version"] != "archive.full128.embedding_cache.v1"
        or manifest["cache_manifest_sha256"] != content_sha256(payload)
        or manifest["dtype"] != "float32_little_endian"
        or manifest["dimension"] != EMBEDDING_DIMENSION
        or manifest["bytes_per_vector"] != _VECTOR_BYTES
        or not isinstance(relative_path, str)
        or Path(relative_path).name != relative_path
        or relative_path in {"", ".", ".."}
        or isinstance(vector_count, bool)
        or not isinstance(vector_count, int)
        or vector_count <= 0
        or isinstance(pack_byte_size, bool)
        or not isinstance(pack_byte_size, int)
        or pack_byte_size != vector_count * _VECTOR_BYTES
        or pack_byte_size > _MAXIMUM_PACK_BYTES
    ):
        raise Full128EvaluationError("Full128 embedding cache contract differs")
    _require_sha256(manifest["pack_sha256"], "embedding pack sha256")
    vectors = manifest["vectors"]
    if not isinstance(vectors, list) or len(vectors) != vector_count:
        raise Full128EvaluationError("Full128 embedding vector count differs")
    seen: set[str] = set()
    row_fields = {
        "sample_id",
        "identity_id",
        "dataset_name",
        "view",
        "role",
        "crop_record_sha256",
        "offset_bytes",
        "byte_size",
        "sha256",
    }
    for index, row in enumerate(vectors):
        _exact_keys(row, row_fields, "Full128 embedding vector")
        sample_id = row["sample_id"]
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in seen
            or row["offset_bytes"] != index * _VECTOR_BYTES
            or row["byte_size"] != _VECTOR_BYTES
            or any(
                not isinstance(row[field], str) or not row[field]
                for field in ("identity_id", "dataset_name", "view", "role")
            )
        ):
            raise Full128EvaluationError("Full128 embedding vector contract differs")
        _require_sha256(row["crop_record_sha256"], "crop record sha256")
        _require_sha256(row["sha256"], "embedding vector sha256")
        seen.add(sample_id)
    return manifest


def _validate_cache_inventory_rows(
    cache_manifest: Mapping[str, Any], inventory_bundle: Mapping[str, Any]
) -> None:
    expected = []
    for record in inventory_bundle["inventory"]["records"]:
        if (
            record["full_status"] not in {"USABLE", "REVIEW"}
            or not record["crop_artifacts_present"]
            or record["identity_evidence_kind"] == "NONE"
            or record["identity_token"] is None
            or record["terminal_role"] not in {"FIT", "DEV", "CAL", "EVAL"}
        ):
            continue
        expected.append(
            {
                "sample_id": record["sample_token"],
                "identity_id": record["identity_token"],
                "dataset_name": record["dataset_name"],
                "view": (
                    "face"
                    if record["view_scope"] in {"FACE_NATIVE", "HEAD_NATIVE"}
                    else "body"
                ),
                "role": record["terminal_role"],
                "crop_record_sha256": record["crop_record_sha256"],
            }
        )
    expected.sort(key=lambda row: row["sample_id"])
    observed = [
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
        for row in cache_manifest["vectors"]
    ]
    if observed != expected:
        raise Full128EvaluationError("training cache inventory rows differ")


def _resolve_regular_directory(path: Path, label: str) -> Path:
    requested = path.absolute()
    if requested.is_symlink():
        raise Full128EvaluationError(f"{label} must not be a symlink")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise Full128EvaluationError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        raise Full128EvaluationError(f"{label} must be a directory")
    return resolved


def _validate_cache_descriptor_shape(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "variant_id",
        "baseline_family_sha256",
        "variant_manifest_sha256",
        "variant_artifact_sha256",
        "training_run_sha256",
        "checkpoint_manifest_file_sha256",
        "checkpoint_manifest_sha256",
        "preprocessing_manifest_file_sha256",
        "preprocessing_manifest_sha256",
        "embedding_manifest_file_sha256",
        "embedding_manifest_sha256",
        "embedding_cache_manifest_file_sha256",
        "embedding_cache_manifest_sha256",
        "inventory_bundle_sha256",
        "inventory_sha256",
        "split_manifest_sha256",
        "split_census_sha256",
        "panel_sha256",
        "embedding_dimension",
        "dtype",
        "normalization",
        "sample_tokens",
        "sample_tokens_sha256",
        "storage",
        "cache_descriptor_sha256",
    }
    _exact_keys(value, fields, "Full128 cache descriptor")
    descriptor = dict(value)
    if descriptor["schema_version"] != CACHE_DESCRIPTOR_SCHEMA:
        raise Full128EvaluationError("Full128 cache descriptor schema differs")
    if descriptor["variant_id"] not in VARIANT_IDS:
        raise Full128EvaluationError("Full128 cache descriptor variant differs")
    for field in fields:
        if field.endswith("_sha256"):
            _require_sha256(descriptor[field], field)
    if (
        descriptor["embedding_dimension"] != EMBEDDING_DIMENSION
        or descriptor["dtype"] != "float32"
        or descriptor["normalization"] != "L2"
    ):
        raise Full128EvaluationError("Full128 cache tensor contract differs")
    samples = descriptor["sample_tokens"]
    if (
        not isinstance(samples, list)
        or any(not isinstance(token, str) or not token for token in samples)
        or samples != sorted(samples)
        or len(samples) != len(set(samples))
        or descriptor["sample_tokens_sha256"] != content_sha256(samples)
    ):
        raise Full128EvaluationError("Full128 cache sample token index differs")
    storage = descriptor["storage"]
    _exact_keys(
        storage,
        {
            "format",
            "relative_path",
            "pack_sha256",
            "pack_byte_size",
            "source_vector_count",
            "vectors",
        },
        "cache storage",
    )
    if (
        storage["format"] != "PACKED_FLOAT32_LITTLE_ENDIAN"
        or not isinstance(storage["relative_path"], str)
        or Path(storage["relative_path"]).name != storage["relative_path"]
        or storage["relative_path"] in {"", ".", ".."}
        or not isinstance(storage["pack_byte_size"], int)
        or isinstance(storage["pack_byte_size"], bool)
        or not 0 < storage["pack_byte_size"] <= _MAXIMUM_PACK_BYTES
        or not isinstance(storage["source_vector_count"], int)
        or isinstance(storage["source_vector_count"], bool)
        or storage["source_vector_count"] <= 0
        or storage["source_vector_count"] < len(samples)
        or storage["source_vector_count"] * EMBEDDING_DIMENSION * 4
        != storage["pack_byte_size"]
    ):
        raise Full128EvaluationError("Full128 cache storage contract differs")
    _require_sha256(storage["pack_sha256"], "storage pack_sha256")
    vectors = storage["vectors"]
    if not isinstance(vectors, list) or len(vectors) != len(samples):
        raise Full128EvaluationError("Full128 cache vector index differs")
    offsets: set[int] = set()
    ordered_offsets: list[int] = []
    for index, row in enumerate(vectors):
        _exact_keys(
            row,
            {"sample_token", "offset_bytes", "byte_size", "sha256"},
            "cache vector",
        )
        if (
            row["sample_token"] != samples[index]
            or not isinstance(row["offset_bytes"], int)
            or isinstance(row["offset_bytes"], bool)
            or row["offset_bytes"] < 0
            or row["offset_bytes"] % (EMBEDDING_DIMENSION * 4) != 0
            or row["offset_bytes"] in offsets
            or row["offset_bytes"] // (EMBEDDING_DIMENSION * 4)
            >= storage["source_vector_count"]
            or row["byte_size"] != EMBEDDING_DIMENSION * 4
            or row["offset_bytes"] + row["byte_size"] > storage["pack_byte_size"]
        ):
            raise Full128EvaluationError("Full128 cache vector contract differs")
        offsets.add(row["offset_bytes"])
        ordered_offsets.append(row["offset_bytes"])
        _require_sha256(row["sha256"], "cache vector sha256")
    if ordered_offsets != sorted(ordered_offsets):
        raise Full128EvaluationError("Full128 cache vector order differs")
    payload = {
        key: item
        for key, item in descriptor.items()
        if key != "cache_descriptor_sha256"
    }
    if descriptor["cache_descriptor_sha256"] != content_sha256(payload):
        raise Full128EvaluationError("Full128 cache descriptor digest differs")
    return descriptor


def _stream_validate_embedding_pack(
    path: Path,
    *,
    cache_manifest: Mapping[str, Any],
    selected_tokens: Sequence[str],
) -> np.ndarray:
    manifest = _validate_embedding_cache_manifest_shape(cache_manifest)
    expected_size = manifest["pack_byte_size"]
    selected_index = {token: index for index, token in enumerate(selected_tokens)}
    retained = np.empty((len(selected_tokens), EMBEDDING_DIMENSION), dtype=np.float32)
    retained_count = 0
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Full128EvaluationError(
            "embedding cache pack cannot be opened safely"
        ) from exc
    pack_digest = hashlib.sha256()
    observed = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise Full128EvaluationError("embedding cache pack size or type differs")
        with os.fdopen(descriptor, "rb", buffering=1 << 20, closefd=False) as stream:
            for row in manifest["vectors"]:
                vector_bytes = stream.read(_VECTOR_BYTES)
                if len(vector_bytes) != _VECTOR_BYTES:
                    raise Full128EvaluationError("embedding cache pack length differs")
                observed += len(vector_bytes)
                pack_digest.update(vector_bytes)
                if hashlib.sha256(vector_bytes).hexdigest() != row["sha256"]:
                    raise Full128EvaluationError(
                        "embedding cache vector digest differs"
                    )
                vector = np.frombuffer(vector_bytes, dtype="<f4")
                if not np.isfinite(vector).all() or not np.isclose(
                    np.linalg.norm(vector.astype(np.float64)),
                    1.0,
                    rtol=1e-5,
                    atol=1e-5,
                ):
                    raise Full128EvaluationError(
                        "embedding cache contains a non-contract Full128 vector"
                    )
                output_row = selected_index.get(row["sample_id"])
                if output_row is not None:
                    retained[output_row] = vector
                    retained_count += 1
            if stream.read(1):
                raise Full128EvaluationError("embedding cache pack length differs")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after) or observed != expected_size:
        raise Full128EvaluationError("embedding cache pack changed while reading")
    if pack_digest.hexdigest() != manifest["pack_sha256"]:
        raise Full128EvaluationError("embedding cache pack digest differs")
    if retained_count != len(selected_tokens):
        raise Full128EvaluationError("embedding cache omitted retained panel vectors")
    _validate_loaded_matrix(retained, len(selected_tokens))
    return retained


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_loaded_matrix(matrix: np.ndarray, rows: int) -> None:
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.dtype != np.float32
        or matrix.shape != (rows, EMBEDDING_DIMENSION)
        or not np.isfinite(matrix).all()
        or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=0.0, atol=1e-5)
    ):
        raise Full128EvaluationError(
            "cache adapter returned a non-contract Full128 matrix"
        )


def _new_gallery_root(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parent = absolute.parent.resolve(strict=True)
    target = parent / absolute.name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite Full128 gallery root: {target}")
    target.mkdir(mode=0o700)
    return target


def _gallery_artifact_binding(
    path: Path, *, dataset: str, enrollment_k: int, scorer_hash: str
) -> dict[str, Any]:
    manifest_path = path / "gallery_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    files = [manifest_path]
    files.extend(path / entry["name"] for entry in manifest["files"].values())
    digest = hashlib.sha256()
    entries = []
    for file_path in sorted(files, key=lambda item: item.name):
        payload = file_path.read_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        entries.append(
            {"name": file_path.name, "sha256": sha256, "byte_size": len(payload)}
        )
    return {
        "dataset": dataset,
        "enrollment_k": enrollment_k,
        "relative_path": path.name,
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "gallery_bytes_sha256": digest.hexdigest(),
        "scorer_hash": scorer_hash,
        "files": entries,
    }


def _validate_common_denominators(
    reports: Sequence[ImmutableFull128EvaluationReport],
) -> None:
    if len(reports) != 3:
        raise Full128EvaluationError("Full128 family requires three reports")
    panels = {report.report["panel_sha256"] for report in reports}
    if len(panels) != 1:
        raise Full128EvaluationError("Full128 variants use different panels")
    signatures = []
    for sealed in reports:
        signature = []
        for dataset in sealed.report["datasets"]:
            identity = dataset["identity_metrics"]
            diagnostic = dataset["diagnostic"]
            signature.append(
                (
                    dataset["dataset"],
                    identity["status"],
                    tuple(
                        (item["enrollment_k"], item["query_count"])
                        for item in identity["by_enrollment_k"]
                    ),
                    diagnostic["status"],
                    None
                    if diagnostic["result"] is None
                    else diagnostic["result"]["query_count"],
                )
            )
        signatures.append(tuple(signature))
    if len(set(signatures)) != 1:
        raise Full128EvaluationError("B0/B1/B2 evaluation denominators differ")


def _validate_report_content(report: Mapping[str, Any]) -> None:
    fields = {
        "schema_version",
        "variant_binding",
        "source_binding",
        "panel_sha256",
        "scorer",
        "datasets",
        "pooled",
        "gallery_artifacts",
        "limitations",
    }
    _exact_keys(report, fields, "Full128 report payload")
    if report["schema_version"] != REPORT_SCHEMA:
        raise Full128EvaluationError("Full128 report payload schema differs")
    _require_sha256(report["panel_sha256"], "panel_sha256")
    binding = report["variant_binding"]
    binding_fields = {
        "variant_id",
        "baseline_family_sha256",
        "variant_manifest_sha256",
        "variant_artifact_sha256",
        "training_run_sha256",
        "checkpoint_manifest_file_sha256",
        "checkpoint_manifest_sha256",
        "preprocessing_manifest_file_sha256",
        "preprocessing_manifest_sha256",
        "embedding_manifest_file_sha256",
        "embedding_manifest_sha256",
        "embedding_cache_manifest_file_sha256",
        "embedding_cache_manifest_sha256",
        "cache_descriptor_sha256",
    }
    _exact_keys(binding, binding_fields, "Full128 report variant binding")
    if binding["variant_id"] not in VARIANT_IDS:
        raise Full128EvaluationError("Full128 report variant differs")
    for field, value in binding.items():
        if field != "variant_id":
            _require_sha256(value, field)
    source_binding = report["source_binding"]
    _exact_keys(
        source_binding,
        {
            "inventory_bundle_sha256",
            "inventory_sha256",
            "split_manifest_sha256",
            "split_census_sha256",
            "baseline_family_sha256",
        },
        "Full128 report source binding",
    )
    for field, value in source_binding.items():
        _require_sha256(value, field)
    if source_binding["baseline_family_sha256"] != binding["baseline_family_sha256"]:
        raise Full128EvaluationError("Full128 report family bindings differ")
    scorer = report["scorer"]
    if scorer != {
        "algorithm": SCORER_ALGORITHM,
        "exact": True,
        "ann": False,
        "open_set": False,
        "channel": FULL128_CHANNEL,
        "dimension": EMBEDDING_DIMENSION,
        "dtype": "float32",
        "normalization": "L2",
    }:
        raise Full128EvaluationError("Full128 report scorer contract differs")
    if not isinstance(report["datasets"], list) or [
        item.get("dataset") for item in report["datasets"]
    ] != [
        "mpdd",
        "sibetan",
        "yt-bb-dog",
        *IDENTITY_NONE_DATASETS,
    ]:
        raise Full128EvaluationError("Full128 report dataset suite differs")
    expected_kinds = {
        "mpdd": "REGISTERED",
        "sibetan": "REGISTERED",
        "yt-bb-dog": "GENERATED",
        **{name: "NONE" for name in IDENTITY_NONE_DATASETS},
    }
    expected_artifacts: dict[tuple[str, int], tuple[str, str]] = {}
    for dataset in report["datasets"]:
        _exact_keys(
            dataset,
            {
                "dataset",
                "identity_evidence_kind",
                "protocol_label",
                "canonical_biometric_claim",
                "coverage",
                "identity_metrics",
                "diagnostic",
            },
            "Full128 report dataset",
        )
        kind = dataset["identity_evidence_kind"]
        if kind != expected_kinds[dataset["dataset"]] or dataset[
            "canonical_biometric_claim"
        ] is not (kind == "REGISTERED"):
            raise Full128EvaluationError("Full128 report identity claim differs")
        _validate_report_coverage(dataset["coverage"])
        identity_metrics = dataset["identity_metrics"]
        diagnostic = dataset["diagnostic"]
        _exact_keys(
            identity_metrics,
            {"status", "reason", "by_enrollment_k"},
            "Full128 identity metrics lane",
        )
        _exact_keys(
            diagnostic,
            {"status", "reason", "result"},
            "Full128 diagnostic lane",
        )
        if kind == "NONE" and (
            identity_metrics["status"] != "NOT_APPLICABLE"
            or identity_metrics["by_enrollment_k"] != []
            or diagnostic["status"] != "NOT_AVAILABLE"
            or diagnostic["result"] is not None
        ):
            raise Full128EvaluationError("identity-NONE metrics must be NOT_APPLICABLE")
        if identity_metrics["status"] == "AVAILABLE":
            results = identity_metrics["by_enrollment_k"]
            if kind != "REGISTERED" or [
                item.get("enrollment_k") for item in results
            ] != list(ENROLLMENT_KS):
                raise Full128EvaluationError("Full128 identity result lanes differ")
            for result in results:
                _validate_report_result(result, canonical=True)
                if result["query_count"] != dataset["coverage"]["query_count"]:
                    raise Full128EvaluationError("Full128 dataset query count differs")
                expected_artifacts[(dataset["dataset"], result["enrollment_k"])] = (
                    result["gallery_bytes_sha256"],
                    result["scorer_hash"],
                )
        elif identity_metrics["status"] not in {"NOT_AVAILABLE", "NOT_APPLICABLE"}:
            raise Full128EvaluationError("Full128 identity metric status differs")
        elif identity_metrics["by_enrollment_k"] != []:
            raise Full128EvaluationError(
                "unavailable Full128 identity lane has results"
            )
        if diagnostic["status"] == "AVAILABLE":
            if kind != "GENERATED" or diagnostic["result"] is None:
                raise Full128EvaluationError("Full128 diagnostic identity kind differs")
            _validate_report_result(diagnostic["result"], canonical=False)
            if (
                diagnostic["result"]["query_count"]
                != dataset["coverage"]["query_count"]
            ):
                raise Full128EvaluationError("Full128 diagnostic query count differs")
        elif diagnostic["status"] not in {"NOT_AVAILABLE", "NOT_APPLICABLE"}:
            raise Full128EvaluationError("Full128 diagnostic status differs")
        elif diagnostic["result"] is not None:
            raise Full128EvaluationError("unavailable Full128 diagnostic has a result")
    if (
        not isinstance(report["pooled"], Mapping)
        or set(report["pooled"]) != {"status", "reason", "metrics"}
        or report["pooled"]["status"] != "NOT_AVAILABLE"
        or not isinstance(report["pooled"]["reason"], str)
        or not report["pooled"]["reason"]
        or report["pooled"]["metrics"] is not None
    ):
        raise Full128EvaluationError("heterogeneous Full128 results must not be pooled")
    artifacts = report["gallery_artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_artifacts):
        raise Full128EvaluationError("Full128 gallery artifact suite differs")
    observed_artifacts: set[tuple[str, int]] = set()
    for artifact in artifacts:
        _exact_keys(
            artifact,
            {
                "dataset",
                "enrollment_k",
                "relative_path",
                "manifest_file_sha256",
                "gallery_bytes_sha256",
                "scorer_hash",
                "files",
            },
            "Full128 gallery artifact",
        )
        key = (artifact["dataset"], artifact["enrollment_k"])
        if (
            key in observed_artifacts
            or key not in expected_artifacts
            or artifact["relative_path"] != f"{key[0]}-K{key[1]}"
            or (artifact["gallery_bytes_sha256"], artifact["scorer_hash"])
            != expected_artifacts[key]
        ):
            raise Full128EvaluationError("Full128 gallery artifact binding differs")
        for field in (
            "manifest_file_sha256",
            "gallery_bytes_sha256",
            "scorer_hash",
        ):
            _require_sha256(artifact[field], field)
        files = artifact["files"]
        if not isinstance(files, list) or not files:
            raise Full128EvaluationError("Full128 gallery artifact files differ")
        names: set[str] = set()
        for file_entry in files:
            _exact_keys(
                file_entry, {"name", "sha256", "byte_size"}, "gallery artifact file"
            )
            if (
                not isinstance(file_entry["name"], str)
                or Path(file_entry["name"]).name != file_entry["name"]
                or file_entry["name"] in names
                or isinstance(file_entry["byte_size"], bool)
                or not isinstance(file_entry["byte_size"], int)
                or file_entry["byte_size"] <= 0
            ):
                raise Full128EvaluationError("gallery artifact file contract differs")
            _require_sha256(file_entry["sha256"], "gallery artifact file sha256")
            names.add(file_entry["name"])
        if "gallery_manifest.json" not in names:
            raise Full128EvaluationError("gallery artifact manifest file is missing")
        observed_artifacts.add(key)
    if observed_artifacts != set(expected_artifacts):
        raise Full128EvaluationError("Full128 gallery artifact suite differs")
    if (
        not isinstance(report["limitations"], list)
        or not report["limitations"]
        or any(
            not isinstance(value, str) or not value for value in report["limitations"]
        )
    ):
        raise Full128EvaluationError("Full128 report limitations differ")


def _validate_report_coverage(value: object) -> None:
    _exact_keys(
        value,
        {
            "candidate_count",
            "query_count",
            "excluded_count",
            "excluded_identity_count",
        },
        "Full128 report coverage",
    )
    if any(
        isinstance(value[field], bool)
        or not isinstance(value[field], int)
        or value[field] < 0
        for field in value
    ):
        raise Full128EvaluationError("Full128 report coverage counts differ")


def _validate_report_result(value: object, *, canonical: bool) -> None:
    fields = {
        "enrollment_k",
        "query_count",
        "gallery_template_count",
        "gallery_identity_count",
        "ranking_unit",
        "aggregation",
        "metrics",
        "query_rows",
        "gallery_bytes_sha256",
        "scorer_hash",
    }
    if canonical:
        fields.add("query_panel_sha256")
    _exact_keys(value, fields, "Full128 report result")
    if canonical:
        if value["enrollment_k"] not in ENROLLMENT_KS:
            raise Full128EvaluationError("Full128 enrollment K differs")
        _require_sha256(value["query_panel_sha256"], "query panel sha256")
        _require_sha256(value["gallery_bytes_sha256"], "gallery bytes sha256")
    elif value["enrollment_k"] is not None or value["gallery_bytes_sha256"] is not None:
        raise Full128EvaluationError("Full128 diagnostic gallery binding differs")
    _require_sha256(value["scorer_hash"], "result scorer hash")
    for field in ("query_count", "gallery_template_count", "gallery_identity_count"):
        if (
            isinstance(value[field], bool)
            or not isinstance(value[field], int)
            or value[field] <= 0
        ):
            raise Full128EvaluationError("Full128 result cardinality differs")
    rows = value["query_rows"]
    if not isinstance(rows, list) or len(rows) != value["query_count"]:
        raise Full128EvaluationError("Full128 query rows differ")
    for row in rows:
        _exact_keys(
            row,
            {
                "sample_token",
                "bootstrap_cluster_id",
                "relevant_rank",
                *_METRIC_NAMES,
            },
            "Full128 query metric row",
        )
        rank = row["relevant_rank"]
        if (
            not isinstance(row["sample_token"], str)
            or not row["sample_token"]
            or not isinstance(row["bootstrap_cluster_id"], str)
            or not row["bootstrap_cluster_id"]
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= 0
            or rank > value["gallery_identity_count"]
            or any(
                row[name] != _query_metric_row("", "", rank)[name]
                for name in _METRIC_NAMES
            )
        ):
            raise Full128EvaluationError("Full128 query metric row differs")
    metrics = value["metrics"]
    _exact_keys(metrics, set(_METRIC_NAMES), "Full128 result metrics")
    for name in _METRIC_NAMES:
        metric = metrics[name]
        _exact_keys(
            metric,
            {
                "status",
                "value",
                "numerator",
                "denominator",
                "reason",
                "confidence_interval",
            },
            "Full128 result metric",
        )
        values = [float(row[name]) for row in rows]
        if (
            metric["status"] != "AVAILABLE"
            or metric["reason"] is not None
            or metric["denominator"] != len(rows)
            or metric["numerator"] != float(math.fsum(values))
            or metric["value"] != float(math.fsum(values) / len(values))
        ):
            raise Full128EvaluationError("Full128 result metric summary differs")
        _validate_report_ci(
            metric["confidence_interval"],
            name,
            len(rows),
            float(np.mean(np.asarray(values, dtype=np.float64))),
        )


def _validate_report_ci(
    value: object, metric: str, query_count: int, estimate: float
) -> None:
    available_fields = {
        "status",
        "metric",
        "estimate",
        "lower_bound",
        "upper_bound",
        "confidence_level",
        "cluster_unit",
        "cluster_count",
        "query_row_count",
        "resamples",
        "seed",
        "interval_method",
        "reason",
    }
    unavailable_fields = {
        "status",
        "lower_bound",
        "upper_bound",
        "cluster_count",
        "resamples",
        "seed",
        "reason",
    }
    if not isinstance(value, Mapping) or value.get("status") not in {
        "AVAILABLE",
        "NOT_AVAILABLE",
    }:
        raise Full128EvaluationError("Full128 confidence interval status differs")
    if value["status"] == "AVAILABLE":
        _exact_keys(value, available_fields, "Full128 confidence interval")
        if (
            value["metric"] != metric
            or value["estimate"] != estimate
            or value["query_row_count"] != query_count
            or value["cluster_unit"] != "query_identity"
            or value["confidence_level"] != 0.95
            or value["interval_method"] != "whole_identity_percentile_bootstrap"
            or value["reason"] is not None
            or value["cluster_count"] < 2
            or value["cluster_count"] > query_count
            or isinstance(value["resamples"], bool)
            or not isinstance(value["resamples"], int)
            or value["resamples"] <= 0
            or isinstance(value["seed"], bool)
            or not isinstance(value["seed"], int)
            or value["seed"] < 0
            or not all(
                isinstance(value[field], (int, float))
                and not isinstance(value[field], bool)
                and math.isfinite(float(value[field]))
                for field in ("estimate", "lower_bound", "upper_bound")
            )
            or value["lower_bound"] > value["upper_bound"]
        ):
            raise Full128EvaluationError("Full128 confidence interval differs")
    else:
        _exact_keys(value, unavailable_fields, "Full128 confidence interval")
        if (
            value["lower_bound"] is not None
            or value["upper_bound"] is not None
            or not isinstance(value["reason"], str)
            or not value["reason"]
        ):
            raise Full128EvaluationError(
                "unavailable Full128 confidence interval differs"
            )


def _validate_bootstrap(resamples: int, seed: int) -> None:
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise Full128EvaluationError("bootstrap resamples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Full128EvaluationError("bootstrap seed must be nonnegative")


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Full128EvaluationError(f"{label} must be lowercase SHA-256")


def _exact_keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Full128EvaluationError(f"{label} fields differ")


__all__ = [
    "CACHE_DESCRIPTOR_SCHEMA",
    "FAMILY_INDEX_SCHEMA",
    "MASTER_TABLE_SCHEMA",
    "PANEL_SCHEMA",
    "REPORT_SCHEMA",
    "Full128EmbeddingCacheAdapter",
    "Full128EvaluationError",
    "ImmutableFull128EvaluationReport",
    "PackedFull128EmbeddingCacheAdapter",
    "build_full128_evaluation_panel",
    "build_full128_family_index",
    "build_full128_gallery_embedding_contract",
    "build_full128_master_table",
    "discover_packed_full128_embedding_cache_adapters",
    "evaluate_full128_family",
    "evaluate_full128_variant",
    "full128_master_table_csv",
    "validate_full128_embedding_cache_descriptor",
    "validate_full128_evaluation_panel",
]
