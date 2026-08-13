from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import evaluation.full_segment.full128 as full128_module
from evaluation.full_segment.full128 import (
    _METRIC_NAMES,
    CACHE_DESCRIPTOR_SCHEMA,
    Full128EvaluationError,
    ImmutableFull128EvaluationReport,
    PackedFull128EmbeddingCacheAdapter,
    _clustered_bootstrap_cis,
    build_full128_evaluation_panel,
    build_full128_gallery_embedding_contract,
    discover_packed_full128_embedding_cache_adapters,
    evaluate_full128_family,
    evaluate_full128_variant,
    validate_full128_embedding_cache_descriptor,
)
from evaluation.retrieval import identity_clustered_bootstrap_ci
from foundation.protected_io import json_document_bytes
from foundation.provenance import content_sha256
from identity.full.full_split_census import (
    FullStatus,
    IdentityEvidenceKind,
    RegionStatus,
    TerminalRole,
    UnifiedFullObservation,
    ViewScope,
    allocate_unified_full_split,
    unified_full_split_bundle,
)
from identity.registry.generated_identity_registry import (
    GENERATED_DOG_NAMESPACE,
    compute_generated_identity_id,
    compute_source_cluster_token,
)
from identity.registry.identity_registry import (
    REGISTERED_DOG_NAMESPACE,
    compute_registered_dog_id,
)
from embedding.methods.full_segment.training.artifacts import file_binding, write_embedding_cache
from embedding.methods.full_segment.preparation.data import Full128Sample
from embedding.methods.full_segment.preparation.inventory import BUNDLE_SCHEMA, INVENTORY_SCHEMA
from embedding.methods.full_segment.training.manifests import (
    build_baseline_family_manifest,
    build_checkpoint_manifest,
    build_embedding_manifest,
    build_model_manifest,
    build_preprocessing_manifest,
)
from retrieval.gallery import IdentityGallery, IdentityRegistryPolicy
from workflows.evaluate_full128_family import main as evaluate_workflow


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _registered(index: int) -> str:
    return compute_registered_dog_id(f"full128-evaluation:{index}")


def _generated(index: int) -> str:
    cluster = compute_source_cluster_token(f"full128-generated:{index}")
    return compute_generated_identity_id("full128-evaluation-test:v1", cluster)


def _row(
    label: str,
    *,
    dataset: str,
    official_split: str,
    kind: str,
    identity: str | None,
    role: str,
    source: str | None = None,
    capture: str | None = None,
    sequence: str | None = None,
    duplicate: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    namespace = {
        "REGISTERED": str(REGISTERED_DOG_NAMESPACE),
        "GENERATED": str(GENERATED_DOG_NAMESPACE),
        "NONE": None,
    }[kind]
    return {
        "dataset_name": dataset,
        "official_split": official_split,
        "identity_evidence_kind": kind,
        "identity_namespace_uuid": namespace,
        "identity_token": identity,
        "sample_token": _sha(f"sample:{label}"),
        "source_group": source or f"source:{label}",
        "capture_group": capture or f"capture:{label}",
        "sequence_group": sequence or f"sequence:{label}",
        "duplicate_component": duplicate or _sha(f"duplicate:{label}"),
        "terminal_role": role,
        "source_observation_sha256": _sha(f"observation:{label}"),
        "effective_source_sha256": content or _sha(f"content:{label}"),
        "crop_artifacts_present": True,
        "full_status": "USABLE",
        "view_scope": "FACE_NATIVE",
        "crop_record_sha256": _sha(f"crop:{label}"),
    }


def _base_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity_index in range(2):
        identity = _registered(identity_index)
        for gallery_index in range(5):
            rows.append(
                _row(
                    f"mpdd-{identity_index}-g{gallery_index}",
                    dataset="mpdd",
                    official_split="gallery",
                    kind="REGISTERED",
                    identity=identity,
                    role="EVAL",
                )
            )
        rows.append(
            _row(
                f"mpdd-{identity_index}-q",
                dataset="mpdd",
                official_split="query",
                kind="REGISTERED",
                identity=identity,
                role="EVAL",
            )
        )
    for identity_index in range(2):
        identity = _generated(identity_index)
        for frame in range(2):
            rows.append(
                _row(
                    f"yt-{identity_index}-{frame}",
                    dataset="yt-bb-dog",
                    official_split="test",
                    kind="GENERATED",
                    identity=identity,
                    role="EVAL",
                    source=f"yt-track:{identity_index}",
                    capture=f"yt-track:{identity_index}",
                    sequence=f"yt-track:{identity_index}",
                )
            )
    for dataset in ("ap10k-dog", "dogflw", "oxford-pets-dog"):
        rows.append(
            _row(
                f"none-{dataset}",
                dataset=dataset,
                official_split="test",
                kind="NONE",
                identity=None,
                role="AUXILIARY",
            )
        )
    return rows


def _inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observations = []
    for row in rows:
        observations.append(
            UnifiedFullObservation(
                dataset_name=row["dataset_name"],
                official_split=row["official_split"],
                identity_evidence_kind=IdentityEvidenceKind(
                    row["identity_evidence_kind"]
                ),
                identity_namespace_uuid=row["identity_namespace_uuid"],
                identity_token=row["identity_token"],
                sample_token=row["sample_token"],
                source_group=row["source_group"],
                capture_group=row["capture_group"],
                sequence_group=row["sequence_group"],
                duplicate_component=row["duplicate_component"],
                gradient_eligible=False,
                validation_only=False,
                full_status=FullStatus.USABLE,
                face_status=RegionStatus.NATIVE,
                nose_status=RegionStatus.NOT_DETECTED,
                view_scope=ViewScope.FACE_NATIVE,
                source_observation_sha256=row["source_observation_sha256"],
                terminal_role=TerminalRole(row["terminal_role"]),
            )
        )
    manifest = allocate_unified_full_split(
        allocation_name="full128-evaluation-test", observations=observations
    )
    split = unified_full_split_bundle(manifest)
    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "records": sorted(rows, key=lambda row: row["sample_token"]),
    }
    family = build_baseline_family_manifest()
    admission = {"schema_version": "test", "datasets": []}
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "artifact_root": "/external/test-only",
        "content_kind": "METADATA_ONLY",
        "source_registry_admission_state": admission,
        "source_registry_admission_sha256": content_sha256(admission),
        "baseline_family_manifest": family,
        "baseline_family_sha256": content_sha256(family),
        "split_manifest_sha256": split["manifest_sha256"],
        "split_census_sha256": split["census_sha256"],
        "split_bundle": split,
        "inventory_sha256": content_sha256(inventory),
        "inventory": inventory,
    }
    return {**payload, "bundle_sha256": content_sha256(payload)}


def _vectors(panel: dict[str, Any]) -> np.ndarray:
    relevance: dict[str, str] = {}
    for dataset in panel["datasets"]:
        for sample in dataset["samples"]:
            relevance[sample["sample_token"]] = sample["relevance_token"]
    identities = {
        identity: index
        for index, identity in enumerate(sorted(set(relevance.values())))
    }
    vectors = np.zeros((len(panel["required_sample_tokens"]), 128), dtype=np.float32)
    for row, token in enumerate(panel["required_sample_tokens"]):
        vectors[row, identities[relevance[token]]] = 1.0
    return vectors


def _run_manifest(inventory: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    source_closure: dict[str, Any] = {}
    bindings = {
        "assembly_sha256": _sha("assembly"),
        "inventory_bundle_sha256": inventory["bundle_sha256"],
        "inventory_sha256": inventory["inventory_sha256"],
        "split_manifest_sha256": inventory["split_manifest_sha256"],
        "split_census_sha256": inventory["split_census_sha256"],
        "baseline_family_sha256": inventory["baseline_family_sha256"],
        "family_manifest_sha256": inventory["baseline_family_sha256"],
        "run_config_sha256": content_sha256(config),
        "source_closure_sha256": content_sha256(source_closure),
        "uv_lock": {"sha256": _sha("uv-lock"), "byte_size": 1},
    }
    payload = {
        "schema_version": "cvi.full128_training_run.v1",
        "run_config": config,
        "bindings": bindings,
        "source_closure": source_closure,
        "runtime_versions": {},
    }
    return {**payload, "run_manifest_sha256": content_sha256(payload)}


def _cache_samples(inventory: dict[str, Any]) -> tuple[Full128Sample, ...]:
    samples = []
    for row in inventory["inventory"]["records"]:
        if (
            row["identity_evidence_kind"] == "NONE"
            or row["identity_token"] is None
            or row["terminal_role"] not in {"FIT", "DEV", "CAL", "EVAL"}
        ):
            continue
        samples.append(
            Full128Sample(
                sample_id=row["sample_token"],
                identity_id=row["identity_token"],
                dataset_name=row["dataset_name"],
                view="face",
                role=row["terminal_role"],
                rgb_path=Path("/external/rgb.png"),
                rgb_sha256=_sha("rgb"),
                mask_path=Path("/external/mask.png"),
                mask_sha256=_sha("mask"),
                crop_record_sha256=row["crop_record_sha256"],
            )
        )
    return tuple(sorted(samples, key=lambda sample: sample.sample_id))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(json_document_bytes(value))


def _write_variant(
    tmp_path: Path,
    inventory: dict[str, Any],
    panel: dict[str, Any],
    variant: str,
    run_manifest: dict[str, Any],
    vectors: np.ndarray | None = None,
) -> dict[str, Any]:
    directory = tmp_path / variant
    directory.mkdir()
    samples = _cache_samples(inventory)
    cache_vectors = _vectors(panel) if vectors is None else vectors
    cache = write_embedding_cache(
        directory / "embeddings.f32le", samples, cache_vectors
    )
    _write_json(directory / "embedding-cache-manifest.json", cache)
    family = inventory["baseline_family_manifest"]
    variant_manifest = next(
        item for item in family["variants"] if item["variant_id"] == variant
    )
    method = variant_manifest["method"]
    preprocessing = build_preprocessing_manifest(method=method)
    embedding = build_embedding_manifest(
        method=method,
        component_metadata={"test_component": [0, 128]} if variant == "B0" else None,
    )
    state_path = directory / "model.safetensors"
    state_path.write_bytes(f"state:{variant}".encode())
    checkpoint = build_checkpoint_manifest(
        method=method,
        checkpoint_sha256=file_binding(state_path)["sha256"],
        preprocessing_manifest=preprocessing,
        embedding_manifest=embedding,
        initialization=variant_manifest["initialization"],
        initialization_sha256=_sha("initialization") if variant == "B2" else None,
        initialization_source_contract_sha256=(
            _sha("source-contract") if variant == "B2" else None
        ),
        initialization_intake_receipt_sha256=(
            _sha("intake") if variant == "B2" else None
        ),
        initialization_usage_lane="RESEARCH_ONLY" if variant == "B2" else None,
        fit_partition="FIT" if variant == "B0" else None,
    )
    documents = {
        "model-manifest.json": build_model_manifest(method=method),
        "preprocessing-manifest.json": preprocessing,
        "embedding-manifest.json": embedding,
        "checkpoint-manifest.json": checkpoint,
    }
    for name, document in documents.items():
        _write_json(directory / name, document)
    artifacts = {
        "state": {"relative_path": state_path.name, **file_binding(state_path)},
        "model_manifest": {
            "relative_path": "model-manifest.json",
            **file_binding(directory / "model-manifest.json"),
        },
        "preprocessing_manifest": {
            "relative_path": "preprocessing-manifest.json",
            **file_binding(directory / "preprocessing-manifest.json"),
        },
        "embedding_manifest": {
            "relative_path": "embedding-manifest.json",
            **file_binding(directory / "embedding-manifest.json"),
        },
        "checkpoint_manifest": {
            "relative_path": "checkpoint-manifest.json",
            **file_binding(directory / "checkpoint-manifest.json"),
        },
        "embedding_cache_manifest": {
            "relative_path": "embedding-cache-manifest.json",
            **file_binding(directory / "embedding-cache-manifest.json"),
            "manifest": cache,
        },
    }
    bindings = {
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        **run_manifest["bindings"],
    }
    if variant == "B2":
        bindings["b2_initialization"] = {"test": True}
    payload = {
        "schema_version": "cvi.full128_variant_run.v1",
        "variant_id": variant,
        "method": method,
        "initialization": variant_manifest["initialization"],
        "bindings": bindings,
        "fit_population": {},
        "training": {},
        "artifacts": artifacts,
    }
    variant_run = {**payload, "variant_run_sha256": content_sha256(payload)}
    _write_json(directory / "variant-run.json", variant_run)
    return variant_run


def _adapter(
    tmp_path: Path,
    inventory: dict[str, Any],
    panel: dict[str, Any],
    variant: str,
    vectors: np.ndarray | None = None,
) -> PackedFull128EmbeddingCacheAdapter:
    run_manifest = _run_manifest(inventory)
    _write_variant(tmp_path, inventory, panel, variant, run_manifest, vectors)
    return PackedFull128EmbeddingCacheAdapter.from_training_variant_directory(
        tmp_path / variant,
        training_run_manifest=run_manifest,
        inventory_bundle=inventory,
        panel=panel,
    )


def _training_run(
    root: Path, inventory: dict[str, Any], panel: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root.mkdir()
    run_manifest = _run_manifest(inventory)
    _write_json(root / "run-manifest.json", run_manifest)
    variants = [
        _write_variant(root, inventory, panel, variant, run_manifest)
        for variant in ("B0", "B1", "B2")
    ]
    payload = {
        "schema_version": "cvi.full128_family_run.v1",
        "family_id": "FULL128_B0_B1_B2",
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "run_config_sha256": run_manifest["bindings"]["run_config_sha256"],
        "family_manifest_sha256": run_manifest["bindings"]["family_manifest_sha256"],
        "variants": [
            {
                "variant_id": item["variant_id"],
                "variant_run_sha256": item["variant_run_sha256"],
            }
            for item in variants
        ],
        "status": "COMPLETE_EXACT_THREE_VARIANT_FAMILY",
    }
    _write_json(
        root / "family-run.json",
        {**payload, "family_run_sha256": content_sha256(payload)},
    )
    return run_manifest, variants


def test_panel_is_order_invariant_and_labels_noncanonical_protocols() -> None:
    rows = _base_rows()
    first = build_full128_evaluation_panel(_inventory(rows))
    second = build_full128_evaluation_panel(_inventory(list(reversed(rows))))

    assert first == second
    by_dataset = {item["dataset"]: item for item in first["datasets"]}
    assert by_dataset["yt-bb-dog"]["identity_evidence_kind"] == "GENERATED"
    assert by_dataset["yt-bb-dog"]["canonical_biometric_claim"] is False
    assert by_dataset["sibetan"]["status"] == "NOT_AVAILABLE"
    assert "DEV/CAL were not repurposed" in by_dataset["sibetan"]["reason"]
    for dataset in ("ap10k-dog", "dogflw", "oxford-pets-dog"):
        assert by_dataset[dataset]["identity_evidence_kind"] == "NONE"
        assert by_dataset[dataset]["status"] == "NOT_AVAILABLE"


def test_sibetan_requires_eval_cross_sequence_and_identity_none_cache_is_unavailable() -> (
    None
):
    rows = _base_rows()
    sibetan_identity = _registered(20)
    rows.extend(
        [
            _row(
                "sibetan-sequence-a",
                dataset="sibetan",
                official_split="unassigned",
                kind="REGISTERED",
                identity=sibetan_identity,
                role="EVAL",
                source="sibetan-source-a",
                capture="sibetan-capture-a",
                sequence="sibetan-sequence-a",
            ),
            _row(
                "sibetan-sequence-b",
                dataset="sibetan",
                official_split="unassigned",
                kind="REGISTERED",
                identity=sibetan_identity,
                role="EVAL",
                source="sibetan-source-b",
                capture="sibetan-capture-b",
                sequence="sibetan-sequence-b",
            ),
            _row(
                "ap10k-second-view",
                dataset="ap10k-dog",
                official_split="test",
                kind="NONE",
                identity=None,
                role="AUXILIARY",
                source="source:none-ap10k-dog",
                capture="independent-capture",
                sequence="independent-sequence",
            ),
        ]
    )
    first_ap10k = next(row for row in rows if row["dataset_name"] == "ap10k-dog")
    second_ap10k = rows[-1]
    second_ap10k["source_group"] = first_ap10k["source_group"]
    panel = build_full128_evaluation_panel(_inventory(rows))
    by_dataset = {item["dataset"]: item for item in panel["datasets"]}

    assert by_dataset["sibetan"]["status"] == "AVAILABLE"
    assert len(by_dataset["sibetan"]["query_sample_tokens"]) == 1
    assert by_dataset["ap10k-dog"]["status"] == "NOT_AVAILABLE"
    assert by_dataset["ap10k-dog"]["protocol_label"] == "INSTANCE_INVARIANCE_RETRIEVAL"
    assert by_dataset["ap10k-dog"]["canonical_biometric_claim"] is False
    assert by_dataset["ap10k-dog"]["query_sample_tokens"] == []
    assert "cache v1 schema excludes" in by_dataset["ap10k-dog"]["reason"]


@pytest.mark.parametrize(
    "field",
    [
        "duplicate_component",
        "effective_source_sha256",
    ],
)
def test_mpdd_leakage_fails_closed(field: str) -> None:
    rows = _base_rows()
    gallery = next(row for row in rows if row["official_split"] == "gallery")
    query = next(
        row
        for row in rows
        if row["official_split"] == "query"
        and row["identity_token"] == gallery["identity_token"]
    )
    query[field] = gallery[field]

    with pytest.raises(Full128EvaluationError, match="leakage"):
        build_full128_evaluation_panel(_inventory(rows))


@pytest.mark.parametrize("field", ["capture_group", "sequence_group"])
def test_mpdd_allows_unverified_pose_view_group_overlap(field: str) -> None:
    rows = _base_rows()
    gallery = next(row for row in rows if row["official_split"] == "gallery")
    query = next(
        row
        for row in rows
        if row["official_split"] == "query"
        and row["identity_token"] == gallery["identity_token"]
    )
    query[field] = gallery[field]

    panel = build_full128_evaluation_panel(_inventory(rows))
    mpdd = next(item for item in panel["datasets"] if item["dataset"] == "mpdd")

    assert mpdd["status"] == "AVAILABLE"
    assert "unverified filename-derived" in mpdd["independence_policy"]
    assert "cross-session" not in mpdd["independence_policy"]


def test_exact_gallery_matches_brute_force_and_k_panels_share_queries(
    tmp_path: Path,
) -> None:
    inventory = _inventory(_base_rows())
    panel = build_full128_evaluation_panel(inventory)
    adapter = _adapter(tmp_path, inventory, panel, "B0")
    report = evaluate_full128_variant(
        inventory_bundle=inventory,
        panel=panel,
        adapter=adapter,
        gallery_root=tmp_path / "galleries-B0",
        bootstrap_resamples=100,
        bootstrap_seed=17,
    ).report
    mpdd = next(item for item in report["datasets"] if item["dataset"] == "mpdd")
    results = mpdd["identity_metrics"]["by_enrollment_k"]
    assert [item["enrollment_k"] for item in results] == [1, 3, 5]
    assert len({item["query_panel_sha256"] for item in results}) == 1
    assert len({item["query_count"] for item in results}) == 1

    matrix = adapter.load_embeddings(panel["required_sample_tokens"])
    vector_by_token = dict(zip(panel["required_sample_tokens"], matrix, strict=True))
    mpdd_panel = next(item for item in panel["datasets"] if item["dataset"] == "mpdd")
    sample_map = {item["sample_token"]: item for item in mpdd_panel["samples"]}
    for result in results:
        gallery_tokens = mpdd_panel["gallery_sample_tokens_by_k"][
            str(result["enrollment_k"])
        ]
        identities = sorted(
            {sample_map[token]["relevance_token"] for token in gallery_tokens}
        )
        for query_row in result["query_rows"]:
            query = vector_by_token[query_row["sample_token"]]
            scores = {
                identity: max(
                    float(query @ vector_by_token[token])
                    for token in gallery_tokens
                    if sample_map[token]["relevance_token"] == identity
                )
                for identity in identities
            }
            ranked = sorted(
                identities, key=lambda identity: (-scores[identity], identity)
            )
            expected = sample_map[query_row["sample_token"]]["relevance_token"]
            assert query_row["relevant_rank"] == ranked.index(expected) + 1

    requested = list(reversed(panel["required_sample_tokens"][::3]))
    random_access = adapter.load_embeddings(requested)
    np.testing.assert_array_equal(
        random_access,
        np.stack([vector_by_token[token] for token in requested]),
    )
    assert adapter.load_embeddings([]).shape == (0, 128)
    assert adapter.descriptor["schema_version"] == CACHE_DESCRIPTOR_SCHEMA
    assert (
        adapter.descriptor["variant_artifact_sha256"]
        == json.loads(
            (tmp_path / "B0" / "variant-run.json").read_text(encoding="utf-8")
        )["variant_run_sha256"]
    )


def test_packed_adapter_retains_only_panel_vectors_with_exact_v1_values(
    tmp_path: Path,
) -> None:
    rows = _base_rows()
    rows.append(
        _row(
            "sibetan-single-sequence-not-in-panel",
            dataset="sibetan",
            official_split="unassigned",
            kind="REGISTERED",
            identity=_registered(40),
            role="EVAL",
        )
    )
    inventory = _inventory(rows)
    panel = build_full128_evaluation_panel(inventory)
    panel_vectors = _vectors(panel)
    by_token = dict(
        zip(panel["required_sample_tokens"], panel_vectors, strict=True)
    )
    cache_samples = _cache_samples(inventory)
    cache_vectors = np.stack(
        [
            by_token.get(
                sample.sample_id,
                np.eye(1, 128, 127, dtype=np.float32)[0],
            )
            for sample in cache_samples
        ]
    )

    adapter = _adapter(
        tmp_path, inventory, panel, "B0", vectors=cache_vectors
    )

    assert adapter.descriptor["storage"]["source_vector_count"] == len(
        cache_samples
    )
    assert len(adapter._panel_vectors) == len(panel["required_sample_tokens"])
    assert len(adapter._panel_vectors) < len(cache_samples)
    np.testing.assert_array_equal(
        adapter.load_embeddings(panel["required_sample_tokens"]), panel_vectors
    )


def test_variant_replay_tamper_report_tamper_and_overwrite_fail(
    tmp_path: Path,
) -> None:
    inventory = _inventory(_base_rows())
    panel = build_full128_evaluation_panel(inventory)
    b0 = _adapter(tmp_path, inventory, panel, "B0")
    b1 = _adapter(tmp_path, inventory, panel, "B1")
    validate_full128_embedding_cache_descriptor(
        b0.descriptor, variant_id="B0", panel=panel, inventory_bundle=inventory
    )
    with pytest.raises(Full128EvaluationError, match="variant"):
        validate_full128_embedding_cache_descriptor(
            b0.descriptor, variant_id="B1", panel=panel, inventory_bundle=inventory
        )
    replayed_panel = dict(panel)
    replayed_panel["panel_sha256"] = _sha("different-panel")
    with pytest.raises(Full128EvaluationError, match="panel content differs"):
        validate_full128_embedding_cache_descriptor(
            b0.descriptor,
            variant_id="B0",
            panel=replayed_panel,
            inventory_bundle=inventory,
        )
    malformed_descriptor = json.loads(json.dumps(b0.descriptor))
    malformed_descriptor["storage"]["source_vector_count"] += 1
    descriptor_payload = {
        key: value
        for key, value in malformed_descriptor.items()
        if key != "cache_descriptor_sha256"
    }
    malformed_descriptor["cache_descriptor_sha256"] = content_sha256(
        descriptor_payload
    )
    with pytest.raises(Full128EvaluationError, match="storage contract"):
        validate_full128_embedding_cache_descriptor(
            malformed_descriptor,
            variant_id="B0",
            panel=panel,
            inventory_bundle=inventory,
        )

    sealed = evaluate_full128_variant(
        inventory_bundle=inventory,
        panel=panel,
        adapter=b0,
        gallery_root=tmp_path / "gallery-first",
        bootstrap_resamples=50,
        bootstrap_seed=3,
    )
    mpdd_gallery = tmp_path / "gallery-first" / "mpdd-K1"
    registered = frozenset(
        sample["relevance_token"]
        for dataset in panel["datasets"]
        if dataset["dataset"] == "mpdd"
        for sample in dataset["samples"]
    )
    with pytest.raises(RuntimeError, match="embedding contract differs"):
        IdentityGallery(
            mpdd_gallery,
            dim=128,
            embedding_contract=build_full128_gallery_embedding_contract(
                b1.descriptor, panel
            ),
            read_only=True,
            registry_policy=IdentityRegistryPolicy(registered_identity_ids=registered),
        )
    tampered_report = sealed.to_dict()
    tampered_report["report"]["datasets"][0]["coverage"]["query_count"] += 1
    with pytest.raises(Full128EvaluationError, match="tampered"):
        ImmutableFull128EvaluationReport.from_dict(tampered_report)
    malformed_report = sealed.to_dict()
    del malformed_report["report"]["variant_binding"][
        "embedding_manifest_sha256"
    ]
    malformed_report["report_sha256"] = content_sha256(
        malformed_report["report"]
    )
    with pytest.raises(Full128EvaluationError, match="variant binding"):
        ImmutableFull128EvaluationReport.from_dict(malformed_report)
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluate_full128_variant(
            inventory_bundle=inventory,
            panel=panel,
            adapter=b0,
            gallery_root=tmp_path / "gallery-first",
            bootstrap_resamples=50,
            bootstrap_seed=3,
        )

    assert b0._panel_vectors.flags.writeable is False
    assert len(b0._panel_vectors) == len(panel["required_sample_tokens"])


def test_whole_packed_cache_tamper_fails_before_adapter_creation(
    tmp_path: Path,
) -> None:
    inventory = _inventory(_base_rows())
    panel = build_full128_evaluation_panel(inventory)
    run_manifest = _run_manifest(inventory)
    _write_variant(tmp_path, inventory, panel, "B0", run_manifest)
    pack = tmp_path / "B0" / "embeddings.f32le"
    payload = bytearray(pack.read_bytes())
    payload[-1] ^= 1
    pack.write_bytes(payload)

    with pytest.raises(Full128EvaluationError, match="vector digest"):
        PackedFull128EmbeddingCacheAdapter.from_training_variant_directory(
            tmp_path / "B0",
            training_run_manifest=run_manifest,
            inventory_bundle=inventory,
            panel=panel,
        )


def test_family_denominators_and_bootstrap_are_deterministic(tmp_path: Path) -> None:
    inventory = _inventory(_base_rows())
    panel = build_full128_evaluation_panel(inventory)
    adapters = [
        _adapter(tmp_path, inventory, panel, variant) for variant in ("B0", "B1", "B2")
    ]
    built_panel, reports, table = evaluate_full128_family(
        inventory_bundle=inventory,
        adapters=adapters,
        gallery_root=tmp_path / "family-galleries",
        bootstrap_resamples=100,
        bootstrap_seed=29,
    )

    assert built_panel == panel
    assert table["schema_version"] == "cvi.full128_master_table.v1"
    mpdd_results = [
        next(item for item in report.report["datasets"] if item["dataset"] == "mpdd")
        for report in reports
    ]
    denominators = [
        [
            result["metrics"]["Rank-1"]["denominator"]
            for result in item["identity_metrics"]["by_enrollment_k"]
        ]
        for item in mpdd_results
    ]
    assert denominators[0] == denominators[1] == denominators[2]
    cis = [
        item["identity_metrics"]["by_enrollment_k"][0]["metrics"]["Rank-1"][
            "confidence_interval"
        ]
        for item in mpdd_results
    ]
    assert cis[0] == cis[1] == cis[2]

    for report in reports:
        by_dataset = {item["dataset"]: item for item in report.report["datasets"]}
        for dataset in ("ap10k-dog", "dogflw", "oxford-pets-dog"):
            assert by_dataset[dataset]["identity_metrics"]["status"] == "NOT_APPLICABLE"
            assert by_dataset[dataset]["diagnostic"]["status"] == "NOT_AVAILABLE"
    identity_none_tokens = {
        row["sample_token"]
        for row in inventory["inventory"]["records"]
        if row["identity_evidence_kind"] == "NONE"
    }
    assert all(
        not identity_none_tokens.intersection(adapter.descriptor["sample_tokens"])
        for adapter in adapters
    )


def test_shared_bootstrap_plan_exactly_matches_metric_by_metric_reference() -> None:
    rng = np.random.default_rng(90210)
    rows: list[dict[str, Any]] = []
    for cluster, row_count in enumerate((1, 2, 5, 11, 3)):
        for _ in range(row_count):
            row: dict[str, Any] = {"bootstrap_cluster_id": f"identity-{cluster}"}
            row.update({metric: float(rng.random()) for metric in _METRIC_NAMES})
            rows.append(row)
    observed = _clustered_bootstrap_cis(rows, resamples=10_003, seed=71)

    assert observed == {
        metric: identity_clustered_bootstrap_ci(
            rows, metric=metric, resamples=10_003, seed=71
        )
        for metric in _METRIC_NAMES
    }


def test_discovered_family_reuses_one_inventory_and_panel_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _inventory(_base_rows())
    panel = build_full128_evaluation_panel(inventory)
    training_run = tmp_path / "training-run"
    _training_run(training_run, inventory, panel)
    calls = 0
    original = full128_module._validate_inventory_metadata

    def counted(value: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(full128_module, "_validate_inventory_metadata", counted)
    adapters = discover_packed_full128_embedding_cache_adapters(
        training_run, inventory_bundle=inventory
    )
    evaluated_panel, _, _ = evaluate_full128_family(
        inventory_bundle=inventory,
        adapters=adapters,
        gallery_root=tmp_path / "family-galleries",
        bootstrap_resamples=20,
        bootstrap_seed=7,
    )

    assert evaluated_panel == panel
    assert calls == 1

    tampered = json.loads(json.dumps(inventory))
    tampered["inventory"]["records"][0]["official_split"] = "tampered"
    with pytest.raises(Full128EvaluationError, match="bundle digest differs"):
        evaluate_full128_family(
            inventory_bundle=tampered,
            adapters=adapters,
            gallery_root=tmp_path / "tampered-family-galleries",
            bootstrap_resamples=20,
            bootstrap_seed=7,
        )


def test_training_family_discovery_and_workflow_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _inventory(_base_rows())
    panel = build_full128_evaluation_panel(inventory)
    training_run = tmp_path / "training-run"
    _training_run(training_run, inventory, panel)

    adapters = discover_packed_full128_embedding_cache_adapters(
        training_run, inventory_bundle=inventory, panel=panel
    )
    assert [adapter.descriptor["variant_id"] for adapter in adapters] == [
        "B0",
        "B1",
        "B2",
    ]
    inventory_path = tmp_path / "inventory.json"
    _write_json(inventory_path, inventory)
    observed_validation_workers: list[int] = []

    def validate_inventory(value: Any, *, validation_workers: int) -> Any:
        observed_validation_workers.append(validation_workers)
        return value

    monkeypatch.setattr(
        "workflows.evaluate_full128_family.validate_full128_experiment_inventory_bundle",
        validate_inventory,
    )
    output = tmp_path / "evaluation"
    assert (
        evaluate_workflow(
            [
                "--training-run",
                str(training_run),
                "--inventory",
                str(inventory_path),
                "--output-directory",
                str(output),
                "--bootstrap-resamples",
                "20",
                "--bootstrap-seed",
                "7",
            ]
        )
        == 0
    )
    index = json.loads((output / "family-index.json").read_text(encoding="utf-8"))
    assert [item["variant_id"] for item in index["reports"]] == ["B0", "B1", "B2"]
    assert observed_validation_workers == [8]
    assert not list(training_run.glob("**/*.npy"))
