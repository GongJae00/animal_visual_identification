from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import operations.embedding_production_runner as embedding_runner
from contracts.runtime_library_provenance import RuntimeLibraryManifest
from data_pipeline.acquisition import sha256_file
from evaluation.benchmark import TimingSummary
from evaluation.control_scoring import (
    ArtifactCacheBinding,
    ArtifactSourceKind,
    ControlScoringInventory,
    EmbeddingCacheEntry,
    EmbeddingCacheManifest,
    EmbeddingCachePolicy,
    ScoringArtifactEntry,
    embedding_cache_key,
    verify_embedding_cache_files,
)
from evaluation.numerical_admission import (
    NumericalAdmissionDecision,
    NumericalDriftPolicy,
    compare_embedding_caches,
)
from evaluation.score_drift_admission import (
    FrozenScoreMarginBoundary,
    RetrievalScoreRequest,
    RetrievalScoreWorkload,
    ScoreDriftAdmissionPlan,
    ScoreDriftAdmissionReceipt,
    ScoreDriftDecision,
    ScoreDriftPolicy,
    ScoreDriftPrecommitment,
    build_score_drift_admission_plan,
    build_score_drift_precommitment,
    compare_score_rank_threshold_drift,
    score_drift_scoring_semantics_sha256,
    validate_retrieval_workload_content_separation,
    verify_score_drift_receipt_external_anchors,
)
from foundation.provenance import content_sha256
from operations.embedding_producer import (
    EmbeddingBackendIdentity,
    EmbeddingProducerConfig,
    EmbeddingProductionCost,
    EmbeddingProductionReceipt,
    EmbeddingRuntimeResources,
)
from operations.embedding_production_runner import (
    EmbeddingFreshWorkerReceipt,
    EmbeddingProductionPrecommitment,
    EmbeddingWorkerExecutionPolicy,
)
from operations.process_supervisor import (
    ProcessSupervisorPolicy,
    SupervisedProcessResult,
    SupervisedProcessStatus,
)
from operations.worker_environment import build_sanitized_worker_environment
from representation_learning.optimization import PromotionDecision

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64


def opaque(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def producer(backend_name: str, lock_hash: str) -> EmbeddingProducerConfig:
    identity = EmbeddingBackendIdentity(
        backend_name=backend_name,
        backend_version="1",
        runtime_version=f"{backend_name}-runtime",
        execution_provider=backend_name,
        device=backend_name,
        precision="fp32" if backend_name == "cpu" else "fp32_tf32_disabled",
        determinism_mode="REQUESTED_NOT_PROVEN",
        backend_config_sha256=HASH_E if backend_name == "cpu" else HASH_F,
    )
    return EmbeddingProducerConfig(
        model_sha256=HASH_A,
        model_lineage_sha256=HASH_B,
        preprocessing_sha256=HASH_C,
        preprocessing_semantics_sha256=HASH_D,
        dependency_lock_sha256=lock_hash,
        code_revision="frozen-score-drift-test",
        backend=identity,
        vector_dimension=3,
        batch_size=2,
        input_width=2,
        input_height=2,
        input_channels=3,
        input_value_bytes=4,
        l2_epsilon=1e-12,
        normalization_tolerance=1e-6,
    )


def inventory(tokens: tuple[str, ...]) -> ControlScoringInventory:
    return ControlScoringInventory(
        plan_sha256=HASH_A,
        scoring_requests_sha256=HASH_B,
        base_artifact_manifest_sha256=HASH_C,
        base_artifact_verification_sha256=HASH_D,
        control_transform_receipt_sha256=HASH_E,
        entries=tuple(
            ScoringArtifactEntry(
                artifact_token=token,
                content_sha256=opaque(f"content:{token}"),
                byte_size=1,
                source_kind=ArtifactSourceKind.BASE,
            )
            for token in tokens
        ),
    )


def cache(
    root: Path,
    config: EmbeddingProducerConfig,
    scoring_inventory: ControlScoringInventory,
    vectors: dict[str, tuple[float, float, float]],
) -> EmbeddingCacheManifest:
    bindings: list[ArtifactCacheBinding] = []
    entries: list[EmbeddingCacheEntry] = []
    inventory_entries = {
        item.artifact_token: item for item in scoring_inventory.entries
    }
    for token in sorted(vectors):
        artifact_hash = inventory_entries[token].content_sha256
        cache_key = embedding_cache_key(
            artifact_content_sha256=artifact_hash,
            model_sha256=config.model_sha256,
            inference_config_sha256=config.config_sha256,
            dependency_lock_sha256=config.dependency_lock_sha256,
            code_revision=config.code_revision,
            precision=config.backend.precision,
            vector_dimension=config.vector_dimension,
        )
        path = root / f"{cache_key}.f32le"
        path.write_bytes(struct.pack("<3f", *vectors[token]))
        bindings.append(ArtifactCacheBinding(token, artifact_hash, cache_key))
        entries.append(
            EmbeddingCacheEntry(
                cache_key=cache_key,
                relative_path=path.name,
                content_sha256=sha256_file(path),
                byte_size=path.stat().st_size,
            )
        )
    return EmbeddingCacheManifest(
        scoring_inventory_sha256=scoring_inventory.inventory_sha256,
        model_sha256=config.model_sha256,
        inference_config_sha256=config.config_sha256,
        dependency_lock_sha256=config.dependency_lock_sha256,
        code_revision=config.code_revision,
        precision=config.backend.precision,
        vector_dimension=config.vector_dimension,
        normalization_tolerance=config.normalization_tolerance,
        bindings=tuple(bindings),
        entries=tuple(entries),
    )


def workload() -> RetrievalScoreWorkload:
    requests = []
    for query in ("q1", "q2"):
        for candidate in ("a", "b", "c"):
            requests.append(
                RetrievalScoreRequest(
                    request_id=opaque(f"request:{query}:{candidate}"),
                    query_group_token=opaque(f"query:{query}"),
                    candidate_slot_token=opaque(f"slot:{candidate}"),
                    query_artifact_token=query,
                    candidate_artifact_token=candidate,
                )
            )
    return RetrievalScoreWorkload(
        gallery_sha256=HASH_B,
        pairing_policy_sha256=HASH_D,
        retrieval_plan_sha256=HASH_E,
        split_manifest_sha256=opaque("optimization-dev-split"),
        workload_construction_receipt_sha256=opaque("workload-construction"),
        data_role="OPTIMIZATION_DEV",
        selection_blind_to_candidate_outputs=True,
        requests=tuple(
            sorted(
                requests,
                key=lambda item: (
                    item.query_group_token,
                    item.candidate_slot_token,
                ),
            )
        )
    )


def boundary(
    reference_config: EmbeddingProducerConfig,
    frozen_workload: RetrievalScoreWorkload,
    policy: ScoreDriftPolicy | None = None,
) -> FrozenScoreMarginBoundary:
    resolved_policy = policy or score_policy()
    return FrozenScoreMarginBoundary(
        score_threshold=0.9,
        margin_threshold=0.15,
        top_k=(1, 2),
        reference_model_sha256=HASH_A,
        reference_producer_config_sha256=reference_config.config_sha256,
        reference_preprocessing_sha256=reference_config.preprocessing_sha256,
        reference_preprocessing_semantics_sha256=(
            reference_config.preprocessing_semantics_sha256
        ),
        gallery_sha256=HASH_B,
        calibration_manifest_sha256=HASH_C,
        calibration_score_receipt_sha256=HASH_E,
        pairing_policy_sha256=HASH_D,
        retrieval_plan_sha256=frozen_workload.retrieval_plan_sha256,
        workload_sha256=frozen_workload.workload_sha256,
        scoring_semantics_sha256=(
            score_drift_scoring_semantics_sha256(resolved_policy)
        ),
    )


def production_receipt(
    config: EmbeddingProducerConfig,
    manifest: EmbeddingCacheManifest,
    verification: object,
    cache_policy: EmbeddingCachePolicy,
) -> EmbeddingProductionReceipt:
    unique_vectors = len(manifest.entries)
    batches = math.ceil(unique_vectors / config.batch_size)
    output_bytes = sum(item.byte_size for item in manifest.entries)
    return EmbeddingProductionReceipt(
        scoring_inventory_sha256=manifest.scoring_inventory_sha256,
        producer_config_sha256=config.config_sha256,
        production_policy_sha256=opaque("production-policy"),
        cache_policy_sha256=cache_policy.policy_sha256,
        model_lineage_sha256=config.model_lineage_sha256,
        cache_manifest=manifest,
        cache_verification=verification,
        batch_timing=TimingSummary.from_samples((1,) * batches),
        runtime_resources=EmbeddingRuntimeResources.unavailable(),
        cost=EmbeddingProductionCost(
            artifact_bindings=len(manifest.bindings),
            unique_content_inputs=unique_vectors,
            content_deduplication_calls_saved=(
                len(manifest.bindings) - unique_vectors
            ),
            warmup_batches=0,
            warmup_artifact_evaluations=0,
            production_batches=batches,
            production_artifact_evaluations=unique_vectors,
            total_backend_artifact_evaluations=unique_vectors,
            warmup_wall_time_ns=0,
            production_wall_time_ns=batches,
            total_backend_wall_time_ns=batches,
            input_integrity_hash_passes=2,
            input_integrity_bytes_read=0,
            provenance_integrity_hash_passes=2,
            provenance_integrity_bytes_read=0,
            output_float_values=unique_vectors * config.vector_dimension,
            output_bytes_written=output_bytes,
            peak_batch_artifacts=min(config.batch_size, unique_vectors),
            peak_batch_input_bytes=0,
            peak_nominal_input_tensor_bytes=0,
            peak_batch_output_bytes=(
                min(config.batch_size, unique_vectors)
                * config.vector_dimension
                * 4
            ),
        ),
    )


def protected_production_receipt(
    config: EmbeddingProducerConfig,
    receipt: EmbeddingProductionReceipt,
    scoring_inventory: ControlScoringInventory,
    *,
    sequence: int,
) -> EmbeddingFreshWorkerReceipt:
    _, environment_identity = build_sanitized_worker_environment(
        os.environ,
        python_executable=sys.executable,
    )
    supervisor = ProcessSupervisorPolicy(
        timeout_seconds=10.0,
        termination_grace_seconds=1.0,
        poll_interval_seconds=0.01,
        maximum_stdout_bytes=0,
        maximum_stderr_bytes=0,
    )
    execution_policy = EmbeddingWorkerExecutionPolicy(
        supervisor=supervisor,
        maximum_worker_result_bytes=1_048_576,
        maximum_snapshot_bytes=1_048_576,
    )
    runtime_policy_sha256 = opaque(f"runtime-policy:{sequence}")
    runtime_manifest = RuntimeLibraryManifest(
        policy_sha256=runtime_policy_sha256,
        entries=(),
        binary_set_sha256=content_sha256([]),
        maps_snapshots=4,
        maps_bytes_read=0,
        binary_bytes_hashed=0,
        provenance_wall_time_ns=0,
        decision="PASS",
        hard_failures=(),
    )
    source_names = embedding_runner._CODE_SOURCE_NAMES
    code_sources = tuple(
        (name, opaque(f"source:{sequence}:{name}"), 1)
        for name in source_names
    )
    precommitment = EmbeddingProductionPrecommitment(
        scoring_inventory_sha256=scoring_inventory.inventory_sha256,
        producer_config_sha256=config.config_sha256,
        production_policy_sha256=receipt.production_policy_sha256,
        cache_policy_sha256=receipt.cache_policy_sha256,
        backend_identity_sha256=config.backend.identity_sha256,
        runtime_library_policy_sha256=runtime_policy_sha256,
        worker_execution_policy_sha256=execution_policy.policy_sha256,
        worker_environment_identity_sha256=(
            environment_identity.identity_sha256
        ),
        artifact_bindings=tuple(
            (
                item.artifact_token,
                item.content_sha256,
                item.byte_size,
            )
            for item in scoring_inventory.entries
        ),
        provenance_sha256=tuple(
            (name, opaque(f"provenance:{sequence}:{name}"), 1)
            for name in (
                "dependency_lock", "model", "model_lineage", "onnx_config",
                "preprocessing",
            )
        ),
        code_source_sha256=code_sources,
        code_source_manifest_sha256=content_sha256(
            [list(item) for item in code_sources]
        ),
        code_source_files=len(code_sources),
        code_source_bytes=len(code_sources),
        worker_bootstrap_sha256=(
            embedding_runner.EMBEDDING_WORKER_BOOTSTRAP_SHA256
        ),
        input_bytes_hashed=sum(
            item.byte_size for item in scoring_inventory.entries
        ),
        provenance_bytes_hashed=5,
        prior_attempt_ledger_sha256=opaque(f"prior:{sequence}"),
        candidate_attempt_token=opaque(f"attempt:{sequence}"),
        precommitment_sequence=sequence,
    )
    worker_request_sha256 = opaque(f"request:{sequence}")
    supervised = SupervisedProcessResult(
        command=(
            environment_identity.python_executable_invocation_path,
            "-I", "-B", "-c", embedding_runner.EMBEDDING_WORKER_BOOTSTRAP,
            f"/tmp/code-{sequence}",
            "--request", f"/tmp/request-{sequence}.json",
            "--result", f"/tmp/result-{sequence}.json",
        ),
        policy_sha256=supervisor.policy_sha256,
        status=SupervisedProcessStatus.COMPLETED,
        return_code=0,
        wall_time_ns=1,
        sampled_peak_rss_bytes=None,
        rss_samples=0,
        rss_scope="UNAVAILABLE",
        stdout_bytes=0,
        stdout_sha256=hashlib.sha256(b"").hexdigest(),
        stderr_bytes=0,
        stderr_sha256=hashlib.sha256(b"").hexdigest(),
        stdout_complete=True,
        stderr_complete=True,
        termination_signal_sent=False,
        kill_signal_sent=False,
    )
    snapshot_unique_files = len({
        item.content_sha256 for item in scoring_inventory.entries
    })
    snapshot_input_bytes = sum({
        item.content_sha256: item.byte_size
        for item in scoring_inventory.entries
    }.values())
    publication_status = "ATOMIC_DIRECTORY_RENAME_COMMITTED"
    publication_strategy = "RENAMEAT2_NOREPLACE"
    actual_provider_options_sha256 = opaque(f"provider-options:{sequence}")
    completed_head = content_sha256({
        "schema_version": "cvi.embedding_completed_attempt.v2",
        "prior_attempt_ledger_sha256": (
            precommitment.prior_attempt_ledger_sha256
        ),
        "precommitment_sha256": precommitment.precommitment_sha256,
        "candidate_attempt_token": precommitment.candidate_attempt_token,
        "precommitment_sequence": precommitment.precommitment_sequence,
        "worker_request_sha256": worker_request_sha256,
        "production_receipt_sha256": receipt.receipt_sha256,
        "runtime_library_manifest_sha256": runtime_manifest.manifest_sha256,
        "supervised_process_result_sha256": supervised.result_sha256,
        "actual_provider_options_sha256": actual_provider_options_sha256,
        "snapshot_unique_files": snapshot_unique_files,
        "snapshot_input_bytes": snapshot_input_bytes,
        "code_snapshot_files": len(code_sources),
        "code_snapshot_bytes": len(code_sources),
        "code_snapshot_manifest_sha256": (
            precommitment.code_source_manifest_sha256
        ),
        "publication_status": publication_status,
        "publication_strategy": publication_strategy,
    })
    provider = (
        "CUDAExecutionProvider"
        if config.backend.device == "cuda"
        else "CPUExecutionProvider"
    )
    return EmbeddingFreshWorkerReceipt(
        precommitment_sha256=precommitment.precommitment_sha256,
        precommitment=precommitment,
        worker_request_sha256=worker_request_sha256,
        production_receipt_sha256=receipt.receipt_sha256,
        production_receipt=receipt,
        runtime_library_manifest_sha256=runtime_manifest.manifest_sha256,
        runtime_library_manifest=runtime_manifest,
        worker_environment_identity_sha256=environment_identity.identity_sha256,
        worker_environment_identity=environment_identity,
        onnxruntime_distribution_name=(
            "onnxruntime-gpu" if provider == "CUDAExecutionProvider"
            else "onnxruntime"
        ),
        onnxruntime_distribution_version="fixture",
        actual_providers=(provider,),
        actual_provider_options_sha256=actual_provider_options_sha256,
        snapshot_unique_files=snapshot_unique_files,
        snapshot_input_bytes=snapshot_input_bytes,
        code_snapshot_files=len(code_sources),
        code_snapshot_bytes=len(code_sources),
        code_snapshot_manifest_sha256=(
            precommitment.code_source_manifest_sha256
        ),
        execution_policy_sha256=execution_policy.policy_sha256,
        execution_policy=execution_policy,
        supervised_process_result_sha256=supervised.result_sha256,
        supervised_process_result=supervised,
        publication_status=publication_status,
        publication_strategy=publication_strategy,
        completed_attempt_ledger_head_sha256=completed_head,
    )


def score_policy(**overrides: object) -> ScoreDriftPolicy:
    values: dict[str, object] = {
        "maximum_absolute_score_drift": 0.0,
        "maximum_mean_absolute_score_drift": 0.0,
        "maximum_absolute_margin_drift": 0.0,
        "maximum_mean_absolute_margin_drift": 0.0,
        "maximum_rank_inversions": 0,
        "maximum_queries_with_rank_change": 0,
        "maximum_rank_displacement": 0,
        "maximum_top1_changes": 0,
        "maximum_top_k_set_changes": 0,
        "maximum_top_k_symmetric_difference_items": 0,
        "maximum_top_k_queries_with_set_change_by_k": (0, 0),
        "maximum_top_k_symmetric_difference_items_by_k": (0, 0),
        "maximum_threshold_decision_flips": 0,
        "maximum_reference_reject_candidate_accept_flips": 0,
        "maximum_reference_accept_candidate_reject_flips": 0,
        "maximum_requests": 10,
        "maximum_queries": 4,
        "maximum_candidates_per_query": 4,
        "maximum_scalar_products": 1_000,
        "maximum_embedding_bytes_read": 10_000,
        "maximum_cache_verification_bytes_read": 10_000,
        "maximum_numerical_recomputation_bytes_read": 10_000,
        "maximum_total_vector_bytes_read": 30_000,
        "dot_chunk_floats": 2,
    }
    values.update(overrides)
    return ScoreDriftPolicy(**values)


class ScoreDriftAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = ("a", "b", "c", "q1", "q2")
        self.reference_vectors = {
            "q1": (1.0, 0.0, 0.0),
            "q2": (0.0, 1.0, 0.0),
            "a": (1.0, 0.0, 0.0),
            "b": (0.8, 0.6, 0.0),
            "c": (0.0, 1.0, 0.0),
        }

    def build_case(
        self,
        root: Path,
        candidate_vectors: dict[str, tuple[float, float, float]],
    ) -> dict[str, object]:
        reference_root = root / "reference"
        candidate_root = root / "candidate"
        reference_root.mkdir()
        candidate_root.mkdir()
        reference_config = producer("cpu", HASH_E)
        candidate_config = producer("cuda", HASH_F)
        scoring_inventory = inventory(self.tokens)
        reference_manifest = cache(
            reference_root,
            reference_config,
            scoring_inventory,
            self.reference_vectors,
        )
        candidate_manifest = cache(
            candidate_root,
            candidate_config,
            scoring_inventory,
            candidate_vectors,
        )
        cache_policy = EmbeddingCachePolicy(
            maximum_artifacts=10,
            maximum_unique_vectors=10,
            maximum_vector_dimension=10,
            maximum_vector_bytes=100,
            maximum_total_cache_bytes=1_000,
            scan_chunk_floats=2,
            maximum_normalization_tolerance=1e-5,
        )
        reference_verification = verify_embedding_cache_files(
            root=reference_root,
            inventory=scoring_inventory,
            manifest=reference_manifest,
            policy=cache_policy,
        )
        candidate_verification = verify_embedding_cache_files(
            root=candidate_root,
            inventory=scoring_inventory,
            manifest=candidate_manifest,
            policy=cache_policy,
        )
        numerical_policy = NumericalDriftPolicy(
            absolute_tolerance=2.0,
            relative_tolerance=2.0,
            relative_floor=1e-12,
            maximum_l2_drift=2.0,
            maximum_cosine_drift=2.0,
            maximum_vectors=10,
            maximum_vector_dimension=10,
            maximum_total_bytes_read=1_000,
        )
        numerical = compare_embedding_caches(
            reference_manifest=reference_manifest,
            candidate_manifest=candidate_manifest,
            reference_config=reference_config,
            candidate_config=candidate_config,
            reference_root=reference_root,
            candidate_root=candidate_root,
            policy=numerical_policy,
        )
        reference_production = production_receipt(
            reference_config,
            reference_manifest,
            reference_verification,
            cache_policy,
        )
        candidate_production = production_receipt(
            candidate_config,
            candidate_manifest,
            candidate_verification,
            cache_policy,
        )
        case_workload = workload()
        case_score_policy = score_policy()
        case_boundary = boundary(
            reference_config,
            case_workload,
            case_score_policy,
        )
        precommitment = build_score_drift_precommitment(
            workload=case_workload,
            inventory=scoring_inventory,
            reference_production=reference_production,
            reference_config=reference_config,
            candidate_config=candidate_config,
            numerical_policy=numerical_policy,
            boundary=case_boundary,
            policy=case_score_policy,
            cache_policy=cache_policy,
            prior_attempt_ledger_sha256=opaque("attempt-ledger-before"),
            candidate_attempt_token=opaque("candidate-attempt-1"),
            precommitment_sequence=1,
        )
        admission_plan = build_score_drift_admission_plan(
            workload=case_workload,
            inventory=scoring_inventory,
            reference_production=reference_production,
            candidate_production=candidate_production,
            reference_config=reference_config,
            candidate_config=candidate_config,
            numerical_admission=numerical,
            numerical_policy=numerical_policy,
            boundary=case_boundary,
            policy=case_score_policy,
            cache_policy=cache_policy,
            precommitment=precommitment,
        )
        return locals()

    def compare(self, case: dict[str, object], **overrides: object):
        resolved_workload = overrides.get("workload", case["case_workload"])
        resolved_policy = overrides.get("policy", case["case_score_policy"])
        resolved_boundary = overrides.get("boundary", case["case_boundary"])
        resolved_numerical = overrides.get("numerical", case["numerical"])
        resolved_numerical_policy = overrides.get(
            "numerical_policy",
            case["numerical_policy"],
        )
        resolved_candidate_manifest = overrides.get(
            "candidate_manifest",
            case["candidate_manifest"],
        )
        resolved_candidate_production = overrides.get(
            "candidate_production",
            case["candidate_production"],
        )
        resolved_candidate_verification = overrides.get(
            "candidate_verification",
            case["candidate_verification"],
        )
        resolved_plan = overrides.get("admission_plan")
        if resolved_plan is None:
            resolved_precommitment = build_score_drift_precommitment(
                workload=resolved_workload,
                inventory=case["scoring_inventory"],
                reference_production=case["reference_production"],
                reference_config=case["reference_config"],
                candidate_config=case["candidate_config"],
                numerical_policy=resolved_numerical_policy,
                boundary=resolved_boundary,
                policy=resolved_policy,
                cache_policy=case["cache_policy"],
                prior_attempt_ledger_sha256=opaque("attempt-ledger-before"),
                candidate_attempt_token=opaque("candidate-attempt-1"),
                precommitment_sequence=1,
            )
            resolved_plan = build_score_drift_admission_plan(
                workload=resolved_workload,
                inventory=case["scoring_inventory"],
                reference_production=case["reference_production"],
                candidate_production=resolved_candidate_production,
                reference_config=case["reference_config"],
                candidate_config=case["candidate_config"],
                numerical_admission=resolved_numerical,
                numerical_policy=resolved_numerical_policy,
                boundary=resolved_boundary,
                policy=resolved_policy,
                cache_policy=case["cache_policy"],
                precommitment=resolved_precommitment,
            )
        return compare_score_rank_threshold_drift(
            workload=resolved_workload,
            inventory=case["scoring_inventory"],
            reference_root=case["reference_root"],
            candidate_root=case["candidate_root"],
            reference_manifest=case["reference_manifest"],
            candidate_manifest=resolved_candidate_manifest,
            reference_verification=case["reference_verification"],
            candidate_verification=resolved_candidate_verification,
            reference_production=case["reference_production"],
            candidate_production=resolved_candidate_production,
            reference_config=case["reference_config"],
            candidate_config=case["candidate_config"],
            numerical_admission=resolved_numerical,
            numerical_policy=resolved_numerical_policy,
            boundary=resolved_boundary,
            policy=resolved_policy,
            cache_policy=case["cache_policy"],
            admission_plan=resolved_plan,
            expected_precommitment_sha256=(
                overrides.get(
                    "expected_precommitment_sha256",
                    resolved_plan.precommitment_sha256,
                )
            ),
            expected_admission_plan_sha256=overrides.get(
                "expected_admission_plan_sha256",
                resolved_plan.plan_sha256,
            ),
        )

    def test_identical_scores_rank_and_boundary_pass_but_cannot_promote(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary), self.reference_vectors)
            receipt = self.compare(case)
            self.assertIs(receipt.decision, ScoreDriftDecision.PASS)
            self.assertIs(receipt.promotion_decision, PromotionDecision.INCONCLUSIVE)
            self.assertEqual(receipt.summary.requests, 6)
            self.assertEqual(receipt.summary.queries, 2)
            self.assertEqual(receipt.summary.scalar_products, 36)
            self.assertEqual(receipt.summary.rank_inversions, 0)
            self.assertEqual(receipt.summary.threshold_decision_flips, 0)
            self.assertEqual(receipt.cost.scoring_vector_bytes_read, 192)
            self.assertEqual(receipt.cost.peak_vector_payload_bytes, 36)
            self.assertEqual(
                ScoreDriftAdmissionReceipt.from_dict(receipt.to_dict()),
                receipt,
            )

    def test_rank_topk_margin_and_frozen_decision_drift_fail(self) -> None:
        candidate = dict(self.reference_vectors)
        candidate["q1"] = (0.85, math.sqrt(1.0 - 0.85**2), 0.0)
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary), candidate)
            receipt = self.compare(case)
            self.assertIs(receipt.decision, ScoreDriftDecision.FAIL)
            self.assertGreater(receipt.summary.maximum_absolute_score_drift, 0.0)
            self.assertEqual(receipt.summary.rank_inversions, 1)
            self.assertEqual(receipt.summary.top1_changes, 1)
            self.assertEqual(receipt.summary.top_k_set_changes, 1)
            self.assertEqual(receipt.summary.threshold_decision_flips, 1)
            self.assertEqual(
                receipt.summary.reference_accept_candidate_reject_flips,
                1,
            )
            self.assertEqual(
                receipt.summary.reference_reject_candidate_accept_flips,
                0,
            )
            self.assertIn("RANK_INVERSIONS", receipt.hard_failures)
            self.assertIn("THRESHOLD_DECISION_FLIPS", receipt.hard_failures)
            self.assertIn("TOP1_CHANGES", receipt.hard_failures)
            self.assertIn(
                "REFERENCE_ACCEPT_CANDIDATE_REJECT_FLIPS",
                receipt.hard_failures,
            )
            self.assertIn(
                "TOP_K_SYMMETRIC_DIFFERENCE_ITEMS",
                receipt.hard_failures,
            )

    def test_lineage_boundary_workload_and_cache_mutation_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary), self.reference_vectors)
            with self.assertRaisesRegex(ValueError, "reference model"):
                self.compare(
                    case,
                    boundary=replace(
                        case["case_boundary"],
                        reference_model_sha256=HASH_F,
                    ),
                )
            with self.assertRaisesRegex(ValueError, "another gallery"):
                self.compare(
                    case,
                    boundary=replace(
                        case["case_boundary"],
                        gallery_sha256=HASH_F,
                    ),
                )
            requests = list(workload().requests)
            with self.assertRaisesRegex(ValueError, "canonically ordered"):
                RetrievalScoreWorkload(
                    gallery_sha256=HASH_B,
                    pairing_policy_sha256=HASH_D,
                    retrieval_plan_sha256=HASH_E,
                    split_manifest_sha256=opaque("optimization-dev-split"),
                    workload_construction_receipt_sha256=opaque(
                        "workload-construction"
                    ),
                    data_role="OPTIMIZATION_DEV",
                    selection_blind_to_candidate_outputs=True,
                    requests=tuple(reversed(requests)),
                )
            reduced = tuple(
                item
                for item in case["case_workload"].requests
                if not (
                    item.query_group_token == opaque("query:q2")
                    and item.candidate_slot_token == opaque("slot:c")
                )
            )
            with self.assertRaisesRegex(ValueError, "identical gallery"):
                RetrievalScoreWorkload(
                    gallery_sha256=HASH_B,
                    pairing_policy_sha256=HASH_D,
                    retrieval_plan_sha256=HASH_E,
                    split_manifest_sha256=opaque("optimization-dev-split"),
                    workload_construction_receipt_sha256=opaque(
                        "workload-construction"
                    ),
                    data_role="OPTIMIZATION_DEV",
                    selection_blind_to_candidate_outputs=True,
                    requests=reduced,
                )
            workload_payload = case["case_workload"].to_dict()
            workload_payload["data_role"] = "FINAL_TEST"
            with self.assertRaisesRegex(ValueError, "OPTIMIZATION_DEV"):
                RetrievalScoreWorkload.from_dict(workload_payload)
            with self.assertRaisesRegex(ValueError, "must be positive"):
                replace(case["case_boundary"], margin_threshold=0.0)
            with self.assertRaisesRegex(ValueError, r"in \[-1, 1\]"):
                replace(case["case_boundary"], score_threshold=999.0)
            with self.assertRaisesRegex(ValueError, r"in \(0, 2\]"):
                replace(case["case_boundary"], margin_threshold=3.0)
            candidate_manifest = case["candidate_manifest"]
            candidate_entry = candidate_manifest.entries[0]
            (case["candidate_root"] / candidate_entry.relative_path).write_bytes(
                b"\x00" * candidate_entry.byte_size
            )
            with self.assertRaisesRegex(ValueError, "hash differs"):
                self.compare(case)

    def test_policy_caps_and_numerical_failure_are_rejected(self) -> None:
        candidate = dict(self.reference_vectors)
        candidate["q1"] = (0.0, 1.0, 0.0)
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary), candidate)
            with self.assertRaisesRegex(ValueError, "requests exceed"):
                self.compare(case, policy=score_policy(maximum_requests=5))
            failed_numerical = replace(
                case["numerical"],
                hard_failures=("EMBEDDING_L2_DRIFT",),
                decision=NumericalAdmissionDecision.FAIL,
            )
            case["numerical"] = failed_numerical
            with self.assertRaisesRegex(ValueError, "numerical PASS"):
                self.compare(case)

    def test_forged_numerical_work_accounting_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary), self.reference_vectors)
            forged_summary = replace(
                case["numerical"].summary,
                vectors=case["numerical"].summary.vectors - 1,
                values=case["numerical"].summary.values - 3,
                bytes_read=case["numerical"].summary.bytes_read - 24,
            )
            case["numerical"] = replace(
                case["numerical"],
                summary=forged_summary,
            )
            with self.assertRaisesRegex(ValueError, "work accounting"):
                self.compare(case)

    def test_exact_workload_plan_and_boundary_reject_subset_substitution(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary), self.reference_vectors)
            altered = tuple(
                replace(
                    item,
                    query_artifact_token=(
                        "q2" if item.query_artifact_token == "q1" else "q1"
                    ),
                )
                for item in case["case_workload"].requests
            )
            substituted = RetrievalScoreWorkload(
                gallery_sha256=case["case_workload"].gallery_sha256,
                pairing_policy_sha256=(
                    case["case_workload"].pairing_policy_sha256
                ),
                retrieval_plan_sha256=(
                    case["case_workload"].retrieval_plan_sha256
                ),
                split_manifest_sha256=(
                    case["case_workload"].split_manifest_sha256
                ),
                workload_construction_receipt_sha256=(
                    case[
                        "case_workload"
                    ].workload_construction_receipt_sha256
                ),
                data_role="OPTIMIZATION_DEV",
                selection_blind_to_candidate_outputs=True,
                requests=altered,
            )
            with self.assertRaisesRegex(ValueError, "exact workload"):
                self.compare(
                    case,
                    workload=substituted,
                    admission_plan=case["admission_plan"],
                )

    def test_receipt_parser_recomputes_policy_failures(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary), self.reference_vectors)
            receipt = self.compare(case)
            forged = receipt.to_dict()
            forged["summary"]["maximum_absolute_score_drift"] = 999.0
            forged["summary"]["mean_absolute_score_drift"] = 999.0
            with self.assertRaisesRegex(ValueError, "disagree with policy"):
                ScoreDriftAdmissionReceipt.from_dict(forged)

    def test_precommitment_and_external_anchor_are_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary), self.reference_vectors)
            self.assertNotIn(
                "attempt_ledger_sha256",
                case["case_workload"].to_dict(),
            )
            second = build_score_drift_precommitment(
                workload=case["case_workload"],
                inventory=case["scoring_inventory"],
                reference_production=case["reference_production"],
                reference_config=case["reference_config"],
                candidate_config=case["candidate_config"],
                numerical_policy=case["numerical_policy"],
                boundary=case["case_boundary"],
                policy=case["case_score_policy"],
                cache_policy=case["cache_policy"],
                prior_attempt_ledger_sha256=opaque("attempt-ledger-after"),
                candidate_attempt_token=opaque("candidate-attempt-2"),
                precommitment_sequence=2,
            )
            self.assertEqual(
                second.workload_sha256,
                case["precommitment"].workload_sha256,
            )
            self.assertNotEqual(
                second.precommitment_sha256,
                case["precommitment"].precommitment_sha256,
            )
            with self.assertRaisesRegex(ValueError, "candidate_config_sha256"):
                build_score_drift_admission_plan(
                    workload=case["case_workload"],
                    inventory=case["scoring_inventory"],
                    reference_production=case["reference_production"],
                    candidate_production=case["candidate_production"],
                    reference_config=case["reference_config"],
                    candidate_config=replace(
                        case["candidate_config"],
                        batch_size=1,
                    ),
                    numerical_admission=case["numerical"],
                    numerical_policy=case["numerical_policy"],
                    boundary=case["case_boundary"],
                    policy=case["case_score_policy"],
                    cache_policy=case["cache_policy"],
                    precommitment=case["precommitment"],
                )
            with self.assertRaisesRegex(ValueError, "ledger transition"):
                replace(
                    case["admission_plan"],
                    completed_attempt_ledger_sha256=opaque("fake-next-head"),
                )
            with self.assertRaisesRegex(ValueError, "external anchor"):
                self.compare(
                    case,
                    expected_admission_plan_sha256=opaque("wrong-plan"),
                )
            receipt = self.compare(case)
            verify_score_drift_receipt_external_anchors(
                receipt,
                expected_precommitment_sha256=(
                    case["precommitment"].precommitment_sha256
                ),
                expected_admission_plan_sha256=(
                    case["admission_plan"].plan_sha256
                ),
                expected_receipt_sha256=receipt.receipt_sha256,
            )
            with self.assertRaisesRegex(ValueError, "anchor differs"):
                verify_score_drift_receipt_external_anchors(
                    receipt,
                    expected_precommitment_sha256=opaque("wrong-precommit"),
                    expected_admission_plan_sha256=(
                        case["admission_plan"].plan_sha256
                    ),
                    expected_receipt_sha256=receipt.receipt_sha256,
                )

    def test_result_anchor_rejects_summary_only_pass_forgery(self) -> None:
        candidate = dict(self.reference_vectors)
        candidate["q1"] = (0.85, math.sqrt(1.0 - 0.85**2), 0.0)
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary), candidate)
            genuine = self.compare(case)
            self.assertIs(genuine.decision, ScoreDriftDecision.FAIL)
            forged_summary = replace(
                genuine.summary,
                maximum_absolute_score_drift=0.0,
                mean_absolute_score_drift=0.0,
                maximum_absolute_margin_drift=0.0,
                mean_absolute_margin_drift=0.0,
                rank_inversions=0,
                queries_with_rank_change=0,
                maximum_rank_displacement=0,
                top1_changes=0,
                top_k_set_changes=0,
                top_k_symmetric_difference_items=0,
                top_k_queries_with_set_change=(0, 0),
                top_k_symmetric_difference_items_by_k=(0, 0),
                threshold_decision_flips=0,
                reference_reject_candidate_accept_flips=0,
                reference_accept_candidate_reject_flips=0,
            )
            structurally_valid_forged_pass = replace(
                genuine,
                summary=forged_summary,
                hard_failures=(),
                decision=ScoreDriftDecision.PASS,
            )
            self.assertEqual(
                structurally_valid_forged_pass.admission_plan_sha256,
                genuine.admission_plan_sha256,
            )
            with self.assertRaisesRegex(ValueError, "result anchor differs"):
                verify_score_drift_receipt_external_anchors(
                    structurally_valid_forged_pass,
                    expected_precommitment_sha256=(
                        genuine.precommitment_sha256
                    ),
                    expected_admission_plan_sha256=(
                        genuine.admission_plan_sha256
                    ),
                    expected_receipt_sha256=genuine.receipt_sha256,
                )
    def test_workload_content_aliases_and_self_matches_are_rejected(self) -> None:
        frozen_workload = workload()
        base_inventory = inventory(self.tokens)
        entries = list(base_inventory.entries)
        positions = {item.artifact_token: index for index, item in enumerate(entries)}
        entries[positions["q1"]] = replace(
            entries[positions["q1"]],
            content_sha256=entries[positions["a"]].content_sha256,
        )
        self_match_inventory = replace(base_inventory, entries=tuple(entries))
        with self.assertRaisesRegex(ValueError, "self-match"):
            validate_retrieval_workload_content_separation(
                frozen_workload,
                self_match_inventory,
            )
        entries = list(base_inventory.entries)
        entries[positions["b"]] = replace(
            entries[positions["b"]],
            content_sha256=entries[positions["a"]].content_sha256,
        )
        duplicate_gallery = replace(base_inventory, entries=tuple(entries))
        with self.assertRaisesRegex(ValueError, "duplicate content aliases"):
            validate_retrieval_workload_content_separation(
                frozen_workload,
                duplicate_gallery,
            )

    def test_normalization_production_numerical_and_io_caps_fail_closed(
        self,
    ) -> None:
        candidate = dict(self.reference_vectors)
        candidate["q1"] = (0.0, 1.0, 0.0)
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary), candidate)
            altered_manifest = replace(
                case["candidate_manifest"],
                normalization_tolerance=1e-5,
            )
            altered_verification = replace(
                case["candidate_verification"],
                cache_manifest_sha256=altered_manifest.manifest_sha256,
            )
            altered_production_for_manifest = replace(
                case["candidate_production"],
                cache_manifest=altered_manifest,
                cache_verification=altered_verification,
            )
            altered_numerical = replace(
                case["numerical"],
                candidate_manifest_sha256=altered_manifest.manifest_sha256,
            )
            with self.assertRaisesRegex(
                ValueError,
                "normalization_tolerance differs",
            ):
                self.compare(
                    case,
                    candidate_manifest=altered_manifest,
                    candidate_verification=altered_verification,
                    candidate_production=altered_production_for_manifest,
                    numerical=altered_numerical,
                )
            altered_production = replace(
                case["candidate_production"],
                producer_config_sha256=HASH_A,
            )
            with self.assertRaisesRegex(ValueError, "production config"):
                self.compare(
                    case,
                    candidate_production=altered_production,
                )
            strict_numerical = NumericalDriftPolicy(
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                relative_floor=1e-12,
                maximum_l2_drift=0.0,
                maximum_cosine_drift=0.0,
                maximum_vectors=10,
                maximum_vector_dimension=10,
                maximum_total_bytes_read=1_000,
            )
            forged_pass = replace(
                case["numerical"],
                policy_sha256=strict_numerical.policy_sha256,
                hard_failures=(),
                decision=NumericalAdmissionDecision.PASS,
            )
            with self.assertRaisesRegex(ValueError, "cache recomputation"):
                self.compare(
                    case,
                    numerical=forged_pass,
                    numerical_policy=strict_numerical,
                )
            with self.assertRaisesRegex(ValueError, "verification reads"):
                self.compare(
                    case,
                    policy=score_policy(
                        maximum_cache_verification_bytes_read=1,
                    ),
                )

    def test_dense_top_k_uses_exact_incremental_set_drift(self) -> None:
        candidate = dict(self.reference_vectors)
        candidate["q1"] = (0.85, math.sqrt(1.0 - 0.85**2), 0.0)
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary), candidate)
            dense_boundary = replace(case["case_boundary"], top_k=(1, 2, 3))
            dense_policy = score_policy(
                maximum_top_k_queries_with_set_change_by_k=(0, 0, 0),
                maximum_top_k_symmetric_difference_items_by_k=(0, 0, 0),
            )
            receipt = self.compare(
                case,
                boundary=dense_boundary,
                policy=dense_policy,
            )
            self.assertEqual(receipt.summary.top_k_set_changes, 1)
            self.assertEqual(
                receipt.summary.top_k_symmetric_difference_items,
                2,
            )
            self.assertEqual(receipt.summary.top_k_values, (1, 2, 3))
            self.assertEqual(
                receipt.summary.top_k_queries_with_set_change,
                (1, 0, 0),
            )
            self.assertEqual(
                receipt.summary.top_k_symmetric_difference_items_by_k,
                (2, 0, 0),
            )

    def test_protected_cli_writes_hashed_private_no_overwrite_bundle(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = self.build_case(root, self.reference_vectors)
            reference_outer = protected_production_receipt(
                case["reference_config"],
                case["reference_production"],
                case["scoring_inventory"],
                sequence=1,
            )
            candidate_outer = protected_production_receipt(
                case["candidate_config"],
                case["candidate_production"],
                case["scoring_inventory"],
                sequence=2,
            )
            inputs = {
                "workload": workload().to_dict(),
                "inventory": case["scoring_inventory"].to_dict(),
                "reference-cache-manifest": case["reference_manifest"].to_dict(),
                "candidate-cache-manifest": case["candidate_manifest"].to_dict(),
                "reference-cache-verification": case[
                    "reference_verification"
                ].to_dict(),
                "candidate-cache-verification": case[
                    "candidate_verification"
                ].to_dict(),
                "reference-producer-config": case["reference_config"].to_dict(),
                "candidate-producer-config": case["candidate_config"].to_dict(),
                "numerical-admission": {
                    "schema_version": "cvi.numerical_admission_bundle.v1",
                    "receipt_sha256": case["numerical"].receipt_sha256,
                    "receipt": case["numerical"].to_dict(),
                },
                "numerical-policy": case["numerical_policy"].to_dict(),
                "reference-production": {
                    "schema_version": "cvi.embedding_production_bundle.v2",
                    "receipt_sha256": reference_outer.receipt_sha256,
                    "receipt": reference_outer.to_dict(),
                },
                "candidate-production": {
                    "schema_version": "cvi.embedding_production_bundle.v2",
                    "receipt_sha256": candidate_outer.receipt_sha256,
                    "receipt": candidate_outer.to_dict(),
                },
                "frozen-boundary": case["case_boundary"].to_dict(),
                "score-drift-policy": case["case_score_policy"].to_dict(),
                "cache-policy": case["cache_policy"].to_dict(),
            }
            paths: dict[str, Path] = {}
            for name, payload in inputs.items():
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(payload, sort_keys=True),
                    encoding="utf-8",
                )
                paths[name] = path
            generated_precommitment = root / "generated-precommitment.json"
            precommitment_command = [
                sys.executable,
                "workflows/create_score_drift_precommitment.py",
            ]
            for name in (
                "workload",
                "inventory",
                "reference-production",
                "reference-producer-config",
                "candidate-producer-config",
                "numerical-policy",
                "frozen-boundary",
                "score-drift-policy",
                "cache-policy",
            ):
                precommitment_command.extend((f"--{name}", str(paths[name])))
            precommitment_command.extend(
                (
                    "--expected-reference-production-receipt-sha256",
                    reference_outer.receipt_sha256,
                    "--expected-reference-completed-attempt-ledger-head-sha256",
                    reference_outer.completed_attempt_ledger_head_sha256,
                    "--prior-attempt-ledger-sha256",
                    opaque("attempt-ledger-before"),
                    "--candidate-attempt-token",
                    opaque("candidate-attempt-1"),
                    "--precommitment-sequence",
                    "1",
                    "--precommitment",
                    str(generated_precommitment),
                )
            )
            legacy_reference = root / "legacy-reference-production.json"
            legacy_reference.write_text(
                json.dumps(
                    {
                        "schema_version": "cvi.embedding_production_bundle.v1",
                        "receipt_sha256": (
                            case["reference_production"].receipt_sha256
                        ),
                        "receipt": case["reference_production"].to_dict(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            legacy_command = list(precommitment_command)
            reference_index = legacy_command.index("--reference-production") + 1
            legacy_command[reference_index] = str(legacy_reference)
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    legacy_command,
                    cwd=Path(__file__).resolve().parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertFalse(generated_precommitment.exists())
            wrong_anchor_command = list(precommitment_command)
            expected_index = wrong_anchor_command.index(
                "--expected-reference-production-receipt-sha256"
            ) + 1
            wrong_anchor_command[expected_index] = opaque("wrong-outer-anchor")
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    wrong_anchor_command,
                    cwd=Path(__file__).resolve().parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertFalse(generated_precommitment.exists())
            precommitment_completed = subprocess.run(
                precommitment_command,
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            precommitment_status = json.loads(precommitment_completed.stdout)
            self.assertEqual(
                precommitment_status["precommitment_sha256"],
                case["precommitment"].precommitment_sha256,
            )
            self.assertEqual(
                os.stat(generated_precommitment).st_mode & 0o777,
                0o600,
            )
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    precommitment_command,
                    cwd=Path(__file__).resolve().parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            paths["precommitment"] = generated_precommitment
            generated_plan = root / "generated-plan.json"
            plan_command = [
                sys.executable,
                "workflows/create_score_drift_plan.py",
            ]
            for name in (
                "workload",
                "inventory",
                "reference-production",
                "candidate-production",
                "reference-producer-config",
                "candidate-producer-config",
                "numerical-admission",
                "numerical-policy",
                "frozen-boundary",
                "score-drift-policy",
                "cache-policy",
                "precommitment",
            ):
                plan_command.extend((f"--{name}", str(paths[name])))
            plan_command.extend(("--plan", str(generated_plan)))
            plan_command.extend(
                (
                    "--expected-reference-production-receipt-sha256",
                    reference_outer.receipt_sha256,
                    "--expected-reference-completed-attempt-ledger-head-sha256",
                    reference_outer.completed_attempt_ledger_head_sha256,
                    "--expected-candidate-production-receipt-sha256",
                    candidate_outer.receipt_sha256,
                    "--expected-candidate-completed-attempt-ledger-head-sha256",
                    candidate_outer.completed_attempt_ledger_head_sha256,
                )
            )
            plan_completed = subprocess.run(
                plan_command,
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            plan_status = json.loads(plan_completed.stdout)
            generated_plan_payload = json.loads(
                generated_plan.read_text(encoding="utf-8")
            )
            self.assertEqual(
                plan_status["plan_sha256"],
                case["admission_plan"].plan_sha256,
            )
            self.assertEqual(
                generated_plan_payload["plan_sha256"],
                case["admission_plan"].plan_sha256,
            )
            self.assertEqual(os.stat(generated_plan).st_mode & 0o777, 0o600)
            paths["admission-plan"] = generated_plan
            output = root / "receipt.json"
            command = [
                sys.executable,
                "workflows/compare_score_drift.py",
                "--workload",
                str(paths["workload"]),
                "--inventory",
                str(paths["inventory"]),
                "--reference-cache-directory",
                str(case["reference_root"]),
                "--candidate-cache-directory",
                str(case["candidate_root"]),
            ]
            for name in (
                "reference-cache-manifest",
                "candidate-cache-manifest",
                "reference-cache-verification",
                "candidate-cache-verification",
                "reference-production",
                "candidate-production",
                "reference-producer-config",
                "candidate-producer-config",
                "numerical-admission",
                "numerical-policy",
                "frozen-boundary",
                "score-drift-policy",
                "cache-policy",
                "admission-plan",
            ):
                command.extend((f"--{name}", str(paths[name])))
            command.extend(("--receipt", str(output)))
            command.extend(
                (
                    "--expected-reference-production-receipt-sha256",
                    reference_outer.receipt_sha256,
                    "--expected-reference-completed-attempt-ledger-head-sha256",
                    reference_outer.completed_attempt_ledger_head_sha256,
                    "--expected-candidate-production-receipt-sha256",
                    candidate_outer.receipt_sha256,
                    "--expected-candidate-completed-attempt-ledger-head-sha256",
                    candidate_outer.completed_attempt_ledger_head_sha256,
                    "--expected-precommitment-sha256",
                    case["precommitment"].precommitment_sha256,
                    "--expected-admission-plan-sha256",
                    case["admission_plan"].plan_sha256,
                )
            )
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            status = json.loads(completed.stdout)
            bundle = json.loads(output.read_text(encoding="utf-8"))
            parsed = ScoreDriftAdmissionReceipt.from_dict(bundle["receipt"])
            self.assertEqual(status["decision"], ScoreDriftDecision.PASS.value)
            self.assertEqual(bundle["receipt_sha256"], parsed.receipt_sha256)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            verify_command = [
                sys.executable,
                "workflows/verify_score_drift_receipt.py",
                "--receipt",
                str(output),
                "--expected-precommitment-sha256",
                case["precommitment"].precommitment_sha256,
                "--expected-admission-plan-sha256",
                case["admission_plan"].plan_sha256,
                "--expected-receipt-sha256",
                parsed.receipt_sha256,
            ]
            verified = subprocess.run(
                verify_command,
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(verified.stdout)["status"], "VERIFIED")
            wrong_receipt_command = list(verify_command)
            wrong_receipt_command[-1] = opaque("wrong-receipt")
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    wrong_receipt_command,
                    cwd=Path(__file__).resolve().parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            tampered = dict(inputs["numerical-admission"])
            tampered["receipt_sha256"] = "0" * 64
            paths["numerical-admission"].write_text(
                json.dumps(tampered, sort_keys=True),
                encoding="utf-8",
            )
            tampered_output = root / "tampered-receipt.json"
            tampered_command = command[:-1] + [str(tampered_output)]
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    tampered_command,
                    cwd=Path(__file__).resolve().parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertFalse(tampered_output.exists())


if __name__ == "__main__":
    unittest.main()
