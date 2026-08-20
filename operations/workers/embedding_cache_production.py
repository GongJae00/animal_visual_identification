"""Produce a protected embedding cache in one sanitized fresh worker.

Commands: produce (default), precommit, verify.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from shared.contracts.runtime_library_provenance import RuntimeLibraryPolicy
from evaluation.controls.control_scoring import ControlScoringInventory, EmbeddingCachePolicy
from shared.foundation.protected_io import (
    read_content_hashed_json_bundle,
    read_strict_json_object,
    write_private_json_bundle,
)
from prototype.export.embedding_producer import (
    EmbeddingProducerConfig,
    EmbeddingProductionPolicy,
)
from prototype.export.onnx_backend import OnnxRuntimeBackendConfig
from operations.workers.embedding_production_runner import (
    EmbeddingFreshWorkerDiscovery,
    EmbeddingFreshWorkerReceipt,
    EmbeddingProductionPrecommitment,
    EmbeddingWorkerExecutionPolicy,
    build_embedding_production_precommitment,
    cleanup_published_embedding_cache,
    embedding_artifact_paths_from_dict,
    read_embedding_production_outer_bundle,
    run_embedding_production_fresh_worker,
)
from operations.workers.worker_environment import build_sanitized_worker_environment


def _run_produce(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--artifact-paths", required=True, type=Path)
    parser.add_argument("--producer-config", required=True, type=Path)
    parser.add_argument("--onnx-config", required=True, type=Path)
    parser.add_argument("--preprocessing-config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-lineage", required=True, type=Path)
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--production-policy", required=True, type=Path)
    parser.add_argument("--cache-policy", required=True, type=Path)
    parser.add_argument("--precommitment", required=True, type=Path)
    parser.add_argument("--expected-precommitment-sha256", required=True)
    parser.add_argument("--runtime-library-policy", required=True, type=Path)
    parser.add_argument("--worker-execution-policy", required=True, type=Path)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--receipt", type=Path)
    output_group.add_argument("--runtime-discovery-output", type=Path)
    args = parser.parse_args(argv)

    precommitment = EmbeddingProductionPrecommitment.from_dict(
        read_content_hashed_json_bundle(
            args.precommitment,
            schema_version="cvi.embedding_production_precommitment_bundle.v1",
            payload_field="precommitment",
            sha256_field="precommitment_sha256",
        )
    )
    execution_policy = EmbeddingWorkerExecutionPolicy.from_dict(
        read_strict_json_object(args.worker_execution_policy)
    )
    discovery = args.runtime_discovery_output is not None
    result = run_embedding_production_fresh_worker(
        backend=args.backend,
        files={
            "inventory": args.inventory,
            "artifact_paths": args.artifact_paths,
            "producer_config": args.producer_config,
            "onnx_config": args.onnx_config,
            "preprocessing": args.preprocessing_config,
            "model": args.model,
            "model_lineage": args.model_lineage,
            "dependency_lock": args.dependency_lock,
            "production_policy": args.production_policy,
            "cache_policy": args.cache_policy,
            "precommitment": args.precommitment,
            "runtime_library_policy": args.runtime_library_policy,
        },
        precommitment=precommitment,
        expected_precommitment_sha256=args.expected_precommitment_sha256,
        python_executable=args.python_executable,
        execution_policy=execution_policy,
        output_directory=args.output_directory,
        discovery=discovery,
    )
    if discovery:
        if not isinstance(result, EmbeddingFreshWorkerDiscovery):
            raise RuntimeError("embedding discovery returned an admission receipt")
        bundle = {
            "schema_version": "cvi.embedding_runtime_discovery_bundle.v1",
            "discovery_sha256": result.discovery_sha256,
            "discovery": result.to_dict(),
        }
        write_private_json_bundle(((args.runtime_discovery_output, bundle),))
        print(json.dumps({
            "status": "DISCOVERED_NOT_ADMITTED",
            "discovery_sha256": result.discovery_sha256,
            "runtime_library_manifest_sha256": (
                result.runtime_library_manifest_sha256
            ),
            "runtime_library_binary_set_sha256": (
                result.runtime_library_manifest.binary_set_sha256
            ),
        }, sort_keys=True))
        return
    if not isinstance(result, EmbeddingFreshWorkerReceipt):
        raise RuntimeError("embedding strict execution returned discovery")
    bundle = {
        "schema_version": "cvi.embedding_production_bundle.v2",
        "receipt_sha256": result.receipt_sha256,
        "receipt": result.to_dict(),
    }
    try:
        write_private_json_bundle(((args.receipt, bundle),))
    except BaseException:
        cleanup_published_embedding_cache(args.output_directory, result)
        raise
    print(json.dumps({
        "status": "CREATED",
        "receipt_sha256": result.receipt_sha256,
        "production_receipt_sha256": result.production_receipt_sha256,
        "completed_attempt_ledger_head_sha256": (
            result.completed_attempt_ledger_head_sha256
        ),
        "unique_vectors": len(result.production_receipt.cache_manifest.entries),
        "cache_bytes": result.production_receipt.cost.output_bytes_written,
    }, sort_keys=True))


def _run_precommit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--artifact-paths", required=True, type=Path)
    parser.add_argument("--producer-config", required=True, type=Path)
    parser.add_argument("--onnx-config", required=True, type=Path)
    parser.add_argument("--preprocessing-config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-lineage", required=True, type=Path)
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--production-policy", required=True, type=Path)
    parser.add_argument("--cache-policy", required=True, type=Path)
    parser.add_argument("--runtime-library-policy", required=True, type=Path)
    parser.add_argument("--worker-execution-policy", required=True, type=Path)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--prior-attempt-ledger-sha256", required=True)
    parser.add_argument("--candidate-attempt-token", required=True)
    parser.add_argument("--precommitment-sequence", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    inventory = ControlScoringInventory.from_dict(
        read_strict_json_object(args.inventory)
    )
    artifact_paths = embedding_artifact_paths_from_dict(
        read_strict_json_object(args.artifact_paths)
    )
    producer = EmbeddingProducerConfig.from_dict(
        read_strict_json_object(args.producer_config)
    )
    onnx_config = OnnxRuntimeBackendConfig.from_dict(
        read_strict_json_object(args.onnx_config)
    )
    if onnx_config.config_sha256 != producer.backend.backend_config_sha256:
        raise ValueError("ONNX config differs from producer backend identity")
    production_policy = EmbeddingProductionPolicy.from_dict(
        read_strict_json_object(args.production_policy)
    )
    cache_policy = EmbeddingCachePolicy.from_dict(
        read_strict_json_object(args.cache_policy)
    )
    runtime_policy = RuntimeLibraryPolicy.from_dict(
        read_strict_json_object(args.runtime_library_policy)
    )
    execution_policy = EmbeddingWorkerExecutionPolicy.from_dict(
        read_strict_json_object(args.worker_execution_policy)
    )
    _, environment = build_sanitized_worker_environment(
        os.environ,
        python_executable=args.python_executable,
    )
    precommitment = build_embedding_production_precommitment(
        inventory=inventory,
        artifact_paths=artifact_paths,
        producer_config=producer,
        provenance_paths={
            "model": args.model,
            "model_lineage": args.model_lineage,
            "onnx_config": args.onnx_config,
            "preprocessing": args.preprocessing_config,
            "dependency_lock": args.dependency_lock,
        },
        production_policy=production_policy,
        cache_policy=cache_policy,
        runtime_library_policy_sha256=runtime_policy.policy_sha256,
        worker_execution_policy_sha256=execution_policy.policy_sha256,
        worker_environment_identity_sha256=environment.identity_sha256,
        prior_attempt_ledger_sha256=args.prior_attempt_ledger_sha256,
        candidate_attempt_token=args.candidate_attempt_token,
        precommitment_sequence=args.precommitment_sequence,
    )
    bundle = {
        "schema_version": "cvi.embedding_production_precommitment_bundle.v1",
        "precommitment_sha256": precommitment.precommitment_sha256,
        "precommitment": precommitment.to_dict(),
    }
    write_private_json_bundle(((args.output, bundle),))
    print(json.dumps({
        "status": "PRECOMMITTED",
        "precommitment_sha256": precommitment.precommitment_sha256,
        "artifacts": len(precommitment.artifact_bindings),
        "precommitment_sequence": precommitment.precommitment_sequence,
    }, sort_keys=True))


def _run_verify(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-precommitment-sha256", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-completed-attempt-ledger-head-sha256",
        required=True,
    )
    args = parser.parse_args(argv)
    receipt = read_embedding_production_outer_bundle(
        args.receipt,
        expected_receipt_sha256=args.expected_receipt_sha256,
        expected_completed_attempt_ledger_head_sha256=(
            args.expected_completed_attempt_ledger_head_sha256
        ),
    )
    if receipt.precommitment_sha256 != args.expected_precommitment_sha256:
        raise ValueError("embedding receipt differs from external precommitment")
    print(json.dumps({
        "status": "VERIFIED",
        "precommitment_sha256": receipt.precommitment_sha256,
        "receipt_sha256": receipt.receipt_sha256,
        "production_receipt_sha256": receipt.production_receipt_sha256,
        "completed_attempt_ledger_head_sha256": (
            receipt.completed_attempt_ledger_head_sha256
        ),
    }, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "produce"
    if argv and argv[0] in {"produce", "precommit", "verify"}:
        command = argv[0]
        argv = argv[1:]
    {
        "produce": _run_produce,
        "precommit": _run_precommit,
        "verify": _run_verify,
    }[command](argv)


if __name__ == "__main__":
    main()
