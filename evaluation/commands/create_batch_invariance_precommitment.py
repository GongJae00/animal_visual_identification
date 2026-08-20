"""Freeze a batch-composition experiment before any candidate inference.

Commands: precommit (default), evaluate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from shared.contracts.runtime_library_provenance import RuntimeLibraryPolicy
from evaluation.integrity.batch_invariance import (
    BatchInvariancePolicy,
    BatchInvariancePrecommitment,
    batch_artifact_paths_from_dict,
    build_batch_invariance_precommitment,
)
from evaluation.controls.control_scoring import ControlScoringInventory
from shared.foundation.protected_io import (
    read_content_hashed_json_bundle,
    read_strict_json_object,
    write_private_json_bundle,
)
from operations.workers.batch_invariance_runner import (
    BatchFreshWorkerDiscovery,
    BatchFreshWorkerReceipt,
    BatchWorkerExecutionPolicy,
    run_batch_invariance_fresh_worker,
)
from prototype.export.embedding_producer import EmbeddingProducerConfig
from operations.workers.worker_environment import build_sanitized_worker_environment


def _run_precommit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--artifact-paths", required=True, type=Path)
    parser.add_argument("--producer-config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-lineage", required=True, type=Path)
    parser.add_argument("--preprocessing-config", required=True, type=Path)
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--runtime-library-policy", required=True, type=Path)
    parser.add_argument("--worker-execution-policy", required=True, type=Path)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--prior-attempt-ledger-sha256", required=True)
    parser.add_argument("--candidate-attempt-token", required=True)
    parser.add_argument("--precommitment-sequence", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    runtime_policy = RuntimeLibraryPolicy.from_dict(
        read_strict_json_object(args.runtime_library_policy)
    )
    execution_policy = BatchWorkerExecutionPolicy.from_dict(
        read_strict_json_object(args.worker_execution_policy)
    )
    _, worker_environment = build_sanitized_worker_environment(
        os.environ,
        python_executable=args.python_executable,
    )
    precommitment = build_batch_invariance_precommitment(
        inventory=ControlScoringInventory.from_dict(
            read_strict_json_object(args.inventory)
        ),
        artifact_paths=batch_artifact_paths_from_dict(
            read_strict_json_object(args.artifact_paths)
        ),
        producer_config=EmbeddingProducerConfig.from_dict(
            read_strict_json_object(args.producer_config)
        ),
        provenance_paths={
            "model": args.model,
            "model_lineage": args.model_lineage,
            "preprocessing": args.preprocessing_config,
            "dependency_lock": args.dependency_lock,
        },
        policy=BatchInvariancePolicy.from_dict(
            read_strict_json_object(args.policy)
        ),
        runtime_library_policy_sha256=runtime_policy.policy_sha256,
        worker_execution_policy_sha256=execution_policy.policy_sha256,
        worker_environment_identity_sha256=worker_environment.identity_sha256,
        prior_attempt_ledger_sha256=args.prior_attempt_ledger_sha256,
        candidate_attempt_token=args.candidate_attempt_token,
        precommitment_sequence=args.precommitment_sequence,
    )
    output = {
        "schema_version": "evaluation.batch_invariance_precommitment_bundle.v1",
        "precommitment_sha256": precommitment.precommitment_sha256,
        "precommitment": precommitment.to_dict(),
    }
    write_private_json_bundle(((args.output, output),))
    print(json.dumps({
        "status": "PRECOMMITTED",
        "precommitment_sha256": precommitment.precommitment_sha256,
        "artifacts": len(precommitment.artifact_bindings),
        "precommitment_sequence": precommitment.precommitment_sequence,
    }, sort_keys=True))


def _run_evaluate(argv: list[str]) -> None:
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
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--precommitment", required=True, type=Path)
    parser.add_argument("--expected-precommitment-sha256", required=True)
    parser.add_argument("--runtime-library-policy", required=True, type=Path)
    parser.add_argument("--worker-execution-policy", required=True, type=Path)
    parser.add_argument("--python-executable", required=True, type=Path)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--receipt", type=Path)
    output_group.add_argument("--runtime-discovery-output", type=Path)
    args = parser.parse_args(argv)

    precommitment = BatchInvariancePrecommitment.from_dict(
        read_content_hashed_json_bundle(
            args.precommitment,
            schema_version="evaluation.batch_invariance_precommitment_bundle.v1",
            payload_field="precommitment",
            sha256_field="precommitment_sha256",
        )
    )
    execution_policy = BatchWorkerExecutionPolicy.from_dict(
        read_strict_json_object(args.worker_execution_policy)
    )
    discovery = args.runtime_discovery_output is not None
    result = run_batch_invariance_fresh_worker(
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
            "batch_policy": args.policy,
            "precommitment": args.precommitment,
            "runtime_library_policy": args.runtime_library_policy,
        },
        precommitment=precommitment,
        expected_precommitment_sha256=args.expected_precommitment_sha256,
        python_executable=args.python_executable,
        execution_policy=execution_policy,
        discovery=discovery,
    )
    if discovery:
        if not isinstance(result, BatchFreshWorkerDiscovery):
            raise RuntimeError("batch discovery returned an admission receipt")
        output = {
            "schema_version": "operations.batch_runtime_library_discovery_bundle.v2",
            "discovery_sha256": result.discovery_sha256,
            "discovery": result.to_dict(),
        }
        write_private_json_bundle(((args.runtime_discovery_output, output),))
        print(json.dumps({
            "status": "DISCOVERED_NOT_ADMITTED",
            "discovery_sha256": result.discovery_sha256,
            "runtime_library_manifest_sha256": (
                result.runtime_library_manifest_sha256
            ),
            "runtime_library_binary_set_sha256": (
                result.runtime_library_manifest.binary_set_sha256
            ),
            "runtime_library_binary_count": len(
                result.runtime_library_manifest.entries
            ),
        }, sort_keys=True))
        return
    if not isinstance(result, BatchFreshWorkerReceipt):
        raise RuntimeError("batch strict evaluation returned discovery evidence")
    output = {
        "schema_version": "evaluation.batch_invariance_bundle.v4",
        "receipt_sha256": result.receipt_sha256,
        "receipt": result.to_dict(),
    }
    write_private_json_bundle(((args.receipt, output),))
    print(json.dumps({
        "status": "CREATED",
        "decision": result.batch_receipt.decision.value,
        "promotion_decision": result.batch_receipt.promotion_decision.value,
        "receipt_sha256": result.receipt_sha256,
        "batch_receipt_sha256": result.batch_receipt_sha256,
        "backend_calls": result.batch_receipt.summary.backend_calls,
        "artifact_evaluations": (
            result.batch_receipt.summary.artifact_evaluations
        ),
    }, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "precommit"
    if argv and argv[0] in {"precommit", "evaluate"}:
        command = argv[0]
        argv = argv[1:]
    {"precommit": _run_precommit, "evaluate": _run_evaluate}[command](argv)


if __name__ == "__main__":
    main()
