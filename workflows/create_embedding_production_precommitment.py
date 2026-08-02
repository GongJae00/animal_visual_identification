"""Freeze one label-blind embedding production attempt before inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaluation.control_scoring import ControlScoringInventory, EmbeddingCachePolicy
from operations.embedding_producer import EmbeddingProducerConfig, EmbeddingProductionPolicy
from operations.embedding_production_runner import (
    EmbeddingWorkerExecutionPolicy,
    build_embedding_production_precommitment,
    embedding_artifact_paths_from_dict,
)
from operations.onnx_backend import OnnxRuntimeBackendConfig
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from artifact_contracts.runtime_library_provenance import RuntimeLibraryPolicy
from operations.worker_environment import build_sanitized_worker_environment


def main() -> None:
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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
