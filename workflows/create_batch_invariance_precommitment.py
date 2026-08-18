"""Freeze a batch-composition experiment before any candidate inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from contracts.runtime_library_provenance import RuntimeLibraryPolicy
from evaluation.integrity.batch_invariance import (
    BatchInvariancePolicy,
    batch_artifact_paths_from_dict,
    build_batch_invariance_precommitment,
)
from evaluation.controls.control_scoring import ControlScoringInventory
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from systems.workers.batch_invariance_runner import BatchWorkerExecutionPolicy
from systems.inference.embedding_producer import EmbeddingProducerConfig
from systems.workers.worker_environment import build_sanitized_worker_environment

def main() -> None:
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
    args = parser.parse_args()

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
        "schema_version": "cvi.batch_invariance_precommitment_bundle.v1",
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


if __name__ == "__main__":
    main()
