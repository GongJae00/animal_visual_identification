"""Produce a protected embedding cache in one sanitized fresh worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from systems.workers.embedding_production_runner import (
    EmbeddingFreshWorkerDiscovery,
    EmbeddingFreshWorkerReceipt,
    EmbeddingProductionPrecommitment,
    EmbeddingWorkerExecutionPolicy,
    cleanup_published_embedding_cache,
    run_embedding_production_fresh_worker,
)
from foundation.protected_io import read_strict_json_object, write_private_json_bundle


def _precommitment(path: Path) -> EmbeddingProductionPrecommitment:
    payload = read_strict_json_object(path)
    if set(payload) != {
        "schema_version", "precommitment_sha256", "precommitment"
    } or payload["schema_version"] != (
        "cvi.embedding_production_precommitment_bundle.v1"
    ):
        raise ValueError("embedding precommitment bundle differs")
    value = EmbeddingProductionPrecommitment.from_dict(payload["precommitment"])
    if value.precommitment_sha256 != payload["precommitment_sha256"]:
        raise ValueError("embedding precommitment bundle hash differs")
    return value


def main() -> None:
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
    args = parser.parse_args()

    precommitment = _precommitment(args.precommitment)
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


if __name__ == "__main__":
    main()
