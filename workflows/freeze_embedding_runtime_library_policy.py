"""Freeze a reviewed strict runtime policy from embedding discoveries."""

from __future__ import annotations

import argparse
from pathlib import Path

from contracts.runtime_library_provenance import (
    RuntimeLibraryPolicy,
    freeze_runtime_library_policy,
)
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from operations.embedding_production_runner import EmbeddingFreshWorkerDiscovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-policy", required=True, type=Path)
    parser.add_argument(
        "--discovery-receipt", required=True, action="append", type=Path
    )
    parser.add_argument(
        "--expected-discovery-sha256",
        required=True,
        action="append",
    )
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--freeze-receipt", required=True, type=Path)
    args = parser.parse_args()
    if len(args.discovery_receipt) < 2:
        raise ValueError("embedding runtime freeze requires two discoveries")
    if len(args.expected_discovery_sha256) != len(args.discovery_receipt):
        raise ValueError("embedding discovery anchors and receipts differ")

    discovery_policy = RuntimeLibraryPolicy.from_dict(
        read_strict_json_object(args.discovery_policy)
    )
    discoveries: list[EmbeddingFreshWorkerDiscovery] = []
    for path, expected_hash in zip(
        args.discovery_receipt,
        args.expected_discovery_sha256,
        strict=True,
    ):
        payload = read_strict_json_object(path)
        if set(payload) != {
            "schema_version", "discovery_sha256", "discovery"
        } or payload["schema_version"] != (
            "cvi.embedding_runtime_discovery_bundle.v1"
        ):
            raise ValueError("embedding runtime discovery bundle differs")
        discovery = EmbeddingFreshWorkerDiscovery.from_dict(payload["discovery"])
        if discovery.discovery_sha256 != payload["discovery_sha256"] or (
            discovery.discovery_sha256 != expected_hash
        ) or (
            discovery.runtime_library_manifest.policy_sha256
            != discovery_policy.policy_sha256
        ):
            raise ValueError("embedding runtime discovery hash differs")
        discoveries.append(discovery)
    if len({item.discovery_sha256 for item in discoveries}) != len(discoveries):
        raise ValueError("embedding runtime discoveries must be distinct")
    if len({
        item.precommitment.candidate_attempt_token for item in discoveries
    }) != len(discoveries):
        raise ValueError("embedding discovery attempt tokens must be distinct")
    for previous, current in zip(discoveries, discoveries[1:]):
        if current.precommitment.precommitment_sequence <= (
            previous.precommitment.precommitment_sequence
        ) or current.precommitment.prior_attempt_ledger_sha256 != (
            previous.completed_attempt_ledger_head_sha256
        ):
            raise ValueError("embedding discovery attempt ledger is not chained")
    first = discoveries[0]
    workload = _workload_identity(first)
    if any(_workload_identity(item) != workload for item in discoveries[1:]):
        raise ValueError("embedding discovery workloads differ")

    strict = freeze_runtime_library_policy(
        discovery_policy,
        tuple(item.runtime_library_manifest for item in discoveries),
    )
    receipt = {
        "schema_version": "cvi.embedding_runtime_policy_freeze_receipt.v1",
        "discovery_policy_sha256": discovery_policy.policy_sha256,
        "discovery_receipt_sha256": [
            item.discovery_sha256 for item in discoveries
        ],
        "discovery_precommitment_sha256": [
            item.precommitment_sha256 for item in discoveries
        ],
        "discovery_manifest_sha256": [
            item.runtime_library_manifest_sha256 for item in discoveries
        ],
        "worker_environment_identity_sha256": (
            first.worker_environment_identity_sha256
        ),
        "worker_execution_policy_sha256": first.execution_policy_sha256,
        "onnxruntime_distribution": [
            first.onnxruntime_distribution_name,
            first.onnxruntime_distribution_version,
        ],
        "discovery_binary_set_sha256": (
            first.runtime_library_manifest.binary_set_sha256
        ),
        "strict_policy_sha256": strict.policy_sha256,
        "expected_binary_count": len(strict.expected_binaries),
        "interpretation": (
            "CANDIDATE_POLICY_REQUIRES_PATH_REVIEW_AND_STRICT_RERUN"
        ),
    }
    write_private_json_bundle(
        ((args.policy, strict.to_dict()), (args.freeze_receipt, receipt))
    )


def _workload_identity(value: EmbeddingFreshWorkerDiscovery) -> tuple[object, ...]:
    precommitment = value.precommitment
    return (
        precommitment.scoring_inventory_sha256,
        precommitment.producer_config_sha256,
        precommitment.production_policy_sha256,
        precommitment.cache_policy_sha256,
        precommitment.backend_identity_sha256,
        precommitment.runtime_library_policy_sha256,
        precommitment.worker_execution_policy_sha256,
        precommitment.worker_environment_identity_sha256,
        precommitment.artifact_bindings,
        precommitment.provenance_sha256,
        precommitment.code_source_manifest_sha256,
        precommitment.code_source_files,
        precommitment.code_source_bytes,
        precommitment.worker_bootstrap_sha256,
        value.onnxruntime_distribution_name,
        value.onnxruntime_distribution_version,
        value.actual_providers,
        value.actual_provider_options_sha256,
    )


if __name__ == "__main__":
    main()
