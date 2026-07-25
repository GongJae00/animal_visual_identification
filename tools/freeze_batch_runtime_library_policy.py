"""Freeze a reviewed strict runtime policy from repeated batch discoveries."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvi.batch_invariance_runner import BatchFreshWorkerDiscovery
from cvi.protected_io import read_strict_json_object, write_private_json_bundle
from cvi.runtime_library_provenance import (
    RuntimeLibraryManifest,
    RuntimeLibraryPolicy,
    freeze_runtime_library_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-policy", required=True, type=Path)
    parser.add_argument(
        "--discovery-manifest", required=True, action="append", type=Path
    )
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--freeze-receipt", required=True, type=Path)
    args = parser.parse_args()
    if len(args.discovery_manifest) < 2:
        raise ValueError("batch runtime policy freeze requires two discoveries")

    discovery_policy = RuntimeLibraryPolicy.from_dict(
        read_strict_json_object(args.discovery_policy)
    )
    manifests: list[RuntimeLibraryManifest] = []
    precommitment_hashes: list[str] = []
    manifest_hashes: list[str] = []
    discovery_hashes: list[str] = []
    environment_hashes: set[str] = set()
    execution_policy_hashes: set[str] = set()
    distribution_identities: set[tuple[str, str]] = set()
    for path in args.discovery_manifest:
        payload = read_strict_json_object(path)
        if set(payload) != {
            "schema_version", "discovery_sha256", "discovery"
        } or payload["schema_version"] != (
            "cvi.batch_runtime_library_discovery_bundle.v2"
        ):
            raise ValueError("batch runtime discovery bundle schema differs")
        discovery = BatchFreshWorkerDiscovery.from_dict(payload["discovery"])
        if discovery.discovery_sha256 != payload["discovery_sha256"]:
            raise ValueError("batch runtime discovery hash differs")
        manifest = discovery.runtime_library_manifest
        if discovery_policy.policy_sha256 != manifest.policy_sha256:
            raise ValueError("batch runtime discovery policy differs")
        manifests.append(manifest)
        precommitment_hashes.append(discovery.precommitment_sha256)
        manifest_hashes.append(manifest.manifest_sha256)
        discovery_hashes.append(discovery.discovery_sha256)
        environment_hashes.add(discovery.worker_environment_identity_sha256)
        execution_policy_hashes.add(discovery.execution_policy_sha256)
        distribution_identities.add((
            discovery.onnxruntime_distribution_name,
            discovery.onnxruntime_distribution_version,
        ))
    if any(
        len(values) != 1
        for values in (
            environment_hashes,
            execution_policy_hashes,
            distribution_identities,
        )
    ):
        raise ValueError("batch runtime discovery worker lanes differ")

    strict = freeze_runtime_library_policy(
        discovery_policy,
        tuple(manifests),
    )
    receipt = {
        "schema_version": "cvi.batch_runtime_library_policy_freeze_receipt.v1",
        "discovery_policy_sha256": discovery_policy.policy_sha256,
        "discovery_precommitment_sha256": precommitment_hashes,
        "discovery_receipt_sha256": discovery_hashes,
        "discovery_manifest_sha256": manifest_hashes,
        "worker_environment_identity_sha256": next(iter(environment_hashes)),
        "worker_execution_policy_sha256": next(iter(execution_policy_hashes)),
        "onnxruntime_distribution": list(next(iter(distribution_identities))),
        "discovery_binary_set_sha256": manifests[0].binary_set_sha256,
        "strict_policy_sha256": strict.policy_sha256,
        "expected_binary_count": len(strict.expected_binaries),
        "interpretation": (
            "CANDIDATE_POLICY_REQUIRES_PATH_REVIEW_AND_STRICT_RERUN"
        ),
    }
    write_private_json_bundle(
        (
            (args.policy, strict.to_dict()),
            (args.freeze_receipt, receipt),
        )
    )


if __name__ == "__main__":
    main()
