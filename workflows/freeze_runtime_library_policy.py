"""Freeze a reviewed strict runtime-binary candidate from discovery evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from contracts.runtime_library_provenance import (
    RuntimeLibraryManifest,
    freeze_runtime_library_policy,
)
from foundation.protected_io import (
    read_content_hashed_json_bundle,
    write_private_json_bundle,
)
from systems.measurement.onnx_inference_benchmark import OnnxInferenceBenchmarkSummary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-receipt", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--freeze-receipt", required=True, type=Path)
    args = parser.parse_args()

    summary = OnnxInferenceBenchmarkSummary.from_dict(
        read_content_hashed_json_bundle(
            args.discovery_receipt,
            schema_version="cvi.onnx_inference_benchmark_receipt.v3",
            payload_field="summary",
            sha256_field="summary_sha256",
        )
    )
    manifests = tuple(
        RuntimeLibraryManifest.from_dict(
            item["measurement"]["runtime_library_manifest"]
        )
        for item in summary.worker_results
    )
    strict = freeze_runtime_library_policy(
        summary.runtime_library_policy,
        manifests,
    )
    receipt = {
        "schema_version": "cvi.runtime_library_policy_freeze_receipt.v1",
        "discovery_summary_sha256": summary.summary_sha256,
        "discovery_policy_sha256": (
            summary.runtime_library_policy.policy_sha256
        ),
        "discovery_binary_set_sha256": (
            summary.runtime_library_binary_set_sha256
        ),
        "strict_policy_sha256": strict.policy_sha256,
        "expected_binary_count": len(strict.expected_binaries),
        "interpretation": (
            "CANDIDATE_POLICY_REQUIRES_REVIEW_AND_STRICT_RERUN"
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
