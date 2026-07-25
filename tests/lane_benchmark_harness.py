"""Test-only two-pass runtime discovery and strict ONNX benchmark harness."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from cvi.onnx_inference_benchmark import (
    OnnxBenchmarkBackend,
    OnnxInferenceBenchmarkPolicy,
    benchmark_onnx_inference,
)
from cvi.protected_io import read_strict_json_object, write_private_json_bundle
from cvi.runtime_library_provenance import (
    RuntimeLibraryManifest,
    freeze_runtime_library_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("CPU", "CUDA"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--preprocessing", required=True, type=Path)
    parser.add_argument("--artifact", required=True, action="append", type=Path)
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--runtime-library-policy", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    backend = OnnxBenchmarkBackend(args.backend)
    policy = OnnxInferenceBenchmarkPolicy.from_dict(
        read_strict_json_object(args.policy)
    )
    common = {
        "backend": backend,
        "model_path": args.model,
        "backend_config_path": args.backend_config,
        "preprocessing_path": args.preprocessing,
        "artifact_paths": tuple(args.artifact),
        "dependency_lock_path": args.dependency_lock,
        "policy": policy,
    }
    discovery = benchmark_onnx_inference(
        **common,
        runtime_library_policy_path=args.runtime_library_policy,
        code_revision=args.code_revision + "-discovery",
    )
    manifests = tuple(
        RuntimeLibraryManifest.from_dict(
            item["measurement"]["runtime_library_manifest"]
        )
        for item in discovery.worker_results
    )
    strict_policy = freeze_runtime_library_policy(
        discovery.runtime_library_policy,
        manifests,
    )
    with TemporaryDirectory(prefix="cvi-test-strict-runtime-") as temporary:
        strict_path = Path(temporary) / "strict-policy.json"
        write_private_json_bundle(((strict_path, strict_policy.to_dict()),))
        strict = benchmark_onnx_inference(
            **common,
            runtime_library_policy_path=strict_path,
            code_revision=args.code_revision,
        )
    write_private_json_bundle(
        (
            (
                args.receipt,
                {
                    "schema_version": (
                        "cvi.onnx_inference_benchmark_receipt.v3"
                    ),
                    "summary_sha256": strict.summary_sha256,
                    "summary": strict.to_dict(),
                },
            ),
        )
    )


if __name__ == "__main__":
    main()
