"""Run a bounded fresh-process ONNX inference measurement."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvi.onnx_inference_benchmark import (
    OnnxBenchmarkBackend,
    OnnxInferenceBenchmarkPolicy,
    benchmark_onnx_inference,
)
from cvi.protected_io import read_strict_json_object, write_private_json_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=[item.value for item in OnnxBenchmarkBackend],
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--preprocessing", required=True, type=Path)
    parser.add_argument(
        "--artifact",
        required=True,
        action="append",
        type=Path,
        help="Repeat in the exact frozen batch order.",
    )
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument(
        "--runtime-library-policy",
        required=True,
        type=Path,
    )
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    policy = OnnxInferenceBenchmarkPolicy.from_dict(
        read_strict_json_object(args.policy)
    )
    summary = benchmark_onnx_inference(
        backend=OnnxBenchmarkBackend(args.backend),
        model_path=args.model,
        backend_config_path=args.backend_config,
        preprocessing_path=args.preprocessing,
        artifact_paths=tuple(args.artifact),
        dependency_lock_path=args.dependency_lock,
        runtime_library_policy_path=args.runtime_library_policy,
        code_revision=args.code_revision,
        policy=policy,
    )
    receipt = {
        "schema_version": "cvi.onnx_inference_benchmark_receipt.v3",
        "summary_sha256": summary.summary_sha256,
        "summary": summary.to_dict(),
    }
    write_private_json_bundle(((args.receipt, receipt),))


if __name__ == "__main__":
    main()
