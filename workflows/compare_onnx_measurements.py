"""Join matched ONNX measurements with numerical admission metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluation.control_scoring import EmbeddingCacheManifest
from operations.embedding_producer import EmbeddingProducerConfig
from evaluation.measurement_comparison import compare_paired_inference_measurements
from evaluation.numerical_admission import NumericalAdmissionReceipt
from operations.onnx_inference_benchmark import OnnxInferenceBenchmarkSummary
from foundation.protected_io import (
    read_content_hashed_json_bundle,
    read_strict_json_object,
    write_private_json_bundle,
)


def _benchmark_summary(path: Path) -> OnnxInferenceBenchmarkSummary:
    return OnnxInferenceBenchmarkSummary.from_dict(
        read_content_hashed_json_bundle(
            path,
            schema_version="cvi.onnx_inference_benchmark_receipt.v3",
            payload_field="summary",
            sha256_field="summary_sha256",
        )
    )


def _numerical_receipt(path: Path) -> NumericalAdmissionReceipt:
    return NumericalAdmissionReceipt.from_dict(
        read_content_hashed_json_bundle(
            path,
            schema_version="cvi.numerical_admission_bundle.v1",
            payload_field="receipt",
            sha256_field="receipt_sha256",
        )
    )


def _typed_payload(path: Path, name: str) -> dict[str, Any]:
    payload = read_strict_json_object(path)
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be an object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-benchmark", required=True, type=Path)
    parser.add_argument("--candidate-benchmark", required=True, type=Path)
    parser.add_argument("--reference-producer-config", required=True, type=Path)
    parser.add_argument("--candidate-producer-config", required=True, type=Path)
    parser.add_argument("--reference-cache-manifest", required=True, type=Path)
    parser.add_argument("--candidate-cache-manifest", required=True, type=Path)
    parser.add_argument("--numerical-admission", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    receipt = compare_paired_inference_measurements(
        reference=_benchmark_summary(args.reference_benchmark),
        candidate=_benchmark_summary(args.candidate_benchmark),
        reference_config=EmbeddingProducerConfig.from_dict(
            _typed_payload(
                args.reference_producer_config,
                "reference producer config",
            )
        ),
        candidate_config=EmbeddingProducerConfig.from_dict(
            _typed_payload(
                args.candidate_producer_config,
                "candidate producer config",
            )
        ),
        reference_manifest=EmbeddingCacheManifest.from_dict(
            _typed_payload(
                args.reference_cache_manifest,
                "reference cache manifest",
            )
        ),
        candidate_manifest=EmbeddingCacheManifest.from_dict(
            _typed_payload(
                args.candidate_cache_manifest,
                "candidate cache manifest",
            )
        ),
        numerical_admission=_numerical_receipt(args.numerical_admission),
    )
    output = {
        "schema_version": "cvi.paired_inference_measurement_bundle.v1",
        "receipt_sha256": receipt.receipt_sha256,
        "receipt": receipt.to_dict(),
    }
    write_private_json_bundle(((args.receipt, output),))


if __name__ == "__main__":
    main()
