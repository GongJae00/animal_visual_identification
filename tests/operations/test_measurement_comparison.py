from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from data.acquisition import sha256_file
from evaluation.controls.control_scoring import (
    ArtifactCacheBinding,
    EmbeddingCacheEntry,
    EmbeddingCacheManifest,
    embedding_cache_key,
)

from tests.repo_root import REPO_ROOT
from evaluation.integrity.measurement_comparison import (
    MeasurementAdmissionDecision,
    PairedInferenceMeasurementReceipt,
    compare_paired_inference_measurements,
)
from evaluation.integrity.numerical_admission import (
    NumericalAdmissionDecision,
    NumericalDriftPolicy,
    compare_embedding_caches,
)
from prototype.export.embedding_producer import (
    EmbeddingBackendIdentity,
    EmbeddingProducerConfig,
)
from operations.measurement.onnx_inference_benchmark import (
    OnnxBenchmarkBackend,
    OnnxInferenceBenchmarkSummary,
)
from identification.training.appearance.optimization import PromotionDecision
from tests.operations.test_onnx_inference_benchmark import (
    OPTIONAL_CUDA_ONNX_AVAILABLE,
    benchmark_policy,
    write_inputs,
)

INVENTORY_SHA256 = "9" * 64
LINEAGE_SHA256 = "8" * 64

def run_isolated_lane_benchmark(
    root: Path,
    paths: dict[str, Path],
    *,
    backend: OnnxBenchmarkBackend,
) -> OnnxInferenceBenchmarkSummary:
    cuda = backend is OnnxBenchmarkBackend.CUDA
    policy_path = root / f"{backend.value.lower()}-policy.json"
    receipt_path = root / f"{backend.value.lower()}-benchmark.json"
    policy_path.write_text(
        json.dumps(
            benchmark_policy(cuda=cuda, fresh_processes=1).to_dict()
        )
        + "\n"
    )
    subprocess.run(
        (
            "uv",
            "run",
            "--isolated",
            "--extra",
            "cuda" if cuda else "cpu",
            "python",
            "tests/lane_benchmark_harness.py",
            "--backend",
            backend.value,
            "--model",
            str(paths["model"]),
            "--backend-config",
            str(paths["backend"]),
            "--preprocessing",
            str(paths["preprocessing"]),
            "--artifact",
            str(paths["image_a"]),
            "--artifact",
            str(paths["image_b"]),
            "--dependency-lock",
            str(paths["lock"]),
            "--runtime-library-policy",
            str(paths["runtime_policy"]),
            "--code-revision",
            "synthetic-paired-measurement",
            "--policy",
            str(policy_path),
            "--receipt",
            str(receipt_path),
        ),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    payload = json.loads(receipt_path.read_text())
    if payload.get("schema_version") != (
        "cvi.onnx_inference_benchmark_receipt.v3"
    ):
        raise AssertionError("isolated benchmark receipt schema differs")
    summary = OnnxInferenceBenchmarkSummary.from_dict(payload["summary"])
    if summary.summary_sha256 != payload["summary_sha256"]:
        raise AssertionError("isolated benchmark summary hash differs")
    return summary

def producer_config(
    summary: OnnxInferenceBenchmarkSummary,
) -> EmbeddingProducerConfig:
    identity = EmbeddingBackendIdentity.from_dict(
        summary.worker_results[0]["measurement"]["backend_identity"]
    )
    shape = summary.tensor_shape
    return EmbeddingProducerConfig(
        model_sha256=summary.model_sha256,
        model_lineage_sha256=LINEAGE_SHA256,
        preprocessing_sha256=summary.preprocessing_file_sha256,
        preprocessing_semantics_sha256=(
            summary.preprocessing_config_sha256
        ),
        dependency_lock_sha256=summary.dependency_lock_sha256,
        code_revision=summary.code_revision,
        backend=identity,
        vector_dimension=summary.vector_dimension,
        batch_size=shape[0],
        input_width=shape[3],
        input_height=shape[2],
        input_channels=shape[1],
        input_value_bytes=4,
        l2_epsilon=1e-12,
        normalization_tolerance=1e-6,
    )

def cache_manifest(
    root: Path,
    config: EmbeddingProducerConfig,
    artifact_hashes: tuple[str, ...],
) -> EmbeddingCacheManifest:
    bindings: list[ArtifactCacheBinding] = []
    entries: list[EmbeddingCacheEntry] = []
    for index, artifact_hash in enumerate(artifact_hashes):
        cache_key = embedding_cache_key(
            artifact_content_sha256=artifact_hash,
            model_sha256=config.model_sha256,
            inference_config_sha256=config.config_sha256,
            dependency_lock_sha256=config.dependency_lock_sha256,
            code_revision=config.code_revision,
            precision=config.backend.precision,
            vector_dimension=config.vector_dimension,
        )
        path = root / f"{cache_key}.f32le"
        vector = (1.0, 0.0) if index % 2 == 0 else (0.0, 1.0)
        path.write_bytes(struct.pack("<2f", *vector))
        bindings.append(
            ArtifactCacheBinding(
                artifact_token=f"artifact-{index}",
                artifact_content_sha256=artifact_hash,
                cache_key=cache_key,
            )
        )
        entries.append(
            EmbeddingCacheEntry(
                cache_key=cache_key,
                relative_path=path.name,
                content_sha256=sha256_file(path),
                byte_size=path.stat().st_size,
            )
        )
    return EmbeddingCacheManifest(
        scoring_inventory_sha256=INVENTORY_SHA256,
        model_sha256=config.model_sha256,
        inference_config_sha256=config.config_sha256,
        dependency_lock_sha256=config.dependency_lock_sha256,
        code_revision=config.code_revision,
        precision=config.backend.precision,
        vector_dimension=config.vector_dimension,
        normalization_tolerance=config.normalization_tolerance,
        bindings=tuple(bindings),
        entries=tuple(entries),
    )

@unittest.skipUnless(
    OPTIONAL_CUDA_ONNX_AVAILABLE,
    "requires ONNX Runtime CUDA",
)
class PairedMeasurementComparisonTests(unittest.TestCase):
    def test_matched_measurement_and_numerical_pass_still_cannot_promote(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cpu_root = root / "cpu"
            cuda_root = root / "cuda"
            cpu_root.mkdir()
            cuda_root.mkdir()
            cpu_paths = write_inputs(cpu_root, cuda=False)
            cuda_paths = write_inputs(cuda_root, cuda=True)
            cpu = run_isolated_lane_benchmark(
                root,
                cpu_paths,
                backend=OnnxBenchmarkBackend.CPU,
            )
            cuda = run_isolated_lane_benchmark(
                root,
                cuda_paths,
                backend=OnnxBenchmarkBackend.CUDA,
            )
            reference_config = producer_config(cpu)
            candidate_config = producer_config(cuda)
            reference_root = root / "reference-cache"
            candidate_root = root / "candidate-cache"
            reference_root.mkdir()
            candidate_root.mkdir()
            reference_manifest = cache_manifest(
                reference_root,
                reference_config,
                cpu.artifact_content_sha256,
            )
            candidate_manifest = cache_manifest(
                candidate_root,
                candidate_config,
                cuda.artifact_content_sha256,
            )
            numerical = compare_embedding_caches(
                reference_manifest=reference_manifest,
                candidate_manifest=candidate_manifest,
                reference_config=reference_config,
                candidate_config=candidate_config,
                reference_root=reference_root,
                candidate_root=candidate_root,
                policy=NumericalDriftPolicy(
                    absolute_tolerance=0.0,
                    relative_tolerance=0.0,
                    relative_floor=1e-12,
                    maximum_l2_drift=0.0,
                    maximum_cosine_drift=0.0,
                    maximum_vectors=4,
                    maximum_vector_dimension=4,
                    maximum_total_bytes_read=1024,
                ),
            )
            receipt = compare_paired_inference_measurements(
                reference=cpu,
                candidate=cuda,
                reference_config=reference_config,
                candidate_config=candidate_config,
                reference_manifest=reference_manifest,
                candidate_manifest=candidate_manifest,
                numerical_admission=numerical,
            )
            self.assertIs(
                receipt.decision,
                MeasurementAdmissionDecision.COMPARABLE_NOT_PROMOTED,
            )
            self.assertIs(
                receipt.promotion_decision,
                PromotionDecision.INCONCLUSIVE,
            )
            self.assertEqual(len(receipt.point_comparisons), 10)
            self.assertIn(
                "NO_BIOMETRIC_SAFETY_INTERVALS",
                receipt.excluded_metric_scopes,
            )
            self.assertEqual(len(receipt.receipt_sha256), 64)
            self.assertEqual(
                PairedInferenceMeasurementReceipt.from_dict(
                    receipt.to_dict()
                ),
                receipt,
            )

            payloads = {
                "reference-benchmark.json": {
                    "schema_version": (
                        "cvi.onnx_inference_benchmark_receipt.v3"
                    ),
                    "summary_sha256": cpu.summary_sha256,
                    "summary": cpu.to_dict(),
                },
                "candidate-benchmark.json": {
                    "schema_version": (
                        "cvi.onnx_inference_benchmark_receipt.v3"
                    ),
                    "summary_sha256": cuda.summary_sha256,
                    "summary": cuda.to_dict(),
                },
                "reference-config.json": reference_config.to_dict(),
                "candidate-config.json": candidate_config.to_dict(),
                "reference-manifest.json": reference_manifest.to_dict(),
                "candidate-manifest.json": candidate_manifest.to_dict(),
                "numerical.json": {
                    "schema_version": "cvi.numerical_admission_bundle.v1",
                    "receipt_sha256": numerical.receipt_sha256,
                    "receipt": numerical.to_dict(),
                },
            }
            for name, payload in payloads.items():
                (root / name).write_text(json.dumps(payload) + "\n")
            cli_receipt = root / "paired-receipt.json"
            subprocess.run(
                (
                    sys.executable,
                    "operations/measurement/compare_onnx_measurements.py",
                    "--reference-benchmark",
                    str(root / "reference-benchmark.json"),
                    "--candidate-benchmark",
                    str(root / "candidate-benchmark.json"),
                    "--reference-producer-config",
                    str(root / "reference-config.json"),
                    "--candidate-producer-config",
                    str(root / "candidate-config.json"),
                    "--reference-cache-manifest",
                    str(root / "reference-manifest.json"),
                    "--candidate-cache-manifest",
                    str(root / "candidate-manifest.json"),
                    "--numerical-admission",
                    str(root / "numerical.json"),
                    "--receipt",
                    str(cli_receipt),
                ),
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            cli_payload = json.loads(cli_receipt.read_text())
            self.assertEqual(
                cli_payload["receipt"]["promotion_decision"],
                "INCONCLUSIVE",
            )
            self.assertEqual(os.stat(cli_receipt).st_mode & 0o777, 0o600)

            failed_numerical = replace(
                numerical,
                hard_failures=("SYNTHETIC_DRIFT",),
                decision=NumericalAdmissionDecision.FAIL,
            )
            with self.assertRaisesRegex(ValueError, "numerical admission PASS"):
                compare_paired_inference_measurements(
                    reference=cpu,
                    candidate=cuda,
                    reference_config=reference_config,
                    candidate_config=candidate_config,
                    reference_manifest=reference_manifest,
                    candidate_manifest=candidate_manifest,
                    numerical_admission=failed_numerical,
                )
            with self.assertRaisesRegex(
                ValueError,
                "numerical producer config binding",
            ):
                compare_paired_inference_measurements(
                    reference=cpu,
                    candidate=cuda,
                    reference_config=replace(
                        reference_config,
                        model_sha256="7" * 64,
                    ),
                    candidate_config=candidate_config,
                    reference_manifest=reference_manifest,
                    candidate_manifest=candidate_manifest,
                    numerical_admission=numerical,
                )

if __name__ == "__main__":
    unittest.main()
