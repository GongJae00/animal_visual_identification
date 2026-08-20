from __future__ import annotations

import json
import math
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
from evaluation.integrity.numerical_admission import (
    NumericalAdmissionDecision,
    NumericalAdmissionReceipt,
    NumericalDriftPolicy,
    compare_embedding_caches,
)
from prototype.export.embedding_producer import (
    EmbeddingBackendIdentity,
    EmbeddingProducerConfig,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64

def _config(
    *,
    backend_name: str,
    lock_sha256: str,
    precision: str,
) -> EmbeddingProducerConfig:
    backend = EmbeddingBackendIdentity(
        backend_name=backend_name,
        backend_version="1",
        runtime_version=f"{backend_name}-runtime",
        execution_provider=backend_name,
        device=backend_name,
        precision=precision,
        determinism_mode="REQUESTED_NOT_PROVEN",
        backend_config_sha256=(
            HASH_E if backend_name == "cpu" else HASH_F
        ),
    )
    return EmbeddingProducerConfig(
        model_sha256=HASH_A,
        model_lineage_sha256=HASH_B,
        preprocessing_sha256=HASH_C,
        preprocessing_semantics_sha256=HASH_D,
        dependency_lock_sha256=lock_sha256,
        code_revision="frozen-revision",
        backend=backend,
        vector_dimension=2,
        batch_size=2,
        input_width=2,
        input_height=2,
        input_channels=3,
        input_value_bytes=4,
        l2_epsilon=1e-12,
        normalization_tolerance=1e-6,
    )

def _cache(
    root: Path,
    config: EmbeddingProducerConfig,
    vectors: dict[str, tuple[float, float]],
) -> EmbeddingCacheManifest:
    bindings: list[ArtifactCacheBinding] = []
    entries: list[EmbeddingCacheEntry] = []
    for index, (artifact_hash, vector) in enumerate(sorted(vectors.items())):
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
        path.write_bytes(struct.pack("<2f", *vector))
        bindings.append(
            ArtifactCacheBinding(
                artifact_token=f"token-{index}",
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
        scoring_inventory_sha256=HASH_E,
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

def _policy(**overrides: object) -> NumericalDriftPolicy:
    values: dict[str, object] = {
        "absolute_tolerance": 1e-4,
        "relative_tolerance": 1e-4,
        "relative_floor": 1e-12,
        "maximum_l2_drift": 1e-4,
        "maximum_cosine_drift": 1e-8,
        "maximum_vectors": 10,
        "maximum_vector_dimension": 10,
        "maximum_total_bytes_read": 1_000,
    }
    values.update(overrides)
    return NumericalDriftPolicy(**values)

class NumericalAdmissionTests(unittest.TestCase):
    def test_identical_and_small_drift_caches_pass_with_receipt_roundtrip(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_root = root / "reference"
            candidate_root = root / "candidate"
            reference_root.mkdir()
            candidate_root.mkdir()
            reference_config = _config(
                backend_name="cpu",
                lock_sha256=HASH_E,
                precision="fp32",
            )
            candidate_config = _config(
                backend_name="cuda",
                lock_sha256=HASH_F,
                precision="fp32_tf32_disabled",
            )
            reference = _cache(
                reference_root,
                reference_config,
                {HASH_A: (1.0, 0.0), HASH_B: (0.0, 1.0)},
            )
            epsilon = 1e-6
            candidate = _cache(
                candidate_root,
                candidate_config,
                {
                    HASH_A: (math.sqrt(1.0 - epsilon**2), epsilon),
                    HASH_B: (0.0, 1.0),
                },
            )
            receipt = compare_embedding_caches(
                reference_manifest=reference,
                candidate_manifest=candidate,
                reference_config=reference_config,
                candidate_config=candidate_config,
                reference_root=reference_root,
                candidate_root=candidate_root,
                policy=_policy(),
            )
            self.assertIs(
                receipt.decision,
                NumericalAdmissionDecision.PASS,
            )
            self.assertEqual(receipt.summary.vectors, 2)
            self.assertEqual(receipt.summary.values, 4)
            self.assertEqual(receipt.summary.violated_values, 0)
            self.assertEqual(receipt.summary.bytes_read, 32)
            self.assertGreater(receipt.summary.maximum_ulp_distance, 0)
            self.assertEqual(
                NumericalAdmissionReceipt.from_dict(receipt.to_dict()),
                receipt,
            )

    def test_drift_failure_reports_element_l2_and_cosine_gates(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_root = root / "reference"
            candidate_root = root / "candidate"
            reference_root.mkdir()
            candidate_root.mkdir()
            reference_config = _config(
                backend_name="cpu",
                lock_sha256=HASH_E,
                precision="fp32",
            )
            candidate_config = _config(
                backend_name="cuda",
                lock_sha256=HASH_F,
                precision="fp16",
            )
            reference = _cache(
                reference_root,
                reference_config,
                {HASH_A: (1.0, 0.0)},
            )
            candidate = _cache(
                candidate_root,
                candidate_config,
                {HASH_A: (0.0, 1.0)},
            )
            receipt = compare_embedding_caches(
                reference_manifest=reference,
                candidate_manifest=candidate,
                reference_config=reference_config,
                candidate_config=candidate_config,
                reference_root=reference_root,
                candidate_root=candidate_root,
                policy=_policy(),
            )
            self.assertIs(
                receipt.decision,
                NumericalAdmissionDecision.FAIL,
            )
            self.assertEqual(
                receipt.hard_failures,
                (
                    "ELEMENTWISE_ABSOLUTE_RELATIVE_TOLERANCE",
                    "EMBEDDING_COSINE_DRIFT",
                    "EMBEDDING_L2_DRIFT",
                ),
            )
            self.assertEqual(receipt.summary.violated_vectors, 1)
            self.assertEqual(receipt.summary.violated_values, 2)

    def test_lineage_binding_and_closed_cache_fail_before_comparison(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_root = root / "reference"
            candidate_root = root / "candidate"
            reference_root.mkdir()
            candidate_root.mkdir()
            reference_config = _config(
                backend_name="cpu",
                lock_sha256=HASH_E,
                precision="fp32",
            )
            candidate_config = _config(
                backend_name="cuda",
                lock_sha256=HASH_F,
                precision="fp32",
            )
            reference = _cache(
                reference_root,
                reference_config,
                {HASH_A: (1.0, 0.0)},
            )
            candidate = _cache(
                candidate_root,
                candidate_config,
                {HASH_A: (1.0, 0.0)},
            )
            with self.assertRaisesRegex(ValueError, "config binding differs"):
                compare_embedding_caches(
                    reference_manifest=reference,
                    candidate_manifest=candidate,
                    reference_config=reference_config,
                    candidate_config=replace(
                        candidate_config,
                        preprocessing_semantics_sha256=HASH_A,
                    ),
                    reference_root=reference_root,
                    candidate_root=candidate_root,
                    policy=_policy(),
                )
            altered_root = root / "altered"
            altered_root.mkdir()
            altered_config = replace(
                candidate_config,
                preprocessing_semantics_sha256=HASH_A,
            )
            altered = _cache(
                altered_root,
                altered_config,
                {HASH_A: (1.0, 0.0)},
            )
            with self.assertRaisesRegex(ValueError, "semantics differ"):
                compare_embedding_caches(
                    reference_manifest=reference,
                    candidate_manifest=altered,
                    reference_config=reference_config,
                    candidate_config=altered_config,
                    reference_root=reference_root,
                    candidate_root=altered_root,
                    policy=_policy(),
                )
            (candidate_root / "extra.f32le").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "not closed"):
                compare_embedding_caches(
                    reference_manifest=reference,
                    candidate_manifest=candidate,
                    reference_config=reference_config,
                    candidate_config=candidate_config,
                    reference_root=reference_root,
                    candidate_root=candidate_root,
                    policy=_policy(),
                )

    def test_resource_cap_and_hash_tamper_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_root = root / "reference"
            candidate_root = root / "candidate"
            reference_root.mkdir()
            candidate_root.mkdir()
            reference_config = _config(
                backend_name="cpu",
                lock_sha256=HASH_E,
                precision="fp32",
            )
            candidate_config = _config(
                backend_name="cuda",
                lock_sha256=HASH_F,
                precision="fp32",
            )
            reference = _cache(
                reference_root,
                reference_config,
                {HASH_A: (1.0, 0.0)},
            )
            candidate = _cache(
                candidate_root,
                candidate_config,
                {HASH_A: (1.0, 0.0)},
            )
            with self.assertRaisesRegex(ValueError, "bytes exceed"):
                compare_embedding_caches(
                    reference_manifest=reference,
                    candidate_manifest=candidate,
                    reference_config=reference_config,
                    candidate_config=candidate_config,
                    reference_root=reference_root,
                    candidate_root=candidate_root,
                    policy=_policy(maximum_total_bytes_read=15),
                )
            candidate_path = next(candidate_root.iterdir())
            candidate_path.write_bytes(struct.pack("<2f", 0.0, 1.0))
            with self.assertRaisesRegex(ValueError, "hash differs"):
                compare_embedding_caches(
                    reference_manifest=reference,
                    candidate_manifest=candidate,
                    reference_config=reference_config,
                    candidate_config=candidate_config,
                    reference_root=reference_root,
                    candidate_root=candidate_root,
                    policy=_policy(),
                )

    def test_protected_cli_writes_hashed_private_no_overwrite_bundle(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_root = root / "reference"
            candidate_root = root / "candidate"
            reference_root.mkdir()
            candidate_root.mkdir()
            reference_config = _config(
                backend_name="cpu",
                lock_sha256=HASH_E,
                precision="fp32",
            )
            candidate_config = _config(
                backend_name="cuda",
                lock_sha256=HASH_F,
                precision="fp32",
            )
            reference = _cache(
                reference_root,
                reference_config,
                {HASH_A: (1.0, 0.0)},
            )
            candidate = _cache(
                candidate_root,
                candidate_config,
                {HASH_A: (1.0, 0.0)},
            )
            payloads = {
                "reference-manifest": reference.to_dict(),
                "candidate-manifest": candidate.to_dict(),
                "reference-config": reference_config.to_dict(),
                "candidate-config": candidate_config.to_dict(),
                "policy": _policy().to_dict(),
            }
            paths: dict[str, Path] = {}
            for name, payload in payloads.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[name] = path
            output = root / "numerical.json"
            command = (
                sys.executable,
                "operations/workers/compare_embedding_caches.py",
                "--reference-cache-directory",
                str(reference_root),
                "--candidate-cache-directory",
                str(candidate_root),
                "--reference-cache-manifest",
                str(paths["reference-manifest"]),
                "--candidate-cache-manifest",
                str(paths["candidate-manifest"]),
                "--reference-producer-config",
                str(paths["reference-config"]),
                "--candidate-producer-config",
                str(paths["candidate-config"]),
                "--policy",
                str(paths["policy"]),
                "--receipt",
                str(output),
            )
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            status = json.loads(completed.stdout)
            bundle = json.loads(output.read_text(encoding="utf-8"))
            parsed = NumericalAdmissionReceipt.from_dict(bundle["receipt"])
            self.assertEqual(
                status["decision"],
                NumericalAdmissionDecision.PASS.value,
            )
            self.assertEqual(bundle["receipt_sha256"], parsed.receipt_sha256)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )

if __name__ == "__main__":
    unittest.main()
