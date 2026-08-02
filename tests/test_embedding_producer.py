from __future__ import annotations

import struct
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from data_pipeline.acquisition import sha256_file
from evaluation.control_scoring import (
    ArtifactSourceKind,
    ControlScoringInventory,
    EmbeddingCachePolicy,
    ScoringArtifactEntry,
)
from operations.embedding_producer import (
    EmbeddingBackendIdentity,
    EmbeddingProducerConfig,
    EmbeddingProductionPolicy,
    EmbeddingProductionReceipt,
    EmbeddingRuntimeResources,
    produce_embedding_cache,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


class FixtureBackend:
    def __init__(
        self,
        identity: EmbeddingBackendIdentity,
        model_sha256: str,
        *,
        mutate_path: Path | None = None,
        malformed: str | None = None,
    ) -> None:
        self._identity = identity
        self._model_sha256 = model_sha256
        self.mutate_path = mutate_path
        self.malformed = malformed
        self.calls: list[tuple[Path, ...]] = []

    @property
    def identity(self) -> EmbeddingBackendIdentity:
        return self._identity

    @property
    def preprocessing_semantics_sha256(self) -> str:
        return HASH_B

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def infer_batch(
        self,
        artifact_paths: tuple[Path, ...],
    ) -> tuple[tuple[float, ...], ...]:
        self.calls.append(artifact_paths)
        if self.mutate_path is not None:
            self.mutate_path.write_bytes(b"mutated!!")
            self.mutate_path = None
        if self.malformed == "zero":
            return tuple((0.0, 0.0) for _ in artifact_paths)
        if self.malformed == "dimension":
            return tuple((1.0,) for _ in artifact_paths)
        return tuple(
            (1e300, 1e-300)
            if path.read_bytes() == b"same"
            else (3.0, 4.0)
            for path in artifact_paths
        )

    def synchronize(self) -> None:
        return None

    def runtime_resources(self) -> EmbeddingRuntimeResources:
        return EmbeddingRuntimeResources.unavailable()


def _identity() -> EmbeddingBackendIdentity:
    return EmbeddingBackendIdentity(
        backend_name="fixture",
        backend_version="1",
        runtime_version="python-test",
        execution_provider="cpu-reference",
        device="cpu",
        precision="fp32",
        determinism_mode="deterministic_fixture",
        backend_config_sha256=HASH_A,
    )


def _create_inputs(root: Path) -> tuple[
    ControlScoringInventory,
    dict[str, Path],
]:
    paths = {
        "token-a": root / "token-a.png",
        "token-b": root / "token-b.png",
        "token-c": root / "token-c.png",
    }
    paths["token-a"].write_bytes(b"same")
    paths["token-b"].write_bytes(b"same")
    paths["token-c"].write_bytes(b"different")
    entries = tuple(
        ScoringArtifactEntry(
            artifact_token=token,
            content_sha256=sha256_file(path),
            byte_size=path.stat().st_size,
            source_kind=ArtifactSourceKind.BASE,
        )
        for token, path in paths.items()
    )
    return (
        ControlScoringInventory(
            plan_sha256=HASH_A,
            scoring_requests_sha256=HASH_B,
            base_artifact_manifest_sha256=HASH_C,
            base_artifact_verification_sha256=HASH_D,
            control_transform_receipt_sha256=HASH_E,
            entries=entries,
        ),
        paths,
    )


def _provenance(root: Path) -> dict[str, Path]:
    paths = {
        "model": root / "model.bin",
        "lineage": root / "lineage.json",
        "preprocess": root / "preprocess.json",
        "lock": root / "uv.lock",
    }
    for name, path in paths.items():
        path.write_bytes(f"{name}-bytes".encode())
    return paths


def _config(paths: dict[str, Path]) -> EmbeddingProducerConfig:
    return EmbeddingProducerConfig(
        model_sha256=sha256_file(paths["model"]),
        model_lineage_sha256=sha256_file(paths["lineage"]),
        preprocessing_sha256=sha256_file(paths["preprocess"]),
        preprocessing_semantics_sha256=HASH_B,
        dependency_lock_sha256=sha256_file(paths["lock"]),
        code_revision="fixture-revision",
        backend=_identity(),
        vector_dimension=2,
        batch_size=1,
        input_width=2,
        input_height=2,
        input_channels=3,
        input_value_bytes=4,
        l2_epsilon=1e-12,
        normalization_tolerance=1e-6,
        warmup_batches=1,
    )


def _produce(
    root: Path,
    *,
    backend: FixtureBackend | None = None,
    config: EmbeddingProducerConfig | None = None,
    policy: EmbeddingProductionPolicy | None = None,
    runtime_phase_callback=None,
):
    inventory, artifact_paths = _create_inputs(root)
    provenance = _provenance(root)
    frozen_config = config or _config(provenance)
    output = root / "cache"
    output.mkdir()
    runtime = backend or FixtureBackend(
        frozen_config.backend,
        frozen_config.model_sha256,
    )
    receipt = produce_embedding_cache(
        inventory=inventory,
        artifact_paths=artifact_paths,
        model_path=provenance["model"],
        model_lineage_path=provenance["lineage"],
        preprocessing_path=provenance["preprocess"],
        dependency_lock_path=provenance["lock"],
        config=frozen_config,
        production_policy=policy or EmbeddingProductionPolicy(),
        cache_policy=EmbeddingCachePolicy(),
        backend=runtime,
        output_directory=output,
        runtime_phase_callback=runtime_phase_callback,
    )
    return receipt, runtime, output


class EmbeddingProducerTests(unittest.TestCase):
    def test_runtime_phase_callback_runs_once_after_first_valid_output(self) -> None:
        with TemporaryDirectory() as temporary:
            phases: list[str] = []
            receipt, _, _ = _produce(
                Path(temporary),
                runtime_phase_callback=phases.append,
            )
            self.assertEqual(phases, ["FIRST_OUTPUT_READY"])
            self.assertEqual(receipt.cost.warmup_batches, 1)

    def test_exact_content_is_deduplicated_and_cache_is_verified(self) -> None:
        with TemporaryDirectory() as temporary:
            receipt, backend, output = _produce(Path(temporary))
            self.assertEqual(receipt.cost.artifact_bindings, 3)
            self.assertEqual(receipt.cost.unique_content_inputs, 2)
            self.assertEqual(
                receipt.cost.content_deduplication_calls_saved,
                1,
            )
            self.assertEqual(receipt.cost.production_batches, 2)
            self.assertEqual(receipt.cost.warmup_artifact_evaluations, 1)
            self.assertEqual(
                receipt.cost.total_backend_artifact_evaluations,
                3,
            )
            self.assertEqual(len(backend.calls), 3)
            self.assertEqual(len(receipt.cache_manifest.entries), 2)
            bindings = {
                binding.artifact_token: binding.cache_key
                for binding in receipt.cache_manifest.bindings
            }
            self.assertEqual(bindings["token-a"], bindings["token-b"])
            self.assertNotEqual(bindings["token-a"], bindings["token-c"])
            self.assertEqual(
                set(path.name for path in output.iterdir()),
                {
                    entry.relative_path
                    for entry in receipt.cache_manifest.entries
                },
            )
            first = output / receipt.cache_manifest.entries[0].relative_path
            values = struct.unpack("<2f", first.read_bytes())
            self.assertAlmostEqual(
                sum(value * value for value in values),
                1.0,
                places=6,
            )
            self.assertEqual(
                receipt.runtime_resources.measurement_scope,
                "UNAVAILABLE",
            )
            self.assertIn(
                "NOT_PROMOTION_EVIDENCE",
                receipt.timing_interpretation,
            )
            self.assertEqual(
                EmbeddingProductionReceipt.from_dict(receipt.to_dict()),
                receipt,
            )
            tampered = receipt.to_dict()
            tampered["cost"]["content_deduplication_calls_saved"] = 2
            with self.assertRaisesRegex(ValueError, "deduplication"):
                EmbeddingProductionReceipt.from_dict(tampered)

    def test_backend_identity_and_provenance_mismatch_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory, artifact_paths = _create_inputs(root)
            provenance = _provenance(root)
            config = _config(provenance)
            output = root / "cache"
            output.mkdir()
            wrong_identity = replace(_identity(), precision="fp16")
            with self.assertRaisesRegex(ValueError, "backend identity"):
                produce_embedding_cache(
                    inventory=inventory,
                    artifact_paths=artifact_paths,
                    model_path=provenance["model"],
                    model_lineage_path=provenance["lineage"],
                    preprocessing_path=provenance["preprocess"],
                    dependency_lock_path=provenance["lock"],
                    config=config,
                    production_policy=EmbeddingProductionPolicy(),
                    cache_policy=EmbeddingCachePolicy(),
                    backend=FixtureBackend(
                        wrong_identity,
                        config.model_sha256,
                    ),
                    output_directory=output,
                )
            self.assertFalse(any(output.iterdir()))
            with self.assertRaisesRegex(ValueError, "model content hash"):
                produce_embedding_cache(
                    inventory=inventory,
                    artifact_paths=artifact_paths,
                    model_path=provenance["model"],
                    model_lineage_path=provenance["lineage"],
                    preprocessing_path=provenance["preprocess"],
                    dependency_lock_path=provenance["lock"],
                    config=replace(config, model_sha256=HASH_A),
                    production_policy=EmbeddingProductionPolicy(),
                    cache_policy=EmbeddingCachePolicy(),
                    backend=FixtureBackend(
                        config.backend,
                        HASH_A,
                    ),
                    output_directory=output,
                )
            self.assertFalse(any(output.iterdir()))
            with self.assertRaisesRegex(ValueError, "runtime loaded model"):
                produce_embedding_cache(
                    inventory=inventory,
                    artifact_paths=artifact_paths,
                    model_path=provenance["model"],
                    model_lineage_path=provenance["lineage"],
                    preprocessing_path=provenance["preprocess"],
                    dependency_lock_path=provenance["lock"],
                    config=config,
                    production_policy=EmbeddingProductionPolicy(),
                    cache_policy=EmbeddingCachePolicy(),
                    backend=FixtureBackend(config.backend, HASH_A),
                    output_directory=output,
                )
            self.assertFalse(any(output.iterdir()))

    def test_malformed_vector_and_input_mutation_publish_nothing(self) -> None:
        for malformed in ("zero", "dimension"):
            with self.subTest(malformed=malformed):
                with TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    inventory, paths = _create_inputs(root)
                    provenance = _provenance(root)
                    config = _config(provenance)
                    output = root / "cache"
                    output.mkdir()
                    with self.assertRaises(ValueError):
                        produce_embedding_cache(
                            inventory=inventory,
                            artifact_paths=paths,
                            model_path=provenance["model"],
                            model_lineage_path=provenance["lineage"],
                            preprocessing_path=provenance["preprocess"],
                            dependency_lock_path=provenance["lock"],
                            config=replace(config, warmup_batches=0),
                            production_policy=EmbeddingProductionPolicy(),
                            cache_policy=EmbeddingCachePolicy(),
                            backend=FixtureBackend(
                                config.backend,
                                config.model_sha256,
                                malformed=malformed,
                            ),
                            output_directory=output,
                        )
                    self.assertFalse(any(output.iterdir()))

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory, paths = _create_inputs(root)
            provenance = _provenance(root)
            config = _config(provenance)
            output = root / "cache"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "content hash"):
                produce_embedding_cache(
                    inventory=inventory,
                    artifact_paths=paths,
                    model_path=provenance["model"],
                    model_lineage_path=provenance["lineage"],
                    preprocessing_path=provenance["preprocess"],
                    dependency_lock_path=provenance["lock"],
                    config=replace(config, warmup_batches=0),
                    production_policy=EmbeddingProductionPolicy(),
                    cache_policy=EmbeddingCachePolicy(),
                    backend=FixtureBackend(
                        config.backend,
                        config.model_sha256,
                        mutate_path=paths["token-c"],
                    ),
                    output_directory=output,
                )
            self.assertFalse(any(output.iterdir()))

    def test_resource_preflight_rejects_before_backend_work(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory, artifact_paths = _create_inputs(root)
            provenance = _provenance(root)
            config = _config(provenance)
            output = root / "cache"
            output.mkdir()
            backend = FixtureBackend(
                config.backend,
                config.model_sha256,
            )
            with self.assertRaisesRegex(ValueError, "unique embedding inputs"):
                produce_embedding_cache(
                    inventory=inventory,
                    artifact_paths=artifact_paths,
                    model_path=provenance["model"],
                    model_lineage_path=provenance["lineage"],
                    preprocessing_path=provenance["preprocess"],
                    dependency_lock_path=provenance["lock"],
                    config=config,
                    production_policy=replace(
                        EmbeddingProductionPolicy(),
                        maximum_unique_inputs=1,
                    ),
                    cache_policy=EmbeddingCachePolicy(),
                    backend=backend,
                    output_directory=output,
                )
            self.assertEqual(backend.calls, [])
            self.assertFalse(any(output.iterdir()))
            with self.assertRaisesRegex(ValueError, "input bytes"):
                produce_embedding_cache(
                    inventory=inventory,
                    artifact_paths=artifact_paths,
                    model_path=provenance["model"],
                    model_lineage_path=provenance["lineage"],
                    preprocessing_path=provenance["preprocess"],
                    dependency_lock_path=provenance["lock"],
                    config=config,
                    production_policy=replace(
                        EmbeddingProductionPolicy(),
                        maximum_total_input_bytes=13,
                    ),
                    cache_policy=EmbeddingCachePolicy(),
                    backend=backend,
                    output_directory=output,
                )
            self.assertEqual(backend.calls, [])
            self.assertFalse(any(output.iterdir()))


if __name__ == "__main__":
    unittest.main()
