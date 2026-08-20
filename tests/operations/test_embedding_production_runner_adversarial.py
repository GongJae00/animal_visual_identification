from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import operations.workers.embedding_production_runner as runner
from shared.contracts.runtime_library_provenance import RuntimeLibraryPolicy
from operations.workers.embedding_production_runner import (
    EmbeddingProductionPrecommitment,
    EmbeddingWorkerExecutionPolicy,
    read_embedding_production_outer_bundle,
    run_embedding_production_fresh_worker,
)
from operations.workers.process_supervisor import ProcessSupervisorPolicy
from operations.workers.worker_environment import build_sanitized_worker_environment

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64

def _supervisor_policy() -> ProcessSupervisorPolicy:
    return ProcessSupervisorPolicy(
        timeout_seconds=10.0,
        termination_grace_seconds=1.0,
        poll_interval_seconds=0.01,
        maximum_stdout_bytes=0,
        maximum_stderr_bytes=0,
    )

def _execution_policy(
    *,
    maximum_worker_result_bytes: int = 8_388_608,
    maximum_snapshot_bytes: int = 8_388_608,
    maximum_code_snapshot_bytes: int = 8_388_608,
) -> EmbeddingWorkerExecutionPolicy:
    return EmbeddingWorkerExecutionPolicy(
        supervisor=_supervisor_policy(),
        maximum_worker_result_bytes=maximum_worker_result_bytes,
        maximum_snapshot_bytes=maximum_snapshot_bytes,
        maximum_code_snapshot_bytes=maximum_code_snapshot_bytes,
    )

def _precommitment(
    *,
    execution_policy_sha256: str,
    environment_sha256: str,
    artifact_bytes: int = 2,
    runtime_policy_sha256: str = HASH_A,
) -> EmbeddingProductionPrecommitment:
    code_sources = runner._code_source_bindings()
    return EmbeddingProductionPrecommitment(
        scoring_inventory_sha256=HASH_A,
        producer_config_sha256=HASH_A,
        production_policy_sha256=HASH_A,
        cache_policy_sha256=HASH_A,
        backend_identity_sha256=HASH_A,
        runtime_library_policy_sha256=runtime_policy_sha256,
        worker_execution_policy_sha256=execution_policy_sha256,
        worker_environment_identity_sha256=environment_sha256,
        artifact_bindings=(("artifact-0", HASH_B, artifact_bytes),),
        provenance_sha256=tuple(
            (name, HASH_C, 1) for name in runner._PROVENANCE_NAMES
        ),
        code_source_sha256=code_sources,
        code_source_manifest_sha256=runner.content_sha256(
            [list(item) for item in code_sources]
        ),
        code_source_files=len(code_sources),
        code_source_bytes=sum(item[2] for item in code_sources),
        worker_bootstrap_sha256=runner.EMBEDDING_WORKER_BOOTSTRAP_SHA256,
        input_bytes_hashed=artifact_bytes,
        provenance_bytes_hashed=len(runner._PROVENANCE_NAMES),
        prior_attempt_ledger_sha256=HASH_A,
        candidate_attempt_token=HASH_B,
        precommitment_sequence=1,
    )

def _input_files(root: Path, runtime_policy: RuntimeLibraryPolicy) -> dict[str, Path]:
    names = {
        "inventory", "artifact_paths", "producer_config", "onnx_config",
        "preprocessing", "model", "model_lineage", "dependency_lock",
        "production_policy", "cache_policy", "precommitment",
        "runtime_library_policy",
    }
    result = {name: root / f"{name}.json" for name in names}
    for name, path in result.items():
        payload = (
            json.dumps(runtime_policy.to_dict(), sort_keys=True)
            if name == "runtime_library_policy"
            else f"fixture-{name}"
        )
        path.write_text(payload, encoding="utf-8")
    return result

class EmbeddingProductionRunnerAdversarialTests(unittest.TestCase):
    def test_historical_bootstrap_parses_but_cannot_execute(self) -> None:
        policy = _execution_policy()
        for bootstrap_sha256 in (
            runner.LEGACY_EMBEDDING_WORKER_BOOTSTRAP_SHA256,
            runner.MIGRATED_EMBEDDING_WORKER_BOOTSTRAP_SHA256,
            runner.SYSTEMS_EMBEDDING_WORKER_BOOTSTRAP_SHA256,
        ):
            with self.subTest(bootstrap_sha256=bootstrap_sha256):
                historical = replace(
                    _precommitment(
                        execution_policy_sha256=policy.policy_sha256,
                        environment_sha256=HASH_A,
                    ),
                    worker_bootstrap_sha256=bootstrap_sha256,
                )

                parsed = EmbeddingProductionPrecommitment.from_dict(
                    historical.to_dict()
                )
                self.assertEqual(parsed, historical)
                with (
                    patch.object(runner, "run_supervised_process") as launch,
                    self.assertRaisesRegex(
                        RuntimeError, "historical embedding bootstrap"
                    ),
                ):
                    run_embedding_production_fresh_worker(
                        backend="cpu",
                        files={},
                        precommitment=parsed,
                        expected_precommitment_sha256=parsed.precommitment_sha256,
                        python_executable=Path(sys.executable),
                        execution_policy=policy,
                        output_directory=Path("unreachable"),
                        discovery=True,
                    )
                launch.assert_not_called()

    def test_historical_source_inventory_parses_but_cannot_execute(self) -> None:
        policy = _execution_policy()
        historical = replace(
            _precommitment(
                execution_policy_sha256=policy.policy_sha256,
                environment_sha256=HASH_A,
            ),
            code_source_sha256=(("data_pipeline/acquisition.py", HASH_A, 1),),
            code_source_manifest_sha256=runner.content_sha256(
                [["data_pipeline/acquisition.py", HASH_A, 1]]
            ),
            code_source_files=1,
            code_source_bytes=1,
        )

        parsed = EmbeddingProductionPrecommitment.from_dict(historical.to_dict())
        self.assertEqual(parsed, historical)
        with (
            patch.object(runner, "run_supervised_process") as launch,
            self.assertRaisesRegex(RuntimeError, "protected Python sources changed"),
        ):
            run_embedding_production_fresh_worker(
                backend="cpu",
                files={},
                precommitment=parsed,
                expected_precommitment_sha256=parsed.precommitment_sha256,
                python_executable=Path(sys.executable),
                execution_policy=policy,
                output_directory=Path("unreachable"),
                discovery=True,
            )
        launch.assert_not_called()

    def test_historical_source_inventory_rejects_unsafe_paths(self) -> None:
        precommitment = _precommitment(
            execution_policy_sha256=HASH_A,
            environment_sha256=HASH_A,
        )
        for name in (
            "../source.py",
            "/absolute/source.py",
            "package\\source.py",
            "source.txt",
        ):
            payload = precommitment.to_dict()
            payload["code_source_sha256"] = [[name, HASH_A, 1]]
            payload["code_source_manifest_sha256"] = runner.content_sha256(
                payload["code_source_sha256"]
            )
            payload["code_source_files"] = 1
            payload["code_source_bytes"] = 1
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                "code-source manifest",
            ):
                EmbeddingProductionPrecommitment.from_dict(payload)

    def test_code_snapshot_is_complete_and_independent_of_workspace_changes(
        self,
    ) -> None:
        self.assertIn("parsing", runner._CODE_SOURCE_PACKAGE_NAMES)
        self.assertTrue(
            any(name.startswith("parsing/") for name in runner._CODE_SOURCE_NAMES)
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "snapshot"
            source.mkdir()
            destination.mkdir()
            sources = {
                "fixture/__init__.py": b"BOUND = 'initial'\n",
                "fixture/worker.py": b"VALUE = 1\n",
            }
            for name, payload in sources.items():
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            expected = tuple(
                (name, hashlib.sha256(payload).hexdigest(), len(payload))
                for name, payload in sorted(sources.items())
            )
            with patch.object(
                runner,
                "_CODE_SOURCE_DIRECTORY",
                source,
            ), patch.object(
                runner,
                "_CODE_SOURCE_NAMES",
                tuple(sorted(sources)),
            ):
                files, byte_size = runner._snapshot_code_sources(
                    expected,
                    destination,
                    maximum_bytes=1_024,
                )
                self.assertEqual(files, 2)
                self.assertEqual(byte_size, sum(map(len, sources.values())))
                (source / "fixture/worker.py").write_bytes(b"VALUE = 999\n")
                runner._verify_code_source_snapshot(destination, expected)
                self.assertEqual(
                    (destination / "fixture/worker.py").read_bytes(),
                    sources["fixture/worker.py"],
                )
                (source / "fixture/omitted.py").write_text("SURPRISE = True\n")
                with self.assertRaisesRegex(RuntimeError, "inventory"):
                    runner._code_source_bindings_at(source)

    def test_execution_policy_and_environment_mismatch_never_launch_worker(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_policy = RuntimeLibraryPolicy(
                expected_binaries=(),
                allow_discovery_only=True,
            )
            files = _input_files(root, runtime_policy)
            policy = _execution_policy()
            _, identity = build_sanitized_worker_environment(
                os.environ,
                python_executable=Path(sys.executable),
            )
            cases = (
                (
                    "execution policy",
                    _precommitment(
                        execution_policy_sha256=HASH_C,
                        environment_sha256=identity.identity_sha256,
                    ),
                ),
                (
                    "worker environment",
                    _precommitment(
                        execution_policy_sha256=policy.policy_sha256,
                        environment_sha256=HASH_C,
                    ),
                ),
            )
            for expected_message, precommitment in cases:
                with self.subTest(expected_message=expected_message), patch.object(
                    runner,
                    "run_supervised_process",
                ) as launch:
                    with self.assertRaisesRegex(ValueError, expected_message):
                        run_embedding_production_fresh_worker(
                            backend="cpu",
                            files=files,
                            precommitment=precommitment,
                            expected_precommitment_sha256=(
                                precommitment.precommitment_sha256
                            ),
                            python_executable=Path(sys.executable),
                            execution_policy=policy,
                            output_directory=root / "unpublished-cache",
                            discovery=True,
                        )
                    launch.assert_not_called()

    def test_snapshot_and_result_caps_reject_before_supervisor_launch(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_policy = RuntimeLibraryPolicy(
                expected_binaries=(),
                maximum_executable_identities=1,
                allow_discovery_only=True,
            )
            files = _input_files(root, runtime_policy)
            _, identity = build_sanitized_worker_environment(
                os.environ,
                python_executable=Path(sys.executable),
            )

            snapshot_policy = _execution_policy(maximum_snapshot_bytes=1)
            snapshot_precommitment = _precommitment(
                execution_policy_sha256=snapshot_policy.policy_sha256,
                environment_sha256=identity.identity_sha256,
            )
            result_policy = _execution_policy(maximum_worker_result_bytes=1)
            result_precommitment = _precommitment(
                execution_policy_sha256=result_policy.policy_sha256,
                environment_sha256=identity.identity_sha256,
                runtime_policy_sha256=runtime_policy.policy_sha256,
            )
            code_policy = _execution_policy(maximum_code_snapshot_bytes=1)
            code_precommitment = _precommitment(
                execution_policy_sha256=code_policy.policy_sha256,
                environment_sha256=identity.identity_sha256,
            )
            cases = (
                ("snapshot exceeds", snapshot_policy, snapshot_precommitment),
                ("result estimate exceeds", result_policy, result_precommitment),
                ("code snapshot exceeds", code_policy, code_precommitment),
            )
            for expected_message, policy, precommitment in cases:
                with self.subTest(expected_message=expected_message), patch.object(
                    runner,
                    "run_supervised_process",
                ) as launch:
                    with self.assertRaisesRegex(ValueError, expected_message):
                        run_embedding_production_fresh_worker(
                            backend="cpu",
                            files=files,
                            precommitment=precommitment,
                            expected_precommitment_sha256=(
                                precommitment.precommitment_sha256
                            ),
                            python_executable=Path(sys.executable),
                            execution_policy=policy,
                            output_directory=root / "unpublished-cache",
                            discovery=True,
                        )
                    launch.assert_not_called()

    def test_outer_bundle_is_v2_only_and_requires_both_external_anchors(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_path = root / "receipt.json"
            legacy = {
                "schema_version": "cvi.embedding_production_bundle.v1",
                "receipt_sha256": HASH_A,
                "receipt": {},
            }
            bundle_path.write_text(json.dumps(legacy), encoding="utf-8")
            with patch.object(
                runner.EmbeddingFreshWorkerReceipt,
                "from_dict",
            ) as parser:
                with self.assertRaisesRegex(ValueError, "bundle schema differs"):
                    read_embedding_production_outer_bundle(
                        bundle_path,
                        expected_receipt_sha256=HASH_A,
                        expected_completed_attempt_ledger_head_sha256=HASH_B,
                    )
                parser.assert_not_called()

            receipt_payload = {"synthetic": "payload"}
            receipt_sha256 = runner.content_sha256(receipt_payload)
            current = {
                "schema_version": "cvi.embedding_production_bundle.v2",
                "receipt_sha256": receipt_sha256,
                "receipt": receipt_payload,
            }
            bundle_path.write_text(json.dumps(current), encoding="utf-8")
            parsed = SimpleNamespace(
                receipt_sha256=receipt_sha256,
                completed_attempt_ledger_head_sha256=HASH_B,
            )
            with patch.object(
                runner.EmbeddingFreshWorkerReceipt,
                "from_dict",
                return_value=parsed,
            ):
                with self.assertRaisesRegex(ValueError, "receipt anchor"):
                    read_embedding_production_outer_bundle(
                        bundle_path,
                        expected_receipt_sha256=HASH_C,
                        expected_completed_attempt_ledger_head_sha256=HASH_B,
                    )
                with self.assertRaisesRegex(ValueError, "attempt anchor"):
                    read_embedding_production_outer_bundle(
                        bundle_path,
                        expected_receipt_sha256=receipt_sha256,
                        expected_completed_attempt_ledger_head_sha256=HASH_C,
                    )
                self.assertIs(
                    read_embedding_production_outer_bundle(
                        bundle_path,
                        expected_receipt_sha256=receipt_sha256,
                        expected_completed_attempt_ledger_head_sha256=HASH_B,
                    ),
                    parsed,
                )

if __name__ == "__main__":
    unittest.main()
