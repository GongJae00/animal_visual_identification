from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from cvi.batch_invariance import (
    BatchInvarianceDecision,
    BatchInvariancePolicy,
    BatchInvariancePrecommitment,
    BatchInvarianceReceipt,
    BatchRuntimeDiscoveryComplete,
    build_batch_invariance_precommitment,
    evaluate_batch_composition_invariance,
    verify_batch_invariance_receipt_external_anchors,
)
from cvi.batch_invariance_runner import (
    BatchFreshWorkerDiscovery,
    BatchFreshWorkerReceipt,
    BatchWorkerExecutionPolicy,
    run_batch_invariance_fresh_worker,
)
from cvi.control_scoring import (
    ArtifactSourceKind,
    ControlScoringInventory,
    ScoringArtifactEntry,
)
from cvi.embedding_producer import (
    EmbeddingBackendIdentity,
    EmbeddingProducerConfig,
    EmbeddingRuntimeResources,
)
from cvi.optimization import PromotionDecision
from cvi.process_supervisor import (
    ProcessSupervisorPolicy,
    SupervisedProcessResult,
    SupervisedProcessStatus,
)
from cvi.provenance import content_sha256
from cvi.runtime_library_provenance import (
    ExpectedRuntimeBinary,
    RuntimeBinaryEntry,
    RuntimeLibraryManifest,
    RuntimeLibraryPhase,
    RuntimeLibraryPolicy,
)
from cvi.worker_environment import build_sanitized_worker_environment


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class FixtureBatchBackend:
    def __init__(
        self,
        identity: EmbeddingBackendIdentity,
        preprocessing_sha256: str,
        model_sha256: str,
        mode: str = "stable",
    ) -> None:
        self._identity = identity
        self._preprocessing_sha256 = preprocessing_sha256
        self._model_sha256 = model_sha256
        self.mode = mode
        self.calls = 0

    @property
    def identity(self) -> EmbeddingBackendIdentity:
        return self._identity

    @property
    def preprocessing_semantics_sha256(self) -> str:
        return self._preprocessing_sha256

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def actual_providers(self) -> tuple[str, ...]:
        return ("FIXTUREExecutionProvider",)

    @property
    def actual_provider_options(self) -> dict[str, dict[str, str]]:
        return {"FIXTUREExecutionProvider": {}}

    def infer_batch(self, paths: tuple[Path, ...]):
        self.calls += 1
        values = [float(path.read_bytes()[0]) for path in paths]
        rows = [[value, value + 1.0, 1.0] for value in values]
        if self.mode == "batch_size":
            for row in rows:
                row[0] += len(paths) * 0.01
        elif self.mode == "slot":
            for index, row in enumerate(rows):
                row[0] += index * 0.01
        elif self.mode == "neighbor":
            adjustment = sum(values) / len(values) * 0.01
            for row in rows:
                row[0] += adjustment
        elif self.mode == "duplicate" and len(set(paths)) != len(paths):
            for row in rows:
                row[0] += 0.1
        elif self.mode == "stateful":
            for row in rows:
                row[0] += self.calls * 0.001
        elif self.mode == "row_swap" and len(rows) > 1:
            rows.reverse()
        return tuple(tuple(row) for row in rows)

    def synchronize(self) -> None:
        return None

    def runtime_resources(self) -> EmbeddingRuntimeResources:
        return EmbeddingRuntimeResources.unavailable()


class FixtureRuntimeTracker:
    def __init__(
        self,
        policy_sha256: str,
        decision: str = "PASS",
    ) -> None:
        self.policy = SimpleNamespace(policy_sha256=policy_sha256)
        self.phases = [RuntimeLibraryPhase.DEPENDENCIES_IMPORTED]
        self.decision = decision

    def capture(self, phase: RuntimeLibraryPhase) -> None:
        expected = tuple(RuntimeLibraryPhase)[len(self.phases)]
        if phase is not expected:
            raise ValueError("fixture runtime phase order differs")
        self.phases.append(phase)

    def finalize(self) -> RuntimeLibraryManifest:
        if self.phases != list(RuntimeLibraryPhase):
            raise ValueError("fixture runtime phases incomplete")
        return RuntimeLibraryManifest(
            policy_sha256=self.policy.policy_sha256,
            entries=(),
            binary_set_sha256=content_sha256([]),
            maps_snapshots=4,
            maps_bytes_read=0,
            binary_bytes_hashed=0,
            provenance_wall_time_ns=0,
            decision=self.decision,
            hard_failures=(),
        )


def policy(**overrides: object) -> BatchInvariancePolicy:
    values: dict[str, object] = {
        "absolute_tolerance": 0.0,
        "relative_tolerance": 0.0,
        "relative_floor": 1e-12,
        "maximum_raw_l2_drift": 0.0,
        "maximum_raw_norm_drift": 0.0,
        "maximum_normalized_l2_drift": 0.0,
        "maximum_cosine_drift": 0.0,
        "maximum_artifacts": 10,
        "maximum_vector_dimension": 10,
        "maximum_backend_calls": 100,
        "maximum_artifact_evaluations": 100,
        "maximum_comparison_values": 1_000,
        "maximum_input_bytes_hashed": 1_000,
        "maximum_provenance_bytes_hashed": 10_000,
        "maximum_anchor_temporary_bytes": 1_000,
    }
    values.update(overrides)
    return BatchInvariancePolicy(**values)


class BatchInvarianceTests(unittest.TestCase):
    def build_case(self, root: Path, mode: str = "stable") -> dict[str, object]:
        artifacts = root / "artifacts"
        artifacts.mkdir()
        artifact_paths: dict[str, Path] = {}
        entries = []
        for index, value in enumerate((11, 29, 47, 83, 101)):
            token = f"artifact-{index}"
            path = artifacts / f"{token}.bin"
            path.write_bytes(bytes((value, index)))
            artifact_paths[token] = path
            entries.append(
                ScoringArtifactEntry(
                    artifact_token=token,
                    content_sha256=digest(path.read_bytes()),
                    byte_size=2,
                    source_kind=ArtifactSourceKind.BASE,
                )
            )
        inventory = ControlScoringInventory(
            plan_sha256="a" * 64,
            scoring_requests_sha256="b" * 64,
            base_artifact_manifest_sha256="c" * 64,
            base_artifact_verification_sha256="d" * 64,
            control_transform_receipt_sha256="e" * 64,
            entries=tuple(entries),
        )
        provenance_paths: dict[str, Path] = {}
        for name in ("model", "model_lineage", "preprocessing", "dependency_lock"):
            path = root / f"{name}.bin"
            path.write_bytes(f"fixture-{name}".encode())
            provenance_paths[name] = path
        identity = EmbeddingBackendIdentity(
            backend_name="fixture.batch",
            backend_version="1",
            runtime_version="fixture-runtime",
            execution_provider="FIXTUREExecutionProvider",
            device="cpu",
            precision="fp32",
            determinism_mode="REQUESTED_NOT_PROVEN",
            backend_config_sha256="f" * 64,
        )
        config = EmbeddingProducerConfig(
            model_sha256=digest(provenance_paths["model"].read_bytes()),
            model_lineage_sha256=digest(
                provenance_paths["model_lineage"].read_bytes()
            ),
            preprocessing_sha256=digest(
                provenance_paths["preprocessing"].read_bytes()
            ),
            preprocessing_semantics_sha256="1" * 64,
            dependency_lock_sha256=digest(
                provenance_paths["dependency_lock"].read_bytes()
            ),
            code_revision="batch-invariance-test",
            backend=identity,
            vector_dimension=3,
            batch_size=4,
            input_width=2,
            input_height=2,
            input_channels=3,
            input_value_bytes=4,
            l2_epsilon=1e-12,
            normalization_tolerance=1e-6,
        )
        backend = FixtureBatchBackend(
            identity,
            config.preprocessing_semantics_sha256,
            config.model_sha256,
            mode,
        )
        return locals()

    def evaluate(self, case: dict[str, object], **overrides: object):
        resolved_policy = overrides.get("policy", policy())
        runtime_policy_sha256 = overrides.get(
            "runtime_policy_sha256", "8" * 64
        )
        worker_execution_policy_sha256 = overrides.get(
            "worker_execution_policy_sha256", "a" * 64
        )
        worker_environment_identity_sha256 = overrides.get(
            "worker_environment_identity_sha256", "b" * 64
        )
        precommitment = build_batch_invariance_precommitment(
            inventory=case["inventory"],
            artifact_paths=case["artifact_paths"],
            producer_config=case["config"],
            provenance_paths=case["provenance_paths"],
            policy=resolved_policy,
            runtime_library_policy_sha256=runtime_policy_sha256,
            worker_execution_policy_sha256=worker_execution_policy_sha256,
            worker_environment_identity_sha256=(
                worker_environment_identity_sha256
            ),
            prior_attempt_ledger_sha256="2" * 64,
            candidate_attempt_token="3" * 64,
            precommitment_sequence=1,
        )
        return evaluate_batch_composition_invariance(
            backend=overrides.get("backend", case["backend"]),
            inventory=case["inventory"],
            artifact_paths=case["artifact_paths"],
            producer_config=case["config"],
            provenance_paths=case["provenance_paths"],
            policy=resolved_policy,
            precommitment=overrides.get("precommitment", precommitment),
            expected_precommitment_sha256=overrides.get(
                "expected_precommitment_sha256",
                precommitment.precommitment_sha256,
            ),
            runtime_library_tracker=overrides.get(
                "runtime_library_tracker",
                FixtureRuntimeTracker(runtime_policy_sha256),
            ),
        )

    def test_stable_backend_passes_all_scenarios_with_bounded_cost(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary))
            tracker = FixtureRuntimeTracker("8" * 64)
            receipt = self.evaluate(case, runtime_library_tracker=tracker)
            self.assertIs(receipt.decision, BatchInvarianceDecision.PASS)
            self.assertIs(receipt.promotion_decision, PromotionDecision.INCONCLUSIVE)
            self.assertEqual(receipt.summary.artifacts, 5)
            self.assertEqual(receipt.summary.artifact_evaluations, 40)
            self.assertEqual(receipt.summary.comparisons, 35)
            self.assertEqual(receipt.summary.compared_values, 105)
            self.assertEqual(receipt.cost.anchor_temporary_bytes, 120)
            self.assertEqual(receipt.cost.peak_nominal_input_tensor_bytes, 192)
            self.assertEqual(tracker.phases, list(RuntimeLibraryPhase))
            self.assertEqual(receipt.runtime_library_manifest.maps_snapshots, 4)
            self.assertEqual(
                BatchInvarianceReceipt.from_dict(receipt.to_dict()),
                receipt,
            )
            self.assertEqual(
                BatchInvariancePrecommitment.from_dict(
                    receipt.precommitment.to_dict()
                ),
                receipt.precommitment,
            )
            verify_batch_invariance_receipt_external_anchors(
                receipt,
                expected_precommitment_sha256=receipt.precommitment_sha256,
                expected_receipt_sha256=receipt.receipt_sha256,
            )

    def test_batch_size_slot_neighbor_duplicate_state_and_row_order_fail(self) -> None:
        for mode in (
            "batch_size", "slot", "neighbor", "duplicate", "stateful", "row_swap"
        ):
            with self.subTest(mode=mode), TemporaryDirectory() as temporary:
                case = self.build_case(Path(temporary), mode)
                receipt = self.evaluate(case)
                self.assertIs(receipt.decision, BatchInvarianceDecision.FAIL)
                self.assertTrue(receipt.hard_failures)

    def test_resource_cap_fails_before_backend_call_and_input_mutation_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary))
            with self.assertRaisesRegex(ValueError, "backend calls exceed"):
                self.evaluate(case, policy=policy(maximum_backend_calls=1))
            self.assertEqual(case["backend"].calls, 0)
            first = next(iter(case["artifact_paths"].values()))
            first.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "metadata differs|content differs"):
                self.evaluate(case)

    def test_receipt_parser_recomputes_failure_and_decision(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary))
            receipt = self.evaluate(case)
            forged = receipt.to_dict()
            forged["summary"]["maximum_raw_l2_drift"] = 1.0
            with self.assertRaisesRegex(ValueError, "failures disagree"):
                BatchInvarianceReceipt.from_dict(forged)

    def test_external_precommitment_mismatch_fails_before_backend_call(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary))
            with self.assertRaisesRegex(ValueError, "external anchor"):
                self.evaluate(
                    case,
                    expected_precommitment_sha256="4" * 64,
                )
            self.assertEqual(case["backend"].calls, 0)

    def test_worker_policy_and_environment_mismatch_fail_before_launch(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = self.build_case(root)
            carrier = root / "worker-carrier.json"
            carrier.write_text("{}\n", encoding="utf-8")
            files = {
                name: carrier
                for name in (
                    "inventory", "artifact_paths", "producer_config",
                    "onnx_config", "preprocessing", "model",
                    "model_lineage", "dependency_lock", "batch_policy",
                    "precommitment", "runtime_library_policy",
                )
            }
            execution_policy = BatchWorkerExecutionPolicy(
                supervisor=ProcessSupervisorPolicy(
                    timeout_seconds=30.0,
                    termination_grace_seconds=1.0,
                    poll_interval_seconds=0.01,
                    maximum_stdout_bytes=1_000,
                    maximum_stderr_bytes=1_000,
                )
            )
            _, environment = build_sanitized_worker_environment(
                os.environ,
                python_executable=sys.executable,
            )
            precommitment = build_batch_invariance_precommitment(
                inventory=case["inventory"],
                artifact_paths=case["artifact_paths"],
                producer_config=case["config"],
                provenance_paths=case["provenance_paths"],
                policy=policy(),
                runtime_library_policy_sha256="8" * 64,
                worker_execution_policy_sha256=execution_policy.policy_sha256,
                worker_environment_identity_sha256=environment.identity_sha256,
                prior_attempt_ledger_sha256="2" * 64,
                candidate_attempt_token="3" * 64,
                precommitment_sequence=1,
            )
            wrong_policy = replace(
                precommitment,
                worker_execution_policy_sha256="4" * 64,
            )
            wrong_environment = replace(
                precommitment,
                worker_environment_identity_sha256="5" * 64,
            )
            with patch(
                "cvi.batch_invariance_runner.run_supervised_process"
            ) as launch:
                with self.assertRaisesRegex(ValueError, "execution policy"):
                    run_batch_invariance_fresh_worker(
                        backend="cpu",
                        files=files,
                        precommitment=wrong_policy,
                        expected_precommitment_sha256=(
                            wrong_policy.precommitment_sha256
                        ),
                        python_executable=Path(sys.executable),
                        execution_policy=execution_policy,
                        discovery=False,
                    )
                with self.assertRaisesRegex(ValueError, "environment"):
                    run_batch_invariance_fresh_worker(
                        backend="cpu",
                        files=files,
                        precommitment=wrong_environment,
                        expected_precommitment_sha256=(
                            wrong_environment.precommitment_sha256
                        ),
                        python_executable=Path(sys.executable),
                        execution_policy=execution_policy,
                        discovery=False,
                    )
                launch.assert_not_called()

    def test_runtime_policy_mismatch_fails_before_backend_call(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary))
            with self.assertRaisesRegex(ValueError, "runtime library policy"):
                self.evaluate(
                    case,
                    runtime_library_tracker=FixtureRuntimeTracker("9" * 64),
                )
            self.assertEqual(case["backend"].calls, 0)

    def test_discovery_completes_without_admission_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary))
            tracker = FixtureRuntimeTracker("8" * 64, "DISCOVERY_ONLY")
            with self.assertRaises(BatchRuntimeDiscoveryComplete) as caught:
                self.evaluate(case, runtime_library_tracker=tracker)
            self.assertGreater(case["backend"].calls, 0)
            self.assertEqual(caught.exception.manifest.decision, "DISCOVERY_ONLY")

    def test_external_final_anchor_rejects_self_consistent_rewrite(self) -> None:
        with TemporaryDirectory() as temporary:
            case = self.build_case(Path(temporary))
            receipt = self.evaluate(case)
            forged = receipt.to_dict()
            forged["summary"]["maximum_absolute_error"] = 1.0
            forged["summary"]["maximum_relative_error"] = 1.0
            forged["summary"]["maximum_ulp_distance"] = 1
            rewritten = BatchInvarianceReceipt.from_dict(forged)
            self.assertNotEqual(rewritten.receipt_sha256, receipt.receipt_sha256)
            with self.assertRaisesRegex(ValueError, "external final anchor"):
                verify_batch_invariance_receipt_external_anchors(
                    rewritten,
                    expected_precommitment_sha256=receipt.precommitment_sha256,
                    expected_receipt_sha256=receipt.receipt_sha256,
                )

    def test_policy_requires_no_padding_and_exact_repeat(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbids padding"):
            replace(policy(), padding_policy="ZERO_PAD")
        with self.assertRaisesRegex(ValueError, "exactness is mandatory"):
            replace(policy(), require_repeated_composition_exact=False)

    def test_precommitment_and_external_verifier_cli_roundtrip(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = self.build_case(root)
            inventory_path = root / "inventory.json"
            artifact_paths_path = root / "artifact_paths.json"
            producer_path = root / "producer.json"
            policy_path = root / "policy.json"
            precommitment_path = root / "precommitment.json"
            runtime_policy_path = root / "runtime_policy.json"
            worker_policy_path = root / "worker_policy.json"
            receipt_path = root / "receipt.json"
            inventory_path.write_text(
                json.dumps(case["inventory"].to_dict()), encoding="utf-8"
            )
            artifact_paths_path.write_text(json.dumps({
                "schema_version": "cvi.batch_artifact_paths.v1",
                "entries": [
                    {"artifact_token": token, "path": str(path)}
                    for token, path in case["artifact_paths"].items()
                ],
            }), encoding="utf-8")
            producer_path.write_text(
                json.dumps(case["config"].to_dict()), encoding="utf-8"
            )
            policy_path.write_text(json.dumps(policy().to_dict()), encoding="utf-8")
            python_path = Path(sys.executable).resolve(strict=True)
            expected_python = ExpectedRuntimeBinary(
                resolved_path=str(python_path),
                byte_size=python_path.stat().st_size,
                content_sha256=digest(python_path.read_bytes()),
            )
            runtime_policy = RuntimeLibraryPolicy(
                expected_binaries=(expected_python,),
                discovery_binary_set_sha256=content_sha256([
                    (
                        expected_python.resolved_path,
                        expected_python.byte_size,
                        expected_python.content_sha256,
                    )
                ]),
            )
            runtime_policy_path.write_text(
                json.dumps(runtime_policy.to_dict()), encoding="utf-8"
            )
            worker_policy = BatchWorkerExecutionPolicy(
                supervisor=ProcessSupervisorPolicy(
                    timeout_seconds=30.0,
                    termination_grace_seconds=1.0,
                    poll_interval_seconds=0.01,
                    maximum_stdout_bytes=65_536,
                    maximum_stderr_bytes=65_536,
                )
            )
            worker_policy_path.write_text(
                json.dumps(worker_policy.to_dict()), encoding="utf-8"
            )
            repository = Path(__file__).resolve().parents[1]
            command = [
                sys.executable,
                "tools/create_batch_invariance_precommitment.py",
                "--inventory", str(inventory_path),
                "--artifact-paths", str(artifact_paths_path),
                "--producer-config", str(producer_path),
                "--model", str(case["provenance_paths"]["model"]),
                "--model-lineage",
                str(case["provenance_paths"]["model_lineage"]),
                "--preprocessing-config",
                str(case["provenance_paths"]["preprocessing"]),
                "--dependency-lock",
                str(case["provenance_paths"]["dependency_lock"]),
                "--policy", str(policy_path),
                "--runtime-library-policy", str(runtime_policy_path),
                "--worker-execution-policy", str(worker_policy_path),
                "--python-executable", sys.executable,
                "--prior-attempt-ledger-sha256", "2" * 64,
                "--candidate-attempt-token", "3" * 64,
                "--precommitment-sequence", "1",
                "--output", str(precommitment_path),
            ]
            completed = subprocess.run(
                command,
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            created = json.loads(completed.stdout)
            precommitment_payload = json.loads(precommitment_path.read_text())
            self.assertEqual(
                created["precommitment_sha256"],
                precommitment_payload["precommitment_sha256"],
            )

            _, worker_environment = build_sanitized_worker_environment(
                os.environ,
                python_executable=sys.executable,
            )
            batch_receipt = self.evaluate(
                case,
                worker_execution_policy_sha256=worker_policy.policy_sha256,
                worker_environment_identity_sha256=(
                    worker_environment.identity_sha256
                ),
            )
            supervised = SupervisedProcessResult(
                command=(
                    sys.executable, "-I", "-B", "-m",
                    "cvi.batch_invariance_worker", "--request", "/tmp/request",
                    "--result", "/tmp/result",
                ),
                policy_sha256=worker_policy.supervisor.policy_sha256,
                status=SupervisedProcessStatus.COMPLETED,
                return_code=0,
                wall_time_ns=1,
                sampled_peak_rss_bytes=1,
                rss_samples=1,
                rss_scope="fixture-main-process",
                stdout_bytes=0,
                stdout_sha256=digest(b""),
                stderr_bytes=0,
                stderr_sha256=digest(b""),
                stdout_complete=True,
                stderr_complete=True,
                termination_signal_sent=False,
                kill_signal_sent=False,
            )
            receipt = BatchFreshWorkerReceipt(
                worker_request_sha256="c" * 64,
                batch_receipt_sha256=batch_receipt.receipt_sha256,
                batch_receipt=batch_receipt,
                worker_environment_identity_sha256=(
                    worker_environment.identity_sha256
                ),
                worker_environment_identity=worker_environment,
                onnxruntime_distribution_name="onnxruntime",
                onnxruntime_distribution_version="1.27.0",
                execution_policy_sha256=worker_policy.policy_sha256,
                execution_policy=worker_policy,
                supervised_process_result_sha256=supervised.result_sha256,
                supervised_process_result=supervised,
            )
            receipt_path.write_text(json.dumps({
                "schema_version": "cvi.batch_invariance_bundle.v4",
                "receipt_sha256": receipt.receipt_sha256,
                "receipt": receipt.to_dict(),
            }), encoding="utf-8")
            verified = subprocess.run(
                [
                    sys.executable,
                    "tools/verify_batch_invariance_receipt.py",
                    "--receipt", str(receipt_path),
                    "--expected-precommitment-sha256",
                    receipt.batch_receipt.precommitment_sha256,
                    "--expected-receipt-sha256", receipt.receipt_sha256,
                ],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(verified.stdout)["status"], "VERIFIED")
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        "tools/verify_batch_invariance_receipt.py",
                        "--receipt", str(receipt_path),
                        "--expected-precommitment-sha256",
                        receipt.batch_receipt.precommitment_sha256,
                        "--expected-receipt-sha256", "9" * 64,
                    ],
                    cwd=repository,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            noisy_supervised = replace(
                supervised,
                stderr_bytes=1,
                stderr_sha256=digest(b"x"),
            )
            with self.assertRaisesRegex(ValueError, "emitted unexpected output"):
                replace(
                    receipt,
                    supervised_process_result=noisy_supervised,
                    supervised_process_result_sha256=(
                        noisy_supervised.result_sha256
                    ),
                )
            wrong_command = replace(
                supervised,
                command=(
                    *supervised.command[:4],
                    "cvi.other_worker",
                    *supervised.command[5:],
                ),
            )
            with self.assertRaisesRegex(ValueError, "command differs"):
                replace(
                    receipt,
                    supervised_process_result=wrong_command,
                    supervised_process_result_sha256=wrong_command.result_sha256,
                )

            rewritten_supervised = replace(supervised, wall_time_ns=2)
            rewritten = replace(
                receipt,
                supervised_process_result=rewritten_supervised,
                supervised_process_result_sha256=(
                    rewritten_supervised.result_sha256
                ),
            )
            self.assertNotEqual(rewritten.receipt_sha256, receipt.receipt_sha256)
            receipt_path.write_text(json.dumps({
                "schema_version": "cvi.batch_invariance_bundle.v4",
                "receipt_sha256": rewritten.receipt_sha256,
                "receipt": rewritten.to_dict(),
            }), encoding="utf-8")
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        "tools/verify_batch_invariance_receipt.py",
                        "--receipt", str(receipt_path),
                        "--expected-precommitment-sha256",
                        receipt.batch_receipt.precommitment_sha256,
                        "--expected-receipt-sha256", receipt.receipt_sha256,
                    ],
                    cwd=repository,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            receipt_path.write_text(json.dumps({
                "schema_version": "cvi.batch_invariance_bundle.v3",
                "receipt_sha256": batch_receipt.receipt_sha256,
                "receipt": batch_receipt.to_dict(),
            }), encoding="utf-8")
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        "tools/verify_batch_invariance_receipt.py",
                        "--receipt", str(receipt_path),
                        "--expected-precommitment-sha256",
                        batch_receipt.precommitment_sha256,
                        "--expected-receipt-sha256",
                        batch_receipt.receipt_sha256,
                    ],
                    cwd=repository,
                    check=True,
                    capture_output=True,
                    text=True,
                )

    def test_batch_runtime_policy_freeze_requires_repeated_discovery(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = Path(__file__).resolve().parents[1]
            discovery_policy = RuntimeLibraryPolicy(
                expected_binaries=(),
                allow_discovery_only=True,
            )
            discovery_policy_path = root / "discovery_policy.json"
            discovery_policy_path.write_text(
                json.dumps(discovery_policy.to_dict()), encoding="utf-8"
            )
            python_path = Path(sys.executable).resolve(strict=True)
            python_stat = python_path.stat()
            entry = RuntimeBinaryEntry(
                resolved_path=str(python_path),
                device_major=0,
                device_minor=0,
                inode=python_stat.st_ino,
                byte_size=python_stat.st_size,
                content_sha256=digest(python_path.read_bytes()),
                first_seen_phase=RuntimeLibraryPhase.DEPENDENCIES_IMPORTED,
                last_seen_phase=RuntimeLibraryPhase.FINAL_OUTPUT_READY,
            )
            binary_set_sha256 = content_sha256([
                (entry.resolved_path, entry.byte_size, entry.content_sha256)
            ])
            manifest = RuntimeLibraryManifest(
                policy_sha256=discovery_policy.policy_sha256,
                entries=(entry,),
                binary_set_sha256=binary_set_sha256,
                maps_snapshots=5,
                maps_bytes_read=100,
                binary_bytes_hashed=entry.byte_size,
                provenance_wall_time_ns=100,
                decision="DISCOVERY_ONLY",
                hard_failures=(),
            )
            execution_policy = BatchWorkerExecutionPolicy(
                supervisor=ProcessSupervisorPolicy(
                    timeout_seconds=30.0,
                    termination_grace_seconds=1.0,
                    poll_interval_seconds=0.01,
                    maximum_stdout_bytes=1_000,
                    maximum_stderr_bytes=1_000,
                )
            )
            _, worker_environment = build_sanitized_worker_environment(
                os.environ,
                python_executable=sys.executable,
            )
            supervised = SupervisedProcessResult(
                command=(
                    sys.executable, "-I", "-B", "-m",
                    "cvi.batch_invariance_worker", "--request", "/tmp/request",
                    "--result", "/tmp/result",
                ),
                policy_sha256=execution_policy.supervisor.policy_sha256,
                status=SupervisedProcessStatus.COMPLETED,
                return_code=0,
                wall_time_ns=1,
                sampled_peak_rss_bytes=1,
                rss_samples=1,
                rss_scope="fixture-main-process",
                stdout_bytes=0,
                stdout_sha256=digest(b""),
                stderr_bytes=0,
                stderr_sha256=digest(b""),
                stdout_complete=True,
                stderr_complete=True,
                termination_signal_sent=False,
                kill_signal_sent=False,
            )
            discovery_paths = []
            for index in range(2):
                discovery = BatchFreshWorkerDiscovery(
                    worker_request_sha256="a" * 64,
                    precommitment_sha256=str(index + 1) * 64,
                    runtime_library_manifest_sha256=manifest.manifest_sha256,
                    runtime_library_manifest=manifest,
                    worker_environment_identity_sha256=(
                        worker_environment.identity_sha256
                    ),
                    worker_environment_identity=worker_environment,
                    onnxruntime_distribution_name="onnxruntime",
                    onnxruntime_distribution_version="1.27.0",
                    execution_policy_sha256=execution_policy.policy_sha256,
                    execution_policy=execution_policy,
                    supervised_process_result_sha256=supervised.result_sha256,
                    supervised_process_result=supervised,
                )
                path = root / f"discovery-{index}.json"
                path.write_text(json.dumps({
                    "schema_version": (
                        "cvi.batch_runtime_library_discovery_bundle.v2"
                    ),
                    "discovery_sha256": discovery.discovery_sha256,
                    "discovery": discovery.to_dict(),
                }), encoding="utf-8")
                discovery_paths.append(path)
            strict_path = root / "strict.json"
            freeze_receipt_path = root / "freeze.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/freeze_batch_runtime_library_policy.py",
                    "--discovery-policy", str(discovery_policy_path),
                    "--discovery-manifest", str(discovery_paths[0]),
                    "--discovery-manifest", str(discovery_paths[1]),
                    "--policy", str(strict_path),
                    "--freeze-receipt", str(freeze_receipt_path),
                ],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.stdout, "")
            strict = RuntimeLibraryPolicy.from_dict(
                json.loads(strict_path.read_text())
            )
            self.assertFalse(strict.allow_discovery_only)
            self.assertEqual(len(strict.expected_binaries), 1)


if __name__ == "__main__":
    unittest.main()
