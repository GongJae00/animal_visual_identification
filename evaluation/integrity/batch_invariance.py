"""Label-blind batch-composition invariance admission for embedding backends."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from shared.contracts.runtime_library_provenance import (
    RuntimeLibraryManifest,
    RuntimeLibraryPhase,
    RuntimeLibraryTracker,
)
from data.acquisition import sha256_file
from evaluation.controls.control_scoring import ControlScoringInventory
from evaluation.integrity.operation_ports import EmbeddingBatchBackend, EmbeddingProducerConfig
from shared.foundation.provenance import content_sha256
from identification.training.appearance.optimization import PromotionDecision

_SCENARIOS = (
    "FULL_BATCH",
    "PERMUTED_NEIGHBORS",
    "DUPLICATE_PACKED",
    "TAIL_SIZE",
    "REPEATED_SAME_COMPOSITION",
)


def batch_artifact_paths_from_dict(payload: dict[str, Any]) -> dict[str, Path]:
    """Parse the strict token-to-path transport schema used by batch CLIs."""

    if set(payload) != {"schema_version", "entries"} or payload[
        "schema_version"
    ] != "cvi.batch_artifact_paths.v1":
        raise ValueError("batch artifact path schema differs")
    if not isinstance(payload["entries"], list) or not payload["entries"]:
        raise ValueError("batch artifact paths must be a nonempty list")
    result: dict[str, Path] = {}
    for entry in payload["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"artifact_token", "path"}:
            raise ValueError("batch artifact path entry keys differ")
        token, value = entry["artifact_token"], entry["path"]
        if not isinstance(token, str) or not token or not isinstance(value, str):
            raise ValueError("batch artifact path entry values differ")
        if token in result:
            raise ValueError("batch artifact path tokens must be unique")
        result[token] = Path(value)
    return result


class BatchInvarianceDecision(StrEnum):
    PASS = "BATCH_COMPOSITION_INVARIANCE_PASS"
    FAIL = "BATCH_COMPOSITION_INVARIANCE_FAIL"


class BatchRuntimeDiscoveryComplete(RuntimeError):
    """Carry discovery evidence without publishing a batch admission receipt."""

    def __init__(self, manifest: RuntimeLibraryManifest) -> None:
        super().__init__("batch runtime-library discovery completed")
        self.manifest = manifest


@dataclass(frozen=True, slots=True)
class BatchInvariancePolicy:
    absolute_tolerance: float
    relative_tolerance: float
    relative_floor: float
    maximum_raw_l2_drift: float
    maximum_raw_norm_drift: float
    maximum_normalized_l2_drift: float
    maximum_cosine_drift: float
    maximum_artifacts: int = 64
    maximum_vector_dimension: int = 65_536
    maximum_backend_calls: int = 1_024
    maximum_artifact_evaluations: int = 10_000
    maximum_comparison_values: int = 100_000_000
    maximum_input_bytes_hashed: int = 17_179_869_184
    maximum_provenance_bytes_hashed: int = 1_073_741_824
    maximum_anchor_temporary_bytes: int = 1_073_741_824
    require_repeated_composition_exact: bool = True
    padding_policy: str = "FORBIDDEN"
    schema_version: str = "cvi.batch_invariance_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.batch_invariance_policy.v1":
            raise ValueError("unsupported batch invariance policy schema")
        for name in (
            "absolute_tolerance",
            "relative_tolerance",
            "maximum_raw_l2_drift",
            "maximum_raw_norm_drift",
            "maximum_normalized_l2_drift",
            "maximum_cosine_drift",
        ):
            _finite_nonnegative(getattr(self, name), name)
        _finite_positive(self.relative_floor, "relative_floor")
        for name in (
            "maximum_artifacts",
            "maximum_vector_dimension",
            "maximum_backend_calls",
            "maximum_artifact_evaluations",
            "maximum_comparison_values",
            "maximum_input_bytes_hashed",
            "maximum_provenance_bytes_hashed",
            "maximum_anchor_temporary_bytes",
        ):
            _positive_int(getattr(self, name), name)
        if self.require_repeated_composition_exact is not True:
            raise ValueError("repeated-composition exactness is mandatory")
        if self.padding_policy != "FORBIDDEN":
            raise ValueError("initial batch invariance gate forbids padding")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BatchInvariancePolicy:
        _exact_keys(payload, set(cls.__dataclass_fields__), "batch policy")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class BatchInvariancePrecommitment:
    inventory_sha256: str
    producer_config_sha256: str
    policy_sha256: str
    schedule_manifest_sha256: str
    backend_identity_sha256: str
    runtime_library_policy_sha256: str
    worker_execution_policy_sha256: str
    worker_environment_identity_sha256: str
    artifact_bindings: tuple[tuple[str, str, int], ...]
    provenance_sha256: tuple[tuple[str, str], ...]
    input_bytes_hashed: int
    provenance_bytes_hashed: int
    prior_attempt_ledger_sha256: str
    candidate_attempt_token: str
    precommitment_sequence: int
    selection_blind_to_candidate_outputs: bool = True
    schema_version: str = "cvi.batch_invariance_precommitment.v3"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.batch_invariance_precommitment.v3":
            raise ValueError("unsupported batch precommitment schema")
        for name in (
            "inventory_sha256",
            "producer_config_sha256",
            "policy_sha256",
            "schedule_manifest_sha256",
            "backend_identity_sha256",
            "runtime_library_policy_sha256",
            "worker_execution_policy_sha256",
            "worker_environment_identity_sha256",
            "prior_attempt_ledger_sha256",
            "candidate_attempt_token",
        ):
            _sha256(getattr(self, name), name)
        if not self.artifact_bindings:
            raise ValueError("batch precommitment artifacts must not be empty")
        tokens: list[str] = []
        contents: list[str] = []
        for token, content, byte_size in self.artifact_bindings:
            if not token:
                raise ValueError("batch precommitment token is empty")
            _sha256(content, "artifact content")
            _positive_int(byte_size, "artifact byte_size")
            tokens.append(token)
            contents.append(content)
        if len(tokens) != len(set(tokens)) or len(contents) != len(set(contents)):
            raise ValueError("batch precommitment artifacts must be unique")
        if tuple(sorted(self.provenance_sha256)) != self.provenance_sha256:
            raise ValueError("batch precommitment provenance must be key sorted")
        for name, digest in self.provenance_sha256:
            if not name:
                raise ValueError("batch precommitment provenance name is empty")
            _sha256(digest, "provenance digest")
        _positive_int(self.input_bytes_hashed, "input_bytes_hashed")
        _positive_int(self.provenance_bytes_hashed, "provenance_bytes_hashed")
        _positive_int(self.precommitment_sequence, "precommitment_sequence")
        if self.input_bytes_hashed != sum(
            item[2] for item in self.artifact_bindings
        ):
            raise ValueError("batch precommitment input-byte accounting differs")
        if self.selection_blind_to_candidate_outputs is not True:
            raise ValueError("batch precommitment must be output blind")

    @property
    def precommitment_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inventory_sha256": self.inventory_sha256,
            "producer_config_sha256": self.producer_config_sha256,
            "policy_sha256": self.policy_sha256,
            "schedule_manifest_sha256": self.schedule_manifest_sha256,
            "backend_identity_sha256": self.backend_identity_sha256,
            "runtime_library_policy_sha256": (
                self.runtime_library_policy_sha256
            ),
            "worker_execution_policy_sha256": (
                self.worker_execution_policy_sha256
            ),
            "worker_environment_identity_sha256": (
                self.worker_environment_identity_sha256
            ),
            "artifact_bindings": [list(item) for item in self.artifact_bindings],
            "provenance_sha256": [list(item) for item in self.provenance_sha256],
            "input_bytes_hashed": self.input_bytes_hashed,
            "provenance_bytes_hashed": self.provenance_bytes_hashed,
            "prior_attempt_ledger_sha256": self.prior_attempt_ledger_sha256,
            "candidate_attempt_token": self.candidate_attempt_token,
            "precommitment_sequence": self.precommitment_sequence,
            "selection_blind_to_candidate_outputs": (
                self.selection_blind_to_candidate_outputs
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> BatchInvariancePrecommitment:
        _exact_keys(payload, set(cls.__dataclass_fields__), "batch precommitment")
        bindings = payload["artifact_bindings"]
        provenance = payload["provenance_sha256"]
        if not isinstance(bindings, list) or not isinstance(provenance, list):
            raise TypeError("batch precommitment collections must be lists")
        values = dict(payload)
        values["artifact_bindings"] = tuple(tuple(item) for item in bindings)
        values["provenance_sha256"] = tuple(tuple(item) for item in provenance)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class BatchScenarioSummary:
    scenario: str
    backend_calls: int
    artifact_evaluations: int
    comparisons: int
    maximum_raw_l2_drift: float
    maximum_normalized_l2_drift: float
    maximum_cosine_drift: float
    schema_version: str = "cvi.batch_scenario_summary.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.batch_scenario_summary.v1":
            raise ValueError("unsupported batch scenario summary schema")
        if self.scenario not in _SCENARIOS:
            raise ValueError("unknown batch invariance scenario")
        for name in ("backend_calls", "artifact_evaluations", "comparisons"):
            _positive_int(getattr(self, name), name)
        if self.artifact_evaluations != self.comparisons:
            raise ValueError("every scenario output must be compared")
        for name in (
            "maximum_raw_l2_drift",
            "maximum_normalized_l2_drift",
            "maximum_cosine_drift",
        ):
            _finite_nonnegative(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BatchScenarioSummary:
        _exact_keys(payload, set(cls.__dataclass_fields__), "scenario summary")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class BatchInvarianceSummary:
    artifacts: int
    vector_dimension: int
    backend_calls: int
    artifact_evaluations: int
    comparisons: int
    compared_values: int
    violated_occurrences: int
    violated_values: int
    repeated_digest_mismatches: int
    maximum_absolute_error: float
    mean_absolute_error: float
    maximum_relative_error: float
    maximum_ulp_distance: int
    maximum_raw_l2_drift: float
    maximum_raw_norm_drift: float
    maximum_normalized_l2_drift: float
    maximum_cosine_drift: float
    worst_artifact_content_sha256: str | None
    worst_coordinate: int | None
    worst_scenario: str | None
    scenario_summaries: tuple[BatchScenarioSummary, ...]
    anchor_digest: str
    output_digest: str
    schema_version: str = "cvi.batch_invariance_summary.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.batch_invariance_summary.v1":
            raise ValueError("unsupported batch invariance summary schema")
        for name in (
            "artifacts", "vector_dimension", "backend_calls",
            "artifact_evaluations", "comparisons", "compared_values",
        ):
            _positive_int(getattr(self, name), name)
        for name in (
            "violated_occurrences", "violated_values",
            "repeated_digest_mismatches", "maximum_ulp_distance",
        ):
            _nonnegative_int(getattr(self, name), name)
        if self.compared_values != self.comparisons * self.vector_dimension:
            raise ValueError("batch comparison value accounting differs")
        if self.violated_occurrences > self.comparisons:
            raise ValueError("violated batch occurrences exceed comparisons")
        if self.violated_values > self.compared_values:
            raise ValueError("violated batch values exceed comparisons")
        for name in (
            "maximum_absolute_error", "mean_absolute_error",
            "maximum_relative_error", "maximum_raw_l2_drift",
            "maximum_raw_norm_drift", "maximum_normalized_l2_drift",
            "maximum_cosine_drift",
        ):
            _finite_nonnegative(getattr(self, name), name)
        if self.mean_absolute_error > self.maximum_absolute_error:
            raise ValueError("mean batch error exceeds maximum")
        if (self.worst_artifact_content_sha256 is None) != (
            self.worst_coordinate is None or self.worst_scenario is None
        ):
            raise ValueError("worst batch drift fields must be jointly present")
        if self.worst_artifact_content_sha256 is not None:
            _sha256(self.worst_artifact_content_sha256, "worst artifact")
            _nonnegative_int(self.worst_coordinate, "worst coordinate")
            if self.worst_scenario not in _SCENARIOS:
                raise ValueError("worst batch scenario is invalid")
        if tuple(item.scenario for item in self.scenario_summaries) != _SCENARIOS:
            raise ValueError("batch scenario summaries are incomplete")
        _sha256(self.anchor_digest, "anchor_digest")
        _sha256(self.output_digest, "output_digest")

    def to_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        payload["scenario_summaries"] = [
            item.to_dict() for item in self.scenario_summaries
        ]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BatchInvarianceSummary:
        _exact_keys(payload, set(cls.__dataclass_fields__), "batch summary")
        values = dict(payload)
        if not isinstance(values["scenario_summaries"], list):
            raise TypeError("scenario summaries must be a list")
        values["scenario_summaries"] = tuple(
            BatchScenarioSummary.from_dict(item)
            for item in values["scenario_summaries"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class BatchInvarianceCost:
    input_integrity_bytes_read: int
    provenance_integrity_bytes_read: int
    anchor_temporary_bytes: int
    anchor_bytes_read: int
    output_bytes_compared: int
    backend_calls: int
    artifact_evaluations: int
    peak_nominal_input_tensor_bytes: int
    peak_nominal_output_bytes: int
    schema_version: str = "cvi.batch_invariance_cost.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.batch_invariance_cost.v1":
            raise ValueError("unsupported batch invariance cost schema")
        for name in self.__dataclass_fields__:
            if name != "schema_version":
                _nonnegative_int(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BatchInvarianceCost:
        _exact_keys(payload, set(cls.__dataclass_fields__), "batch cost")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class BatchInvarianceReceipt:
    precommitment_sha256: str
    precommitment: BatchInvariancePrecommitment
    runtime_library_policy_sha256: str
    runtime_library_manifest_sha256: str
    runtime_library_binary_set_sha256: str
    runtime_library_manifest: RuntimeLibraryManifest
    inventory_sha256: str
    producer_config_sha256: str
    policy_sha256: str
    schedule_manifest_sha256: str
    backend_identity_sha256: str
    actual_providers: tuple[str, ...]
    actual_provider_options_sha256: str
    artifact_content_sha256: tuple[str, ...]
    provenance_sha256: tuple[tuple[str, str], ...]
    policy: BatchInvariancePolicy
    summary: BatchInvarianceSummary
    cost: BatchInvarianceCost
    hard_failures: tuple[str, ...]
    decision: BatchInvarianceDecision
    promotion_decision: PromotionDecision = PromotionDecision.INCONCLUSIVE
    interpretation: str = (
        "BATCH_INVARIANCE_ONLY_NOT_BIOMETRIC_NONINFERIORITY_"
        "OR_OPTIMIZATION_PROMOTION"
    )
    schema_version: str = "cvi.batch_invariance_receipt.v3"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.batch_invariance_receipt.v3":
            raise ValueError("unsupported batch invariance receipt schema")
        for name in (
            "precommitment_sha256", "runtime_library_policy_sha256",
            "runtime_library_manifest_sha256",
            "runtime_library_binary_set_sha256", "inventory_sha256",
            "producer_config_sha256", "policy_sha256",
            "schedule_manifest_sha256", "backend_identity_sha256",
            "actual_provider_options_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.precommitment.precommitment_sha256 != self.precommitment_sha256:
            raise ValueError("embedded batch precommitment hash differs")
        if self.precommitment.runtime_library_policy_sha256 != (
            self.runtime_library_policy_sha256
        ):
            raise ValueError("batch runtime policy differs from precommitment")
        if self.runtime_library_manifest.policy_sha256 != (
            self.runtime_library_policy_sha256
        ):
            raise ValueError("batch runtime manifest policy differs")
        if self.runtime_library_manifest.manifest_sha256 != (
            self.runtime_library_manifest_sha256
        ):
            raise ValueError("embedded batch runtime manifest hash differs")
        if self.runtime_library_manifest.binary_set_sha256 != (
            self.runtime_library_binary_set_sha256
        ):
            raise ValueError("batch runtime binary set differs from manifest")
        if self.runtime_library_manifest.decision != "PASS":
            raise ValueError("batch runtime library admission must pass")
        precommitment_bindings = (
            ("inventory_sha256", self.inventory_sha256),
            ("producer_config_sha256", self.producer_config_sha256),
            ("policy_sha256", self.policy_sha256),
            ("schedule_manifest_sha256", self.schedule_manifest_sha256),
            ("backend_identity_sha256", self.backend_identity_sha256),
        )
        for name, receipt_value in precommitment_bindings:
            if getattr(self.precommitment, name) != receipt_value:
                raise ValueError(f"batch receipt {name} differs from precommitment")
        if not self.actual_providers or any(
            not isinstance(item, str) or not item for item in self.actual_providers
        ):
            raise ValueError("batch receipt providers must be non-empty")
        if not self.artifact_content_sha256:
            raise ValueError("batch receipt artifact contents must not be empty")
        for value in self.artifact_content_sha256:
            _sha256(value, "artifact content")
        if tuple(
            item[1] for item in self.precommitment.artifact_bindings
        ) != self.artifact_content_sha256:
            raise ValueError("batch receipt artifacts differ from precommitment")
        if tuple(sorted(self.provenance_sha256)) != self.provenance_sha256:
            raise ValueError("batch provenance must be key sorted")
        for name, value in self.provenance_sha256:
            if not name:
                raise ValueError("batch provenance name must not be empty")
            _sha256(value, "provenance digest")
        if self.precommitment.provenance_sha256 != self.provenance_sha256:
            raise ValueError("batch receipt provenance differs from precommitment")
        if self.policy.policy_sha256 != self.policy_sha256:
            raise ValueError("embedded batch policy hash differs")
        expected_failures = _hard_failures(self.summary, self.policy)
        if self.hard_failures != expected_failures:
            raise ValueError("batch invariance failures disagree with policy")
        expected_decision = (
            BatchInvarianceDecision.FAIL
            if expected_failures else BatchInvarianceDecision.PASS
        )
        if self.decision is not expected_decision:
            raise ValueError("batch invariance decision differs from failures")
        if self.promotion_decision is not PromotionDecision.INCONCLUSIVE:
            raise ValueError("batch invariance cannot promote an optimization")
        if self.cost.backend_calls != self.summary.backend_calls or (
            self.cost.artifact_evaluations != self.summary.artifact_evaluations
        ):
            raise ValueError("batch invariance cost accounting differs")
        if self.interpretation != (
            "BATCH_INVARIANCE_ONLY_NOT_BIOMETRIC_NONINFERIORITY_"
            "OR_OPTIMIZATION_PROMOTION"
        ):
            raise ValueError("batch invariance interpretation is fixed")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "precommitment_sha256": self.precommitment_sha256,
            "precommitment": self.precommitment.to_dict(),
            "runtime_library_policy_sha256": (
                self.runtime_library_policy_sha256
            ),
            "runtime_library_manifest_sha256": (
                self.runtime_library_manifest_sha256
            ),
            "runtime_library_binary_set_sha256": (
                self.runtime_library_binary_set_sha256
            ),
            "runtime_library_manifest": self.runtime_library_manifest.to_dict(),
            "inventory_sha256": self.inventory_sha256,
            "producer_config_sha256": self.producer_config_sha256,
            "policy_sha256": self.policy_sha256,
            "schedule_manifest_sha256": self.schedule_manifest_sha256,
            "backend_identity_sha256": self.backend_identity_sha256,
            "actual_providers": list(self.actual_providers),
            "actual_provider_options_sha256": self.actual_provider_options_sha256,
            "artifact_content_sha256": list(self.artifact_content_sha256),
            "provenance_sha256": [list(item) for item in self.provenance_sha256],
            "policy": self.policy.to_dict(),
            "summary": self.summary.to_dict(),
            "cost": self.cost.to_dict(),
            "hard_failures": list(self.hard_failures),
            "decision": self.decision.value,
            "promotion_decision": self.promotion_decision.value,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BatchInvarianceReceipt:
        _exact_keys(payload, set(cls.__dataclass_fields__), "batch receipt")
        values = dict(payload)
        values["precommitment"] = BatchInvariancePrecommitment.from_dict(
            values["precommitment"]
        )
        values["runtime_library_manifest"] = RuntimeLibraryManifest.from_dict(
            values["runtime_library_manifest"]
        )
        for name in ("actual_providers", "artifact_content_sha256", "hard_failures"):
            if not isinstance(values[name], list):
                raise TypeError(f"{name} must be a list")
            values[name] = tuple(values[name])
        if not isinstance(values["provenance_sha256"], list):
            raise TypeError("provenance_sha256 must be a list")
        values["provenance_sha256"] = tuple(
            tuple(item) for item in values["provenance_sha256"]
        )
        values["policy"] = BatchInvariancePolicy.from_dict(values["policy"])
        values["summary"] = BatchInvarianceSummary.from_dict(values["summary"])
        values["cost"] = BatchInvarianceCost.from_dict(values["cost"])
        values["decision"] = BatchInvarianceDecision(values["decision"])
        values["promotion_decision"] = PromotionDecision(values["promotion_decision"])
        return cls(**values)


@dataclass(slots=True)
class _Accumulator:
    absolute_sum: float = 0.0
    absolute_compensation: float = 0.0
    maximum_absolute: float = 0.0
    maximum_relative: float = 0.0
    maximum_ulp: int = 0
    maximum_raw_l2: float = 0.0
    maximum_raw_norm: float = 0.0
    maximum_normalized_l2: float = 0.0
    maximum_cosine: float = 0.0
    violated_occurrences: int = 0
    violated_values: int = 0
    repeated_digest_mismatches: int = 0
    worst_content: str | None = None
    worst_coordinate: int | None = None
    worst_scenario: str | None = None

    def add_absolute(self, value: float) -> None:
        updated = self.absolute_sum + value
        if abs(self.absolute_sum) >= abs(value):
            self.absolute_compensation += (self.absolute_sum - updated) + value
        else:
            self.absolute_compensation += (value - updated) + self.absolute_sum
        self.absolute_sum = updated

    @property
    def absolute_total(self) -> float:
        return self.absolute_sum + self.absolute_compensation


def build_batch_invariance_precommitment(
    *,
    inventory: ControlScoringInventory,
    artifact_paths: Mapping[str, Path],
    producer_config: EmbeddingProducerConfig,
    provenance_paths: Mapping[str, Path],
    policy: BatchInvariancePolicy,
    runtime_library_policy_sha256: str,
    worker_execution_policy_sha256: str,
    worker_environment_identity_sha256: str,
    prior_attempt_ledger_sha256: str,
    candidate_attempt_token: str,
    precommitment_sequence: int,
) -> BatchInvariancePrecommitment:
    """Freeze the complete label-blind batch experiment before inference."""

    ordered = _validate_inputs(inventory, artifact_paths)
    provenance = _validate_provenance(provenance_paths, producer_config)
    input_bytes = sum(path.stat().st_size for _, _, path in ordered)
    provenance_bytes = sum(
        path.resolve(strict=True).stat().st_size
        for path in provenance_paths.values()
    )
    schedules, _, _, _ = _validate_plan_limits(
        artifact_count=len(ordered),
        input_bytes=input_bytes,
        provenance_bytes=provenance_bytes,
        producer_config=producer_config,
        policy=policy,
    )
    return _precommitment_from_validated(
        inventory=inventory,
        ordered=ordered,
        producer_config=producer_config,
        provenance=provenance,
        policy=policy,
        schedules=schedules,
        input_bytes=input_bytes,
        provenance_bytes=provenance_bytes,
        runtime_library_policy_sha256=runtime_library_policy_sha256,
        worker_execution_policy_sha256=worker_execution_policy_sha256,
        worker_environment_identity_sha256=(
            worker_environment_identity_sha256
        ),
        prior_attempt_ledger_sha256=prior_attempt_ledger_sha256,
        candidate_attempt_token=candidate_attempt_token,
        precommitment_sequence=precommitment_sequence,
    )


def evaluate_batch_composition_invariance(
    *,
    backend: EmbeddingBatchBackend,
    inventory: ControlScoringInventory,
    artifact_paths: Mapping[str, Path],
    producer_config: EmbeddingProducerConfig,
    provenance_paths: Mapping[str, Path],
    policy: BatchInvariancePolicy,
    precommitment: BatchInvariancePrecommitment,
    expected_precommitment_sha256: str,
    runtime_library_tracker: RuntimeLibraryTracker,
    temporary_directory_parent: Path | None = None,
) -> BatchInvarianceReceipt:
    """Compare fixed batch scenarios to disk-backed singleton anchors."""

    _sha256(expected_precommitment_sha256, "expected_precommitment_sha256")
    if precommitment.precommitment_sha256 != expected_precommitment_sha256:
        raise ValueError("batch precommitment differs from external anchor")
    if runtime_library_tracker.policy.policy_sha256 != (
        precommitment.runtime_library_policy_sha256
    ):
        raise ValueError("batch runtime library policy differs from precommitment")
    _validate_backend_lineage(backend, producer_config)
    ordered = _validate_inputs(inventory, artifact_paths)
    provenance = _validate_provenance(provenance_paths, producer_config)
    input_bytes = sum(path.stat().st_size for _, _, path in ordered)
    provenance_bytes = sum(
        path.resolve(strict=True).stat().st_size
        for path in provenance_paths.values()
    )
    dimension = producer_config.vector_dimension
    schedules, backend_calls, evaluations, comparison_values = (
        _validate_plan_limits(
            artifact_count=len(ordered),
            input_bytes=input_bytes,
            provenance_bytes=provenance_bytes,
            producer_config=producer_config,
            policy=policy,
        )
    )
    comparisons = evaluations - len(ordered)
    anchor_temp_bytes = len(ordered) * dimension * 4 * 2
    derived_precommitment = _precommitment_from_validated(
        inventory=inventory,
        ordered=ordered,
        producer_config=producer_config,
        provenance=provenance,
        policy=policy,
        schedules=schedules,
        input_bytes=input_bytes,
        provenance_bytes=provenance_bytes,
        runtime_library_policy_sha256=(
            precommitment.runtime_library_policy_sha256
        ),
        worker_execution_policy_sha256=(
            precommitment.worker_execution_policy_sha256
        ),
        worker_environment_identity_sha256=(
            precommitment.worker_environment_identity_sha256
        ),
        prior_attempt_ledger_sha256=(
            precommitment.prior_attempt_ledger_sha256
        ),
        candidate_attempt_token=precommitment.candidate_attempt_token,
        precommitment_sequence=precommitment.precommitment_sequence,
    )
    if derived_precommitment != precommitment:
        raise ValueError("batch execution inputs differ from precommitment")
    runtime_library_tracker.capture(RuntimeLibraryPhase.SESSION_READY)

    accumulator = _Accumulator()
    anchor_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    scenario_summaries: list[BatchScenarioSummary] = []
    full_digests: dict[int, str] = {}
    with TemporaryDirectory(
        prefix="cvi-batch-anchor-",
        dir=temporary_directory_parent,
    ) as temporary:
        anchor_root = Path(temporary)
        for index, (content, _, path) in enumerate(ordered):
            rows = _infer(backend, (path,), dimension)
            if index == 0:
                runtime_library_tracker.capture(
                    RuntimeLibraryPhase.FIRST_OUTPUT_READY
                )
            raw = rows[0]
            normalized = _normalized_bytes(raw, producer_config.l2_epsilon)
            (anchor_root / f"{index:06d}.raw").write_bytes(raw)
            (anchor_root / f"{index:06d}.norm").write_bytes(normalized)
            anchor_digest.update(bytes.fromhex(content))
            anchor_digest.update(raw)
            anchor_digest.update(normalized)

        for scenario, batches in schedules:
            scenario_calls = 0
            scenario_evaluations = 0
            scenario_raw_l2 = 0.0
            scenario_normalized_l2 = 0.0
            scenario_cosine = 0.0
            for batch_number, indices in enumerate(batches):
                paths = tuple(ordered[index][2] for index in indices)
                rows = _infer(backend, paths, dimension)
                scenario_calls += 1
                scenario_evaluations += len(indices)
                for slot, (index, raw) in enumerate(zip(indices, rows, strict=True)):
                    content = ordered[index][0]
                    reference_raw = (anchor_root / f"{index:06d}.raw").read_bytes()
                    reference_norm = (anchor_root / f"{index:06d}.norm").read_bytes()
                    normalized = _normalized_bytes(raw, producer_config.l2_epsilon)
                    metrics = _compare_vectors(
                        reference_raw,
                        raw,
                        reference_norm,
                        normalized,
                        policy,
                        accumulator,
                        content,
                        scenario,
                    )
                    scenario_raw_l2 = max(scenario_raw_l2, metrics[0])
                    scenario_normalized_l2 = max(
                        scenario_normalized_l2, metrics[1]
                    )
                    scenario_cosine = max(scenario_cosine, metrics[2])
                    occurrence = content_sha256(
                        {
                            "content_sha256": content,
                            "scenario": scenario,
                            "batch": batch_number,
                            "slot": slot,
                        }
                    )
                    output_digest.update(bytes.fromhex(occurrence))
                    output_digest.update(raw)
                    raw_digest = hashlib.sha256(raw).hexdigest()
                    if scenario == "FULL_BATCH":
                        full_digests[index] = raw_digest
                    elif scenario == "REPEATED_SAME_COMPOSITION" and (
                        full_digests.get(index) != raw_digest
                    ):
                        accumulator.repeated_digest_mismatches += 1
            scenario_summaries.append(
                BatchScenarioSummary(
                    scenario=scenario,
                    backend_calls=scenario_calls,
                    artifact_evaluations=scenario_evaluations,
                    comparisons=scenario_evaluations,
                    maximum_raw_l2_drift=scenario_raw_l2,
                    maximum_normalized_l2_drift=scenario_normalized_l2,
                    maximum_cosine_drift=scenario_cosine,
                )
            )

    _validate_inputs(inventory, artifact_paths)
    _validate_provenance(provenance_paths, producer_config)
    runtime_library_tracker.capture(RuntimeLibraryPhase.FINAL_OUTPUT_READY)
    runtime_library_manifest = runtime_library_tracker.finalize()
    if runtime_library_manifest.decision == "DISCOVERY_ONLY":
        raise BatchRuntimeDiscoveryComplete(runtime_library_manifest)
    if runtime_library_manifest.decision != "PASS":
        raise ValueError("batch runtime library admission did not pass")
    actual_providers = tuple(getattr(backend, "actual_providers", ("UNAVAILABLE",)))
    actual_options = getattr(backend, "actual_provider_options", {})
    summary = BatchInvarianceSummary(
        artifacts=len(ordered),
        vector_dimension=dimension,
        backend_calls=backend_calls,
        artifact_evaluations=evaluations,
        comparisons=comparisons,
        compared_values=comparison_values,
        violated_occurrences=accumulator.violated_occurrences,
        violated_values=accumulator.violated_values,
        repeated_digest_mismatches=accumulator.repeated_digest_mismatches,
        maximum_absolute_error=accumulator.maximum_absolute,
        mean_absolute_error=accumulator.absolute_total / comparison_values,
        maximum_relative_error=accumulator.maximum_relative,
        maximum_ulp_distance=accumulator.maximum_ulp,
        maximum_raw_l2_drift=accumulator.maximum_raw_l2,
        maximum_raw_norm_drift=accumulator.maximum_raw_norm,
        maximum_normalized_l2_drift=accumulator.maximum_normalized_l2,
        maximum_cosine_drift=accumulator.maximum_cosine,
        worst_artifact_content_sha256=accumulator.worst_content,
        worst_coordinate=accumulator.worst_coordinate,
        worst_scenario=accumulator.worst_scenario,
        scenario_summaries=tuple(scenario_summaries),
        anchor_digest=anchor_digest.hexdigest(),
        output_digest=output_digest.hexdigest(),
    )
    failures = _hard_failures(summary, policy)
    return BatchInvarianceReceipt(
        precommitment_sha256=precommitment.precommitment_sha256,
        precommitment=precommitment,
        runtime_library_policy_sha256=(
            precommitment.runtime_library_policy_sha256
        ),
        runtime_library_manifest_sha256=(
            runtime_library_manifest.manifest_sha256
        ),
        runtime_library_binary_set_sha256=(
            runtime_library_manifest.binary_set_sha256
        ),
        runtime_library_manifest=runtime_library_manifest,
        inventory_sha256=inventory.inventory_sha256,
        producer_config_sha256=producer_config.config_sha256,
        policy_sha256=policy.policy_sha256,
        schedule_manifest_sha256=precommitment.schedule_manifest_sha256,
        backend_identity_sha256=content_sha256(backend.identity.to_dict()),
        actual_providers=actual_providers,
        actual_provider_options_sha256=content_sha256(actual_options),
        artifact_content_sha256=tuple(item[0] for item in ordered),
        provenance_sha256=tuple(sorted(provenance.items())),
        policy=policy,
        summary=summary,
        cost=BatchInvarianceCost(
            input_integrity_bytes_read=input_bytes * 2,
            provenance_integrity_bytes_read=provenance_bytes * 2,
            anchor_temporary_bytes=anchor_temp_bytes,
            anchor_bytes_read=comparisons * dimension * 4 * 2,
            output_bytes_compared=comparisons * dimension * 4,
            backend_calls=backend_calls,
            artifact_evaluations=evaluations,
            peak_nominal_input_tensor_bytes=(
                producer_config.batch_size
                * producer_config.input_width
                * producer_config.input_height
                * producer_config.input_channels
                * producer_config.input_value_bytes
            ),
            peak_nominal_output_bytes=(
                producer_config.batch_size * dimension * 4
            ),
        ),
        hard_failures=failures,
        decision=(
            BatchInvarianceDecision.FAIL if failures
            else BatchInvarianceDecision.PASS
        ),
    )


def _build_schedules(
    count: int,
    batch_size: int,
) -> tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]:
    canonical = tuple(range(count))
    shift = max(1, count // 2)
    permuted = canonical[shift:] + canonical[:shift]
    remainder = count % batch_size
    tail_size = remainder if remainder else batch_size - 1
    duplicate = tuple((index, (index + 1) % count, index) for index in canonical)
    return (
        ("FULL_BATCH", _chunks(canonical, batch_size)),
        ("PERMUTED_NEIGHBORS", _chunks(permuted, batch_size)),
        ("DUPLICATE_PACKED", duplicate),
        ("TAIL_SIZE", _chunks(canonical, tail_size)),
        ("REPEATED_SAME_COMPOSITION", _chunks(canonical, batch_size)),
    )


def _schedule_manifest(
    ordered: tuple[tuple[str, str, Path], ...],
    batch_size: int,
    schedules: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...],
) -> dict[str, Any]:
    return {
        "schema_version": "cvi.batch_invariance_schedule.v1",
        "artifact_content_sha256": [item[0] for item in ordered],
        "batch_size": batch_size,
        "padding_policy": "FORBIDDEN",
        "scenarios": [
            {"scenario": name, "batches": [list(batch) for batch in batches]}
            for name, batches in schedules
        ],
    }


def _validate_plan_limits(
    *,
    artifact_count: int,
    input_bytes: int,
    provenance_bytes: int,
    producer_config: EmbeddingProducerConfig,
    policy: BatchInvariancePolicy,
) -> tuple[
    tuple[tuple[str, tuple[tuple[int, ...], ...]], ...],
    int,
    int,
    int,
]:
    if artifact_count < 3:
        raise ValueError("batch invariance requires at least three artifacts")
    if artifact_count > policy.maximum_artifacts:
        raise ValueError("batch invariance artifacts exceed policy")
    if producer_config.vector_dimension > policy.maximum_vector_dimension:
        raise ValueError("batch invariance dimension exceeds policy")
    if producer_config.batch_size < 3:
        raise ValueError("batch invariance requires production batch size >= 3")
    if input_bytes * 2 > policy.maximum_input_bytes_hashed:
        raise ValueError("batch input integrity bytes exceed policy")
    if provenance_bytes * 2 > policy.maximum_provenance_bytes_hashed:
        raise ValueError("batch provenance integrity bytes exceed policy")
    anchor_temp_bytes = artifact_count * producer_config.vector_dimension * 4 * 2
    if anchor_temp_bytes > policy.maximum_anchor_temporary_bytes:
        raise ValueError("batch anchor temporary bytes exceed policy")
    schedules = _build_schedules(artifact_count, producer_config.batch_size)
    backend_calls = artifact_count + sum(
        len(batches) for _, batches in schedules
    )
    evaluations = artifact_count + sum(
        len(batch) for _, batches in schedules for batch in batches
    )
    comparison_values = (
        evaluations - artifact_count
    ) * producer_config.vector_dimension
    if backend_calls > policy.maximum_backend_calls:
        raise ValueError("batch backend calls exceed policy")
    if evaluations > policy.maximum_artifact_evaluations:
        raise ValueError("batch artifact evaluations exceed policy")
    if comparison_values > policy.maximum_comparison_values:
        raise ValueError("batch comparison values exceed policy")
    return schedules, backend_calls, evaluations, comparison_values


def _precommitment_from_validated(
    *,
    inventory: ControlScoringInventory,
    ordered: tuple[tuple[str, str, Path], ...],
    producer_config: EmbeddingProducerConfig,
    provenance: Mapping[str, str],
    policy: BatchInvariancePolicy,
    schedules: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...],
    input_bytes: int,
    provenance_bytes: int,
    runtime_library_policy_sha256: str,
    worker_execution_policy_sha256: str,
    worker_environment_identity_sha256: str,
    prior_attempt_ledger_sha256: str,
    candidate_attempt_token: str,
    precommitment_sequence: int,
) -> BatchInvariancePrecommitment:
    return BatchInvariancePrecommitment(
        inventory_sha256=inventory.inventory_sha256,
        producer_config_sha256=producer_config.config_sha256,
        policy_sha256=policy.policy_sha256,
        schedule_manifest_sha256=content_sha256(
            _schedule_manifest(ordered, producer_config.batch_size, schedules)
        ),
        backend_identity_sha256=content_sha256(
            producer_config.backend.to_dict()
        ),
        runtime_library_policy_sha256=runtime_library_policy_sha256,
        worker_execution_policy_sha256=worker_execution_policy_sha256,
        worker_environment_identity_sha256=(
            worker_environment_identity_sha256
        ),
        artifact_bindings=tuple(
            (token, content, path.stat().st_size)
            for content, token, path in ordered
        ),
        provenance_sha256=tuple(sorted(provenance.items())),
        input_bytes_hashed=input_bytes,
        provenance_bytes_hashed=provenance_bytes,
        prior_attempt_ledger_sha256=prior_attempt_ledger_sha256,
        candidate_attempt_token=candidate_attempt_token,
        precommitment_sequence=precommitment_sequence,
    )


def verify_batch_invariance_receipt_external_anchors(
    receipt: BatchInvarianceReceipt,
    *,
    expected_precommitment_sha256: str,
    expected_receipt_sha256: str,
) -> None:
    """Reject structurally valid receipts that differ from external anchors."""

    _sha256(expected_precommitment_sha256, "expected_precommitment_sha256")
    _sha256(expected_receipt_sha256, "expected_receipt_sha256")
    if receipt.precommitment_sha256 != expected_precommitment_sha256:
        raise ValueError("batch receipt precommitment differs from external anchor")
    if receipt.receipt_sha256 != expected_receipt_sha256:
        raise ValueError("batch receipt differs from external final anchor")


def _chunks(values: tuple[int, ...], size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(values[index:index + size] for index in range(0, len(values), size))


def _infer(
    backend: EmbeddingBatchBackend,
    paths: tuple[Path, ...],
    dimension: int,
) -> tuple[bytes, ...]:
    backend.synchronize()
    rows = backend.infer_batch(paths)
    backend.synchronize()
    if not isinstance(rows, Sequence) or len(rows) != len(paths):
        raise ValueError("batch backend output row count differs")
    encoded: list[bytes] = []
    for row in rows:
        if not isinstance(row, Sequence) or len(row) != dimension:
            raise ValueError("batch backend output dimension differs")
        values: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("batch backend output value type differs")
            resolved = float(value)
            if not math.isfinite(resolved):
                raise ValueError("batch backend output is non-finite")
            values.append(resolved)
        encoded.append(struct.pack(f"<{dimension}f", *values))
    return tuple(encoded)


def _normalized_bytes(raw: bytes, epsilon: float) -> bytes:
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm <= epsilon:
        raise ValueError("batch embedding norm is too small")
    return struct.pack(f"<{len(values)}f", *(value / norm for value in values))


def _compare_vectors(
    reference_raw: bytes,
    candidate_raw: bytes,
    reference_norm: bytes,
    candidate_norm: bytes,
    policy: BatchInvariancePolicy,
    accumulator: _Accumulator,
    content: str,
    scenario: str,
) -> tuple[float, float, float]:
    dimension = len(reference_raw) // 4
    raw_squared: list[float] = []
    normalized_squared: list[float] = []
    dot: list[float] = []
    reference_norm_values: list[float] = []
    candidate_norm_values: list[float] = []
    reference_raw_squares: list[float] = []
    candidate_raw_squares: list[float] = []
    occurrence_violated = False
    for coordinate in range(dimension):
        offset = coordinate * 4
        reference = struct.unpack_from("<f", reference_raw, offset)[0]
        candidate = struct.unpack_from("<f", candidate_raw, offset)[0]
        absolute = abs(reference - candidate)
        relative = absolute / max(
            abs(reference), abs(candidate), policy.relative_floor
        )
        allowed = policy.absolute_tolerance + policy.relative_tolerance * max(
            abs(reference), abs(candidate)
        )
        if absolute > allowed:
            accumulator.violated_values += 1
            occurrence_violated = True
        if absolute > accumulator.maximum_absolute:
            accumulator.maximum_absolute = absolute
            accumulator.worst_content = content
            accumulator.worst_coordinate = coordinate
            accumulator.worst_scenario = scenario
        accumulator.maximum_relative = max(accumulator.maximum_relative, relative)
        accumulator.maximum_ulp = max(
            accumulator.maximum_ulp,
            _ulp_distance(
                reference_raw[offset:offset + 4],
                candidate_raw[offset:offset + 4],
            ),
        )
        accumulator.add_absolute(absolute)
        raw_squared.append((reference - candidate) ** 2)
        reference_raw_squares.append(reference * reference)
        candidate_raw_squares.append(candidate * candidate)
        ref_n = struct.unpack_from("<f", reference_norm, offset)[0]
        can_n = struct.unpack_from("<f", candidate_norm, offset)[0]
        normalized_squared.append((ref_n - can_n) ** 2)
        dot.append(ref_n * can_n)
        reference_norm_values.append(ref_n * ref_n)
        candidate_norm_values.append(can_n * can_n)
    if occurrence_violated:
        accumulator.violated_occurrences += 1
    raw_l2 = math.sqrt(math.fsum(raw_squared))
    raw_norm_drift = abs(
        math.sqrt(math.fsum(reference_raw_squares))
        - math.sqrt(math.fsum(candidate_raw_squares))
    )
    normalized_l2 = math.sqrt(math.fsum(normalized_squared))
    cosine = math.fsum(dot) / math.sqrt(
        math.fsum(reference_norm_values) * math.fsum(candidate_norm_values)
    )
    cosine_drift = 1.0 - min(1.0, max(-1.0, cosine))
    accumulator.maximum_raw_l2 = max(accumulator.maximum_raw_l2, raw_l2)
    accumulator.maximum_raw_norm = max(
        accumulator.maximum_raw_norm, raw_norm_drift
    )
    accumulator.maximum_normalized_l2 = max(
        accumulator.maximum_normalized_l2, normalized_l2
    )
    accumulator.maximum_cosine = max(accumulator.maximum_cosine, cosine_drift)
    return raw_l2, normalized_l2, cosine_drift


def _hard_failures(
    summary: BatchInvarianceSummary,
    policy: BatchInvariancePolicy,
) -> tuple[str, ...]:
    failures: list[str] = []
    if summary.violated_values:
        failures.append("ELEMENTWISE_ABSOLUTE_RELATIVE_TOLERANCE")
    checks = (
        (summary.maximum_raw_l2_drift > policy.maximum_raw_l2_drift, "RAW_L2_DRIFT"),
        (summary.maximum_raw_norm_drift > policy.maximum_raw_norm_drift, "RAW_NORM_DRIFT"),
        (
            summary.maximum_normalized_l2_drift
            > policy.maximum_normalized_l2_drift,
            "NORMALIZED_L2_DRIFT",
        ),
        (summary.maximum_cosine_drift > policy.maximum_cosine_drift, "COSINE_DRIFT"),
        (summary.repeated_digest_mismatches > 0, "REPEATED_COMPOSITION_NOT_EXACT"),
    )
    failures.extend(code for failed, code in checks if failed)
    return tuple(sorted(failures))


def _validate_backend_lineage(
    backend: EmbeddingBatchBackend,
    config: EmbeddingProducerConfig,
) -> None:
    if backend.identity != config.backend:
        raise ValueError("batch backend identity differs from producer config")
    if backend.preprocessing_semantics_sha256 != config.preprocessing_semantics_sha256:
        raise ValueError("batch preprocessing semantics differ")
    if backend.model_sha256 != config.model_sha256:
        raise ValueError("batch backend model differs")


def _validate_inputs(
    inventory: ControlScoringInventory,
    artifact_paths: Mapping[str, Path],
) -> tuple[tuple[str, str, Path], ...]:
    if set(artifact_paths) != {item.artifact_token for item in inventory.entries}:
        raise ValueError("batch artifact path tokens differ from inventory")
    ordered: list[tuple[str, str, Path]] = []
    seen_content: set[str] = set()
    for item in inventory.entries:
        if item.content_sha256 in seen_content:
            raise ValueError("batch gate requires unique artifact content")
        seen_content.add(item.content_sha256)
        path = artifact_paths[item.artifact_token]
        if path.is_symlink():
            raise ValueError("batch artifact must not be a symlink")
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        if not resolved.is_file() or before.st_size != item.byte_size:
            raise ValueError("batch artifact file metadata differs")
        if sha256_file(resolved) != item.content_sha256:
            raise ValueError("batch artifact content differs")
        after = resolved.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino
        ):
            raise RuntimeError("batch artifact changed while hashing")
        ordered.append((item.content_sha256, item.artifact_token, resolved))
    return tuple(sorted(ordered))


def _validate_provenance(
    paths: Mapping[str, Path],
    config: EmbeddingProducerConfig,
) -> dict[str, str]:
    expected = {
        "model": config.model_sha256,
        "model_lineage": config.model_lineage_sha256,
        "preprocessing": config.preprocessing_sha256,
        "dependency_lock": config.dependency_lock_sha256,
    }
    if set(paths) != set(expected):
        raise ValueError("batch provenance path names differ")
    result: dict[str, str] = {}
    for name in sorted(expected):
        path = paths[name]
        if path.is_symlink():
            raise ValueError("batch provenance must not be a symlink")
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        digest = sha256_file(resolved)
        after = resolved.stat()
        if not resolved.is_file() or before.st_size <= 0:
            raise ValueError("batch provenance must be a nonempty file")
        if digest != expected[name]:
            raise ValueError(f"batch {name} provenance differs")
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino
        ):
            raise RuntimeError("batch provenance changed while hashing")
        result[name] = digest
    return result


def _ulp_distance(first: bytes, second: bytes) -> int:
    left = struct.unpack("<I", first)[0]
    right = struct.unpack("<I", second)[0]
    left_ordered = 0x80000000 - left if left & 0x80000000 else left + 0x80000000
    right_ordered = 0x80000000 - right if right & 0x80000000 else right + 0x80000000
    return abs(left_ordered - right_ordered)


def _exact_keys(payload: Any, expected: set[str], name: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{name} keys mismatch")


def _sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")


def _finite_nonnegative(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (
        not math.isfinite(value) or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _finite_positive(value: Any, name: str) -> None:
    _finite_nonnegative(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
