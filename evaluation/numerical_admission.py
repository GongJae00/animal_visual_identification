"""Label-blind numerical admission for canonical embedding caches."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

from evaluation.control_scoring import EmbeddingCacheManifest
from operations.embedding_producer import EmbeddingProducerConfig
from foundation.provenance import content_sha256


class NumericalAdmissionDecision(StrEnum):
    PASS = "NUMERICAL_PASS_ON_FROZEN_WORKLOAD"
    FAIL = "NUMERICAL_FAIL"


@dataclass(frozen=True, slots=True)
class NumericalDriftPolicy:
    absolute_tolerance: float
    relative_tolerance: float
    relative_floor: float
    maximum_l2_drift: float
    maximum_cosine_drift: float
    maximum_vectors: int = 100_000
    maximum_vector_dimension: int = 65_536
    maximum_total_bytes_read: int = 17_179_869_184
    schema_version: str = "cvi.numerical_drift_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.numerical_drift_policy.v1":
            raise ValueError("unsupported numerical drift policy schema")
        for name in (
            "absolute_tolerance",
            "relative_tolerance",
            "maximum_l2_drift",
            "maximum_cosine_drift",
        ):
            _require_finite_nonnegative(getattr(self, name), name)
        _require_finite_positive(self.relative_floor, "relative_floor")
        for name in (
            "maximum_vectors",
            "maximum_vector_dimension",
            "maximum_total_bytes_read",
        ):
            _require_positive_int(getattr(self, name), name)

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "schema_version": self.schema_version,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "relative_floor": self.relative_floor,
            "maximum_l2_drift": self.maximum_l2_drift,
            "maximum_cosine_drift": self.maximum_cosine_drift,
            "maximum_vectors": self.maximum_vectors,
            "maximum_vector_dimension": self.maximum_vector_dimension,
            "maximum_total_bytes_read": self.maximum_total_bytes_read,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NumericalDriftPolicy:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "absolute_tolerance",
                "relative_tolerance",
                "relative_floor",
                "maximum_l2_drift",
                "maximum_cosine_drift",
                "maximum_vectors",
                "maximum_vector_dimension",
                "maximum_total_bytes_read",
            },
            "numerical drift policy",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class NumericalDriftSummary:
    vectors: int
    values: int
    violated_vectors: int
    violated_values: int
    maximum_absolute_error: float
    mean_absolute_error: float
    maximum_relative_error: float
    maximum_ulp_distance: int
    maximum_l2_drift: float
    maximum_cosine_drift: float
    worst_artifact_content_sha256: str | None
    worst_coordinate: int | None
    bytes_read: int
    schema_version: str = "cvi.numerical_drift_summary.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.numerical_drift_summary.v1":
            raise ValueError("unsupported numerical drift summary schema")
        for name in (
            "vectors",
            "values",
            "violated_vectors",
            "violated_values",
            "maximum_ulp_distance",
            "bytes_read",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        if self.vectors == 0 or self.values == 0:
            raise ValueError("numerical drift summary must not be empty")
        if self.violated_vectors > self.vectors:
            raise ValueError("violated vector count exceeds vector count")
        if self.violated_values > self.values:
            raise ValueError("violated value count exceeds value count")
        for name in (
            "maximum_absolute_error",
            "mean_absolute_error",
            "maximum_relative_error",
            "maximum_l2_drift",
            "maximum_cosine_drift",
        ):
            _require_finite_nonnegative(getattr(self, name), name)
        if (self.worst_artifact_content_sha256 is None) != (
            self.worst_coordinate is None
        ):
            raise ValueError("worst drift artifact and coordinate must pair")
        if self.worst_artifact_content_sha256 is not None:
            _validate_sha256(
                self.worst_artifact_content_sha256,
                "worst_artifact_content_sha256",
            )
            _require_nonnegative_int(
                self.worst_coordinate,
                "worst_coordinate",
            )

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "schema_version": self.schema_version,
            "vectors": self.vectors,
            "values": self.values,
            "violated_vectors": self.violated_vectors,
            "violated_values": self.violated_values,
            "maximum_absolute_error": self.maximum_absolute_error,
            "mean_absolute_error": self.mean_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "maximum_ulp_distance": self.maximum_ulp_distance,
            "maximum_l2_drift": self.maximum_l2_drift,
            "maximum_cosine_drift": self.maximum_cosine_drift,
            "worst_artifact_content_sha256": (
                self.worst_artifact_content_sha256
            ),
            "worst_coordinate": self.worst_coordinate,
            "bytes_read": self.bytes_read,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NumericalDriftSummary:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "vectors",
                "values",
                "violated_vectors",
                "violated_values",
                "maximum_absolute_error",
                "mean_absolute_error",
                "maximum_relative_error",
                "maximum_ulp_distance",
                "maximum_l2_drift",
                "maximum_cosine_drift",
                "worst_artifact_content_sha256",
                "worst_coordinate",
                "bytes_read",
            },
            "numerical drift summary",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class NumericalAdmissionReceipt:
    reference_manifest_sha256: str
    candidate_manifest_sha256: str
    reference_config_sha256: str
    candidate_config_sha256: str
    comparable_semantics_sha256: str
    scoring_inventory_sha256: str
    policy_sha256: str
    summary: NumericalDriftSummary
    hard_failures: tuple[str, ...]
    decision: NumericalAdmissionDecision
    interpretation: str = (
        "NUMERICAL_ADMISSION_ONLY_NOT_OPTIMIZATION_PROMOTION"
    )
    schema_version: str = "cvi.numerical_admission_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.numerical_admission_receipt.v1":
            raise ValueError("unsupported numerical admission receipt schema")
        for name in (
            "reference_manifest_sha256",
            "candidate_manifest_sha256",
            "reference_config_sha256",
            "candidate_config_sha256",
            "comparable_semantics_sha256",
            "scoring_inventory_sha256",
            "policy_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if len(self.hard_failures) != len(set(self.hard_failures)):
            raise ValueError("numerical hard failures must be unique")
        if tuple(sorted(self.hard_failures)) != self.hard_failures:
            raise ValueError("numerical hard failures must be sorted")
        if any(not item.strip() for item in self.hard_failures):
            raise ValueError("numerical hard failures must be non-empty")
        expected = (
            NumericalAdmissionDecision.PASS
            if not self.hard_failures
            else NumericalAdmissionDecision.FAIL
        )
        if self.decision is not expected:
            raise ValueError("numerical decision and failures differ")
        if (
            self.interpretation
            != "NUMERICAL_ADMISSION_ONLY_NOT_OPTIMIZATION_PROMOTION"
        ):
            raise ValueError("numerical interpretation is fixed")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "reference_config_sha256": self.reference_config_sha256,
            "candidate_config_sha256": self.candidate_config_sha256,
            "comparable_semantics_sha256": (
                self.comparable_semantics_sha256
            ),
            "scoring_inventory_sha256": self.scoring_inventory_sha256,
            "policy_sha256": self.policy_sha256,
            "summary": self.summary.to_dict(),
            "hard_failures": list(self.hard_failures),
            "decision": self.decision.value,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NumericalAdmissionReceipt:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "reference_manifest_sha256",
                "candidate_manifest_sha256",
                "reference_config_sha256",
                "candidate_config_sha256",
                "comparable_semantics_sha256",
                "scoring_inventory_sha256",
                "policy_sha256",
                "summary",
                "hard_failures",
                "decision",
                "interpretation",
            },
            "numerical admission receipt",
        )
        if not isinstance(payload["summary"], dict):
            raise TypeError("numerical summary must be an object")
        failures = payload["hard_failures"]
        if not isinstance(failures, list):
            raise TypeError("numerical hard failures must be a list")
        return cls(
            schema_version=payload["schema_version"],
            reference_manifest_sha256=payload[
                "reference_manifest_sha256"
            ],
            candidate_manifest_sha256=payload[
                "candidate_manifest_sha256"
            ],
            reference_config_sha256=payload["reference_config_sha256"],
            candidate_config_sha256=payload["candidate_config_sha256"],
            comparable_semantics_sha256=payload[
                "comparable_semantics_sha256"
            ],
            scoring_inventory_sha256=payload["scoring_inventory_sha256"],
            policy_sha256=payload["policy_sha256"],
            summary=NumericalDriftSummary.from_dict(payload["summary"]),
            hard_failures=tuple(failures),
            decision=NumericalAdmissionDecision(payload["decision"]),
            interpretation=payload["interpretation"],
        )


def compare_embedding_caches(
    *,
    reference_manifest: EmbeddingCacheManifest,
    candidate_manifest: EmbeddingCacheManifest,
    reference_config: EmbeddingProducerConfig,
    candidate_config: EmbeddingProducerConfig,
    reference_root: Path,
    candidate_root: Path,
    policy: NumericalDriftPolicy,
) -> NumericalAdmissionReceipt:
    """Compare canonical float32 caches without labels or pair outcomes."""

    semantics = _validate_comparison_lineage(
        reference_manifest,
        candidate_manifest,
        reference_config,
        candidate_config,
    )
    vector_dimension = reference_manifest.vector_dimension
    if vector_dimension > policy.maximum_vector_dimension:
        raise ValueError("numerical comparison dimension exceeds policy")
    content_bindings = {
        binding.artifact_content_sha256
        for binding in reference_manifest.bindings
    }
    if len(content_bindings) > policy.maximum_vectors:
        raise ValueError("numerical comparison vectors exceed policy")
    bytes_read = len(content_bindings) * vector_dimension * 4 * 2
    if bytes_read > policy.maximum_total_bytes_read:
        raise ValueError("numerical comparison bytes exceed policy")
    reference_files = _closed_cache_files(
        reference_root,
        reference_manifest,
    )
    candidate_files = _closed_cache_files(
        candidate_root,
        candidate_manifest,
    )
    reference_by_content = _entries_by_content(
        reference_manifest,
        reference_files,
    )
    candidate_by_content = _entries_by_content(
        candidate_manifest,
        candidate_files,
    )
    if set(reference_by_content) != set(candidate_by_content):
        raise ValueError("numerical cache contents differ")

    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_ulp = 0
    maximum_l2 = 0.0
    maximum_cosine = 0.0
    violated_values = 0
    violated_vectors = 0
    worst_content: str | None = None
    worst_coordinate: int | None = None
    absolute_sum = _CompensatedSum()
    for artifact_content_sha256 in sorted(reference_by_content):
        reference_bytes = _read_verified_vector(
            reference_by_content[artifact_content_sha256],
            vector_dimension,
            reference_manifest.normalization_tolerance,
        )
        candidate_bytes = _read_verified_vector(
            candidate_by_content[artifact_content_sha256],
            vector_dimension,
            candidate_manifest.normalization_tolerance,
        )
        squared_differences = _CompensatedSum()
        dot_products = _CompensatedSum()
        reference_squares = _CompensatedSum()
        candidate_squares = _CompensatedSum()
        vector_violated = False
        for coordinate in range(vector_dimension):
            offset = coordinate * 4
            reference = struct.unpack_from(
                "<f",
                reference_bytes,
                offset,
            )[0]
            candidate = struct.unpack_from(
                "<f",
                candidate_bytes,
                offset,
            )[0]
            absolute = abs(reference - candidate)
            relative = absolute / max(
                abs(reference),
                abs(candidate),
                policy.relative_floor,
            )
            ulp = _float32_ulp_distance(
                reference_bytes[coordinate * 4 : coordinate * 4 + 4],
                candidate_bytes[coordinate * 4 : coordinate * 4 + 4],
            )
            allowed = policy.absolute_tolerance + (
                policy.relative_tolerance
                * max(abs(reference), abs(candidate))
            )
            if absolute > allowed:
                violated_values += 1
                vector_violated = True
            if absolute > maximum_absolute:
                maximum_absolute = absolute
                worst_content = artifact_content_sha256
                worst_coordinate = coordinate
            maximum_relative = max(maximum_relative, relative)
            maximum_ulp = max(maximum_ulp, ulp)
            absolute_sum.add(absolute)
            difference = reference - candidate
            squared_differences.add(difference * difference)
            dot_products.add(reference * candidate)
            reference_squares.add(reference * reference)
            candidate_squares.add(candidate * candidate)
        if vector_violated:
            violated_vectors += 1
        l2_drift = math.sqrt(squared_differences.value)
        reference_norm = math.sqrt(reference_squares.value)
        candidate_norm = math.sqrt(candidate_squares.value)
        cosine = dot_products.value / (
            reference_norm * candidate_norm
        )
        cosine_drift = 1.0 - min(1.0, max(-1.0, cosine))
        maximum_l2 = max(maximum_l2, l2_drift)
        maximum_cosine = max(maximum_cosine, cosine_drift)

    vector_count = len(reference_by_content)
    value_count = vector_count * vector_dimension
    failures: list[str] = []
    if violated_values:
        failures.append("ELEMENTWISE_ABSOLUTE_RELATIVE_TOLERANCE")
    if maximum_l2 > policy.maximum_l2_drift:
        failures.append("EMBEDDING_L2_DRIFT")
    if maximum_cosine > policy.maximum_cosine_drift:
        failures.append("EMBEDDING_COSINE_DRIFT")
    summary = NumericalDriftSummary(
        vectors=vector_count,
        values=value_count,
        violated_vectors=violated_vectors,
        violated_values=violated_values,
        maximum_absolute_error=maximum_absolute,
        mean_absolute_error=absolute_sum.value / value_count,
        maximum_relative_error=maximum_relative,
        maximum_ulp_distance=maximum_ulp,
        maximum_l2_drift=maximum_l2,
        maximum_cosine_drift=maximum_cosine,
        worst_artifact_content_sha256=worst_content,
        worst_coordinate=worst_coordinate,
        bytes_read=bytes_read,
    )
    hard_failures = tuple(sorted(failures))
    return NumericalAdmissionReceipt(
        reference_manifest_sha256=reference_manifest.manifest_sha256,
        candidate_manifest_sha256=candidate_manifest.manifest_sha256,
        reference_config_sha256=reference_config.config_sha256,
        candidate_config_sha256=candidate_config.config_sha256,
        comparable_semantics_sha256=content_sha256(semantics),
        scoring_inventory_sha256=(
            reference_manifest.scoring_inventory_sha256
        ),
        policy_sha256=policy.policy_sha256,
        summary=summary,
        hard_failures=hard_failures,
        decision=(
            NumericalAdmissionDecision.PASS
            if not hard_failures
            else NumericalAdmissionDecision.FAIL
        ),
    )


def _validate_comparison_lineage(
    reference_manifest: EmbeddingCacheManifest,
    candidate_manifest: EmbeddingCacheManifest,
    reference_config: EmbeddingProducerConfig,
    candidate_config: EmbeddingProducerConfig,
) -> dict[str, Any]:
    if reference_config.backend == candidate_config.backend:
        raise ValueError("numerical comparison backends must differ")
    if reference_manifest.inference_config_sha256 != (
        reference_config.config_sha256
    ):
        raise ValueError("reference manifest/config binding differs")
    if candidate_manifest.inference_config_sha256 != (
        candidate_config.config_sha256
    ):
        raise ValueError("candidate manifest/config binding differs")
    reference_semantics = _comparable_semantics(reference_config)
    candidate_semantics = _comparable_semantics(candidate_config)
    if reference_semantics != candidate_semantics:
        raise ValueError("numerical comparison semantics differ")
    if reference_manifest.scoring_inventory_sha256 != (
        candidate_manifest.scoring_inventory_sha256
    ):
        raise ValueError("numerical scoring inventories differ")
    reference_bindings = tuple(
        (binding.artifact_token, binding.artifact_content_sha256)
        for binding in reference_manifest.bindings
    )
    candidate_bindings = tuple(
        (binding.artifact_token, binding.artifact_content_sha256)
        for binding in candidate_manifest.bindings
    )
    if reference_bindings != candidate_bindings:
        raise ValueError("numerical artifact bindings differ")
    for manifest, config, name in (
        (reference_manifest, reference_config, "reference"),
        (candidate_manifest, candidate_config, "candidate"),
    ):
        if manifest.model_sha256 != config.model_sha256:
            raise ValueError(f"{name} model binding differs")
        if manifest.dependency_lock_sha256 != config.dependency_lock_sha256:
            raise ValueError(f"{name} dependency lock binding differs")
        if manifest.code_revision != config.code_revision:
            raise ValueError(f"{name} code revision binding differs")
        if manifest.precision != config.backend.precision:
            raise ValueError(f"{name} precision binding differs")
        if manifest.vector_dimension != config.vector_dimension:
            raise ValueError(f"{name} vector dimension binding differs")
        if manifest.vector_format != config.output_vector_format:
            raise ValueError(f"{name} vector format binding differs")
    return reference_semantics


def _comparable_semantics(config: EmbeddingProducerConfig) -> dict[str, Any]:
    payload = config.to_dict()
    payload.pop("backend")
    payload.pop("dependency_lock_sha256")
    return payload


def _closed_cache_files(
    root: Path,
    manifest: EmbeddingCacheManifest,
) -> dict[str, Path]:
    if root.is_symlink():
        raise ValueError("embedding cache root must not be a symlink")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("embedding cache root must be a directory")
    expected = {entry.relative_path for entry in manifest.entries}
    actual: set[str] = set()
    paths: dict[str, Path] = {}
    for child in resolved.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ValueError("embedding cache must contain regular files only")
        actual.add(child.name)
        paths[child.name] = child
    if actual != expected:
        raise ValueError("embedding cache directory is not closed")
    return paths


def _entries_by_content(
    manifest: EmbeddingCacheManifest,
    paths: dict[str, Path],
) -> dict[str, tuple[Path, str, int]]:
    entries_by_key = {entry.cache_key: entry for entry in manifest.entries}
    by_content: dict[str, tuple[Path, str, int]] = {}
    for binding in manifest.bindings:
        entry = entries_by_key[binding.cache_key]
        value = (
            paths[entry.relative_path],
            entry.content_sha256,
            entry.byte_size,
        )
        previous = by_content.setdefault(
            binding.artifact_content_sha256,
            value,
        )
        if previous != value:
            raise ValueError("content alias maps to multiple cache entries")
    return by_content


def _read_verified_vector(
    entry: tuple[Path, str, int],
    dimension: int,
    normalization_tolerance: float,
) -> bytes:
    path, expected_sha256, expected_bytes = entry
    before = path.stat()
    if before.st_size != expected_bytes or expected_bytes != dimension * 4:
        raise ValueError("embedding cache vector byte size differs")
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise ValueError("embedding cache vector changed while reading")
    if sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("embedding cache vector hash differs")
    square_sum = _CompensatedSum()
    values = 0
    for (value,) in struct.iter_unpack("<f", payload):
        if not math.isfinite(value):
            raise ValueError("embedding cache vector is malformed")
        square_sum.add(value * value)
        values += 1
    if values != dimension:
        raise ValueError("embedding cache vector is malformed")
    norm = math.sqrt(square_sum.value)
    if abs(norm - 1.0) > normalization_tolerance:
        raise ValueError("embedding cache vector is not normalized")
    return payload


def _float32_ulp_distance(reference: bytes, candidate: bytes) -> int:
    reference_bits = struct.unpack("<I", reference)[0]
    candidate_bits = struct.unpack("<I", candidate)[0]

    def ordered(bits: int) -> int:
        if bits & 0x7FFFFFFF == 0:
            return 0x80000000
        if bits & 0x80000000:
            return (~bits) & 0xFFFFFFFF
        return bits ^ 0x80000000

    return abs(ordered(reference_bits) - ordered(candidate_bits))


class _CompensatedSum:
    __slots__ = ("_compensation", "_sum")

    def __init__(self) -> None:
        self._sum = 0.0
        self._compensation = 0.0

    def add(self, value: float) -> None:
        total = self._sum + value
        if abs(self._sum) >= abs(value):
            self._compensation += (self._sum - total) + value
        else:
            self._compensation += (value - total) + self._sum
        self._sum = total

    @property
    def value(self) -> float:
        return self._sum + self._compensation


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite_nonnegative(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _require_finite_positive(value: float, name: str) -> None:
    _require_finite_nonnegative(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
