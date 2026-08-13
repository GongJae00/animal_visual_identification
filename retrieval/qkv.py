"""Explicit query, gallery-key, and gallery-value retrieval contracts.

These roles describe exact retrieval data flow. They do not implement attention,
learned projections, softmax weighting, or value mixing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

SCORER_ALGORITHM = "exact_available_intersection_weighted_cosine.v1"
FULL128_CHANNEL = "Full128"


class IdentityEvidenceKind(str, Enum):
    """Identity namespace asserted by enrollment evidence."""

    REGISTERED = "REGISTERED"
    PROVISIONAL_GENID = "PROVISIONAL_GENID"


class EnrollmentRank(str, Enum):
    """Deterministic enrollment panel represented by a template."""

    K1 = "K1"
    K3 = "K3"
    K5 = "K5"


@dataclass(frozen=True, slots=True)
class QueryExclusions:
    """Content-bound template exclusions applied before identity aggregation."""

    template_ids: frozenset[str] = field(default_factory=frozenset)
    content_sha256s: frozenset[str] = field(default_factory=frozenset)
    duplicate_group_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for name in ("template_ids", "content_sha256s", "duplicate_group_ids"):
            values = getattr(self, name)
            if not isinstance(values, frozenset):
                try:
                    values = frozenset(values)
                except TypeError as exc:
                    raise TypeError(f"query exclusion {name} must be a string set") from exc
                object.__setattr__(self, name, values)
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"query exclusion {name} must contain non-empty text")
        for name in ("template_ids", "content_sha256s"):
            if any(not _is_sha256(value) for value in getattr(self, name)):
                raise ValueError(f"query exclusion {name} must contain SHA-256 digests")


@dataclass(frozen=True, slots=True)
class EvidenceChannelSpec:
    name: str
    dimension: int
    optional: bool
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("evidence channel name must be non-empty")
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension <= 0
        ):
            raise ValueError("evidence channel dimension must be positive")
        if not isinstance(self.optional, bool):
            raise TypeError("evidence channel optional flag must be boolean")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight < 0.0
        ):
            raise ValueError("evidence channel weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    vectors: Mapping[str, np.ndarray]
    availability: Mapping[str, bool]
    exclusions: QueryExclusions = field(default_factory=QueryExclusions)

    def __post_init__(self) -> None:
        if not isinstance(self.exclusions, QueryExclusions):
            raise TypeError("query exclusions must be a QueryExclusions value")
        vectors, availability = _snapshot_vector_set(self.vectors, self.availability)
        _validate_vector_set(
            vectors, availability, role="query", require_unit=False
        )
        _freeze_vector_set(
            self, vectors, availability, normalize=True
        )


@dataclass(frozen=True, slots=True)
class GalleryKey:
    template_row: int
    vectors: Mapping[str, np.ndarray]
    availability: Mapping[str, bool]

    def __post_init__(self) -> None:
        if (
            isinstance(self.template_row, bool)
            or not isinstance(self.template_row, int)
            or self.template_row < 0
        ):
            raise ValueError("gallery key template row must be non-negative")
        vectors, availability = _snapshot_vector_set(self.vectors, self.availability)
        _validate_vector_set(
            vectors, availability, role="gallery key", require_unit=True
        )
        _freeze_vector_set(
            self, vectors, availability, normalize=False
        )


@dataclass(frozen=True, slots=True)
class GalleryValue:
    template_row: int
    registered_identity_id: str
    template_id: str
    content_sha256: str
    idempotency_key: str
    template_schema: str
    breed: str
    metadata: dict[str, Any]
    identity_evidence_kind: IdentityEvidenceKind = IdentityEvidenceKind.REGISTERED
    enrollment_rank: EnrollmentRank | None = None
    enrollment_view: str | None = None
    duplicate_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.template_row, bool)
            or not isinstance(self.template_row, int)
            or self.template_row < 0
        ):
            raise ValueError("gallery value template row must be non-negative")
        for name in (
            "registered_identity_id",
            "template_id",
            "content_sha256",
            "idempotency_key",
            "template_schema",
            "breed",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"gallery value {name} must be non-empty text")
        if not isinstance(self.metadata, dict):
            raise TypeError("gallery value metadata must be an object")
        if not isinstance(self.identity_evidence_kind, IdentityEvidenceKind):
            raise TypeError("gallery value identity evidence kind is invalid")
        if self.enrollment_rank is not None and not isinstance(
            self.enrollment_rank, EnrollmentRank
        ):
            raise TypeError("gallery value enrollment rank is invalid")
        if (self.enrollment_rank is None) != (self.enrollment_view is None):
            raise ValueError("gallery enrollment rank and view must be provided together")
        if self.enrollment_view is not None and (
            not isinstance(self.enrollment_view, str) or not self.enrollment_view
        ):
            raise ValueError("gallery enrollment view must be non-empty text")
        if (
            not isinstance(self.duplicate_group_ids, tuple)
            or any(
                not isinstance(value, str) or not value
                for value in self.duplicate_group_ids
            )
            or tuple(sorted(set(self.duplicate_group_ids))) != self.duplicate_group_ids
        ):
            raise ValueError("gallery duplicate group IDs must be sorted unique text")


@dataclass(frozen=True, slots=True)
class QueryKeyScore:
    similarity: float
    evidence: dict[str, float]
    evidence_availability: dict[str, bool]


@dataclass(frozen=True, slots=True)
class ScoredGalleryValue:
    value: GalleryValue
    query_key_score: QueryKeyScore
    template_availability: dict[str, bool]


class IdentityMatchAccumulator:
    """Retain only the best template value observed for each identity."""

    def __init__(self) -> None:
        self._best_by_identity: dict[str, ScoredGalleryValue] = {}

    def add(self, candidate: ScoredGalleryValue) -> None:
        if not isinstance(candidate, ScoredGalleryValue):
            raise TypeError("identity match candidate must be a scored gallery value")
        identity_id = candidate.value.registered_identity_id
        current = self._best_by_identity.get(identity_id)
        if current is None or _is_better_template(candidate, current):
            self._best_by_identity[identity_id] = candidate

    def ranked(self, *, top_k: int) -> list[ScoredGalleryValue]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return sorted(
            self._best_by_identity.values(),
            key=lambda item: (
                -item.query_key_score.similarity,
                item.value.registered_identity_id,
            ),
        )[:top_k]


class AvailableIntersectionScorer:
    """Exact weighted cosine over channels available in both Q and K."""

    def __init__(self, channels: tuple[EvidenceChannelSpec, ...]) -> None:
        if not channels or any(not isinstance(item, EvidenceChannelSpec) for item in channels):
            raise ValueError("QK scorer requires evidence channel specifications")
        if len({item.name for item in channels}) != len(channels):
            raise ValueError("QK scorer channel names must be unique")
        if sum(item.weight for item in channels) <= 0.0:
            raise ValueError("QK scorer weights must have a positive sum")
        if sum(item.weight for item in channels if not item.optional) <= 0.0:
            raise ValueError("required QK channels must retain positive weight")
        self.channels = channels

    @property
    def scorer_hash(self) -> str:
        total = sum(channel.weight for channel in self.channels)
        payload = {
            "algorithm": SCORER_ALGORITHM,
            "channels": [
                {
                    "name": channel.name,
                    "dimension": channel.dimension,
                    "optional": channel.optional,
                }
                for channel in self.channels
            ],
            "weights": [channel.weight / total for channel in self.channels],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def score(self, query: RetrievalQuery, key: GalleryKey) -> QueryKeyScore:
        expected_names = {channel.name for channel in self.channels}
        if set(query.availability) != expected_names:
            raise ValueError("query availability differs from QK scorer channels")
        if set(key.availability) != expected_names:
            raise ValueError("gallery-key availability differs from QK scorer channels")
        intersection = [
            channel
            for channel in self.channels
            if query.availability[channel.name] and key.availability[channel.name]
        ]
        for channel in intersection:
            if query.vectors[channel.name].shape != (channel.dimension,):
                raise ValueError(f"query channel {channel.name!r} dimension differs")
            if key.vectors[channel.name].shape != (channel.dimension,):
                raise ValueError(f"gallery-key channel {channel.name!r} dimension differs")
        total_weight = sum(channel.weight for channel in intersection)
        if total_weight <= 0.0:
            raise RuntimeError("no positively weighted QK evidence intersection remains")
        evidence = {
            channel.name: float(
                np.dot(query.vectors[channel.name], key.vectors[channel.name])
            )
            for channel in intersection
        }
        similarity = sum(
            channel.weight * evidence[channel.name] for channel in intersection
        ) / total_weight
        return QueryKeyScore(
            similarity=float(similarity),
            evidence=evidence,
            evidence_availability={
                channel.name: (
                    query.availability[channel.name] and key.availability[channel.name]
                )
                for channel in self.channels
            },
        )


def canonical_channel_weights(
    channel_count: int, values: list[float] | None
) -> np.ndarray:
    if isinstance(channel_count, bool) or not isinstance(channel_count, int) or channel_count <= 0:
        raise ValueError("channel count must be positive")
    raw = values if values is not None else [1.0 / channel_count] * channel_count
    if len(raw) != channel_count:
        raise ValueError("channel weight count must match channel count")
    weights = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("channel weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("channel weights must have a positive sum")
    return (weights / total).astype(np.float32)


def aggregate_identity_matches(
    candidates: list[ScoredGalleryValue], *, top_k: int
) -> list[ScoredGalleryValue]:
    accumulator = IdentityMatchAccumulator()
    for candidate in candidates:
        accumulator.add(candidate)
    return accumulator.ranked(top_k=top_k)


def _is_better_template(
    candidate: ScoredGalleryValue, current: ScoredGalleryValue
) -> bool:
    candidate_score = candidate.query_key_score.similarity
    current_score = current.query_key_score.similarity
    return candidate_score > current_score or (
        candidate_score == current_score
        and candidate.value.template_id < current.value.template_id
    )


def _validate_vector_set(
    vectors: Mapping[str, np.ndarray],
    availability: Mapping[str, bool],
    *,
    role: str,
    require_unit: bool,
) -> None:
    if not isinstance(vectors, Mapping) or not isinstance(availability, Mapping):
        raise TypeError(f"{role} vectors and availability must be objects")
    if any(not isinstance(name, str) or not name for name in availability):
        raise ValueError(f"{role} availability channel names differ")
    if any(not isinstance(value, bool) for value in availability.values()):
        raise TypeError(f"{role} availability values must be boolean")
    available_names = {name for name, available in availability.items() if available}
    if set(vectors) != available_names:
        raise ValueError(f"{role} vectors must exactly match available channels")
    for name, vector in vectors.items():
        if (
            not isinstance(vector, np.ndarray)
            or vector.dtype != np.float32
            or vector.ndim != 1
            or not np.isfinite(vector).all()
        ):
            raise ValueError(f"{role} channel {name!r} must be a finite float32 vector")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError(f"{role} channel {name!r} must have non-zero finite norm")
        if require_unit and not np.isclose(norm, 1.0, atol=1e-5):
            raise ValueError(f"{role} channel {name!r} must be a unit float32 vector")


def _is_sha256(value: str) -> bool:
    if len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _snapshot_vector_set(
    vectors: Mapping[str, np.ndarray], availability: Mapping[str, bool]
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    if not isinstance(vectors, Mapping) or not isinstance(availability, Mapping):
        raise TypeError("vectors and availability must be objects")
    vector_snapshot = {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in vectors.items()
    }
    return vector_snapshot, dict(availability)


def _freeze_vector_set(
    target: RetrievalQuery | GalleryKey,
    vectors: Mapping[str, np.ndarray],
    availability: Mapping[str, bool],
    *,
    normalize: bool,
) -> None:
    frozen_vectors: dict[str, np.ndarray] = {}
    for name, vector in vectors.items():
        value = (
            np.asarray(vector / float(np.linalg.norm(vector)), dtype=np.float32)
            if normalize
            else vector
        )
        payload = value.tobytes(order="C")
        frozen_vectors[name] = np.frombuffer(payload, dtype=np.float32).reshape(value.shape)
    object.__setattr__(target, "vectors", MappingProxyType(frozen_vectors))
    object.__setattr__(target, "availability", MappingProxyType(dict(availability)))


__all__ = [
    "FULL128_CHANNEL",
    "SCORER_ALGORITHM",
    "AvailableIntersectionScorer",
    "EnrollmentRank",
    "EvidenceChannelSpec",
    "GalleryKey",
    "GalleryValue",
    "IdentityEvidenceKind",
    "IdentityMatchAccumulator",
    "QueryExclusions",
    "QueryKeyScore",
    "RetrievalQuery",
    "ScoredGalleryValue",
    "aggregate_identity_matches",
    "canonical_channel_weights",
]
