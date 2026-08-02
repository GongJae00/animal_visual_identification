"""Label-blind score, rank, and frozen-boundary drift admission."""

from __future__ import annotations

import hashlib
import math
import struct
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator

from evaluation.control_scoring import (
    ControlScoringInventory,
    EmbeddingCacheEntry,
    EmbeddingCacheManifest,
    EmbeddingCachePolicy,
    EmbeddingCacheVerification,
    verify_embedding_cache_files,
)
from operations.embedding_producer import (
    EmbeddingProducerConfig,
    EmbeddingProductionReceipt,
)
from evaluation.numerical_admission import (
    NumericalAdmissionDecision,
    NumericalAdmissionReceipt,
    NumericalDriftPolicy,
    compare_embedding_caches,
)
from representation_learning.optimization import PromotionDecision
from foundation.provenance import content_sha256


class ScoreDriftDecision(StrEnum):
    PASS = "SCORE_RANK_THRESHOLD_PASS_ON_FROZEN_WORKLOAD"
    FAIL = "SCORE_RANK_THRESHOLD_FAIL"


@dataclass(frozen=True, slots=True)
class RetrievalScoreRequest:
    request_id: str
    query_group_token: str
    candidate_slot_token: str
    query_artifact_token: str
    candidate_artifact_token: str

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "query_group_token",
            "candidate_slot_token",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in ("query_artifact_token", "candidate_artifact_token"):
            _require_nonempty(getattr(self, name), name)

    def to_dict(self) -> dict[str, str]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RetrievalScoreRequest:
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            "retrieval score request",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class RetrievalScoreWorkload:
    gallery_sha256: str
    pairing_policy_sha256: str
    retrieval_plan_sha256: str
    split_manifest_sha256: str
    workload_construction_receipt_sha256: str
    data_role: str
    selection_blind_to_candidate_outputs: bool
    requests: tuple[RetrievalScoreRequest, ...]
    schema_version: str = "cvi.retrieval_score_workload.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.retrieval_score_workload.v2":
            raise ValueError("unsupported retrieval score workload schema")
        for name in (
            "gallery_sha256",
            "pairing_policy_sha256",
            "retrieval_plan_sha256",
            "split_manifest_sha256",
            "workload_construction_receipt_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if self.data_role != "OPTIMIZATION_DEV":
            raise ValueError("score drift workload role must be OPTIMIZATION_DEV")
        if self.selection_blind_to_candidate_outputs is not True:
            raise ValueError(
                "workload selection must attest candidate-output blindness"
            )
        if not self.requests:
            raise ValueError("retrieval workload must not be empty")
        request_ids = tuple(item.request_id for item in self.requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("retrieval request IDs must be unique")
        ordering = tuple(
            (item.query_group_token, item.candidate_slot_token)
            for item in self.requests
        )
        if ordering != tuple(sorted(ordering)):
            raise ValueError("retrieval workload must be canonically ordered")
        if len(ordering) != len(set(ordering)):
            raise ValueError("query/candidate slots must be unique")
        query_artifacts: dict[str, str] = {}
        candidate_artifacts: dict[str, str] = {}
        candidate_slots_by_query: dict[str, list[str]] = {}
        counts: Counter[str] = Counter()
        for item in self.requests:
            previous = query_artifacts.setdefault(
                item.query_group_token,
                item.query_artifact_token,
            )
            if previous != item.query_artifact_token:
                raise ValueError("one query group maps to multiple artifacts")
            previous_candidate = candidate_artifacts.setdefault(
                item.candidate_slot_token,
                item.candidate_artifact_token,
            )
            if previous_candidate != item.candidate_artifact_token:
                raise ValueError("one candidate slot maps to multiple artifacts")
            counts[item.query_group_token] += 1
            candidate_slots_by_query.setdefault(
                item.query_group_token,
                [],
            ).append(item.candidate_slot_token)
        if any(count < 2 for count in counts.values()):
            raise ValueError("every query requires at least two candidates")
        if len(set(query_artifacts.values())) != len(query_artifacts):
            raise ValueError("query groups must map to distinct artifacts")
        galleries = {
            tuple(slots) for slots in candidate_slots_by_query.values()
        }
        if len(galleries) != 1:
            raise ValueError("every query must search the identical gallery")

    @property
    def workload_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gallery_sha256": self.gallery_sha256,
            "pairing_policy_sha256": self.pairing_policy_sha256,
            "retrieval_plan_sha256": self.retrieval_plan_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "workload_construction_receipt_sha256": (
                self.workload_construction_receipt_sha256
            ),
            "data_role": self.data_role,
            "selection_blind_to_candidate_outputs": (
                self.selection_blind_to_candidate_outputs
            ),
            "requests": [item.to_dict() for item in self.requests],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RetrievalScoreWorkload:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "gallery_sha256",
                "pairing_policy_sha256",
                "retrieval_plan_sha256",
                "split_manifest_sha256",
                "workload_construction_receipt_sha256",
                "data_role",
                "selection_blind_to_candidate_outputs",
                "requests",
            },
            "retrieval score workload",
        )
        if not isinstance(payload["requests"], list):
            raise TypeError("retrieval requests must be a list")
        return cls(
            schema_version=payload["schema_version"],
            gallery_sha256=payload["gallery_sha256"],
            pairing_policy_sha256=payload["pairing_policy_sha256"],
            retrieval_plan_sha256=payload["retrieval_plan_sha256"],
            split_manifest_sha256=payload["split_manifest_sha256"],
            workload_construction_receipt_sha256=payload[
                "workload_construction_receipt_sha256"
            ],
            data_role=payload["data_role"],
            selection_blind_to_candidate_outputs=payload[
                "selection_blind_to_candidate_outputs"
            ],
            requests=tuple(
                RetrievalScoreRequest.from_dict(item)
                for item in payload["requests"]
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenScoreMarginBoundary:
    score_threshold: float
    margin_threshold: float
    top_k: tuple[int, ...]
    reference_model_sha256: str
    reference_producer_config_sha256: str
    reference_preprocessing_sha256: str
    reference_preprocessing_semantics_sha256: str
    gallery_sha256: str
    calibration_manifest_sha256: str
    calibration_score_receipt_sha256: str
    pairing_policy_sha256: str
    retrieval_plan_sha256: str
    workload_sha256: str
    scoring_semantics_sha256: str
    score_rule: str = "TOP1_SCORE_GREATER_THAN_OR_EQUAL"
    margin_rule: str = "TOP1_MINUS_TOP2_GREATER_THAN_OR_EQUAL"
    ranking_rule: str = "DESCENDING_SCORE_THEN_ASCENDING_OPAQUE_SLOT"
    schema_version: str = "cvi.frozen_score_margin_boundary.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.frozen_score_margin_boundary.v1":
            raise ValueError("unsupported frozen score-margin boundary schema")
        _require_finite(self.score_threshold, "score_threshold")
        _require_finite_positive(self.margin_threshold, "margin_threshold")
        if not -1.0 <= self.score_threshold <= 1.0:
            raise ValueError("cosine score threshold must be in [-1, 1]")
        if self.margin_threshold > 2.0:
            raise ValueError("cosine margin threshold must be in (0, 2]")
        if (
            not self.top_k
            or self.top_k != tuple(sorted(set(self.top_k)))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.top_k
            )
        ):
            raise ValueError("top_k must be unique sorted positive integers")
        for name in (
            "reference_model_sha256",
            "reference_producer_config_sha256",
            "reference_preprocessing_sha256",
            "reference_preprocessing_semantics_sha256",
            "gallery_sha256",
            "calibration_manifest_sha256",
            "calibration_score_receipt_sha256",
            "pairing_policy_sha256",
            "retrieval_plan_sha256",
            "workload_sha256",
            "scoring_semantics_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if self.score_rule != "TOP1_SCORE_GREATER_THAN_OR_EQUAL":
            raise ValueError("score threshold rule is fixed")
        if self.margin_rule != "TOP1_MINUS_TOP2_GREATER_THAN_OR_EQUAL":
            raise ValueError("margin threshold rule is fixed")
        if self.ranking_rule != "DESCENDING_SCORE_THEN_ASCENDING_OPAQUE_SLOT":
            raise ValueError("ranking rule is fixed")

    @property
    def boundary_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "score_threshold": self.score_threshold,
            "margin_threshold": self.margin_threshold,
            "top_k": list(self.top_k),
            "reference_model_sha256": self.reference_model_sha256,
            "reference_producer_config_sha256": (
                self.reference_producer_config_sha256
            ),
            "reference_preprocessing_sha256": (
                self.reference_preprocessing_sha256
            ),
            "reference_preprocessing_semantics_sha256": (
                self.reference_preprocessing_semantics_sha256
            ),
            "gallery_sha256": self.gallery_sha256,
            "calibration_manifest_sha256": self.calibration_manifest_sha256,
            "calibration_score_receipt_sha256": (
                self.calibration_score_receipt_sha256
            ),
            "pairing_policy_sha256": self.pairing_policy_sha256,
            "retrieval_plan_sha256": self.retrieval_plan_sha256,
            "workload_sha256": self.workload_sha256,
            "scoring_semantics_sha256": self.scoring_semantics_sha256,
            "score_rule": self.score_rule,
            "margin_rule": self.margin_rule,
            "ranking_rule": self.ranking_rule,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrozenScoreMarginBoundary:
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            "frozen score-margin boundary",
        )
        values = dict(payload)
        if not isinstance(values["top_k"], list):
            raise TypeError("top_k must be a list")
        values["top_k"] = tuple(values["top_k"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ScoreDriftPolicy:
    maximum_absolute_score_drift: float
    maximum_mean_absolute_score_drift: float
    maximum_absolute_margin_drift: float
    maximum_mean_absolute_margin_drift: float
    maximum_rank_inversions: int
    maximum_queries_with_rank_change: int
    maximum_rank_displacement: int
    maximum_top1_changes: int
    maximum_top_k_set_changes: int
    maximum_top_k_symmetric_difference_items: int
    maximum_top_k_queries_with_set_change_by_k: tuple[int, ...]
    maximum_top_k_symmetric_difference_items_by_k: tuple[int, ...]
    maximum_threshold_decision_flips: int
    maximum_reference_reject_candidate_accept_flips: int
    maximum_reference_accept_candidate_reject_flips: int
    maximum_requests: int = 1_000_000
    maximum_queries: int = 100_000
    maximum_candidates_per_query: int = 100_000
    maximum_scalar_products: int = 2_000_000_000
    maximum_embedding_bytes_read: int = 17_179_869_184
    maximum_cache_verification_bytes_read: int = 17_179_869_184
    maximum_numerical_recomputation_bytes_read: int = 17_179_869_184
    maximum_total_vector_bytes_read: int = 51_539_607_552
    dot_chunk_floats: int = 4_096
    metric: str = "cosine_l2_dot"
    accumulation: str = "float64_chunk_fsum_neumaier"
    schema_version: str = "cvi.score_drift_policy.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.score_drift_policy.v2":
            raise ValueError("unsupported score drift policy schema")
        for name in (
            "maximum_absolute_score_drift",
            "maximum_mean_absolute_score_drift",
            "maximum_absolute_margin_drift",
            "maximum_mean_absolute_margin_drift",
        ):
            _require_finite_nonnegative(getattr(self, name), name)
        for name in (
            "maximum_rank_inversions",
            "maximum_queries_with_rank_change",
            "maximum_rank_displacement",
            "maximum_top1_changes",
            "maximum_top_k_set_changes",
            "maximum_top_k_symmetric_difference_items",
            "maximum_threshold_decision_flips",
            "maximum_reference_reject_candidate_accept_flips",
            "maximum_reference_accept_candidate_reject_flips",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        for name in (
            "maximum_top_k_queries_with_set_change_by_k",
            "maximum_top_k_symmetric_difference_items_by_k",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not value:
                raise ValueError(f"{name} must be a non-empty tuple")
            for item in value:
                _require_nonnegative_int(item, name)
        for name in (
            "maximum_requests",
            "maximum_queries",
            "maximum_candidates_per_query",
            "maximum_scalar_products",
            "maximum_embedding_bytes_read",
            "maximum_cache_verification_bytes_read",
            "maximum_numerical_recomputation_bytes_read",
            "maximum_total_vector_bytes_read",
            "dot_chunk_floats",
        ):
            _require_positive_int(getattr(self, name), name)
        if self.metric != "cosine_l2_dot":
            raise ValueError("score drift metric is fixed to cosine_l2_dot")
        if self.accumulation != "float64_chunk_fsum_neumaier":
            raise ValueError("unsupported score drift accumulation")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
        for name in (
            "maximum_top_k_queries_with_set_change_by_k",
            "maximum_top_k_symmetric_difference_items_by_k",
        ):
            payload[name] = list(payload[name])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreDriftPolicy:
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            "score drift policy",
        )
        values = dict(payload)
        for name in (
            "maximum_top_k_queries_with_set_change_by_k",
            "maximum_top_k_symmetric_difference_items_by_k",
        ):
            if not isinstance(values[name], list):
                raise TypeError(f"{name} must be a list")
            values[name] = tuple(values[name])
        return cls(**values)


def score_drift_scoring_semantics_sha256(
    policy: ScoreDriftPolicy,
) -> str:
    return content_sha256(
        {
            "schema_version": "cvi.score_drift_scoring_semantics.v1",
            "metric": policy.metric,
            "accumulation": policy.accumulation,
            "dot_chunk_floats": policy.dot_chunk_floats,
            "vector_format": "float32_le",
            "vectors_l2_normalized": True,
            "ranking_rule": (
                "DESCENDING_SCORE_THEN_ASCENDING_OPAQUE_SLOT"
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class ScoreDriftSummary:
    requests: int
    queries: int
    scalar_products: int
    maximum_candidates_in_query: int
    maximum_absolute_score_drift: float
    mean_absolute_score_drift: float
    worst_score_request_id: str
    maximum_absolute_margin_drift: float
    mean_absolute_margin_drift: float
    worst_margin_query_token: str
    rank_inversions: int
    queries_with_rank_change: int
    maximum_rank_displacement: int
    top1_changes: int
    top_k_set_changes: int
    top_k_symmetric_difference_items: int
    top_k_values: tuple[int, ...]
    top_k_queries_with_set_change: tuple[int, ...]
    top_k_symmetric_difference_items_by_k: tuple[int, ...]
    threshold_decision_flips: int
    reference_reject_candidate_accept_flips: int
    reference_accept_candidate_reject_flips: int
    reference_acceptances: int
    candidate_acceptances: int
    reference_tie_pairs: int
    candidate_tie_pairs: int
    minimum_reference_score_boundary_distance: float
    minimum_candidate_score_boundary_distance: float
    minimum_reference_margin_boundary_distance: float
    minimum_candidate_margin_boundary_distance: float
    reference_score_digest: str
    candidate_score_digest: str
    schema_version: str = "cvi.score_drift_summary.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.score_drift_summary.v1":
            raise ValueError("unsupported score drift summary schema")
        for name in (
            "requests",
            "queries",
            "scalar_products",
            "maximum_candidates_in_query",
            "rank_inversions",
            "queries_with_rank_change",
            "maximum_rank_displacement",
            "top1_changes",
            "top_k_set_changes",
            "top_k_symmetric_difference_items",
            "threshold_decision_flips",
            "reference_reject_candidate_accept_flips",
            "reference_accept_candidate_reject_flips",
            "reference_acceptances",
            "candidate_acceptances",
            "reference_tie_pairs",
            "candidate_tie_pairs",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        if self.requests == 0 or self.queries == 0:
            raise ValueError("score drift summary must not be empty")
        if self.maximum_candidates_in_query < 2:
            raise ValueError("score drift queries require two candidates")
        if self.requests < self.queries * 2 or self.requests > (
            self.queries * self.maximum_candidates_in_query
        ):
            raise ValueError("score drift request/query accounting differs")
        if self.scalar_products == 0:
            raise ValueError("score drift scalar products must be positive")
        for name in (
            "maximum_absolute_score_drift",
            "mean_absolute_score_drift",
            "maximum_absolute_margin_drift",
            "mean_absolute_margin_drift",
            "minimum_reference_score_boundary_distance",
            "minimum_candidate_score_boundary_distance",
            "minimum_reference_margin_boundary_distance",
            "minimum_candidate_margin_boundary_distance",
        ):
            _require_finite_nonnegative(getattr(self, name), name)
        if self.mean_absolute_score_drift > self.maximum_absolute_score_drift:
            raise ValueError("mean score drift exceeds maximum")
        if self.mean_absolute_margin_drift > self.maximum_absolute_margin_drift:
            raise ValueError("mean margin drift exceeds maximum")
        if self.maximum_rank_displacement >= self.maximum_candidates_in_query:
            raise ValueError("rank displacement exceeds query gallery")
        _validate_sha256(self.worst_score_request_id, "worst_score_request_id")
        _validate_sha256(self.worst_margin_query_token, "worst_margin_query_token")
        _validate_sha256(self.reference_score_digest, "reference_score_digest")
        _validate_sha256(self.candidate_score_digest, "candidate_score_digest")
        if self.reference_acceptances > self.queries:
            raise ValueError("reference acceptances exceed query count")
        if self.candidate_acceptances > self.queries:
            raise ValueError("candidate acceptances exceed query count")
        if self.top1_changes > self.queries:
            raise ValueError("top1 changes exceed query count")
        if self.queries_with_rank_change > self.queries:
            raise ValueError("rank-change queries exceed query count")
        if self.threshold_decision_flips > self.queries:
            raise ValueError("decision flips exceed query count")
        if self.threshold_decision_flips != (
            self.reference_reject_candidate_accept_flips
            + self.reference_accept_candidate_reject_flips
        ):
            raise ValueError("directional decision flips do not sum to total")
        if (
            not self.top_k_values
            or self.top_k_values != tuple(sorted(set(self.top_k_values)))
            or any(value <= 0 for value in self.top_k_values)
        ):
            raise ValueError("summary Top-K values are invalid")
        if not (
            len(self.top_k_values)
            == len(self.top_k_queries_with_set_change)
            == len(self.top_k_symmetric_difference_items_by_k)
        ):
            raise ValueError("summary Top-K vectors differ in length")
        if any(
            value < 0 or value > self.queries
            for value in self.top_k_queries_with_set_change
        ):
            raise ValueError("summary per-K query count is invalid")
        if any(
            value < 0
            for value in self.top_k_symmetric_difference_items_by_k
        ):
            raise ValueError("summary per-K difference count is invalid")
        if sum(self.top_k_queries_with_set_change) != self.top_k_set_changes:
            raise ValueError("summary Top-K change aggregate differs")
        if sum(self.top_k_symmetric_difference_items_by_k) != (
            self.top_k_symmetric_difference_items
        ):
            raise ValueError("summary Top-K difference aggregate differs")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
        for name in (
            "top_k_values",
            "top_k_queries_with_set_change",
            "top_k_symmetric_difference_items_by_k",
        ):
            payload[name] = list(payload[name])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreDriftSummary:
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            "score drift summary",
        )
        values = dict(payload)
        for name in (
            "top_k_values",
            "top_k_queries_with_set_change",
            "top_k_symmetric_difference_items_by_k",
        ):
            if not isinstance(values[name], list):
                raise TypeError("score drift Top-K summaries must be lists")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ScoreDriftCost:
    cache_verification_bytes_read: int
    numerical_recomputation_bytes_read: int
    scoring_vector_bytes_read: int
    total_vector_bytes_read: int
    scalar_products: int
    peak_vector_payload_bytes: int
    rank_pair_relations_summarized: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_nonnegative_int(getattr(self, name), name)

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreDriftCost:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "score drift cost")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ScoreDriftPrecommitment:
    workload_sha256: str
    inventory_sha256: str
    reference_production_receipt_sha256: str
    reference_config_sha256: str
    candidate_config_sha256: str
    numerical_policy_sha256: str
    frozen_boundary_sha256: str
    score_drift_policy_sha256: str
    cache_policy_sha256: str
    prior_attempt_ledger_sha256: str
    candidate_attempt_token: str
    precommitment_sequence: int
    schema_version: str = "cvi.score_drift_precommitment.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.score_drift_precommitment.v1":
            raise ValueError("unsupported score drift precommitment schema")
        for name in self.__dataclass_fields__:
            if name not in ("schema_version", "precommitment_sequence"):
                _validate_sha256(getattr(self, name), name)
        _require_positive_int(
            self.precommitment_sequence,
            "precommitment_sequence",
        )

    @property
    def precommitment_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreDriftPrecommitment:
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            "score drift precommitment",
        )
        return cls(**payload)


def build_score_drift_precommitment(
    *,
    workload: RetrievalScoreWorkload,
    inventory: ControlScoringInventory,
    reference_production: EmbeddingProductionReceipt,
    reference_config: EmbeddingProducerConfig,
    candidate_config: EmbeddingProducerConfig,
    numerical_policy: NumericalDriftPolicy,
    boundary: FrozenScoreMarginBoundary,
    policy: ScoreDriftPolicy,
    cache_policy: EmbeddingCachePolicy,
    prior_attempt_ledger_sha256: str,
    candidate_attempt_token: str,
    precommitment_sequence: int,
) -> ScoreDriftPrecommitment:
    if boundary.workload_sha256 != workload.workload_sha256:
        raise ValueError("precommitment boundary exact workload differs")
    if reference_production.producer_config_sha256 != (
        reference_config.config_sha256
    ):
        raise ValueError("precommitment reference production config differs")
    if reference_production.scoring_inventory_sha256 != (
        inventory.inventory_sha256
    ):
        raise ValueError("precommitment reference inventory differs")
    if boundary.scoring_semantics_sha256 != (
        score_drift_scoring_semantics_sha256(policy)
    ):
        raise ValueError("precommitment scoring semantics differ")
    return ScoreDriftPrecommitment(
        workload_sha256=workload.workload_sha256,
        inventory_sha256=inventory.inventory_sha256,
        reference_production_receipt_sha256=(
            reference_production.receipt_sha256
        ),
        reference_config_sha256=reference_config.config_sha256,
        candidate_config_sha256=candidate_config.config_sha256,
        numerical_policy_sha256=numerical_policy.policy_sha256,
        frozen_boundary_sha256=boundary.boundary_sha256,
        score_drift_policy_sha256=policy.policy_sha256,
        cache_policy_sha256=cache_policy.policy_sha256,
        prior_attempt_ledger_sha256=prior_attempt_ledger_sha256,
        candidate_attempt_token=candidate_attempt_token,
        precommitment_sequence=precommitment_sequence,
    )


@dataclass(frozen=True, slots=True)
class ScoreDriftAdmissionPlan:
    precommitment_sha256: str
    precommitment: ScoreDriftPrecommitment
    completed_attempt_ledger_sha256: str
    workload_sha256: str
    inventory_sha256: str
    reference_production_receipt_sha256: str
    candidate_production_receipt_sha256: str
    reference_config_sha256: str
    candidate_config_sha256: str
    numerical_admission_receipt_sha256: str
    numerical_policy_sha256: str
    frozen_boundary_sha256: str
    score_drift_policy_sha256: str
    cache_policy_sha256: str
    schema_version: str = "cvi.score_drift_admission_plan.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.score_drift_admission_plan.v2":
            raise ValueError("unsupported score drift admission plan schema")
        for name in self.__dataclass_fields__:
            if name not in ("schema_version", "precommitment"):
                _validate_sha256(getattr(self, name), name)
        if self.precommitment.precommitment_sha256 != self.precommitment_sha256:
            raise ValueError("embedded score drift precommitment hash differs")
        expected_ledger_head = score_drift_attempt_ledger_head(
            precommitment=self.precommitment,
            candidate_production_receipt_sha256=(
                self.candidate_production_receipt_sha256
            ),
            numerical_admission_receipt_sha256=(
                self.numerical_admission_receipt_sha256
            ),
        )
        if self.completed_attempt_ledger_sha256 != expected_ledger_head:
            raise ValueError("completed attempt ledger transition differs")

    @property
    def plan_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
        payload["precommitment"] = self.precommitment.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreDriftAdmissionPlan:
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            "score drift admission plan",
        )
        values = dict(payload)
        if not isinstance(values["precommitment"], dict):
            raise TypeError("score drift precommitment must be an object")
        values["precommitment"] = ScoreDriftPrecommitment.from_dict(
            values["precommitment"]
        )
        return cls(**values)


def build_score_drift_admission_plan(
    *,
    workload: RetrievalScoreWorkload,
    inventory: ControlScoringInventory,
    reference_production: EmbeddingProductionReceipt,
    candidate_production: EmbeddingProductionReceipt,
    reference_config: EmbeddingProducerConfig,
    candidate_config: EmbeddingProducerConfig,
    numerical_admission: NumericalAdmissionReceipt,
    numerical_policy: NumericalDriftPolicy,
    boundary: FrozenScoreMarginBoundary,
    policy: ScoreDriftPolicy,
    cache_policy: EmbeddingCachePolicy,
    precommitment: ScoreDriftPrecommitment,
) -> ScoreDriftAdmissionPlan:
    expected_precommitment = {
        "workload_sha256": workload.workload_sha256,
        "inventory_sha256": inventory.inventory_sha256,
        "reference_production_receipt_sha256": (
            reference_production.receipt_sha256
        ),
        "reference_config_sha256": reference_config.config_sha256,
        "candidate_config_sha256": candidate_config.config_sha256,
        "numerical_policy_sha256": numerical_policy.policy_sha256,
        "frozen_boundary_sha256": boundary.boundary_sha256,
        "score_drift_policy_sha256": policy.policy_sha256,
        "cache_policy_sha256": cache_policy.policy_sha256,
    }
    for name, value in expected_precommitment.items():
        if getattr(precommitment, name) != value:
            raise ValueError(f"score drift precommitment {name} differs")
    completed_attempt_ledger_sha256 = score_drift_attempt_ledger_head(
        precommitment=precommitment,
        candidate_production_receipt_sha256=(
            candidate_production.receipt_sha256
        ),
        numerical_admission_receipt_sha256=(
            numerical_admission.receipt_sha256
        ),
    )
    return ScoreDriftAdmissionPlan(
        precommitment_sha256=precommitment.precommitment_sha256,
        precommitment=precommitment,
        completed_attempt_ledger_sha256=completed_attempt_ledger_sha256,
        workload_sha256=workload.workload_sha256,
        inventory_sha256=inventory.inventory_sha256,
        reference_production_receipt_sha256=(
            reference_production.receipt_sha256
        ),
        candidate_production_receipt_sha256=(
            candidate_production.receipt_sha256
        ),
        reference_config_sha256=reference_config.config_sha256,
        candidate_config_sha256=candidate_config.config_sha256,
        numerical_admission_receipt_sha256=(
            numerical_admission.receipt_sha256
        ),
        numerical_policy_sha256=numerical_policy.policy_sha256,
        frozen_boundary_sha256=boundary.boundary_sha256,
        score_drift_policy_sha256=policy.policy_sha256,
        cache_policy_sha256=cache_policy.policy_sha256,
    )


def score_drift_attempt_ledger_head(
    *,
    precommitment: ScoreDriftPrecommitment,
    candidate_production_receipt_sha256: str,
    numerical_admission_receipt_sha256: str,
) -> str:
    _validate_sha256(
        candidate_production_receipt_sha256,
        "candidate_production_receipt_sha256",
    )
    _validate_sha256(
        numerical_admission_receipt_sha256,
        "numerical_admission_receipt_sha256",
    )
    return content_sha256(
        {
            "schema_version": "cvi.score_drift_attempt_ledger_entry.v1",
            "prior_attempt_ledger_sha256": (
                precommitment.prior_attempt_ledger_sha256
            ),
            "candidate_attempt_token": precommitment.candidate_attempt_token,
            "precommitment_sequence": precommitment.precommitment_sequence,
            "precommitment_sha256": precommitment.precommitment_sha256,
            "candidate_production_receipt_sha256": (
                candidate_production_receipt_sha256
            ),
            "numerical_admission_receipt_sha256": (
                numerical_admission_receipt_sha256
            ),
            "phase": "CANDIDATE_PRODUCTION_AND_NUMERICAL_ADMISSION_BOUND",
        }
    )


@dataclass(frozen=True, slots=True)
class ScoreDriftAdmissionReceipt:
    precommitment_sha256: str
    admission_plan_sha256: str
    admission_plan: ScoreDriftAdmissionPlan
    workload_sha256: str
    inventory_sha256: str
    reference_production_receipt_sha256: str
    candidate_production_receipt_sha256: str
    reference_manifest_sha256: str
    candidate_manifest_sha256: str
    reference_cache_verification_sha256: str
    candidate_cache_verification_sha256: str
    reference_config_sha256: str
    candidate_config_sha256: str
    numerical_admission_receipt_sha256: str
    numerical_policy_sha256: str
    frozen_boundary_sha256: str
    score_drift_policy_sha256: str
    cache_policy_sha256: str
    frozen_boundary: FrozenScoreMarginBoundary
    score_drift_policy: ScoreDriftPolicy
    summary: ScoreDriftSummary
    cost: ScoreDriftCost
    hard_failures: tuple[str, ...]
    decision: ScoreDriftDecision
    promotion_decision: PromotionDecision = PromotionDecision.INCONCLUSIVE
    interpretation: str = (
        "LABEL_BLIND_SCORE_RANK_THRESHOLD_ADMISSION_ONLY_"
        "NOT_BIOMETRIC_NONINFERIORITY_OR_OPTIMIZATION_PROMOTION"
    )
    schema_version: str = "cvi.score_drift_admission_receipt.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.score_drift_admission_receipt.v2":
            raise ValueError("unsupported score drift receipt schema")
        for name in (
            "precommitment_sha256",
            "workload_sha256",
            "inventory_sha256",
            "reference_production_receipt_sha256",
            "candidate_production_receipt_sha256",
            "reference_manifest_sha256",
            "candidate_manifest_sha256",
            "reference_cache_verification_sha256",
            "candidate_cache_verification_sha256",
            "reference_config_sha256",
            "candidate_config_sha256",
            "numerical_admission_receipt_sha256",
            "numerical_policy_sha256",
            "frozen_boundary_sha256",
            "score_drift_policy_sha256",
            "cache_policy_sha256",
            "admission_plan_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if self.hard_failures != tuple(sorted(set(self.hard_failures))):
            raise ValueError("hard failures must be unique and sorted")
        if self.admission_plan.plan_sha256 != self.admission_plan_sha256:
            raise ValueError("embedded score drift plan hash differs")
        if self.admission_plan.precommitment_sha256 != (
            self.precommitment_sha256
        ):
            raise ValueError("embedded score drift precommitment differs")
        expected_plan_bindings = {
            "workload_sha256": self.workload_sha256,
            "inventory_sha256": self.inventory_sha256,
            "reference_production_receipt_sha256": (
                self.reference_production_receipt_sha256
            ),
            "candidate_production_receipt_sha256": (
                self.candidate_production_receipt_sha256
            ),
            "reference_config_sha256": self.reference_config_sha256,
            "candidate_config_sha256": self.candidate_config_sha256,
            "numerical_admission_receipt_sha256": (
                self.numerical_admission_receipt_sha256
            ),
            "numerical_policy_sha256": self.numerical_policy_sha256,
            "frozen_boundary_sha256": self.frozen_boundary_sha256,
            "score_drift_policy_sha256": self.score_drift_policy_sha256,
            "cache_policy_sha256": self.cache_policy_sha256,
        }
        for name, value in expected_plan_bindings.items():
            if getattr(self.admission_plan, name) != value:
                raise ValueError(f"embedded score drift plan {name} differs")
        if self.frozen_boundary.boundary_sha256 != self.frozen_boundary_sha256:
            raise ValueError("embedded frozen boundary hash differs")
        if self.score_drift_policy.policy_sha256 != (
            self.score_drift_policy_sha256
        ):
            raise ValueError("embedded score drift policy hash differs")
        _validate_top_k_policy(self.score_drift_policy, self.frozen_boundary)
        expected_failures = _hard_failures(
            self.summary,
            self.score_drift_policy,
        )
        if self.hard_failures != expected_failures:
            raise ValueError("score drift failures disagree with policy")
        if self.summary.requests > self.score_drift_policy.maximum_requests:
            raise ValueError("receipt requests exceed score drift policy")
        if self.summary.queries > self.score_drift_policy.maximum_queries:
            raise ValueError("receipt queries exceed score drift policy")
        if self.summary.maximum_candidates_in_query > (
            self.score_drift_policy.maximum_candidates_per_query
        ):
            raise ValueError("receipt query gallery exceeds score drift policy")
        if self.summary.scalar_products > (
            self.score_drift_policy.maximum_scalar_products
        ):
            raise ValueError("receipt scalar products exceed score drift policy")
        if self.summary.top_k_set_changes > (
            self.summary.queries * len(self.frozen_boundary.top_k)
        ):
            raise ValueError("receipt Top-K change count is impossible")
        if self.summary.top_k_values != self.frozen_boundary.top_k:
            raise ValueError("receipt Top-K summary differs from boundary")
        expected = (
            ScoreDriftDecision.FAIL
            if self.hard_failures
            else ScoreDriftDecision.PASS
        )
        if self.decision is not expected:
            raise ValueError("score drift decision disagrees with hard failures")
        if self.promotion_decision is not PromotionDecision.INCONCLUSIVE:
            raise ValueError("score drift admission cannot promote an optimization")
        if self.cost.scalar_products != self.summary.scalar_products:
            raise ValueError("score drift scalar-product accounting differs")
        if self.cost.total_vector_bytes_read != (
            self.cost.cache_verification_bytes_read
            + self.cost.numerical_recomputation_bytes_read
            + self.cost.scoring_vector_bytes_read
        ):
            raise ValueError("score drift byte accounting differs")
        if self.cost.scoring_vector_bytes_read > (
            self.score_drift_policy.maximum_embedding_bytes_read
        ):
            raise ValueError("receipt scoring bytes exceed policy")
        if self.cost.cache_verification_bytes_read > (
            self.score_drift_policy.maximum_cache_verification_bytes_read
        ):
            raise ValueError("receipt cache verification bytes exceed policy")
        if self.cost.numerical_recomputation_bytes_read > (
            self.score_drift_policy.maximum_numerical_recomputation_bytes_read
        ):
            raise ValueError("receipt numerical recomputation exceeds policy")
        if self.cost.total_vector_bytes_read > (
            self.score_drift_policy.maximum_total_vector_bytes_read
        ):
            raise ValueError("receipt total vector bytes exceed policy")
        if self.interpretation != (
            "LABEL_BLIND_SCORE_RANK_THRESHOLD_ADMISSION_ONLY_"
            "NOT_BIOMETRIC_NONINFERIORITY_OR_OPTIMIZATION_PROMOTION"
        ):
            raise ValueError("score drift interpretation is fixed")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "precommitment_sha256": self.precommitment_sha256,
            "admission_plan_sha256": self.admission_plan_sha256,
            "admission_plan": self.admission_plan.to_dict(),
            "workload_sha256": self.workload_sha256,
            "inventory_sha256": self.inventory_sha256,
            "reference_production_receipt_sha256": (
                self.reference_production_receipt_sha256
            ),
            "candidate_production_receipt_sha256": (
                self.candidate_production_receipt_sha256
            ),
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "reference_cache_verification_sha256": (
                self.reference_cache_verification_sha256
            ),
            "candidate_cache_verification_sha256": (
                self.candidate_cache_verification_sha256
            ),
            "reference_config_sha256": self.reference_config_sha256,
            "candidate_config_sha256": self.candidate_config_sha256,
            "numerical_admission_receipt_sha256": (
                self.numerical_admission_receipt_sha256
            ),
            "numerical_policy_sha256": self.numerical_policy_sha256,
            "frozen_boundary_sha256": self.frozen_boundary_sha256,
            "score_drift_policy_sha256": self.score_drift_policy_sha256,
            "cache_policy_sha256": self.cache_policy_sha256,
            "frozen_boundary": self.frozen_boundary.to_dict(),
            "score_drift_policy": self.score_drift_policy.to_dict(),
            "summary": self.summary.to_dict(),
            "cost": self.cost.to_dict(),
            "hard_failures": list(self.hard_failures),
            "decision": self.decision.value,
            "promotion_decision": self.promotion_decision.value,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreDriftAdmissionReceipt:
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            "score drift receipt",
        )
        if not isinstance(payload["summary"], dict) or not isinstance(
            payload["cost"], dict
        ):
            raise TypeError("score drift summary and cost must be objects")
        if not isinstance(payload["admission_plan"], dict):
            raise TypeError("score drift admission plan must be an object")
        if not isinstance(payload["frozen_boundary"], dict) or not isinstance(
            payload["score_drift_policy"], dict
        ):
            raise TypeError("score drift boundary and policy must be objects")
        if not isinstance(payload["hard_failures"], list):
            raise TypeError("score drift hard failures must be a list")
        values = dict(payload)
        values["admission_plan"] = ScoreDriftAdmissionPlan.from_dict(
            values["admission_plan"]
        )
        values["summary"] = ScoreDriftSummary.from_dict(values["summary"])
        values["cost"] = ScoreDriftCost.from_dict(values["cost"])
        values["frozen_boundary"] = FrozenScoreMarginBoundary.from_dict(
            values["frozen_boundary"]
        )
        values["score_drift_policy"] = ScoreDriftPolicy.from_dict(
            values["score_drift_policy"]
        )
        values["hard_failures"] = tuple(values["hard_failures"])
        values["decision"] = ScoreDriftDecision(values["decision"])
        values["promotion_decision"] = PromotionDecision(values["promotion_decision"])
        return cls(**values)


def compare_score_rank_threshold_drift(
    *,
    workload: RetrievalScoreWorkload,
    inventory: ControlScoringInventory,
    reference_root: Path,
    candidate_root: Path,
    reference_manifest: EmbeddingCacheManifest,
    candidate_manifest: EmbeddingCacheManifest,
    reference_verification: EmbeddingCacheVerification,
    candidate_verification: EmbeddingCacheVerification,
    reference_production: EmbeddingProductionReceipt,
    candidate_production: EmbeddingProductionReceipt,
    reference_config: EmbeddingProducerConfig,
    candidate_config: EmbeddingProducerConfig,
    numerical_admission: NumericalAdmissionReceipt,
    numerical_policy: NumericalDriftPolicy,
    boundary: FrozenScoreMarginBoundary,
    policy: ScoreDriftPolicy,
    cache_policy: EmbeddingCachePolicy,
    admission_plan: ScoreDriftAdmissionPlan,
    expected_precommitment_sha256: str,
    expected_admission_plan_sha256: str,
) -> ScoreDriftAdmissionReceipt:
    """Compare exact opaque retrieval work without identity labels."""

    _validate_lineage(
        workload=workload,
        inventory=inventory,
        reference_manifest=reference_manifest,
        candidate_manifest=candidate_manifest,
        reference_verification=reference_verification,
        candidate_verification=candidate_verification,
        reference_config=reference_config,
        candidate_config=candidate_config,
        reference_production=reference_production,
        candidate_production=candidate_production,
        numerical_admission=numerical_admission,
        numerical_policy=numerical_policy,
        boundary=boundary,
        policy=policy,
        cache_policy=cache_policy,
    )
    _validate_sha256(
        expected_precommitment_sha256,
        "expected_precommitment_sha256",
    )
    _validate_sha256(
        expected_admission_plan_sha256,
        "expected_admission_plan_sha256",
    )
    if admission_plan.precommitment_sha256 != expected_precommitment_sha256:
        raise ValueError("score drift precommitment differs from external anchor")
    if admission_plan.plan_sha256 != expected_admission_plan_sha256:
        raise ValueError("score drift plan differs from external anchor")
    expected_plan = build_score_drift_admission_plan(
        workload=workload,
        inventory=inventory,
        reference_production=reference_production,
        candidate_production=candidate_production,
        reference_config=reference_config,
        candidate_config=candidate_config,
        numerical_admission=numerical_admission,
        numerical_policy=numerical_policy,
        boundary=boundary,
        policy=policy,
        cache_policy=cache_policy,
        precommitment=admission_plan.precommitment,
    )
    if admission_plan != expected_plan:
        raise ValueError("score drift admission plan differs from exact inputs")
    query_counts = Counter(item.query_group_token for item in workload.requests)
    if len(workload.requests) > policy.maximum_requests:
        raise ValueError("retrieval requests exceed score drift policy")
    if len(query_counts) > policy.maximum_queries:
        raise ValueError("retrieval queries exceed score drift policy")
    maximum_candidates = max(query_counts.values())
    if maximum_candidates > policy.maximum_candidates_per_query:
        raise ValueError("retrieval candidates per query exceed policy")
    if boundary.top_k[-1] > min(query_counts.values()):
        raise ValueError("top_k exceeds at least one query gallery")
    _validate_top_k_policy(policy, boundary)
    scalar_products = (
        2 * len(workload.requests) * reference_manifest.vector_dimension
    )
    if scalar_products > policy.maximum_scalar_products:
        raise ValueError("score drift scalar-product work exceeds policy")
    scoring_bytes = (
        2
        * (len(workload.requests) + len(query_counts))
        * reference_manifest.vector_dimension
        * 4
    )
    if scoring_bytes > policy.maximum_embedding_bytes_read:
        raise ValueError("score drift embedding reads exceed policy")
    cache_verification_bytes = 2 * (
        sum(item.byte_size for item in reference_manifest.entries)
        + sum(item.byte_size for item in candidate_manifest.entries)
    )
    if cache_verification_bytes > (
        policy.maximum_cache_verification_bytes_read
    ):
        raise ValueError("score drift cache verification reads exceed policy")
    numerical_bytes = numerical_admission.summary.bytes_read
    if numerical_bytes > policy.maximum_numerical_recomputation_bytes_read:
        raise ValueError("score drift numerical recomputation exceeds policy")
    total_vector_bytes = cache_verification_bytes + numerical_bytes + scoring_bytes
    if total_vector_bytes > policy.maximum_total_vector_bytes_read:
        raise ValueError("score drift total vector reads exceed policy")

    recomputed_numerical = compare_embedding_caches(
        reference_manifest=reference_manifest,
        candidate_manifest=candidate_manifest,
        reference_config=reference_config,
        candidate_config=candidate_config,
        reference_root=reference_root,
        candidate_root=candidate_root,
        policy=numerical_policy,
    )
    if recomputed_numerical != numerical_admission:
        raise ValueError("numerical admission does not match cache recomputation")

    current_reference = verify_embedding_cache_files(
        root=reference_root,
        inventory=inventory,
        manifest=reference_manifest,
        policy=cache_policy,
    )
    current_candidate = verify_embedding_cache_files(
        root=candidate_root,
        inventory=inventory,
        manifest=candidate_manifest,
        policy=cache_policy,
    )
    if current_reference != reference_verification:
        raise ValueError("reference cache changed before score drift admission")
    if current_candidate != candidate_verification:
        raise ValueError("candidate cache changed before score drift admission")

    reference_paths = _paths_by_token(reference_root, reference_manifest)
    candidate_paths = _paths_by_token(candidate_root, candidate_manifest)
    reference_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    total_score_drift = 0.0
    score_drift_compensation = 0.0
    maximum_score_drift = -1.0
    worst_score_request = workload.requests[0].request_id
    total_margin_drift = 0.0
    margin_drift_compensation = 0.0
    maximum_margin_drift = -1.0
    worst_margin_query = workload.requests[0].query_group_token
    rank_inversions = 0
    queries_with_rank_change = 0
    maximum_rank_displacement = 0
    top1_changes = 0
    top_k_set_changes = 0
    top_k_symmetric_difference = 0
    top_k_change_counts = {value: 0 for value in boundary.top_k}
    top_k_difference_counts = {value: 0 for value in boundary.top_k}
    decision_flips = 0
    reject_to_accept_flips = 0
    accept_to_reject_flips = 0
    reference_acceptances = 0
    candidate_acceptances = 0
    reference_tie_pairs = 0
    candidate_tie_pairs = 0
    minimum_reference_score_distance = math.inf
    minimum_candidate_score_distance = math.inf
    minimum_reference_margin_distance = math.inf
    minimum_candidate_margin_distance = math.inf

    for query_token, requests in _group_requests(workload.requests):
        query_artifact = requests[0].query_artifact_token
        reference_query = _read_verified_payload(reference_paths[query_artifact])
        candidate_query = _read_verified_payload(candidate_paths[query_artifact])
        rows: list[tuple[RetrievalScoreRequest, float, float]] = []
        for request in requests:
            reference_candidate = _read_verified_payload(
                reference_paths[request.candidate_artifact_token]
            )
            reference_score = _dot_payloads(
                reference_query,
                reference_candidate,
                policy.dot_chunk_floats,
            )
            del reference_candidate
            candidate_candidate = _read_verified_payload(
                candidate_paths[request.candidate_artifact_token]
            )
            candidate_score = _dot_payloads(
                candidate_query,
                candidate_candidate,
                policy.dot_chunk_floats,
            )
            del candidate_candidate
            drift = abs(reference_score - candidate_score)
            total_score_drift, score_drift_compensation = _neumaier_add(
                total_score_drift,
                score_drift_compensation,
                drift,
            )
            if drift > maximum_score_drift:
                maximum_score_drift = drift
                worst_score_request = request.request_id
            _update_score_digest(reference_digest, request.request_id, reference_score)
            _update_score_digest(candidate_digest, request.request_id, candidate_score)
            rows.append((request, reference_score, candidate_score))

        reference_order = sorted(
            rows,
            key=lambda row: (-row[1], row[0].candidate_slot_token),
        )
        candidate_order = sorted(
            rows,
            key=lambda row: (-row[2], row[0].candidate_slot_token),
        )
        reference_slots = tuple(row[0].candidate_slot_token for row in reference_order)
        candidate_slots = tuple(row[0].candidate_slot_token for row in candidate_order)
        if reference_slots != candidate_slots:
            queries_with_rank_change += 1
        if reference_slots[0] != candidate_slots[0]:
            top1_changes += 1
        inversions, displacement = _rank_drift(reference_slots, candidate_slots)
        rank_inversions += inversions
        maximum_rank_displacement = max(maximum_rank_displacement, displacement)
        top_k_differences = _top_k_drift(
            reference_slots,
            candidate_slots,
            boundary.top_k,
        )
        for top_k, difference in top_k_differences:
            if difference:
                top_k_set_changes += 1
                top_k_symmetric_difference += difference
                top_k_change_counts[top_k] += 1
                top_k_difference_counts[top_k] += difference

        reference_top1 = reference_order[0][1]
        reference_margin = reference_top1 - reference_order[1][1]
        candidate_top1 = candidate_order[0][2]
        candidate_margin = candidate_top1 - candidate_order[1][2]
        margin_drift = abs(reference_margin - candidate_margin)
        total_margin_drift, margin_drift_compensation = _neumaier_add(
            total_margin_drift,
            margin_drift_compensation,
            margin_drift,
        )
        if margin_drift > maximum_margin_drift:
            maximum_margin_drift = margin_drift
            worst_margin_query = query_token
        reference_accepted = (
            reference_top1 >= boundary.score_threshold
            and reference_margin >= boundary.margin_threshold
        )
        candidate_accepted = (
            candidate_top1 >= boundary.score_threshold
            and candidate_margin >= boundary.margin_threshold
        )
        reference_acceptances += int(reference_accepted)
        candidate_acceptances += int(candidate_accepted)
        decision_flips += int(reference_accepted != candidate_accepted)
        reject_to_accept_flips += int(
            not reference_accepted and candidate_accepted
        )
        accept_to_reject_flips += int(
            reference_accepted and not candidate_accepted
        )
        reference_tie_pairs += _tie_pairs(row[1] for row in rows)
        candidate_tie_pairs += _tie_pairs(row[2] for row in rows)
        minimum_reference_score_distance = min(
            minimum_reference_score_distance,
            abs(reference_top1 - boundary.score_threshold),
        )
        minimum_candidate_score_distance = min(
            minimum_candidate_score_distance,
            abs(candidate_top1 - boundary.score_threshold),
        )
        minimum_reference_margin_distance = min(
            minimum_reference_margin_distance,
            abs(reference_margin - boundary.margin_threshold),
        )
        minimum_candidate_margin_distance = min(
            minimum_candidate_margin_distance,
            abs(candidate_margin - boundary.margin_threshold),
        )

    final_reference = verify_embedding_cache_files(
        root=reference_root,
        inventory=inventory,
        manifest=reference_manifest,
        policy=cache_policy,
    )
    final_candidate = verify_embedding_cache_files(
        root=candidate_root,
        inventory=inventory,
        manifest=candidate_manifest,
        policy=cache_policy,
    )
    if (
        final_reference != reference_verification
        or final_candidate != candidate_verification
    ):
        raise RuntimeError("embedding cache changed during score drift admission")

    summary = ScoreDriftSummary(
        requests=len(workload.requests),
        queries=len(query_counts),
        scalar_products=scalar_products,
        maximum_candidates_in_query=maximum_candidates,
        maximum_absolute_score_drift=maximum_score_drift,
        mean_absolute_score_drift=(
            total_score_drift + score_drift_compensation
        )
        / len(workload.requests),
        worst_score_request_id=worst_score_request,
        maximum_absolute_margin_drift=maximum_margin_drift,
        mean_absolute_margin_drift=(
            total_margin_drift + margin_drift_compensation
        )
        / len(query_counts),
        worst_margin_query_token=worst_margin_query,
        rank_inversions=rank_inversions,
        queries_with_rank_change=queries_with_rank_change,
        maximum_rank_displacement=maximum_rank_displacement,
        top1_changes=top1_changes,
        top_k_set_changes=top_k_set_changes,
        top_k_symmetric_difference_items=top_k_symmetric_difference,
        top_k_values=boundary.top_k,
        top_k_queries_with_set_change=tuple(
            top_k_change_counts[value] for value in boundary.top_k
        ),
        top_k_symmetric_difference_items_by_k=tuple(
            top_k_difference_counts[value] for value in boundary.top_k
        ),
        threshold_decision_flips=decision_flips,
        reference_reject_candidate_accept_flips=reject_to_accept_flips,
        reference_accept_candidate_reject_flips=accept_to_reject_flips,
        reference_acceptances=reference_acceptances,
        candidate_acceptances=candidate_acceptances,
        reference_tie_pairs=reference_tie_pairs,
        candidate_tie_pairs=candidate_tie_pairs,
        minimum_reference_score_boundary_distance=minimum_reference_score_distance,
        minimum_candidate_score_boundary_distance=minimum_candidate_score_distance,
        minimum_reference_margin_boundary_distance=minimum_reference_margin_distance,
        minimum_candidate_margin_boundary_distance=minimum_candidate_margin_distance,
        reference_score_digest=reference_digest.hexdigest(),
        candidate_score_digest=candidate_digest.hexdigest(),
    )
    failures = _hard_failures(summary, policy)
    return ScoreDriftAdmissionReceipt(
        precommitment_sha256=admission_plan.precommitment_sha256,
        admission_plan_sha256=admission_plan.plan_sha256,
        admission_plan=admission_plan,
        workload_sha256=workload.workload_sha256,
        inventory_sha256=inventory.inventory_sha256,
        reference_production_receipt_sha256=(
            reference_production.receipt_sha256
        ),
        candidate_production_receipt_sha256=(
            candidate_production.receipt_sha256
        ),
        reference_manifest_sha256=reference_manifest.manifest_sha256,
        candidate_manifest_sha256=candidate_manifest.manifest_sha256,
        reference_cache_verification_sha256=reference_verification.verification_sha256,
        candidate_cache_verification_sha256=candidate_verification.verification_sha256,
        reference_config_sha256=reference_config.config_sha256,
        candidate_config_sha256=candidate_config.config_sha256,
        numerical_admission_receipt_sha256=numerical_admission.receipt_sha256,
        numerical_policy_sha256=numerical_policy.policy_sha256,
        frozen_boundary_sha256=boundary.boundary_sha256,
        score_drift_policy_sha256=policy.policy_sha256,
        cache_policy_sha256=cache_policy.policy_sha256,
        frozen_boundary=boundary,
        score_drift_policy=policy,
        summary=summary,
        cost=ScoreDriftCost(
            cache_verification_bytes_read=cache_verification_bytes,
            numerical_recomputation_bytes_read=numerical_bytes,
            scoring_vector_bytes_read=scoring_bytes,
            total_vector_bytes_read=total_vector_bytes,
            scalar_products=scalar_products,
            peak_vector_payload_bytes=3 * reference_manifest.vector_dimension * 4,
            rank_pair_relations_summarized=sum(
                count * (count - 1) // 2 for count in query_counts.values()
            ),
        ),
        hard_failures=failures,
        decision=ScoreDriftDecision.FAIL if failures else ScoreDriftDecision.PASS,
    )


def _validate_lineage(
    *,
    workload: RetrievalScoreWorkload,
    inventory: ControlScoringInventory,
    reference_manifest: EmbeddingCacheManifest,
    candidate_manifest: EmbeddingCacheManifest,
    reference_verification: EmbeddingCacheVerification,
    candidate_verification: EmbeddingCacheVerification,
    reference_config: EmbeddingProducerConfig,
    candidate_config: EmbeddingProducerConfig,
    reference_production: EmbeddingProductionReceipt,
    candidate_production: EmbeddingProductionReceipt,
    numerical_admission: NumericalAdmissionReceipt,
    numerical_policy: NumericalDriftPolicy,
    boundary: FrozenScoreMarginBoundary,
    policy: ScoreDriftPolicy,
    cache_policy: EmbeddingCachePolicy,
) -> None:
    if numerical_admission.decision is not NumericalAdmissionDecision.PASS:
        raise ValueError("score drift admission requires numerical PASS")
    if numerical_admission.policy_sha256 != numerical_policy.policy_sha256:
        raise ValueError("score drift numerical policy binding differs")
    if reference_manifest.manifest_sha256 != (
        numerical_admission.reference_manifest_sha256
    ):
        raise ValueError("reference numerical manifest binding differs")
    if candidate_manifest.manifest_sha256 != (
        numerical_admission.candidate_manifest_sha256
    ):
        raise ValueError("candidate numerical manifest binding differs")
    if reference_config.config_sha256 != numerical_admission.reference_config_sha256:
        raise ValueError("reference numerical config binding differs")
    if candidate_config.config_sha256 != numerical_admission.candidate_config_sha256:
        raise ValueError("candidate numerical config binding differs")
    if inventory.inventory_sha256 != reference_manifest.scoring_inventory_sha256 or (
        inventory.inventory_sha256 != candidate_manifest.scoring_inventory_sha256
    ) or inventory.inventory_sha256 != numerical_admission.scoring_inventory_sha256:
        raise ValueError("score drift inventory lineage differs")
    if boundary.reference_model_sha256 != reference_config.model_sha256:
        raise ValueError("frozen boundary belongs to another reference model")
    if boundary.reference_producer_config_sha256 != reference_config.config_sha256:
        raise ValueError("frozen boundary reference producer differs")
    if boundary.reference_preprocessing_sha256 != (
        reference_config.preprocessing_sha256
    ):
        raise ValueError("frozen boundary preprocessing file differs")
    if boundary.reference_preprocessing_semantics_sha256 != (
        reference_config.preprocessing_semantics_sha256
    ):
        raise ValueError("frozen boundary preprocessing semantics differ")
    if boundary.gallery_sha256 != workload.gallery_sha256:
        raise ValueError("frozen boundary belongs to another gallery")
    if boundary.pairing_policy_sha256 != workload.pairing_policy_sha256:
        raise ValueError("frozen boundary pairing policy differs")
    if boundary.retrieval_plan_sha256 != workload.retrieval_plan_sha256:
        raise ValueError("frozen boundary retrieval plan differs")
    if boundary.workload_sha256 != workload.workload_sha256:
        raise ValueError("frozen boundary exact workload differs")
    if boundary.scoring_semantics_sha256 != (
        score_drift_scoring_semantics_sha256(policy)
    ):
        raise ValueError("frozen boundary scoring semantics differ")
    reference_semantics = _producer_comparable_semantics(reference_config)
    if reference_semantics != _producer_comparable_semantics(candidate_config):
        raise ValueError("score drift producer semantics differ")
    if content_sha256(reference_semantics) != (
        numerical_admission.comparable_semantics_sha256
    ):
        raise ValueError("score drift numerical semantics binding differs")
    for name, manifest, config in (
        ("reference", reference_manifest, reference_config),
        ("candidate", candidate_manifest, candidate_config),
    ):
        expected = {
            "inference_config_sha256": config.config_sha256,
            "model_sha256": config.model_sha256,
            "dependency_lock_sha256": config.dependency_lock_sha256,
            "code_revision": config.code_revision,
            "precision": config.backend.precision,
            "vector_dimension": config.vector_dimension,
            "vector_format": config.output_vector_format,
            "normalization_tolerance": config.normalization_tolerance,
        }
        for field, value in expected.items():
            if getattr(manifest, field) != value:
                raise ValueError(f"{name} manifest/config {field} differs")
    for name, production, manifest, verification, config in (
        (
            "reference",
            reference_production,
            reference_manifest,
            reference_verification,
            reference_config,
        ),
        (
            "candidate",
            candidate_production,
            candidate_manifest,
            candidate_verification,
            candidate_config,
        ),
    ):
        if production.cache_manifest != manifest:
            raise ValueError(f"{name} production cache manifest differs")
        if production.cache_verification != verification:
            raise ValueError(f"{name} production cache verification differs")
        if production.producer_config_sha256 != config.config_sha256:
            raise ValueError(f"{name} production config binding differs")
        if production.cache_policy_sha256 != cache_policy.policy_sha256:
            raise ValueError(f"{name} production cache policy differs")
        if production.model_lineage_sha256 != config.model_lineage_sha256:
            raise ValueError(f"{name} production model lineage differs")
        if production.scoring_inventory_sha256 != inventory.inventory_sha256:
            raise ValueError(f"{name} production inventory differs")
    reference_bindings = tuple(
        (item.artifact_token, item.artifact_content_sha256)
        for item in reference_manifest.bindings
    )
    candidate_bindings = tuple(
        (item.artifact_token, item.artifact_content_sha256)
        for item in candidate_manifest.bindings
    )
    if reference_bindings != candidate_bindings:
        raise ValueError("score drift artifact bindings differ")
    unique_contents = {content for _, content in reference_bindings}
    expected_values = len(unique_contents) * reference_manifest.vector_dimension
    if (
        numerical_admission.summary.vectors != len(unique_contents)
        or numerical_admission.summary.values != expected_values
        or numerical_admission.summary.bytes_read != expected_values * 4 * 2
    ):
        raise ValueError("score drift numerical work accounting differs")
    request_tokens = {
        token
        for item in workload.requests
        for token in (item.query_artifact_token, item.candidate_artifact_token)
    }
    inventory_tokens = {item.artifact_token for item in inventory.entries}
    if request_tokens != inventory_tokens:
        raise ValueError("retrieval workload does not cover inventory exactly")
    validate_retrieval_workload_content_separation(workload, inventory)


def validate_retrieval_workload_content_separation(
    workload: RetrievalScoreWorkload,
    inventory: ControlScoringInventory,
) -> None:
    """Reject source-content aliases and query/gallery self-matches."""

    content_by_token = {
        item.artifact_token: item.content_sha256
        for item in inventory.entries
    }
    query_tokens = {
        item.query_artifact_token for item in workload.requests
    }
    candidate_by_slot = {
        item.candidate_slot_token: item.candidate_artifact_token
        for item in workload.requests
    }
    required_tokens = query_tokens | set(candidate_by_slot.values())
    missing = required_tokens - set(content_by_token)
    if missing:
        raise ValueError("retrieval workload references absent inventory artifacts")
    query_contents = [content_by_token[token] for token in query_tokens]
    candidate_contents = [
        content_by_token[token] for token in candidate_by_slot.values()
    ]
    if len(query_contents) != len(set(query_contents)):
        raise ValueError("retrieval queries contain duplicate content aliases")
    if len(candidate_contents) != len(set(candidate_contents)):
        raise ValueError("retrieval gallery contains duplicate content aliases")
    if set(query_contents) & set(candidate_contents):
        raise ValueError("retrieval workload contains a query/gallery self-match")


def verify_score_drift_receipt_external_anchors(
    receipt: ScoreDriftAdmissionReceipt,
    *,
    expected_precommitment_sha256: str,
    expected_admission_plan_sha256: str,
    expected_receipt_sha256: str,
) -> None:
    """Verify externally archived anchors after structural receipt parsing."""

    _validate_sha256(
        expected_precommitment_sha256,
        "expected_precommitment_sha256",
    )
    _validate_sha256(
        expected_admission_plan_sha256,
        "expected_admission_plan_sha256",
    )
    _validate_sha256(expected_receipt_sha256, "expected_receipt_sha256")
    if receipt.precommitment_sha256 != expected_precommitment_sha256:
        raise ValueError("score drift receipt precommitment anchor differs")
    if receipt.admission_plan_sha256 != expected_admission_plan_sha256:
        raise ValueError("score drift receipt plan anchor differs")
    if receipt.receipt_sha256 != expected_receipt_sha256:
        raise ValueError("score drift receipt result anchor differs")


def _producer_comparable_semantics(
    config: EmbeddingProducerConfig,
) -> dict[str, Any]:
    payload = config.to_dict()
    payload.pop("backend")
    payload.pop("dependency_lock_sha256")
    return payload


def _paths_by_token(
    root: Path,
    manifest: EmbeddingCacheManifest,
) -> dict[str, tuple[Path, EmbeddingCacheEntry]]:
    resolved = root.resolve(strict=True)
    entries = {item.cache_key: item for item in manifest.entries}
    return {
        binding.artifact_token: (
            resolved / entries[binding.cache_key].relative_path,
            entries[binding.cache_key],
        )
        for binding in manifest.bindings
    }


def _read_verified_payload(binding: tuple[Path, EmbeddingCacheEntry]) -> bytes:
    path, entry = binding
    before = path.stat()
    if before.st_size != entry.byte_size:
        raise ValueError("embedding vector byte size differs")
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise RuntimeError("embedding vector changed while reading")
    if hashlib.sha256(payload).hexdigest() != entry.content_sha256:
        raise ValueError("embedding vector content hash differs")
    return payload


def _dot_payloads(first: bytes, second: bytes, chunk_floats: int) -> float:
    if len(first) != len(second) or len(first) % 4:
        raise ValueError("embedding payload dimensions differ")
    total = 0.0
    compensation = 0.0
    chunk_bytes = chunk_floats * 4
    first_view = memoryview(first)
    second_view = memoryview(second)
    for offset in range(0, len(first), chunk_bytes):
        end = min(offset + chunk_bytes, len(first))
        left = struct.iter_unpack("<f", first_view[offset:end])
        right = struct.iter_unpack("<f", second_view[offset:end])
        subtotal = math.fsum(
            a[0] * b[0] for a, b in zip(left, right, strict=True)
        )
        total, compensation = _neumaier_add(
            total,
            compensation,
            subtotal,
        )
    result = total + compensation
    if not math.isfinite(result):
        raise ValueError("embedding score is non-finite")
    return result


def _group_requests(
    requests: tuple[RetrievalScoreRequest, ...],
) -> Iterator[tuple[str, tuple[RetrievalScoreRequest, ...]]]:
    start = 0
    while start < len(requests):
        token = requests[start].query_group_token
        end = start + 1
        while end < len(requests) and requests[end].query_group_token == token:
            end += 1
        yield token, requests[start:end]
        start = end


def _rank_drift(
    reference: tuple[str, ...],
    candidate: tuple[str, ...],
) -> tuple[int, int]:
    if set(reference) != set(candidate):
        raise ValueError("ranked candidate sets differ")
    candidate_positions = {token: index for index, token in enumerate(candidate)}
    sequence = [candidate_positions[token] for token in reference]
    displacement = max(
        abs(index - candidate_positions[token])
        for index, token in enumerate(reference)
    )
    return _count_inversions(sequence), displacement


def _top_k_drift(
    reference: tuple[str, ...],
    candidate: tuple[str, ...],
    top_k: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    requested = set(top_k)
    symmetric_difference: set[str] = set()
    differences: list[tuple[int, int]] = []
    for index in range(top_k[-1]):
        for token in (reference[index], candidate[index]):
            if token in symmetric_difference:
                symmetric_difference.remove(token)
            else:
                symmetric_difference.add(token)
        current_k = index + 1
        if current_k in requested:
            differences.append((current_k, len(symmetric_difference)))
    return tuple(differences)


def _count_inversions(values: list[int]) -> int:
    if len(values) < 2:
        return 0
    source = list(values)
    target = [0] * len(values)
    width = 1
    inversions = 0
    while width < len(source):
        for start in range(0, len(source), 2 * width):
            middle = min(start + width, len(source))
            end = min(start + 2 * width, len(source))
            left = start
            right = middle
            output = start
            while left < middle and right < end:
                if source[left] <= source[right]:
                    target[output] = source[left]
                    left += 1
                else:
                    target[output] = source[right]
                    right += 1
                    inversions += middle - left
                output += 1
            while left < middle:
                target[output] = source[left]
                left += 1
                output += 1
            while right < end:
                target[output] = source[right]
                right += 1
                output += 1
        source, target = target, source
        width *= 2
    return inversions


def _tie_pairs(scores: Iterator[float]) -> int:
    return sum(count * (count - 1) // 2 for count in Counter(scores).values())


def _update_score_digest(digest: Any, request_id: str, score: float) -> None:
    digest.update(bytes.fromhex(request_id))
    digest.update(struct.pack(">d", score))


def _neumaier_add(
    total: float,
    compensation: float,
    value: float,
) -> tuple[float, float]:
    updated = total + value
    if abs(total) >= abs(value):
        compensation += (total - updated) + value
    else:
        compensation += (value - updated) + total
    return updated, compensation


def _hard_failures(
    summary: ScoreDriftSummary,
    policy: ScoreDriftPolicy,
) -> tuple[str, ...]:
    failures: list[str] = []
    checks = (
        (
            summary.maximum_absolute_score_drift
            > policy.maximum_absolute_score_drift,
            "MAXIMUM_SCORE_DRIFT",
        ),
        (
            summary.mean_absolute_score_drift
            > policy.maximum_mean_absolute_score_drift,
            "MEAN_SCORE_DRIFT",
        ),
        (
            summary.maximum_absolute_margin_drift
            > policy.maximum_absolute_margin_drift,
            "MAXIMUM_MARGIN_DRIFT",
        ),
        (
            summary.mean_absolute_margin_drift
            > policy.maximum_mean_absolute_margin_drift,
            "MEAN_MARGIN_DRIFT",
        ),
        (summary.rank_inversions > policy.maximum_rank_inversions, "RANK_INVERSIONS"),
        (
            summary.queries_with_rank_change
            > policy.maximum_queries_with_rank_change,
            "RANK_CHANGED_QUERIES",
        ),
        (
            summary.maximum_rank_displacement
            > policy.maximum_rank_displacement,
            "MAXIMUM_RANK_DISPLACEMENT",
        ),
        (summary.top1_changes > policy.maximum_top1_changes, "TOP1_CHANGES"),
        (
            summary.top_k_set_changes > policy.maximum_top_k_set_changes,
            "TOP_K_SET_CHANGES",
        ),
        (
            summary.top_k_symmetric_difference_items
            > policy.maximum_top_k_symmetric_difference_items,
            "TOP_K_SYMMETRIC_DIFFERENCE_ITEMS",
        ),
        (
            summary.threshold_decision_flips
            > policy.maximum_threshold_decision_flips,
            "THRESHOLD_DECISION_FLIPS",
        ),
        (
            summary.reference_reject_candidate_accept_flips
            > policy.maximum_reference_reject_candidate_accept_flips,
            "REFERENCE_REJECT_CANDIDATE_ACCEPT_FLIPS",
        ),
        (
            summary.reference_accept_candidate_reject_flips
            > policy.maximum_reference_accept_candidate_reject_flips,
            "REFERENCE_ACCEPT_CANDIDATE_REJECT_FLIPS",
        ),
    )
    failures.extend(code for failed, code in checks if failed)
    failures.extend(
        f"TOP_K_QUERIES_WITH_SET_CHANGE_AT_INDEX_{index}"
        for index, (observed, maximum) in enumerate(
            zip(
                summary.top_k_queries_with_set_change,
                policy.maximum_top_k_queries_with_set_change_by_k,
                strict=True,
            )
        )
        if observed > maximum
    )
    failures.extend(
        f"TOP_K_SYMMETRIC_DIFFERENCE_ITEMS_AT_INDEX_{index}"
        for index, (observed, maximum) in enumerate(
            zip(
                summary.top_k_symmetric_difference_items_by_k,
                policy.maximum_top_k_symmetric_difference_items_by_k,
                strict=True,
            )
        )
        if observed > maximum
    )
    return tuple(sorted(failures))


def _validate_top_k_policy(
    policy: ScoreDriftPolicy,
    boundary: FrozenScoreMarginBoundary,
) -> None:
    expected = len(boundary.top_k)
    if len(policy.maximum_top_k_queries_with_set_change_by_k) != expected:
        raise ValueError("per-K query drift caps differ from frozen Top-K")
    if (
        len(policy.maximum_top_k_symmetric_difference_items_by_k)
        != expected
    ):
        raise ValueError("per-K item drift caps differ from frozen Top-K")


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    if set(payload) != expected:
        raise ValueError(f"{context} keys mismatch")


def _require_nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _validate_sha256(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")


def _require_finite(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")


def _require_finite_nonnegative(value: Any, name: str) -> None:
    _require_finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_finite_positive(value: Any, name: str) -> None:
    _require_finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
