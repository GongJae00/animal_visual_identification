"""Exact, score-blind open-set threshold calibration contracts.

This module receives opaque query and distinct-gallery-identity scores only.
Development labels, known-query labels, test roles, and registered dog IDs are
outside the API.  The calibration threshold is a one-sided nonparametric
order-statistic tolerance bound, not a threshold grid search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from foundation.provenance import content_sha256
from identity.splits.protected_public_split import ProtectedPublicSplitPolicy


class OpenSetDisposition(StrEnum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True, slots=True)
class DistinctIdentityScore:
    identity_slot_token: str
    score: float
    schema_version: str = "cvi.distinct_identity_score.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.distinct_identity_score.v1":
            raise ValueError("unsupported distinct identity score schema")
        _sha256(self.identity_slot_token, "identity_slot_token")
        _cosine(self.score, "score")

    def to_dict(self) -> dict[str, str | float]:
        return {
            "schema_version": self.schema_version,
            "identity_slot_token": self.identity_slot_token,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DistinctIdentityScore":
        _exact_keys(payload, set(cls.__dataclass_fields__), "distinct identity score")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class BlindOpenSetScoreRow:
    query_token: str
    gallery_size: int
    shot: int
    scores: tuple[DistinctIdentityScore, ...]
    score_semantics: str = "DISTINCT_GALLERY_IDENTITY_AGGREGATED_COSINE"
    schema_version: str = "cvi.blind_open_set_score_row.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.blind_open_set_score_row.v1":
            raise ValueError("unsupported blind open-set score row schema")
        _sha256(self.query_token, "query_token")
        _positive_int(self.gallery_size, "gallery_size")
        _positive_int(self.shot, "shot")
        if self.gallery_size < 2:
            raise ValueError("open-set gallery must contain at least two identities")
        if len(self.scores) != self.gallery_size:
            raise ValueError("score row cardinality differs from gallery size")
        if any(not isinstance(item, DistinctIdentityScore) for item in self.scores):
            raise TypeError("score row entries must be DistinctIdentityScore")
        tokens = tuple(item.identity_slot_token for item in self.scores)
        if tokens != tuple(sorted(tokens)) or len(tokens) != len(set(tokens)):
            raise ValueError("identity score slots must be unique and canonically sorted")
        if self.score_semantics != "DISTINCT_GALLERY_IDENTITY_AGGREGATED_COSINE":
            raise ValueError("open-set score semantics differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_token": self.query_token,
            "gallery_size": self.gallery_size,
            "shot": self.shot,
            "scores": [item.to_dict() for item in self.scores],
            "score_semantics": self.score_semantics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlindOpenSetScoreRow":
        _exact_keys(payload, set(cls.__dataclass_fields__), "blind open-set score row")
        if not isinstance(payload["scores"], list):
            raise TypeError("open-set scores must be a list")
        values = dict(payload)
        values["scores"] = tuple(
            DistinctIdentityScore.from_dict(item) for item in values["scores"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class OpenSetCalibrationPolicy:
    target_fpir: float = 0.01
    one_sided_alpha: float = 0.05
    calibration_unknown_identities: int = 300
    registered_gallery_sizes: tuple[int, ...] = (39, 64, 100, 300)
    registered_shots: tuple[int, ...] = (1, 3)
    primary_gallery_size: int = 300
    primary_shot: int = 3
    margin_rule: str = "TOP1_MINUS_NEXT_DISTINCT_IDENTITY_GREATER_THAN_OR_EQUAL"
    tie_rule: str = "DISTINCT_IDENTITY_TOP_TIE_REVIEW_REQUIRED"
    threshold_rule: str = "EXACT_BINOMIAL_ORDER_STATISTIC_NEXTAFTER"
    threshold_interpolation: str = "PROHIBITED_RECALIBRATION_REQUIRED"
    schema_version: str = "cvi.open_set_calibration_policy.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.open_set_calibration_policy.v2":
            raise ValueError("unsupported open-set calibration policy schema")
        _probability(self.target_fpir, "target_fpir")
        _probability(self.one_sided_alpha, "one_sided_alpha")
        _positive_int(
            self.calibration_unknown_identities,
            "calibration_unknown_identities",
        )
        for values, name in (
            (self.registered_gallery_sizes, "registered_gallery_sizes"),
            (self.registered_shots, "registered_shots"),
        ):
            if (
                not values
                or values != tuple(sorted(set(values)))
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values)
            ):
                raise ValueError(f"{name} must be unique sorted positive integers")
        if self.primary_gallery_size not in self.registered_gallery_sizes:
            raise ValueError("primary gallery size is not registered")
        if self.primary_shot not in self.registered_shots:
            raise ValueError("primary shot is not registered")
        expected_strings = {
            "margin_rule": "TOP1_MINUS_NEXT_DISTINCT_IDENTITY_GREATER_THAN_OR_EQUAL",
            "tie_rule": "DISTINCT_IDENTITY_TOP_TIE_REVIEW_REQUIRED",
            "threshold_rule": "EXACT_BINOMIAL_ORDER_STATISTIC_NEXTAFTER",
            "threshold_interpolation": "PROHIBITED_RECALIBRATION_REQUIRED",
        }
        for name, expected in expected_strings.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} differs from the frozen rule")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values["registered_gallery_sizes"] = list(self.registered_gallery_sizes)
        values["registered_shots"] = list(self.registered_shots)
        return values

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OpenSetCalibrationPolicy":
        _exact_keys(payload, set(cls.__dataclass_fields__), "open-set policy")
        values = dict(payload)
        for name in ("registered_gallery_sizes", "registered_shots"):
            if not isinstance(values[name], list):
                raise TypeError(f"{name} must be a list")
            values[name] = tuple(values[name])
        candidate = cls(**values)
        if candidate != cls():
            raise ValueError("open-set calibration policy constants differ")
        return candidate


@dataclass(frozen=True, slots=True)
class AuthenticatedOpenSetCalibrationPanel:
    split_assignment_sha256: str
    split_policy_sha256: str
    gallery_size: int
    shot: int
    gallery_identity_slot_tokens: tuple[str, ...]
    unknown_query_event_tokens: tuple[str, ...]
    protocol: str = "YT_CALIBRATION_OPEN_SET"
    episode: str = "N_300"
    schema_version: str = "cvi.authenticated_open_set_calibration_panel.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.authenticated_open_set_calibration_panel.v1":
            raise ValueError("unsupported authenticated calibration panel schema")
        for value, name in (
            (self.split_assignment_sha256, "split_assignment_sha256"),
            (self.split_policy_sha256, "split_policy_sha256"),
        ):
            _sha256(value, name)
        _positive_int(self.gallery_size, "gallery_size")
        _positive_int(self.shot, "shot")
        for values, name, expected in (
            (
                self.gallery_identity_slot_tokens,
                "gallery_identity_slot_tokens",
                self.gallery_size,
            ),
            (
                self.unknown_query_event_tokens,
                "unknown_query_event_tokens",
                OpenSetCalibrationPolicy().calibration_unknown_identities,
            ),
        ):
            if values != tuple(sorted(set(values))) or len(values) != expected:
                raise ValueError(f"{name} cardinality or canonical order differs")
            for value in values:
                _sha256(value, name)
        if self.protocol != "YT_CALIBRATION_OPEN_SET":
            raise ValueError("calibration panel protocol differs")
        if self.episode != f"N_{self.gallery_size}":
            raise ValueError("calibration panel episode differs from gallery size")

    @property
    def gallery_identity_set_sha256(self) -> str:
        return content_sha256({
            "schema_version": "cvi.gallery_identity_set.v1",
            "identity_slot_tokens": list(self.gallery_identity_slot_tokens),
        })

    @property
    def query_event_set_sha256(self) -> str:
        return content_sha256({
            "schema_version": "cvi.calibration_unknown_query_event_set.v1",
            "query_event_tokens": list(self.unknown_query_event_tokens),
        })

    @property
    def panel_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_assignment_sha256": self.split_assignment_sha256,
            "split_policy_sha256": self.split_policy_sha256,
            "gallery_size": self.gallery_size,
            "shot": self.shot,
            "gallery_identity_slot_tokens": list(
                self.gallery_identity_slot_tokens
            ),
            "unknown_query_event_tokens": list(
                self.unknown_query_event_tokens
            ),
            "protocol": self.protocol,
            "episode": self.episode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AuthenticatedOpenSetCalibrationPanel":
        _exact_keys(payload, set(cls.__dataclass_fields__), "authenticated calibration panel")
        values = dict(payload)
        for name in ("gallery_identity_slot_tokens", "unknown_query_event_tokens"):
            if not isinstance(values[name], list):
                raise TypeError(f"{name} must be a list")
            values[name] = tuple(values[name])
        return cls(**values)


def authenticate_open_set_calibration_panel(
    assignment: dict[str, Any],
    *,
    split_policy: ProtectedPublicSplitPolicy,
    calibration_policy: OpenSetCalibrationPolicy,
    gallery_size: int,
    shot: int,
) -> AuthenticatedOpenSetCalibrationPanel:
    """Derive an exact label-free calibration panel from a protected assignment."""

    if split_policy != ProtectedPublicSplitPolicy():
        raise ValueError("protected split policy is not frozen")
    if calibration_policy != OpenSetCalibrationPolicy():
        raise ValueError("open-set calibration policy is not frozen")
    _validate_policy_compatibility(split_policy, calibration_policy)
    expected_assignment_keys = {
        "schema_version", "status", "seed_commitment", "evidence_root_sha256",
        "policy_sha256", "strict_external_boundary", "score_inputs_used",
        "label_fields_present", "capacity", "protocol_cohorts", "records",
        "interpretation",
    }
    _exact_keys(assignment, expected_assignment_keys, "protected split assignment")
    if assignment["schema_version"] != "cvi.protected_public_split_assignment.v1":
        raise ValueError("protected split assignment schema differs")
    if assignment["status"] != "PASS_PROTECTED_SPLIT_CONSTRUCTION":
        raise ValueError("protected split assignment is not passing")
    if assignment["policy_sha256"] != split_policy.policy_sha256:
        raise ValueError("protected split policy hash differs")
    if assignment["score_inputs_used"] is not False or assignment["label_fields_present"] is not False:
        raise ValueError("protected split assignment is not score-blind and label-free")
    if not isinstance(assignment["records"], list):
        raise TypeError("protected split records must be a list")
    if gallery_size not in calibration_policy.registered_gallery_sizes:
        raise ValueError("RECALIBRATION_REQUIRED: gallery size is not registered")
    if shot not in calibration_policy.registered_shots:
        raise ValueError("RECALIBRATION_REQUIRED: shot is not registered")

    gallery_counts: dict[str, int] = {}
    known_queries: dict[str, str] = {}
    unknown_queries: dict[str, str] = {}
    episode = f"N_{gallery_size}"
    record_keys = {
        "sample_token", "identity_token", "component_token", "dataset_name",
        "source_variant", "identity_role", "model_access", "sample_disposition",
        "paired_original_token", "uses",
    }
    use_keys = {
        "protocol", "episode", "gallery_size", "shot", "role", "event_token",
        "primary_query_event_token", "bootstrap_cluster_token",
    }
    for record in assignment["records"]:
        _exact_keys(record, record_keys, "protected split record")
        _sha256(record["sample_token"], "sample_token")
        _sha256(record["identity_token"], "identity_token")
        if not isinstance(record["uses"], list):
            raise TypeError("protected split uses must be a list")
        for use in record["uses"]:
            _exact_keys(use, use_keys, "protected split use")
            if use["protocol"] != "YT_CALIBRATION_OPEN_SET":
                continue
            if use["gallery_size"] != gallery_size or use["shot"] != shot:
                continue
            if use["episode"] != episode:
                raise ValueError("calibration use episode differs")
            _sha256(use["event_token"], "event_token")
            role = use["role"]
            identity = record["identity_token"]
            if role == "GALLERY":
                if record["identity_role"] != "YT_CALIBRATION_KNOWN":
                    raise ValueError("calibration-unknown identity entered gallery")
                if use["primary_query_event_token"] is not None or use["bootstrap_cluster_token"] is not None:
                    raise ValueError("gallery use carries query-only tokens")
                gallery_counts[identity] = gallery_counts.get(identity, 0) + 1
            elif role in {"KNOWN_QUERY", "UNKNOWN_QUERY"}:
                event = use["primary_query_event_token"]
                cluster = use["bootstrap_cluster_token"]
                _sha256(event, "primary_query_event_token")
                _sha256(cluster, "bootstrap_cluster_token")
                target = known_queries if role == "KNOWN_QUERY" else unknown_queries
                expected_role = (
                    "YT_CALIBRATION_KNOWN"
                    if role == "KNOWN_QUERY"
                    else "YT_CALIBRATION_UNKNOWN"
                )
                if record["identity_role"] != expected_role or identity in target:
                    raise ValueError("calibration query role or uniqueness differs")
                target[identity] = event
            else:
                raise ValueError("calibration use role differs")

    gallery_identities = set(gallery_counts)
    if (
        len(gallery_identities) != gallery_size
        or any(value != shot for value in gallery_counts.values())
        or gallery_identities != set(known_queries)
        or len(unknown_queries) != calibration_policy.calibration_unknown_identities
        or gallery_identities & set(unknown_queries)
    ):
        raise ValueError("calibration gallery/query panel capacity differs")
    return AuthenticatedOpenSetCalibrationPanel(
        split_assignment_sha256=content_sha256(assignment),
        split_policy_sha256=split_policy.policy_sha256,
        gallery_size=gallery_size,
        shot=shot,
        gallery_identity_slot_tokens=tuple(sorted(gallery_identities)),
        unknown_query_event_tokens=tuple(sorted(unknown_queries.values())),
        episode=episode,
    )


def _validate_policy_compatibility(
    split_policy: ProtectedPublicSplitPolicy,
    calibration_policy: OpenSetCalibrationPolicy,
) -> None:
    if (
        split_policy.yt_calibration_gallery_sizes
        != calibration_policy.registered_gallery_sizes
        or split_policy.shot_counts != calibration_policy.registered_shots
        or split_policy.yt_primary_open_set_gallery_size
        != calibration_policy.primary_gallery_size
        or split_policy.yt_primary_open_set_shot != calibration_policy.primary_shot
        or split_policy.yt_calibration_unknown_identities
        != calibration_policy.calibration_unknown_identities
    ):
        raise ValueError("protected split and open-set calibration policies differ")


@dataclass(frozen=True, slots=True)
class TopIdentityEvidence:
    top_identity_slot_token: str | None
    top1_score: float
    top2_score: float
    margin: float
    unique_top_identity: bool
    margin_passed: bool
    effective_score: float | None


@dataclass(frozen=True, slots=True)
class FrozenOpenSetBoundary:
    gallery_size: int
    shot: int
    margin_threshold: float
    score_threshold: float
    automatic_accept_enabled: bool
    allowed_calibration_accepts: int
    calibration_unknown_identities: int
    target_fpir: float
    one_sided_alpha: float
    policy_sha256: str
    split_assignment_sha256: str
    split_policy_sha256: str
    gallery_identity_set_sha256: str
    query_event_set_sha256: str
    calibration_panel_sha256: str
    margin_selection_receipt_sha256: str
    calibration_score_receipt_sha256: str
    model_sha256: str
    preprocessing_sha256: str
    scoring_semantics_sha256: str
    precision: str
    score_dtype: str
    threshold_rule: str = "EFFECTIVE_SCORE_GREATER_THAN_OR_EQUAL"
    schema_version: str = "cvi.frozen_open_set_boundary.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.frozen_open_set_boundary.v2":
            raise ValueError("unsupported frozen open-set boundary schema")
        for value, name in ((self.gallery_size, "gallery_size"), (self.shot, "shot")):
            _positive_int(value, name)
        _finite(self.margin_threshold, "margin_threshold")
        if not 0.0 < self.margin_threshold <= 2.0:
            raise ValueError("margin threshold must be in (0, 2]")
        _finite(self.score_threshold, "score_threshold")
        if not isinstance(self.automatic_accept_enabled, bool):
            raise TypeError("automatic_accept_enabled must be boolean")
        for value, name in (
            (self.allowed_calibration_accepts, "allowed_calibration_accepts"),
            (self.calibration_unknown_identities, "calibration_unknown_identities"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _probability(self.target_fpir, "target_fpir")
        _probability(self.one_sided_alpha, "one_sided_alpha")
        for name in (
            "policy_sha256",
            "split_assignment_sha256",
            "split_policy_sha256",
            "gallery_identity_set_sha256",
            "query_event_set_sha256",
            "calibration_panel_sha256",
            "margin_selection_receipt_sha256",
            "calibration_score_receipt_sha256",
            "model_sha256",
            "preprocessing_sha256",
            "scoring_semantics_sha256",
        ):
            _sha256(getattr(self, name), name)
        policy = OpenSetCalibrationPolicy()
        if (
            self.policy_sha256 != policy.policy_sha256
            or self.gallery_size not in policy.registered_gallery_sizes
            or self.shot not in policy.registered_shots
            or self.calibration_unknown_identities
            != policy.calibration_unknown_identities
            or self.target_fpir != policy.target_fpir
            or self.one_sided_alpha != policy.one_sided_alpha
        ):
            raise ValueError("frozen open-set boundary differs from fixed policy")
        expected_allowed = maximum_allowed_calibration_accepts(
            trials=self.calibration_unknown_identities,
            target_fpir=self.target_fpir,
            one_sided_alpha=self.one_sided_alpha,
        )
        if (
            expected_allowed is None
            or self.allowed_calibration_accepts != expected_allowed
        ):
            raise ValueError("allowed calibration accepts differ from exact rule")
        if self.automatic_accept_enabled != (self.score_threshold <= 1.0):
            raise ValueError("automatic accept state differs from cosine threshold")
        for name in ("precision", "score_dtype"):
            _text(getattr(self, name), name, 64)
        if self.threshold_rule != "EFFECTIVE_SCORE_GREATER_THAN_OR_EQUAL":
            raise ValueError("open-set threshold rule differs")

    @property
    def boundary_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrozenOpenSetBoundary":
        _exact_keys(payload, set(cls.__dataclass_fields__), "frozen open-set boundary")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class OpenSetCalibrationSummary:
    unknown_identity_events: int
    unique_top_events: int
    margin_pass_events: int
    top_tie_events: int
    observed_calibration_accepts: int
    allowed_calibration_accepts: int
    order_statistic_rank_one_based: int
    order_statistic_value: float | str
    zero_event_fpir_upper_bound: float

    def __post_init__(self) -> None:
        for name in (
            "unknown_identity_events",
            "unique_top_events",
            "margin_pass_events",
            "top_tie_events",
            "observed_calibration_accepts",
            "order_statistic_rank_one_based",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.allowed_calibration_accepts, bool)
            or not isinstance(self.allowed_calibration_accepts, int)
            or self.allowed_calibration_accepts < -1
        ):
            raise ValueError("allowed_calibration_accepts must be at least -1")
        if self.unique_top_events + self.top_tie_events > self.unknown_identity_events:
            raise ValueError("open-set event counts exceed unknown identities")
        if self.margin_pass_events > self.unique_top_events:
            raise ValueError("margin-pass events exceed unique-top events")
        if self.observed_calibration_accepts > self.margin_pass_events:
            raise ValueError("calibration accepts exceed margin-pass events")
        if self.allowed_calibration_accepts > self.unknown_identity_events:
            raise ValueError("allowed accepts exceed unknown identities")
        if (
            self.allowed_calibration_accepts >= 0
            and self.observed_calibration_accepts
            > self.allowed_calibration_accepts
        ):
            raise ValueError("observed accepts exceed exact allowance")
        if not isinstance(self.order_statistic_value, (float, str)):
            raise TypeError("order_statistic_value must be float or sentinel")
        if isinstance(self.order_statistic_value, float) and not (
            math.isfinite(self.order_statistic_value)
            and -1.0 <= self.order_statistic_value <= 1.0
        ):
            raise ValueError("order statistic float must be a finite cosine score")
        if isinstance(self.order_statistic_value, str) and self.order_statistic_value not in {
            "NEGATIVE_INFINITY",
            "UNAVAILABLE",
        }:
            raise ValueError("order statistic sentinel differs")
        _unit_interval(
            self.zero_event_fpir_upper_bound,
            "zero_event_fpir_upper_bound",
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OpenSetCalibrationSummary":
        _exact_keys(payload, set(cls.__dataclass_fields__), "open-set calibration summary")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class OpenSetCalibrationReceipt:
    status: str
    boundary: FrozenOpenSetBoundary | None
    summary: OpenSetCalibrationSummary
    panel: AuthenticatedOpenSetCalibrationPanel
    interpretation: str = (
        "CALIBRATION_UNKNOWN_ORDER_STATISTIC_ONLY_NOT_TEST_OR_PERFORMANCE_EVIDENCE"
    )
    schema_version: str = "cvi.open_set_calibration_receipt.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.open_set_calibration_receipt.v2":
            raise ValueError("unsupported open-set calibration receipt schema")
        if not isinstance(self.panel, AuthenticatedOpenSetCalibrationPanel):
            raise TypeError("calibration receipt requires an authenticated panel")
        if self.status == "PASS_EXACT_OPEN_SET_CALIBRATION":
            if self.boundary is None:
                raise ValueError("passing calibration receipt requires a boundary")
            if (
                self.summary.unknown_identity_events
                != self.boundary.calibration_unknown_identities
                or self.summary.allowed_calibration_accepts
                != self.boundary.allowed_calibration_accepts
            ):
                raise ValueError("calibration summary and boundary differ")
            if (
                self.boundary.split_assignment_sha256
                != self.panel.split_assignment_sha256
                or self.boundary.split_policy_sha256
                != self.panel.split_policy_sha256
                or self.boundary.gallery_identity_set_sha256
                != self.panel.gallery_identity_set_sha256
                or self.boundary.query_event_set_sha256
                != self.panel.query_event_set_sha256
                or self.boundary.calibration_panel_sha256
                != self.panel.panel_sha256
                or self.boundary.gallery_size != self.panel.gallery_size
                or self.boundary.shot != self.panel.shot
            ):
                raise ValueError("calibration boundary and authenticated panel differ")
            if (
                self.summary.unique_top_events + self.summary.top_tie_events
                != self.summary.unknown_identity_events
                or self.summary.order_statistic_rank_one_based
                != self.summary.unknown_identity_events
                - self.summary.allowed_calibration_accepts
            ):
                raise ValueError("passing calibration summary event counts differ")
            expected_upper = zero_event_one_sided_upper_bound(
                trials=self.summary.unknown_identity_events,
                alpha=self.boundary.one_sided_alpha,
            )
            if self.summary.zero_event_fpir_upper_bound != expected_upper:
                raise ValueError("calibration upper bound differs from exact rule")
            if self.summary.order_statistic_value == "UNAVAILABLE":
                raise ValueError("passing calibration requires an order statistic")
            order_value = (
                -math.inf
                if self.summary.order_statistic_value == "NEGATIVE_INFINITY"
                else self.summary.order_statistic_value
            )
            assert isinstance(order_value, float)
            if self.boundary.score_threshold != math.nextafter(
                order_value, math.inf
            ):
                raise ValueError("calibration threshold differs from order statistic")
        elif self.status == "CALIBRATION_CAPACITY_FAILED":
            if self.boundary is not None:
                raise ValueError("failed calibration receipt cannot contain a boundary")
        else:
            raise ValueError("open-set calibration status differs")
        if self.interpretation != (
            "CALIBRATION_UNKNOWN_ORDER_STATISTIC_ONLY_NOT_TEST_OR_PERFORMANCE_EVIDENCE"
        ):
            raise ValueError("open-set calibration interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "boundary": self.boundary.to_dict() if self.boundary is not None else None,
            "summary": self.summary.to_dict(),
            "panel": self.panel.to_dict(),
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OpenSetCalibrationReceipt":
        _exact_keys(payload, set(cls.__dataclass_fields__), "open-set calibration receipt")
        values = dict(payload)
        if values["boundary"] is not None:
            values["boundary"] = FrozenOpenSetBoundary.from_dict(values["boundary"])
        values["summary"] = OpenSetCalibrationSummary.from_dict(values["summary"])
        values["panel"] = AuthenticatedOpenSetCalibrationPanel.from_dict(values["panel"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class OpenSetDispositionResult:
    query_token: str
    disposition: OpenSetDisposition
    predicted_identity_slot_token: str | None
    top1_score: float
    top2_score: float
    margin: float
    boundary_sha256: str


def top_identity_evidence(
    row: BlindOpenSetScoreRow,
    *,
    margin_threshold: float,
) -> TopIdentityEvidence:
    _finite(margin_threshold, "margin_threshold")
    if not 0.0 < margin_threshold <= 2.0:
        raise ValueError("margin threshold must be in (0, 2]")
    ranked = sorted(
        row.scores,
        key=lambda item: (-item.score, item.identity_slot_token),
    )
    top1, top2 = ranked[0], ranked[1]
    unique = top1.score > top2.score
    margin = top1.score - top2.score
    margin_passed = unique and margin >= margin_threshold
    return TopIdentityEvidence(
        top_identity_slot_token=top1.identity_slot_token if unique else None,
        top1_score=top1.score,
        top2_score=top2.score,
        margin=margin,
        unique_top_identity=unique,
        margin_passed=margin_passed,
        effective_score=top1.score if margin_passed else None,
    )


def maximum_allowed_calibration_accepts(
    *,
    trials: int,
    target_fpir: float,
    one_sided_alpha: float,
) -> int | None:
    """Return the greatest binomial count whose lower-tail mass is <= alpha."""

    _positive_int(trials, "trials")
    _probability(target_fpir, "target_fpir")
    _probability(one_sided_alpha, "one_sided_alpha")
    log_alpha = math.log(one_sided_alpha)
    log_pmf = trials * math.log1p(-target_fpir)
    log_cdf = log_pmf
    allowed: int | None = 0 if log_cdf <= log_alpha else None
    odds = target_fpir / (1.0 - target_fpir)
    for count in range(1, trials + 1):
        log_pmf += (
            math.log(trials - count + 1)
            - math.log(count)
            + math.log(odds)
        )
        log_cdf = _logaddexp(log_cdf, log_pmf)
        if log_cdf <= log_alpha:
            allowed = count
        else:
            break
    return allowed


def zero_event_one_sided_upper_bound(*, trials: int, alpha: float) -> float:
    _positive_int(trials, "trials")
    _probability(alpha, "alpha")
    return -math.expm1(math.log(alpha) / trials)


def freeze_open_set_threshold(
    rows: tuple[BlindOpenSetScoreRow, ...],
    *,
    policy: OpenSetCalibrationPolicy,
    panel: AuthenticatedOpenSetCalibrationPanel,
    margin_threshold: float,
    margin_selection_receipt_sha256: str,
    calibration_score_receipt_sha256: str,
    model_sha256: str,
    preprocessing_sha256: str,
    scoring_semantics_sha256: str,
    precision: str,
    score_dtype: str,
) -> OpenSetCalibrationReceipt:
    """Freeze one registered N/k threshold from calibration-unknown rows only."""

    if policy != OpenSetCalibrationPolicy():
        raise ValueError("open-set calibration policy is not the frozen protocol")
    if not isinstance(panel, AuthenticatedOpenSetCalibrationPanel):
        raise TypeError("authenticated calibration panel is required")
    gallery_size = panel.gallery_size
    shot = panel.shot
    if gallery_size not in policy.registered_gallery_sizes:
        raise ValueError("RECALIBRATION_REQUIRED: gallery size is not registered")
    if shot not in policy.registered_shots:
        raise ValueError("RECALIBRATION_REQUIRED: shot is not registered")
    if (gallery_size, shot) != (policy.primary_gallery_size, policy.primary_shot):
        raise ValueError(
            "FAMILYWISE_ALLOCATION_REQUIRED: only the primary N/k safety boundary may freeze"
        )
    _finite(margin_threshold, "margin_threshold")
    if not 0.0 < margin_threshold <= 2.0:
        raise ValueError("margin threshold must be in (0, 2]")
    for name, digest in (
        ("margin_selection_receipt_sha256", margin_selection_receipt_sha256),
        ("calibration_score_receipt_sha256", calibration_score_receipt_sha256),
        ("model_sha256", model_sha256),
        ("preprocessing_sha256", preprocessing_sha256),
        ("scoring_semantics_sha256", scoring_semantics_sha256),
    ):
        _sha256(digest, name)
    for value, name in ((precision, "precision"), (score_dtype, "score_dtype")):
        _text(value, name, 64)
    query_tokens = tuple(row.query_token for row in rows)
    if query_tokens != tuple(sorted(query_tokens)) or len(query_tokens) != len(set(query_tokens)):
        raise ValueError("calibration rows must have unique canonically sorted queries")
    if any(row.gallery_size != gallery_size or row.shot != shot for row in rows):
        raise ValueError("calibration row N/k differs from requested boundary")
    if not set(query_tokens) <= set(panel.unknown_query_event_tokens):
        raise ValueError("calibration score queries differ from authenticated panel")
    if (
        len(rows) == policy.calibration_unknown_identities
        and set(query_tokens) != set(panel.unknown_query_event_tokens)
    ):
        raise ValueError("complete calibration query set differs from authenticated panel")
    expected_slots = panel.gallery_identity_slot_tokens
    if any(
        tuple(item.identity_slot_token for item in row.scores) != expected_slots
        for row in rows
    ):
        raise ValueError("calibration score gallery slots differ from authenticated panel")

    allowed = maximum_allowed_calibration_accepts(
        trials=len(rows) if rows else 1,
        target_fpir=policy.target_fpir,
        one_sided_alpha=policy.one_sided_alpha,
    )
    if len(rows) != policy.calibration_unknown_identities or allowed is None:
        summary = OpenSetCalibrationSummary(
            unknown_identity_events=len(rows),
            unique_top_events=0,
            margin_pass_events=0,
            top_tie_events=0,
            observed_calibration_accepts=0,
            allowed_calibration_accepts=-1 if allowed is None else allowed,
            order_statistic_rank_one_based=0,
            order_statistic_value="UNAVAILABLE",
            zero_event_fpir_upper_bound=(
                zero_event_one_sided_upper_bound(
                    trials=len(rows), alpha=policy.one_sided_alpha
                )
                if rows
                else 1.0
            ),
        )
        return OpenSetCalibrationReceipt(
            status="CALIBRATION_CAPACITY_FAILED",
            boundary=None,
            summary=summary,
            panel=panel,
        )

    evidence = tuple(
        top_identity_evidence(row, margin_threshold=margin_threshold) for row in rows
    )
    effective = sorted(
        value.effective_score if value.effective_score is not None else -math.inf
        for value in evidence
    )
    rank = len(effective) - allowed
    order_value = effective[rank - 1]
    score_threshold = math.nextafter(order_value, math.inf)
    automatic_enabled = math.isfinite(score_threshold) and score_threshold <= 1.0
    if not math.isfinite(score_threshold):
        score_threshold = math.nextafter(1.0, math.inf)
    observed_accepts = sum(
        value.effective_score is not None
        and value.effective_score >= score_threshold
        for value in evidence
    )
    if observed_accepts > allowed:
        raise RuntimeError("order-statistic threshold exceeds allowed accepts")

    boundary = FrozenOpenSetBoundary(
        gallery_size=gallery_size,
        shot=shot,
        margin_threshold=margin_threshold,
        score_threshold=score_threshold,
        automatic_accept_enabled=automatic_enabled,
        allowed_calibration_accepts=allowed,
        calibration_unknown_identities=len(rows),
        target_fpir=policy.target_fpir,
        one_sided_alpha=policy.one_sided_alpha,
        policy_sha256=policy.policy_sha256,
        split_assignment_sha256=panel.split_assignment_sha256,
        split_policy_sha256=panel.split_policy_sha256,
        gallery_identity_set_sha256=panel.gallery_identity_set_sha256,
        query_event_set_sha256=panel.query_event_set_sha256,
        calibration_panel_sha256=panel.panel_sha256,
        margin_selection_receipt_sha256=margin_selection_receipt_sha256,
        calibration_score_receipt_sha256=calibration_score_receipt_sha256,
        model_sha256=model_sha256,
        preprocessing_sha256=preprocessing_sha256,
        scoring_semantics_sha256=scoring_semantics_sha256,
        precision=precision,
        score_dtype=score_dtype,
    )
    summary = OpenSetCalibrationSummary(
        unknown_identity_events=len(rows),
        unique_top_events=sum(value.unique_top_identity for value in evidence),
        margin_pass_events=sum(value.margin_passed for value in evidence),
        top_tie_events=sum(not value.unique_top_identity for value in evidence),
        observed_calibration_accepts=observed_accepts,
        allowed_calibration_accepts=allowed,
        order_statistic_rank_one_based=rank,
        order_statistic_value=(
            "NEGATIVE_INFINITY" if order_value == -math.inf else order_value
        ),
        zero_event_fpir_upper_bound=zero_event_one_sided_upper_bound(
            trials=len(rows), alpha=policy.one_sided_alpha
        ),
    )
    return OpenSetCalibrationReceipt(
        status="PASS_EXACT_OPEN_SET_CALIBRATION",
        boundary=boundary,
        summary=summary,
        panel=panel,
    )


def apply_open_set_boundary(
    row: BlindOpenSetScoreRow,
    boundary: FrozenOpenSetBoundary,
) -> OpenSetDispositionResult:
    if row.gallery_size != boundary.gallery_size or row.shot != boundary.shot:
        raise ValueError("RECALIBRATION_REQUIRED: score row N/k differs")
    value = top_identity_evidence(row, margin_threshold=boundary.margin_threshold)
    if not boundary.automatic_accept_enabled:
        disposition = OpenSetDisposition.REVIEW_REQUIRED
        predicted = None
    elif not value.unique_top_identity or not value.margin_passed:
        disposition = OpenSetDisposition.REVIEW_REQUIRED
        predicted = None
    elif value.effective_score is not None and value.effective_score >= boundary.score_threshold:
        disposition = OpenSetDisposition.KNOWN
        predicted = value.top_identity_slot_token
    else:
        disposition = OpenSetDisposition.UNKNOWN
        predicted = None
    return OpenSetDispositionResult(
        query_token=row.query_token,
        disposition=disposition,
        predicted_identity_slot_token=predicted,
        top1_score=value.top1_score,
        top2_score=value.top2_score,
        margin=value.margin,
        boundary_sha256=boundary.boundary_sha256,
    )


def _logaddexp(left: float, right: float) -> float:
    larger = max(left, right)
    smaller = min(left, right)
    if larger == -math.inf:
        return -math.inf
    return larger + math.log1p(math.exp(smaller - larger))


def _cosine(value: object, name: str) -> None:
    _finite(value, name)
    if not -1.0 <= value <= 1.0:  # type: ignore[operator]
        raise ValueError(f"{name} must be in [-1, 1]")


def _finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _probability(value: object, name: str) -> None:
    _finite(value, name)
    if not 0.0 < value < 1.0:  # type: ignore[operator]
        raise ValueError(f"{name} must be in (0, 1)")


def _unit_interval(value: object, name: str) -> None:
    _finite(value, name)
    if not 0.0 <= value <= 1.0:  # type: ignore[operator]
        raise ValueError(f"{name} must be in [0, 1]")


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _text(value: object, name: str, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be bounded canonical text")


def _exact_keys(payload: object, expected: set[str], name: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{name} fields differ")
