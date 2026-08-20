"""Frozen-threshold verification evaluation with explicit uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import ceil, isfinite, sqrt
from random import Random
from statistics import NormalDist
from typing import Any

from shared.foundation.binomial import (
    required_zero_event_trials as _required_zero_event_trials,
)
from shared.foundation.binomial import (
    zero_event_exact_upper_bound as _zero_event_exact_upper_bound,
)
from shared.foundation.provenance import content_sha256

required_zero_event_trials = _required_zero_event_trials
zero_event_exact_upper_bound = _zero_event_exact_upper_bound


class VerificationDirection(StrEnum):
    RGB_TO_RGB = "RGB_TO_RGB"
    IR_TO_IR = "IR_TO_IR"
    RGB_TO_IR = "RGB_TO_IR"
    IR_TO_RGB = "IR_TO_RGB"


class ClusterUnit(StrEnum):
    QUERY_DOG = "QUERY_DOG"
    QUERY_SESSION = "QUERY_SESSION"


@dataclass(frozen=True, slots=True)
class ClusterBootstrapConfig:
    cluster_unit: ClusterUnit
    resamples: int
    seed: int
    confidence_level: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.resamples, bool)
            or not isinstance(self.resamples, int)
            or self.resamples < 1_000
        ):
            raise ValueError("cluster bootstrap requires at least 1000 resamples")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("bootstrap seed must be a non-negative integer")
        _validate_open_fraction(self.confidence_level, "confidence_level")

    @property
    def config_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "cluster_unit": self.cluster_unit.value,
            "resamples": self.resamples,
            "seed": self.seed,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClusterBootstrapConfig:
        _require_exact_keys(
            payload,
            {
                "cluster_unit",
                "resamples",
                "seed",
                "confidence_level",
            },
            "cluster bootstrap config",
        )
        return cls(
            cluster_unit=ClusterUnit(payload["cluster_unit"]),
            resamples=payload["resamples"],
            seed=payload["seed"],
            confidence_level=payload["confidence_level"],
        )


@dataclass(frozen=True, slots=True)
class FrozenVerificationThreshold:
    score_threshold: float
    target_fmr: float
    confidence_level: float
    direction: VerificationDirection
    model_sha256: str
    gallery_sha256: str
    calibration_manifest_sha256: str

    def __post_init__(self) -> None:
        _validate_finite(self.score_threshold)
        _validate_open_fraction(self.target_fmr, "target_fmr")
        _validate_open_fraction(self.confidence_level, "confidence_level")
        for name, digest in (
            ("model_sha256", self.model_sha256),
            ("gallery_sha256", self.gallery_sha256),
            (
                "calibration_manifest_sha256",
                self.calibration_manifest_sha256,
            ),
        ):
            _validate_sha256(digest, name)

    @property
    def threshold_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | float]:
        return {
            "score_threshold": self.score_threshold,
            "target_fmr": self.target_fmr,
            "confidence_level": self.confidence_level,
            "direction": self.direction.value,
            "model_sha256": self.model_sha256,
            "gallery_sha256": self.gallery_sha256,
            "calibration_manifest_sha256": (
                self.calibration_manifest_sha256
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> FrozenVerificationThreshold:
        _require_exact_keys(
            payload,
            {
                "score_threshold",
                "target_fmr",
                "confidence_level",
                "direction",
                "model_sha256",
                "gallery_sha256",
                "calibration_manifest_sha256",
            },
            "frozen verification threshold",
        )
        return cls(
            score_threshold=payload["score_threshold"],
            target_fmr=payload["target_fmr"],
            confidence_level=payload["confidence_level"],
            direction=VerificationDirection(payload["direction"]),
            model_sha256=payload["model_sha256"],
            gallery_sha256=payload["gallery_sha256"],
            calibration_manifest_sha256=payload[
                "calibration_manifest_sha256"
            ],
        )


@dataclass(frozen=True, slots=True)
class ScoredVerificationPair:
    pair_id: str
    query_track_id: str
    reference_template_id: str
    query_dog_id: str
    reference_dog_id: str
    query_session_id: str
    reference_session_id: str
    score: float

    def __post_init__(self) -> None:
        for name, value in (
            ("pair_id", self.pair_id),
            ("query_track_id", self.query_track_id),
            ("reference_template_id", self.reference_template_id),
            ("query_dog_id", self.query_dog_id),
            ("reference_dog_id", self.reference_dog_id),
            ("query_session_id", self.query_session_id),
            ("reference_session_id", self.reference_session_id),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        _validate_finite(self.score)
        if self.query_session_id == self.reference_session_id:
            raise ValueError(
                "oracle verification pair must be session-disjoint"
            )

    @property
    def same_identity(self) -> bool:
        return self.query_dog_id == self.reference_dog_id


@dataclass(frozen=True, slots=True)
class RateEstimate:
    events: int
    trials: int
    estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    interval_method: str

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class ClusterBootstrapRate:
    estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    cluster_unit: ClusterUnit
    cluster_count: int
    resamples: int
    seed: int
    interval_method: str

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            field_name: (
                value.value if isinstance(value, StrEnum) else value
            )
            for field_name, value in (
                (name, getattr(self, name))
                for name in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class VerificationEvaluation:
    test_manifest_sha256: str
    threshold_sha256: str
    direction: VerificationDirection
    score_rule: str
    pair_count: int
    positive_pairs: int
    negative_pairs: int
    distinct_query_dogs: int
    distinct_query_sessions: int
    false_match_rate: RateEstimate
    false_non_match_rate: RateEstimate
    cluster_bootstrap_config_sha256: str
    false_match_cluster_rate: ClusterBootstrapRate
    false_non_match_cluster_rate: ClusterBootstrapRate
    dependence_warning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "evaluation.verification_evaluation.v1",
            "test_manifest_sha256": self.test_manifest_sha256,
            "threshold_sha256": self.threshold_sha256,
            "direction": self.direction.value,
            "score_rule": self.score_rule,
            "pair_count": self.pair_count,
            "positive_pairs": self.positive_pairs,
            "negative_pairs": self.negative_pairs,
            "distinct_query_dogs": self.distinct_query_dogs,
            "distinct_query_sessions": self.distinct_query_sessions,
            "false_match_rate": self.false_match_rate.to_dict(),
            "false_non_match_rate": self.false_non_match_rate.to_dict(),
            "cluster_bootstrap_config_sha256": (
                self.cluster_bootstrap_config_sha256
            ),
            "false_match_cluster_rate": (
                self.false_match_cluster_rate.to_dict()
            ),
            "false_non_match_cluster_rate": (
                self.false_non_match_cluster_rate.to_dict()
            ),
            "dependence_warning": self.dependence_warning,
        }


def evaluate_frozen_verification_threshold(
    pairs: tuple[ScoredVerificationPair, ...],
    *,
    threshold: FrozenVerificationThreshold,
    test_manifest_sha256: str,
    bootstrap: ClusterBootstrapConfig,
) -> VerificationEvaluation:
    """Evaluate a pre-frozen threshold; no threshold search is performed."""

    _validate_sha256(test_manifest_sha256, "test_manifest_sha256")
    if test_manifest_sha256 == threshold.calibration_manifest_sha256:
        raise ValueError("calibration and test manifests must be distinct")
    if bootstrap.confidence_level != threshold.confidence_level:
        raise ValueError(
            "bootstrap and threshold confidence levels must match"
        )
    if not pairs:
        raise ValueError("at least one scored pair is required")
    pair_ids = tuple(pair.pair_id for pair in pairs)
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair IDs must be unique")
    positive = tuple(pair for pair in pairs if pair.same_identity)
    negative = tuple(pair for pair in pairs if not pair.same_identity)
    if not positive or not negative:
        raise ValueError("both positive and negative pairs are required")
    false_matches = sum(
        pair.score >= threshold.score_threshold for pair in negative
    )
    false_non_matches = sum(
        pair.score < threshold.score_threshold for pair in positive
    )
    negative_cluster_events = tuple(
        (
            _cluster_id(pair, bootstrap.cluster_unit),
            pair.score >= threshold.score_threshold,
        )
        for pair in negative
    )
    positive_cluster_events = tuple(
        (
            _cluster_id(pair, bootstrap.cluster_unit),
            pair.score < threshold.score_threshold,
        )
        for pair in positive
    )
    return VerificationEvaluation(
        test_manifest_sha256=test_manifest_sha256,
        threshold_sha256=threshold.threshold_sha256,
        direction=threshold.direction,
        score_rule="accept_same_identity_if_score_greater_than_or_equal",
        pair_count=len(pairs),
        positive_pairs=len(positive),
        negative_pairs=len(negative),
        distinct_query_dogs=len({pair.query_dog_id for pair in pairs}),
        distinct_query_sessions=len(
            {pair.query_session_id for pair in pairs}
        ),
        false_match_rate=wilson_rate(
            false_matches,
            len(negative),
            confidence_level=threshold.confidence_level,
        ),
        false_non_match_rate=wilson_rate(
            false_non_matches,
            len(positive),
            confidence_level=threshold.confidence_level,
        ),
        cluster_bootstrap_config_sha256=bootstrap.config_sha256,
        false_match_cluster_rate=cluster_bootstrap_rate(
            negative_cluster_events,
            config=bootstrap,
        ),
        false_non_match_cluster_rate=cluster_bootstrap_rate(
            positive_cluster_events,
            config=bootstrap,
        ),
        dependence_warning=(
            "Pair-level Wilson intervals assume independent events. Cluster "
            "bootstrap preserves declared query-cluster dependence but can "
            "still be underpowered with few clusters or zero rare events; "
            "retain exact rare-event upper bounds."
        ),
    )


def wilson_rate(
    events: int,
    trials: int,
    *,
    confidence_level: float,
) -> RateEstimate:
    if (
        isinstance(events, bool)
        or isinstance(trials, bool)
        or not isinstance(events, int)
        or not isinstance(trials, int)
    ):
        raise TypeError("events and trials must be integers")
    if trials <= 0 or events < 0 or events > trials:
        raise ValueError("require 0 <= events <= trials and trials > 0")
    _validate_open_fraction(confidence_level, "confidence_level")
    estimate = events / trials
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (estimate + z_squared / (2 * trials)) / denominator
    half_width = (
        z
        * sqrt(
            estimate * (1 - estimate) / trials
            + z_squared / (4 * trials * trials)
        )
        / denominator
    )
    return RateEstimate(
        events=events,
        trials=trials,
        estimate=estimate,
        lower_bound=max(0.0, center - half_width),
        upper_bound=min(1.0, center + half_width),
        confidence_level=confidence_level,
        interval_method="two_sided_wilson_score",
    )


def cluster_bootstrap_rate(
    cluster_events: tuple[tuple[str, bool], ...],
    *,
    config: ClusterBootstrapConfig,
) -> ClusterBootstrapRate:
    """Percentile interval from whole-cluster resampling with replacement."""

    if not cluster_events:
        raise ValueError("cluster bootstrap requires observations")
    groups: dict[str, list[bool]] = {}
    for cluster_id, event in cluster_events:
        if not cluster_id.strip():
            raise ValueError("cluster ID must be non-empty")
        groups.setdefault(cluster_id, []).append(event)
    if len(groups) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    grouped_counts = tuple(
        (sum(events), len(events)) for events in groups.values()
    )
    total_events = sum(item[0] for item in grouped_counts)
    total_trials = sum(item[1] for item in grouped_counts)
    rng = Random(config.seed)
    sampled_rates: list[float] = []
    cluster_count = len(grouped_counts)
    for _ in range(config.resamples):
        sampled_events = 0
        sampled_trials = 0
        for _ in range(cluster_count):
            events, trials = grouped_counts[rng.randrange(cluster_count)]
            sampled_events += events
            sampled_trials += trials
        sampled_rates.append(sampled_events / sampled_trials)
    sampled_rates.sort()
    alpha = (1.0 - config.confidence_level) / 2.0
    lower_index = max(0, ceil(alpha * config.resamples) - 1)
    upper_index = min(
        config.resamples - 1,
        ceil((1.0 - alpha) * config.resamples) - 1,
    )
    return ClusterBootstrapRate(
        estimate=total_events / total_trials,
        lower_bound=sampled_rates[lower_index],
        upper_bound=sampled_rates[upper_index],
        confidence_level=config.confidence_level,
        cluster_unit=config.cluster_unit,
        cluster_count=cluster_count,
        resamples=config.resamples,
        seed=config.seed,
        interval_method="whole_cluster_percentile_bootstrap",
    )


def _validate_open_fraction(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 < value < 1
    ):
        raise ValueError(f"{name} must be finite and in (0, 1)")


def _validate_finite(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError("score must be finite")


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _cluster_id(
    pair: ScoredVerificationPair,
    unit: ClusterUnit,
) -> str:
    if unit is ClusterUnit.QUERY_DOG:
        return pair.query_dog_id
    return pair.query_session_id


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
