"""Fail-closed A/F/N teacher agreement for candidate mask supervision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from foundation.provenance import content_sha256

INTERPRETATION = (
    "MODEL_AGREEMENT_CANDIDATE_NOT_VERIFIED_SEMANTIC_SEGMENTATION_OR_"
    "BIOMETRIC_VALIDATION"
)
REGION_CLASS_MAPS: dict[str, dict[str, str]] = {
    "A": {"0": "background", "1": "dog"},
    "F": {"0": "background", "1": "ears", "2": "face", "3": "neck"},
    "N": {"0": "context", "1": "nasal_surface", "2": "nostril"},
}


class ConsensusState(StrEnum):
    HARD_CANDIDATE = "HARD_CANDIDATE"
    SOFT_CANDIDATE = "SOFT_CANDIDATE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True, slots=True)
class RegionTeacherBinding:
    teacher_id: str
    model_family: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        for name in ("teacher_id", "model_family"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"region teacher {name} must be canonical text")
        if (
            not isinstance(self.artifact_sha256, str)
            or len(self.artifact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.artifact_sha256)
        ):
            raise ValueError("region teacher artifact SHA-256 differs")

    def to_dict(self) -> dict[str, str]:
        return {
            "teacher_id": self.teacher_id,
            "model_family": self.model_family,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class RegionConsensusPolicy:
    minimum_teacher_count: int = 2
    minimum_distinct_families_for_hard: int = 2
    hard_minimum_pairwise_iou: float = 0.80
    hard_minimum_confidence: float = 0.85
    hard_maximum_mean_entropy: float = 0.30
    soft_minimum_pairwise_iou: float = 0.55
    soft_minimum_confidence: float = 0.60
    soft_maximum_mean_entropy: float = 0.65

    def __post_init__(self) -> None:
        for name in ("minimum_teacher_count", "minimum_distinct_families_for_hard"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"region consensus {name} must be at least two")
        for name in (
            "hard_minimum_pairwise_iou",
            "hard_minimum_confidence",
            "hard_maximum_mean_entropy",
            "soft_minimum_pairwise_iou",
            "soft_minimum_confidence",
            "soft_maximum_mean_entropy",
        ):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"region consensus {name} must be a float in [0,1]")
        if (
            self.hard_minimum_pairwise_iou < self.soft_minimum_pairwise_iou
            or self.hard_minimum_confidence < self.soft_minimum_confidence
            or self.hard_maximum_mean_entropy > self.soft_maximum_mean_entropy
        ):
            raise ValueError("hard consensus thresholds must be stricter than soft thresholds")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


_DEFAULT_POLICY = RegionConsensusPolicy()


@dataclass(frozen=True, slots=True)
class RegionConsensusResult:
    region: str
    state: ConsensusState
    class_map: dict[str, str]
    hard_mask: np.ndarray | None
    soft_probabilities: np.ndarray | None
    uncertainty: np.ndarray | None
    metrics: dict[str, Any]
    producer: dict[str, Any]
    interpretation: str = INTERPRETATION

    def to_record(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "state": self.state.value,
            "qualification": (
                None if self.state is ConsensusState.ABSTAIN else "MODEL_GENERATED_CANDIDATE"
            ),
            "class_map": self.class_map,
            "metrics": self.metrics,
            "producer": self.producer,
            "interpretation": self.interpretation,
        }


def region_teacher_consensus(
    region: str,
    probabilities: np.ndarray,
    *,
    teachers: tuple[RegionTeacherBinding, ...],
    source_validity: np.ndarray,
    geometry_support: np.ndarray,
    teacher_weights: tuple[float, ...] | None = None,
    policy: RegionConsensusPolicy = _DEFAULT_POLICY,
) -> RegionConsensusResult:
    """Fuse teacher probabilities without promoting agreement to ground truth."""

    if region not in REGION_CLASS_MAPS:
        raise ValueError("region consensus target must be A, F, or N")
    values = np.asarray(probabilities, dtype=np.float64)
    classes = len(REGION_CLASS_MAPS[region])
    if values.ndim != 4 or values.shape[1] != classes or values.shape[0] != len(teachers):
        raise ValueError("teacher probabilities must be [teacher,class,height,width]")
    if not teachers or len({item.teacher_id for item in teachers}) != len(teachers):
        raise ValueError("region consensus teacher IDs must be non-empty and unique")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("teacher probabilities must be finite values in [0,1]")
    totals = values.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("teacher class probabilities must sum to one")
    valid = np.asarray(source_validity, dtype=bool)
    geometry = np.asarray(geometry_support, dtype=bool)
    if valid.shape != values.shape[2:] or geometry.shape != values.shape[2:]:
        raise ValueError("region consensus support geometry shape differs")
    support = valid & geometry
    if not np.any(support):
        return _abstain(
            region,
            teachers,
            policy,
            reason="NO_VALID_GEOMETRY_SUPPORT",
            valid_pixels=int(valid.sum()),
            support_pixels=0,
        )
    weights = _teacher_weights(teacher_weights, len(teachers))
    constrained = values.copy()
    constrained[:, 1:, ~support] = 0.0
    constrained[:, 0, ~support] = 1.0
    hard_teachers = np.argmax(constrained, axis=1).astype(np.uint8)
    pairwise = _pairwise_macro_iou(hard_teachers, support, classes)
    consensus = np.tensordot(weights, constrained, axes=(0, 0))
    hard_mask = np.argmax(consensus, axis=0).astype(np.uint8)
    confidence_map = np.max(consensus, axis=0)
    entropy = -(consensus * np.log(consensus.clip(1e-12))).sum(axis=0) / math.log(classes)
    mean_confidence = float(confidence_map[support].mean())
    mean_entropy = float(entropy[support].mean())
    distinct_families = len({item.model_family for item in teachers})
    reasons: list[str] = []
    if len(teachers) < policy.minimum_teacher_count:
        reasons.append("INSUFFICIENT_TEACHERS")
    if pairwise < policy.soft_minimum_pairwise_iou:
        reasons.append("LOW_PAIRWISE_IOU")
    if mean_confidence < policy.soft_minimum_confidence:
        reasons.append("LOW_CONSENSUS_CONFIDENCE")
    if mean_entropy > policy.soft_maximum_mean_entropy:
        reasons.append("HIGH_CONSENSUS_ENTROPY")
    hard_eligible = (
        not reasons
        and distinct_families >= policy.minimum_distinct_families_for_hard
        and pairwise >= policy.hard_minimum_pairwise_iou
        and mean_confidence >= policy.hard_minimum_confidence
        and mean_entropy <= policy.hard_maximum_mean_entropy
    )
    state = (
        ConsensusState.HARD_CANDIDATE
        if hard_eligible
        else ConsensusState.SOFT_CANDIDATE
        if not reasons
        else ConsensusState.ABSTAIN
    )
    if not hard_eligible and not reasons and distinct_families < policy.minimum_distinct_families_for_hard:
        reasons.append("INSUFFICIENT_TEACHER_FAMILY_DIVERSITY_FOR_HARD")
    metrics = {
        "teacher_count": len(teachers),
        "distinct_model_family_count": distinct_families,
        "mean_pairwise_macro_iou": pairwise,
        "mean_consensus_confidence": mean_confidence,
        "mean_normalized_entropy": mean_entropy,
        "valid_pixel_count": int(valid.sum()),
        "geometry_support_pixel_count": int(support.sum()),
        "decision_reasons": reasons,
    }
    producer = _producer(teachers, weights, policy)
    return RegionConsensusResult(
        region=region,
        state=state,
        class_map=dict(REGION_CLASS_MAPS[region]),
        hard_mask=None if state is ConsensusState.ABSTAIN else hard_mask,
        soft_probabilities=(
            None if state is ConsensusState.ABSTAIN else consensus.astype(np.float32)
        ),
        uncertainty=None if state is ConsensusState.ABSTAIN else entropy.astype(np.float32),
        metrics=metrics,
        producer=producer,
    )


def _teacher_weights(values: tuple[float, ...] | None, count: int) -> np.ndarray:
    if values is None:
        return np.full(count, 1.0 / count, dtype=np.float64)
    if len(values) != count or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in values
    ):
        raise ValueError("region consensus teacher weights must be positive finite values")
    weights = np.asarray(values, dtype=np.float64)
    return weights / weights.sum()


def _pairwise_macro_iou(masks: np.ndarray, support: np.ndarray, classes: int) -> float:
    scores: list[float] = []
    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            class_scores: list[float] = []
            for class_index in range(1, classes):
                left = (masks[first] == class_index) & support
                right = (masks[second] == class_index) & support
                union = int(np.logical_or(left, right).sum())
                if union:
                    class_scores.append(float(np.logical_and(left, right).sum() / union))
            scores.append(float(np.mean(class_scores)) if class_scores else 1.0)
    return float(np.mean(scores)) if scores else 0.0


def _producer(
    teachers: tuple[RegionTeacherBinding, ...],
    weights: np.ndarray,
    policy: RegionConsensusPolicy,
) -> dict[str, Any]:
    body = {
        "schema_version": "cvi.region_teacher_consensus_producer.v1",
        "teachers": [item.to_dict() for item in teachers],
        "normalized_teacher_weights": [float(item) for item in weights],
        "policy": policy.to_dict(),
        "algorithm": "WEIGHTED_CLASS_PROBABILITY_MEAN_WITH_MACRO_FOREGROUND_IOU_ENTROPY_AND_GEOMETRY_GATES",
    }
    return {**body, "producer_sha256": content_sha256(body)}


def _abstain(
    region: str,
    teachers: tuple[RegionTeacherBinding, ...],
    policy: RegionConsensusPolicy,
    *,
    reason: str,
    valid_pixels: int,
    support_pixels: int,
) -> RegionConsensusResult:
    weights = _teacher_weights(None, len(teachers))
    return RegionConsensusResult(
        region=region,
        state=ConsensusState.ABSTAIN,
        class_map=dict(REGION_CLASS_MAPS[region]),
        hard_mask=None,
        soft_probabilities=None,
        uncertainty=None,
        metrics={
            "teacher_count": len(teachers),
            "distinct_model_family_count": len({item.model_family for item in teachers}),
            "mean_pairwise_macro_iou": 0.0,
            "mean_consensus_confidence": 0.0,
            "mean_normalized_entropy": 1.0,
            "valid_pixel_count": valid_pixels,
            "geometry_support_pixel_count": support_pixels,
            "decision_reasons": [reason],
        },
        producer=_producer(teachers, weights, policy),
    )


__all__ = [
    "INTERPRETATION",
    "REGION_CLASS_MAPS",
    "ConsensusState",
    "RegionConsensusPolicy",
    "RegionConsensusResult",
    "RegionTeacherBinding",
    "region_teacher_consensus",
]
