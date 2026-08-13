"""Sealed matched-panel evaluation for blind visual-control scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from evaluation.controls.policy import (
    ControlEvaluationBinding,
    ControlPanelSummary,
    VisualControlKind,
)
from evaluation.controls.control_scoring import (
    ControlBlindScoreReceipt,
    EmbeddingCacheManifest,
)
from evaluation import (
    ClusterBootstrapConfig,
    FrozenVerificationThreshold,
    ScoredVerificationPair,
    VerificationEvaluation,
    evaluate_frozen_verification_threshold,
)
from evaluation.controls.pairing import (
    PairConstructionResult,
    PairGroundTruth,
    PairingPolicy,
    PairStratum,
)
from foundation.provenance import content_sha256


@dataclass(frozen=True, slots=True)
class ControlEvaluationPolicy:
    maximum_bindings: int = 1_000_000
    maximum_panels: int = 100
    maximum_total_auc_sort_items: int = 1_000_000
    require_original_control: bool = True
    schema_version: str = "cvi.control_evaluation_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.control_evaluation_policy.v1":
            raise ValueError("unsupported control evaluation policy schema")
        for name in (
            "maximum_bindings",
            "maximum_panels",
            "maximum_total_auc_sort_items",
        ):
            _require_positive_int(getattr(self, name), name)
        if self.require_original_control is not True:
            raise ValueError("ORIGINAL control is mandatory for evaluation")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "schema_version": self.schema_version,
            "maximum_bindings": self.maximum_bindings,
            "maximum_panels": self.maximum_panels,
            "maximum_total_auc_sort_items": (
                self.maximum_total_auc_sort_items
            ),
            "require_original_control": self.require_original_control,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlEvaluationPolicy:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "maximum_bindings",
                "maximum_panels",
                "maximum_total_auc_sort_items",
                "require_original_control",
            },
            "control evaluation policy",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class DescriptiveScoreSeparation:
    positive_pairs: int
    negative_pairs: int
    positive_mean_score: float
    negative_mean_score: float
    roc_auc: float
    auc_method: str = "mann_whitney_midrank_point_estimate"

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "positive_pairs": self.positive_pairs,
            "negative_pairs": self.negative_pairs,
            "positive_mean_score": self.positive_mean_score,
            "negative_mean_score": self.negative_mean_score,
            "roc_auc": self.roc_auc,
            "auc_method": self.auc_method,
        }


@dataclass(frozen=True, slots=True)
class PairedScoreDelta:
    positive_pairs: int
    negative_pairs: int
    positive_mean_control_minus_original: float
    negative_mean_control_minus_original: float
    uncertainty_status: str = "DESCRIPTIVE_POINT_ESTIMATE_ONLY"

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "positive_pairs": self.positive_pairs,
            "negative_pairs": self.negative_pairs,
            "positive_mean_control_minus_original": (
                self.positive_mean_control_minus_original
            ),
            "negative_mean_control_minus_original": (
                self.negative_mean_control_minus_original
            ),
            "uncertainty_status": self.uncertainty_status,
        }


@dataclass(frozen=True, slots=True)
class ControlKindEvaluation:
    control_kind: VisualControlKind
    threshold_evaluation: VerificationEvaluation
    separation: DescriptiveScoreSeparation
    paired_delta: PairedScoreDelta

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_kind": self.control_kind.value,
            "threshold_evaluation": self.threshold_evaluation.to_dict(),
            "separation": self.separation.to_dict(),
            "paired_delta": self.paired_delta.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ControlPanelEvaluation:
    panel_id: str
    matched_base_pairs: int
    controls: tuple[ControlKindEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "matched_base_pairs": self.matched_base_pairs,
            "controls": [control.to_dict() for control in self.controls],
        }


@dataclass(frozen=True, slots=True)
class ControlEvaluationCost:
    bindings_joined: int
    control_metric_rows: int
    auc_sort_items: int
    paired_delta_terms: int
    maximum_rows_per_control: int

    def to_dict(self) -> dict[str, int]:
        return {
            "bindings_joined": self.bindings_joined,
            "control_metric_rows": self.control_metric_rows,
            "auc_sort_items": self.auc_sort_items,
            "paired_delta_terms": self.paired_delta_terms,
            "maximum_rows_per_control": self.maximum_rows_per_control,
        }


@dataclass(frozen=True, slots=True)
class ControlEvaluationReceipt:
    plan_sha256: str
    pair_set_sha256: str
    blind_score_receipt_sha256: str
    embedding_cache_manifest_sha256: str
    threshold_sha256: str
    bootstrap_config_sha256: str
    policy_sha256: str
    panels: tuple[ControlPanelEvaluation, ...]
    cost: ControlEvaluationCost
    limitations: tuple[str, ...]
    schema_version: str = "cvi.control_evaluation_receipt.v1"

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "pair_set_sha256": self.pair_set_sha256,
            "blind_score_receipt_sha256": (
                self.blind_score_receipt_sha256
            ),
            "embedding_cache_manifest_sha256": (
                self.embedding_cache_manifest_sha256
            ),
            "threshold_sha256": self.threshold_sha256,
            "bootstrap_config_sha256": self.bootstrap_config_sha256,
            "policy_sha256": self.policy_sha256,
            "panels": [panel.to_dict() for panel in self.panels],
            "cost": self.cost.to_dict(),
            "limitations": list(self.limitations),
        }


def control_evaluation_bindings_from_payload(
    payload: dict[str, Any],
) -> tuple[
    str,
    str,
    tuple[ControlEvaluationBinding, ...],
    tuple[ControlPanelSummary, ...],
]:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "plan_sha256",
            "pair_set_sha256",
            "bindings",
            "panel_summaries",
        },
        "control evaluation bindings",
    )
    if (
        payload["schema_version"]
        != "cvi.visual_control_evaluation_bindings.v1"
    ):
        raise ValueError("unsupported control evaluation binding schema")
    _validate_sha256(payload["plan_sha256"], "plan_sha256")
    _validate_sha256(payload["pair_set_sha256"], "pair_set_sha256")
    bindings = payload["bindings"]
    summaries = payload["panel_summaries"]
    if not isinstance(bindings, list) or not isinstance(summaries, list):
        raise TypeError("control bindings and summaries must be lists")
    return (
        payload["plan_sha256"],
        payload["pair_set_sha256"],
        tuple(ControlEvaluationBinding.from_dict(item) for item in bindings),
        tuple(ControlPanelSummary.from_dict(item) for item in summaries),
    )


def evaluate_sealed_control_scores(
    *,
    construction: PairConstructionResult,
    pairing_policy: PairingPolicy,
    plan_sha256: str,
    pair_set_sha256: str,
    bindings: tuple[ControlEvaluationBinding, ...],
    panel_summaries: tuple[ControlPanelSummary, ...],
    blind_scores: ControlBlindScoreReceipt,
    embedding_cache_manifest: EmbeddingCacheManifest,
    threshold: FrozenVerificationThreshold,
    bootstrap: ClusterBootstrapConfig,
    policy: ControlEvaluationPolicy,
) -> ControlEvaluationReceipt:
    """Join labels only here and evaluate exact matched panel cohorts."""

    _validate_sha256(plan_sha256, "plan_sha256")
    _validate_sha256(pair_set_sha256, "pair_set_sha256")
    if pair_set_sha256 != construction.result_sha256:
        raise ValueError("sealed control bindings use another pair set")
    if pairing_policy.policy_sha256 != construction.pairing_policy_sha256:
        raise ValueError(
            "pairing policy does not match pair construction"
        )
    if threshold.direction is not pairing_policy.direction:
        raise ValueError(
            "threshold direction does not match pairing policy"
        )
    if blind_scores.plan_sha256 != plan_sha256:
        raise ValueError("blind scores belong to another control plan")
    if (
        blind_scores.embedding_cache_manifest_sha256
        != embedding_cache_manifest.manifest_sha256
    ):
        raise ValueError("blind scores and embedding cache differ")
    if threshold.model_sha256 != embedding_cache_manifest.model_sha256:
        raise ValueError("threshold and embedding model differ")
    if threshold.gallery_sha256 != blind_scores.gallery_sha256:
        raise ValueError("threshold and scoring gallery differ")
    if len(bindings) > policy.maximum_bindings:
        raise ValueError("control bindings exceed evaluation policy")
    if not bindings:
        raise ValueError("control evaluation bindings must not be empty")
    if len(panel_summaries) > policy.maximum_panels:
        raise ValueError("control panels exceed evaluation policy")
    binding_ids = tuple(binding.request_id for binding in bindings)
    if len(binding_ids) != len(set(binding_ids)):
        raise ValueError("control evaluation request IDs must be unique")
    score_by_id = {score.request_id: score.score for score in blind_scores.scores}
    if set(binding_ids) != set(score_by_id):
        raise ValueError("blind score and sealed binding ID sets differ")
    truth_by_id = {
        truth.pair_id: truth for truth in construction.ground_truth
    }
    if any(binding.base_pair_id not in truth_by_id for binding in bindings):
        raise ValueError("control binding references unknown base pair")
    summary_by_panel = {
        summary.panel_id: summary for summary in panel_summaries
    }
    if len(summary_by_panel) != len(panel_summaries):
        raise ValueError("control panel summaries must be unique")
    binding_panels = {binding.panel_id for binding in bindings}
    if set(summary_by_panel) != binding_panels:
        raise ValueError("panel summary and binding panel sets differ")

    grouped: dict[
        str,
        dict[VisualControlKind, dict[str, float]],
    ] = {}
    for binding in bindings:
        control_scores = grouped.setdefault(
            binding.panel_id,
            {},
        ).setdefault(binding.control_kind, {})
        if binding.base_pair_id in control_scores:
            raise ValueError(
                "duplicate panel/control/base-pair evaluation binding"
            )
        control_scores[binding.base_pair_id] = score_by_id[
            binding.request_id
        ]

    total_sort_items = 0
    metric_rows = 0
    paired_terms = 0
    maximum_rows = 0
    panel_results: list[ControlPanelEvaluation] = []
    for panel_id in sorted(grouped):
        controls = grouped[panel_id]
        if VisualControlKind.ORIGINAL not in controls:
            raise ValueError("control panel lacks ORIGINAL")
        if len(controls) < 2:
            raise ValueError("control panel requires an intervention")
        base_sets = {frozenset(scores) for scores in controls.values()}
        if len(base_sets) != 1:
            raise ValueError("control panel cohorts are not pair-matched")
        base_pair_ids = next(iter(base_sets))
        summary = summary_by_panel[panel_id]
        if not summary.minimum_met:
            raise ValueError("control panel did not meet its frozen minimum")
        if summary.selected_pairs != len(base_pair_ids):
            raise ValueError("panel selected count differs from bindings")
        _validate_selected_strata(
            summary,
            base_pair_ids,
            truth_by_id,
        )
        original = controls[VisualControlKind.ORIGINAL]
        control_results: list[ControlKindEvaluation] = []
        for kind in sorted(
            controls,
            key=lambda item: (
                item is not VisualControlKind.ORIGINAL,
                item.value,
            ),
        ):
            scores = controls[kind]
            rows = tuple(
                (
                    truth_by_id[pair_id],
                    scores[pair_id],
                    scores[pair_id] - original[pair_id],
                )
                for pair_id in sorted(base_pair_ids)
            )
            total_sort_items += len(rows)
            if total_sort_items > policy.maximum_total_auc_sort_items:
                raise ValueError("control AUC work exceeds evaluation policy")
            metric_rows += len(rows)
            paired_terms += len(rows)
            maximum_rows = max(maximum_rows, len(rows))
            scored_pairs = tuple(
                _scored_pair(panel_id, kind, truth, score)
                for truth, score, _ in rows
            )
            test_manifest_sha256 = content_sha256(
                {
                    "pair_set_sha256": construction.result_sha256,
                    "plan_sha256": plan_sha256,
                    "blind_score_receipt_sha256": (
                        blind_scores.receipt_sha256
                    ),
                    "panel_id": panel_id,
                    "control_kind": kind.value,
                    "base_pair_ids": sorted(base_pair_ids),
                }
            )
            threshold_result = evaluate_frozen_verification_threshold(
                scored_pairs,
                threshold=threshold,
                test_manifest_sha256=test_manifest_sha256,
                bootstrap=bootstrap,
            )
            separation = _score_separation(rows)
            delta = _paired_delta(rows)
            control_results.append(
                ControlKindEvaluation(
                    control_kind=kind,
                    threshold_evaluation=threshold_result,
                    separation=separation,
                    paired_delta=delta,
                )
            )
        panel_results.append(
            ControlPanelEvaluation(
                panel_id=panel_id,
                matched_base_pairs=len(base_pair_ids),
                controls=tuple(control_results),
            )
        )
    return ControlEvaluationReceipt(
        plan_sha256=plan_sha256,
        pair_set_sha256=construction.result_sha256,
        blind_score_receipt_sha256=blind_scores.receipt_sha256,
        embedding_cache_manifest_sha256=(
            embedding_cache_manifest.manifest_sha256
        ),
        threshold_sha256=threshold.threshold_sha256,
        bootstrap_config_sha256=bootstrap.config_sha256,
        policy_sha256=policy.policy_sha256,
        panels=tuple(panel_results),
        cost=ControlEvaluationCost(
            bindings_joined=len(bindings),
            control_metric_rows=metric_rows,
            auc_sort_items=total_sort_items,
            paired_delta_terms=paired_terms,
            maximum_rows_per_control=maximum_rows,
        ),
        limitations=(
            "AUC and mean score deltas are descriptive point estimates; "
            "do not infer equivalence or causality from them.",
            "Threshold error intervals preserve the declared query-cluster "
            "dependence but can remain underpowered for rare events.",
            "Matched transforms isolate declared pixel interventions only; "
            "mask errors, silhouette holes, and retained coarse cues remain.",
        ),
    )


def _validate_selected_strata(
    summary: ControlPanelSummary,
    base_pair_ids: frozenset[str],
    truth_by_id: dict[str, PairGroundTruth],
) -> None:
    observed = {stratum: 0 for stratum in PairStratum}
    for pair_id in base_pair_ids:
        observed[truth_by_id[pair_id].stratum] += 1
    declared = {
        item.stratum: item.selected_pairs for item in summary.strata
    }
    if any(observed[stratum] != declared.get(stratum, 0) for stratum in observed):
        raise ValueError("panel selected stratum counts differ from truth")


def _scored_pair(
    panel_id: str,
    kind: VisualControlKind,
    truth: PairGroundTruth,
    score: float,
) -> ScoredVerificationPair:
    prefix = content_sha256(
        {
            "panel_id": panel_id,
            "control_kind": kind.value,
            "base_pair_id": truth.pair_id,
        }
    )[:24]
    return ScoredVerificationPair(
        pair_id=f"control-eval-{prefix}",
        query_track_id=f"query-{truth.pair_id}",
        reference_template_id=f"reference-{truth.pair_id}",
        query_dog_id=truth.query_dog_id,
        reference_dog_id=truth.reference_dog_id,
        query_session_id=truth.query_session_id,
        reference_session_id=truth.reference_session_id,
        score=score,
    )


def _score_separation(
    rows: tuple[tuple[PairGroundTruth, float, float], ...],
) -> DescriptiveScoreSeparation:
    positive = tuple(score for truth, score, _ in rows if _same(truth))
    negative = tuple(score for truth, score, _ in rows if not _same(truth))
    if not positive or not negative:
        raise ValueError("control metrics require positive and negative pairs")
    return DescriptiveScoreSeparation(
        positive_pairs=len(positive),
        negative_pairs=len(negative),
        positive_mean_score=math.fsum(positive) / len(positive),
        negative_mean_score=math.fsum(negative) / len(negative),
        roc_auc=_midrank_auc(positive, negative),
    )


def _paired_delta(
    rows: tuple[tuple[PairGroundTruth, float, float], ...],
) -> PairedScoreDelta:
    positive = tuple(delta for truth, _, delta in rows if _same(truth))
    negative = tuple(delta for truth, _, delta in rows if not _same(truth))
    if not positive or not negative:
        raise ValueError("paired deltas require positive and negative pairs")
    return PairedScoreDelta(
        positive_pairs=len(positive),
        negative_pairs=len(negative),
        positive_mean_control_minus_original=(
            math.fsum(positive) / len(positive)
        ),
        negative_mean_control_minus_original=(
            math.fsum(negative) / len(negative)
        ),
    )


def _midrank_auc(
    positive: tuple[float, ...],
    negative: tuple[float, ...],
) -> float:
    ranked = sorted(
        tuple((score, True) for score in positive)
        + tuple((score, False) for score in negative),
        key=lambda item: item[0],
    )
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            is_positive for _, is_positive in ranked[start:end]
        )
        start = end
    positive_count = len(positive)
    negative_count = len(negative)
    return (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def _same(truth: PairGroundTruth) -> bool:
    return truth.query_dog_id == truth.reference_dog_id


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


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
