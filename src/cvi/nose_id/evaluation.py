"""Common appearance, nose, and late-fusion closed-set evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from cvi.evaluation.retrieval import (
    compute_cosine_score_matrix,
    evaluate_multi_template_closed_set,
)


def evaluate_noseid_ablation(
    *,
    query_identity_ids: np.ndarray,
    gallery_identity_ids: np.ndarray,
    query_template_ids: np.ndarray,
    gallery_template_ids: np.ndarray,
    query_capture_ids: np.ndarray,
    gallery_capture_ids: np.ndarray,
    query_appearance: np.ndarray,
    gallery_appearance: np.ndarray,
    query_nose: np.ndarray,
    gallery_nose: np.ndarray,
    query_nose_utility: np.ndarray | None = None,
    nose_base_weight: float = 0.30,
) -> dict[str, dict[str, Any]]:
    query_captures = np.asarray(query_capture_ids)
    gallery_captures = np.asarray(gallery_capture_ids)
    if query_captures.shape != (len(query_identity_ids),) or gallery_captures.shape != (len(gallery_identity_ids),):
        raise ValueError("capture IDs must align with query and gallery templates")
    capture_values = query_captures.tolist() + gallery_captures.tolist()
    for value in capture_values:
        if value is None:
            raise ValueError("capture IDs must be non-null")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("capture IDs must be finite")
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError("capture IDs must be hashable") from exc
    overlap = set(query_captures.tolist()) & set(gallery_captures.tolist())
    if overlap:
        raise ValueError("query and gallery captures must be disjoint")
    appearance_scores = compute_cosine_score_matrix(query_appearance, gallery_appearance)
    nose_scores = compute_cosine_score_matrix(query_nose, gallery_nose)
    if query_nose_utility is None:
        nose_weight = np.full((len(query_identity_ids), 1), nose_base_weight, dtype=np.float64)
    else:
        utility = np.asarray(query_nose_utility, dtype=np.float64)
        if utility.shape != (len(query_identity_ids),) or not np.isfinite(utility).all():
            raise ValueError("query nose utility must be a finite query vector")
        nose_weight = (nose_base_weight * np.clip(utility, 0.0, 1.0))[:, None]
    fused_scores = (1.0 - nose_weight) * appearance_scores + nose_weight * nose_scores
    results = {}
    for name, scores in (
        ("appearance", appearance_scores),
        ("nose", nose_scores),
        ("fused", fused_scores),
    ):
        results[name] = evaluate_multi_template_closed_set(
            scores,
            query_identity_ids,
            gallery_identity_ids,
            self_match_policy="exclude",
            query_template_ids=query_template_ids,
            gallery_template_ids=gallery_template_ids,
            rank_ks=(1, 5),
        )
    return results


__all__ = ["evaluate_noseid_ablation"]
