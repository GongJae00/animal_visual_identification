from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class OpenSetError(ValueError):
    pass


@dataclass(frozen=True)
class OpenSetResult:
    known_vs_unknown_auroc: float
    known_vs_unknown_aupr: float
    dir_at_fpir: dict[str, float]
    false_accept_count: int
    false_reject_count: int
    num_enrolled_queries: int
    num_unknown_queries: int
    num_gallery_identities: int


def _normalize_rows(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    zero = norms.ravel() < 1e-8
    if zero.any():
        raise OpenSetError(
            f"{int(zero.sum())} embedding(s) ha{'' if zero.sum() > 1 else 's'} zero norm"
        )
    return embs / norms


def evaluate_open_set(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    fpir_targets: tuple[float, ...] = (0.01, 0.001),
    metric: Literal["cosine", "dot"] = "cosine",
    closed_set_no_match_policy: str = "error",
) -> OpenSetResult:
    n_query = len(query_embs)
    n_gallery = len(gallery_embs)
    if n_query == 0:
        raise OpenSetError("empty query set")
    if n_gallery == 0:
        raise OpenSetError("empty gallery set")
    q = _normalize_rows(query_embs)
    g = _normalize_rows(gallery_embs)
    sims = q @ g.T
    enrolled_ids = set(gallery_ids.tolist())
    if closed_set_no_match_policy == "error":
        for i in range(n_query):
            if query_ids[i] not in enrolled_ids:
                raise OpenSetError(
                    f"query {i} (id={query_ids[i]}) not in gallery "
                    f"identities under closed-set-equivalent policy"
                )
    unknown_mask = np.array(
        [qid not in enrolled_ids for qid in query_ids], dtype=bool
    )
    known_mask = ~unknown_mask
    n_known = int(known_mask.sum())
    n_unknown = int(unknown_mask.sum())
    if n_known == 0 or n_unknown == 0:
        raise OpenSetError(
            f"need both known ({n_known}) and unknown ({n_unknown}) queries"
        )
    max_scores = np.max(sims, axis=1)
    known_scores = max_scores[known_mask]
    unknown_scores = max_scores[unknown_mask]
    labels = np.concatenate([
        np.ones(n_known, dtype=np.int64),
        np.zeros(n_unknown, dtype=np.int64),
    ])
    all_scores = np.concatenate([known_scores, unknown_scores])
    auroc = float(roc_auc_score(labels, all_scores))
    aupr = float(average_precision_score(labels, all_scores))
    sorted_idx = np.argsort(-all_scores)
    sorted_labels = labels[sorted_idx]
    fp = np.cumsum(sorted_labels == 0)
    tp = np.cumsum(sorted_labels == 1)
    far = fp / max(n_unknown, 1)
    tpr = tp / max(n_known, 1)
    dir_at_fpir = {}
    for target in fpir_targets:
        valid = np.where(far <= target)[0]
        if len(valid) > 0:
            dir_at_fpir[f"DIR@FPIR={target:.0e}"] = float(tpr[valid[-1]])
        else:
            dir_at_fpir[f"DIR@FPIR={target:.0e}"] = 0.0
    top1_ids = np.argmax(sims, axis=1)
    top1_gallery_ids = gallery_ids[top1_ids]
    false_accept = int(((top1_ids >= 0) & unknown_mask).sum())
    false_reject = 0
    for i in np.where(known_mask)[0]:
        if top1_gallery_ids[i] != query_ids[i]:
            false_reject += 1
    return OpenSetResult(
        known_vs_unknown_auroc=auroc,
        known_vs_unknown_aupr=aupr,
        dir_at_fpir=dir_at_fpir,
        false_accept_count=false_accept,
        false_reject_count=false_reject,
        num_enrolled_queries=n_known,
        num_unknown_queries=n_unknown,
        num_gallery_identities=len(enrolled_ids),
    )
