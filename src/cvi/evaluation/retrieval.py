from __future__ import annotations

from typing import Literal

import numpy as np


class RetrievalError(ValueError):
    pass


class EmbeddingNormError(RetrievalError):
    pass


class NonFiniteEmbeddingError(RetrievalError):
    pass


class MissingGalleryIdentityError(RetrievalError):
    pass


class ClosedSetViolation(RetrievalError):
    pass


def _validate_embeddings(
    embs: np.ndarray,
    ids: np.ndarray,
    name: str,
) -> None:
    if not np.all(np.isfinite(embs)):
        raise NonFiniteEmbeddingError(
            f"{name} embeddings contain non-finite values"
        )


def _normalize_rows(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    zero_norm = norms.ravel() < 1e-8
    if zero_norm.any():
        raise EmbeddingNormError(
            f"{int(zero_norm.sum())} {', '.join(['row'] if zero_norm.sum() == 1 else ['rows'])} "
            f"ha{s if zero_norm.sum() > 1 else ''} zero norm"
        )
    return embs / norms


def _compute_ap_inp(
    sims: np.ndarray,
    is_positive: np.ndarray,
    n_relevant: int,
) -> tuple[float, float]:
    order = np.argsort(-sims, kind="stable")
    ranked_pos = is_positive[order]
    tp = np.cumsum(ranked_pos)
    rank = np.arange(1, len(sims) + 1, dtype=np.float64)
    precision = tp / rank
    ap = float(np.sum(precision * ranked_pos) / n_relevant)
    penetration_rates = rank / n_relevant
    inp_candidates = penetration_rates * ranked_pos
    minp = float(np.max(inp_candidates)) if inp_candidates.sum() > 0 else 0.0
    return ap, minp


def compute_retrieval_metrics(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    metric: Literal["cosine", "dot"] = "cosine",
    rank_ks: tuple[int, ...] = (1, 5, 10),
    exclude_self: np.ndarray | None = None,
    closed_set: bool = True,
) -> dict:
    n_query = len(query_embs)
    n_gallery = len(gallery_embs)
    if n_query == 0:
        raise RetrievalError("empty query set")
    if n_gallery == 0:
        raise RetrievalError("empty gallery set")
    _validate_embeddings(query_embs, query_ids, "query")
    _validate_embeddings(gallery_embs, gallery_ids, "gallery")
    q = _normalize_rows(query_embs)
    g = _normalize_rows(gallery_embs)
    if exclude_self is not None:
        if exclude_self.shape != (n_query, n_gallery):
            raise RetrievalError(
                f"exclude_self shape {exclude_self.shape} != ({n_query}, {n_gallery})"
            )
        if exclude_self.dtype != np.bool_:
            raise RetrievalError("exclude_self must be bool")
    sims = q @ g.T
    rank_hits = {f"Rank-{k}": 0 for k in rank_ks}
    aps = []
    minps = []
    num_valid = 0
    num_no_relevant = 0
    for i in range(n_query):
        row_sims = sims[i].copy()
        if exclude_self is not None:
            row_sims[exclude_self[i]] = -np.inf
        gallery_ids_i = gallery_ids.copy()
        is_positive = gallery_ids_i == query_ids[i]
        n_relevant = int(is_positive.sum())
        if n_relevant == 0:
            if closed_set:
                raise ClosedSetViolation(
                    f"query {i} (id={query_ids[i]}) has no matching gallery identity "
                    f"but closed_set=True"
                )
            aps.append(0.0)
            minps.append(0.0)
            num_no_relevant += 1
            continue
        num_valid += 1
        order = np.argsort(-row_sims, kind="stable")
        ranked_pos = is_positive[order]
        first_pos = int(np.where(ranked_pos)[0][0]) if ranked_pos.any() else n_gallery
        for k in rank_ks:
            if first_pos < k:
                rank_hits[f"Rank-{k}"] += 1
        ap, minp = _compute_ap_inp(row_sims, is_positive, n_relevant)
        aps.append(ap)
        minps.append(minp)
    mAP = float(np.mean(aps)) if aps else 0.0
    mINP = float(np.mean(minps)) if minps else 0.0
    result = {
        "num_queries": n_query,
        "num_gallery": n_gallery,
        "num_valid_queries": num_valid,
        "num_queries_without_relevant_gallery": num_no_relevant,
        "metric": metric,
        "closed_set": closed_set,
        "mAP": mAP,
        "mINP": mINP,
    }
    for k in rank_ks:
        result[f"Rank-{k}"] = round(rank_hits[f"Rank-{k}"] / max(num_valid, 1), 4)
    if exclude_self is not None:
        result["self_match_excluded"] = True
    return result
