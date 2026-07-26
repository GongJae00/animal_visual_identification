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


class MetricInvariantError(RetrievalError):
    pass


class SampleIdValidationError(RetrievalError):
    pass


def _validate_embeddings(
    embs: np.ndarray,
    ids: np.ndarray,
    name: str,
) -> None:
    if embs.ndim != 2:
        raise RetrievalError(
            f"{name} embeddings must be 2-d, got shape {embs.shape}"
        )
    if ids.ndim != 1:
        raise RetrievalError(
            f"{name} ids must be 1-d, got shape {ids.shape}"
        )
    if len(embs) != len(ids):
        raise RetrievalError(
            f"{name} embeddings count {len(embs)} != ids count {len(ids)}"
        )
    if not np.all(np.isfinite(embs)):
        raise NonFiniteEmbeddingError(
            f"{name} embeddings contain non-finite values"
        )


def _validate_sample_ids(
    query_sample_ids: np.ndarray | None,
    gallery_sample_ids: np.ndarray | None,
    n_query: int,
    n_gallery: int,
) -> None:
    if query_sample_ids is None and gallery_sample_ids is None:
        return
    if query_sample_ids is None or gallery_sample_ids is None:
        raise SampleIdValidationError(
            "both query_sample_ids and gallery_sample_ids must be provided together"
        )
    if query_sample_ids.ndim != 1:
        raise SampleIdValidationError(
            f"query_sample_ids must be 1-d, got shape {query_sample_ids.shape}"
        )
    if gallery_sample_ids.ndim != 1:
        raise SampleIdValidationError(
            f"gallery_sample_ids must be 1-d, got shape {gallery_sample_ids.shape}"
        )
    if len(query_sample_ids) != n_query:
        raise SampleIdValidationError(
            f"query_sample_ids length {len(query_sample_ids)} != n_query {n_query}"
        )
    if len(gallery_sample_ids) != n_gallery:
        raise SampleIdValidationError(
            f"gallery_sample_ids length {len(gallery_sample_ids)} != n_gallery {n_gallery}"
        )
    unique_queries, q_counts = np.unique(query_sample_ids, return_counts=True)
    dup = unique_queries[q_counts > 1]
    if len(dup) > 0:
        raise SampleIdValidationError(
            f"duplicate sample IDs in query_sample_ids: {list(dup)}"
        )
    unique_gallery, g_counts = np.unique(gallery_sample_ids, return_counts=True)
    dup = unique_gallery[g_counts > 1]
    if len(dup) > 0:
        raise SampleIdValidationError(
            f"duplicate sample IDs in gallery_sample_ids: {list(dup)}"
        )


def _normalize_rows(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    zero_norm = norms.ravel() < 1e-8
    if zero_norm.any():
        n_zero = int(zero_norm.sum())
        raise EmbeddingNormError(
            f"{n_zero} row{'s' if n_zero > 1 else ''} ha{'ve' if n_zero > 1 else 's'} zero norm"
        )
    return embs / norms


def _compute_ap_inp(
    ranked_pos: np.ndarray,
    n_relevant: int,
) -> tuple[float, float]:
    tp = np.cumsum(ranked_pos)
    rank = np.arange(1, len(ranked_pos) + 1, dtype=np.float64)
    precision = tp / rank
    ap = float(np.sum(precision * ranked_pos.astype(np.float64)) / n_relevant)
    positive_ranks = np.flatnonzero(ranked_pos) + 1
    last_positive_rank = int(positive_ranks[-1])
    inp = n_relevant / last_positive_rank
    if not 0.0 < inp <= 1.0:
        raise MetricInvariantError(
            f"INP={inp} outside (0,1]; "
            f"n_relevant={n_relevant}, last_positive_rank={last_positive_rank}"
        )
    return ap, inp


def compute_retrieval_metrics(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    metric: Literal["cosine", "dot"] = "cosine",
    rank_ks: tuple[int, ...] = (1, 5, 10),
    query_sample_ids: np.ndarray | None = None,
    gallery_sample_ids: np.ndarray | None = None,
    closed_set: bool = True,
) -> dict:
    if metric != "cosine":
        raise RetrievalError(f"unsupported metric: {metric!r} (only 'cosine' supported)")
    n_query = len(query_embs)
    n_gallery = len(gallery_embs)
    if n_query == 0:
        raise RetrievalError("empty query set")
    if n_gallery == 0:
        raise RetrievalError("empty gallery set")
    _validate_embeddings(query_embs, query_ids, "query")
    _validate_embeddings(gallery_embs, gallery_ids, "gallery")
    _validate_sample_ids(query_sample_ids, gallery_sample_ids, n_query, n_gallery)
    q = _normalize_rows(query_embs)
    g = _normalize_rows(gallery_embs)
    exclude_self: np.ndarray | None = None
    if query_sample_ids is not None and gallery_sample_ids is not None:
        qs = np.asarray(query_sample_ids)
        gs = np.asarray(gallery_sample_ids)
        exclude_self = qs[:, None] == gs[None, :]
    sims = q @ g.T
    rank_hits = {f"Rank-{k}": 0 for k in rank_ks}
    aps = []
    inps = []
    num_valid = 0
    num_no_relevant = 0
    for i in range(n_query):
        row_sims = sims[i].copy()
        is_positive: np.ndarray = gallery_ids == query_ids[i]
        if exclude_self is not None:
            row_sims[exclude_self[i]] = -np.inf
            is_positive = is_positive & ~exclude_self[i]
        n_relevant = int(is_positive.sum())
        if n_relevant == 0:
            if closed_set:
                raise ClosedSetViolation(
                    f"query {i} (id={query_ids[i]}) has no matching gallery identity "
                    f"but closed_set=True"
                )
            aps.append(0.0)
            inps.append(0.0)
            num_no_relevant += 1
            continue
        num_valid += 1
        order = np.argsort(-row_sims, kind="stable")
        ranked_pos = is_positive[order]
        first_pos = int(np.where(ranked_pos)[0][0]) if ranked_pos.any() else n_gallery
        for k in rank_ks:
            if first_pos < k:
                rank_hits[f"Rank-{k}"] += 1
        ap, inp = _compute_ap_inp(ranked_pos, n_relevant)
        aps.append(ap)
        inps.append(inp)
    mAP = float(np.mean(aps)) if aps else 0.0
    mINP = float(np.mean(inps)) if inps else 0.0
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
        result[f"Rank-{k}"] = rank_hits[f"Rank-{k}"] / max(num_valid, 1)
    if exclude_self is not None:
        result["self_match_excluded"] = True
    return result
