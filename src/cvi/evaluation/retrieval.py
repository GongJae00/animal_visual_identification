from __future__ import annotations

import numpy as np


def _ap_for_query(
    query_emb: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_ids: np.ndarray,
    query_id: int,
) -> tuple[float, np.ndarray]:
    sims = np.dot(gallery_embs, query_emb)
    order = np.argsort(sims)[::-1]
    ranked_ids = gallery_ids[order]
    is_positive = ranked_ids == query_id
    n_relevant = int(is_positive.sum())
    if n_relevant == 0:
        return 0.0, np.array([])
    tp = np.cumsum(is_positive)
    precision = tp / np.arange(1, len(tp) + 1)
    ap = float(np.sum(precision * is_positive) / n_relevant)
    return ap, is_positive


def compute_retrieval_metrics(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    top_k: int = 10,
) -> dict:
    n_query = len(query_embs)
    if n_query == 0:
        return {"error": "empty query set", "num_queries": 0}
    rank_k_hits = {f"Rank-{k}": 0 for k in [1, 5, 10]}
    aps = []
    for i in range(n_query):
        ap, is_pos = _ap_for_query(
            query_embs[i], gallery_embs, gallery_ids, query_ids[i]
        )
        aps.append(ap)
        if ap > 0:
            rank_positions = np.where(is_pos)[0]
            for k in [1, 5, 10]:
                if len(rank_positions) > 0 and rank_positions[0] < k:
                    rank_k_hits[f"Rank-{k}"] += 1
    mAP = float(np.mean(aps))
    for k in [1, 5, 10]:
        rank_k_hits[f"Rank-{k}"] = round(rank_k_hits[f"Rank-{k}"] / n_query, 4)
    return {
        "num_queries": n_query,
        "num_gallery": len(gallery_embs),
        "mAP": round(mAP, 4),
        **rank_k_hits,
    }
