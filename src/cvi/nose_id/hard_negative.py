"""Exact hard-identity mining for NoseID-v1 training."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def mine_hard_neighbors(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    *,
    top_k: int = 50,
) -> dict[str, tuple[str, ...]]:
    values = np.asarray(embeddings, dtype=np.float32)
    identities = np.asarray(identity_ids)
    if values.ndim != 2 or len(values) != len(identities) or len(values) == 0:
        raise ValueError("hard-negative embeddings and identities must align")
    if not np.isfinite(values).all() or top_k <= 0:
        raise ValueError("hard-negative inputs must be finite with positive top_k")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("hard-negative embeddings must be non-zero")
    values = values / norms
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, identity in enumerate(identities.tolist()):
        grouped[str(identity)].append(index)
    prototypes = {
        identity: np.mean(values[indices], axis=0)
        for identity, indices in grouped.items()
    }
    ordered = sorted(prototypes)
    matrix = np.stack([prototypes[identity] for identity in ordered])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
    scores = matrix @ matrix.T
    result: dict[str, tuple[str, ...]] = {}
    for index, identity in enumerate(ordered):
        order = np.argsort(-scores[index], kind="stable")
        result[identity] = tuple(
            ordered[candidate]
            for candidate in order
            if candidate != index
        )[:top_k]
    return result


__all__ = ["mine_hard_neighbors"]
