from __future__ import annotations

import numpy as np


class TemporalAggregator:
    def __init__(self, strategy: str = "weighted_median"):
        self._strategy = strategy

    def aggregate(self, embeddings: list[np.ndarray]) -> np.ndarray:
        if not embeddings:
            return np.zeros(0, dtype=np.float32)
        stack = np.stack(embeddings)
        if self._strategy == "mean":
            return stack.mean(axis=0)
        elif self._strategy == "median":
            return np.median(stack, axis=0)
        elif self._strategy == "geometric":
            logs = np.log(np.abs(stack) + 1e-8)
            return np.exp(logs.mean(axis=0))
        elif self._strategy == "weighted_median":
            median = np.median(stack, axis=0)
            dists = np.abs(stack - median).sum(axis=1)
            inv = 1.0 / (dists + 1e-8)
            weights = inv / inv.sum()
            return np.average(stack, axis=0, weights=weights)
        return stack[-1]

    def aggregate_with_reliability(self, embeddings: list[np.ndarray],
                                   qualities: list[float]) -> np.ndarray:
        weights = np.array(qualities)
        weights = weights / max(weights.sum(), 1e-8)
        return np.average(np.stack(embeddings), axis=0, weights=weights)

    def temporal_search(self, embeddings: list[np.ndarray],
                        index: object, top_k: int = 5
                        ) -> list[tuple[int, float, dict]]:
        fused = self.aggregate(embeddings)
        return index.search(fused, top_k)
