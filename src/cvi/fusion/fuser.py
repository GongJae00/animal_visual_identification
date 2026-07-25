from __future__ import annotations

import numpy as np


class LearnedWeightFuser:
    def __init__(self, channel_names: list[str],
                 initial_weights: list[float] | None = None):
        self.channel_names = channel_names
        n = len(channel_names)
        if n == 0:
            self._weights = np.array([], dtype=np.float32)
        else:
            self._weights = np.array(initial_weights or [1.0 / n] * n, dtype=np.float32)

    def fuse(self, scores: dict[str, float],
             uncertainties: dict[str, float] | None = None,
             qualities: dict[str, float] | None = None) -> float:
        if not scores:
            return 0.0
        w = self._weights.copy().astype(np.float64)
        for i, name in enumerate(self.channel_names):
            if uncertainties and name in uncertainties:
                u = uncertainties[name]
                if not np.isfinite(u):
                    u = 1.0
                w[i] *= np.exp(-u)
            if qualities and name in qualities:
                q = qualities[name]
                if not np.isfinite(q):
                    q = 0.0
                w[i] *= q
        total = max(w.sum(), 1e-8)
        result = 0.0
        for i, name in enumerate(self.channel_names):
            if name in scores:
                s = scores[name]
                if not np.isfinite(s):
                    continue
                result += (w[i] / total) * s
        return float(np.clip(result, 0.0, 1.0))

    def update_weights(self, val_scores: np.ndarray, val_labels: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs")
        lr.fit(val_scores, val_labels)
        self._weights = np.abs(lr.coef_[0])
        self._weights = self._weights / max(self._weights.sum(), 1e-8)
