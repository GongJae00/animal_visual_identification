from __future__ import annotations

import hashlib
import json

import numpy as np


class LearnedWeightFuser:
    def __init__(self, channel_names: list[str],
                 initial_weights: list[float] | None = None,
                 *,
                 channel_dimensions: dict[str, int] | None = None,
                 optional_channels: set[str] | frozenset[str] | None = None):
        if not channel_names or any(not name for name in channel_names):
            raise ValueError("at least one non-empty channel name is required")
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("channel names must be unique")
        self.channel_names = list(channel_names)
        self._channel_dimensions = dict(channel_dimensions or {})
        if self._channel_dimensions and set(self._channel_dimensions) != set(channel_names):
            raise ValueError("channel dimensions must match fusion channels")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self._channel_dimensions.values()
        ):
            raise ValueError("channel dimensions must be positive integers")
        self._optional_channels = frozenset(optional_channels or ())
        if not self._optional_channels <= set(channel_names):
            raise ValueError("optional fusion channels must be configured channels")
        n = len(channel_names)
        values = initial_weights if initial_weights is not None else [1.0 / n] * n
        if len(values) != n:
            raise ValueError("fusion weight count must match channel count")
        weights = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("fusion weights must be finite and non-negative")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("fusion weights must have a positive sum")
        self._weights = (weights / total).astype(np.float32)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def embedding_scales(self) -> np.ndarray:
        """Scales whose concatenated cosine equals the weighted score sum."""
        return np.sqrt(self._weights.astype(np.float64)).astype(np.float32)

    @property
    def scorer_hash(self) -> str:
        weights = self._weights.astype(np.float64)
        weights /= float(weights.sum())
        payload = {
            "algorithm": "exact_available_intersection_weighted_cosine.v1",
            "channels": [
                {
                    "name": name,
                    "dimension": self._channel_dimensions.get(name),
                    "optional": name in self._optional_channels,
                }
                for name in self.channel_names
            ],
            "weights": weights.tolist(),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def normalized_weights(self, available_channels: set[str]) -> dict[str, float]:
        unknown = available_channels - set(self.channel_names)
        if unknown:
            raise ValueError(f"unknown fusion channels: {sorted(unknown)}")
        selected = {
            name: float(self._weights[index])
            for index, name in enumerate(self.channel_names)
            if name in available_channels
        }
        total = sum(selected.values())
        if total <= 0.0:
            raise ValueError("no positively weighted available channel remains")
        return {name: weight / total for name, weight in selected.items()}

    def fuse(self, scores: dict[str, float],
             uncertainties: dict[str, float] | None = None,
             qualities: dict[str, float] | None = None) -> float:
        if not scores:
            return 0.0
        w = self._weights.copy().astype(np.float64)
        available = np.zeros(len(self.channel_names), dtype=bool)
        for i, name in enumerate(self.channel_names):
            if name not in scores or not np.isfinite(scores[name]):
                continue
            available[i] = True
            if uncertainties and name in uncertainties:
                u = uncertainties[name]
                if not np.isfinite(u) or u < 0.0:
                    raise ValueError("uncertainty must be finite and non-negative")
                w[i] *= np.exp(-u)
            if qualities and name in qualities:
                q = qualities[name]
                if not np.isfinite(q) or not 0.0 <= q <= 1.0:
                    raise ValueError("quality must be finite and in [0, 1]")
                w[i] *= q
        w[~available] = 0.0
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("no usable channel score remains after gating")
        result = 0.0
        for i, name in enumerate(self.channel_names):
            if available[i]:
                result += (w[i] / total) * scores[name]
        return float(result)

    def update_weights(self, val_scores: np.ndarray, val_labels: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs")
        lr.fit(val_scores, val_labels)
        weights = np.abs(lr.coef_[0])
        total = float(weights.sum())
        if not np.all(np.isfinite(weights)) or total <= 0.0:
            raise RuntimeError("learned fusion weights are invalid")
        self._weights = (weights / total).astype(np.float32)
