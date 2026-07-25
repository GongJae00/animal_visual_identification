from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ScoreCalibrator:
    def __init__(self) -> None:
        self._calibrators: dict[str, Any] = {}

    def fit(self, scores: dict[str, np.ndarray],
            labels: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression
        for name, vals in scores.items():
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(vals, labels.astype(np.float64))
            self._calibrators[name] = iso

    def transform(self, scores: dict[str, np.ndarray]
                  ) -> dict[str, np.ndarray]:
        return {
            name: self._calibrators[name].transform(vals)
            for name, vals in scores.items()
        }

    def calibrate(self, raw_score: float, channel: str = "all") -> float:
        if channel not in self._calibrators:
            return raw_score
        result = self._calibrators[channel].predict([[raw_score]])[0]
        return float(np.clip(result, 0.0, 1.0))

    def save(self, path: Path) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self._calibrators, f)

    @classmethod
    def load(cls, path: Path) -> ScoreCalibrator:
        import pickle
        obj = cls()
        with open(path, "rb") as f:
            obj._calibrators = pickle.load(f)
        return obj


class AdaptiveFusion:
    def __init__(self, quality_threshold: float = 0.3) -> None:
        self._quality_threshold = quality_threshold

    def fuse(self, scores: dict[str, float],
             qualities: dict[str, float] | None = None,
             weights: dict[str, float] | None = None) -> float:
        if not scores:
            return 0.0
        weights = weights or {k: 1.0 for k in scores}
        total_weight = 0.0
        fused = 0.0
        for channel, score in scores.items():
            w = weights.get(channel, 1.0)
            if qualities and channel in qualities:
                if qualities[channel] < self._quality_threshold:
                    w *= 0.1
            fused += w * score
            total_weight += w
        return fused / max(total_weight, 1e-8)


class OpenSetDecision:
    def __init__(self, distance_ratio_threshold: float = 0.15,
                 min_similarity: float = 0.5) -> None:
        self._ratio_threshold = distance_ratio_threshold
        self._min_similarity = min_similarity

    @staticmethod
    def distance_ratio(first: float, second: float) -> float:
        return (1.0 - first) / max(1.0 - second, 1e-8)

    def reject(self, top_result: tuple[int, float, dict],
               all_results: list[tuple[int, float, dict]]
               ) -> tuple[bool, str]:
        idx, score, meta = top_result
        if len(all_results) < 2:
            if score < self._min_similarity:
                return True, f"low_similarity:{score:.3f}"
            return False, ""
        second_score = all_results[1][1]
        ratio = self.distance_ratio(score, second_score)
        if ratio > self._ratio_threshold:
            return True, f"ambiguous_ratio:{ratio:.3f}"
        if score < self._min_similarity:
            return True, f"low_similarity:{score:.3f}"
        return False, ""


class TemporalAggregator:
    def __init__(self, aggregation: str = "median") -> None:
        allowed = {"mean", "median", "geometric_mean"}
        if aggregation not in allowed:
            raise ValueError(f"aggregation must be one of {allowed}")
        self._aggregation = aggregation

    def aggregate(self, embeddings: list[np.ndarray]) -> np.ndarray:
        if not embeddings:
            raise ValueError("embeddings must be non-empty")
        stacked = np.stack([e.ravel() for e in embeddings])
        if self._aggregation == "mean":
            return stacked.mean(axis=0)
        elif self._aggregation == "median":
            return np.median(stacked, axis=0)
        else:
            log = np.log(stacked.clip(min=1e-8))
            return np.exp(log.mean(axis=0))

    def aggregate_weighted(self, embeddings: list[np.ndarray],
                           qualities: list[float]) -> np.ndarray:
        if not embeddings:
            raise ValueError("embeddings must be non-empty")
        stacked = np.stack([e.ravel() for e in embeddings])
        w = np.array(qualities, dtype=np.float32)
        w /= max(w.sum(), 1e-8)
        return (stacked.T * w).sum(axis=1)
