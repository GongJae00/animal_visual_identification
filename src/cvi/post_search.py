from __future__ import annotations

import numpy as np

from cvi.fusion.calibrator import PerChannelCalibrator


ScoreCalibrator = PerChannelCalibrator


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
