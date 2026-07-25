from __future__ import annotations

import numpy as np


class EvidentialOpenSet:
    def __init__(self, epistemic_threshold: float = 0.3,
                 distance_ratio_threshold: float = 0.15,
                 min_similarity: float = 0.4,
                 calibrator: object | None = None):
        self._epi_thresh = epistemic_threshold
        self._ratio_thresh = distance_ratio_threshold
        self._min_sim = min_similarity
        self._calibrator = calibrator

    def reject(self, top_result: tuple[int, float, dict],
               all_results: list[tuple[int, float, dict]],
               epistemic: float | None = None) -> tuple[bool, str]:
        sim = top_result[1]
        if sim < self._min_sim:
            return True, "below_min_similarity"
        if epistemic is not None and epistemic > self._epi_thresh:
            return True, "high_epistemic_uncertainty"
        if len(all_results) >= 2:
            top_sim = top_result[1]
            second_sim = all_results[1][1] if all_results[0][0] == top_result[0] else all_results[0][1]
            ratio = (1.0 - top_sim) / max(1.0 - second_sim, 1e-8)
            if ratio > self._ratio_thresh:
                return True, "high_distance_ratio"
        return False, ""

    def estimate_far_frr(self, sim_scores: np.ndarray, labels: np.ndarray,
                         thresholds: np.ndarray | None = None
                         ) -> tuple[np.ndarray, np.ndarray, float]:
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.99, 200)
        far, frr = [], []
        for thresh in thresholds:
            fp = np.sum((sim_scores >= thresh) & (labels == 0))
            fn = np.sum((sim_scores < thresh) & (labels == 1))
            tp = np.sum((sim_scores >= thresh) & (labels == 1))
            tn = np.sum((sim_scores < thresh) & (labels == 0))
            far.append(fp / max(fp + tn, 1))
            frr.append(fn / max(fn + tp, 1))
        far = np.array(far)
        frr = np.array(frr)
        eer_idx = np.argmin(np.abs(far - frr))
        eer = (far[eer_idx] + frr[eer_idx]) / 2
        return far, frr, eer

    def optimal_threshold(self, sim_scores: np.ndarray, labels: np.ndarray,
                          target_far: float = 0.01) -> float:
        far, frr, _ = self.estimate_far_frr(sim_scores, labels)
        thresholds = np.linspace(0.1, 0.99, 200)
        valid_idx = np.where(far <= target_far)[0]
        if len(valid_idx) == 0:
            return thresholds[-1]
        best_idx = valid_idx[np.argmin(frr[valid_idx])]
        return float(thresholds[best_idx])
