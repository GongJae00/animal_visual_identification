from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class PerChannelCalibrator:
    def __init__(self):
        self._calibrators: dict[str, Any] = {}

    def fit(self, scores: dict[str, np.ndarray], labels: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression
        for name, vals in scores.items():
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(vals, labels.astype(np.float64))
            self._calibrators[name] = iso

    def transform(self, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            name: self._calibrators[name].transform(vals)
            for name, vals in scores.items() if name in self._calibrators
        }

    def calibrate(self, raw_score: float, channel: str = "all") -> float:
        if channel not in self._calibrators:
            return raw_score
        result = self._calibrators[channel].predict([[raw_score]])[0]
        return float(np.clip(result, 0.0, 1.0))

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._calibrators, f)

    @classmethod
    def load(cls, path: Path) -> PerChannelCalibrator:
        obj = cls()
        with open(path, "rb") as f:
            obj._calibrators = pickle.load(f)
        return obj
