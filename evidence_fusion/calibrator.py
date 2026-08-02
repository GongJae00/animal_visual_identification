from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from foundation.protected_io import read_strict_json_object, write_private_json_bundle


_SCHEMA_VERSION = "cvi.isotonic_calibrator.v1"


class CalibrationError(ValueError):
    pass


def fit_isotonic_calibration(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
) -> IsotonicRegression:
    if cal_scores.ndim != 1 or cal_labels.ndim != 1:
        raise CalibrationError("calibration scores and labels must be 1-d")
    if len(cal_scores) != len(cal_labels):
        raise CalibrationError("calibration score and label lengths differ")
    if len(cal_scores) < 2:
        raise CalibrationError(f"need >= 2 calibration samples, got {len(cal_scores)}")
    if not np.all(np.isfinite(cal_scores)):
        raise CalibrationError("calibration scores contain non-finite values")
    if not np.all(np.isfinite(cal_labels)) or not np.all(
        (cal_labels == 0) | (cal_labels == 1)
    ):
        raise CalibrationError("calibration labels must be exactly {0, 1}")
    if len(np.unique(cal_labels)) != 2:
        raise CalibrationError("calibration requires both classes")
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(cal_scores, cal_labels.astype(np.int64))
    return model


class PerChannelCalibrator:
    def __init__(self) -> None:
        self._calibrators: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, scores: dict[str, np.ndarray], labels: np.ndarray) -> None:
        if not scores:
            raise ValueError("at least one calibration channel is required")
        fitted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, values in sorted(scores.items()):
            if not isinstance(name, str) or not name:
                raise ValueError("calibration channel names must be non-empty")
            array = np.asarray(values, dtype=np.float64)
            model = fit_isotonic_calibration(array, np.asarray(labels))
            fitted[name] = (
                np.asarray(model.X_thresholds_, dtype=np.float64),
                np.asarray(model.y_thresholds_, dtype=np.float64),
            )
        self._calibrators = fitted

    def transform(self, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for name, values in scores.items():
            if name not in self._calibrators:
                raise KeyError(f"channel {name!r} has no fitted calibrator")
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError("calibration scores must be a finite vector")
            x, y = self._calibrators[name]
            output[name] = np.interp(array, x, y, left=y[0], right=y[-1])
        return output

    def calibrate(self, raw_score: float, channel: str = "all") -> float:
        if channel not in self._calibrators:
            raise KeyError(f"channel {channel!r} has no fitted calibrator")
        if not np.isfinite(raw_score):
            raise ValueError("raw score must be finite")
        x, y = self._calibrators[channel]
        return float(np.interp(raw_score, x, y, left=y[0], right=y[-1]))

    def save(self, path: Path) -> None:
        if not self._calibrators:
            raise RuntimeError("cannot save an unfitted calibrator")
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "channels": {
                name: {
                    "x_thresholds": x.tolist(),
                    "y_thresholds": y.tolist(),
                }
                for name, (x, y) in sorted(self._calibrators.items())
            },
        }
        write_private_json_bundle(((path, payload),))

    @classmethod
    def load(cls, path: Path) -> "PerChannelCalibrator":
        payload = read_strict_json_object(path)
        if set(payload) != {"schema_version", "channels"} or payload[
            "schema_version"
        ] != _SCHEMA_VERSION:
            raise ValueError("unsupported calibrator schema")
        channels = payload["channels"]
        if not isinstance(channels, dict) or not channels:
            raise ValueError("calibrator channels must be a non-empty object")
        obj = cls()
        for name, state in sorted(channels.items()):
            if not isinstance(name, str) or not name or not isinstance(state, dict):
                raise ValueError("invalid calibrator channel")
            if set(state) != {"x_thresholds", "y_thresholds"}:
                raise ValueError("invalid calibrator channel schema")
            x = np.asarray(state["x_thresholds"], dtype=np.float64)
            y = np.asarray(state["y_thresholds"], dtype=np.float64)
            if (
                x.ndim != 1
                or y.ndim != 1
                or len(x) < 1
                or len(x) != len(y)
                or not np.all(np.isfinite(x))
                or not np.all(np.isfinite(y))
                or not np.all(np.diff(x) > 0.0)
                or not np.all(np.diff(y) >= 0.0)
                or np.any((y < 0.0) | (y > 1.0))
            ):
                raise ValueError("invalid isotonic calibrator thresholds")
            obj._calibrators[name] = (x, y)
        return obj
