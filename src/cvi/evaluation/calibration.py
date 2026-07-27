from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression


class CalibrationError(ValueError):
    pass


class InvalidProbabilityError(CalibrationError):
    pass


def _validate_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    if len(probabilities) == 0:
        raise CalibrationError("empty probability array")
    if len(probabilities) != len(labels):
        raise CalibrationError(
            f"probabilities length {len(probabilities)} != labels length {len(labels)}"
        )
    if probabilities.ndim != 1:
        raise CalibrationError(
            f"probabilities must be 1-d, got shape {probabilities.shape}"
        )
    if labels.ndim != 1:
        raise CalibrationError(f"labels must be 1-d, got shape {labels.shape}")
    if not np.all(np.isfinite(probabilities)):
        raise CalibrationError("probabilities contain non-finite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise InvalidProbabilityError(
            f"probabilities must be in [0, 1], "
            f"got range [{float(probabilities.min())}, {float(probabilities.max())}]"
        )
    if not np.all(np.isfinite(labels)) or not np.all((labels == 0) | (labels == 1)):
        raise CalibrationError(
            f"labels must be in {{0, 1}}, got {np.unique(labels)}"
        )
    return labels.astype(np.int64)


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


def compute_probability_calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        raise CalibrationError("n_bins must be a positive integer")
    lb = _validate_probabilities(probabilities, labels)
    n = len(probabilities)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_counts = []
    bin_confidences = []
    bin_positive_rates = []
    for i in range(n_bins):
        mask = (probabilities >= bins[i]) & (probabilities < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (probabilities == 1.0)
        n_bin = int(mask.sum())
        bin_counts.append(n_bin)
        if n_bin == 0:
            bin_confidences.append(0.0)
            bin_positive_rates.append(0.0)
            continue
        conf = float(probabilities[mask].mean())
        acc = float(lb[mask].mean())
        bin_confidences.append(conf)
        bin_positive_rates.append(acc)
        ece += (n_bin / n) * abs(acc - conf)
    brier = float(brier_score_loss(lb, probabilities))
    eps = 1e-15
    p_clip = np.clip(probabilities, eps, 1 - eps)
    nll = -float(np.mean(
        lb * np.log(p_clip) + (1 - lb) * np.log(1 - p_clip)
    ))
    return {
        "ECE": ece,
        "Brier": brier,
        "NLL": nll,
        "n_bins": n_bins,
        "num_pairs": n,
        "positive_fraction": float(lb.mean()),
        "bin_counts": bin_counts,
        "bin_confidences": bin_confidences,
        "bin_positive_rates": bin_positive_rates,
    }
