from __future__ import annotations

import numpy as np


def compute_calibration_metrics(
    sims: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    if len(sims) == 0:
        return {"error": "empty input"}
    labels = labels.astype(np.int64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ece = 0.0
    n_total = len(sims)
    for i in range(n_bins):
        mask = (sims >= bins[i]) & (sims < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (sims == 1.0)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        acc = float(labels[mask].mean())
        conf = float(sims[mask].mean())
        ece += (n_bin / n_total) * abs(acc - conf)
    return {
        "ECE": round(ece, 4),
        "n_bins": n_bins,
        "num_pairs": len(sims),
        "positive_fraction": float(labels.mean()),
    }
