from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_verification_metrics(
    sims: np.ndarray,
    labels: np.ndarray,
) -> dict:
    if len(sims) == 0:
        return {"error": "empty input", "num_pairs": 0}
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    pos_sims = sims[labels == 1]
    neg_sims = sims[labels == 0]
    if n_pos == 0 or n_neg == 0:
        return {
            "num_pairs": len(sims),
            "num_positive": n_pos,
            "num_negative": n_neg,
            "error": "single-class input",
        }
    try:
        auc = float(roc_auc_score(labels, sims))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(labels, sims))
    except ValueError:
        ap = float("nan")
    thresholds = np.linspace(0.0, 1.0, 1001)
    tpr_curve = np.zeros_like(thresholds)
    fpr_curve = np.zeros_like(thresholds)
    fnr_curve = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        pred = (sims >= t).astype(np.int64)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        tpr_curve[i] = tp / max(tp + fn, 1)
        fpr_curve[i] = fp / max(fp + tn, 1)
        fnr_curve[i] = fn / max(fn + tp, 1)
    eer_idx = int(np.argmin(np.abs(fpr_curve - fnr_curve)))
    eer = float((fpr_curve[eer_idx] + fnr_curve[eer_idx]) / 2)
    tar_at_far = {}
    for target_far in [0.001, 0.01, 0.1]:
        valid = np.where(fpr_curve <= target_far)[0]
        if len(valid) > 0:
            tar_at_far[f"TAR@FAR={target_far:.0e}"] = float(tpr_curve[valid[-1]])
        else:
            tar_at_far[f"TAR@FAR={target_far:.0e}"] = 0.0
    d_prime = float(
        (pos_sims.mean() - neg_sims.mean())
        / max(np.sqrt((pos_sims.var() + neg_sims.var()) / 2), 1e-8)
    )
    return {
        "num_pairs": len(sims),
        "num_positive": n_pos,
        "num_negative": n_neg,
        "mean_positive_sim": float(pos_sims.mean()),
        "mean_negative_sim": float(neg_sims.mean()),
        "d_prime": round(d_prime, 4),
        "AUC": round(auc, 4),
        "AP": round(ap, 4),
        "EER": round(eer, 4),
        **tar_at_far,
    }
