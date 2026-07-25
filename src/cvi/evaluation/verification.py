from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class EvaluationError(ValueError):
    pass


class LengthMismatchError(EvaluationError):
    pass


class InvalidLabelError(EvaluationError):
    pass


class NonFiniteScoreError(EvaluationError):
    pass


class EmptyInputError(EvaluationError):
    pass


class SingleClassError(EvaluationError):
    pass


@dataclass(frozen=True)
class VerificationCurve:
    thresholds: np.ndarray
    far: np.ndarray
    frr: np.ndarray
    tar: np.ndarray


@dataclass(frozen=True)
class OperatingThreshold:
    threshold: float
    target_far: float
    calibration_far: float
    calibration_tar: float
    calibration_num_negative: int
    calibration_false_accepts: int
    calibration_num_positive: int
    calibration_false_rejects: int


def _validate_scores_labels(
    scores: np.ndarray,
    labels: np.ndarray,
    require_both_classes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if len(scores) != len(labels):
        raise LengthMismatchError(
            f"scores length {len(scores)} != labels length {len(labels)}"
        )
    if scores.ndim != 1:
        raise EvaluationError(
            f"scores must be 1-d, got shape {scores.shape}"
        )
    if labels.ndim != 1:
        raise EvaluationError(
            f"labels must be 1-d, got shape {labels.shape}"
        )
    if len(scores) == 0:
        raise EmptyInputError("scores and labels are empty")
    lb = labels.astype(np.int64)
    if not np.all((lb == 0) | (lb == 1)):
        raise InvalidLabelError(
            f"labels must be in {{0, 1}}, got values {np.unique(lb)}"
        )
    if not np.all(np.isfinite(scores)):
        raise NonFiniteScoreError("scores contain non-finite values")
    n_pos = int(lb.sum())
    n_neg = int((1 - lb).sum())
    if require_both_classes and (n_pos == 0 or n_neg == 0):
        raise SingleClassError(
            f"both classes required: n_pos={n_pos}, n_neg={n_neg}"
        )
    return lb, n_pos, n_neg


def compute_verification_curve(
    scores: np.ndarray,
    labels: np.ndarray,
) -> VerificationCurve:
    lb, n_pos, n_neg = _validate_scores_labels(scores, labels)
    unique = np.sort(np.unique(scores))[::-1]
    thresholds = np.concatenate([
        [np.inf],
        unique,
        [-np.inf],
    ])
    n = len(thresholds)
    far_arr = np.zeros(n, dtype=np.float64)
    frr_arr = np.zeros(n, dtype=np.float64)
    tar_arr = np.zeros(n, dtype=np.float64)
    for i, t in enumerate(thresholds):
        pred = (scores >= t).astype(np.int64)
        tp = ((pred == 1) & (lb == 1)).sum()
        fp = ((pred == 1) & (lb == 0)).sum()
        fn = ((pred == 0) & (lb == 1)).sum()
        tn = ((pred == 0) & (lb == 0)).sum()
        far_arr[i] = fp / max(fp + tn, 1)
        frr_arr[i] = fn / max(fn + tp, 1)
        tar_arr[i] = tp / max(tp + fn, 1)
    return VerificationCurve(
        thresholds=thresholds,
        far=far_arr,
        frr=frr_arr,
        tar=tar_arr,
    )


def select_threshold_at_far(
    calibration_curve: VerificationCurve,
    target_far: float,
) -> OperatingThreshold:
    valid = np.where(calibration_curve.far <= target_far)[0]
    if len(valid) == 0:
        valid = np.array([len(calibration_curve.thresholds) - 1])
    max_tar_idx = valid[np.argmax(calibration_curve.tar[valid])]
    t = float(calibration_curve.thresholds[max_tar_idx])
    cal_far = float(calibration_curve.far[max_tar_idx])
    cal_tar = float(calibration_curve.tar[max_tar_idx])
    far_at_max = calibration_curve.far[max_tar_idx]
    frr_at_max = calibration_curve.frr[max_tar_idx]
    n_pos_approx = None
    n_neg_approx = None
    return OperatingThreshold(
        threshold=t,
        target_far=target_far,
        calibration_far=cal_far,
        calibration_tar=cal_tar,
        calibration_num_negative=-1,
        calibration_false_accepts=-1,
        calibration_num_positive=-1,
        calibration_false_rejects=-1,
    )


def evaluate_at_threshold(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
) -> dict:
    lb, n_pos, n_neg = _validate_scores_labels(test_scores, test_labels)
    pred = (test_scores >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (lb == 1)).sum())
    fp = int(((pred == 1) & (lb == 0)).sum())
    fn = int(((pred == 0) & (lb == 1)).sum())
    tn = int(((pred == 0) & (lb == 0)).sum())
    tar = tp / max(tp + fn, 1)
    far = fp / max(fp + tn, 1)
    frr = fn / max(fn + tp, 1)
    return {
        "threshold": float(threshold),
        "true_accepts": tp,
        "false_accepts": fp,
        "false_rejects": fn,
        "true_rejects": tn,
        "num_positive": n_pos,
        "num_negative": n_neg,
        "TAR": float(tar),
        "FAR": float(far),
        "FRR": float(frr),
    }


def compute_verification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    lb, n_pos, n_neg = _validate_scores_labels(scores, labels)
    pos_scores = scores[lb == 1]
    neg_scores = scores[lb == 0]
    auc = float(roc_auc_score(lb, scores))
    ap = float(average_precision_score(lb, scores))
    curve = compute_verification_curve(scores, lb)
    eer_idx = int(np.argmin(np.abs(curve.far - curve.frr)))
    eer = float((curve.far[eer_idx] + curve.frr[eer_idx]) / 2)
    eer_threshold = float(curve.thresholds[eer_idx])
    d_prime_numer = pos_scores.mean() - neg_scores.mean()
    d_prime_denom = max(
        np.sqrt((pos_scores.var(ddof=0) + neg_scores.var(ddof=0)) / 2), 1e-8
    )
    d_prime = float(d_prime_numer / d_prime_denom)
    return {
        "num_pairs": len(scores),
        "num_positive": n_pos,
        "num_negative": n_neg,
        "mean_positive_score": float(pos_scores.mean()),
        "mean_negative_score": float(neg_scores.mean()),
        "std_positive_score": float(pos_scores.std(ddof=0)),
        "std_negative_score": float(neg_scores.std(ddof=0)),
        "ROC_AUC": auc,
        "PR_AUC": ap,
        "EER": eer,
        "EER_threshold": eer_threshold,
        "d_prime": d_prime,
    }
