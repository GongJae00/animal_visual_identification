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
    n_pos: int
    n_neg: int


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
) -> tuple[int, int]:
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
    if not np.all((labels == 0) | (labels == 1)):
        raise InvalidLabelError(
            f"labels must be in {{0, 1}}, got values {np.unique(labels)}"
        )
    if not np.all(np.isfinite(scores)):
        raise NonFiniteScoreError("scores contain non-finite values")
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if require_both_classes and (n_pos == 0 or n_neg == 0):
        raise SingleClassError(
            f"both classes required: n_pos={n_pos}, n_neg={n_neg}"
        )
    return n_pos, n_neg


def compute_verification_curve(
    scores: np.ndarray,
    labels: np.ndarray,
) -> VerificationCurve:
    n_pos, n_neg = _validate_scores_labels(scores, labels)
    lb = labels.astype(np.int64)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = lb[order]
    unique, first_indices, counts = np.unique(
        sorted_scores, return_index=True, return_counts=True
    )
    group_order = np.argsort(first_indices)
    unique = unique[group_order]
    counts = counts[group_order]
    thresholds = np.concatenate([
        [np.inf],
        unique,
        [-np.inf],
    ])
    far_arr = np.empty(len(thresholds), dtype=np.float64)
    frr_arr = np.empty(len(thresholds), dtype=np.float64)
    tar_arr = np.empty(len(thresholds), dtype=np.float64)
    tp = 0
    fp = 0
    far_arr[0] = 0.0
    frr_arr[0] = 1.0
    tar_arr[0] = 0.0
    offset = 0
    for position, count in enumerate(counts, start=1):
        group_labels = sorted_labels[offset:offset + count]
        tp += int(np.sum(group_labels == 1))
        fp += int(np.sum(group_labels == 0))
        far_arr[position] = fp / n_neg
        tar_arr[position] = tp / n_pos
        frr_arr[position] = 1.0 - tar_arr[position]
        offset += int(count)
    far_arr[-1] = 1.0
    frr_arr[-1] = 0.0
    tar_arr[-1] = 1.0
    return VerificationCurve(
        thresholds=thresholds,
        far=far_arr,
        frr=frr_arr,
        tar=tar_arr,
        n_pos=n_pos,
        n_neg=n_neg,
    )


def select_threshold_at_far(
    scores: np.ndarray,
    labels: np.ndarray,
    target_far: float,
) -> OperatingThreshold:
    if not 0.0 <= target_far <= 1.0:
        raise EvaluationError(f"target_far must be in [0, 1], got {target_far}")
    n_pos, n_neg = _validate_scores_labels(scores, labels)
    lb = labels.astype(np.int64)
    curve = compute_verification_curve(scores, lb)
    valid = np.where(curve.far <= target_far)[0]
    if len(valid) == 0:
        idx = len(curve.thresholds) - 1
    else:
        idx = valid[np.argmax(curve.tar[valid])]
    t = float(curve.thresholds[idx])
    if not np.isfinite(t):
        raise EvaluationError(
            "no finite verification threshold satisfies the requested FAR"
        )
    pred = (scores >= t).astype(np.int64)
    tp = int(((pred == 1) & (lb == 1)).sum())
    fp = int(((pred == 1) & (lb == 0)).sum())
    fn = n_pos - tp
    cal_far = fp / max(n_neg, 1)
    cal_tar = tp / max(n_pos, 1)
    return OperatingThreshold(
        threshold=t,
        target_far=target_far,
        calibration_far=cal_far,
        calibration_tar=cal_tar,
        calibration_num_negative=n_neg,
        calibration_false_accepts=fp,
        calibration_num_positive=n_pos,
        calibration_false_rejects=fn,
    )


def evaluate_at_threshold(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
) -> dict:
    n_pos, n_neg = _validate_scores_labels(test_scores, test_labels)
    lb = test_labels.astype(np.int64)
    pred = (test_scores >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (lb == 1)).sum())
    fp = int(((pred == 1) & (lb == 0)).sum())
    fn = n_pos - tp
    tn = n_neg - fp
    tar = tp / max(n_pos, 1)
    far = fp / max(n_neg, 1)
    frr = fn / max(n_pos, 1)
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
    n_pos, n_neg = _validate_scores_labels(scores, labels)
    lb = labels.astype(np.int64)
    pos_scores = scores[lb == 1]
    neg_scores = scores[lb == 0]
    auc = float(roc_auc_score(lb, scores))
    ap = float(average_precision_score(lb, scores))
    curve = compute_verification_curve(scores, lb)
    finite_indices = np.flatnonzero(np.isfinite(curve.thresholds))
    eer_idx = int(finite_indices[
        np.argmin(np.abs(curve.far[finite_indices] - curve.frr[finite_indices]))
    ])
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
