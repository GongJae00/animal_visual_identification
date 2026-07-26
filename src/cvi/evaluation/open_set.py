from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class OpenSetError(ValueError):
    pass


@dataclass(frozen=True)
class OpenSetResult:
    known_detection_auroc: float
    known_detection_aupr: float
    dir_at_fpir: dict[str, float]
    fpir_thresholds: dict[str, float]
    per_target: dict[str, dict]
    known_correct_accept_count: int
    known_misidentification_count: int
    known_rejection_count: int
    unknown_accept_count: int
    unknown_rejection_count: int
    num_enrolled_queries: int
    num_unknown_queries: int
    num_gallery_identities: int


def _normalize_rows(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    zero = norms.ravel() < 1e-8
    if zero.any():
        n_zero = int(zero.sum())
        s = "s" if n_zero == 1 else "ve"
        raise OpenSetError(
            f"{n_zero} embedding(s) ha{s} zero norm"
        )
    return embs / norms


def _classify_queries(
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
) -> np.ndarray:
    enrolled = set(gallery_ids.tolist())
    return np.array([qid in enrolled for qid in query_ids], dtype=bool)


def _compute_detection_metrics(
    max_scores: np.ndarray,
    is_known: np.ndarray,
) -> tuple[float, float]:
    n_known = int(is_known.sum())
    n_unknown = int((~is_known).sum())
    if n_known == 0 or n_unknown == 0:
        return 0.0, 0.0
    labels = np.concatenate([
        np.ones(n_known, dtype=np.int64),
        np.zeros(n_unknown, dtype=np.int64),
    ])
    scores = np.concatenate([
        max_scores[is_known],
        max_scores[~is_known],
    ])
    auroc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))
    return auroc, aupr


def _select_thresholds_from_calibration(
    cal_query_embs: np.ndarray,
    cal_gallery_embs: np.ndarray,
    cal_query_ids: np.ndarray,
    cal_gallery_ids: np.ndarray,
    fpir_targets: tuple[float, ...],
) -> dict[str, dict]:
    cal_q = _normalize_rows(cal_query_embs)
    cal_g = _normalize_rows(cal_gallery_embs)
    cal_sims = cal_q @ cal_g.T
    cal_max = np.max(cal_sims, axis=1)
    cal_is_known = _classify_queries(cal_query_ids, cal_gallery_ids)
    n_known = int(cal_is_known.sum())
    n_unknown = int((~cal_is_known).sum())
    if n_known == 0 or n_unknown == 0:
        raise OpenSetError(
            f"calibration needs both known ({n_known}) and unknown "
            f"({n_unknown}) queries"
        )
    known_correct_scores = []
    for i in np.where(cal_is_known)[0]:
        same_id = cal_gallery_ids == cal_query_ids[i]
        if same_id.any():
            known_correct_scores.append(float(np.max(cal_sims[i][same_id])))
        else:
            known_correct_scores.append(-1.0)
    known_correct_arr = np.array(known_correct_scores, dtype=np.float64)
    unknown_scores = cal_max[~cal_is_known]
    all_scores = np.concatenate([known_correct_arr, unknown_scores])
    all_labels = np.concatenate([
        np.ones(n_known, dtype=np.int64),
        np.zeros(n_unknown, dtype=np.int64),
    ])
    sorted_idx = np.argsort(-all_scores)
    sorted_labels = all_labels[sorted_idx]
    fp = np.cumsum(sorted_labels == 0)
    tp = np.cumsum(sorted_labels == 1)
    far = fp / max(n_unknown, 1)
    dir_rate = tp / max(n_known, 1)
    per_target: dict[str, dict] = {}
    for target in fpir_targets:
        valid = np.where(far <= target)[0]
        if len(valid) > 0:
            threshold = float(all_scores[sorted_idx[valid[-1]]])
            cal_dir = float(dir_rate[valid[-1]])
            cal_fpir = float(far[valid[-1]])
        else:
            threshold = float(all_scores[sorted_idx[-1]] + 1.0)
            cal_dir = 0.0
            cal_fpir = 0.0
        per_target[str(target)] = {
            "target_fpir": target,
            "selected_threshold": threshold,
            "calibration": {
                "known_queries": n_known,
                "unknown_queries": n_unknown,
                "correct_known_accepts": int(np.sum(
                    (known_correct_arr >= threshold)
                )),
                "unknown_accepts": int(np.sum(unknown_scores >= threshold)),
                "DIR": cal_dir,
                "FPIR": cal_fpir,
            },
        }
    return per_target


def _evaluate_on_test(
    test_query_embs: np.ndarray,
    test_gallery_embs: np.ndarray,
    test_query_ids: np.ndarray,
    test_gallery_ids: np.ndarray,
    per_target: dict[str, dict],
) -> tuple[
    dict[str, dict],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    q = _normalize_rows(test_query_embs)
    g = _normalize_rows(test_gallery_embs)
    sims = q @ g.T
    max_scores = np.max(sims, axis=1)
    top1_indices = np.argmax(sims, axis=1)
    top1_gallery_ids = test_gallery_ids[top1_indices]
    is_known = _classify_queries(test_query_ids, test_gallery_ids)
    n_known = int(is_known.sum())
    n_unknown = int((~is_known).sum())
    top1_is_correct = np.array([
        top1_gallery_ids[i] == test_query_ids[i] if is_known[i] else False
        for i in range(len(test_query_ids))
    ], dtype=bool)
    for target_key, info in per_target.items():
        t = info["selected_threshold"]
        correct_accept = 0
        misid = 0
        rejection = 0
        unk_accept = 0
        unk_reject = 0
        for i in range(len(max_scores)):
            if is_known[i]:
                if top1_is_correct[i] and max_scores[i] >= t:
                    correct_accept += 1
                elif not top1_is_correct[i]:
                    misid += 1
                else:
                    rejection += 1
            else:
                if max_scores[i] >= t:
                    unk_accept += 1
                else:
                    unk_reject += 1
        test_dir = correct_accept / max(n_known, 1)
        test_fpir = unk_accept / max(n_unknown, 1)
        info["test"] = {
            "known_queries": n_known,
            "unknown_queries": n_unknown,
            "correct_known_accepts": correct_accept,
            "known_misidentifications": misid,
            "known_rejections": rejection,
            "unknown_accepts": unk_accept,
            "unknown_rejections": unk_reject,
            "DIR": test_dir,
            "FPIR": test_fpir,
        }
    return per_target, max_scores, is_known, top1_is_correct, top1_gallery_ids


def evaluate_open_set(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    fpir_targets: tuple[float, ...] = (0.01, 0.001),
    calibration_query_embs: np.ndarray | None = None,
    calibration_gallery_embs: np.ndarray | None = None,
    calibration_query_ids: np.ndarray | None = None,
    calibration_gallery_ids: np.ndarray | None = None,
) -> OpenSetResult:
    n_query = len(query_embs)
    n_gallery = len(gallery_embs)
    if n_query == 0:
        raise OpenSetError("empty query set")
    if n_gallery == 0:
        raise OpenSetError("empty gallery set")
    has_calibration = all(x is not None for x in [
        calibration_query_embs, calibration_gallery_embs,
        calibration_query_ids, calibration_gallery_ids,
    ])
    if has_calibration:
        per_target = _select_thresholds_from_calibration(
            calibration_query_embs,
            calibration_gallery_embs,
            calibration_query_ids,
            calibration_gallery_ids,
            fpir_targets,
        )
    else:
        per_target = _select_thresholds_from_calibration(
            query_embs, gallery_embs, query_ids, gallery_ids,
            fpir_targets,
        )
    per_target, max_scores, is_known, top1_is_correct, top1_gallery_ids = _evaluate_on_test(
        query_embs, gallery_embs, query_ids, gallery_ids,
        per_target,
    )
    n_known = int(is_known.sum())
    n_unknown = int((~is_known).sum())
    if n_known == 0 or n_unknown == 0:
        raise OpenSetError(
            f"test needs both known ({n_known}) and unknown ({n_unknown}) queries"
        )
    known_detection_auroc, known_detection_aupr = _compute_detection_metrics(
        max_scores, is_known,
    )
    dir_at_fpir: dict[str, float] = {}
    fpir_thresholds: dict[str, float] = {}
    for target_key, info in per_target.items():
        dir_at_fpir[f"DIR@FPIR={target_key}"] = info["test"]["DIR"]
        fpir_thresholds[f"threshold@FPIR={target_key}"] = info["selected_threshold"]

    first_target_key = next(iter(per_target.keys()))
    first_info = per_target[first_target_key]["test"]
    known_correct_accept = first_info["correct_known_accepts"]
    known_misidentification = first_info["known_misidentifications"]
    known_rejection = first_info["known_rejections"]
    unknown_accept = first_info["unknown_accepts"]
    unknown_rejection = first_info["unknown_rejections"]
    enrolled_ids = set(gallery_ids.tolist())
    return OpenSetResult(
        known_detection_auroc=known_detection_auroc,
        known_detection_aupr=known_detection_aupr,
        dir_at_fpir=dir_at_fpir,
        fpir_thresholds=fpir_thresholds,
        per_target=per_target,
        known_correct_accept_count=known_correct_accept,
        known_misidentification_count=known_misidentification,
        known_rejection_count=known_rejection,
        unknown_accept_count=unknown_accept,
        unknown_rejection_count=unknown_rejection,
        num_enrolled_queries=n_known,
        num_unknown_queries=n_unknown,
        num_gallery_identities=len(enrolled_ids),
    )
