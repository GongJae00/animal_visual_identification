"""Multi-channel evaluation framework.

Protocols:
  evaluate verification   --calibration-pairs CAL  --test-pairs TEST  ...
  evaluate retrieval      --gallery FILE  --queries FILE  ...
  evaluate open-set       --gallery FILE  --calibration-queries CAL  --test-queries TEST  ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from PIL import Image

from cvi.evaluation._legacy import (
    required_zero_event_trials,
    wilson_rate,
    zero_event_exact_upper_bound,
)
from cvi.evaluation.calibration import CalibrationError, compute_probability_calibration_metrics, fit_isotonic_calibration
from cvi.evaluation.open_set import OpenSetError, OpenSetResult, evaluate_open_set
from cvi.evaluation.retrieval import RetrievalError, compute_retrieval_metrics
from cvi.evaluation.verification import (
    EvaluationError,
    compute_verification_curve,
    compute_verification_metrics,
    evaluate_at_threshold,
    select_threshold_at_far,
)
from cvi.evidence.appearance import Dinov2WithUncertainty
from cvi.evidence.base import AbstractEvidencer
from cvi.evidence.landmark_graph import LandmarkEvidencer

SCHEMA_VERSION = "cvi.evaluation.report.v2"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "cvi.evaluation.report.v2.schema.json"


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception as exc:
        print(json.dumps({"warning": f"git commit failed: {exc}"}), file=sys.stderr)
        return "__GIT_FAILED__"


def _git_branch() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception as exc:
        print(json.dumps({"warning": f"git branch failed: {exc}"}), file=sys.stderr)
        return "__GIT_FAILED__"


def _git_dirty() -> bool | str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5,
        )
        return len(result.stdout.strip()) > 0
    except Exception as exc:
        print(json.dumps({"warning": f"git dirty check failed: {exc}"}), file=sys.stderr)
        return "__GIT_FAILED__"


def _file_sha256(path: Path) -> dict:
    r: dict = {"path": str(path)}
    try:
        r["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        r["status"] = "VERIFIED"
    except Exception as exc:
        r["sha256"] = None
        r["status"] = "UNVERIFIED"
        r["reason"] = str(exc)
    return r


def _provenance(start: str | None = None) -> dict:
    p = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "dirty_state": _git_dirty(),
    }
    p["start_timestamp"] = start or datetime.now(timezone.utc).isoformat()
    p["python_version"] = sys.version.split()[0]
    p["numpy_version"] = np.__version__
    p["scikit_learn_version"] = __import__("sklearn").__version__
    try:
        p["jsonschema_version"] = __import__("jsonschema").__version__
    except Exception:
        p["jsonschema_version"] = "N/A"
    p["platform"] = platform.platform()
    p["machine"] = platform.machine()
    p["processor"] = platform.processor() or platform.machine()
    try:
        p["python_argv"] = " ".join(sys.argv)
    except Exception:
        p["python_argv"] = "N/A"
    try:
        p["cwd"] = str(Path.cwd())
    except Exception:
        p["cwd"] = "N/A"
    p["baseline_commit"] = "0ba3b1bef4ad6bd18ee516260cf938e9e43ca659"
    return p


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class ReportSchemaValidationError(ValueError):
    pass


def _validate_report(report: dict) -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"schema not found: {SCHEMA_PATH}")
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda e: list(e.path))
    if errors:
        lines = ["; ".join(e.message for e in errors)]
        raise ReportSchemaValidationError("schema validation failed: " + "; ".join(lines))
    report["schema_sha256"] = hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# CIs
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values_per_query: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 7,
    unit: str = "query",
) -> dict:
    if len(values_per_query) == 0:
        return {"estimate": 0.0, "ci_method": "none", "bootstrap_unit": unit}
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(values_per_query, size=len(values_per_query), replace=True).mean()
        for _ in range(n_resamples)
    ], dtype=np.float64)
    alpha = 1 - confidence_level
    low = float(np.percentile(means, 100 * alpha / 2))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {
        "estimate": float(np.mean(values_per_query)),
        "lower_bound": low,
        "upper_bound": high,
        "n_resamples": n_resamples,
        "ci_method": "bootstrap_percentile",
        "confidence_level": confidence_level,
        "bootstrap_unit": unit,
        "bootstrap_seed": seed,
    }


def _wilson_ci(events: int, trials: int, level: float = 0.95) -> dict:
    if trials == 0:
        return {"estimate": 0.0, "ci_method": "none"}
    est = wilson_rate(events, trials, confidence_level=level)
    return {
        "estimate": est.estimate,
        "lower_bound": est.lower_bound,
        "upper_bound": est.upper_bound,
        "interval_method": est.interval_method,
        "confidence_level": level,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"pairs file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list, got {type(data)}")
    if len(data) == 0:
        raise ValueError("empty pairs list")
    return data


def load_embedding_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    for k in ("embeddings", "identities"):
        if k not in data:
            raise ValueError(f"manifest missing '{k}'")
    return data


# ---------------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------------

def validate_split_disjoint(
    cal: list[dict],
    test: list[dict],
) -> list[str]:
    warnings: list[str] = []
    for field_key in ("image_a", "image_b"):
        cal_set = {p.get(field_key, "") for p in cal if p.get(field_key)}
        test_set = {p.get(field_key, "") for p in test if p.get(field_key)}
        overlap = cal_set & test_set
        if overlap:
            warnings.append(f"image path leakage in {field_key}: {len(overlap)} path(s) in both splits")
    for id_key in ("registered_dog_id", "identity_a", "identity_b", "identity"):
        cal_ids = {str(p[id_key]) for p in cal if id_key in p}
        test_ids = {str(p[id_key]) for p in test if id_key in p}
        if cal_ids and test_ids and (cal_ids & test_ids):
            warnings.append(f"identity leakage in {id_key}: identities in both splits")
    for gk in ("video_id", "video_id_a", "video_id_b", "session_id", "capture_session_id", "camera_id", "source_dataset"):
        cal_vals = {p.get(gk) for p in cal if p.get(gk) is not None}
        test_vals = {p.get(gk) for p in test if p.get(gk) is not None}
        if cal_vals and test_vals and (cal_vals & test_vals):
            warnings.append(f"group leakage in {gk}: values in both splits")
    pair_set = set()
    for p in cal:
        a, b = p.get("image_a", ""), p.get("image_b", "")
        pair_set.add((a, b))
        pair_set.add((b, a))
    for p in test:
        a, b = p.get("image_a", ""), p.get("image_b", "")
        if (a, b) in pair_set:
            warnings.append(f"reversed pair leakage: ({a}, {b}) across splits")
    return warnings


def enforce_split_disjoint(
    cal: list[dict],
    test: list[dict],
) -> str:
    warnings = validate_split_disjoint(cal, test)
    if not warnings:
        return "VERIFIED"
    return "INVALID"


def relaxed_status_from_warnings(warnings: list[str], relaxed: bool) -> str:
    if not warnings:
        return "VERIFIED"
    if relaxed:
        return "RELAXED_UNSAFE"
    return "INVALID"


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def compute_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    na = np.linalg.norm(emb_a)
    nb = np.linalg.norm(emb_b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(emb_a, emb_b) / (na * nb))


def _extract_sims(ev: AbstractEvidencer, pairs: list[dict]) -> tuple[list[float], list[int]]:
    sims: list[float] = []
    labels: list[int] = []
    for p in pairs:
        emb_a = ev.extract(Image.open(p["image_a"]).convert("RGB"))
        emb_b = ev.extract(Image.open(p["image_b"]).convert("RGB"))
        sims.append(compute_similarity(emb_a, emb_b))
        labels.append(p["label"])
    return sims, labels


# ---------------------------------------------------------------------------
# Verification protocol
# ---------------------------------------------------------------------------

def cmd_verification(args: argparse.Namespace) -> None:
    start_ts = datetime.now(timezone.utc).isoformat()
    config = json.loads(args.evidence_config.read_text())
    ev_map: dict[str, AbstractEvidencer] = {}
    for name, spec in config.get("channels", {}).items():
        kind = spec.get("type", "")
        if kind == "dinov2":
            ev_map[name] = Dinov2WithUncertainty()
        elif kind == "landmark":
            ev_map[name] = LandmarkEvidencer()
    active = list(ev_map.keys())
    if not active:
        print(json.dumps({"error": "no active evidence channels"}))
        raise SystemExit(1)
    cal_pairs = load_pairs(args.calibration_pairs)
    test_pairs = load_pairs(args.test_pairs)
    warnings = validate_split_disjoint(cal_pairs, test_pairs)
    split_status = relaxed_status_from_warnings(warnings, args.relaxed_split)
    if split_status == "INVALID" and not args.relaxed_split:
        print(json.dumps({
            "error": "fatal split leakage detected",
            "protocol_status": split_status,
            "warnings": warnings,
            "valid_for_model_selection": False,
            "valid_for_final_reporting": False,
        }))
        raise SystemExit(1)

    prov = _provenance(start_ts)
    schema_hash = _file_sha256(SCHEMA_PATH) if SCHEMA_PATH.exists() else {"status": "NOT_FOUND"}
    report: dict[str, Any] = {
        "protocol": "verification",
        "protocol_status": split_status,
        "provenance": prov,
        "evidence_config": _file_sha256(args.evidence_config),
        "calibration_pairs": _file_sha256(args.calibration_pairs),
        "test_pairs": _file_sha256(args.test_pairs),
        "schema": schema_hash,
        "warnings": warnings if warnings else None,
        "split_policy": "relaxed" if args.relaxed_split else "strict",
        "calibration_pair_count": len(cal_pairs),
        "test_pair_count": len(test_pairs),
        "channels": [],
        "calibration": {},
        "thresholds": {},
    }
    if split_status == "RELAXED_UNSAFE":
        report["valid_for_model_selection"] = False
        report["valid_for_final_reporting"] = False

    target_fars = [0.001, 0.01, 0.1]
    for name in active:
        ev = ev_map[name]
        cal_sims, cal_labels = _extract_sims(ev, cal_pairs)
        test_sims, test_labels = _extract_sims(ev, test_pairs)
        cal_scores = np.array(cal_sims, dtype=np.float32)
        cal_lbs = np.array(cal_labels, dtype=np.int64)
        test_scores = np.array(test_sims, dtype=np.float32)
        test_lbs = np.array(test_labels, dtype=np.int64)

        descriptive = compute_verification_metrics(test_scores, test_lbs)
        descriptive["channel"] = name
        report["channels"].append(descriptive)

        thresholds: dict[str, dict] = {}
        for tfar in target_fars:
            op = select_threshold_at_far(cal_scores, cal_lbs, tfar)
            te = evaluate_at_threshold(test_scores, test_lbs, op.threshold)
            n_neg_cal = int((1 - cal_lbs).sum())
            n_neg_test = int((1 - test_lbs).sum())
            thresholds[str(tfar)] = {
                "target_far": tfar,
                "selected_threshold": op.threshold,
                "calibration_far": op.calibration_far,
                "calibration_tar": op.calibration_tar,
                "calibration_negatives": op.calibration_num_negative,
                "calibration_false_accepts": op.calibration_false_accepts,
                "calib_negative_ci": _wilson_ci(op.calibration_false_accepts, op.calibration_num_negative),
                "required_zero_event_trials": required_zero_event_trials(tfar, 0.95),
                "max_zero_event_upper_bound": zero_event_exact_upper_bound(n_neg_cal, 0.95),
                "zero_event_feasible": n_neg_cal >= required_zero_event_trials(tfar, 0.95),
                **te,
                "test_tar_ci": _wilson_ci(te["true_accepts"], te["num_positive"]),
                "test_far_ci": _wilson_ci(te["false_accepts"], n_neg_test),
            }
        report["thresholds"][name] = thresholds

        try:
            iso = fit_isotonic_calibration(cal_scores.astype(np.float64), cal_lbs)
            cal_probs = iso.transform(test_scores.astype(np.float64))
            cal_metrics = compute_probability_calibration_metrics(cal_probs, test_lbs)
            cal_metrics["channel"] = name
            report["calibration"][name] = cal_metrics
        except Exception as exc:
            print(json.dumps({
                "error": f"calibration failed for channel {name}: {exc}",
                "protocol_status": "INVALID",
            }))
            raise SystemExit(1)

    prov["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    report["provenance"] = prov
    _validate_report(report)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "verification_done", "output": str(args.output)}))


# ---------------------------------------------------------------------------
# Retrieval protocol
# ---------------------------------------------------------------------------

def cmd_retrieval(args: argparse.Namespace) -> None:
    start_ts = datetime.now(timezone.utc).isoformat()
    gallery = load_embedding_manifest(args.gallery)
    queries = load_embedding_manifest(args.queries)
    g_embs = np.array(gallery["embeddings"], dtype=np.float32)
    g_ids = np.array(gallery["identities"])
    q_embs = np.array(queries["embeddings"], dtype=np.float32)
    q_ids = np.array(queries["identities"])
    g_sample_ids = np.array(gallery.get("sample_ids", [])) if "sample_ids" in gallery else None
    q_sample_ids = np.array(queries.get("sample_ids", [])) if "sample_ids" in queries else None
    if args.no_self_match:
        if q_sample_ids is None or g_sample_ids is None:
            print(json.dumps({"error": "--no-self-match requires 'sample_ids' in both gallery and query manifests"}))
            raise SystemExit(1)
    try:
        result = compute_retrieval_metrics(
            q_embs, g_embs, q_ids, g_ids,
            metric="cosine",
            rank_ks=(1, 5, 10),
            query_sample_ids=q_sample_ids,
            gallery_sample_ids=g_sample_ids,
            closed_set=not args.open_set,
        )
    except RetrievalError as e:
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)
    prov = _provenance(start_ts)
    prov["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "protocol": "retrieval",
        "protocol_status": "VERIFIED",
        "provenance": prov,
        "gallery": _file_sha256(args.gallery),
        "queries": _file_sha256(args.queries),
        "schema": _file_sha256(SCHEMA_PATH) if SCHEMA_PATH.exists() else {"status": "NOT_FOUND"},
        "self_match_excluded": args.no_self_match and q_sample_ids is not None,
        **result,
    }
    rank_keys = [k for k in result if k.startswith("Rank-")]
    if rank_keys:
        rank_vals = np.array([result[k] for k in rank_keys], dtype=np.float64)
        report["rank_bootstrap_ci"] = _bootstrap_ci(rank_vals, unit="rank-k")
    query_aps = _per_query_aps(q_embs, g_embs, q_ids, g_ids, q_sample_ids, g_sample_ids)
    if query_aps is not None and len(query_aps) > 0:
        report["mAP_bootstrap_ci"] = _bootstrap_ci(np.array(query_aps, dtype=np.float64), unit="query")
    _validate_report(report)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "retrieval_done", "output": str(args.output)}))


def _per_query_aps(
    q_embs, g_embs, q_ids, g_ids, q_sids, g_sids,
) -> list[float] | None:
    try:
        from cvi.evaluation.retrieval import _compute_ap_inp
        q = q_embs.copy()
        g = g_embs.copy()
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        g_norm = np.linalg.norm(g, axis=1, keepdims=True)
        q = q / q_norm
        g = g / g_norm
        sims = q @ g.T
        aps = []
        for i in range(len(q_embs)):
            row = sims[i].copy()
            ip = g_ids == q_ids[i]
            if q_sids is not None and g_sids is not None:
                excl = np.array(q_sids[i] == g_sids, dtype=bool)
                row[excl] = -np.inf
                ip = ip & ~excl
            nr = int(ip.sum())
            if nr == 0:
                aps.append(0.0)
                continue
            order = np.argsort(-row, kind="stable")
            rp = ip[order]
            ap, _ = _compute_ap_inp(rp, nr)
            aps.append(ap)
        return aps
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Open-set protocol
# ---------------------------------------------------------------------------

def cmd_open_set(args: argparse.Namespace) -> None:
    start_ts = datetime.now(timezone.utc).isoformat()
    gallery = load_embedding_manifest(args.gallery)
    test_queries = load_embedding_manifest(args.test_queries)
    g_embs = np.array(gallery["embeddings"], dtype=np.float32)
    g_ids = np.array(gallery["identities"])
    q_embs = np.array(test_queries["embeddings"], dtype=np.float32)
    q_ids = np.array(test_queries["identities"])
    if args.calibration_queries:
        cal_q = load_embedding_manifest(args.calibration_queries)
        cal_q_embs = np.array(cal_q["embeddings"], dtype=np.float32)
        cal_q_ids = np.array(cal_q["identities"])
        cal_g_embs = np.array(gallery["embeddings"], dtype=np.float32)
        cal_g_ids = np.array(gallery["identities"])
    else:
        cal_q_embs = None
        cal_q_ids = None
        cal_g_embs = None
        cal_g_ids = None
    try:
        result: OpenSetResult = evaluate_open_set(
            q_embs, g_embs, q_ids, g_ids,
            fpir_targets=(0.01, 0.001),
            calibration_query_embs=cal_q_embs,
            calibration_gallery_embs=cal_g_embs,
            calibration_query_ids=cal_q_ids,
            calibration_gallery_ids=cal_g_ids,
        )
    except OpenSetError as e:
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)
    prov = _provenance(start_ts)
    prov["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "protocol": "open_set",
        "protocol_status": "VERIFIED",
        "provenance": prov,
        "gallery": _file_sha256(args.gallery),
        "test_queries": _file_sha256(args.test_queries),
        "schema": _file_sha256(SCHEMA_PATH) if SCHEMA_PATH.exists() else {"status": "NOT_FOUND"},
        "known_detection_AUROC": result.known_detection_auroc,
        "known_detection_AUPR": result.known_detection_aupr,
        "DIR_at_FPIR": dict(result.dir_at_fpir),
        "FPIR_thresholds": dict(result.fpir_thresholds),
        "per_target": result.per_target,
        "known_correct_accept_count": result.known_correct_accept_count,
        "known_misidentification_count": result.known_misidentification_count,
        "known_rejection_count": result.known_rejection_count,
        "unknown_accept_count": result.unknown_accept_count,
        "unknown_rejection_count": result.unknown_rejection_count,
        "num_enrolled_queries": result.num_enrolled_queries,
        "num_unknown_queries": result.num_unknown_queries,
        "num_gallery_identities": result.num_gallery_identities,
    }
    for target_key, info in result.per_target.items():
        t = info["test"]
        dir_events = t["correct_known_accepts"]
        dir_trials = t["known_queries"]
        fpir_events = t["unknown_accepts"]
        fpir_trials = t["unknown_queries"]
        if target_key not in report:
            report.setdefault("DIR_CI", {})
            report["DIR_CI"][target_key] = _wilson_ci(dir_events, dir_trials)
        report.setdefault("FPIR_CI", {})
        report["FPIR_CI"][target_key] = _wilson_ci(fpir_events, fpir_trials)
    _validate_report(report)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "open_set_done", "output": str(args.output)}))


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="protocol", required=True)

    p_ver = sub.add_parser("verification")
    p_ver.add_argument("--evidence-config", type=Path, required=True)
    p_ver.add_argument("--calibration-pairs", type=Path, required=True)
    p_ver.add_argument("--test-pairs", type=Path, required=True)
    p_ver.add_argument("--output", type=Path, required=True)
    p_ver.add_argument("--relaxed-split", action="store_true")
    p_ver.set_defaults(func=cmd_verification)

    p_ret = sub.add_parser("retrieval")
    p_ret.add_argument("--gallery", type=Path, required=True)
    p_ret.add_argument("--queries", type=Path, required=True)
    p_ret.add_argument("--output", type=Path, required=True)
    p_ret.add_argument("--no-self-match", action="store_true")
    p_ret.add_argument("--open-set", action="store_true")
    p_ret.set_defaults(func=cmd_retrieval)

    p_os = sub.add_parser("open-set")
    p_os.add_argument("--gallery", type=Path, required=True)
    p_os.add_argument("--calibration-queries", type=Path, default=None)
    p_os.add_argument("--test-queries", type=Path, required=True)
    p_os.add_argument("--output", type=Path, required=True)
    p_os.set_defaults(func=cmd_open_set)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
