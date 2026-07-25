"""Multi-channel evaluation framework.

Protocols:
  evaluate verification   --calibration-pairs CAL  --test-pairs TEST  ...
  evaluate retrieval      --gallery FILE  --queries FILE  ...
  evaluate open-set       --gallery FILE  --queries FILE  ...

Usage:
  uv run python tools/evaluate_multichannel.py verification \\
      --evidence-config config.json \\
      --calibration-pairs cal.json \\
      --test-pairs test.json \\
      --output report.json

  uv run python tools/evaluate_multichannel.py retrieval \\
      --gallery gallery.json --queries queries.json \\
      --evidence-config config.json --output report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.isotonic import IsotonicRegression

from cvi.evidence.base import AbstractEvidencer
from cvi.evidence.nose_print import MiewIDNoseExtractor
from cvi.evidence.landmark_graph import LandmarkEvidencer
from cvi.evidence.appearance import Dinov2WithUncertainty
from cvi.evaluation.verification import (
    compute_verification_curve,
    select_threshold_at_far,
    evaluate_at_threshold,
    compute_verification_metrics,
    EvaluationError,
)
from cvi.evaluation.retrieval import (
    compute_retrieval_metrics,
    RetrievalError,
)
from cvi.evaluation.open_set import (
    evaluate_open_set,
    OpenSetError,
    OpenSetResult,
)
from cvi.evaluation.calibration import (
    fit_isotonic_calibration,
    compute_probability_calibration_metrics,
    CalibrationError,
)
from cvi.evaluation._legacy import (
    wilson_rate,
    zero_event_exact_upper_bound,
    required_zero_event_trials,
)
from cvi.pipeline.enroll import MultiEvidencePipeline

SCHEMA_VERSION = "cvi.evaluation.report.v2"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return "UNVERIFIED"


def _git_branch() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return "UNVERIFIED"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return len(result.stdout.strip()) > 0
    except Exception:
        return False


def _provenance() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "dirty_state": _git_dirty(),
        "start_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }


def _confidence_intervals(
    events: int,
    trials: int,
    confidence_level: float = 0.95,
) -> dict:
    if trials == 0:
        return {"estimate": 0.0, "ci_method": "none"}
    est = wilson_rate(events, trials, confidence_level)
    return {
        "events": events,
        "trials": trials,
        "estimate": est.estimate,
        "lower_bound": est.lower_bound,
        "upper_bound": est.upper_bound,
        "interval_method": est.interval_method,
        "confidence_level": confidence_level,
    }


def report_provenance() -> dict:
    return _provenance()


def compute_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    na = np.linalg.norm(emb_a)
    nb = np.linalg.norm(emb_b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(emb_a, emb_b) / (na * nb))


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"pairs file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list, got {type(data)}")
    if len(data) == 0:
        raise ValueError("empty pairs list")
    return data


def load_split_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    for key in ("calibration", "test"):
        if key not in data:
            raise ValueError(f"split manifest missing '{key}'")
        if not isinstance(data[key], list):
            raise ValueError(f"split manifest '{key}' must be a list")
    return data


def validate_split_disjoint(
    cal_pairs: list[dict],
    test_pairs: list[dict],
) -> list[str]:
    warnings: list[str] = []
    fields_checked: list[str] = []
    for field_key, group_label in [
        ("image_a", "image_a path"),
        ("image_b", "image_b path"),
    ]:
        cal_paths = {p.get(field_key, "") for p in cal_pairs if p.get(field_key)}
        test_paths = {p.get(field_key, "") for p in test_pairs if p.get(field_key)}
        overlap = cal_paths & test_paths
        if overlap:
            warnings.append(
                f"image path leakage in {field_key}: {len(overlap)} path(s) "
                f"appear in both calibration and test"
            )
    fields_checked.append("image_a, image_b")
    for id_key, label in [
        ("registered_dog_id", "registered_dog_id"),
        ("identity_a", "identity_a"),
        ("identity_b", "identity_b"),
        ("identity", "identity"),
    ]:
        cal_ids = set()
        test_ids = set()
        for p in cal_pairs:
            v = p.get(id_key)
            if v is not None:
                cal_ids.add(str(v))
        for p in test_pairs:
            v = p.get(id_key)
            if v is not None:
                test_ids.add(str(v))
        if cal_ids and test_ids:
            fields_checked.append(id_key)
            if cal_ids & test_ids:
                warnings.append(
                    f"identity leakage in {id_key}: identities appear "
                    f"in both calibration and test"
                )
    for group_key, label in [
        ("video_id", "video_id"),
        ("video_id_a", "video_id_a"),
        ("video_id_b", "video_id_b"),
        ("session_id", "session_id"),
        ("capture_session_id", "capture_session_id"),
        ("camera_id", "camera_id"),
        ("source_dataset", "source_dataset"),
        ("sample_id", "sample_id"),
    ]:
        cal_vals = {p.get(group_key) for p in cal_pairs if p.get(group_key) is not None}
        test_vals = {p.get(group_key) for p in test_pairs if p.get(group_key) is not None}
        if cal_vals and test_vals:
            fields_checked.append(group_key)
            if cal_vals & test_vals:
                warnings.append(
                    f"potential group leakage in {group_key}: values "
                    f"appear in both calibration and test"
                )
    pair_keys = []
    for p in cal_pairs:
        a = p.get("image_a", ""); b = p.get("image_b", "")
        pair_keys.append((a, b))
        pair_keys.append((b, a))
    cal_pair_set = set(pair_keys)
    for p in test_pairs:
        a = p.get("image_a", ""); b = p.get("image_b", "")
        if (a, b) in cal_pair_set:
            warnings.append(f"reversed pair leakage: pair ({a}, {b}) matches across splits")
    return warnings


def build_evidence_map(config: dict) -> dict[str, AbstractEvidencer]:
    emap: dict[str, AbstractEvidencer] = {}
    for name, spec in config.get("channels", {}).items():
        kind = spec.get("type", "")
        if kind == "miewid":
            path = Path(spec.get("path", ""))
            if path.exists():
                from cvi.evidence.nose_print import MiewIDNoseExtractor
                emap[name] = MiewIDNoseExtractor(path)
        elif kind == "dinov2":
            emap[name] = Dinov2WithUncertainty()
        elif kind == "landmark":
            emap[name] = LandmarkEvidencer()
    return emap


# ---------- verification protocol ----------

def cmd_verification(args: argparse.Namespace) -> None:
    config = json.loads(args.evidence_config.read_text())
    evidencer_map = build_evidence_map(config)
    active = list(evidencer_map.keys())
    if not active:
        print(json.dumps({"error": "no active evidence channels"}))
        raise SystemExit(1)
    cal_pairs = load_pairs(args.calibration_pairs)
    test_pairs = load_pairs(args.test_pairs)
    leakage_warnings = validate_split_disjoint(cal_pairs, test_pairs)
    provenance = report_provenance()
    report: dict[str, Any] = {
        "protocol": "verification",
        "provenance": provenance,
        "evidence_config_sha256": "UNVERIFIED",
        "warnings": leakage_warnings,
        "calibration_pairs": len(cal_pairs),
        "test_pairs": len(test_pairs),
        "channels": [],
        "calibration": {},
        "thresholds": {},
    }
    for name in active:
        ev = evidencer_map[name]
        cal_sims, cal_labels = _extract_sims(ev, cal_pairs)
        test_sims, test_labels = _extract_sims(ev, test_pairs)
        descriptive = compute_verification_metrics(
            np.array(test_sims, dtype=np.float32),
            np.array(test_labels, dtype=np.int64),
        )
        descriptive["channel"] = name
        report["channels"].append(descriptive)
        curve = compute_verification_curve(
            np.array(cal_sims, dtype=np.float32),
            np.array(cal_labels, dtype=np.int64),
        )
        target_fars = [0.001, 0.01, 0.1]
        thresholds = {}
        for target_far in target_fars:
            op = select_threshold_at_far(curve, target_far)
            test_eval = evaluate_at_threshold(
                np.array(test_sims, dtype=np.float32),
                np.array(test_labels, dtype=np.int64),
                op.threshold,
            )
            n_neg_cal = cal_labels.count(0)
            n_neg_test = test_labels.count(0)
            required = required_zero_event_trials(target_far, confidence_level=0.95)
            max_zero_upper = zero_event_exact_upper_bound(n_neg_cal, confidence_level=0.95)
            thresholds[str(target_far)] = {
                "target_far": target_far,
                "selected_threshold": op.threshold,
                "calibration_far": op.calibration_far,
                "calibration_tar": op.calibration_tar,
                "calibration_negatives": n_neg_cal,
                "calibration_false_accepts": int(round(op.calibration_far * n_neg_cal)),
                "calib_negative_ci": _confidence_intervals(
                    int(round(op.calibration_far * n_neg_cal)), n_neg_cal,
                ),
                "required_zero_event_trials": required,
                "max_zero_event_upper_bound": max_zero_upper,
                "zero_event_feasible": n_neg_cal >= required,
                "test_false_accepts": test_eval["false_accepts"],
                "test_negatives": n_neg_test,
                "test_TAR": test_eval["TAR"],
                "test_FAR": test_eval["FAR"],
                "test_tar_ci": _confidence_intervals(
                    test_eval["true_accepts"],
                    test_eval["num_positive"],
                ),
                "test_far_ci": _confidence_intervals(
                    test_eval["false_accepts"],
                    n_neg_test,
                ),
            }
        report["thresholds"][name] = thresholds
        cal_probs = _calibrate_channel(ev, cal_pairs, test_pairs)
        if cal_probs is not None:
            try:
                cal_metrics = compute_probability_calibration_metrics(
                    np.array(cal_probs, dtype=np.float64),
                    np.array(test_labels, dtype=np.int64),
                )
                cal_metrics["channel"] = name
                report["calibration"][name] = cal_metrics
            except CalibrationError as e:
                report["calibration"][name] = {"error": str(e)}
    provenance["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    report["provenance"] = provenance
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"event": "verification_done", "output": str(args.output)}))


def _extract_sims(
    ev: AbstractEvidencer, pairs: list[dict],
) -> tuple[list[float], list[int]]:
    sims: list[float] = []
    labels: list[int] = []
    for p in pairs:
        img_a = Image.open(p["image_a"]).convert("RGB")
        img_b = Image.open(p["image_b"]).convert("RGB")
        emb_a = ev.extract(img_a)
        emb_b = ev.extract(img_b)
        sims.append(compute_similarity(emb_a, emb_b))
        labels.append(p["label"])
    return sims, labels


def _calibrate_channel(
    ev: AbstractEvidencer,
    cal_pairs: list[dict],
    test_pairs: list[dict],
) -> list[float] | None:
    try:
        cal_scores = []
        for p in cal_pairs:
            img_a = Image.open(p["image_a"]).convert("RGB")
            img_b = Image.open(p["image_b"]).convert("RGB")
            emb_a = ev.extract(img_a)
            emb_b = ev.extract(img_b)
            cal_scores.append(compute_similarity(emb_a, emb_b))
        cal_labels = [p["label"] for p in cal_pairs]
        iso = fit_isotonic_calibration(
            np.array(cal_scores, dtype=np.float64),
            np.array(cal_labels, dtype=np.int64),
        )
        test_scores = []
        for p in test_pairs:
            img_a = Image.open(p["image_a"]).convert("RGB")
            img_b = Image.open(p["image_b"]).convert("RGB")
            emb_a = ev.extract(img_a)
            emb_b = ev.extract(img_b)
            test_scores.append(compute_similarity(emb_a, emb_b))
        return iso.transform(np.array(test_scores, dtype=np.float64)).tolist()
    except Exception:
        return None


# ---------- retrieval protocol ----------

def cmd_retrieval(args: argparse.Namespace) -> None:
    gallery = json.loads(args.gallery.read_text())
    queries = json.loads(args.queries.read_text())
    g_embs = np.array(gallery["embeddings"], dtype=np.float32)
    g_ids = np.array(gallery["identities"], dtype=np.int64)
    q_embs = np.array(queries["embeddings"], dtype=np.float32)
    q_ids = np.array(queries["identities"], dtype=np.int64)
    exclude_self = None
    if args.no_self_match:
        if q_embs.shape[0] == g_embs.shape[0]:
            exclude_self = np.eye(q_embs.shape[0], dtype=bool)
    try:
        result = compute_retrieval_metrics(
            q_embs, g_embs, q_ids, g_ids,
            metric="cosine",
            rank_ks=(1, 5, 10),
            exclude_self=exclude_self,
            closed_set=not args.open_set,
        )
    except RetrievalError as e:
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)
    provenance = report_provenance()
    provenance["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    report = {
        "protocol": "retrieval",
        "provenance": provenance,
        **result,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"event": "retrieval_done", "output": str(args.output)}))


# ---------- open-set protocol ----------

def cmd_open_set(args: argparse.Namespace) -> None:
    gallery = json.loads(args.gallery.read_text())
    queries = json.loads(args.queries.read_text())
    g_embs = np.array(gallery["embeddings"], dtype=np.float32)
    g_ids = np.array(gallery["identities"], dtype=np.int64)
    q_embs = np.array(queries["embeddings"], dtype=np.float32)
    q_ids = np.array(queries["identities"], dtype=np.int64)
    try:
        result = evaluate_open_set(
            q_embs, g_embs, q_ids, g_ids,
            fpir_targets=(0.01, 0.001),
        )
    except OpenSetError as e:
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)
    provenance = report_provenance()
    provenance["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    report = {
        "protocol": "open_set",
        "provenance": provenance,
        "known_vs_unknown_AUROC": result.known_vs_unknown_auroc,
        "known_vs_unknown_AUPR": result.known_vs_unknown_aupr,
        "DIR_at_FPIR": dict(result.dir_at_fpir),
        "false_accept_count": result.false_accept_count,
        "false_reject_count": result.false_reject_count,
        "num_enrolled_queries": result.num_enrolled_queries,
        "num_unknown_queries": result.num_unknown_queries,
        "num_gallery_identities": result.num_gallery_identities,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"event": "open_set_done", "output": str(args.output)}))


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="protocol", required=True)

    p_ver = sub.add_parser("verification")
    p_ver.add_argument("--evidence-config", type=Path, required=True)
    p_ver.add_argument("--calibration-pairs", type=Path, required=True)
    p_ver.add_argument("--test-pairs", type=Path, required=True)
    p_ver.add_argument("--output", type=Path, required=True)
    p_ver.set_defaults(func=cmd_verification)

    p_ret = sub.add_parser("retrieval")
    p_ret.add_argument("--evidence-config", type=Path, required=True)
    p_ret.add_argument("--gallery", type=Path, required=True)
    p_ret.add_argument("--queries", type=Path, required=True)
    p_ret.add_argument("--output", type=Path, required=True)
    p_ret.add_argument("--no-self-match", action="store_true")
    p_ret.add_argument("--open-set", action="store_true")
    p_ret.set_defaults(func=cmd_retrieval)

    p_os = sub.add_parser("open-set")
    p_os.add_argument("--gallery", type=Path, required=True)
    p_os.add_argument("--queries", type=Path, required=True)
    p_os.add_argument("--output", type=Path, required=True)
    p_os.set_defaults(func=cmd_open_set)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
