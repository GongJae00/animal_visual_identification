"""Evaluation CLI.

Run: ``uv run python -m evaluation.commands.evaluate --help``

Protocols: verification, retrieval, open-set, protected.
Also: parsed-body, pairs, controls, drift, identity-kfold,
localization-kfold, localization-benchmark, oracle-crops,
protected-split, unified-split, oxford-pet, protected-prepare,
protected-verify, role-exposure, research-cycle, research-plan,
batch-precommit, batch-verify, registry-build, registry-bind,
split-check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from PIL import Image

from shared.contracts.source_provenance import build_offline_tool_provenance
from evaluation.calibration import (
    compute_probability_calibration_metrics,
    fit_isotonic_calibration,
)
from evaluation.open_set import OpenSetError, OpenSetResult, evaluate_open_set
from evaluation.protected_evaluation import (
    REPORT_INTERPRETATION,
    REPORT_PROTOCOL_STATUS,
    REPORT_SCHEMA_VERSION,
    load_protected_evaluation,
    publish_protected_evaluation_output,
)
from evaluation.protected_verification import (
    required_zero_event_trials,
    wilson_rate,
    zero_event_exact_upper_bound,
)
from evaluation.search_metrics.metrics import (
    RetrievalError,
    compute_cosine_score_matrix,
    compute_retrieval_metrics,
    evaluate_multi_template_closed_set,
    identity_clustered_bootstrap_ci,
)
from evaluation.verification import (
    compute_verification_metrics,
    evaluate_at_threshold,
    select_threshold_at_far,
)
from representation.evidence.base import AbstractEvidencer
from shared.foundation.protected_io import write_private_json_bundle
from shared.foundation.provenance import content_sha256
from identification.export.appearance import ReceiptBoundDinov2Small

SCHEMA_VERSION = "evaluation.report.v2"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared" / "contracts"
    / "schemas"
    / "evaluation.report.v2.schema.json"
)

def _git_text(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"git {' '.join(args)} failed") from exc
    return completed.stdout.strip()

def _file_sha256(path: Path) -> dict[str, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "status": "VERIFIED"}

def _provenance(start: str | None = None) -> dict[str, Any]:
    import jsonschema
    import sklearn

    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_text("rev-parse", "HEAD"),
        "git_branch": _git_text("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_state": bool(_git_text("status", "--porcelain")),
        "start_timestamp": start or datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "jsonschema_version": jsonschema.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python_argv": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
    }

class ReportSchemaValidationError(ValueError):
    pass

def _validate_report(report: dict) -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"schema not found: {SCHEMA_PATH}")
    schema = json.loads(SCHEMA_PATH.read_text())
    report["schema_sha256"] = hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda e: list(e.path))
    if errors:
        lines = ["; ".join(e.message for e in errors)]
        raise ReportSchemaValidationError("schema validation failed: " + "; ".join(lines))

def _write_report(path: Path, report: dict[str, Any]) -> None:
    _validate_report(report)
    write_private_json_bundle(((path, report),))

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

def _template_ids(manifest: dict, name: str) -> tuple[np.ndarray | None, str | None]:
    template_ids = manifest.get("template_ids")
    sample_ids = manifest.get("sample_ids")
    if template_ids is not None and sample_ids is not None:
        if template_ids != sample_ids:
            raise ValueError(
                f"{name} template_ids and deprecated sample_ids differ"
            )
        return np.asarray(template_ids), "template_ids+sample_ids"
    if template_ids is not None:
        return np.asarray(template_ids), "template_ids"
    if sample_ids is not None:
        return np.asarray(sample_ids), "sample_ids"
    return None, None

def validate_split_disjoint(
    cal: list[dict],
    test: list[dict],
) -> list[str]:
    warnings: list[str] = []
    def values(records: list[dict], keys: tuple[str, ...]) -> set[str]:
        return {
            str(record[key])
            for record in records
            for key in keys
            if key in record and record[key] not in (None, "")
        }

    image_keys = ("image_a", "image_b", "sample_id_a", "sample_id_b")
    image_overlap = values(cal, image_keys) & values(test, image_keys)
    if image_overlap:
        warnings.append(
            f"image path leakage across pair sides: {len(image_overlap)} item(s)"
        )
    identity_keys = (
        "registered_dog_id", "identity_a", "identity_b", "identity"
    )
    identity_overlap = values(cal, identity_keys) & values(test, identity_keys)
    if identity_overlap:
        warnings.append(
            f"identity leakage across aliases: {len(identity_overlap)} identity(s)"
        )
    group_namespaces = {
        "video": ("video_id", "video_id_a", "video_id_b"),
        "session": (
            "session_id", "session_id_a", "session_id_b", "capture_session_id"
        ),
        "camera": ("camera_id", "camera_id_a", "camera_id_b"),
        "source": ("source_dataset", "source_dataset_a", "source_dataset_b"),
    }
    for namespace, keys in group_namespaces.items():
        overlap = values(cal, keys) & values(test, keys)
        if overlap:
            warnings.append(
                f"group leakage in {namespace} namespace: {len(overlap)} value(s)"
            )
    pair_set: set[tuple[str, str]] = set()
    for p in cal:
        a, b = p.get("image_a", ""), p.get("image_b", "")
        if a and b:
            pair_set.add(tuple(sorted((str(a), str(b)))))
    for p in test:
        a, b = p.get("image_a", ""), p.get("image_b", "")
        if a and b and tuple(sorted((str(a), str(b)))) in pair_set:
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

def cmd_verification(args: argparse.Namespace) -> None:
    start_ts = datetime.now(timezone.utc).isoformat()
    config = json.loads(args.evidence_config.read_text())
    ev_map: dict[str, AbstractEvidencer] = {}
    for name, spec in config.get("channels", {}).items():
        kind = spec.get("type", "")
        if kind == "dinov2_local":
            required = {
                "type",
                "model_dir",
                "weight_intake_bundle",
                "preprocessor_intake_bundle",
                "device",
            }
            if set(spec) != required:
                raise ValueError(
                    "dinov2_local verification requires exact receipt-bound fields"
                )
            ev_map[name] = ReceiptBoundDinov2Small(
                model_directory=Path(spec["model_dir"]),
                weight_intake_bundle=Path(spec["weight_intake_bundle"]),
                preprocessor_intake_bundle=Path(
                    spec["preprocessor_intake_bundle"]
                ),
                device=spec["device"],
            )
        elif kind in {"dinov2", "appearance"}:
            raise ValueError(
                "unpinned DINOv2 verification is disabled; use dinov2_local"
            )
        elif kind == "landmark":
            raise ValueError(
                "landmark evaluation is disabled until trained heatmap and "
                "graph artifacts have a verified loading contract"
            )
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
    schema_hash = _file_sha256(SCHEMA_PATH)
    report: dict[str, Any] = {
        "protocol": "verification",
        "protocol_status": "UNVERIFIED" if split_status == "VERIFIED" else split_status,
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
    _write_report(args.output, report)
    print(json.dumps({"event": "verification_done", "output": str(args.output)}))

def cmd_retrieval(args: argparse.Namespace) -> None:
    start_ts = datetime.now(timezone.utc).isoformat()
    gallery = load_embedding_manifest(args.gallery)
    queries = load_embedding_manifest(args.queries)
    g_embs = np.array(gallery["embeddings"], dtype=np.float32)
    g_ids = np.array(gallery["identities"])
    q_embs = np.array(queries["embeddings"], dtype=np.float32)
    q_ids = np.array(queries["identities"])
    try:
        g_template_ids, g_template_id_field = _template_ids(gallery, "gallery")
        q_template_ids, q_template_id_field = _template_ids(queries, "queries")
    except ValueError as error:
        print(json.dumps({"error": str(error)}))
        raise SystemExit(1)
    if args.self_match_policy == "exclude" and (
        q_template_ids is None or g_template_ids is None
    ):
        print(json.dumps({
            "error": "self-match exclusion requires template_ids in both manifests "
            "(deprecated sample_ids is accepted as an alias)"
        }))
        raise SystemExit(1)
    try:
        if args.open_set:
            result = compute_retrieval_metrics(
                q_embs,
                g_embs,
                q_ids,
                g_ids,
                metric="cosine",
                rank_ks=(1, 5, 10),
                query_sample_ids=(
                    q_template_ids if args.self_match_policy == "exclude" else None
                ),
                gallery_sample_ids=(
                    g_template_ids if args.self_match_policy == "exclude" else None
                ),
                closed_set=False,
            )
            evaluation_variant = "legacy_open_set_retrieval_diagnostic"
        else:
            scores = compute_cosine_score_matrix(q_embs, g_embs)
            result = evaluate_multi_template_closed_set(
                scores,
                q_ids,
                g_ids,
                self_match_policy=args.self_match_policy,
                query_template_ids=q_template_ids,
                gallery_template_ids=g_template_ids,
                rank_ks=(1, 5, 10),
            )
            evaluation_variant = "multi_template_closed_set"
    except RetrievalError as e:
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)
    prov = _provenance(start_ts)
    prov["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "protocol": "retrieval",
        "protocol_status": "UNVERIFIED",
        "provenance": prov,
        "gallery": _file_sha256(args.gallery),
        "queries": _file_sha256(args.queries),
        "schema": _file_sha256(SCHEMA_PATH),
        "evaluation_variant": evaluation_variant,
        "self_match_policy": args.self_match_policy,
        "self_match_excluded": args.self_match_policy == "exclude",
        "template_id_fields": {
            "gallery": g_template_id_field,
            "queries": q_template_id_field,
        },
        "valid_for_model_selection": False,
        "valid_for_final_reporting": False,
        **result,
    }
    if evaluation_variant == "multi_template_closed_set":
        query_rows = result["query_rows"]
        cluster_count = len({row["bootstrap_cluster_id"] for row in query_rows})
        if cluster_count < 2:
            report["identity_clustered_bootstrap"] = {
                "state": "UNAVAILABLE",
                "reason": "at least two query identities are required",
                "cluster_count": cluster_count,
            }
        else:
            metrics = ("AP", "INP", "reciprocal_rank", "Rank-1", "Rank-5", "Rank-10")
            report["identity_clustered_bootstrap"] = {
                "state": "AVAILABLE",
                "metrics": {
                    metric: identity_clustered_bootstrap_ci(
                        query_rows, metric=metric, resamples=1000, seed=7
                    )
                    for metric in metrics
                },
            }
    _write_report(args.output, report)
    print(json.dumps({"event": "retrieval_done", "output": str(args.output)}))

def cmd_open_set(args: argparse.Namespace) -> None:
    start_ts = datetime.now(timezone.utc).isoformat()
    gallery = load_embedding_manifest(args.gallery)
    calibration_gallery = load_embedding_manifest(args.calibration_gallery)
    calibration_queries = load_embedding_manifest(args.calibration_queries)
    test_queries = load_embedding_manifest(args.test_queries)
    g_embs = np.array(gallery["embeddings"], dtype=np.float32)
    g_ids = np.array(gallery["identities"])
    q_embs = np.array(test_queries["embeddings"], dtype=np.float32)
    q_ids = np.array(test_queries["identities"])
    cal_q_embs = np.array(calibration_queries["embeddings"], dtype=np.float32)
    cal_q_ids = np.array(calibration_queries["identities"])
    cal_g_embs = np.array(calibration_gallery["embeddings"], dtype=np.float32)
    cal_g_ids = np.array(calibration_gallery["identities"])
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
        "protocol_status": "UNVERIFIED",
        "provenance": prov,
        "gallery": _file_sha256(args.gallery),
        "calibration_gallery": _file_sha256(args.calibration_gallery),
        "calibration_queries": _file_sha256(args.calibration_queries),
        "test_queries": _file_sha256(args.test_queries),
        "schema": _file_sha256(SCHEMA_PATH),
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
    _write_report(args.output, report)
    print(json.dumps({"event": "open_set_done", "output": str(args.output)}))

def cmd_protected(args: argparse.Namespace) -> None:
    provenance = build_offline_tool_provenance(
        Path(__file__),
        additional_paths=(
            Path(__file__).resolve().parents[2] / "evaluation" / "protected_evaluation.py",
            Path(__file__).resolve().parents[2] / "shared" / "foundation" / "protected_io.py",
            Path(__file__).resolve().parents[2]
            / "evaluation"
            / "search_metrics"
            / "metrics.py",
        ),
    )
    prepared = load_protected_evaluation(
        preparation_directory=args.preparation_directory,
        expected_plan_receipt_sha256=args.expected_plan_receipt_sha256,
        expected_advanced_exposure_declaration_sha256=(
            args.expected_advanced_exposure_declaration_sha256
        ),
        policy_path=args.policy,
        split_assignment_path=args.split_assignment,
        split_receipt_path=args.split_receipt,
        exposure_ledger_path=args.exposure_ledger,
        exposure_receipt_path=args.exposure_receipt,
        gallery_path=args.gallery,
        queries_path=args.queries,
    )
    # All byte, count, dimension, and score-matrix caps have passed before
    # these dense arrays are allocated.
    gallery_embeddings = np.asarray(
        [record.embedding for record in prepared.gallery.records], dtype=np.float64
    )
    query_embeddings = np.asarray(
        [record.embedding for record in prepared.queries.records], dtype=np.float64
    )
    gallery_identities = np.asarray(
        [record.identity_token for record in prepared.gallery.records]
    )
    query_identities = np.asarray(
        [record.identity_token for record in prepared.queries.records]
    )
    gallery_templates = np.asarray(
        [record.template_token for record in prepared.gallery.records]
    )
    query_templates = np.asarray(
        [record.template_token for record in prepared.queries.records]
    )
    scores = compute_cosine_score_matrix(query_embeddings, gallery_embeddings)
    result = evaluate_multi_template_closed_set(
        scores,
        query_identities,
        gallery_identities,
        self_match_policy=prepared.policy.self_match_policy,
        query_template_ids=query_templates,
        gallery_template_ids=gallery_templates,
        rank_ks=prepared.policy.rank_ks,
    )
    bootstrap_metrics = ("AP", "INP", "reciprocal_rank", *(
        f"Rank-{value}" for value in prepared.policy.rank_ks
    ))
    bootstrap = [
        identity_clustered_bootstrap_ci(
            result["query_rows"],
            metric=metric,
            resamples=prepared.policy.bootstrap_resamples,
            seed=prepared.policy.bootstrap_seed,
        )
        for metric in bootstrap_metrics
    ]
    plan = prepared.plan_receipt
    policy = prepared.policy
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol": "protected_retrieval",
        "protocol_status": REPORT_PROTOCOL_STATUS,
        "receipt_chain_verified": True,
        "valid_for_model_selection": False,
        "valid_for_final_reporting": False,
        "evaluation_token": plan.evaluation_token,
        "receipt_chain": {
            "plan_receipt_sha256": plan.receipt_sha256,
            "policy_receipt_sha256": prepared.policy_receipt.receipt_sha256,
            "input_receipt_sha256": prepared.input_receipt.receipt_sha256,
            "advanced_exposure_declaration_sha256": plan.advanced_exposure_declaration_sha256,
            "split_assignment_sha256": plan.split_assignment_sha256,
            "prior_exposure_ledger_sha256": plan.prior_exposure_ledger_sha256,
            "prior_exposure_receipt_sha256": plan.prior_exposure_receipt_sha256,
        },
        "protocol_configuration": {
            "metric": policy.metric,
            "score_dtype": policy.score_dtype,
            "self_match_policy": policy.self_match_policy,
            "aggregation": result["aggregation"],
            "tie_policy": result["tie_policy"],
            "rank_ks": list(policy.rank_ks),
            "bootstrap_resamples": policy.bootstrap_resamples,
            "bootstrap_seed": policy.bootstrap_seed,
        },
        "input_summary": {
            "gallery_templates": len(prepared.gallery.records),
            "query_templates": len(prepared.queries.records),
            "gallery_identities": len(set(gallery_identities.tolist())),
            "query_identities": len(set(query_identities.tolist())),
            "embedding_dimension": prepared.input_receipt.embedding_dimension,
            "total_embedding_values": prepared.input_receipt.total_embedding_values,
            "score_matrix_elements": prepared.input_receipt.score_matrix_elements,
        },
        "metrics": {
            "mAP": result["mAP"],
            "mINP": result["mINP"],
            "MRR": result["MRR"],
            "rank_at_k": [
                {"k": value, "value": result[f"Rank-{value}"]}
                for value in policy.rank_ks
            ],
            "identity_clustered_bootstrap": bootstrap,
        },
        "resource_bounds": {
            "maximum_samples_per_input": policy.maximum_samples_per_input,
            "maximum_embedding_dimension": policy.maximum_embedding_dimension,
            "maximum_total_embedding_values": policy.maximum_total_embedding_values,
            "maximum_score_matrix_elements": policy.maximum_score_matrix_elements,
        },
        "evaluator_provenance_sha256": content_sha256(provenance),
        "interpretation": REPORT_INTERPRETATION,
    }
    receipt = publish_protected_evaluation_output(
        output_directory=args.output_directory,
        preparation=prepared,
        report=report,
        evaluator_provenance=provenance,
    )
    print(json.dumps({
        "event": "protected_retrieval_done",
        "output_directory": str(args.output_directory),
        "plan_receipt_sha256": plan.receipt_sha256,
        "output_receipt_sha256": receipt.receipt_sha256,
    }, sort_keys=True))

_ABSORBED = {
    "parsed-body": "evaluation.parsed_body",
    "pairs": "evaluation.controls.construct_pairs",
    "controls": "evaluation.controls.visual_controls",
    "drift": "evaluation.commands.compare_score_drift",
    "identity-kfold": "evaluation.splits.research.build_kfold",
    "localization-kfold": "evaluation.localization_kfold_cli",
    "localization-benchmark": "evaluation.localization_benchmark",
    "oracle-crops": "evaluation.controls.oracle_crop_export",
    "protected-split": "evaluation.splits.build_protected_public_split",
    "unified-split": "evaluation.splits.build_unified_full_split",
    "oxford-pet": "evaluation.oxford_pet_foreground",
    "protected-prepare": "evaluation.protected_prepare",
    "protected-verify": "evaluation.protected_verify",
    "role-exposure": "evaluation.splits.assemble_role_exposure_ledger",
    "research-cycle": "evaluation.splits.research.build_research_cycle_manifest",
    "research-plan": "evaluation.splits.research.build_research_task_plan",
    "batch-precommit": "evaluation.commands.create_batch_invariance_precommitment",
    "batch-verify": "operations.workers.verify_batch_invariance_receipt",
    "registry-build": "evaluation.splits.registry_cli",
    "registry-bind": "evaluation.splits.registry_cli",
    "split-check": "evaluation.splits.registry_cli",
}

_ABSORBED_SUBCOMMAND = {
    "registry-bind": "bind",
    "split-check": "check",
}


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in _ABSORBED:
        import importlib

        module = importlib.import_module(_ABSORBED[argv[0]])
        injected = _ABSORBED_SUBCOMMAND.get(argv[0])
        rest = argv[1:] if injected is None else [injected, *argv[1:]]
        sys.argv = [sys.argv[0], *rest]
        result = module.main()
        if isinstance(result, int):
            raise SystemExit(result)
        return

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
    self_match = p_ret.add_mutually_exclusive_group(required=True)
    self_match.add_argument(
        "--self-match-policy", choices=("include", "exclude")
    )
    self_match.add_argument(
        "--no-self-match",
        action="store_const",
        const="exclude",
        dest="self_match_policy",
        help="deprecated alias for --self-match-policy exclude",
    )
    p_ret.add_argument("--open-set", action="store_true")
    p_ret.set_defaults(func=cmd_retrieval)

    p_os = sub.add_parser("open-set")
    p_os.add_argument("--gallery", type=Path, required=True)
    p_os.add_argument("--calibration-gallery", type=Path, required=True)
    p_os.add_argument("--calibration-queries", type=Path, required=True)
    p_os.add_argument("--test-queries", type=Path, required=True)
    p_os.add_argument("--output", type=Path, required=True)
    p_os.set_defaults(func=cmd_open_set)

    p_protected = sub.add_parser(
        "protected",
        help="run receipt-bound protected retrieval from pinned embeddings",
    )
    p_protected.add_argument("--preparation-directory", required=True, type=Path)
    p_protected.add_argument("--expected-plan-receipt-sha256", required=True)
    p_protected.add_argument(
        "--expected-advanced-exposure-declaration-sha256", required=True
    )
    p_protected.add_argument("--policy", required=True, type=Path)
    p_protected.add_argument("--split-assignment", required=True, type=Path)
    p_protected.add_argument("--split-receipt", required=True, type=Path)
    p_protected.add_argument("--exposure-ledger", required=True, type=Path)
    p_protected.add_argument("--exposure-receipt", required=True, type=Path)
    p_protected.add_argument("--gallery", required=True, type=Path)
    p_protected.add_argument("--queries", required=True, type=Path)
    p_protected.add_argument("--output-directory", required=True, type=Path)
    p_protected.set_defaults(func=cmd_protected)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
