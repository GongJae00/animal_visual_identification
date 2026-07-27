"""Visual-shortcut baseline evaluation for the public frozen experiment.

Computes simple image-similarity baselines (color histograms, mean pixel)
on the protected split protocols. Features are pre-computed once per image
and reused across all protocols and extractors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from cvi.evaluation.verification import (
    compute_verification_metrics,
    select_threshold_at_far,
)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _hsv_hist(img: np.ndarray, bins: int = 50) -> np.ndarray:
    hsv = Image.fromarray(img.astype("uint8")).convert("HSV")
    h, s, v = np.array(hsv, dtype=np.float32).reshape(-1, 3).T
    hh, _ = np.histogram(h, bins=bins, range=(0, 256))
    sh, _ = np.histogram(s, bins=bins, range=(0, 256))
    vh, _ = np.histogram(v, bins=bins, range=(0, 256))
    return _l2(np.concatenate([hh, sh, vh]).astype(np.float64))


def _gray_hist(img: np.ndarray, bins: int = 256) -> np.ndarray:
    g = np.mean(img, axis=2)
    h, _ = np.histogram(g, bins=bins, range=(0, 256))
    return _l2(h.astype(np.float64))


def _mean_col(img: np.ndarray) -> np.ndarray:
    return _l2(img.reshape(-1, 3).mean(axis=0).astype(np.float64))


_FEATURE_FNS = {
    "hsv_hist50": lambda img: _hsv_hist(img, 50),
    "gray_hist256": lambda img: _gray_hist(img, 256),
    "mean_color": _mean_col,
}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _roc_auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.5
    scores = np.asarray(pos + neg, dtype=np.float64)
    labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    return compute_verification_metrics(scores, labels)["ROC_AUC"]


def _tar_at_far(pos, neg, target: float) -> float:
    if not pos or not neg:
        return 0.0
    scores = np.asarray(pos + neg, dtype=np.float64)
    labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    return select_threshold_at_far(scores, labels, target).calibration_tar


def _eer(pos, neg) -> float:
    if not pos or not neg:
        return 0.5
    scores = np.asarray(pos + neg, dtype=np.float64)
    labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    return compute_verification_metrics(scores, labels)["EER"]


# ---------------------------------------------------------------------------
# Feature cache — compute once per token, reuse everywhere
# ---------------------------------------------------------------------------

def _crop_path(crops_dir: Path, protocol: str, shot: str, role: str,
               token: str, gallery_size: int | None = None) -> Path:
    parts = [f"protocol={protocol}", f"shot={shot}", f"role={role}"]
    if gallery_size:
        parts.append(f"gallery_size={gallery_size}")
    return crops_dir / "/".join(parts) / f"{token}.jpg"


def _build_feature_cache(
    assignment: dict, crops_dir: Path, extract,
) -> dict[str, np.ndarray]:
    seen: set[str] = set()
    cache: dict[str, np.ndarray] = {}
    for rec in assignment.get("records", []):
        token = rec["sample_token"]
        if token in seen:
            continue
        seen.add(token)
        for use in rec.get("uses", []):
            p, shot, role, gs = use["protocol"], use["shot"], use["role"], use.get("gallery_size")
            path = _crop_path(crops_dir, p, shot, role, token, gs)
            if path.exists():
                cache[token] = extract(_load_rgb(path))
                break
    return cache


# ---------------------------------------------------------------------------
# Per-protocol evaluators
# ---------------------------------------------------------------------------

_QUERY_ROLES = frozenset({"KNOWN_QUERY", "UNKNOWN_QUERY"})


def _build_gallery_query(
    records: list[dict],
    labels_by_token: dict[str, dict],
    feat_cache: dict[str, np.ndarray],
    protocol: str,
    *query_roles: str,
):
    g_identities: list[str] = []
    g_feats: list[np.ndarray] = []
    q_identities: list[str] = []
    q_feats: list[np.ndarray] = []

    for rec in records:
        token = rec["sample_token"]
        label = labels_by_token.get(token)
        if label is None:
            continue
        identity = label.get("registered_dog_id", label["dataset_identity_id"])
        for use in rec.get("uses", []):
            if use["protocol"] != protocol:
                continue
            role = use["role"]
            feat = feat_cache.get(token)
            if feat is None:
                continue
            if role == "GALLERY":
                g_identities.append(identity)
                g_feats.append(feat)
            elif role in query_roles:
                q_identities.append(identity)
                q_feats.append(feat)

    if not g_feats or not q_feats:
        return np.empty((0, 0)), [], [], []

    G = np.stack(g_feats)
    Q = np.stack(q_feats)
    sim = Q @ G.T
    return sim, g_identities, q_identities, len(g_feats)


def evaluate_closed_set(
    records: list[dict],
    labels_by_token: dict[str, dict],
    feat_cache: dict[str, np.ndarray],
    protocol: str,
) -> dict:
    sim, g_ids, q_ids, _ = _build_gallery_query(
        records, labels_by_token, feat_cache, protocol, "KNOWN_QUERY",
    )
    if sim.size == 0:
        nq = sum(1 for rec in records for u in rec.get("uses", [])
                 if u["protocol"] == protocol and u["role"] != "GALLERY")
        return {"error": "no gallery or query features", "n_queries_expected": nq}

    g_id_set = sorted(set(g_ids))
    g_id_to_cols = {gid: [i for i, x in enumerate(g_ids) if x == gid] for gid in g_id_set}

    all_pos, all_neg = [], []
    rank1_ok = 0
    for q_idx, q_identity in enumerate(q_ids):
        best_sim = -1.0
        best_id = None
        for g_id in g_id_set:
            cols = g_id_to_cols[g_id]
            s = float(np.max(sim[q_idx, cols]))
            if s > best_sim:
                best_sim, best_id = s, g_id
            if g_id == q_identity:
                all_pos.append(s)
            else:
                all_neg.append(s)
        if best_id == q_identity:
            rank1_ok += 1

    nq = len(q_ids)
    return {
        "rank1_accuracy": round(rank1_ok / nq, 4) if nq else 0.0,
        "auc": round(_roc_auc(all_pos, all_neg), 4),
        "tar@1%far": round(_tar_at_far(all_pos, all_neg, 0.01), 4),
        "tar@0.1%far": round(_tar_at_far(all_pos, all_neg, 0.001), 4),
        "eer": round(_eer(all_pos, all_neg), 4),
        "n_pos": len(all_pos),
        "n_neg": len(all_neg),
        "n_queries": nq,
        "n_gallery_ids": len(g_id_set),
        "mean_pos_score": round(float(np.mean(all_pos)), 4) if all_pos else 0.0,
        "mean_neg_score": round(float(np.mean(all_neg)), 4) if all_neg else 0.0,
    }


def evaluate_open_set(
    records: list[dict],
    labels_by_token: dict[str, dict],
    feat_cache: dict[str, np.ndarray],
    protocol: str,
) -> dict:
    g_identities: list[str] = []
    g_feats: list[np.ndarray] = []
    kq_identities: list[str] = []
    kq_feats: list[np.ndarray] = []
    uq_feats: list[np.ndarray] = []

    for rec in records:
        token = rec["sample_token"]
        label = labels_by_token.get(token)
        if label is None:
            continue
        identity = label.get("registered_dog_id", label["dataset_identity_id"])
        for use in rec.get("uses", []):
            if use["protocol"] != protocol:
                continue
            role = use["role"]
            feat = feat_cache.get(token)
            if feat is None:
                continue
            if role == "GALLERY":
                g_identities.append(identity)
                g_feats.append(feat)
            elif role == "KNOWN_QUERY":
                kq_identities.append(identity)
                kq_feats.append(feat)
            elif role == "UNKNOWN_QUERY":
                uq_feats.append(feat)

    if not g_feats:
        return {"error": "no gallery features"}

    G = np.stack(g_feats)
    g_id_set = sorted(set(g_identities))
    g_id_to_cols = {gid: [i for i, x in enumerate(g_identities) if x == gid] for gid in g_id_set}

    pos, neg = [], []
    if kq_feats:
        KQ = np.stack(kq_feats)
        kq_sim = KQ @ G.T
        for qi, q_id in enumerate(kq_identities):
            pos_cols = g_id_to_cols.get(q_id, [])
            if not pos_cols:
                continue
            pos.append(float(np.max(kq_sim[qi, pos_cols])))
            # best negative: max over all other gallery identities
            neg_cols = [c for g, cols in g_id_to_cols.items()
                        if g != q_id for c in cols]
            neg.append(float(np.max(kq_sim[qi, neg_cols])) if neg_cols else -1.0)

    uk_scores: list[float] = []
    if uq_feats:
        UQ = np.stack(uq_feats)
        uq_sim = UQ @ G.T
        for qi in range(len(uq_feats)):
            uk_scores.append(float(np.max(uq_sim[qi, :])))

    return {
        "auc": round(_roc_auc(pos, neg), 4),
        "mean_pos_score": round(float(np.mean(pos)), 4) if pos else 0.0,
        "mean_neg_score": round(float(np.mean(neg)), 4) if neg else 0.0,
        "mean_unknown_score": round(float(np.mean(uk_scores)), 4) if uk_scores else 0.0,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n_unknown_queries": len(uk_scores),
        "n_known_queries": len(kq_identities),
        "n_gallery_ids": len(g_id_set),
    }


def evaluate_paired_delta(
    records: list[dict],
    labels_by_token: dict[str, dict],
    feat_cache: dict[str, np.ndarray],
    protocol: str,
    token_to_paired_token: dict[str, str] | None = None,
) -> dict:
    scores: list[float] = []
    if not token_to_paired_token:
        return {"mean_score": 0.0, "std_score": 0.0, "n_pairs": 0}

    for rec in records:
        token = rec["sample_token"]
        paired_token = token_to_paired_token.get(token)
        if paired_token is None:
            continue
        feat_q = feat_cache.get(token)
        feat_p = feat_cache.get(paired_token)
        if feat_q is None or feat_p is None:
            continue
        scores.append(cosine_sim(feat_q, feat_p))

    return {
        "mean_score": round(float(np.mean(scores)), 4) if scores else 0.0,
        "std_score": round(float(np.std(scores)), 4) if scores else 0.0,
        "n_pairs": len(scores),
    }


def evaluate_cross_sequence(
    records: list[dict],
    labels_by_token: dict[str, dict],
    feat_cache: dict[str, np.ndarray],
    protocol: str,
) -> dict:
    sim, g_ids, q_ids, _ = _build_gallery_query(
        records, labels_by_token, feat_cache, protocol, "KNOWN_QUERY",
    )
    if sim.size == 0:
        return {"error": "no gallery or query features"}

    g_id_set = sorted(set(g_ids))
    g_id_to_cols = {gid: [i for i, x in enumerate(g_ids) if x == gid] for gid in g_id_set}

    all_pos, all_neg = [], []
    for qi, q_id in enumerate(q_ids):
        for g_id in g_id_set:
            cols = g_id_to_cols[g_id]
            s = float(np.max(sim[qi, cols]))
            if g_id == q_id:
                all_pos.append(s)
            else:
                all_neg.append(s)

    nq = len(q_ids)
    return {
        "auc": round(_roc_auc(all_pos, all_neg), 4),
        "tar@1%far": round(_tar_at_far(all_pos, all_neg, 0.01), 4),
        "eer": round(_eer(all_pos, all_neg), 4),
        "n_pos": len(all_pos),
        "n_neg": len(all_neg),
        "n_queries": nq,
        "n_gallery_ids": len(g_id_set),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_EVALUATORS = {
    "YT_CLOSED_SET": evaluate_closed_set,
    "YT_CLOSED_SET_DIAGNOSTIC": evaluate_closed_set,
    "DOGFACE_CLOSED_SET": evaluate_closed_set,
    "MPDD_CLOSED_SET": evaluate_closed_set,
    "YT_OPEN_SET": evaluate_open_set,
    "YT_DEVELOPMENT_OPEN_SET": evaluate_open_set,
    "YT_CALIBRATION_OPEN_SET": evaluate_open_set,
    "MPDD_OPEN_SET": evaluate_open_set,
    "SIBETAN_OPEN_SET": evaluate_open_set,
    "YT_RANDOM_BACKGROUND_PAIRED_DELTA": evaluate_paired_delta,
    "SIBETAN_CROSS_SEQUENCE": evaluate_cross_sequence,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--source-bundle", required=False, type=Path, default=None,
                        help="source_bundle.json (needed for paired_source_sample_id)")
    parser.add_argument("--crops-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature", choices=list(_FEATURE_FNS) + ["all"], default="all")
    parser.add_argument(
        "--use-registered-ids", action="store_true",
        help="Replace dataset_identity_id with deterministic registered_dog_id (UUIDv5)",
    )
    args = parser.parse_args()

    raise RuntimeError(
        "visual shortcut baseline publication is disabled until records are "
        "isolated by protocol, episode, gallery_size, shot, modality, and role, "
        "and every use is bound to its exact crop artifact"
    )

    t0 = time.time()
    assignment = json.loads(args.assignment.read_text())
    labels = json.loads(args.labels.read_text())
    labels_by_token = {r["sample_token"]: r for r in labels["records"]}

    if args.use_registered_ids:
        from cvi.identity_registry import compute_registered_dog_id as _crid
        n_resolved = 0
        for rec in labels.get("records", []):
            did = rec.get("dataset_identity_id", "")
            if did:
                rec["registered_dog_id"] = _crid(did)
                n_resolved += 1
        print(f"Registered IDs: {n_resolved} identities resolved", flush=True)
    print(f"Loaded assignment+labels in {time.time()-t0:.1f}s", flush=True)

    src_bundle = None
    token_to_paired_token: dict[str, str] = {}
    if args.source_bundle:
        src_bundle = json.loads(args.source_bundle.read_text())
        sid_to_token: dict[str, str] = {}
        for s in src_bundle.get("samples", []):
            sid_to_token[s["source_sample_id"]] = s["sample_token"]
        for s in src_bundle.get("samples", []):
            psid = s.get("paired_source_sample_id")
            if psid and psid in sid_to_token:
                token_to_paired_token[s["sample_token"]] = sid_to_token[psid]
        print(
            f"Loaded source_bundle: {len(token_to_paired_token)} token→paired_token mappings",
            flush=True,
        )

    records_by_protocol: dict[str, list[dict]] = defaultdict(list)
    seen_by_proto: dict[str, set[str]] = defaultdict(set)
    for rec in assignment.get("records", []):
        for use in rec.get("uses", []):
            proto = use["protocol"]
            tok = rec["sample_token"]
            if tok not in seen_by_proto[proto]:
                seen_by_proto[proto].add(tok)
                records_by_protocol[proto].append(rec)

    feature_names = list(_FEATURE_FNS) if args.feature == "all" else [args.feature]

    results: dict[str, dict] = {}
    for feat_name in feature_names:
        extract_fn = _FEATURE_FNS[feat_name]
        t1 = time.time()
        feat_cache = _build_feature_cache(assignment, args.crops_dir, extract_fn)
        print(
            f"Cached {len(feat_cache)} features for {feat_name} in {time.time()-t1:.1f}s",
            flush=True,
        )
        for proto, evaluator in _EVALUATORS.items():
            if proto not in records_by_protocol:
                print(f"  {feat_name}/{proto}: SKIP", flush=True)
                continue
            key = f"{feat_name}/{proto}"
            t2 = time.time()
            print(f"  {key}: starting ...", flush=True)
            try:
                kwargs: dict = {}
                if proto == "YT_RANDOM_BACKGROUND_PAIRED_DELTA":
                    kwargs["token_to_paired_token"] = token_to_paired_token
                result = evaluator(
                    records_by_protocol[proto], labels_by_token,
                    feat_cache, proto, **kwargs,
                )
                results[key] = result
            except Exception as exc:
                import traceback
                results[key] = {"error": str(exc)}
                print(f"    ERROR: {exc}", flush=True)
                traceback.print_exc()
            print(f"  {key}: {time.time()-t2:.1f}s", flush=True)

    summary = {
        "schema_version": "cvi.visual_shortcut_baseline_results.v1",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps({"status": "DONE", "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
