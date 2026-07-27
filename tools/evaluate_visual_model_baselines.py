"""Model-based visual evaluation with ONNX Runtime.

Loads a pre-trained ONNX model, extracts embeddings from oracle crop
images via ONNX Runtime, and runs the same vectorized evaluation as
the visual-shortcut baselines.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from cvi.evaluation.verification import (
    compute_verification_metrics,
    select_threshold_at_far,
)
from cvi.identity_registry import compute_registered_dog_id


# ---------------------------------------------------------------------------
# Image preprocessing  (standard ImageNet: 224×224, RGB, norm to approx [-1,1])
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_and_preprocess(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))  # HWC → CHW


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def _build_embedding_cache(
    assignment: dict, crops_dir: Path, sess: ort.InferenceSession,
) -> dict[str, np.ndarray]:
    seen: set[str] = set()
    cache: dict[str, np.ndarray] = {}
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    for i, rec in enumerate(assignment.get("records", [])):
        token = rec["sample_token"]
        if token in seen:
            continue
        seen.add(token)
        for use in rec.get("uses", []):
            p, shot, role, gs = (
                use["protocol"], use["shot"], use["role"], use.get("gallery_size"),
            )
            path = _crop_path(crops_dir, p, shot, role, token, gs)
            if path.exists():
                tensor = _load_and_preprocess(path)[np.newaxis, :]
                emb = sess.run([out_name], {inp_name: tensor})[0]
                n = np.linalg.norm(emb)
                cache[token] = emb.squeeze(0) / n if n > 0 else emb.squeeze(0)
                break
        if (i + 1) % 2000 == 0:
            print(f"  cache progress: {len(cache)} / {i+1} recs", flush=True)
    return cache


def _crop_path(crops_dir: Path, protocol: str, shot: str, role: str,
               token: str, gallery_size: int | None = None) -> Path:
    parts = [f"protocol={protocol}", f"shot={shot}", f"role={role}"]
    if gallery_size:
        parts.append(f"gallery_size={gallery_size}")
    return crops_dir / "/".join(parts) / f"{token}.jpg"


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
# Evaluation
# ---------------------------------------------------------------------------

def _build_gallery_query(records, labels_by_token, feat_cache, protocol, *query_roles):
    g_ids, g_feats = [], []
    q_ids, q_feats = [], []
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
                g_ids.append(identity)
                g_feats.append(feat)
            elif role in query_roles:
                q_ids.append(identity)
                q_feats.append(feat)
    if not g_feats or not q_feats:
        return np.empty((0, 0)), [], [], []
    G = np.stack(g_feats)
    Q = np.stack(q_feats)
    return Q @ G.T, g_ids, q_ids, len(g_feats)


def _eval_closed_set(records, labels_by_token, feat_cache, protocol):
    sim, g_ids, q_ids, _ = _build_gallery_query(
        records, labels_by_token, feat_cache, protocol, "KNOWN_QUERY",
    )
    if sim.size == 0:
        return {"error": "no features"}
    g_id_set = sorted(set(g_ids))
    g_id_to_cols = {gid: [i for i, x in enumerate(g_ids) if x == gid] for gid in g_id_set}
    all_pos, all_neg, rank1_ok = [], [], 0
    for qi, q_id in enumerate(q_ids):
        best_sim, best_id = -1.0, None
        for g_id in g_id_set:
            cols = g_id_to_cols[g_id]
            s = float(np.max(sim[qi, cols]))
            if s > best_sim:
                best_sim, best_id = s, g_id
            if g_id == q_id:
                all_pos.append(s)
            else:
                all_neg.append(s)
        if best_id == q_id:
            rank1_ok += 1
    nq = len(q_ids)
    return {
        "auc": round(_roc_auc(all_pos, all_neg), 4),
        "rank1_accuracy": round(rank1_ok / nq, 4) if nq else 0.0,
        "tar@1%far": round(_tar_at_far(all_pos, all_neg, 0.01), 4),
        "tar@0.1%far": round(_tar_at_far(all_pos, all_neg, 0.001), 4),
        "eer": round(_eer(all_pos, all_neg), 4),
        "n_queries": nq,
        "n_gallery_ids": len(g_id_set),
    }


def _eval_open_set(records, labels_by_token, feat_cache, protocol):
    g_ids, g_feats = [], []
    kq_ids, kq_feats = [], []
    uq_feats = []
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
                g_ids.append(identity)
                g_feats.append(feat)
            elif role == "KNOWN_QUERY":
                kq_ids.append(identity)
                kq_feats.append(feat)
            elif role == "UNKNOWN_QUERY":
                uq_feats.append(feat)
    if not g_feats:
        return {"error": "no gallery features"}
    G = np.stack(g_feats)
    g_id_set = sorted(set(g_ids))
    g_id_to_cols = {gid: [i for i, x in enumerate(g_ids) if x == gid] for gid in g_id_set}
    pos, neg = [], []
    if kq_feats:
        KQ = np.stack(kq_feats)
        kq_sim = KQ @ G.T
        for qi, q_id in enumerate(kq_ids):
            pos_cols = g_id_to_cols.get(q_id, [])
            if not pos_cols:
                continue
            pos.append(float(np.max(kq_sim[qi, pos_cols])))
            neg_cols = [c for g, cols in g_id_to_cols.items()
                        if g != q_id for c in cols]
            neg.append(float(np.max(kq_sim[qi, neg_cols])) if neg_cols else -1.0)
    uk_scores = []
    if uq_feats:
        UQ = np.stack(uq_feats)
        uq_sim = UQ @ G.T
        for qi in range(len(uq_feats)):
            uk_scores.append(float(np.max(uq_sim[qi, :])))
    return {
        "auc": round(_roc_auc(pos, neg), 4),
        "mean_known_score": round(float(np.mean(pos)), 4) if pos else 0.0,
        "mean_unknown_score": round(float(np.mean(uk_scores)), 4) if uk_scores else 0.0,
        "n_known_queries": len(kq_ids),
        "n_unknown_queries": len(uq_feats),
        "n_gallery_ids": len(g_id_set),
    }


def _eval_paired_delta(records, labels_by_token, feat_cache, protocol, token_to_paired_token):
    scores = []
    if not token_to_paired_token:
        return {"mean_score": 0.0, "std_score": 0.0, "n_pairs": 0}
    for rec in records:
        token = rec["sample_token"]
        paired_token = token_to_paired_token.get(token)
        if paired_token is None:
            continue
        fq = feat_cache.get(token)
        fp = feat_cache.get(paired_token)
        if fq is None or fp is None:
            continue
        scores.append(float(np.dot(fq, fp)))
    return {
        "mean_score": round(float(np.mean(scores)), 4) if scores else 0.0,
        "std_score": round(float(np.std(scores)), 4) if scores else 0.0,
        "n_pairs": len(scores),
    }


def _eval_cross_sequence(records, labels_by_token, feat_cache, protocol):
    sim, g_ids, q_ids, _ = _build_gallery_query(
        records, labels_by_token, feat_cache, protocol, "KNOWN_QUERY",
    )
    if sim.size == 0:
        return {"error": "no features"}
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
    return {
        "auc": round(_roc_auc(all_pos, all_neg), 4),
        "tar@1%far": round(_tar_at_far(all_pos, all_neg, 0.01), 4),
        "eer": round(_eer(all_pos, all_neg), 4),
        "n_queries": len(q_ids),
        "n_gallery_ids": len(g_id_set),
    }


_EVALUATORS = {
    "YT_CLOSED_SET": _eval_closed_set,
    "YT_CLOSED_SET_DIAGNOSTIC": _eval_closed_set,
    "DOGFACE_CLOSED_SET": _eval_closed_set,
    "MPDD_CLOSED_SET": _eval_closed_set,
    "YT_OPEN_SET": _eval_open_set,
    "YT_DEVELOPMENT_OPEN_SET": _eval_open_set,
    "YT_CALIBRATION_OPEN_SET": _eval_open_set,
    "MPDD_OPEN_SET": _eval_open_set,
    "SIBETAN_OPEN_SET": _eval_open_set,
    "YT_RANDOM_BACKGROUND_PAIRED_DELTA": _eval_paired_delta,
    "SIBETAN_CROSS_SEQUENCE": _eval_cross_sequence,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True, type=Path)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--crops-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--use-registered-ids", action="store_true",
        help="Replace dataset_identity_id with deterministic registered_dog_id (UUIDv5)",
    )
    args = parser.parse_args()

    raise RuntimeError(
        "visual model baseline publication is disabled until records are "
        "isolated by protocol, episode, gallery_size, shot, modality, and role, "
        "and every use is bound to its exact crop artifact"
    )

    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}", flush=True)
        raise SystemExit(1)

    t0 = time.time()
    print(f"Loading data ...", flush=True)
    assignment = json.loads(args.assignment.read_text())
    labels = json.loads(args.labels.read_text())
    labels_by_token = {r["sample_token"]: r for r in labels["records"]}

    if args.use_registered_ids:
        n_resolved = 0
        for rec in labels.get("records", []):
            did = rec.get("dataset_identity_id", "")
            if did:
                rec["registered_dog_id"] = compute_registered_dog_id(did)
                n_resolved += 1
        print(f"Registered IDs: {n_resolved} identities resolved", flush=True)

    src_bundle = json.loads(args.source_bundle.read_text())
    sid_to_token = {s["source_sample_id"]: s["sample_token"] for s in src_bundle.get("samples", [])}
    token_to_paired_token = {}
    for s in src_bundle.get("samples", []):
        psid = s.get("paired_source_sample_id")
        if psid and psid in sid_to_token:
            token_to_paired_token[s["sample_token"]] = sid_to_token[psid]

    records_by_protocol: dict[str, list[dict]] = defaultdict(list)
    seen_by_proto: dict[str, set[str]] = defaultdict(set)
    for rec in assignment.get("records", []):
        for use in rec.get("uses", []):
            proto = use["protocol"]
            tok = rec["sample_token"]
            if tok not in seen_by_proto[proto]:
                seen_by_proto[proto].add(tok)
                records_by_protocol[proto].append(rec)

    print(f"Data loaded in {time.time()-t0:.1f}s", flush=True)

    # Load ONNX model
    t1 = time.time()
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(args.model), providers=providers)
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"Model '{args.model.name}': {inp.name} → {out.name} "
          f"(dim={out.shape[1]}) loaded in {time.time()-t1:.1f}s", flush=True)

    # Build embedding cache
    t2 = time.time()
    feat_cache = _build_embedding_cache(assignment, args.crops_dir, sess)
    print(f"Embedding cache: {len(feat_cache)} vectors in {time.time()-t2:.1f}s", flush=True)

    # Evaluate all protocols
    results: dict[str, dict] = {}
    for proto, evaluator in _EVALUATORS.items():
        if proto not in records_by_protocol:
            print(f"  {proto}: SKIP", flush=True)
            continue
        t3 = time.time()
        print(f"  {proto}: starting ...", flush=True)
        try:
            kwargs: dict = {}
            if proto == "YT_RANDOM_BACKGROUND_PAIRED_DELTA":
                kwargs["token_to_paired_token"] = token_to_paired_token
            result = evaluator(
                records_by_protocol[proto], labels_by_token,
                feat_cache, proto, **kwargs,
            )
            results[proto] = result
        except Exception as exc:
            import traceback
            results[proto] = {"error": str(exc)}
            print(f"    ERROR: {exc}", flush=True)
            traceback.print_exc()
        print(f"  {proto}: {time.time()-t3:.1f}s", flush=True)

    summary = {
        "schema_version": "cvi.visual_model_baseline_results.v1",
        "model": str(args.model),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps({"status": "DONE", "output": str(args.output)}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
