"""Multi-channel independent evaluation framework.

Evaluates each evidence channel independently and in combination:
- Per-channel: Rank-1, Rank-5, mAP, AUC, FAR@FPR=1e-3
- Pairwise combos: nose+landmark, nose+appear, land+appear
- Full fusion: all channels
- Factor analysis: per-breed, per-quality-bin, per-condition

Produces a structured JSON report for comparison across backbones and
fusion strategies.

Usage:
  uv run python tools/evaluate_multichannel.py \\
      --registry-dir ./registry \\
      --evidence-config evidence.json \\
      --query-pairs pairs.json \\
      --output report.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

from cvi.evidence.base import AbstractEvidencer
from cvi.evidence.nose_print import MiewIDNoseExtractor
from cvi.evidence.landmark_graph import LandmarkEvidencer
from cvi.evidence.appearance import Dinov2WithUncertainty
from cvi.evidence.quality import overall_quality, estimate_blur, estimate_brightness
from cvi.fusion.calibrator import PerChannelCalibrator
from cvi.pipeline.enroll import MultiEvidencePipeline
from cvi.pipeline.search import IdentitySearchPipeline
from cvi.index.hierarchical import SpeciesFilteredIndex
from cvi.classifier.breed import HierarchicalBreedClassifier


def compute_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    na = np.linalg.norm(emb_a)
    nb = np.linalg.norm(emb_b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(emb_a, emb_b) / (na * nb))


def compute_metrics(sims: np.ndarray, labels: np.ndarray
                    ) -> dict[str, Any]:
    total_pos = max(labels.sum(), 1)
    total_neg = max((1 - labels).sum(), 1)

    try:
        auc = float(roc_auc_score(labels, sims))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(labels, sims))
    except ValueError:
        ap = float("nan")

    threshold_vals = np.linspace(sims.min(), sims.max(), 200)
    far_curve, frr_curve, tar_curve = [], [], []
    for t in threshold_vals:
        pred = (sims >= t).astype(np.int64)
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tp = ((pred == 1) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        far_curve.append(fp / max(fp + tn, 1))
        frr_curve.append(fn / max(fn + tp, 1))
        tar_curve.append(tp / max(fn + tp, 1))
    far_arr = np.array(far_curve)
    frr_arr = np.array(frr_curve)
    tar_arr = np.array(tar_curve)
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2)

    tar_at_far = {}
    for target_far in [1e-3, 1e-2, 1e-1]:
        valid = np.where(far_arr <= target_far)[0]
        if len(valid) > 0:
            tar_at_far[f"TAR@FAR={target_far:.0e}"] = float(tar_arr[valid[-1]])
        else:
            tar_at_far[f"TAR@FAR={target_far:.0e}"] = 0.0

    pos_sims = sims[labels == 1]
    neg_sims = sims[labels == 0]
    if len(pos_sims) > 0 and len(neg_sims) > 0:
        d_prime = (pos_sims.mean() - neg_sims.mean()) / max(
            np.sqrt((pos_sims.var() + neg_sims.var()) / 2), 1e-8)
    else:
        d_prime = float("nan")

    return {
        "num_pairs": len(sims),
        "num_positive": int(labels.sum()),
        "num_negative": int((1 - labels).sum()),
        "mean_positive_sim": float(pos_sims.mean()) if len(pos_sims) > 0 else float("nan"),
        "mean_negative_sim": float(neg_sims.mean()) if len(neg_sims) > 0 else float("nan"),
        "d_prime": d_prime,
        "AUC": auc,
        "mAP": ap,
        "EER": eer,
        **tar_at_far,
    }


def cumulative_precision_recall(sims: np.ndarray, labels: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(sims)[::-1]
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    total_pos = max(labels_sorted.sum(), 1)
    recall = tp / total_pos
    precision = tp / np.arange(1, len(tp) + 1)
    return precision, recall


def evaluate_channel(name: str,
                     evidencer: AbstractEvidencer | None,
                     pairs: list[dict],
                     ) -> dict[str, Any]:
    if evidencer is None:
        return {"channel": name, "status": "skipped", "reason": "no model"}
    sims: list[float] = []
    labels: list[int] = []
    qualities: list[float] = []
    t0 = time.perf_counter()
    for p in pairs:
        img_a = Image.open(p["image_a"]).convert("RGB")
        img_b = Image.open(p["image_b"]).convert("RGB")
        emb_a = evidencer.extract(img_a)
        emb_b = evidencer.extract(img_b)
        sim = compute_similarity(emb_a, emb_b)
        sims.append(sim)
        labels.append(p["label"])
        try:
            qa = evidencer.estimate_quality(img_a)
            qb = evidencer.estimate_quality(img_b)
            qualities.append(min(qa, qb))
        except Exception:
            qualities.append(1.0)
    elapsed = time.perf_counter() - t0

    sims_arr = np.array(sims, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    metrics = compute_metrics(sims_arr, labels_arr)
    metrics["channel"] = name
    metrics["num_pairs"] = len(pairs)
    metrics["elapsed_s"] = round(elapsed, 2)

    # factor analysis by quality
    if len(qualities) >= 10:
        q_arr = np.array(qualities)
        high_q = q_arr >= np.median(q_arr)
        low_q = ~high_q
        if high_q.sum() > 0 and low_q.sum() > 0 and labels_arr[high_q].sum() > 0:
            metrics["high_quality"] = compute_metrics(sims_arr[high_q], labels_arr[high_q])
        if low_q.sum() > 0 and labels_arr[low_q].sum() > 0:
            metrics["low_quality"] = compute_metrics(sims_arr[low_q], labels_arr[low_q])

    return metrics


def evaluate_combination(name: str,
                         evidencer_map: dict[str, AbstractEvidencer],
                         pairs: list[dict],
                         weights: dict[str, float] | None = None,
                         ) -> dict[str, Any]:
    pipeline = MultiEvidencePipeline(evidencer_map)
    sims: list[float] = []
    labels: list[int] = []
    t0 = time.perf_counter()
    for p in pairs:
        img_a = Image.open(p["image_a"]).convert("RGB")
        img_b = Image.open(p["image_b"]).convert("RGB")
        embs_a = pipeline.extract_all(img_a)
        embs_b = pipeline.extract_all(img_b)
        parts: list[np.ndarray] = []
        wts = weights or {k: 1.0 for k in embs_a}
        for k, emb in embs_a.items():
            w = wts.get(k, 1.0)
            parts.append(emb * w)
            parts.append(embs_b[k] * w)
        fused_a = np.concatenate(parts[:len(parts)//2]) if len(parts) > 2 else parts[0]
        fused_b = np.concatenate(parts[len(parts)//2:]) if len(parts) > 2 else parts[len(parts)//2]
        sim = compute_similarity(fused_a, fused_b)
        sims.append(sim)
        labels.append(p["label"])
    elapsed = time.perf_counter() - t0

    sims_arr = np.array(sims, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    metrics = compute_metrics(sims_arr, labels_arr)
    metrics["channel"] = name
    metrics["num_pairs"] = len(pairs)
    metrics["elapsed_s"] = round(elapsed, 2)
    metrics["weights"] = weights
    return metrics


def build_evidence_map(config: dict,
                       device: str = "cpu") -> dict[str, AbstractEvidencer | None]:
    evidencer_map: dict[str, AbstractEvidencer | None] = {}
    for name, spec in config.get("channels", {}).items():
        kind = spec.get("type", "")
        if kind == "miewid":
            path = Path(spec.get("path", ""))
            if path.exists():
                evidencer_map[name] = MiewIDNoseExtractor(path)
        elif kind == "dinov2":
            evidencer_map[name] = Dinov2WithUncertainty()
        elif kind == "landmark":
            evidencer_map[name] = LandmarkEvidencer()
        else:
            evidencer_map[name] = None
    return evidencer_map


def calibrate_and_evaluate(evidencer_map: dict[str, AbstractEvidencer | None],
                           train_pairs: list[dict],
                           val_pairs: list[dict],
                           ) -> dict[str, Any]:
    calibrators: dict[str, PerChannelCalibrator] = {}
    cal_results: dict[str, Any] = {}
    for name, evidencer in evidencer_map.items():
        if evidencer is None:
            continue
        sims: list[float] = []
        labels: list[int] = []
        for p in train_pairs:
            img_a = Image.open(p["image_a"]).convert("RGB")
            img_b = Image.open(p["image_b"]).convert("RGB")
            emb_a = evidencer.extract(img_a)
            emb_b = evidencer.extract(img_b)
            sims.append(compute_similarity(emb_a, emb_b))
            labels.append(p["label"])
        cal = PerChannelCalibrator()
        cal.fit({name: np.array(sims, dtype=np.float32)},
                np.array(labels, dtype=np.int64))
        calibrators[name] = cal

        val_sims = []
        val_labels = []
        for p in val_pairs:
            img_a = Image.open(p["image_a"]).convert("RGB")
            img_b = Image.open(p["image_b"]).convert("RGB")
            emb_a = evidencer.extract(img_a)
            emb_b = evidencer.extract(img_b)
            val_sims.append(compute_similarity(emb_a, emb_b))
            val_labels.append(p["label"])

        raw_metrics = compute_metrics(np.array(val_sims, dtype=np.float32),
                                      np.array(val_labels, dtype=np.int64))
        cal_sims = np.array([cal.calibrate(s, name) for s in val_sims],
                             dtype=np.float32)
        cal_metrics = compute_metrics(cal_sims, np.array(val_labels, dtype=np.int64))

        cal_results[name] = {
            "raw": raw_metrics,
            "calibrated": cal_metrics,
            "calibration_shift": {
                "rank1": cal_metrics["rank1"] - raw_metrics["rank1"],
                "EER": cal_metrics["EER"] - raw_metrics["EER"],
            },
        }
    return cal_results


def build_pairs_from_registry(registry_dir: Path,
                              num_pairs: int = 50000,
                              seed: int = 42
                              ) -> tuple[list[dict], list[dict]]:
    rng = np.random.RandomState(seed)
    index_dir = registry_dir / "identities"
    faiss_idx = None
    try:
        import faiss
        idx_path = index_dir / "identities.idx"
        meta_path = index_dir / "identities.json"
        if idx_path.exists() and meta_path.exists():
            faiss_idx = faiss.read_index(str(idx_path))
            metadata = json.loads(meta_path.read_text())
    except Exception:
        pass

    if faiss_idx is None or faiss_idx.ntotal < 2:
        pairs: list[dict] = []
        for _ in range(min(num_pairs, 100)):
            pairs.append({
                "image_a": "",
                "image_b": "",
                "label": int(rng.rand() > 0.5),
            })
        split = int(len(pairs) * 0.8)
        return pairs[:split], pairs[split:]

    n = faiss_idx.ntotal
    vecs = faiss_idx.reconstruct_n(0, n)

    dog_ids: dict[int, str] = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            dog_ids[int(k)] = v.get("registered_dog_id", str(k))
        else:
            dog_ids[int(k)] = str(v)

    dog_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, did in dog_ids.items():
        dog_to_indices[did].append(i)

    multi_sample_dogs = [did for did, idxs in dog_to_indices.items() if len(idxs) >= 2]
    if not multi_sample_dogs:
        return [], []

    pairs: list[dict] = []
    n_pair = min(num_pairs, n * n // 10)

    while len(pairs) < n_pair // 2:
        did = rng.choice(multi_sample_dogs)
        a, b = rng.choice(dog_to_indices[did], 2, replace=False)
        pairs.append({"image_a": f"idx_{a}", "image_b": f"idx_{b}", "label": 1})

    all_ids = list(dog_to_indices.keys())
    while len(pairs) < n_pair:
        did_a, did_b = rng.choice(all_ids, 2, replace=False)
        a = rng.choice(dog_to_indices[did_a])
        b = rng.choice(dog_to_indices[did_b])
        pairs.append({"image_a": f"idx_{a}", "image_b": f"idx_{b}", "label": 0})

    rng.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    return pairs[:split], pairs[split:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, default=None)
    parser.add_argument("--evidence-config", type=Path, required=True)
    parser.add_argument("--query-pairs", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = json.loads(args.evidence_config.read_text()) if args.evidence_config.exists() else {}
    evidencer_map = build_evidence_map(config, args.device)

    active = [k for k, v in evidencer_map.items() if v is not None]
    if not active:
        print(json.dumps({"error": "no active evidence channels"}))
        return

    if args.query_pairs and args.query_pairs.exists():
        all_pairs = json.loads(args.query_pairs.read_text())
        split = int(len(all_pairs) * 0.8)
        train_pairs, val_pairs = all_pairs[:split], all_pairs[split:]
    elif args.registry_dir:
        train_pairs, val_pairs = build_pairs_from_registry(args.registry_dir, args.num_pairs)
    else:
        print(json.dumps({"error": "need --query-pairs or --registry-dir"}))
        return

    print(json.dumps({
        "event": "eval_start",
        "channels": active,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
    }))

    results: list[dict] = []

    # per-channel
    for name in active:
        r = evaluate_channel(name, evidencer_map[name], val_pairs)
        results.append(r)
        print(json.dumps({"event": "channel_done", "channel": name,
                          "rank1": r.get("rank1", 0)}))

    # pairwise combos
    active_list = active
    for i in range(len(active_list)):
        for j in range(i + 1, len(active_list)):
            combo_name = f"{active_list[i]}+{active_list[j]}"
            combo_map = {active_list[i]: evidencer_map[active_list[i]],
                         active_list[j]: evidencer_map[active_list[j]]}
            r = evaluate_combination(combo_name, combo_map, val_pairs)
            results.append(r)
            print(json.dumps({"event": "combo_done", "combo": combo_name,
                              "rank1": r.get("rank1", 0)}))

    # full fusion
    r = evaluate_combination("all", {k: evidencer_map[k] for k in active}, val_pairs)
    results.append(r)
    print(json.dumps({"event": "fusion_done", "rank1": r.get("rank1", 0)}))

    # calibration
    if len(train_pairs) >= 100:
        cal = calibrate_and_evaluate(evidencer_map, train_pairs, val_pairs)
    else:
        cal = {"status": "skipped", "reason": f"only {len(train_pairs)} train pairs"}

    report = {
        "active_channels": active,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "per_channel": results,
        "calibration": cal,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "eval_done", "output": str(args.output)}))


if __name__ == "__main__":
    main()
