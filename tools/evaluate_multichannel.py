"""Multi-channel independent evaluation framework.

Evaluates each evidence channel independently and in combination.
Two evaluation modes:

  Verification:
    Compares image pairs (known match/non-match).
    Metrics: AUC, AP, EER, TAR@FAR, d-prime.

  Retrieval:
    Given gallery embeddings + query embeddings with identity labels.
    Metrics: Rank-1/5/10, mAP.

Usage:
  uv run python tools/evaluate_multichannel.py \\
      --evidence-config evidence.json \\
      --query-pairs pairs.json \\
      --output report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cvi.evidence.base import AbstractEvidencer
from cvi.evidence.nose_print import MiewIDNoseExtractor
from cvi.evidence.landmark_graph import LandmarkEvidencer
from cvi.evidence.appearance import Dinov2WithUncertainty
from cvi.evaluation.verification import compute_verification_metrics
from cvi.evaluation.retrieval import compute_retrieval_metrics
from cvi.evaluation.calibration import compute_calibration_metrics
from cvi.pipeline.enroll import MultiEvidencePipeline


def compute_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    na = np.linalg.norm(emb_a)
    nb = np.linalg.norm(emb_b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(emb_a, emb_b) / (na * nb))


def _validate_pair(p: dict) -> None:
    if "image_a" not in p or "image_b" not in p:
        raise ValueError(f"pair missing image_a or image_b: {p}")
    if "label" not in p:
        raise ValueError(f"pair missing label: {p}")
    path_a = p["image_a"]
    path_b = p["image_b"]
    if not path_a or not path_b:
        raise ValueError(f"empty image path in pair: {p}")
    if path_a.startswith("idx_"):
        raise ValueError(
            f"image path uses FAISS idx_ format instead of file path: {path_a}. "
            f"The --registry-dir auto-pair builder has been removed."
        )


def evaluate_channel_verification(
    name: str,
    evidencer: AbstractEvidencer | None,
    pairs: list[dict],
) -> dict[str, Any]:
    if evidencer is None:
        return {"channel": name, "status": "skipped", "reason": "no model"}
    if not pairs:
        return {"channel": name, "status": "skipped", "reason": "empty pair set"}
    sims: list[float] = []
    labels: list[int] = []
    t0 = time.perf_counter()
    for p in pairs:
        _validate_pair(p)
        img_a = Image.open(p["image_a"]).convert("RGB")
        img_b = Image.open(p["image_b"]).convert("RGB")
        emb_a = evidencer.extract(img_a)
        emb_b = evidencer.extract(img_b)
        sim = compute_similarity(emb_a, emb_b)
        sims.append(sim)
        labels.append(p["label"])
    elapsed = time.perf_counter() - t0
    sims_arr = np.array(sims, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    metrics = compute_verification_metrics(sims_arr, labels_arr)
    metrics["channel"] = name
    metrics["num_pairs"] = len(pairs)
    metrics["elapsed_s"] = round(elapsed, 2)
    return metrics


def evaluate_channel_retrieval(
    name: str,
    evidencer: AbstractEvidencer | None,
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    top_k: int = 10,
) -> dict:
    if evidencer is None:
        return {"channel": name, "status": "skipped", "reason": "no model"}
    metrics = compute_retrieval_metrics(
        query_embeddings, gallery_embeddings, query_ids, gallery_ids, top_k
    )
    metrics["channel"] = name
    return metrics


def evaluate_combination(
    name: str,
    evidencer_map: dict[str, AbstractEvidencer],
    pairs: list[dict],
) -> dict[str, Any]:
    if not pairs:
        return {"channel": name, "status": "skipped", "reason": "empty pair set"}
    pipeline = MultiEvidencePipeline(evidencer_map)
    sims: list[float] = []
    labels: list[int] = []
    t0 = time.perf_counter()
    for p in pairs:
        _validate_pair(p)
        img_a = Image.open(p["image_a"]).convert("RGB")
        img_b = Image.open(p["image_b"]).convert("RGB")
        embs_a = pipeline.extract_all(img_a)
        embs_b = pipeline.extract_all(img_b)
        fused_a = _fuse_embeddings(embs_a)
        fused_b = _fuse_embeddings(embs_b)
        sim = compute_similarity(fused_a, fused_b)
        sims.append(sim)
        labels.append(p["label"])
    elapsed = time.perf_counter() - t0
    sims_arr = np.array(sims, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    metrics = compute_verification_metrics(sims_arr, labels_arr)
    metrics["channel"] = name
    metrics["num_pairs"] = len(pairs)
    metrics["elapsed_s"] = round(elapsed, 2)
    return metrics


def _fuse_embeddings(
    embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    parts = list(embeddings.values())
    if not parts:
        return np.zeros(1, dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    fused = np.concatenate(parts)
    norm = np.linalg.norm(fused)
    return fused / norm if norm > 0 else fused


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"query pairs file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list of pair dicts, got {type(data)}")
    if len(data) == 0:
        raise ValueError("pair list is empty")
    return data


def split_pairs(
    pairs: list[dict],
    train_frac: float = 0.60,
    cal_frac: float = 0.20,
) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(pairs)
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    return pairs[:n_train], pairs[n_train : n_train + n_cal], pairs[n_train + n_cal :]


def build_evidence_map(
    config: dict,
) -> dict[str, AbstractEvidencer | None]:
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


def calibrate_and_evaluate(
    evidencer_map: dict[str, AbstractEvidencer | None],
    cal_pairs: list[dict],
    test_pairs: list[dict],
) -> dict:
    from cvi.fusion.calibrator import PerChannelCalibrator
    calibrators: dict[str, PerChannelCalibrator] = {}
    cal_results: dict[str, Any] = {}
    for name, evidencer in evidencer_map.items():
        if evidencer is None:
            continue
        cal_sims: list[float] = []
        cal_labels: list[int] = []
        for p in cal_pairs:
            _validate_pair(p)
            img_a = Image.open(p["image_a"]).convert("RGB")
            img_b = Image.open(p["image_b"]).convert("RGB")
            emb_a = evidencer.extract(img_a)
            emb_b = evidencer.extract(img_b)
            cal_sims.append(compute_similarity(emb_a, emb_b))
            cal_labels.append(p["label"])
        cal = PerChannelCalibrator()
        cal.fit(
            {name: np.array(cal_sims, dtype=np.float32)},
            np.array(cal_labels, dtype=np.int64),
        )
        calibrators[name] = cal
        test_sims = []
        test_labels = []
        for p in test_pairs:
            _validate_pair(p)
            img_a = Image.open(p["image_a"]).convert("RGB")
            img_b = Image.open(p["image_b"]).convert("RGB")
            emb_a = evidencer.extract(img_a)
            emb_b = evidencer.extract(img_b)
            test_sims.append(compute_similarity(emb_a, emb_b))
            test_labels.append(p["label"])
        raw_metrics = compute_verification_metrics(
            np.array(test_sims, dtype=np.float32),
            np.array(test_labels, dtype=np.int64),
        )
        cal_sims_arr = np.array(
            [cal.calibrate(s, name) for s in test_sims], dtype=np.float32
        )
        cal_metrics = compute_verification_metrics(
            cal_sims_arr, np.array(test_labels, dtype=np.int64)
        )
        cal_results[name] = {
            "raw": raw_metrics,
            "calibrated": cal_metrics,
        }
    return cal_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-config", type=Path, required=True)
    parser.add_argument("--query-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.evidence_config.read_text())
    evidencer_map = build_evidence_map(config)
    active = [k for k, v in evidencer_map.items() if v is not None]
    if not active:
        print(json.dumps({"error": "no active evidence channels"}))
        raise SystemExit(1)

    pairs = load_pairs(args.query_pairs)
    train_pairs, cal_pairs, test_pairs = split_pairs(pairs)

    print(json.dumps({
        "event": "eval_start",
        "channels": active,
        "train_pairs": len(train_pairs),
        "cal_pairs": len(cal_pairs),
        "test_pairs": len(test_pairs),
    }))

    results: list[dict] = []

    for name in active:
        r = evaluate_channel_verification(name, evidencer_map[name], test_pairs)
        results.append(r)

    active_list = active
    for i in range(len(active_list)):
        for j in range(i + 1, len(active_list)):
            combo_name = f"{active_list[i]}+{active_list[j]}"
            combo_map = {
                active_list[i]: evidencer_map[active_list[i]],
                active_list[j]: evidencer_map[active_list[j]],
            }
            r = evaluate_combination(combo_name, combo_map, test_pairs)
            results.append(r)

    r = evaluate_combination(
        "all", {k: evidencer_map[k] for k in active}, test_pairs
    )
    results.append(r)

    if len(cal_pairs) >= 100:
        cal = calibrate_and_evaluate(evidencer_map, cal_pairs, test_pairs)
    else:
        cal = {"status": "skipped", "reason": f"only {len(cal_pairs)} calibration pairs"}

    report = {
        "schema_version": "cvi.evaluation.report.v1",
        "pinned_commit": "0ba3b1bef4ad6bd18ee516260cf938e9e43ca659",
        "active_channels": active,
        "train_pairs": len(train_pairs),
        "cal_pairs": len(cal_pairs),
        "test_pairs": len(test_pairs),
        "per_channel": results,
        "calibration": cal,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"event": "eval_done", "output": str(args.output)}))


if __name__ == "__main__":
    main()
