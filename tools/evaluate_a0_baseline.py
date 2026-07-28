"""Evaluate A0 — frozen DINOv2 384D appearance baseline on yt-bb-dog DEV."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from cvi.canid_data.adapters import adapt_yt_bb_dog
from cvi.evidence.appearance import ReceiptBoundDinov2Small
from cvi.evaluation.retrieval import (
    compute_cosine_score_matrix,
    evaluate_multi_template_closed_set,
)
from cvi.visualization import contact_sheet, attention_heatmap


def main() -> None:
    data_root = Path("/mnt/r/research-data/canine_video_identity_secure/datasets/yt-bb-dog")
    model_dir = Path(
        "/mnt/r/research-data/canine_video_identity_secure/checkpoints/"
        "deployment-eligible/hf-facebook-dinov2-small-"
        "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    )
    weight_bundle = Path(
        "/mnt/r/research-data/canine_video_identity_secure/manifests/"
        "pretrained-weights/2026-07-22-v1/dinov2-small-weight-intake.json"
    )
    preproc_bundle = Path(
        "/mnt/r/research-data/canine_video_identity_secure/manifests/"
        "pretrained-weights/2026-07-22-v1/dinov2-small-preprocessor-intake.json"
    )

    print("[A0] loading samples...")
    all_samples = adapt_yt_bb_dog(data_root)
    test_samples = [s for s in all_samples if s.split_role == "test"]
    print(f"[A0] yt-bb-dog test: {len(test_samples)} images, "
          f"{len({s.registered_identity_id for s in test_samples})} identities")

    by_identity: dict[str, list] = {}
    for s in test_samples:
        by_identity.setdefault(s.registered_identity_id, []).append(s)

    gallery_samples = []
    query_samples = []
    for identity, samples in sorted(by_identity.items()):
        sorted_samples = sorted(samples, key=lambda s: s.sample_id)
        gallery_samples.append(sorted_samples[0])
        query_samples.extend(sorted_samples[1:])

    print(f"[A0] gallery={len(gallery_samples)} query={len(query_samples)}")

    print("[A0] loading DINOv2 receipt-bound model...")
    evidencer = ReceiptBoundDinov2Small(
        model_directory=str(model_dir),
        weight_intake_bundle=str(weight_bundle),
        preprocessor_intake_bundle=str(preproc_bundle),
        device="cuda",
        max_batch_size=32,
    )

    def extract(samples, label):
        print(f"[A0] extracting {label} embeddings ({len(samples)} images)...")
        embeddings = np.empty((len(samples), 384), dtype=np.float32)
        batch_size = 32
        t0 = time.perf_counter()
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            images = [
                Image.open(data_root / s.image_path).convert("RGB") for s in batch
            ]
            embeddings[start : start + len(batch)] = evidencer.extract_batch(images)
            if start % 256 == 0:
                print(f"  [{label}] {start}/{len(samples)}")
        elapsed = time.perf_counter() - t0
        print(f"  [{label}] done in {elapsed:.1f}s, {len(samples)/elapsed:.1f} img/s")
        return embeddings

    gallery_embs = extract(gallery_samples, "gallery")
    query_embs = extract(query_samples, "query")

    print("[A0] contact sheets...")
    thumb_images = [
        Image.open(data_root / s.image_path).convert("RGB")
        for s in test_samples[:64]
    ]
    contact_sheet(thumb_images, stage="a0_baseline", title="yt_bb_dog_test")

    print("[A0] computing cosine score matrix...")
    scores = compute_cosine_score_matrix(query_embs, gallery_embs)

    print("[A0] evaluating retrieval...")
    metrics = evaluate_multi_template_closed_set(
        scores,
        query_identity_ids=np.asarray([s.registered_identity_id for s in query_samples]),
        gallery_template_identity_ids=np.asarray([s.registered_identity_id for s in gallery_samples]),
        self_match_policy="exclude",
        query_template_ids=np.asarray([s.sample_id for s in query_samples]),
        gallery_template_ids=np.asarray([s.sample_id for s in gallery_samples]),
        rank_ks=(1, 5, 10),
    )

    report = {
        "model": "DINOv2-small (frozen, receipt-bound)",
        "dataset": "yt-bb-dog (test only)",
        "gallery_count": len(gallery_samples),
        "query_count": len(query_samples),
        "identity_count": len(by_identity),
        "dimension": 384,
        "Rank-1": metrics["Rank-1"],
        "Rank-5": metrics["Rank-5"],
        "Rank-10": metrics.get("Rank-10", 0),
        "mAP": metrics["mAP"],
        "mINP": metrics["mINP"],
        "MRR": metrics["MRR"],
    }
    print(json.dumps(report, indent=2))

    report_path = Path("visualization/a0_baseline/report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[A0] report saved to {report_path}")


if __name__ == "__main__":
    main()
