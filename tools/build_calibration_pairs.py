"""Build match/non-match pairs for score calibration.

Reads oracle crops and their registered_dog_id labels, extracts
embeddings via a trained backbone, and produces a labeled pair dataset
for IsotonicRegression calibration (one score = cosine similarity per pair).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cvi.trainer import ArcFaceModel, TrainConfig, compute_embeddings


def build_pairs(labels: np.ndarray, embeddings: np.ndarray,
                num_match_pairs: int = 5000,
                num_nonmatch_pairs: int = 5000,
                seed: int = 42
                ) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    label_to_indices: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        label_to_indices.setdefault(int(lab), []).append(i)

    scores: list[float] = []
    pair_labels: list[int] = []

    for _ in range(num_match_pairs):
        lab = rng.choice(list(label_to_indices.keys()))
        a, b = rng.sample(label_to_indices[lab], 2)
        cos = float(np.dot(embeddings[a], embeddings[b]))
        scores.append(cos)
        pair_labels.append(1)

    for _ in range(num_nonmatch_pairs):
        lab_a, lab_b = rng.sample(list(label_to_indices.keys()), 2)
        a = rng.choice(label_to_indices[lab_a])
        b = rng.choice(label_to_indices[lab_b])
        cos = float(np.dot(embeddings[a], embeddings[b]))
        scores.append(cos)
        pair_labels.append(0)

    return np.array(scores, dtype=np.float32), np.array(pair_labels, dtype=np.int64)


def calibrate_from_checkpoint(
    checkpoint_path: Path,
    crop_root: Path,
    train_binding: list[dict],
    output_path: Path,
    device: torch.device = torch.device("cpu"),
    embedding_dim: int = 384,
    num_match: int = 5000,
    num_nonmatch: int = 5000,
) -> None:
    from cvi.post_search import ScoreCalibrator
    from cvi.trainer import OracleCropDataset, _build_label_index, _build_dataloader

    label_to_index = _build_label_index(train_binding)
    dataset = OracleCropDataset(crop_root, train_binding, label_to_index,
                                 use_cache=True)
    loader = _build_dataloader(dataset, None, TrainConfig(), device, shuffle=False)

    cfg = TrainConfig(embedding_dim=embedding_dim, num_classes=len(label_to_index))
    model = ArcFaceModel(cfg)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device)

    embs, labels = compute_embeddings(model, loader, device)

    scores, pair_labels = build_pairs(
        labels, embs,
        num_match_pairs=num_match,
        num_nonmatch_pairs=num_nonmatch,
    )

    cal = ScoreCalibrator()
    cal.fit({"all": scores}, pair_labels)
    cal.save(output_path)
    print(f"Calibrator saved: {output_path}")
    print(f"  {num_match} match pairs, {num_nonmatch} non-match pairs")
    print(f"  Score range: [{scores[pair_labels == 1].min():.3f}, "
          f"{scores[pair_labels == 1].max():.3f}] (match)")
    print(f"  Score range: [{scores[pair_labels == 0].min():.3f}, "
          f"{scores[pair_labels == 0].max():.3f}] (non-match)")
