"""Build match/non-match pairs for score calibration.

The legacy checkpoint-to-pairs path is retained only as an explicit
fail-closed boundary. Pair construction remains available for already
admitted score/label arrays.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


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
    raise RuntimeError(
        "legacy pair calibration is disabled: it uses training identities and "
        "does not consume the protected calibration gallery/query contract"
    )
