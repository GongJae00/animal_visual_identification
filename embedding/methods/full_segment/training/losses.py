"""Triplet-only objective for the Full128 baseline family."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def batch_hard_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, *, margin: float = 0.2
) -> torch.Tensor:
    """Return mean batch-hard Euclidean triplet loss for valid P x K batches."""

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("triplet embeddings must have shape [B,D] with B >= 2")
    if labels.shape != (embeddings.shape[0],) or labels.dtype != torch.long:
        raise ValueError("triplet labels must be int64 [B]")
    if not torch.isfinite(embeddings).all():
        raise ValueError("triplet embeddings must be finite")
    if margin != 0.2:
        raise ValueError("Full128 batch-hard triplet margin is fixed at 0.2")
    normalized = F.normalize(embeddings, dim=1)
    distances = torch.cdist(normalized, normalized, p=2)
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    negatives = labels[:, None] != labels[None, :]
    if not torch.all(positives.any(dim=1)):
        raise ValueError("every triplet anchor requires a distinct positive")
    if not torch.all(negatives.any(dim=1)):
        raise ValueError("every triplet anchor requires a negative identity")
    hardest_positive = distances.masked_fill(~positives, float("-inf")).max(dim=1).values
    hardest_negative = distances.masked_fill(~negatives, float("inf")).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


__all__ = ["batch_hard_triplet_loss"]
