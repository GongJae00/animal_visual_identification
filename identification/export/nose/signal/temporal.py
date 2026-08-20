"""Quality-aware temporal aggregation for already-extracted Nose embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalEmbeddingResult:
    embedding: np.ndarray
    normalized_qualities: tuple[float, ...]
    consensus_cosines: tuple[float, ...]
    accepted_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    angular_dispersion: float
    aggregation: str

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema_version": "identification.nose.nose_temporal_embedding.v1",
            "normalized_qualities": list(self.normalized_qualities),
            "consensus_cosines": list(self.consensus_cosines),
            "accepted_indices": list(self.accepted_indices),
            "rejected_indices": list(self.rejected_indices),
            "angular_dispersion": self.angular_dispersion,
            "aggregation": self.aggregation,
        }


def aggregate_nose_embeddings(
    embeddings: Sequence[np.ndarray],
    qualities: Sequence[float] | None = None,
    *,
    reject_outliers: bool = False,
    outlier_mad_scale: float = 2.5,
    minimum_consensus_cosine: float = -1.0,
) -> TemporalEmbeddingResult:
    """Aggregate observed embeddings while retaining at least two temporal views."""

    if not isinstance(reject_outliers, bool):
        raise TypeError("reject_outliers must be bool")
    if not math.isfinite(outlier_mad_scale) or outlier_mad_scale < 0.0:
        raise ValueError("outlier_mad_scale must be finite and non-negative")
    if (
        not math.isfinite(minimum_consensus_cosine)
        or minimum_consensus_cosine < -1.0
        or minimum_consensus_cosine > 1.0
    ):
        raise ValueError("minimum_consensus_cosine must be finite and in [-1,1]")
    if len(embeddings) < 2:
        raise ValueError("temporal Nose aggregation requires at least two embeddings")
    rows: list[np.ndarray] = []
    dimension = None
    for embedding in embeddings:
        value = np.asarray(embedding, dtype=np.float32)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError("temporal Nose embeddings must be finite vectors")
        if dimension is None:
            dimension = value.shape[0]
        if value.shape != (dimension,):
            raise ValueError("temporal Nose embedding dimensions differ")
        norm = float(np.linalg.norm(value))
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError("temporal Nose embeddings must have non-zero norm")
        rows.append(np.asarray(value / norm, dtype=np.float32))
    stack = np.stack(rows)

    if qualities is None:
        quality = np.ones(len(rows), dtype=np.float64)
    else:
        quality = np.asarray(qualities, dtype=np.float64)
        if (
            quality.shape != (len(rows),)
            or not np.isfinite(quality).all()
            or np.any((quality < 0.0) | (quality > 1.0))
            or not np.any(quality > 0.0)
        ):
            raise ValueError("temporal Nose qualities must be finite [0,1] values")
    quality = np.maximum(quality, 1e-6)
    quality /= quality.sum()

    initial = np.sum(stack * quality[:, None], axis=0)
    initial_norm = float(np.linalg.norm(initial))
    if initial_norm <= 1e-8:
        raise ValueError("temporal Nose embeddings cancel to zero")
    initial /= initial_norm
    cosines = np.asarray(stack @ initial, dtype=np.float64)
    median = float(np.median(cosines))
    mad = float(np.median(np.abs(cosines - median)))
    # Repeated near-identical observations can make MAD exactly zero. A small
    # angular floor avoids treating harmless numerical variation as an outlier.
    threshold = max(
        float(minimum_consensus_cosine),
        median - outlier_mad_scale * max(mad, 0.02),
    )
    if reject_outliers:
        accepted = np.flatnonzero(cosines >= threshold)
        if accepted.size < 2:
            accepted = np.argsort(-cosines, kind="stable")[:2]
        accepted = np.sort(accepted)
    else:
        accepted = np.arange(len(rows), dtype=np.int64)
    rejected = np.asarray(
        [index for index in range(len(rows)) if index not in set(accepted.tolist())],
        dtype=np.int64,
    )

    accepted_weights = quality[accepted]
    accepted_weights /= accepted_weights.sum()
    fused = np.sum(stack[accepted] * accepted_weights[:, None], axis=0)
    fused_norm = float(np.linalg.norm(fused))
    if not math.isfinite(fused_norm) or fused_norm <= 1e-8:
        raise ValueError("temporal Nose aggregate has zero norm")
    fused = np.asarray(fused / fused_norm, dtype=np.float32)
    dispersion = float(
        np.sum(accepted_weights * (1.0 - np.clip(stack[accepted] @ fused, -1.0, 1.0)))
    )
    quality_weighted = qualities is not None
    if reject_outliers:
        aggregation = (
            "QUALITY_WEIGHTED_CONSENSUS_L2_MEAN"
            if quality_weighted
            else "UNWEIGHTED_CONSENSUS_L2_MEAN"
        )
    else:
        aggregation = (
            "QUALITY_WEIGHTED_L2_MEAN"
            if quality_weighted
            else "UNWEIGHTED_L2_MEAN"
        )
    return TemporalEmbeddingResult(
        embedding=fused,
        normalized_qualities=tuple(float(value) for value in quality),
        consensus_cosines=tuple(float(value) for value in cosines),
        accepted_indices=tuple(int(value) for value in accepted),
        rejected_indices=tuple(int(value) for value in rejected),
        angular_dispersion=dispersion,
        aggregation=aggregation,
    )


__all__ = ["TemporalEmbeddingResult", "aggregate_nose_embeddings"]
