"""Face ReID evaluation — cosine retrieval with identity-disjoint DEV folds."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from evaluation.search_metrics.metrics import (
    compute_cosine_score_matrix,
    evaluate_multi_template_closed_set,
    identity_clustered_bootstrap_ci,
)


@torch.no_grad()
def extract_face_embeddings(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    identity_ids: list[str] = []
    template_ids: list[str] = []
    qualities: list[torch.Tensor] = []
    baselines: list[torch.Tensor] = []
    baseline_available: bool | None = None

    for batch in loader:
        rgb = batch["rgb"].to(device=device, dtype=torch.float32)
        landmarks = batch.get("landmarks")
        if landmarks is not None:
            landmarks = landmarks.to(device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(rgb, landmarks)
        embeddings.append(output["embedding"].float().cpu())
        qualities.append(output["quality"].float().cpu())
        baseline = output.get("baseline_embedding")
        current_available = isinstance(baseline, torch.Tensor)
        if baseline_available is None:
            baseline_available = current_available
        elif baseline_available != current_available:
            raise RuntimeError("Face model baseline output availability changed by batch")
        if current_available:
            baselines.append(baseline.float().cpu())
        identity_ids.extend(batch["registered_dog_id"])
        template_ids.extend(batch["sample_id"])

    result = {
        "embeddings": F.normalize(torch.cat(embeddings), dim=1).numpy(),
        "identity_ids": np.asarray(identity_ids),
        "template_ids": np.asarray(template_ids),
        "qualities": torch.cat(qualities).numpy(),
    }
    if baselines:
        result["baseline_embeddings"] = F.normalize(
            torch.cat(baselines), dim=1
        ).numpy()
    return result


def evaluate_face_retrieval(
    *,
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_identity_ids: np.ndarray,
    gallery_identity_ids: np.ndarray,
    query_template_ids: np.ndarray,
    gallery_template_ids: np.ndarray,
) -> dict[str, Any]:
    scores = compute_cosine_score_matrix(query_embeddings, gallery_embeddings)
    return evaluate_multi_template_closed_set(
        scores,
        query_identity_ids,
        gallery_identity_ids,
        self_match_policy="exclude",
        query_template_ids=query_template_ids,
        gallery_template_ids=gallery_template_ids,
        rank_ks=(1, 5),
    )


def paired_face_retrieval_comparison(
    *,
    baseline_query_embeddings: np.ndarray,
    baseline_gallery_embeddings: np.ndarray,
    candidate_query_embeddings: np.ndarray,
    candidate_gallery_embeddings: np.ndarray,
    query_identity_ids: np.ndarray,
    gallery_identity_ids: np.ndarray,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare exact-query ranks with whole-identity paired bootstrap."""

    identities = np.asarray(gallery_identity_ids)
    queries = np.asarray(query_identity_ids)
    if len(set(identities.tolist())) != len(identities):
        raise ValueError("paired Face comparison requires one gallery row per identity")
    index = {identity: position for position, identity in enumerate(identities.tolist())}
    if any(identity not in index for identity in queries.tolist()):
        raise ValueError("paired Face comparison query identity is absent from gallery")

    def rows(query: np.ndarray, gallery: np.ndarray) -> list[dict[str, float]]:
        scores = compute_cosine_score_matrix(query, gallery)
        output: list[dict[str, float]] = []
        for values, identity in zip(scores, queries.tolist(), strict=True):
            order = np.argsort(-values, kind="stable")
            rank = int(np.flatnonzero(order == index[identity])[0]) + 1
            output.append(
                {
                    "Rank-1": float(rank == 1),
                    "Rank-5": float(rank <= 5),
                    "MRR": 1.0 / rank,
                }
            )
        return output

    baseline = rows(baseline_query_embeddings, baseline_gallery_embeddings)
    candidate = rows(candidate_query_embeddings, candidate_gallery_embeddings)
    metrics = ("Rank-1", "Rank-5", "MRR")
    delta_rows = [
        {
            "bootstrap_cluster_id": identity,
            **{
                metric: candidate_row[metric] - baseline_row[metric]
                for metric in metrics
            },
        }
        for identity, baseline_row, candidate_row in zip(
            queries.tolist(), baseline, candidate, strict=True
        )
    ]
    return {
        "paired_query_count": len(queries),
        "paired_identity_count": len(set(queries.tolist())),
        "baseline_metrics": {
            metric: float(np.mean([row[metric] for row in baseline]))
            for metric in metrics
        },
        "candidate_metrics": {
            metric: float(np.mean([row[metric] for row in candidate]))
            for metric in metrics
        },
        "delta_bootstrap_cis": {
            metric: identity_clustered_bootstrap_ci(
                delta_rows,
                metric=metric,
                resamples=resamples,
                seed=seed + offset,
            )
            for offset, metric in enumerate(metrics)
        },
    }


__all__ = [
    "evaluate_face_retrieval",
    "extract_face_embeddings",
    "paired_face_retrieval_comparison",
]
