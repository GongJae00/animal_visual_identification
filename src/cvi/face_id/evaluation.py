"""Face ReID evaluation — cosine retrieval with identity-disjoint DEV folds."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from cvi.evaluation.retrieval import (
    compute_cosine_score_matrix,
    evaluate_multi_template_closed_set,
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

    for batch in loader:
        rgb = batch["rgb"].to(device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(rgb)
        embeddings.append(output["embedding"].float().cpu())
        qualities.append(output["quality"].float().cpu())
        identity_ids.extend(batch["registered_dog_id"])
        template_ids.extend(batch["sample_id"])

    return {
        "embeddings": F.normalize(torch.cat(embeddings), dim=1).numpy(),
        "identity_ids": np.asarray(identity_ids),
        "template_ids": np.asarray(template_ids),
        "qualities": torch.cat(qualities).numpy(),
    }


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


__all__ = ["evaluate_face_retrieval", "extract_face_embeddings"]
