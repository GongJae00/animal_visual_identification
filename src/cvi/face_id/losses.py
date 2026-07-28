"""Face ReID losses — reuses NoseID-v1 SubCenter ArcFace, SupCon, triplet."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cvi.nose_id.losses import (
    SubCenterArcFace,
    batch_hard_triplet_loss,
    supervised_contrastive_loss,
    view_consistency_loss,
)


class FaceIDObjective(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.arcface = SubCenterArcFace(embedding_dim, num_classes)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        session_ids: torch.Tensor,
        *,
        second_view_embedding: torch.Tensor | None = None,
        margin_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        emb = output["embedding"]
        arc = F.cross_entropy(
            self.arcface(emb, labels, margin_scale=margin_scale), labels
        )
        supcon = supervised_contrastive_loss(emb, labels, session_ids)
        triplet = batch_hard_triplet_loss(emb, labels, session_ids)
        consistency = emb.sum() * 0.0
        if second_view_embedding is not None:
            consistency = view_consistency_loss(
                emb, second_view_embedding,
                torch.zeros((emb.shape[0], 6), device=emb.device),
                torch.zeros((emb.shape[0], 6), device=emb.device),
            )
        total = arc + 0.4 * supcon + 0.2 * triplet + 0.1 * consistency
        return {
            "total": total,
            "subcenter_arcface": arc,
            "supervised_contrastive": supcon,
            "batch_hard_triplet": triplet,
            "view_consistency": consistency,
        }


__all__ = ["FaceIDObjective"]
