"""Face encoder losses — reuses NoseID-v1 SubCenter ArcFace, SupCon, triplet."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from identification.training.nose.losses import (
    SubCenterArcFace,
    batch_hard_triplet_loss,
    supervised_contrastive_loss,
    view_consistency_loss,
)


def objective_anchor_coverage(
    labels: torch.Tensor,
    session_ids: torch.Tensor,
    *,
    second_view_available: bool,
) -> dict[str, torch.Tensor]:
    """Report whether scheduled metric objectives have usable observations."""

    if labels.ndim != 1 or session_ids.shape != labels.shape or len(labels) == 0:
        raise ValueError("Face objective labels and sessions must be aligned 1D tensors")
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same_identity = (labels[:, None] == labels[None, :]) & ~self_mask
    cross_session = same_identity & (session_ids[:, None] != session_ids[None, :])
    negatives = labels[:, None] != labels[None, :]
    dtype = torch.float32
    return {
        "supcon_valid_anchor_fraction": same_identity.any(dim=1).to(dtype).mean(),
        "cross_session_triplet_valid_anchor_fraction": (
            cross_session.any(dim=1) & negatives.any(dim=1)
        ).to(dtype).mean(),
        "second_view_coverage": labels.new_tensor(
            float(second_view_available), dtype=dtype
        ),
    }


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
        quality_target: torch.Tensor | None = None,
        curriculum_stage: int = 0,
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
                emb,
                second_view_embedding,
                torch.zeros((emb.shape[0], 6), device=emb.device),
                torch.zeros((emb.shape[0], 6), device=emb.device),
            )
        quality = emb.sum() * 0.0
        if quality_target is not None:
            quality = F.smooth_l1_loss(output["quality"], quality_target)
        total = arc + 0.1 * quality
        if curriculum_stage >= 1:
            total = total + 0.4 * supcon
        if curriculum_stage >= 2:
            total = total + 0.2 * triplet
        if curriculum_stage >= 3:
            total = total + 0.1 * consistency
        return {
            "total": total,
            "subcenter_arcface": arc,
            "supervised_contrastive": supcon,
            "batch_hard_triplet": triplet,
            "view_consistency": consistency,
            "quality": quality,
            **objective_anchor_coverage(
                labels,
                session_ids,
                second_view_available=second_view_embedding is not None,
            ),
        }


class FaceResidualObjective(nn.Module):
    """F5 objective with no unavailable cross-session loss term."""

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
        quality_target: torch.Tensor | None = None,
        curriculum_stage: int = 0,
        margin_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        del curriculum_stage
        if second_view_embedding is None:
            raise ValueError("F5 training requires a paired second view")
        embedding = output["embedding"]
        first_arc = F.cross_entropy(
            self.arcface(embedding, labels, margin_scale=margin_scale), labels
        )
        second_arc = F.cross_entropy(
            self.arcface(second_view_embedding, labels, margin_scale=margin_scale),
            labels,
        )
        arc = 0.5 * (first_arc + second_arc)
        paired_embeddings = torch.cat((embedding, second_view_embedding), dim=0)
        paired_labels = torch.cat((labels, labels), dim=0)
        paired_sessions = torch.cat((session_ids, session_ids), dim=0)
        supcon = supervised_contrastive_loss(
            paired_embeddings, paired_labels, paired_sessions
        )
        consistency = 1.0 - F.cosine_similarity(
            embedding, second_view_embedding, dim=1
        ).mean()
        quality = embedding.sum() * 0.0
        if quality_target is not None:
            quality = F.smooth_l1_loss(output["quality"], quality_target)
        anchor = 1.0 - F.cosine_similarity(
            embedding, output["baseline_embedding"], dim=1
        ).mean()
        total = arc + 0.4 * supcon + 0.1 * consistency + 0.1 * quality + anchor
        zero = embedding.sum() * 0.0
        return {
            "total": total,
            "subcenter_arcface": arc,
            "supervised_contrastive": supcon,
            "batch_hard_triplet": zero,
            "view_consistency": consistency,
            "quality": quality,
            "baseline_anchor": anchor,
            **objective_anchor_coverage(
                paired_labels,
                paired_sessions,
                second_view_available=True,
            ),
        }


__all__ = [
    "FaceIDObjective",
    "FaceResidualObjective",
    "objective_anchor_coverage",
]
