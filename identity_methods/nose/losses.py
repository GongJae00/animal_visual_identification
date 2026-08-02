"""Canonical NoseID-v1 metric-learning objectives."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SubCenterArcFace(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        *,
        subcenters: int = 3,
        scale: float = 32.0,
        margin: float = 0.30,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.subcenters = subcenters
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, subcenters, embedding_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor, *, margin_scale: float = 1.0
    ) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.weight.shape[2]:
            raise ValueError("ArcFace embeddings have an invalid shape")
        if not torch.isfinite(embeddings).all():
            raise ValueError("ArcFace embeddings must be finite")
        if labels.shape != (embeddings.shape[0],) or labels.dtype != torch.long:
            raise ValueError("ArcFace labels must be int64 [B]")
        if torch.any((labels < 0) | (labels >= self.num_classes)):
            raise ValueError("ArcFace label is outside the class range")
        if not 0.0 <= margin_scale <= 1.0:
            raise ValueError("ArcFace margin_scale must be in [0, 1]")
        normalized_weight = F.normalize(self.weight, dim=2)
        cosine = torch.einsum("bd,ckd->bck", F.normalize(embeddings, dim=1), normalized_weight).max(dim=2).values
        target = cosine.gather(1, labels[:, None]).squeeze(1).clamp(-1 + 1e-7, 1 - 1e-7)
        margin = self.margin * margin_scale
        cos_m = target.new_tensor(math.cos(margin))
        sin_m = target.new_tensor(math.sin(margin))
        threshold = math.cos(math.pi - margin)
        mm = target.new_tensor(math.sin(math.pi - margin) * margin)
        sine = torch.sqrt((1.0 - target.square()).clamp_min(1e-7))
        phi_raw = target * cos_m - sine * sin_m
        phi = torch.where(target > threshold, phi_raw, target - mm)
        logits = cosine.clone()
        logits.scatter_(1, labels[:, None], phi[:, None].to(dtype=logits.dtype))
        return logits * self.scale


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    session_ids: torch.Tensor | None = None,
    *,
    temperature: float = 0.07,
    weak_positive_weight: float = 0.5,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    count = embeddings.shape[0]
    logits = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    logits = logits.masked_fill(self_mask, -1e4)
    same_identity = labels[:, None] == labels[None, :]
    positive = same_identity & ~self_mask
    weights = positive.float()
    if session_ids is not None:
        cross_session = session_ids[:, None] != session_ids[None, :]
        has_cross_session = (positive & cross_session).any(dim=1, keepdim=True)
        weights = torch.where(
            has_cross_session,
            (positive & cross_session).float(),
            positive.float() * weak_positive_weight,
        )
    valid = weights.sum(dim=1) > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(weights * log_probability).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return per_anchor[valid].mean()


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    session_ids: torch.Tensor,
    *,
    margin: float = 0.20,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    distance = 1.0 - embeddings @ embeddings.T
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = (labels[:, None] == labels[None, :]) & (session_ids[:, None] != session_ids[None, :]) & ~self_mask
    negative = labels[:, None] != labels[None, :]
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    hardest_positive = distance.masked_fill(~positive, float("-inf")).max(dim=1).values
    closest_negative = distance.masked_fill(~negative, float("inf")).min(dim=1).values
    return F.relu(hardest_positive[valid] - closest_negative[valid] + margin).mean()


def view_consistency_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    first_degradation: torch.Tensor,
    second_degradation: torch.Tensor,
) -> torch.Tensor:
    for value in (first_degradation, second_degradation):
        if value.shape != (first.shape[0], 6) or not torch.isfinite(value).all():
            raise ValueError("consistency degradation must be finite [B,6]")
    first_information_loss = torch.maximum(
        torch.maximum(first_degradation[:, 0], first_degradation[:, 1]),
        torch.maximum(first_degradation[:, 2], first_degradation[:, 3]),
    )
    second_information_loss = torch.maximum(
        torch.maximum(second_degradation[:, 0], second_degradation[:, 1]),
        torch.maximum(second_degradation[:, 2], second_degradation[:, 3]),
    )
    weight = torch.minimum(
        (1.0 - first_information_loss).clamp(0.25, 1.0),
        (1.0 - second_information_loss).clamp(0.25, 1.0),
    )
    per_sample = 1.0 - F.cosine_similarity(first, second, dim=1)
    return torch.sum(per_sample * weight) / weight.sum().clamp_min(1e-6)


class NoseIDObjective(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.arcface = SubCenterArcFace(embedding_dim, num_classes)
        self.register_buffer("quality_auxiliary_weight", torch.tensor(0.05))

    def forward(
        self,
        output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        session_ids: torch.Tensor,
        *,
        second_view_output: dict[str, torch.Tensor] | None = None,
        first_degradation_target: torch.Tensor | None = None,
        second_degradation_target: torch.Tensor | None = None,
        margin_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        embedding = output["embedding"]
        arc = F.cross_entropy(self.arcface(embedding, labels, margin_scale=margin_scale), labels)
        supcon = supervised_contrastive_loss(embedding, labels, session_ids)
        triplet = batch_hard_triplet_loss(embedding, labels, session_ids)
        has_second = second_view_output is not None
        has_targets = (
            first_degradation_target is not None
            and second_degradation_target is not None
        )
        if has_second != has_targets:
            raise ValueError("second view and both degradation targets are required together")
        consistency = (
            embedding.sum() * 0.0
            if second_view_output is None
            else view_consistency_loss(
                embedding,
                second_view_output["embedding"],
                first_degradation_target,
                second_degradation_target,
            )
        )
        branch_aux = supervised_contrastive_loss(output["z_rgb"], labels, session_ids) + supervised_contrastive_loss(output["z_texture"], labels, session_ids)
        quality_aux = (
            embedding.sum() * 0.0
            if second_view_output is None
            else 0.5
            * (
                F.smooth_l1_loss(
                    output["degradation_predictions"], first_degradation_target
                )
                + F.smooth_l1_loss(
                    second_view_output["degradation_predictions"],
                    second_degradation_target,
                )
            )
        )
        total = (
            arc
            + 0.4 * supcon
            + 0.2 * triplet
            + 0.1 * consistency
            + 0.1 * branch_aux
            + self.quality_auxiliary_weight * quality_aux
        )
        return {
            "total": total,
            "subcenter_arcface": arc,
            "supervised_contrastive": supcon,
            "batch_hard_triplet": triplet,
            "view_consistency": consistency,
            "branch_auxiliary": branch_aux,
            "quality_auxiliary": quality_aux,
        }


__all__ = [
    "NoseIDObjective",
    "SubCenterArcFace",
    "batch_hard_triplet_loss",
    "supervised_contrastive_loss",
    "view_consistency_loss",
]
