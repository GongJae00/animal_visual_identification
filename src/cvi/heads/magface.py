"""MagFace: magnitude-aware angular margin variant with norm penalty."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MagArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int,
                 scale: float = 64.0, margin: float = 0.45, h: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_dim, num_classes) * 0.01)
        self._scale = scale
        self._margin = margin
        self._h = h

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = F.normalize(self.W, p=2, dim=0)
        x = F.normalize(embeddings, p=2, dim=1)
        cos_theta = torch.mm(x, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        one_hot = F.one_hot(labels, num_classes=self.W.shape[1]).to(embeddings.dtype)
        target_logits = torch.cos(theta + self._margin)
        other_logits = cos_theta
        output = one_hot * target_logits + (1.0 - one_hot) * other_logits
        output *= self._scale
        norm_batch = torch.norm(embeddings, dim=1)
        mag_loss = torch.mean(torch.exp(-self._h * norm_batch))
        return output, mag_loss
