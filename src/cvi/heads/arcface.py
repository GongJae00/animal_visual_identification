"""ArcFace: additive angular margin loss."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int,
                 scale: float = 30.0, margin: float = 0.50):
        super().__init__()
        self._scale = scale
        self._margin = margin
        self._w = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_normal_(self._w)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self._w, p=2, dim=1)
        cos_theta = F.linear(features, w).clamp(-1, 1)
        sin_theta = torch.sqrt((1.0 - cos_theta.pow(2)).clamp(min=1e-12))
        cos_m = math.cos(self._margin)
        sin_m = math.sin(self._margin)
        phi = cos_theta * cos_m - sin_theta * sin_m
        one_hot = F.one_hot(labels, num_classes=self._w.size(0)).float()
        logits = one_hot * phi + (1.0 - one_hot) * cos_theta
        return logits * self._scale


class MagArcFace(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int,
                 scale: float = 64.0, margin: float = 0.45):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_dim, num_classes) * 0.01)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.W, p=2, dim=0)
        x = F.normalize(embeddings, p=2, dim=1)
        cos_theta = torch.mm(x, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        one_hot = F.one_hot(labels, num_classes=self.W.shape[1]).to(embeddings.dtype)
        target_logits = torch.cos(theta + self.margin)
        other_logits = cos_theta
        output = one_hot * target_logits + (1.0 - one_hot) * other_logits
        output *= self.scale
        return output
