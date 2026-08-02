"""Full model: backbone + classification head."""

from __future__ import annotations

import torch
import torch.nn as nn

from representation_learning.heads.arcface import MagArcFace


class PetFaceArcFace(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_dim: int,
                 num_classes: int, scale: float = 64.0):
        super().__init__()
        self._backbone = backbone
        self._head = MagArcFace(embedding_dim, num_classes, scale=scale)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        emb = self._backbone(x)
        if labels is None:
            return emb
        return self._head(emb, labels)
