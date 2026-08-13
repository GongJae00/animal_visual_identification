"""Evidential Deep Learning head for uncertainty estimation.

WARNING: Head is UNTRAINED by default. Random weights produce random
uncertainty values.  Must be trained with an evidential loss function
(negative log marginal likelihood + KL regularization) on labeled pairs
before the epistemic uncertainty output is meaningful.

Without training, any uncertainty-based rejection decision is effectively random.
Do not use untrained heads for identity decisions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EvidentialHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self._fc = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._fc(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        total_evidence = alpha.sum(dim=1, keepdim=True)
        num_classes = alpha.shape[1]
        epistemic = num_classes / total_evidence
        aleatoric = alpha / total_evidence
        return epistemic.squeeze(-1), aleatoric
