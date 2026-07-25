from __future__ import annotations

import torch
import torch.nn as nn

from cvi.train.dataset import PetFaceDataset


class HierarchicalBreedClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_dim: int,
                 num_species: int, num_breeds: int,
                 num_colors: int = 20):
        super().__init__()
        self._backbone = backbone
        self._species_head = nn.Linear(embedding_dim, num_species)
        self._breed_head = nn.Linear(embedding_dim, num_breeds)
        self._color_head = nn.Linear(embedding_dim, num_colors)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self._backbone(x)
        emb_norm = nn.functional.normalize(emb, p=2, dim=1)
        species_logits = self._species_head(emb_norm)
        breed_logits = self._breed_head(emb_norm)
        color_logits = self._color_head(emb_norm)
        return emb_norm, species_logits, breed_logits, color_logits

    def predict_breed(self, x: torch.Tensor, top_k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            emb, _, breed_logits, _ = self.forward(x)
            probs = torch.softmax(breed_logits, dim=1)
            scores, idx = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)
        return idx, scores

    def train_step(self, x: torch.Tensor, species: torch.Tensor,
                   breeds: torch.Tensor, colors: torch.Tensor | None = None,
                   weights: tuple[float, float, float] = (0.1, 0.8, 0.1)
                   ) -> torch.Tensor:
        _, sp_logits, br_logits, co_logits = self.forward(x)
        loss_sp = nn.functional.cross_entropy(sp_logits, species)
        loss_br = nn.functional.cross_entropy(br_logits, breeds)
        loss_co = nn.functional.cross_entropy(co_logits, colors) if colors is not None else 0
        return weights[0] * loss_sp + weights[1] * loss_br + weights[2] * loss_co


def build_breed_index(breeds: list[str]) -> dict[str, int]:
    return {b: i for i, b in enumerate(sorted(set(breeds)))}


def filter_search_space(breed_scores: dict[str, float],
                        threshold: float = 0.05,
                        max_breeds: int = 5) -> list[str]:
    return sorted(breed_scores, key=breed_scores.get, reverse=True)[:max_breeds]
