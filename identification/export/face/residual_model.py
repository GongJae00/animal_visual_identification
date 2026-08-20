"""FaceID-F5: aligned frozen CLS with a bounded residual adapter."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class FaceCLSResidualAdapter(nn.Module):
    def __init__(self, dimension: int = 384, scale: float = 0.1) -> None:
        super().__init__()
        self.scale = scale
        self.network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, baseline: torch.Tensor) -> torch.Tensor:
        residual = self.network(baseline)
        residual = residual / torch.linalg.vector_norm(
            residual, dim=1, keepdim=True
        ).clamp_min(1.0)
        return F.normalize(baseline + self.scale * residual, dim=1)


class FaceIDResidualModel(nn.Module):
    output_dim = 384

    def __init__(
        self,
        dino_backbone: nn.Module,
        *,
        image_mean: tuple[float, float, float],
        image_std: tuple[float, float, float],
        rescale_factor: float,
    ) -> None:
        super().__init__()
        self.dino = dino_backbone
        for parameter in self.dino.parameters():
            parameter.requires_grad = False
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1))
        self.register_buffer("pixel_scale", torch.tensor(255.0 * rescale_factor))
        self.encoder = FaceCLSResidualAdapter()
        self.quality_head = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, rgb: torch.Tensor, landmarks: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        del landmarks
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 224, 224):
            raise ValueError("F5 Face RGB must be [B,3,224,224]")
        normalized = (
            rgb * self.pixel_scale.to(dtype=rgb.dtype)
            - self.image_mean.to(dtype=rgb.dtype)
        ) / self.image_std.to(dtype=rgb.dtype)
        with torch.no_grad():
            output = self.dino(pixel_values=normalized)
        baseline = getattr(output, "pooler_output", None)
        if not isinstance(baseline, torch.Tensor) or baseline.shape != (
            rgb.shape[0],
            self.output_dim,
        ):
            raise RuntimeError("DINO backbone must return a 384D pooler output")
        baseline = F.normalize(baseline.float(), dim=1)
        embedding = self.encoder(baseline)
        quality = torch.sigmoid(self.quality_head(embedding)).squeeze(1)
        return {
            "embedding": embedding,
            "baseline_embedding": baseline,
            "quality": quality,
        }


__all__: Sequence[str] = ("FaceCLSResidualAdapter", "FaceIDResidualModel")
