"""Face ReID model — DINOv2 patch tokens + anatomical regional pooling."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

_FACE_REGION_CENTERS = (
    (0.50, 0.50, 0.4, 0.4),
    (0.35, 0.35, 0.15, 0.12),
    (0.65, 0.35, 0.15, 0.12),
    (0.50, 0.50, 0.18, 0.22),
    (0.50, 0.25, 0.22, 0.14),
)


class FaceRegionalEncoder(nn.Module):
    """DINOv2-S/14 patch tokens + region-aware GeM pooling → 256D."""

    hidden_indices = (8, 9, 10, 11)

    def __init__(self) -> None:
        super().__init__()
        self.layer_mix_logits = nn.Parameter(torch.zeros(4))
        self.hidden_norms = nn.ModuleList([nn.LayerNorm(384) for _ in range(4)])
        self.landmark_projection = nn.Sequential(
            nn.Linear(17 * 3, 128), nn.GELU(), nn.LayerNorm(128)
        )
        self.projection = nn.Sequential(
            nn.Linear(384 * 10 + 128, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
        )

    def _region_masks(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        regions = []
        for cx, cy, sx, sy in _FACE_REGION_CENTERS:
            regions.append(
                torch.exp(-0.5 * (((x - cx) / sx).square() + ((y - cy) / sy).square()))
            )
        return torch.stack(regions, dim=0)

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        landmarks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mixture = torch.softmax(self.layer_mix_logits, dim=0)
        tokens = sum(
            mixture[idx] * norm(hidden_states[layer + 1][:, 1:])
            for idx, (layer, norm) in enumerate(
                zip(self.hidden_indices, self.hidden_norms, strict=True)
            )
        )
        b, n, d = tokens.shape
        features = tokens.transpose(1, 2).reshape(b, d, 16, 16)

        regions = self._region_masks(
            16, 16, device=features.device, dtype=features.dtype
        )
        power = torch.tensor(2.0, device=features.device, dtype=features.dtype)
        pooled: list[torch.Tensor] = []
        for region_idx in range(5):
            weights = regions[region_idx : region_idx + 1].unsqueeze(0)
            w_sum = weights.sum().clamp_min(1e-6)
            channel_attn = torch.softmax(
                (
                    features.mean(dim=1, keepdim=True) + weights.clamp_min(1e-6).log()
                ).flatten(2),
                dim=-1,
            ).view(b, 1, 16, 16)
            attn_pool = torch.sum(features * channel_attn, dim=(2, 3))
            pos = torch.sum(F.relu(features).pow(power) * weights, dim=(2, 3)) / w_sum
            neg = torch.sum(F.relu(-features).pow(power) * weights, dim=(2, 3)) / w_sum
            gem = pos.pow(1.0 / power) - neg.pow(1.0 / power)
            pooled.extend([attn_pool, gem])
        if landmarks is None:
            landmark_features = features.new_zeros((b, 128))
        else:
            if landmarks.shape != (b, 17, 3):
                raise ValueError("face landmarks must be [B,17,3]")
            landmark_features = self.landmark_projection(landmarks.flatten(1))
        return F.normalize(
            self.projection(torch.cat([*pooled, landmark_features], dim=1)), dim=1
        )


class FaceIDModel(nn.Module):
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
        for param in self.dino.parameters():
            param.requires_grad = False
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1))
        self.register_buffer("pixel_scale", torch.tensor(255.0 * rescale_factor))
        self.encoder = FaceRegionalEncoder()
        self.quality_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def _dino_forward(self, rgb: torch.Tensor) -> Sequence[torch.Tensor]:
        with torch.no_grad():
            output = self.dino(pixel_values=rgb, output_hidden_states=True)
        hidden = getattr(output, "hidden_states", None)
        if not isinstance(hidden, (tuple, list)) or len(hidden) < 13:
            raise RuntimeError("DINO backbone must return 13 hidden states")
        return hidden

    def forward(
        self, rgb: torch.Tensor, landmarks: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("Face RGB must be [B,3,224,224]")
        mean = self.image_mean.to(dtype=rgb.dtype)
        std = self.image_std.to(dtype=rgb.dtype)
        ps = self.pixel_scale.to(dtype=rgb.dtype)
        normalized = (rgb * ps - mean) / std
        hidden = self._dino_forward(normalized)
        embedding = self.encoder(hidden, landmarks)
        quality = torch.sigmoid(self.quality_head(embedding))
        return {"embedding": embedding, "quality": quality.squeeze(1)}


__all__ = ["FaceIDModel", "FaceRegionalEncoder"]
