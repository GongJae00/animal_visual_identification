from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MultiHeadConfig:
    model_name: str = "dinov2-small"
    embedding_dim: int = 384
    texture_dim: int = 128
    structural_dim: int = 128
    num_classes: int = 0
    arcface_scale: float = 30.0
    arcface_margin: float = 0.50
    batch_size: int = 128
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lr_min: float = 1e-6
    warmup_epochs: int = 5
    grad_clip_norm: float = 5.0
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50
    save_every_n_epochs: int = 5
    mixed_precision: bool = True
    num_workers: int = 0
    gradient_checkpointing: bool = False
    compile_model: bool = False
    preload_images: bool = True
    fusion_weight_v: float = 1.0
    fusion_weight_t: float = 0.5
    fusion_weight_s: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultiHeadConfig:
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Shared backbone
# ---------------------------------------------------------------------------


class MultiHeadBackbone(nn.Module):
    def __init__(self, use_gradient_checkpointing: bool = False) -> None:
        super().__init__()
        from transformers import AutoModel
        self._model = AutoModel.from_pretrained(
            "facebook/dinov2-small", attn_implementation="sdpa"
        )
        if use_gradient_checkpointing:
            self._model.gradient_checkpointing_enable()
        self._hidden_dim = 384
        self._num_patches = 256

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self._model(x, output_hidden_states=True)
        last_hidden = out.last_hidden_state
        hs = out.hidden_states
        cls_token = last_hidden[:, 0, :]
        patch_tokens = last_hidden[:, 1:, :]
        mid_features = hs[6][:, 1:, :]
        return cls_token, patch_tokens, mid_features


# ---------------------------------------------------------------------------
# Visual head
# ---------------------------------------------------------------------------


class VisualHead(nn.Module):
    def __init__(self, hidden_dim: int = 384, output_dim: int = 384) -> None:
        super().__init__()
        self._proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        emb = self._proj(cls_token)
        return F.normalize(emb, p=2, dim=1)


# ---------------------------------------------------------------------------
# Texture head
# ---------------------------------------------------------------------------


class TextureHead(nn.Module):
    def __init__(self, patch_dim: int = 384, output_dim: int = 128,
                 num_patches: int = 256) -> None:
        super().__init__()
        self._attn = nn.Linear(patch_dim, 1)
        self._proj = nn.Linear(patch_dim, output_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        attn_logits = self._attn(patch_tokens).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)
        agg = (patch_tokens * attn_weights).sum(dim=1)
        emb = self._proj(agg)
        return F.normalize(emb, p=2, dim=1)


# ---------------------------------------------------------------------------
# Structural head (implicit keypoints)
# ---------------------------------------------------------------------------


class StructuralHead(nn.Module):
    def __init__(self, patch_dim: int = 384, grid_size: int = 16,
                 num_keypoints: int = 8, output_dim: int = 128) -> None:
        super().__init__()
        self._num_keypoints = num_keypoints
        self._grid_size = grid_size
        self._heatmap_conv = nn.Sequential(
            nn.Conv2d(patch_dim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, num_keypoints, kernel_size=1),
        )
        n_dists = num_keypoints * (num_keypoints - 1) // 2
        n_angles = n_dists
        geom_dim = n_dists + n_angles
        self._mlp = nn.Sequential(
            nn.Linear(geom_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, mid_features: torch.Tensor) -> torch.Tensor:
        b, n, d = mid_features.shape
        gs = self._grid_size
        feat = mid_features.transpose(1, 2).view(b, d, gs, gs)
        heatmaps = self._heatmap_conv(feat)
        coords = self._soft_argmax(heatmaps)
        geom_desc = self._pairwise_geometry(coords)
        emb = self._mlp(geom_desc)
        return F.normalize(emb, p=2, dim=1)

    def _soft_argmax(self, heatmaps: torch.Tensor) -> torch.Tensor:
        b, k, h, w = heatmaps.shape
        heat_flat = heatmaps.view(b, k, -1)
        attn = F.softmax(heat_flat * 10.0, dim=-1)
        ys = torch.linspace(-1, 1, h, device=heatmaps.device).view(1, 1, h, 1)
        xs = torch.linspace(-1, 1, w, device=heatmaps.device).view(1, 1, 1, w)
        y_hat = (attn.view(b, k, h, w) * ys).sum(dim=(2, 3))
        x_hat = (attn.view(b, k, h, w) * xs).sum(dim=(2, 3))
        return torch.stack([x_hat, y_hat], dim=-1)

    def _pairwise_geometry(self, coords: torch.Tensor) -> torch.Tensor:
        b, k, _ = coords.shape
        i, j = torch.triu_indices(k, k, offset=1, device=coords.device)
        pi = coords[:, i, :]
        pj = coords[:, j, :]
        vecs = pj - pi
        dists = torch.norm(vecs, dim=-1)
        angles = torch.atan2(vecs[..., 1], vecs[..., 0])
        return torch.cat([dists, angles], dim=-1)


# ---------------------------------------------------------------------------
# Multi-head model
# ---------------------------------------------------------------------------


class MultiHeadModel(nn.Module):
    def __init__(self, config: MultiHeadConfig) -> None:
        super().__init__()
        self._config = config
        self._backbone = MultiHeadBackbone(config.gradient_checkpointing)
        self._visual = VisualHead(384, config.embedding_dim)
        self._texture = TextureHead(384, config.texture_dim)
        self._structural = StructuralHead(384, output_dim=config.structural_dim)

        if config.num_classes > 0:
            self._arcface_v = _ArcFaceHead(config.embedding_dim, config.num_classes,
                                           config.arcface_scale, config.arcface_margin)
            self._arcface_t = _ArcFaceHead(config.texture_dim, config.num_classes,
                                           config.arcface_scale, config.arcface_margin)
            self._arcface_s = _ArcFaceHead(config.structural_dim, config.num_classes,
                                           config.arcface_scale, config.arcface_margin)
        else:
            self._arcface_v = None
            self._arcface_t = None
            self._arcface_s = None

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        cls_t, patch_t, mid_f = self._backbone(images)
        e_v = self._visual(cls_t)
        e_t = self._texture(patch_t)
        e_s = self._structural(mid_f)
        if labels is not None and self.training and self._arcface_v is not None:
            logits_v = self._arcface_v(e_v, labels)
            logits_t = self._arcface_t(e_t, labels)
            logits_s = self._arcface_s(e_s, labels)
            return logits_v, logits_t, logits_s
        full = torch.cat([e_v, e_t, e_s], dim=-1)
        return full

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            cls_t, patch_t, mid_f = self._backbone(images)
            e_v = self._visual(cls_t)
            e_t = self._texture(patch_t)
            e_s = self._structural(mid_f)
            return torch.cat([e_v, e_t, e_s], dim=-1)

    def extract_split(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            cls_t, patch_t, mid_f = self._backbone(images)
            return {
                "visual": self._visual(cls_t),
                "texture": self._texture(patch_t),
                "structural": self._structural(mid_f),
            }

    def export_to_onnx(self, output_path: str, device: torch.device = torch.device("cpu")) -> None:
        self.eval().to(device)
        dummy = torch.randn(1, 3, 224, 224, device=device)
        torch.onnx.export(
            self,
            dummy,
            output_path,
            input_names=["images"],
            output_names=["embedding"],
            dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=18,
        )


class _ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int,
                 scale: float, margin: float) -> None:
        super().__init__()
        self._scale = scale
        self._margin = margin
        self._w = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_normal_(self._w)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self._w, p=2, dim=1)
        cos_theta = F.linear(features, w)
        sin_theta = torch.sqrt(1.0 - cos_theta.clamp(-1, 1).pow(2))
        cos_m = math.cos(self._margin)
        sin_m = math.sin(self._margin)
        phi = cos_theta * cos_m - sin_theta * sin_m
        one_hot = F.one_hot(labels, num_classes=self._w.size(0)).float()
        logits = one_hot * phi + (1.0 - one_hot) * cos_theta
        return logits * self._scale
