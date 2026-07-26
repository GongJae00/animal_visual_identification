"""Backbone factory: pluggable feature extractors.

Each backbone outputs L2-normalized embeddings that directly enter the
ArcFace/MagFace head during training, or the FAISS index at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dinov2Backbone(nn.Module):
    """DINOv2-small backbone.  384-d L2-normalized output."""

    def __init__(self, embedding_dim: int = 384,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        self._backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        if use_gradient_checkpointing:
            self._backbone.set_grad_checkpointing(True)
        self._project = nn.Linear(384, embedding_dim, bias=False) if embedding_dim != 384 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self._backbone(x)
        emb = self._project(emb)
        return F.normalize(emb, p=2, dim=1)


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt-Base backbone.  768-d output, project to embedding_dim."""

    def __init__(self, embedding_dim: int = 768,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        from transformers import ConvNextModel
        self._backbone = ConvNextModel.from_pretrained("facebook/convnext-base-224")
        if use_gradient_checkpointing:
            self._backbone.gradient_checkpointing_enable()
        self._project = nn.Linear(1024, embedding_dim, bias=False) if embedding_dim != 1024 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._backbone(x)
        emb = out.last_hidden_state.mean(dim=[-1, -2])
        emb = self._project(emb)
        return F.normalize(emb, p=2, dim=1)


class TinyViTBackbone(nn.Module):
    """Disabled alias for a former untrained CNN incorrectly named TinyViT."""

    def __init__(self, embedding_dim: int = 512,
                 use_gradient_checkpointing: bool = False):
        raise RuntimeError(
            "tinyvit is disabled: the implementation was an untrained "
            "three-layer CNN, not a TinyViT architecture."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("tinyvit is disabled")


_BACKBONE_REGISTRY: dict[str, type[nn.Module]] = {
    "dinov2-small": Dinov2Backbone,
    "dinov2-base": Dinov2Backbone,
    "convnext-base": ConvNeXtBackbone,
}


def get_backbone(name: str, embedding_dim: int = 384,
                 use_gradient_checkpointing: bool = False) -> nn.Module:
    if name not in _BACKBONE_REGISTRY:
        raise KeyError(f"Unknown backbone: {name}. Available: {list(_BACKBONE_REGISTRY)}")
    return _BACKBONE_REGISTRY[name](
        embedding_dim=embedding_dim,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
