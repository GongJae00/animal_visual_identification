"""DINOv2 appearance embedding with lazy loading.

Backbone is loaded on first .extract() call, not at import time.
This prevents 84 MB auto-download on `import cvi`.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from cvi.evidence.base import AbstractEvidencer


class Dinov2WithUncertainty(AbstractEvidencer):
    name = "appearance"
    output_dim = 384

    def __init__(self, backbone: object | None = None,
                 embedding_dim: int = 384,
                 evidential_checkpoint: str | None = None):
        self._backbone = backbone
        self._embedding_dim = embedding_dim
        self._evidential_checkpoint = evidential_checkpoint
        self._evidential = None
        self._loaded = backbone is not None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import torch
        if self._backbone is None:
            self._backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self._backbone.eval()
        for p in self._backbone.parameters():
            p.requires_grad = False
        if self._evidential_checkpoint:
            raise RuntimeError(
                "Evidential uncertainty is disabled: no strict, calibrated "
                "checkpoint-loading contract is implemented."
            )
        self._loaded = True

    def extract(self, image: Image.Image) -> np.ndarray:
        self._ensure_loaded()
        import torch
        img = np.array(image.resize((224, 224)))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        with torch.no_grad():
            emb = self._backbone(tensor)
        emb_np = emb.squeeze(0).numpy()
        return emb_np / max(np.linalg.norm(emb_np), 1e-8)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        self._ensure_loaded()
        import torch
        imgs = [np.array(img.resize((224, 224))) for img in images]
        batch = np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        with torch.no_grad():
            embs = self._backbone(tensor)
        embs_np = embs.numpy()
        norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
        return embs_np / np.maximum(norms, 1e-8)

    def extract_with_uncertainty(self, image: Image.Image
                                 ) -> tuple[np.ndarray, None, None]:
        return self.extract(image), None, None
