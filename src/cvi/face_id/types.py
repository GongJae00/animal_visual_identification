"""Face ReID types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AlignedFace:
    rgb: np.ndarray
    landmarks_xyc: np.ndarray | None

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.float32)
        if rgb.shape != (3, 224, 224) or not np.isfinite(rgb).all():
            raise ValueError("aligned face RGB must be finite [3,224,224]")
        object.__setattr__(self, "rgb", rgb)


@dataclass(frozen=True, slots=True)
class FaceIDOutput:
    embedding: np.ndarray
    quality: float

    def __post_init__(self) -> None:
        emb = np.asarray(self.embedding, dtype=np.float32)
        if emb.shape != (256,) or not np.isfinite(emb).all():
            raise ValueError("face embedding must be finite float32 [256]")
        object.__setattr__(self, "embedding", emb)


__all__ = ["AlignedFace", "FaceIDOutput"]
