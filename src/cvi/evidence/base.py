from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image


class AbstractEvidencer(ABC):
    name: str = "base"
    output_dim: int = 0

    @abstractmethod
    def extract(self, image: Image.Image) -> np.ndarray:
        ...

    @abstractmethod
    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        ...

    def estimate_quality(self, image: Image.Image) -> float:
        return 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "output_dim": self.output_dim}
