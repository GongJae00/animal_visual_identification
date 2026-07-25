from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cvi.evidence.base import AbstractEvidencer
from cvi.evidence.appearance import Dinov2WithUncertainty
from cvi.evidence.nose_print import MiewIDNoseExtractor
from cvi.evidence.landmark_graph import LandmarkEvidencer
from cvi.evidence.quality import overall_quality


class MultiEvidencePipeline:
    def __init__(self, evidencer_map: dict[str, AbstractEvidencer | None]):
        self._evidencer_map = {
            k: v for k, v in evidencer_map.items() if v is not None
        }

    @property
    def active_channels(self) -> list[str]:
        return list(self._evidencer_map.keys())

    def extract_all(self, image: Image.Image
                    ) -> dict[str, np.ndarray]:
        return {
            name: ev.extract(image)
            for name, ev in self._evidencer_map.items()
        }

    def extract_with_quality(self, image: Image.Image
                             ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        embs: dict[str, np.ndarray] = {}
        quals: dict[str, float] = {}
        for name, ev in self._evidencer_map.items():
            embs[name] = ev.extract(image)
            quals[name] = ev.estimate_quality(image)
        return embs, quals

    def extract_with_uncertainty(
        self, image: Image.Image
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        embs: dict[str, np.ndarray] = {}
        uncertainties: dict[str, float] = {}
        for name, ev in self._evidencer_map.items():
            if isinstance(ev, Dinov2WithUncertainty):
                emb, epi, ale = ev.extract_with_uncertainty(image)
                embs[name] = emb
                uncertainties[name] = epi
            elif isinstance(ev, (MiewIDNoseExtractor, LandmarkEvidencer)):
                embs[name] = ev.extract(image)
                uncertainties[name] = 0.05
            else:
                embs[name] = ev.extract(image)
                uncertainties[name] = 0.1
        return embs, uncertainties

    def estimate_quality(self, image: Image.Image) -> dict[str, float]:
        q = overall_quality(image)
        return {name: q for name in self._evidencer_map}
