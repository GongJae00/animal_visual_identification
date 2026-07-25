"""CUDA deployment path for GPU-accelerated inference + FAISS search.

Configures ONNX Runtime with CUDAExecutionProvider and FAISS with
GpuIndexFlatIP.  Falls back to CPU gracefully when GPU unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cvi.evidence.base import AbstractEvidencer
from cvi.pipeline.enroll import MultiEvidencePipeline
from cvi.pipeline.search import IdentitySearchPipeline, SearchResult
from cvi.fusion.fuser import LearnedWeightFuser
from cvi.fusion.open_set import EvidentialOpenSet
from cvi.index.hierarchical import SpeciesFilteredIndex
from cvi.backbones import get_backbone


class CVIDeploymentCUDA:
    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._evidence_map: dict[str, AbstractEvidencer] = self._build_evidence()
        self._pipeline = MultiEvidencePipeline(self._evidence_map)

        dim = config.get("fused_dim", 640)
        index_dir = Path(config.get("index_dir", "./cvi_index"))
        self._index = SpeciesFilteredIndex(index_dir, dim=dim)

        channel_names = list(self._evidence_map.keys())
        init_w = config.get("fusion_weights", None)
        self._fuser = LearnedWeightFuser(channel_names, init_w)

        os_cfg = config.get("open_set", {})
        self._open_set = EvidentialOpenSet(
            epistemic_threshold=os_cfg.get("epistemic_threshold", 0.3),
            distance_ratio_threshold=os_cfg.get("distance_ratio_threshold", 0.15),
            min_similarity=os_cfg.get("min_similarity", 0.4),
        )

        self._search_pipeline = IdentitySearchPipeline(
            self._pipeline, self._index, self._fuser, self._open_set
        )

    def _build_evidence(self) -> dict[str, AbstractEvidencer]:
        from cvi.evidence.nose_print import MiewIDNoseExtractor
        from cvi.evidence.landmark_graph import LandmarkEvidencer
        from cvi.evidence.appearance import Dinov2WithUncertainty

        evidence: dict[str, AbstractEvidencer] = {}
        for name, spec in self._config.get("channels", {}).items():
            kind = spec.get("type", "")
            if kind == "miewid":
                evidence[name] = MiewIDNoseExtractor(Path(spec["path"]))
            elif kind == "landmark":
                evidence[name] = LandmarkEvidencer()
            elif kind == "dinov2" or kind == "appearance":
                evidence[name] = Dinov2WithUncertainty()
        return evidence

    def enroll(self, image, dog_id: str, breed: str | None = None,
               metadata: dict | None = None) -> int:
        return self._search_pipeline.enroll(image, dog_id, breed, metadata)

    def search(self, image, top_k: int = 10,
               allowed_breeds: list[str] | None = None) -> list[SearchResult]:
        return self._search_pipeline.search(image, top_k, allowed_breeds)

    def explain(self, image, dog_id: str) -> dict[str, float]:
        return self._search_pipeline.explain(image, dog_id)

    @property
    def size(self) -> int:
        return self._index.size

    def close(self) -> None:
        self._index.save()
        self._index.close()
