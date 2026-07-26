"""Public API: single entry point for identity enrollment and search.

Users never touch internal modules (index, fusion, evidence) directly.
All configuration lives in a single JSON/ dict.

Usage:
    from cvi import CVI

    cvi = CVI(config="configs/deployment/production.json")

    # 등록
    cvi.enroll(image, dog_id="뽀삐", breed="비글")

    # 검색
    results = cvi.search(query_image, top_k=5)

    # 저장
    cvi.save()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class Match:
    dog_id: str
    similarity: float
    evidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dog_id": self.dog_id,
            "similarity": round(self.similarity, 4),
            "evidence": {k: round(v, 4) for k, v in self.evidence.items()},
        }


class CVI:
    def __init__(self, config: dict[str, Any] | str | Path | None = None):
        if config is None:
            config = {}
        elif isinstance(config, (str, Path)):
            raw = str(config) if isinstance(config, Path) else config
            if isinstance(config, str) and raw.strip().startswith("{"):
                config = json.loads(raw)
            else:
                config = json.loads(Path(raw).read_text())
        self._config = config
        self._pipeline = None
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        from cvi.pipeline.enroll import MultiEvidencePipeline
        from cvi.pipeline.search import IdentitySearchPipeline
        from cvi.index.hierarchical import SpeciesFilteredIndex
        from cvi.fusion.fuser import LearnedWeightFuser
        from cvi.fusion.open_set import EvidentialOpenSet
        from cvi.evidence.base import AbstractEvidencer

        evidence = self._build_evidence()
        self._pipeline = MultiEvidencePipeline(evidence)

        index_dir = Path(self._config.get("index_dir", "./cvi_index"))
        dim = self._config.get("fused_dim", self._compute_fused_dim(evidence))
        self._index = SpeciesFilteredIndex(index_dir, dim=dim)

        channels = list(evidence.keys())
        weights = self._config.get("fusion_weights", None)
        self._fuser = LearnedWeightFuser(channels, weights)

        os_cfg = self._config.get("open_set", {})
        self._open_set = EvidentialOpenSet(
            epistemic_threshold=os_cfg.get("epistemic_threshold", 0.3),
            distance_ratio_threshold=os_cfg.get("distance_ratio_threshold", 0.15),
            min_similarity=os_cfg.get("min_similarity", 0.4),
        )

        self._search = IdentitySearchPipeline(
            self._pipeline, self._index, self._fuser, self._open_set
        )

    def _build_evidence(self) -> dict[str, Any]:
        from cvi.evidence.miewid import MiewIDReIDExtractor
        from cvi.evidence.appearance import Dinov2WithUncertainty

        evidence: dict[str, Any] = {}
        for name, spec in self._config.get("channels", {}).items():
            kind = spec.get("type", "")
            if kind in ("miewid", "miewid_reid", "wildlife_reid"):
                evidence[name] = MiewIDReIDExtractor(Path(spec["path"]))
            elif kind == "landmark":
                raise ValueError(
                    "landmark channel is disabled until trained heatmap and "
                    "graph artifacts have a verified loading contract"
                )
            elif kind in ("dinov2", "appearance"):
                evidence[name] = Dinov2WithUncertainty()
        if not evidence:
            evidence["appearance"] = Dinov2WithUncertainty()
        return evidence

    @staticmethod
    def _compute_fused_dim(evidence: dict) -> int:
        return sum(getattr(ev, "output_dim", 384) for ev in evidence.values())

    def enroll(self, image: Image.Image, dog_id: str,
               breed: str | None = None,
               metadata: dict | None = None) -> int:
        return self._search.enroll(image, dog_id, breed, metadata)

    def search(self, image: Image.Image, top_k: int = 5,
               breed_filter: list[str] | None = None) -> list[Match]:
        raw = self._search.search(image, top_k, breed_filter)
        return [Match(r.registered_dog_id, r.similarity, r.evidence) for r in raw]

    def explain(self, image: Image.Image, dog_id: str) -> dict[str, float]:
        return self._search.explain(image, dog_id)

    @property
    def size(self) -> int:
        return self._index.size

    def save(self) -> None:
        self._index.save()

    def close(self) -> None:
        self.save()
