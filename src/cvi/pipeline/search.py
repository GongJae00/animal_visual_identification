from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cvi.evidence.base import AbstractEvidencer
from cvi.evidence.appearance import Dinov2WithUncertainty
from cvi.index.base import AbstractIdentityIndex
from cvi.index.hierarchical import SpeciesFilteredIndex
from cvi.fusion.fuser import LearnedWeightFuser
from cvi.fusion.open_set import EvidentialOpenSet
from cvi.pipeline.enroll import MultiEvidencePipeline


@dataclass
class SearchResult:
    registered_dog_id: str
    similarity: float
    evidence: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


class IdentitySearchPipeline:
    def __init__(self, pipeline: MultiEvidencePipeline,
                 index: AbstractIdentityIndex | SpeciesFilteredIndex,
                 fuser: LearnedWeightFuser | None = None,
                 open_set: EvidentialOpenSet | None = None):
        self._pipeline = pipeline
        self._index = index
        self._fuser = fuser
        self._open_set = open_set

    def enroll(self, image: Image.Image, dog_id: str,
               breed: str | None = None,
               metadata: dict | None = None) -> int:
        embs, quals = self._pipeline.extract_with_quality(image)
        fused = self._fuse_embeddings(embs, quals)
        if isinstance(self._index, SpeciesFilteredIndex) and breed:
            return self._index.enroll_with_breed(fused, dog_id, breed, metadata)
        return self._index.enroll(fused, dog_id, metadata)

    def search(self, image: Image.Image, top_k: int = 10,
               allowed_breeds: list[str] | None = None
               ) -> list[SearchResult]:
        embs, uncertainties = self._pipeline.extract_with_uncertainty(image)
        fused = self._fuse_embeddings(embs)
        if isinstance(self._index, SpeciesFilteredIndex) and allowed_breeds:
            raw = self._index.search_filtered(fused, allowed_breeds, top_k * 2)
        else:
            raw = self._index.search(fused, top_k * 2)

        results: list[SearchResult] = []
        for idx, score, meta in raw:
            vec = self._index._index.reconstruct(int(idx))
            evidence = {}
            offset = 0
            for name, emb in embs.items():
                d = len(emb)
                q_s = fused[offset:offset + d]
                e_s = vec[offset:offset + d]
                offset += d
                q_s = q_s / max(np.linalg.norm(q_s), 1e-8)
                e_s = e_s / max(np.linalg.norm(e_s), 1e-8)
                evidence[name] = float(np.dot(q_s, e_s))
            results.append(SearchResult(
                registered_dog_id=meta.get("registered_dog_id", "unknown"),
                similarity=float(score),
                evidence=evidence,
                metadata=meta.get("metadata", {}),
            ))
            if len(results) >= top_k:
                break

        if self._open_set and results:
            aggr_uncertainty = (
                float(np.mean(list(uncertainties.values())))
                if uncertainties else None
            )
            rej, reason = self._open_set.reject(
                (0, results[0].similarity, {}),
                [(i, r.similarity, {}) for i, r in enumerate(results)],
                epistemic=aggr_uncertainty,
            )
            if rej:
                return []
        return results

    def explain(self, image: Image.Image, dog_id: str) -> dict[str, float]:
        results = self.search(image)
        for r in results:
            if r.registered_dog_id == dog_id:
                evidence = dict(r.evidence)
                evidence["fused"] = r.similarity
                return evidence
        return {}

    def _fuse_embeddings(self, embeddings: dict[str, np.ndarray],
                         qualities: dict[str, float] | None = None
                         ) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self._fuser:
            for name, emb in embeddings.items():
                w = 1.0
                if name in self._fuser.channel_names:
                    idx = self._fuser.channel_names.index(name)
                    w = self._fuser._weights[idx]
                if qualities and name in qualities:
                    w *= qualities[name]
                parts.append(emb * w)
        else:
            for emb in embeddings.values():
                parts.append(emb)
        if not parts:
            return np.zeros(next(iter(embeddings.values())).shape[0], dtype=np.float32) if embeddings else np.zeros(1, dtype=np.float32)
        fused = np.concatenate(parts) if len(parts) > 1 else parts[0]
        norm = np.linalg.norm(fused)
        return fused / norm if norm > 0 else fused
