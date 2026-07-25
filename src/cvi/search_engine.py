from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cvi.evidence_extractor import EvidenceExtractor, EvidenceExtractorRegistry, OnnxExtractor
from cvi.identity_index import (
    EMBEDDING_DIM,
    TEXTURE_SLICE,
    STRUCTURAL_SLICE,
    IdentityIndex,
    SearchResult,
)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class FusePlan:
    visual_weight: float = 1.0
    texture_weight: float = 0.5
    structural_weight: float = 0.5
    nose_weight: float = 0.3

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.visual_weight, self.texture_weight, self.structural_weight)


@dataclass
class MultiEvidenceEmbedding:
    visual: np.ndarray | None = None
    texture: np.ndarray | None = None
    structural: np.ndarray | None = None
    nose: np.ndarray | None = None

    def fused(self, plan: FusePlan | None = None) -> np.ndarray:
        p = plan or FusePlan()
        parts: list[np.ndarray] = []
        w_total = 0.0
        for emb, w in [(self.visual, p.visual_weight),
                        (self.texture, p.texture_weight),
                        (self.structural, p.structural_weight)]:
            if emb is not None and w > 0:
                parts.append(emb * w)
                w_total += w
        if not parts:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        fused = np.concatenate(parts) if len(parts) > 1 else parts[0]
        norm = np.linalg.norm(fused)
        return fused / norm if norm > 0 else fused

    def nose_separate(self) -> np.ndarray | None:
        return self.nose


# ---------------------------------------------------------------------------
# Legacy FeatureExtractor (backward compatible)
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Legacy single-ONNX feature extractor (wraps OnnxExtractor).

    Input:  224×224 RGB PIL Image
    Output: 640-d L2-normalized embedding [visual 384 | texture 128 | structural 128]

    Deprecated: use EvidenceExtractorRegistry + SearchEngine directly.
    """

    def __init__(self, model_path: Path) -> None:
        self._ext = OnnxExtractor(model_path, input_size=224)

    @property
    def output_dim(self) -> int:
        return self._ext.output_dim

    def preprocess(self, image: Image.Image) -> np.ndarray:
        return self._ext.preprocess(image)

    def extract(self, image: Image.Image) -> np.ndarray:
        return self._ext.extract(image)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self._ext.extract_batch(images)


# ---------------------------------------------------------------------------
# SearchEngine
# ---------------------------------------------------------------------------


class SearchEngine:
    """Unified identity search engine with pluggable evidence extractors.

    Accepts either:
      - A legacy ``model_path`` (single ONNX, backward compat), or
      - An ``EvidenceExtractorRegistry`` with per-channel extractors.
    """

    def __init__(self, model_path: Path | None = None,
                 index_dir: Path | None = None,
                 registry: EvidenceExtractorRegistry | None = None,
                 fuse_plan: FusePlan | None = None) -> None:
        if registry is not None:
            self._registry = registry
        elif model_path is not None:
            ext = OnnxExtractor(model_path, input_size=224)
            self._registry = EvidenceExtractorRegistry()
            self._registry.register("onnx", ext)
        else:
            raise ValueError("provide either model_path or registry")
        if index_dir is None:
            index_dir = Path.cwd()

        # compute fused dimension dynamically
        visual, texture, structural = (self._registry.visual,
                                       self._registry.texture,
                                       self._registry.structural)
        if visual or texture or structural:
            fused_dim = (visual.output_dim if visual else 0) \
                      + (texture.output_dim if texture else 0) \
                      + (structural.output_dim if structural else 0)
        else:
            fused_dim = self._registry.get("onnx").output_dim
        self._index = IdentityIndex(
            index_path=index_dir / "identities.idx",
            metadata_path=index_dir / "identities.json",
            dim=fused_dim,
        )
        self._index_dir = index_dir
        self._fuse_plan = fuse_plan or FusePlan()

    @property
    def size(self) -> int:
        return self._index.size

    @property
    def fused_dim(self) -> int:
        visual, texture, structural = (self._registry.visual,
                                       self._registry.texture,
                                       self._registry.structural)
        if visual or texture or structural:
            return (visual.output_dim if visual else 0) \
                 + (texture.output_dim if texture else 0) \
                 + (structural.output_dim if structural else 0)
        return self._registry.get("onnx").output_dim

    def _evidence_slices(self) -> list[tuple[int, int, str]]:
        from cvi.identity_index import make_evidence_slices
        visual, texture, structural = (self._registry.visual,
                                       self._registry.texture,
                                       self._registry.structural)
        if visual or texture or structural:
            return make_evidence_slices(
                visual_dim=visual.output_dim if visual else 0,
                texture_dim=texture.output_dim if texture else 0,
                structural_dim=structural.output_dim if structural else 0,
            )
        return make_evidence_slices()

    def _extract_multi(self, image: Image.Image) -> MultiEvidenceEmbedding:
        visual = self._registry.visual
        texture = self._registry.texture
        structural = self._registry.structural
        nose = self._registry.nose
        if visual or texture or structural:
            return MultiEvidenceEmbedding(
                visual=visual.extract(image) if visual else None,
                texture=texture.extract(image) if texture else None,
                structural=structural.extract(image) if structural else None,
                nose=nose.extract(image) if nose else None,
            )
        onnx_ext = self._registry.get("onnx")
        emb = onnx_ext.extract(image)
        return MultiEvidenceEmbedding(
            visual=emb[:TEXTURE_SLICE[0]],
            texture=emb[TEXTURE_SLICE[0]:TEXTURE_SLICE[1]],
            structural=emb[STRUCTURAL_SLICE[0]:STRUCTURAL_SLICE[1]],
        )

    def enroll(self, image: Image.Image, registered_dog_id: str,
               dataset_name: str | None = None,
               metadata: dict | None = None) -> int:
        m = self._extract_multi(image)
        fused = m.fused(self._fuse_plan)
        return self._index.enroll(fused, registered_dog_id, dataset_name, metadata)

    def enroll_batch(self, images: list[Image.Image],
                     registered_dog_ids: list[str],
                     dataset_names: list[str | None] | None = None,
                     metadata_list: list[dict] | None = None) -> list[int]:
        fused_list = [self._extract_multi(img).fused(self._fuse_plan) for img in images]
        embs = np.stack(fused_list)
        return self._index.enroll_batch(embs, registered_dog_ids, dataset_names, metadata_list)

    def search(self, image: Image.Image, top_k: int = 5,
               fusion_weights: tuple[float, float, float] | None = None
               ) -> QueryResponse:
        t0 = time.perf_counter()
        m = self._extract_multi(image)
        fused = m.fused(self._fuse_plan)
        fw = fusion_weights or (self._fuse_plan.visual_weight,
                                self._fuse_plan.texture_weight,
                                self._fuse_plan.structural_weight)
        results = self._index.search(fused, top_k, fw)
        elapsed = (time.perf_counter() - t0) * 1000
        return QueryResponse(
            matches=tuple(results),
            elapsed_ms=elapsed,
        )

    def search_batch(self, images: list[Image.Image], top_k: int = 5,
                     fusion_weights: tuple[float, float, float] | None = None
                     ) -> list[QueryResponse]:
        t0 = time.perf_counter()
        fw = fusion_weights or (self._fuse_plan.visual_weight,
                                self._fuse_plan.texture_weight,
                                self._fuse_plan.structural_weight)
        responses: list[QueryResponse] = []
        for img in images:
            m = self._extract_multi(img)
            fused = m.fused(self._fuse_plan)
            results = self._index.search(fused, top_k, fw)
            responses.append(QueryResponse(matches=tuple(results), elapsed_ms=0.0))
        elapsed = (time.perf_counter() - t0) * 1000
        for r in responses:
            object.__setattr__(r, "elapsed_ms", elapsed / len(images))
        return responses

    def explain(self, image: Image.Image, registered_dog_id: str
                ) -> dict[str, float]:
        m = self._extract_multi(image)
        fused = m.fused(self._fuse_plan)
        results = self._index.search(fused, top_k=self._index.size)
        for r in results:
            if r.registered_dog_id == registered_dog_id:
                return {
                    "visual": r.evidence.visual,
                    "texture": r.evidence.texture,
                    "structural": r.evidence.structural,
                    "fused": r.similarity,
                }
        return {"visual": 0.0, "texture": 0.0, "structural": 0.0, "fused": 0.0}

    def close(self) -> None:
        self._registry.close()
        self._index.close()


class QueryResponse:
    def __init__(self, matches: tuple[SearchResult, ...], elapsed_ms: float) -> None:
        self.matches = matches
        self.elapsed_ms = elapsed_ms

    def to_dict(self, include_evidence: bool = True) -> dict[str, Any]:
        return {
            "matches": [
                {"registered_dog_id": m.registered_dog_id,
                 "similarity": round(m.similarity, 6),
                 "evidence": m.evidence.to_dict() if include_evidence else None}
                for m in self.matches
            ],
            "elapsed_ms": round(self.elapsed_ms, 2),
        }
