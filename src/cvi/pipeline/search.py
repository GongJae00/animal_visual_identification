from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

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
    evidence_availability: dict[str, bool] = field(default_factory=dict)
    scorer_hash: str = ""
    exact: bool = False
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
               metadata: dict | None = None,
               idempotency_key: str | None = None) -> int:
        content_sha256 = _image_content_sha256(image)
        observations = self._pipeline.extract_observations(image)
        embs = {
            name: observation.embedding
            for name, observation in observations.items()
            if observation.is_available and observation.embedding is not None
        }
        if isinstance(self._index, SpeciesFilteredIndex) and breed:
            return self._index.enroll_with_breed(
                embs, dog_id, breed, metadata, idempotency_key, content_sha256
            )
        if isinstance(self._index, SpeciesFilteredIndex):
            return self._index.enroll(
                embs, dog_id, metadata, idempotency_key, content_sha256
            )
        if idempotency_key is None:
            return self._index.enroll(embs, dog_id, metadata)
        return self._index.enroll(embs, dog_id, metadata, idempotency_key)

    def search(self, image: Image.Image, top_k: int = 10,
               allowed_breeds: list[str] | None = None
               ) -> list[SearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if hasattr(self._pipeline, "extract_observations_with_uncertainty"):
            observations, uncertainties = (
                self._pipeline.extract_observations_with_uncertainty(image)
            )
            embs = {
                name: observation.embedding
                for name, observation in observations.items()
                if observation.is_available and observation.embedding is not None
            }
        else:
            embs, uncertainties = self._pipeline.extract_with_uncertainty(image)
        decision_k = max(top_k, 2) if self._open_set is not None else top_k
        if isinstance(self._index, SpeciesFilteredIndex) and allowed_breeds:
            raw = self._index.search_filtered(embs, allowed_breeds, decision_k)
        else:
            raw = self._index.search(embs, decision_k)

        results: list[SearchResult] = []
        for idx, score, meta in raw:
            evidence = dict(meta.pop("_evidence"))
            availability = dict(meta.pop("_evidence_availability"))
            scorer_hash = meta.pop("_scorer_hash")
            exact = meta.pop("_exact")
            query_availability = meta.pop("_query_availability")
            template_availability = meta.pop("_template_availability")
            registered_dog_id = meta.get("registered_dog_id")
            if not isinstance(registered_dog_id, str) or not registered_dog_id:
                raise RuntimeError("gallery metadata is missing registered_dog_id")
            results.append(SearchResult(
                registered_dog_id=registered_dog_id,
                similarity=float(score),
                evidence=evidence,
                evidence_availability=availability,
                scorer_hash=scorer_hash,
                exact=exact,
                metadata={
                    **meta.get("metadata", {}),
                    "template_id": meta["template_id"],
                    "content_sha256": meta["content_sha256"],
                    "idempotency_key": meta["idempotency_key"],
                    "template_schema": meta["template_schema"],
                    "query_evidence_availability": query_availability,
                    "template_evidence_availability": template_availability,
                },
            ))
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
        return results[:top_k]

    def explain(self, image: Image.Image, dog_id: str) -> dict[str, Any]:
        if not isinstance(self._index, SpeciesFilteredIndex):
            raise RuntimeError("exact explain requires a v4 gallery index")
        observations, _ = self._pipeline.extract_observations_with_uncertainty(image)
        embs = {
            name: observation.embedding
            for name, observation in observations.items()
            if observation.is_available and observation.embedding is not None
        }
        row = self._index.explain_identity(embs, dog_id)
        if row is None:
            return {}
        _, score, meta = row
        return {
            "registered_dog_id": dog_id,
            "similarity": float(score),
            "evidence": dict(meta["_evidence"]),
            "evidence_availability": dict(meta["_evidence_availability"]),
            "query_evidence_availability": dict(meta["_query_availability"]),
            "template_evidence_availability": dict(
                meta["_template_availability"]
            ),
            "scorer_hash": meta["_scorer_hash"],
            "exact": meta["_exact"],
            "template_id": meta["template_id"],
        }

    def _fuse_embeddings(self, embeddings: dict[str, np.ndarray]) -> np.ndarray:
        if not embeddings:
            raise ValueError("at least one embedding is required")
        parts: list[np.ndarray] = []
        if self._fuser:
            if set(embeddings) != set(self._fuser.channel_names):
                raise ValueError("embedding channels do not match fusion channels")
            for name, scale in zip(
                self._fuser.channel_names, self._fuser.embedding_scales
            ):
                emb = np.asarray(embeddings[name], dtype=np.float32)
                if emb.ndim != 1 or not np.all(np.isfinite(emb)):
                    raise ValueError(f"channel {name!r} must be a finite vector")
                norm = float(np.linalg.norm(emb))
                if not np.isfinite(norm) or norm <= 1e-8:
                    raise ValueError(f"channel {name!r} has a zero embedding")
                parts.append((emb / norm) * scale)
        else:
            for name, value in embeddings.items():
                emb = np.asarray(value, dtype=np.float32)
                if emb.ndim != 1 or not np.all(np.isfinite(emb)):
                    raise ValueError(f"channel {name!r} must be a finite vector")
                norm = float(np.linalg.norm(emb))
                if not np.isfinite(norm) or norm <= 1e-8:
                    raise ValueError(f"channel {name!r} has a zero embedding")
                parts.append(emb / norm)
        fused = np.concatenate(parts) if len(parts) > 1 else parts[0]
        norm = float(np.linalg.norm(fused))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("fused embedding has zero or non-finite norm")
        return (fused / norm).astype(np.float32, copy=False)


def _image_content_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(b"cvi.enrollment_pixels.v1\0")
    mode = image.mode.encode("utf-8")
    digest.update(len(mode).to_bytes(4, "big"))
    digest.update(mode)
    digest.update(image.width.to_bytes(8, "big"))
    digest.update(image.height.to_bytes(8, "big"))
    digest.update(image.tobytes())
    return digest.hexdigest()
