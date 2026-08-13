from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from identity_retrieval.gallery import IdentityGallery
from identity_retrieval.pipeline.extraction import EvidenceExtractionPipeline
from identity_retrieval.qkv import QueryExclusions


@dataclass
class RetrievalResult:
    registered_dog_id: str
    similarity: float
    evidence: dict[str, float]
    evidence_availability: dict[str, bool] = field(default_factory=dict)
    scorer_hash: str = ""
    exact: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class IdentityRetrievalPipeline:
    def __init__(
        self, extraction: EvidenceExtractionPipeline, gallery: IdentityGallery
    ) -> None:
        self._extraction = extraction
        self._gallery = gallery

    def enroll(self, image: Image.Image, dog_id: str,
               breed: str | None = None,
               metadata: dict | None = None,
               idempotency_key: str | None = None) -> int:
        content_sha256 = _image_content_sha256(image)
        observations = self._extraction.extract_observations(image)
        embs = {
            name: observation.embedding
            for name, observation in observations.items()
            if observation.is_available and observation.embedding is not None
        }
        availability = {
            name: observation.is_available for name, observation in observations.items()
        }
        if breed:
            return self._gallery.enroll_with_breed(
                embs,
                dog_id,
                breed,
                metadata,
                idempotency_key,
                content_sha256,
                availability=availability,
            )
        return self._gallery.enroll(
            embs,
            dog_id,
            metadata,
            idempotency_key,
            content_sha256,
            availability=availability,
        )

    def search(self, image: Image.Image, top_k: int = 10,
               allowed_breeds: list[str] | None = None,
               *,
               exclude_content_match: bool = False,
               query_template_id: str | None = None,
               duplicate_group_ids: frozenset[str] = frozenset(),
               ) -> list[RetrievalResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        observations = self._extraction.extract_observations(image)
        vectors = {
            name: observation.embedding
            for name, observation in observations.items()
            if observation.is_available and observation.embedding is not None
        }
        exclusions = QueryExclusions(
            template_ids=(
                frozenset({query_template_id})
                if query_template_id is not None
                else frozenset()
            ),
            content_sha256s=(
                frozenset({_image_content_sha256(image)})
                if exclude_content_match
                else frozenset()
            ),
            duplicate_group_ids=duplicate_group_ids,
        )
        query = self._gallery.prepare_query(
            vectors,
            {
                name: observation.is_available
                for name, observation in observations.items()
            },
            exclusions,
        )
        if allowed_breeds:
            raw = self._gallery.search_filtered(query, allowed_breeds, top_k)
        else:
            raw = self._gallery.search(query, top_k)

        results: list[RetrievalResult] = []
        for idx, score, meta in raw:
            evidence = dict(meta.pop("_evidence"))
            availability = dict(meta.pop("_evidence_availability"))
            scorer_hash = meta.pop("_scorer_hash")
            exact = meta.pop("_exact")
            query_availability = meta.pop("_query_availability")
            template_availability = meta.pop("_template_availability")
            identity_evidence_kind = meta.pop("_identity_evidence_kind")
            enrollment_rank = meta.pop("_enrollment_rank")
            enrollment_view = meta.pop("_enrollment_view")
            enrolled_duplicate_groups = meta.pop("_duplicate_group_ids")
            winning_template_row = meta.pop("_winning_template_row")
            registered_dog_id = meta.get("registered_dog_id")
            if not isinstance(registered_dog_id, str) or not registered_dog_id:
                raise RuntimeError("gallery metadata is missing registered_dog_id")
            results.append(RetrievalResult(
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
                    "identity_evidence_kind": identity_evidence_kind,
                    "enrollment_rank": enrollment_rank,
                    "enrollment_view": enrollment_view,
                    "duplicate_group_ids": enrolled_duplicate_groups,
                    "winning_template_row": winning_template_row,
                },
            ))
        return results

    def explain(self, image: Image.Image, dog_id: str) -> dict[str, Any]:
        observations = self._extraction.extract_observations(image)
        vectors = {
            name: observation.embedding
            for name, observation in observations.items()
            if observation.is_available and observation.embedding is not None
        }
        query = self._gallery.prepare_query(
            vectors,
            {
                name: observation.is_available
                for name, observation in observations.items()
            },
        )
        row = self._gallery.explain_identity(query, dog_id)
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
