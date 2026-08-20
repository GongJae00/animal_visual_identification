from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from representation.channels.extraction import EvidenceExtractionPipeline
from search.scoring.roles import QueryExclusions


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
        self, extraction: EvidenceExtractionPipeline, gallery: Any
    ) -> None:
        self._extraction = extraction
        self._gallery = gallery

    def search(
        self,
        image: Image.Image,
        top_k: int = 10,
        allowed_breeds: list[str] | None = None,
        *,
        exclude_content_match: bool = False,
        query_template_id: str | None = None,
        duplicate_group_ids: frozenset[str] = frozenset(),
    ) -> list[RetrievalResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        vectors, availability = self._extract_available_evidence(image)
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
            availability,
            exclusions,
        )
        if allowed_breeds:
            raw = self._gallery.search_filtered(query, allowed_breeds, top_k)
        else:
            raw = self._gallery.search(query, top_k)

        results: list[RetrievalResult] = []
        for _, score, meta in raw:
            registered_dog_id = meta.get("registered_dog_id")
            if not isinstance(registered_dog_id, str) or not registered_dog_id:
                raise RuntimeError("gallery metadata is missing registered_dog_id")
            results.append(
                RetrievalResult(
                    registered_dog_id=registered_dog_id,
                    similarity=float(score),
                    evidence=dict(meta["_evidence"]),
                    evidence_availability=dict(meta["_evidence_availability"]),
                    scorer_hash=meta["_scorer_hash"],
                    exact=meta["_exact"],
                    metadata={
                        **meta.get("metadata", {}),
                        "template_id": meta["template_id"],
                        "content_sha256": meta["content_sha256"],
                        "idempotency_key": meta["idempotency_key"],
                        "template_schema": meta["template_schema"],
                        "query_evidence_availability": meta[
                            "_query_availability"
                        ],
                        "template_evidence_availability": meta[
                            "_template_availability"
                        ],
                        "identity_evidence_kind": meta[
                            "_identity_evidence_kind"
                        ],
                        "enrollment_rank": meta["_enrollment_rank"],
                        "enrollment_view": meta["_enrollment_view"],
                        "duplicate_group_ids": meta["_duplicate_group_ids"],
                        "winning_template_row": meta["_winning_template_row"],
                    },
                )
            )
        return results

    def explain(self, image: Image.Image, dog_id: str) -> dict[str, Any]:
        vectors, availability = self._extract_available_evidence(image)
        query = self._gallery.prepare_query(vectors, availability)
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

    def _extract_available_evidence(
        self, image: Image.Image
    ) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
        observations = self._extraction.extract_observations(image)
        vectors = {
            name: observation.embedding
            for name, observation in observations.items()
            if observation.is_available and observation.embedding is not None
        }
        availability = {
            name: observation.is_available
            for name, observation in observations.items()
        }
        return vectors, availability


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
