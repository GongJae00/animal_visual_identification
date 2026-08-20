"""Write extracted channel vectors into a gallery store."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from PIL import Image

from representation.channels.extraction import EvidenceExtractionPipeline


class EnrollmentWriter:
    def __init__(self, extraction: EvidenceExtractionPipeline, gallery: Any) -> None:
        self._extraction = extraction
        self._gallery = gallery

    def enroll(
        self,
        image: Image.Image,
        dog_id: str,
        breed: str | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> int:
        content_sha256 = _image_content_sha256(image)
        embs, availability = self._extract_available_evidence(image)
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
    digest.update(b"search.enrollment_pixels.v1\0")
    mode = image.mode.encode("utf-8")
    digest.update(len(mode).to_bytes(4, "big"))
    digest.update(mode)
    digest.update(image.width.to_bytes(8, "big"))
    digest.update(image.height.to_bytes(8, "big"))
    digest.update(image.tobytes())
    return digest.hexdigest()
