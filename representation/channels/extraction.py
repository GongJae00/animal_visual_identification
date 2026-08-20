from __future__ import annotations

import numpy as np
from PIL import Image

from representation.evidence.base import (
    AbstractEvidencer,
    EvidenceObservation,
    RequiredEvidenceUnavailableError,
)
from representation.quality.quality import QualityObservation


class EvidenceExtractionPipeline:
    def __init__(
        self,
        evidencer_map: dict[str, AbstractEvidencer | None],
        optional_channels: set[str] | frozenset[str] | None = None,
    ) -> None:
        self._evidencer_map = {
            name: evidencer
            for name, evidencer in evidencer_map.items()
            if evidencer is not None
        }
        self._optional_channels = frozenset(optional_channels or ())
        if not self._optional_channels <= set(self._evidencer_map):
            raise ValueError("optional channels must be active evidence channels")
        if not set(self._evidencer_map) - self._optional_channels:
            raise ValueError("at least one evidence channel must be required")

    def extract_observations(
        self, image: Image.Image
    ) -> dict[str, EvidenceObservation]:
        return {
            name: self._extract_observation(name, evidencer, image)
            for name, evidencer in self._evidencer_map.items()
        }

    def extract_with_quality(
        self, image: Image.Image
    ) -> tuple[dict[str, np.ndarray], dict[str, QualityObservation]]:
        embs: dict[str, np.ndarray] = {}
        quals: dict[str, QualityObservation] = {}
        for name, observation in self.extract_observations(image).items():
            if not observation.is_available:
                continue
            embs[name] = observation.embedding
            quals[name] = self._evidencer_map[name].estimate_quality(
                image, channel=name
            )
        return embs, quals

    def estimate_quality(
        self,
        image: Image.Image,
    ) -> dict[str, QualityObservation]:
        return {
            name: ev.estimate_quality(image, channel=name)
            for name, ev in self._evidencer_map.items()
        }

    def _extract_observation(
        self,
        name: str,
        evidencer: AbstractEvidencer,
        image: Image.Image,
    ) -> EvidenceObservation:
        observation = _as_observation(name, evidencer.extract(image))
        if not observation.is_available and name not in self._optional_channels:
            raise RequiredEvidenceUnavailableError(
                f"required evidence channel {name!r} is unavailable: "
                f"{observation.reason.value}"
            )
        return observation


def _as_observation(
    channel: str, value: np.ndarray | EvidenceObservation
) -> EvidenceObservation:
    if isinstance(value, EvidenceObservation):
        if value.channel == channel:
            return value
        return EvidenceObservation(
            channel=channel,
            availability=value.availability,
            embedding=value.embedding,
            reason=value.reason,
            details=value.details,
        )
    return EvidenceObservation.available(channel, value)
