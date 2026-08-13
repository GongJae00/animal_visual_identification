from __future__ import annotations

import numpy as np
from PIL import Image

from evidence_fusion.base import (
    AbstractEvidencer,
    EvidenceObservation,
    RequiredEvidenceUnavailableError,
)
from evidence_fusion.quality import QualityObservation


class EvidenceExtractionPipeline:
    def __init__(
        self,
        evidencer_map: dict[str, AbstractEvidencer | None],
        optional_channels: set[str] | frozenset[str] | None = None,
    ):
        self._evidencer_map = {
            k: v for k, v in evidencer_map.items() if v is not None
        }
        self._optional_channels = frozenset(optional_channels or ())
        if not self._optional_channels <= set(self._evidencer_map):
            raise ValueError("optional channels must be active evidence channels")
        if not set(self._evidencer_map) - self._optional_channels:
            raise ValueError("at least one evidence channel must be required")

    @property
    def active_channels(self) -> list[str]:
        return list(self._evidencer_map.keys())

    @property
    def optional_channels(self) -> frozenset[str]:
        return self._optional_channels

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset(self._evidencer_map) - self._optional_channels

    def extract_observations(
        self, image: Image.Image
    ) -> dict[str, EvidenceObservation]:
        observations: dict[str, EvidenceObservation] = {}
        for name, evidencer in self._evidencer_map.items():
            value = evidencer.extract(image)
            observation = _as_observation(name, value)
            if not observation.is_available and name not in self._optional_channels:
                raise RequiredEvidenceUnavailableError(
                    f"required evidence channel {name!r} is unavailable: "
                    f"{observation.reason.value}"
                )
            observations[name] = observation
        return observations

    def extract_all(self, image: Image.Image
                    ) -> dict[str, np.ndarray]:
        return {
            name: observation.embedding
            for name, observation in self.extract_observations(image).items()
            if observation.is_available and observation.embedding is not None
        }

    def extract_with_quality(self, image: Image.Image
                             ) -> tuple[
                                 dict[str, np.ndarray],
                                 dict[str, QualityObservation],
                             ]:
        embs: dict[str, np.ndarray] = {}
        quals: dict[str, QualityObservation] = {}
        for name, ev in self._evidencer_map.items():
            observation = _as_observation(name, ev.extract(image))
            if not observation.is_available:
                if name not in self._optional_channels:
                    raise RequiredEvidenceUnavailableError(
                        f"required evidence channel {name!r} is unavailable: "
                        f"{observation.reason.value}"
                    )
                continue
            assert observation.embedding is not None
            embs[name] = observation.embedding
            quals[name] = ev.estimate_quality(image, channel=name)
        return embs, quals

    def estimate_quality(
        self,
        image: Image.Image,
    ) -> dict[str, QualityObservation]:
        return {
            name: ev.estimate_quality(image, channel=name)
            for name, ev in self._evidencer_map.items()
        }


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
