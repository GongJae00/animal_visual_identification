from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from PIL import Image

from evidence_fusion.quality import (
    QualityDiagnostics,
    QualityMapping,
    QualityObservation,
    QualityReason,
    QualityState,
    RoiBox,
    observe_quality,
)


class EvidenceAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class EvidenceUnavailableReason(str, Enum):
    NO_ROI = "NO_ROI"
    ROI_MISSING = "NO_ROI"
    ROI_TOO_SMALL = "ROI_TOO_SMALL"
    ROI_LOW_RESOLUTION = "ROI_LOW_RESOLUTION"
    INSUFFICIENT_LANDMARKS = "INSUFFICIENT_LANDMARKS"


class EvidenceInsufficiency(RuntimeError):
    """Expected image-content insufficiency, never an operational failure."""

    def __init__(
        self,
        reason: EvidenceUnavailableReason,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.details = dict(details or {})


class RequiredEvidenceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EvidenceObservation:
    channel: str
    availability: EvidenceAvailability
    embedding: np.ndarray | None = None
    reason: EvidenceUnavailableReason | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel:
            raise ValueError("evidence channel must be a non-empty string")
        if not isinstance(self.availability, EvidenceAvailability):
            raise TypeError("availability must be an EvidenceAvailability")
        if not isinstance(self.details, dict):
            raise TypeError("evidence details must be an object")
        object.__setattr__(self, "details", dict(self.details))
        if self.availability is EvidenceAvailability.UNAVAILABLE:
            if self.embedding is not None or not isinstance(
                self.reason, EvidenceUnavailableReason
            ):
                raise ValueError(
                    "unavailable evidence requires a typed reason and no embedding"
                )
            return
        if self.reason is not None:
            raise ValueError("available evidence cannot carry an unavailable reason")
        if (
            not isinstance(self.embedding, np.ndarray)
            or self.embedding.dtype != np.float32
            or self.embedding.ndim != 1
            or not np.isfinite(self.embedding).all()
        ):
            raise ValueError("available evidence must contain a finite float32 vector")
        norm = float(np.linalg.norm(self.embedding))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("available evidence must have non-zero norm")

    @classmethod
    def available(
        cls,
        channel: str,
        embedding: np.ndarray,
        *,
        details: dict[str, Any] | None = None,
    ) -> EvidenceObservation:
        return cls(
            channel,
            EvidenceAvailability.AVAILABLE,
            np.asarray(embedding, dtype=np.float32),
            details=dict(details or {}),
        )

    @classmethod
    def unavailable(
        cls,
        channel: str,
        reason: EvidenceUnavailableReason,
        *,
        details: dict[str, Any] | None = None,
    ) -> EvidenceObservation:
        return cls(
            channel,
            EvidenceAvailability.UNAVAILABLE,
            reason=reason,
            details=dict(details or {}),
        )

    @property
    def is_available(self) -> bool:
        return self.availability is EvidenceAvailability.AVAILABLE

    @property
    def abstained(self) -> bool:
        return not self.is_available

    @property
    def abstain_reason(self) -> EvidenceUnavailableReason | None:
        return self.reason

    @property
    def roi_box(self) -> tuple[int, int, int, int] | None:
        value = self.details.get("roi_box")
        return None if value is None else tuple(value)

    @property
    def detection_confidence(self) -> float | None:
        value = self.details.get("detection_confidence")
        return None if value is None else float(value)


class AbstractEvidencer(ABC):
    name: str = "base"
    output_dim: int = 0

    @abstractmethod
    def extract(self, image: Image.Image) -> np.ndarray | EvidenceObservation:
        ...

    @abstractmethod
    def extract_batch(
        self, images: list[Image.Image]
    ) -> np.ndarray | list[EvidenceObservation]:
        ...

    def quality_roi(self, image: Image.Image) -> RoiBox | None:
        return None

    def map_quality(self, diagnostics: QualityDiagnostics) -> QualityMapping:
        return QualityMapping(
            state=QualityState.UNAVAILABLE,
            reason_codes=(QualityReason.MAPPING_NOT_CONFIGURED,),
        )

    def estimate_quality(
        self,
        image: Image.Image,
        *,
        channel: str | None = None,
        roi_box: RoiBox | None = None,
    ) -> QualityObservation:
        selected_roi = self.quality_roi(image) if roi_box is None else roi_box
        return observe_quality(
            image,
            channel=channel or self.name,
            roi_box=selected_roi,
            mapper=self.map_quality,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "output_dim": self.output_dim}
