"""Runtime-enforced state and evidence boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any


class StreamStatus(StrEnum):
    OK = "STREAM_OK"
    DEGRADED = "STREAM_DEGRADED"
    FAILED = "STREAM_FAILED"


class OccupancyStatus(StrEnum):
    NOT_EVALUATED = "NOT_EVALUATED"
    NO_DOG = "NO_DOG"
    SINGLE_DOG = "SINGLE_DOG"
    MULTIPLE_DOGS = "MULTIPLE_DOGS"
    UNCERTAIN = "OCCUPANCY_UNCERTAIN"


class EvidenceStatus(StrEnum):
    NOT_EVALUATED = "NOT_EVALUATED"
    USABLE = "USABLE"
    NO_USABLE_EVIDENCE = "NO_USABLE_EVIDENCE"


class VisualIdentityStatus(StrEnum):
    NOT_EVALUATED = "NOT_EVALUATED"
    PENDING = "PENDING"
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ConflictStatus(StrEnum):
    NONE = "NONE"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"


class DecisionStatus(StrEnum):
    NO_DOG = "NO_DOG"
    MULTIPLE_DOGS = "MULTIPLE_DOGS"
    DOG_PRESENT = "DOG_PRESENT"
    NO_USABLE_EVIDENCE = "NO_USABLE_EVIDENCE"
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"


class Modality(StrEnum):
    RGB = "RGB"
    IR = "IR"
    MIXED = "RGB_IR_MIXED"


def _require_nonempty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class TrackKey:
    """A track identifier in its required camera/session namespace."""

    camera_id: str
    session_id: str
    track_id: str

    def __post_init__(self) -> None:
        _require_nonempty("camera_id", self.camera_id)
        _require_nonempty("session_id", self.session_id)
        _require_nonempty("track_id", self.track_id)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    registered_dog_id: str
    score: float

    def __post_init__(self) -> None:
        _require_nonempty("registered_dog_id", self.registered_dog_id)
        if not isfinite(self.score):
            raise ValueError("candidate score must be finite")


@dataclass(frozen=True, slots=True)
class EvidenceFrameRef:
    """Reference to source evidence without copying image data into the result."""

    camera_id: str
    frame_detection_id: str
    timestamp_ns: int
    modality: Modality

    def __post_init__(self) -> None:
        _require_nonempty("camera_id", self.camera_id)
        _require_nonempty("frame_detection_id", self.frame_detection_id)
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")


@dataclass(frozen=True, slots=True)
class VisualIdentityResult:
    """Visual-only output. Operational priors are intentionally absent."""

    track: TrackKey
    status: VisualIdentityStatus
    modality: Modality
    input_quality: float
    candidates: tuple[CandidateScore, ...]
    top1_top2_margin: float | None
    evidence_frames: tuple[EvidenceFrameRef, ...]
    model_version: str
    gallery_version: str
    predicted_dog_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty("model_version", self.model_version)
        _require_nonempty("gallery_version", self.gallery_version)
        if not isfinite(self.input_quality) or not 0.0 <= self.input_quality <= 1.0:
            raise ValueError("input_quality must be finite and in [0, 1]")
        if self.status in {
            VisualIdentityStatus.NOT_EVALUATED,
            VisualIdentityStatus.PENDING,
        }:
            raise ValueError("VisualIdentityResult requires an evaluated status")
        if not self.evidence_frames:
            raise ValueError("evaluated visual result requires evidence frames")
        if any(ref.camera_id != self.track.camera_id for ref in self.evidence_frames):
            raise ValueError("evidence camera must match the track namespace")

        dog_ids = tuple(candidate.registered_dog_id for candidate in self.candidates)
        if len(dog_ids) != len(set(dog_ids)):
            raise ValueError("candidate dog IDs must be unique")
        if any(
            earlier.score < later.score
            for earlier, later in zip(self.candidates, self.candidates[1:])
        ):
            raise ValueError("candidates must be sorted by descending score")

        if len(self.candidates) >= 2:
            expected_margin = self.candidates[0].score - self.candidates[1].score
            if self.top1_top2_margin is None or not isfinite(self.top1_top2_margin):
                raise ValueError("two or more candidates require a finite margin")
            if abs(self.top1_top2_margin - expected_margin) > 1e-9:
                raise ValueError("top1_top2_margin does not match candidate scores")
        elif self.top1_top2_margin is not None:
            raise ValueError("margin is undefined with fewer than two candidates")

        if self.status is VisualIdentityStatus.KNOWN:
            if self.predicted_dog_id is None:
                raise ValueError("KNOWN requires predicted_dog_id")
            if not self.candidates:
                raise ValueError("KNOWN requires at least one candidate")
            if self.predicted_dog_id != self.candidates[0].registered_dog_id:
                raise ValueError("predicted dog must be the top-ranked candidate")
        elif self.predicted_dog_id is not None:
            raise ValueError("only KNOWN may assign predicted_dog_id")


@dataclass(frozen=True, slots=True)
class OperationalContext:
    """Metadata joined after visual scoring."""

    camera_id: str
    cage_id: str
    expected_dog_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty("camera_id", self.camera_id)
        _require_nonempty("cage_id", self.cage_id)
        if self.expected_dog_id is not None:
            _require_nonempty("expected_dog_id", self.expected_dog_id)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """A validated snapshot of all orthogonal state axes."""

    timestamp_ns: int
    context: OperationalContext
    stream: StreamStatus
    occupancy: OccupancyStatus
    evidence: EvidenceStatus
    visual_status: VisualIdentityStatus
    conflict: ConflictStatus = ConflictStatus.NONE
    visual_result: VisualIdentityResult | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if (
            self.visual_result is not None
            and self.visual_result.track.camera_id != self.context.camera_id
        ):
            raise ValueError("visual track camera must match operational context")
        if self.visual_result is not None and any(
            ref.timestamp_ns > self.timestamp_ns
            for ref in self.visual_result.evidence_frames
        ):
            raise ValueError("decision cannot use evidence from the future")

        if self.stream is StreamStatus.FAILED:
            if self.occupancy is not OccupancyStatus.NOT_EVALUATED:
                raise ValueError("failed stream blocks occupancy evaluation")
            self._require_no_identity()
            return

        if self.occupancy in {
            OccupancyStatus.NOT_EVALUATED,
            OccupancyStatus.NO_DOG,
            OccupancyStatus.MULTIPLE_DOGS,
            OccupancyStatus.UNCERTAIN,
        }:
            self._require_no_identity()
            return

        if self.evidence is EvidenceStatus.NO_USABLE_EVIDENCE:
            self._require_visual_not_evaluated()
            return
        if self.evidence is EvidenceStatus.NOT_EVALUATED:
            if self.visual_status is not VisualIdentityStatus.PENDING:
                raise ValueError("single dog awaiting evidence must be PENDING")
            if self.visual_result is not None:
                raise ValueError("pending identity cannot have a visual result")
            if self.conflict is not ConflictStatus.NONE:
                raise ValueError("pending identity cannot have a conflict")
            return
        if self.visual_status in {
            VisualIdentityStatus.NOT_EVALUATED,
            VisualIdentityStatus.PENDING,
        }:
            raise ValueError("usable evidence requires an evaluated visual status")
        if self.visual_result is None:
            raise ValueError("evaluated identity requires visual_result")
        if self.visual_result.status is not self.visual_status:
            raise ValueError("visual status axes disagree")

        if self.conflict is ConflictStatus.IDENTITY_CONFLICT:
            if self.visual_status is not VisualIdentityStatus.KNOWN:
                raise ValueError("conflict requires a KNOWN visual result")
            if self.context.expected_dog_id is None:
                raise ValueError("conflict requires expected_dog_id")
            if self.context.expected_dog_id == self.visual_result.predicted_dog_id:
                raise ValueError("matching expected and visual IDs are not a conflict")
        elif (
            self.visual_status is VisualIdentityStatus.KNOWN
            and self.context.expected_dog_id is not None
            and self.context.expected_dog_id != self.visual_result.predicted_dog_id
        ):
            raise ValueError("mismatched expected and visual IDs require conflict")

    def _require_no_identity(self) -> None:
        if self.evidence is not EvidenceStatus.NOT_EVALUATED:
            raise ValueError("occupancy without one dog cannot evaluate evidence")
        self._require_visual_not_evaluated()

    def _require_visual_not_evaluated(self) -> None:
        if self.visual_status is not VisualIdentityStatus.NOT_EVALUATED:
            raise ValueError("identity must be NOT_EVALUATED")
        if self.visual_result is not None:
            raise ValueError("unevaluated identity cannot have a visual result")
        if self.conflict is not ConflictStatus.NONE:
            raise ValueError("unevaluated identity cannot have a conflict")

    @property
    def status(self) -> DecisionStatus | None:
        """Project the orthogonal state without discarding the source axes."""

        if self.stream is StreamStatus.FAILED:
            return None
        if self.occupancy is OccupancyStatus.NO_DOG:
            return DecisionStatus.NO_DOG
        if self.occupancy is OccupancyStatus.MULTIPLE_DOGS:
            return DecisionStatus.MULTIPLE_DOGS
        if self.occupancy in {
            OccupancyStatus.NOT_EVALUATED,
            OccupancyStatus.UNCERTAIN,
        }:
            return None
        if self.evidence is EvidenceStatus.NO_USABLE_EVIDENCE:
            return DecisionStatus.NO_USABLE_EVIDENCE
        if self.evidence is EvidenceStatus.NOT_EVALUATED:
            return DecisionStatus.DOG_PRESENT
        if self.conflict is ConflictStatus.IDENTITY_CONFLICT:
            return DecisionStatus.IDENTITY_CONFLICT
        return DecisionStatus(self.visual_status.value)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable API-shaped representation for logs and fixtures."""

        visual = self.visual_result
        return {
            "timestamp_ns": self.timestamp_ns,
            "operational_context": {
                "camera_id": self.context.camera_id,
                "cage_id": self.context.cage_id,
                "expected_dog_id": self.context.expected_dog_id,
            },
            "state": {
                "status": self.status.value if self.status is not None else None,
                "stream_status": self.stream.value,
                "occupancy_status": self.occupancy.value,
                "evidence_status": self.evidence.value,
                "visual_identity_status": self.visual_status.value,
                "conflict_status": self.conflict.value,
            },
            "visual_result": None if visual is None else _visual_result_dict(visual),
        }


def _visual_result_dict(result: VisualIdentityResult) -> dict[str, Any]:
    return {
        "camera_id": result.track.camera_id,
        "session_id": result.track.session_id,
        "track_id": result.track.track_id,
        "predicted_dog_id": result.predicted_dog_id,
        "top_k_candidates": [
            {
                "registered_dog_id": candidate.registered_dog_id,
                "score": candidate.score,
            }
            for candidate in result.candidates
        ],
        "top1_top2_score_margin": result.top1_top2_margin,
        "input_quality": result.input_quality,
        "modality": result.modality.value,
        "evidence_frames": [
            {
                "camera_id": ref.camera_id,
                "frame_detection_id": ref.frame_detection_id,
                "timestamp_ns": ref.timestamp_ns,
                "modality": ref.modality.value,
            }
            for ref in result.evidence_frames
        ],
        "model_version": result.model_version,
        "gallery_version": result.gallery_version,
    }
