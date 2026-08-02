"""Deterministic capture-disjoint NoseID DEV_N3 protocol construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from identity_methods.nose.dataset import NoseIDSample


_MAX_SAMPLES_PER_CAPTURE = 4
_PREFERRED_SEPARATION_MS = 500


def capture_id(sample: NoseIDSample) -> str:
    """Return the canonical session-NUL-camera-NUL-video capture identity."""

    values = (sample.session_id, sample.camera_id, sample.video_id)
    if any(not value or "\0" in value for value in values):
        raise ValueError("capture components must be non-empty and contain no NUL")
    return "\0".join(values)


def stable_capture_order_key(seed: int, identity: str, value: str) -> bytes:
    """Hash the complete seed, identity, and capture tuple for stable ordering."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("protocol seed must be an integer")
    if (
        not isinstance(identity, str)
        or not identity
        or not isinstance(value, str)
        or not value
    ):
        raise ValueError("identity and capture must be non-empty strings")
    return hashlib.sha256(f"{seed}\0{identity}\0{value}".encode("utf-8")).digest()


def select_temporally_farthest(
    samples: tuple[NoseIDSample, ...] | list[NoseIDSample],
    *,
    limit: int = _MAX_SAMPLES_PER_CAPTURE,
    preferred_separation_ms: int = _PREFERRED_SEPARATION_MS,
) -> tuple[NoseIDSample, ...]:
    """Select temporal coverage only, preferring at least 500 ms separation."""

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 4:
        raise ValueError("temporal selection limit must be in [1, 4]")
    if (
        isinstance(preferred_separation_ms, bool)
        or not isinstance(preferred_separation_ms, int)
        or preferred_separation_ms < 0
    ):
        raise ValueError("preferred temporal separation must be non-negative")
    ordered = sorted(
        samples,
        key=lambda sample: (sample.timestamp_ms, sample.frame_index, sample.sample_id),
    )
    if not ordered:
        raise ValueError("capture must contain at least one sample")
    if len({sample.sample_id for sample in ordered}) != len(ordered):
        raise ValueError("capture contains duplicate sample_id values")
    if len(ordered) <= limit:
        return tuple(ordered)

    selected = [ordered[0]]
    remaining = ordered[1:]
    while len(selected) < limit:
        def rank(sample: NoseIDSample) -> tuple[int, int, int, int, str]:
            minimum_distance = min(
                abs(sample.timestamp_ms - chosen.timestamp_ms) for chosen in selected
            )
            return (
                -int(minimum_distance >= preferred_separation_ms),
                -minimum_distance,
                sample.timestamp_ms,
                sample.frame_index,
                sample.sample_id,
            )

        chosen = min(remaining, key=rank)
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(
        sorted(
            selected,
            key=lambda sample: (sample.timestamp_ms, sample.frame_index, sample.sample_id),
        )
    )


@dataclass(frozen=True, slots=True)
class NoseIDCaptureTemplate:
    identity_id: str
    capture_id: str
    samples: tuple[NoseIDSample, ...]

    def __post_init__(self) -> None:
        if not self.samples or len(self.samples) > _MAX_SAMPLES_PER_CAPTURE:
            raise ValueError("capture template must contain between one and four samples")
        if any(sample.registered_dog_id != self.identity_id for sample in self.samples):
            raise ValueError("capture template identity differs")
        if any(capture_id(sample) != self.capture_id for sample in self.samples):
            raise ValueError("capture template capture differs")


@dataclass(frozen=True, slots=True)
class NoseIDProtocolFold:
    fold_index: int
    gallery: tuple[NoseIDCaptureTemplate, ...]
    queries: tuple[NoseIDCaptureTemplate, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or self.fold_index < 0
            or not self.gallery
            or not self.queries
        ):
            raise ValueError("protocol fold must have gallery and query templates")
        gallery_identities = [template.identity_id for template in self.gallery]
        if len(gallery_identities) != len(set(gallery_identities)):
            raise ValueError("protocol fold must have one gallery capture per identity")
        if set(gallery_identities) != {template.identity_id for template in self.queries}:
            raise ValueError("protocol query identities differ from gallery identities")
        gallery_captures = {template.capture_id for template in self.gallery}
        query_captures = {template.capture_id for template in self.queries}
        if gallery_captures & query_captures:
            raise ValueError("protocol gallery and query captures must be disjoint")


def build_dev_n3_folds(
    samples: tuple[NoseIDSample, ...] | list[NoseIDSample],
    *,
    seed: int,
) -> tuple[NoseIDProtocolFold, ...]:
    """Build up to three DEV folds with one gallery capture per identity."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("protocol seed must be an integer")
    if not samples:
        raise ValueError("DEV protocol samples must not be empty")
    identities: dict[str, dict[str, list[NoseIDSample]]] = {}
    sample_ids: set[str] = set()
    capture_owners: dict[str, str] = {}
    for sample in samples:
        if not isinstance(sample, NoseIDSample):
            raise TypeError("DEV protocol rows must be NoseIDSample values")
        if sample.split_role != "DEV":
            raise ValueError("DEV protocol accepts only DEV samples")
        if sample.sample_id in sample_ids:
            raise ValueError(f"duplicate DEV sample_id: {sample.sample_id}")
        sample_ids.add(sample.sample_id)
        current_capture = capture_id(sample)
        previous_owner = capture_owners.setdefault(
            current_capture, sample.registered_dog_id
        )
        if previous_owner != sample.registered_dog_id:
            raise ValueError("one capture cannot contain multiple DEV identities")
        identities.setdefault(sample.registered_dog_id, {}).setdefault(
            current_capture, []
        ).append(sample)

    minimum_captures = min(len(captures) for captures in identities.values())
    fold_count = min(3, minimum_captures)
    if fold_count < 2:
        raise ValueError("every DEV identity needs at least two captures")

    ordered_captures: dict[str, tuple[str, ...]] = {}
    templates: dict[tuple[str, str], NoseIDCaptureTemplate] = {}
    for identity in sorted(identities):
        captures = identities[identity]
        ordered_captures[identity] = tuple(
            sorted(
                captures,
                key=lambda value: (stable_capture_order_key(seed, identity, value), value),
            )
        )
        for current_capture, capture_samples in captures.items():
            templates[(identity, current_capture)] = NoseIDCaptureTemplate(
                identity_id=identity,
                capture_id=current_capture,
                samples=select_temporally_farthest(capture_samples),
            )

    folds: list[NoseIDProtocolFold] = []
    for fold_index in range(fold_count):
        gallery: list[NoseIDCaptureTemplate] = []
        queries: list[NoseIDCaptureTemplate] = []
        for identity in sorted(ordered_captures):
            captures = ordered_captures[identity]
            gallery_capture = captures[fold_index]
            gallery.append(templates[(identity, gallery_capture)])
            queries.extend(
                templates[(identity, current_capture)]
                for current_capture in captures
                if current_capture != gallery_capture
            )
        folds.append(
            NoseIDProtocolFold(
                fold_index=fold_index,
                gallery=tuple(gallery),
                queries=tuple(queries),
            )
        )
    return tuple(folds)


__all__ = [
    "NoseIDCaptureTemplate",
    "NoseIDProtocolFold",
    "build_dev_n3_folds",
    "capture_id",
    "select_temporally_farthest",
    "stable_capture_order_key",
]
