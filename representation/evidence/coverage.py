"""Fixed-memory, duration-weighted G1 evidence coverage summaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from shared.contracts.contracts import Modality
from shared.foundation.provenance import content_sha256

PIXEL_BIN_UPPER_EDGES = (64, 96, 128, 224, 256, 384, 512)
UNIT_BIN_UPPER_EDGES = tuple(index / 10 for index in range(1, 11))


@dataclass(frozen=True, slots=True)
class CoveragePolicy:
    name: str
    expected_sample_period_ns: int
    maximum_hold_periods: float
    minimum_dog_height_px: int
    minimum_head_long_edge_px: int
    minimum_face_min_edge_px: int
    minimum_visible_fraction: float
    maximum_occlusion_fraction: float
    maximum_motion_blur_score: float
    maximum_defocus_blur_score: float
    maximum_cage_bar_occlusion_fraction: float
    minimum_localization_confidence: float
    maximum_ir_saturation_fraction: float
    minimum_usable_tracklet_duration_ns: int

    def __post_init__(self) -> None:
        _require_nonempty("name", self.name)
        _positive_int("expected_sample_period_ns", self.expected_sample_period_ns)
        _positive_int(
            "minimum_usable_tracklet_duration_ns",
            self.minimum_usable_tracklet_duration_ns,
        )
        for field_name in (
            "minimum_dog_height_px",
            "minimum_head_long_edge_px",
            "minimum_face_min_edge_px",
        ):
            _positive_int(field_name, getattr(self, field_name))
        if (
            not isfinite(self.maximum_hold_periods)
            or self.maximum_hold_periods < 1.0
        ):
            raise ValueError("maximum_hold_periods must be finite and >= 1")
        for field_name in (
            "minimum_visible_fraction",
            "maximum_occlusion_fraction",
            "maximum_motion_blur_score",
            "maximum_defocus_blur_score",
            "maximum_cage_bar_occlusion_fraction",
            "minimum_localization_confidence",
            "maximum_ir_saturation_fraction",
        ):
            _unit_interval(field_name, getattr(self, field_name))

    @property
    def maximum_hold_ns(self) -> int:
        return round(self.expected_sample_period_ns * self.maximum_hold_periods)

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "name": self.name,
            "expected_sample_period_ns": self.expected_sample_period_ns,
            "maximum_hold_periods": self.maximum_hold_periods,
            "minimum_dog_height_px": self.minimum_dog_height_px,
            "minimum_head_long_edge_px": self.minimum_head_long_edge_px,
            "minimum_face_min_edge_px": self.minimum_face_min_edge_px,
            "minimum_visible_fraction": self.minimum_visible_fraction,
            "maximum_occlusion_fraction": self.maximum_occlusion_fraction,
            "maximum_motion_blur_score": self.maximum_motion_blur_score,
            "maximum_defocus_blur_score": self.maximum_defocus_blur_score,
            "maximum_cage_bar_occlusion_fraction": (
                self.maximum_cage_bar_occlusion_fraction
            ),
            "minimum_localization_confidence": (
                self.minimum_localization_confidence
            ),
            "maximum_ir_saturation_fraction": (
                self.maximum_ir_saturation_fraction
            ),
            "minimum_usable_tracklet_duration_ns": (
                self.minimum_usable_tracklet_duration_ns
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CoveragePolicy:
        _reject_unknown_keys(payload, set(cls.__dataclass_fields__), "coverage policy")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class CoverageObservation:
    timestamp_ns: int
    modality: Modality
    dog_count: int
    dog_crop_height_px: int | None
    head_long_edge_px: int | None
    face_min_edge_px: int | None
    visible_fraction: float | None
    occlusion_fraction: float | None
    motion_blur_score: float | None
    defocus_blur_score: float | None
    cage_bar_occlusion_fraction: float | None
    localization_confidence: float | None
    exposure_ok: bool | None
    ir_saturation_fraction: float | None
    camera_id: str | None = None
    session_id: str | None = None
    track_id: str | None = None

    def __post_init__(self) -> None:
        _nonnegative_int("timestamp_ns", self.timestamp_ns)
        _nonnegative_int("dog_count", self.dog_count)
        for field_name in (
            "dog_crop_height_px",
            "head_long_edge_px",
            "face_min_edge_px",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _positive_int(field_name, value)
        for field_name in (
            "visible_fraction",
            "occlusion_fraction",
            "motion_blur_score",
            "defocus_blur_score",
            "cage_bar_occlusion_fraction",
            "localization_confidence",
            "ir_saturation_fraction",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _unit_interval(field_name, value)
        if self.exposure_ok is not None and not isinstance(self.exposure_ok, bool):
            raise TypeError("exposure_ok must be boolean or null")
        namespace = (self.camera_id, self.session_id, self.track_id)
        if any(value is not None for value in namespace):
            if not all(value is not None for value in namespace):
                raise ValueError(
                    "camera_id, session_id, and track_id must be provided together"
                )
            for field_name, value in zip(
                ("camera_id", "session_id", "track_id"),
                namespace,
            ):
                _require_nonempty(field_name, value)

    @property
    def track_key(self) -> tuple[str, str, str] | None:
        if self.camera_id is None:
            return None
        return (self.camera_id, self.session_id, self.track_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "modality": self.modality.value,
            "dog_count": self.dog_count,
            "dog_crop_height_px": self.dog_crop_height_px,
            "head_long_edge_px": self.head_long_edge_px,
            "face_min_edge_px": self.face_min_edge_px,
            "visible_fraction": self.visible_fraction,
            "occlusion_fraction": self.occlusion_fraction,
            "motion_blur_score": self.motion_blur_score,
            "defocus_blur_score": self.defocus_blur_score,
            "cage_bar_occlusion_fraction": self.cage_bar_occlusion_fraction,
            "localization_confidence": self.localization_confidence,
            "exposure_ok": self.exposure_ok,
            "ir_saturation_fraction": self.ir_saturation_fraction,
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "track_id": self.track_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CoverageObservation:
        _reject_unknown_keys(
            payload,
            set(cls.__dataclass_fields__),
            "coverage observation",
        )
        kwargs = dict(payload)
        kwargs["modality"] = Modality(payload["modality"])
        return cls(**kwargs)


class _CoverageBucket:
    __slots__ = (
        "cage_bar_occlusion_histogram_ns",
        "defocus_blur_histogram_ns",
        "dog_crop_histogram_ns",
        "dog_size_available_ns",
        "exposure_failure_ns",
        "face_histogram_ns",
        "face_size_available_ns",
        "face_usable_ns",
        "full_body_usable_ns",
        "head_histogram_ns",
        "head_size_available_ns",
        "head_usable_ns",
        "ir_saturation_histogram_ns",
        "missing_quality_ns",
        "motion_blur_histogram_ns",
        "multiple_dogs_ns",
        "no_dog_ns",
        "observed_ns",
        "occlusion_histogram_ns",
        "single_dog_ns",
        "visibility_histogram_ns",
    )

    def __init__(self) -> None:
        bins = len(PIXEL_BIN_UPPER_EDGES) + 1
        self.observed_ns = 0
        self.no_dog_ns = 0
        self.single_dog_ns = 0
        self.multiple_dogs_ns = 0
        self.missing_quality_ns = 0
        self.dog_size_available_ns = 0
        self.head_size_available_ns = 0
        self.face_size_available_ns = 0
        self.exposure_failure_ns = 0
        self.full_body_usable_ns = 0
        self.head_usable_ns = 0
        self.face_usable_ns = 0
        self.dog_crop_histogram_ns = [0] * bins
        self.head_histogram_ns = [0] * bins
        self.face_histogram_ns = [0] * bins
        unit_bins = len(UNIT_BIN_UPPER_EDGES)
        self.visibility_histogram_ns = [0] * unit_bins
        self.occlusion_histogram_ns = [0] * unit_bins
        self.motion_blur_histogram_ns = [0] * unit_bins
        self.defocus_blur_histogram_ns = [0] * unit_bins
        self.cage_bar_occlusion_histogram_ns = [0] * unit_bins
        self.ir_saturation_histogram_ns = [0] * unit_bins

    def add(
        self,
        observation: CoverageObservation,
        duration_ns: int,
        policy: CoveragePolicy,
    ) -> None:
        self.observed_ns += duration_ns
        if observation.dog_count == 0:
            self.no_dog_ns += duration_ns
            return
        if observation.dog_count > 1:
            self.multiple_dogs_ns += duration_ns
            return
        self.single_dog_ns += duration_ns
        if observation.dog_crop_height_px is not None:
            self.dog_size_available_ns += duration_ns
        if observation.head_long_edge_px is not None:
            self.head_size_available_ns += duration_ns
        if observation.face_min_edge_px is not None:
            self.face_size_available_ns += duration_ns
        if observation.exposure_ok is False:
            self.exposure_failure_ns += duration_ns
        _add_histogram(
            self.dog_crop_histogram_ns,
            observation.dog_crop_height_px,
            duration_ns,
        )
        _add_unit_histogram(
            self.visibility_histogram_ns,
            observation.visible_fraction,
            duration_ns,
        )
        _add_unit_histogram(
            self.occlusion_histogram_ns,
            observation.occlusion_fraction,
            duration_ns,
        )
        _add_unit_histogram(
            self.motion_blur_histogram_ns,
            observation.motion_blur_score,
            duration_ns,
        )
        _add_unit_histogram(
            self.defocus_blur_histogram_ns,
            observation.defocus_blur_score,
            duration_ns,
        )
        _add_unit_histogram(
            self.cage_bar_occlusion_histogram_ns,
            observation.cage_bar_occlusion_fraction,
            duration_ns,
        )
        _add_unit_histogram(
            self.ir_saturation_histogram_ns,
            observation.ir_saturation_fraction,
            duration_ns,
        )
        _add_histogram(
            self.head_histogram_ns,
            observation.head_long_edge_px,
            duration_ns,
        )
        _add_histogram(
            self.face_histogram_ns,
            observation.face_min_edge_px,
            duration_ns,
        )
        quality = _base_quality(observation, policy)
        if quality is None:
            self.missing_quality_ns += duration_ns
            return
        if not quality:
            return
        if (
            observation.dog_crop_height_px is not None
            and observation.dog_crop_height_px >= policy.minimum_dog_height_px
        ):
            self.full_body_usable_ns += duration_ns
        if (
            observation.head_long_edge_px is not None
            and observation.head_long_edge_px >= policy.minimum_head_long_edge_px
        ):
            self.head_usable_ns += duration_ns
        if (
            observation.face_min_edge_px is not None
            and observation.face_min_edge_px >= policy.minimum_face_min_edge_px
        ):
            self.face_usable_ns += duration_ns

    def to_dict(self) -> dict[str, Any]:
        denominator = self.single_dog_ns
        return {
            "observed_duration_ns": self.observed_ns,
            "no_dog_duration_ns": self.no_dog_ns,
            "single_dog_duration_ns": self.single_dog_ns,
            "multiple_dogs_duration_ns": self.multiple_dogs_ns,
            "missing_quality_duration_ns": self.missing_quality_ns,
            "dog_size_available_duration_ns": self.dog_size_available_ns,
            "head_size_available_duration_ns": self.head_size_available_ns,
            "face_size_available_duration_ns": self.face_size_available_ns,
            "exposure_failure_duration_ns": self.exposure_failure_ns,
            "full_body_usable_duration_ns": self.full_body_usable_ns,
            "head_usable_duration_ns": self.head_usable_ns,
            "face_usable_duration_ns": self.face_usable_ns,
            "full_body_coverage_given_single_dog": _safe_ratio(
                self.full_body_usable_ns,
                denominator,
            ),
            "head_coverage_given_single_dog": _safe_ratio(
                self.head_usable_ns,
                denominator,
            ),
            "face_coverage_given_single_dog": _safe_ratio(
                self.face_usable_ns,
                denominator,
            ),
            "missing_quality_given_single_dog": _safe_ratio(
                self.missing_quality_ns,
                denominator,
            ),
            "dog_size_availability_given_single_dog": _safe_ratio(
                self.dog_size_available_ns,
                denominator,
            ),
            "head_size_availability_given_single_dog": _safe_ratio(
                self.head_size_available_ns,
                denominator,
            ),
            "face_size_availability_given_single_dog": _safe_ratio(
                self.face_size_available_ns,
                denominator,
            ),
            "pixel_bin_upper_edges": list(PIXEL_BIN_UPPER_EDGES),
            "dog_crop_height_histogram_ns": self.dog_crop_histogram_ns,
            "head_long_edge_histogram_ns": self.head_histogram_ns,
            "face_min_edge_histogram_ns": self.face_histogram_ns,
            "unit_interval_bin_upper_edges": list(UNIT_BIN_UPPER_EDGES),
            "visibility_histogram_ns": self.visibility_histogram_ns,
            "occlusion_histogram_ns": self.occlusion_histogram_ns,
            "motion_blur_histogram_ns": self.motion_blur_histogram_ns,
            "defocus_blur_histogram_ns": self.defocus_blur_histogram_ns,
            "cage_bar_occlusion_histogram_ns": (
                self.cage_bar_occlusion_histogram_ns
            ),
            "ir_saturation_histogram_ns": self.ir_saturation_histogram_ns,
        }


class CoverageAccumulator:
    """Chronological fixed-memory accumulator."""

    __slots__ = (
        "_active_run_duration_ns",
        "_active_run_key",
        "_aggregate",
        "_closed",
        "_last_observation",
        "_missing_track_key_duration_ns",
        "_modalities",
        "_policy",
        "_timeline_start_ns",
        "_unobserved_gap_ns",
        "_usable_tracklet_count",
        "_usable_tracklet_duration_ns",
    )

    def __init__(
        self,
        policy: CoveragePolicy,
        *,
        timeline_start_ns: int | None = None,
    ) -> None:
        if timeline_start_ns is not None:
            _nonnegative_int("timeline_start_ns", timeline_start_ns)
        self._policy = policy
        self._aggregate = _CoverageBucket()
        self._modalities = {modality: _CoverageBucket() for modality in Modality}
        self._last_observation: CoverageObservation | None = None
        self._active_run_key: tuple[str, str, str, Modality] | None = None
        self._active_run_duration_ns = 0
        self._usable_tracklet_count = 0
        self._usable_tracklet_duration_ns = 0
        self._missing_track_key_duration_ns = 0
        self._unobserved_gap_ns = 0
        self._timeline_start_ns = timeline_start_ns
        self._closed = False

    def observe(self, observation: CoverageObservation) -> None:
        if self._closed:
            raise RuntimeError("coverage accumulator is finalized")
        if self._last_observation is None and self._timeline_start_ns is not None:
            if observation.timestamp_ns < self._timeline_start_ns:
                raise ValueError("observation precedes the declared timeline")
            self._unobserved_gap_ns += (
                observation.timestamp_ns - self._timeline_start_ns
            )
        if self._last_observation is not None:
            gap = observation.timestamp_ns - self._last_observation.timestamp_ns
            if gap <= 0:
                raise ValueError("observation timestamps must be strictly increasing")
            represented = min(gap, self._policy.maximum_hold_ns)
            self._add(self._last_observation, represented)
            if gap > self._policy.maximum_hold_ns:
                self._close_run()
            self._unobserved_gap_ns += gap - represented
        self._last_observation = observation

    def finalize(self, *, timeline_end_ns: int | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("coverage accumulator is already finalized")
        if timeline_end_ns is not None:
            _nonnegative_int("timeline_end_ns", timeline_end_ns)
            if (
                self._timeline_start_ns is not None
                and timeline_end_ns < self._timeline_start_ns
            ):
                raise ValueError("timeline end precedes timeline start")
        if self._last_observation is not None:
            if (
                timeline_end_ns is not None
                and timeline_end_ns < self._last_observation.timestamp_ns
            ):
                raise ValueError("timeline end precedes the final observation")
            final_gap = (
                timeline_end_ns - self._last_observation.timestamp_ns
                if timeline_end_ns is not None
                else self._policy.expected_sample_period_ns
            )
            represented = min(final_gap, self._policy.maximum_hold_ns)
            self._add(self._last_observation, represented)
            self._unobserved_gap_ns += final_gap - represented
        elif (
            self._timeline_start_ns is not None and timeline_end_ns is not None
        ):
            self._unobserved_gap_ns += timeline_end_ns - self._timeline_start_ns
        self._close_run()
        self._closed = True
        single_dog_hours = self._aggregate.single_dog_ns / 3_600_000_000_000
        return {
            "schema_version": "cvi.coverage.v1",
            "policy": self._policy.to_dict(),
            "policy_sha256": self._policy.policy_sha256,
            "timeline_start_ns": self._timeline_start_ns,
            "timeline_end_ns": timeline_end_ns,
            "unobserved_gap_duration_ns": self._unobserved_gap_ns,
            "usable_tracklet_opportunities": self._usable_tracklet_count,
            "usable_tracklet_duration_ns": self._usable_tracklet_duration_ns,
            "usable_tracklet_opportunities_per_single_dog_hour": (
                self._usable_tracklet_count / single_dog_hours
                if single_dog_hours
                else None
            ),
            "usable_evidence_missing_track_key_duration_ns": (
                self._missing_track_key_duration_ns
            ),
            "aggregate": self._aggregate.to_dict(),
            "by_modality": {
                modality.value: self._modalities[modality].to_dict()
                for modality in Modality
            },
        }

    def _add(self, observation: CoverageObservation, duration_ns: int) -> None:
        self._aggregate.add(observation, duration_ns, self._policy)
        self._modalities[observation.modality].add(
            observation,
            duration_ns,
            self._policy,
        )
        if not _evidence_is_usable(observation, self._policy):
            self._close_run()
            return
        track_key = observation.track_key
        if track_key is None:
            self._missing_track_key_duration_ns += duration_ns
            self._close_run()
            return
        run_key = (*track_key, observation.modality)
        if self._active_run_key != run_key:
            self._close_run()
            self._active_run_key = run_key
        self._active_run_duration_ns += duration_ns

    def _close_run(self) -> None:
        if (
            self._active_run_key is not None
            and self._active_run_duration_ns
            >= self._policy.minimum_usable_tracklet_duration_ns
        ):
            self._usable_tracklet_count += 1
            self._usable_tracklet_duration_ns += self._active_run_duration_ns
        self._active_run_key = None
        self._active_run_duration_ns = 0


def _base_quality(
    observation: CoverageObservation,
    policy: CoveragePolicy,
) -> bool | None:
    required = (
        observation.visible_fraction,
        observation.occlusion_fraction,
        observation.motion_blur_score,
        observation.defocus_blur_score,
        observation.cage_bar_occlusion_fraction,
        observation.localization_confidence,
        observation.exposure_ok,
    )
    if any(value is None for value in required):
        return None
    if (
        observation.modality in {Modality.IR, Modality.MIXED}
        and observation.ir_saturation_fraction is None
    ):
        return None
    return bool(
        observation.visible_fraction >= policy.minimum_visible_fraction
        and observation.occlusion_fraction <= policy.maximum_occlusion_fraction
        and observation.motion_blur_score <= policy.maximum_motion_blur_score
        and observation.defocus_blur_score <= policy.maximum_defocus_blur_score
        and observation.cage_bar_occlusion_fraction
        <= policy.maximum_cage_bar_occlusion_fraction
        and observation.localization_confidence
        >= policy.minimum_localization_confidence
        and observation.exposure_ok
        and (
            observation.modality is Modality.RGB
            or observation.ir_saturation_fraction
            <= policy.maximum_ir_saturation_fraction
        )
    )


def _evidence_is_usable(
    observation: CoverageObservation,
    policy: CoveragePolicy,
) -> bool:
    if observation.dog_count != 1 or _base_quality(observation, policy) is not True:
        return False
    return bool(
        (
            observation.dog_crop_height_px is not None
            and observation.dog_crop_height_px >= policy.minimum_dog_height_px
        )
        or (
            observation.head_long_edge_px is not None
            and observation.head_long_edge_px >= policy.minimum_head_long_edge_px
        )
        or (
            observation.face_min_edge_px is not None
            and observation.face_min_edge_px >= policy.minimum_face_min_edge_px
        )
    )


def _add_histogram(
    histogram: list[int],
    value: int | None,
    duration_ns: int,
) -> None:
    if value is None:
        return
    for index, upper_edge in enumerate(PIXEL_BIN_UPPER_EDGES):
        if value < upper_edge:
            histogram[index] += duration_ns
            return
    histogram[-1] += duration_ns


def _add_unit_histogram(
    histogram: list[int],
    value: float | None,
    duration_ns: int,
) -> None:
    if value is None:
        return
    index = min(int(value * 10), 9)
    histogram[index] += duration_ns


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _require_nonempty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _unit_interval(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _reject_unknown_keys(
    payload: dict[str, Any],
    allowed: set[str],
    object_name: str,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{object_name} must be an object")
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            f"{object_name} has unknown fields: {', '.join(sorted(unknown))}"
        )
