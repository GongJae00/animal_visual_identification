"""G0 camera, source-video, modality, and timestamp intake contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any

from shared.foundation.provenance import content_sha256


class IRMechanism(StrEnum):
    UNKNOWN = "UNKNOWN"
    DAY_NIGHT_SWITCHING = "DAY_NIGHT_SWITCHING"
    DUAL_RGB_NIR = "DUAL_RGB_NIR"
    THERMAL_AUXILIARY = "THERMAL_AUXILIARY"


class ModalityState(StrEnum):
    RGB = "RGB"
    IR = "IR"
    TRANSITION = "RGB_IR_TRANSITION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class CameraSpecification:
    """Camera facts required before interpreting RGB/IR samples.

    Unknown values remain explicit `None` values. `missing_for_g0` determines
    whether the inventory is complete enough for sample admission.
    """

    camera_id: str
    camera_setting_version: str
    sensor_model: str | None = None
    ir_mechanism: IRMechanism = IRMechanism.UNKNOWN
    ir_spectral_band: str | None = None
    width: int | None = None
    height: int | None = None
    stored_fps: float | None = None
    shutter: str | None = None
    gain_mode: str | None = None
    exposure_mode: str | None = None
    white_balance_mode: str | None = None
    wdr_enabled: bool | None = None
    ir_cut_behavior: str | None = None
    codec: str | None = None
    target_bitrate_mbps: float | None = None
    gop_length: int | None = None
    focus_mode: str | None = None
    focal_length_mm: float | None = None
    horizontal_fov_deg: float | None = None
    installation_height_m: float | None = None
    cage_center_distance_m: float | None = None
    pan_deg: float | None = None
    tilt_deg: float | None = None
    timestamp_accuracy_ms: float | None = None
    measured_frame_drop_rate: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty("camera_id", self.camera_id)
        _require_nonempty("camera_setting_version", self.camera_setting_version)
        _optional_positive_int("width", self.width)
        _optional_positive_int("height", self.height)
        _optional_positive_int("gop_length", self.gop_length)
        for name in (
            "stored_fps",
            "target_bitrate_mbps",
            "focal_length_mm",
            "horizontal_fov_deg",
            "installation_height_m",
            "cage_center_distance_m",
            "timestamp_accuracy_ms",
        ):
            _optional_positive_float(name, getattr(self, name))
        if self.measured_frame_drop_rate is not None and not (
            0.0 <= self.measured_frame_drop_rate <= 1.0
        ):
            raise ValueError("measured_frame_drop_rate must be in [0, 1]")

    @property
    def missing_for_g0(self) -> tuple[str, ...]:
        required_values = {
            "sensor_model": self.sensor_model,
            "ir_mechanism": (
                None if self.ir_mechanism is IRMechanism.UNKNOWN else self.ir_mechanism
            ),
            "ir_spectral_band": self.ir_spectral_band,
            "width": self.width,
            "height": self.height,
            "stored_fps": self.stored_fps,
            "shutter": self.shutter,
            "gain_mode": self.gain_mode,
            "exposure_mode": self.exposure_mode,
            "white_balance_mode": self.white_balance_mode,
            "wdr_enabled": self.wdr_enabled,
            "ir_cut_behavior": self.ir_cut_behavior,
            "codec": self.codec,
            "target_bitrate_mbps": self.target_bitrate_mbps,
            "gop_length": self.gop_length,
            "focus_mode": self.focus_mode,
            "focal_length_mm": self.focal_length_mm,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "installation_height_m": self.installation_height_m,
            "cage_center_distance_m": self.cage_center_distance_m,
            "pan_deg": self.pan_deg,
            "tilt_deg": self.tilt_deg,
            "timestamp_accuracy_ms": self.timestamp_accuracy_ms,
            "measured_frame_drop_rate": self.measured_frame_drop_rate,
        }
        return tuple(
            name for name, value in required_values.items() if value is None
        )

    @property
    def g0_ready(self) -> bool:
        return not self.missing_for_g0

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "camera_setting_version": self.camera_setting_version,
            "sensor_model": self.sensor_model,
            "ir_mechanism": self.ir_mechanism.value,
            "ir_spectral_band": self.ir_spectral_band,
            "width": self.width,
            "height": self.height,
            "stored_fps": self.stored_fps,
            "shutter": self.shutter,
            "gain_mode": self.gain_mode,
            "exposure_mode": self.exposure_mode,
            "white_balance_mode": self.white_balance_mode,
            "wdr_enabled": self.wdr_enabled,
            "ir_cut_behavior": self.ir_cut_behavior,
            "codec": self.codec,
            "target_bitrate_mbps": self.target_bitrate_mbps,
            "gop_length": self.gop_length,
            "focus_mode": self.focus_mode,
            "focal_length_mm": self.focal_length_mm,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "installation_height_m": self.installation_height_m,
            "cage_center_distance_m": self.cage_center_distance_m,
            "pan_deg": self.pan_deg,
            "tilt_deg": self.tilt_deg,
            "timestamp_accuracy_ms": self.timestamp_accuracy_ms,
            "measured_frame_drop_rate": self.measured_frame_drop_rate,
            "missing_for_g0": list(self.missing_for_g0),
            "g0_ready": self.g0_ready,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CameraSpecification:
        allowed = {
            "camera_id",
            "camera_setting_version",
            "sensor_model",
            "ir_mechanism",
            "ir_spectral_band",
            "width",
            "height",
            "stored_fps",
            "shutter",
            "gain_mode",
            "exposure_mode",
            "white_balance_mode",
            "wdr_enabled",
            "ir_cut_behavior",
            "codec",
            "target_bitrate_mbps",
            "gop_length",
            "focus_mode",
            "focal_length_mm",
            "horizontal_fov_deg",
            "installation_height_m",
            "cage_center_distance_m",
            "pan_deg",
            "tilt_deg",
            "timestamp_accuracy_ms",
            "measured_frame_drop_rate",
            "missing_for_g0",
            "g0_ready",
        }
        _reject_unknown_keys(payload, allowed, "camera specification")
        kwargs = {
            key: value
            for key, value in payload.items()
            if key not in {"ir_mechanism", "missing_for_g0", "g0_ready"}
        }
        kwargs["ir_mechanism"] = IRMechanism(
            payload.get("ir_mechanism", IRMechanism.UNKNOWN.value)
        )
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class VideoProbeSummary:
    codec: str
    format_name: str
    width: int
    height: int
    average_fps: float
    duration_seconds: float
    bitrate_bps: int | None
    frame_count: int | None
    time_base: str

    def __post_init__(self) -> None:
        _require_nonempty("codec", self.codec)
        _require_nonempty("format_name", self.format_name)
        _require_nonempty("time_base", self.time_base)
        _optional_positive_int("width", self.width)
        _optional_positive_int("height", self.height)
        _optional_positive_float("average_fps", self.average_fps)
        _optional_positive_float("duration_seconds", self.duration_seconds)
        if self.bitrate_bps is not None:
            _optional_positive_int("bitrate_bps", self.bitrate_bps)
        if self.frame_count is not None:
            _optional_positive_int("frame_count", self.frame_count)

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "codec": self.codec,
            "format_name": self.format_name,
            "width": self.width,
            "height": self.height,
            "average_fps": self.average_fps,
            "duration_seconds": self.duration_seconds,
            "bitrate_bps": self.bitrate_bps,
            "frame_count": self.frame_count,
            "time_base": self.time_base,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VideoProbeSummary:
        allowed = {
            "codec",
            "format_name",
            "width",
            "height",
            "average_fps",
            "duration_seconds",
            "bitrate_bps",
            "frame_count",
            "time_base",
        }
        _reject_unknown_keys(payload, allowed, "video probe")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ModalityInterval:
    start_timestamp_ns: int
    end_timestamp_ns: int
    state: ModalityState

    def __post_init__(self) -> None:
        if self.start_timestamp_ns < 0:
            raise ValueError("modality interval start must be non-negative")
        if self.end_timestamp_ns <= self.start_timestamp_ns:
            raise ValueError("modality interval end must be after start")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "start_timestamp_ns": self.start_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModalityInterval:
        allowed = {"start_timestamp_ns", "end_timestamp_ns", "state"}
        _reject_unknown_keys(payload, allowed, "modality interval")
        return cls(
            start_timestamp_ns=payload["start_timestamp_ns"],
            end_timestamp_ns=payload["end_timestamp_ns"],
            state=ModalityState(payload["state"]),
        )


@dataclass(frozen=True, slots=True)
class RawVideoRecord:
    source_id: str
    source_uri: str
    source_sha256: str
    byte_size: int
    camera_id: str
    cage_id: str
    camera_setting_version: str
    recording_start_ns: int
    recording_end_ns: int
    probe: VideoProbeSummary
    modality_intervals: tuple[ModalityInterval, ...]

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "source_uri",
            "camera_id",
            "cage_id",
            "camera_setting_version",
        ):
            _require_nonempty(name, getattr(self, name))
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")
        _optional_positive_int("byte_size", self.byte_size)
        if self.recording_start_ns < 0:
            raise ValueError("recording_start_ns must be non-negative")
        if self.recording_end_ns <= self.recording_start_ns:
            raise ValueError("recording_end_ns must be after recording_start_ns")
        observed_duration_ns = self.recording_end_ns - self.recording_start_ns
        probed_duration_ns = round(self.probe.duration_seconds * 1_000_000_000)
        duration_tolerance_ns = max(
            1_000_000,
            round(1_000_000_000 / self.probe.average_fps),
        )
        if abs(observed_duration_ns - probed_duration_ns) > duration_tolerance_ns:
            raise ValueError("recording interval disagrees with probed duration")
        if not self.modality_intervals:
            raise ValueError("at least one modality interval is required")
        previous_end: int | None = None
        for index, interval in enumerate(self.modality_intervals):
            if not (
                self.recording_start_ns
                <= interval.start_timestamp_ns
                < interval.end_timestamp_ns
                <= self.recording_end_ns
            ):
                raise ValueError("modality interval lies outside the recording")
            if previous_end is not None and interval.start_timestamp_ns < previous_end:
                raise ValueError("modality intervals overlap or are unsorted")
            if previous_end is not None and interval.start_timestamp_ns > previous_end:
                raise ValueError("modality interval gaps must be labeled UNKNOWN")
            if index == 0 and interval.start_timestamp_ns != self.recording_start_ns:
                raise ValueError("modality intervals must cover recording start")
            previous_end = interval.end_timestamp_ns
        if previous_end != self.recording_end_ns:
            raise ValueError("modality intervals must cover recording end")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_uri": self.source_uri,
            "source_sha256": self.source_sha256,
            "byte_size": self.byte_size,
            "camera_id": self.camera_id,
            "cage_id": self.cage_id,
            "camera_setting_version": self.camera_setting_version,
            "recording_start_ns": self.recording_start_ns,
            "recording_end_ns": self.recording_end_ns,
            "probe": self.probe.to_dict(),
            "modality_intervals": [
                interval.to_dict() for interval in self.modality_intervals
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RawVideoRecord:
        allowed = {
            "source_id",
            "source_uri",
            "source_sha256",
            "byte_size",
            "camera_id",
            "cage_id",
            "camera_setting_version",
            "recording_start_ns",
            "recording_end_ns",
            "probe",
            "modality_intervals",
        }
        _reject_unknown_keys(payload, allowed, "raw video record")
        return cls(
            source_id=payload["source_id"],
            source_uri=payload["source_uri"],
            source_sha256=payload["source_sha256"],
            byte_size=payload["byte_size"],
            camera_id=payload["camera_id"],
            cage_id=payload["cage_id"],
            camera_setting_version=payload["camera_setting_version"],
            recording_start_ns=payload["recording_start_ns"],
            recording_end_ns=payload["recording_end_ns"],
            probe=VideoProbeSummary.from_dict(payload["probe"]),
            modality_intervals=tuple(
                ModalityInterval.from_dict(interval)
                for interval in payload["modality_intervals"]
            ),
        )


@dataclass(frozen=True, slots=True)
class AcquisitionManifest:
    """Content-addressed G0 inventory and protected-source index."""

    cameras: tuple[CameraSpecification, ...]
    videos: tuple[RawVideoRecord, ...]
    schema_version: str = "data.acquisition.v1"

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("at least one camera specification is required")
        _require_unique(
            tuple(
                (camera.camera_id, camera.camera_setting_version)
                for camera in self.cameras
            ),
            "camera ID/setting version",
        )
        _require_unique(
            tuple(video.source_id for video in self.videos),
            "source_id",
        )
        _require_unique(
            tuple(video.source_sha256 for video in self.videos),
            "source_sha256",
        )
        admitted_settings = {
            (camera.camera_id, camera.camera_setting_version)
            for camera in self.cameras
        }
        for video in self.videos:
            if (video.camera_id, video.camera_setting_version) not in admitted_settings:
                raise ValueError(
                    "video references an unknown camera ID/setting version"
                )

    def gate_blockers(
        self,
        *,
        minimum_cages: int = 3,
        minimum_contiguous_seconds_per_cage: float = 86_400.0,
    ) -> tuple[str, ...]:
        """Return deterministic G0 blockers without claiming sample quality."""

        _optional_positive_int("minimum_cages", minimum_cages)
        _optional_positive_float(
            "minimum_contiguous_seconds_per_cage",
            minimum_contiguous_seconds_per_cage,
        )
        blockers: list[str] = []
        for camera in self.cameras:
            blockers.extend(
                f"camera:{camera.camera_id}:{camera.camera_setting_version}:missing:{field}"
                for field in camera.missing_for_g0
            )

        cage_ids = sorted({video.cage_id for video in self.videos})
        if len(cage_ids) < minimum_cages:
            blockers.append(
                f"samples:distinct_cages:{len(cage_ids)}<{minimum_cages}"
            )
        required_duration_ns = round(
            minimum_contiguous_seconds_per_cage * 1_000_000_000
        )
        mechanism_by_setting = {
            (camera.camera_id, camera.camera_setting_version): camera.ir_mechanism
            for camera in self.cameras
        }
        for cage_id in cage_ids:
            cage_videos = tuple(
                video for video in self.videos if video.cage_id == cage_id
            )
            intervals = sorted(
                (
                    video.recording_start_ns,
                    video.recording_end_ns,
                )
                for video in cage_videos
            )
            if _maximum_contiguous_duration(intervals) < required_duration_ns:
                blockers.append(
                    f"samples:cage:{cage_id}:contiguous_duration_below_requirement"
                )
            mechanisms = {
                mechanism_by_setting[
                    (video.camera_id, video.camera_setting_version)
                ]
                for video in cage_videos
            }
            observed_modalities = {
                interval.state
                for video in cage_videos
                for interval in video.modality_intervals
            }
            required_modalities = {ModalityState.RGB, ModalityState.IR}
            if IRMechanism.DAY_NIGHT_SWITCHING in mechanisms:
                required_modalities.add(ModalityState.TRANSITION)
            for state in sorted(required_modalities, key=lambda item: item.value):
                if state not in observed_modalities:
                    blockers.append(
                        f"samples:cage:{cage_id}:missing_modality:{state.value}"
                    )
        return tuple(blockers)

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cameras": [camera.to_dict() for camera in self.cameras],
            "videos": [video.to_dict() for video in self.videos],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AcquisitionManifest:
        allowed = {"schema_version", "cameras", "videos"}
        _reject_unknown_keys(payload, allowed, "acquisition manifest")
        schema_version = payload.get("schema_version")
        if schema_version != "data.acquisition.v1":
            raise ValueError(f"unsupported acquisition schema: {schema_version!r}")
        cameras = payload.get("cameras")
        videos = payload.get("videos")
        if not isinstance(cameras, list) or not isinstance(videos, list):
            raise ValueError("cameras and videos must be lists")
        return cls(
            cameras=tuple(CameraSpecification.from_dict(item) for item in cameras),
            videos=tuple(RawVideoRecord.from_dict(item) for item in videos),
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True)
class TimestampAudit:
    observed_frames: int
    unavailable_timestamps: int
    first_timestamp_ns: int | None
    last_timestamp_ns: int | None
    inversions: int
    duplicates: int
    estimated_missing_frames: int
    maximum_forward_gap_ns: int


class TimestampAuditAccumulator:
    """O(1)-memory audit for decoded or packet timestamps."""

    __slots__ = (
        "_duplicates",
        "_estimated_missing",
        "_expected_period_ns",
        "_first",
        "_inversions",
        "_last",
        "_maximum_gap",
        "_observed",
        "_unavailable",
    )

    def __init__(self, expected_period_ns: int) -> None:
        _optional_positive_int("expected_period_ns", expected_period_ns)
        self._expected_period_ns = expected_period_ns
        self._observed = 0
        self._unavailable = 0
        self._first: int | None = None
        self._last: int | None = None
        self._inversions = 0
        self._duplicates = 0
        self._estimated_missing = 0
        self._maximum_gap = 0

    def observe(self, timestamp_ns: int) -> None:
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise TypeError("timestamp_ns must be an integer")
        if self._first is None:
            self._first = timestamp_ns
        if self._last is not None:
            gap = timestamp_ns - self._last
            if gap < 0:
                self._inversions += 1
            elif gap == 0:
                self._duplicates += 1
            else:
                self._maximum_gap = max(self._maximum_gap, gap)
                nominal_periods = (gap + self._expected_period_ns // 2) // (
                    self._expected_period_ns
                )
                self._estimated_missing += max(0, nominal_periods - 1)
        self._last = timestamp_ns
        self._observed += 1

    def observe_unavailable(self) -> None:
        self._unavailable += 1

    def snapshot(self) -> TimestampAudit:
        return TimestampAudit(
            observed_frames=self._observed,
            unavailable_timestamps=self._unavailable,
            first_timestamp_ns=self._first,
            last_timestamp_ns=self._last,
            inversions=self._inversions,
            duplicates=self._duplicates,
            estimated_missing_frames=self._estimated_missing,
            maximum_forward_gap_ns=self._maximum_gap,
        )


def audit_timestamp_lines(
    lines: Iterable[str],
    *,
    expected_fps: float,
) -> TimestampAudit:
    """Consume one ffprobe CSV timestamp per line in constant memory."""

    _optional_positive_float("expected_fps", expected_fps)
    expected_period_ns = round(1_000_000_000 / expected_fps)
    audit = TimestampAuditAccumulator(expected_period_ns)
    for line in lines:
        token = line.strip().split(",", maxsplit=1)[0].strip()
        if not token or token.upper() == "N/A":
            audit.observe_unavailable()
            continue
        try:
            timestamp_ns = int(
                (Decimal(token) * Decimal(1_000_000_000)).to_integral_value(
                    rounding=ROUND_HALF_UP
                )
            )
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"invalid ffprobe timestamp: {token!r}") from error
        audit.observe(timestamp_ns)
    return audit.snapshot()


def parse_ffprobe(payload: dict[str, Any]) -> VideoProbeSummary:
    """Parse the JSON emitted by the local ffprobe stream/format query."""

    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ValueError("ffprobe payload has no streams list")
    video = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        None,
    )
    if video is None:
        raise ValueError("ffprobe payload has no video stream")
    format_info = payload.get("format")
    if not isinstance(format_info, dict):
        raise ValueError("ffprobe payload has no format object")

    frame_rate_text = video.get("avg_frame_rate") or video.get("r_frame_rate")
    try:
        frame_rate = float(Fraction(str(frame_rate_text)))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid ffprobe frame rate") from error
    duration_text = video.get("duration") or format_info.get("duration")
    try:
        duration_seconds = float(duration_text)
    except (TypeError, ValueError) as error:
        raise ValueError("ffprobe duration is unavailable") from error
    bitrate_text = video.get("bit_rate") or format_info.get("bit_rate")
    frame_count_text = video.get("nb_frames")

    return VideoProbeSummary(
        codec=str(video["codec_name"]),
        format_name=str(format_info["format_name"]),
        width=int(video["width"]),
        height=int(video["height"]),
        average_fps=frame_rate,
        duration_seconds=duration_seconds,
        bitrate_bps=int(bitrate_text) if bitrate_text is not None else None,
        frame_count=int(frame_count_text) if frame_count_text is not None else None,
        time_base=str(video["time_base"]),
    )


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a source in bounded memory without modifying it."""

    _optional_positive_int("chunk_bytes", chunk_bytes)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video_file(path: Path, timeout_seconds: float = 60.0) -> VideoProbeSummary:
    """Run a read-only ffprobe query against one local source."""

    if not path.is_file():
        raise FileNotFoundError(path)
    _optional_positive_float("timeout_seconds", timeout_seconds)
    command = (
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return parse_ffprobe(json.loads(completed.stdout))


def _require_nonempty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _optional_positive_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _optional_positive_float(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_unique(values: tuple[Any, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} values must be unique")


def _maximum_contiguous_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    maximum = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            maximum = max(maximum, current_end - current_start)
            current_start, current_end = start, end
    return max(maximum, current_end - current_start)


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
