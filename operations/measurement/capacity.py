"""First-order event-driven compute and peak-memory capacity models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import floor, inf, isfinite, nextafter
from typing import Any

from shared.foundation.provenance import content_sha256


class ComputeResource(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    VIDEO_DECODE = "video_decode"


@dataclass(frozen=True, slots=True)
class StageRateBudget:
    """Measured invocation rates and service time for one pipeline stage."""

    name: str
    resource: ComputeResource
    idle_calls_per_stream_second: float
    occupied_calls_per_stream_second: float
    service_seconds_per_call: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stage name must be non-empty")
        _validate_nonnegative_finite(
            self.idle_calls_per_stream_second,
            self.occupied_calls_per_stream_second,
            self.service_seconds_per_call,
        )

    def blended_rate(self, occupied_fraction: float) -> float:
        _validate_fraction(occupied_fraction, "occupied_fraction")
        return (
            (1.0 - occupied_fraction) * self.idle_calls_per_stream_second
            + occupied_fraction * self.occupied_calls_per_stream_second
        )

    @property
    def peak_state_rate(self) -> float:
        return max(
            self.idle_calls_per_stream_second,
            self.occupied_calls_per_stream_second,
        )

    def to_dict(self) -> dict[str, str | float]:
        return {
            "name": self.name,
            "resource": self.resource.value,
            "idle_calls_per_stream_second": (
                self.idle_calls_per_stream_second
            ),
            "occupied_calls_per_stream_second": (
                self.occupied_calls_per_stream_second
            ),
            "service_seconds_per_call": self.service_seconds_per_call,
        }


@dataclass(frozen=True, slots=True)
class ResourceCapacity:
    resource: ComputeResource
    parallel_service_units: float
    target_utilization: float

    def __post_init__(self) -> None:
        _validate_positive_finite(self.parallel_service_units)
        _validate_fraction(self.target_utilization, "target_utilization")
        if self.target_utilization == 0:
            raise ValueError("target_utilization must be positive")

    def to_dict(self) -> dict[str, str | float]:
        return {
            "resource": self.resource.value,
            "parallel_service_units": self.parallel_service_units,
            "target_utilization": self.target_utilization,
        }


@dataclass(frozen=True, slots=True)
class ResourceLoad:
    resource: ComputeResource
    expected_utilization: float
    peak_state_utilization: float
    target_utilization: float
    expected_within_target: bool
    peak_state_within_target: bool
    maximum_cameras_at_expected_mix: int | None
    maximum_cameras_at_peak_state: int | None

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return {
            field_name: (
                value.value if isinstance(value, StrEnum) else value
            )
            for field_name, value in (
                (name, getattr(self, name))
                for name in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class CapacityPlan:
    camera_count: int
    occupied_fraction: float
    stages: tuple[StageRateBudget, ...]
    capacities: tuple[ResourceCapacity, ...]

    def __post_init__(self) -> None:
        _validate_positive_int(self.camera_count, "camera_count")
        _validate_fraction(self.occupied_fraction, "occupied_fraction")
        if not self.stages:
            raise ValueError("at least one stage is required")
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError("stage names must be unique")
        capacity_resources = tuple(item.resource for item in self.capacities)
        if len(set(capacity_resources)) != len(capacity_resources):
            raise ValueError("resource capacities must be unique")
        missing = {stage.resource for stage in self.stages} - set(
            capacity_resources
        )
        if missing:
            raise ValueError(f"missing capacity for resources: {sorted(missing)}")

    @property
    def config_sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CapacityPlan:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "camera_count",
                "occupied_fraction",
                "stages",
                "capacities",
            },
            "capacity plan",
        )
        if payload["schema_version"] != "cvi.capacity_plan.v1":
            raise ValueError("unsupported capacity plan schema version")
        stages_payload = payload["stages"]
        capacities_payload = payload["capacities"]
        if not isinstance(stages_payload, list):
            raise TypeError("stages must be a list")
        if not isinstance(capacities_payload, list):
            raise TypeError("capacities must be a list")
        stages: list[StageRateBudget] = []
        for item in stages_payload:
            if not isinstance(item, dict):
                raise TypeError("each stage must be an object")
            _require_exact_keys(
                item,
                {
                    "name",
                    "resource",
                    "idle_calls_per_stream_second",
                    "occupied_calls_per_stream_second",
                    "service_seconds_per_call",
                },
                "stage",
            )
            stages.append(
                StageRateBudget(
                    name=item["name"],
                    resource=ComputeResource(item["resource"]),
                    idle_calls_per_stream_second=(
                        item["idle_calls_per_stream_second"]
                    ),
                    occupied_calls_per_stream_second=(
                        item["occupied_calls_per_stream_second"]
                    ),
                    service_seconds_per_call=item["service_seconds_per_call"],
                )
            )
        capacities: list[ResourceCapacity] = []
        for item in capacities_payload:
            if not isinstance(item, dict):
                raise TypeError("each capacity must be an object")
            _require_exact_keys(
                item,
                {
                    "resource",
                    "parallel_service_units",
                    "target_utilization",
                },
                "resource capacity",
            )
            capacities.append(
                ResourceCapacity(
                    resource=ComputeResource(item["resource"]),
                    parallel_service_units=item["parallel_service_units"],
                    target_utilization=item["target_utilization"],
                )
            )
        return cls(
            camera_count=payload["camera_count"],
            occupied_fraction=payload["occupied_fraction"],
            stages=tuple(stages),
            capacities=tuple(capacities),
        )

    def expected_stage_calls(self, duration_seconds: float) -> dict[str, float]:
        _validate_nonnegative_finite(duration_seconds)
        return {
            stage.name: (
                self.camera_count
                * duration_seconds
                * stage.blended_rate(self.occupied_fraction)
            )
            for stage in self.stages
        }

    def peak_state_stage_calls(
        self, duration_seconds: float
    ) -> dict[str, float]:
        _validate_nonnegative_finite(duration_seconds)
        return {
            stage.name: (
                self.camera_count
                * duration_seconds
                * stage.peak_state_rate
            )
            for stage in self.stages
        }

    def resource_loads(self) -> tuple[ResourceLoad, ...]:
        loads: list[ResourceLoad] = []
        for capacity in self.capacities:
            stages = tuple(
                stage
                for stage in self.stages
                if stage.resource is capacity.resource
            )
            expected_demand_per_camera = sum(
                stage.blended_rate(self.occupied_fraction)
                * stage.service_seconds_per_call
                for stage in stages
            )
            peak_demand_per_camera = sum(
                stage.peak_state_rate * stage.service_seconds_per_call
                for stage in stages
            )
            expected_utilization = (
                self.camera_count
                * expected_demand_per_camera
                / capacity.parallel_service_units
            )
            peak_utilization = (
                self.camera_count
                * peak_demand_per_camera
                / capacity.parallel_service_units
            )
            loads.append(
                ResourceLoad(
                    resource=capacity.resource,
                    expected_utilization=expected_utilization,
                    peak_state_utilization=peak_utilization,
                    target_utilization=capacity.target_utilization,
                    expected_within_target=(
                        expected_utilization <= capacity.target_utilization
                    ),
                    peak_state_within_target=(
                        peak_utilization <= capacity.target_utilization
                    ),
                    maximum_cameras_at_expected_mix=_maximum_cameras(
                        capacity, expected_demand_per_camera
                    ),
                    maximum_cameras_at_peak_state=_maximum_cameras(
                        capacity, peak_demand_per_camera
                    ),
                )
            )
        return tuple(loads)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cvi.capacity_plan.v1",
            "camera_count": self.camera_count,
            "occupied_fraction": self.occupied_fraction,
            "stages": [stage.to_dict() for stage in self.stages],
            "capacities": [
                capacity.to_dict() for capacity in self.capacities
            ],
        }


@dataclass(frozen=True, slots=True)
class MemoryComponent:
    name: str
    shared_bytes: int = 0
    per_stream_bytes: int = 0
    per_active_track_bytes: int = 0
    per_batch_item_bytes: int = 0
    workspace_bytes_per_replica: int = 0
    workspace_replicas: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("memory component name must be non-empty")
        _validate_nonnegative_int(
            self.shared_bytes,
            self.per_stream_bytes,
            self.per_active_track_bytes,
            self.per_batch_item_bytes,
            self.workspace_bytes_per_replica,
            self.workspace_replicas,
        )

    def peak_bytes(
        self,
        *,
        stream_count: int,
        active_tracks: int,
        batch_items: int,
    ) -> int:
        _validate_nonnegative_int(stream_count, active_tracks, batch_items)
        return (
            self.shared_bytes
            + stream_count * self.per_stream_bytes
            + active_tracks * self.per_active_track_bytes
            + batch_items * self.per_batch_item_bytes
            + self.workspace_replicas * self.workspace_bytes_per_replica
        )


def peak_memory_bytes(
    components: tuple[MemoryComponent, ...],
    *,
    stream_count: int,
    active_tracks: int,
    batch_items: int,
) -> int:
    """Add preclassified terms; callers must not double-count shared memory."""

    if not components:
        raise ValueError("at least one memory component is required")
    if len({component.name for component in components}) != len(components):
        raise ValueError("memory component names must be unique")
    return sum(
        component.peak_bytes(
            stream_count=stream_count,
            active_tracks=active_tracks,
            batch_items=batch_items,
        )
        for component in components
    )


def adaptive_call_reduction_fraction(
    *,
    idle_calls_per_second: float,
    occupied_calls_per_second: float,
    occupied_fraction: float,
) -> float:
    """Reduction versus always running at the larger declared state rate."""

    _validate_nonnegative_finite(
        idle_calls_per_second, occupied_calls_per_second
    )
    _validate_fraction(occupied_fraction, "occupied_fraction")
    fixed_rate = max(idle_calls_per_second, occupied_calls_per_second)
    if fixed_rate == 0:
        return 0.0
    adaptive_rate = (
        (1 - occupied_fraction) * idle_calls_per_second
        + occupied_fraction * occupied_calls_per_second
    )
    return 1.0 - adaptive_rate / fixed_rate


def _maximum_cameras(
    capacity: ResourceCapacity,
    demand_per_camera: float,
) -> int | None:
    if demand_per_camera == 0:
        return None
    ratio = (
        capacity.target_utilization
        * capacity.parallel_service_units
        / demand_per_camera
    )
    return floor(nextafter(ratio, inf))


def _validate_fraction(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _validate_positive_finite(*values: float) -> None:
    _validate_nonnegative_finite(*values)
    if any(value == 0 for value in values):
        raise ValueError("values must be positive")


def _validate_nonnegative_finite(*values: float) -> None:
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("values must be finite and non-negative")


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_int(*values: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("memory dimensions must be integers")
    if any(value < 0 for value in values):
        raise ValueError("memory dimensions must be non-negative")


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
