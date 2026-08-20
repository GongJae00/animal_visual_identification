"""Strict observation and role contracts for the common Full segment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from shared.foundation.provenance import content_sha256

OBSERVATION_SCHEMA = "parsing.full_segment_observation.v1"
ROLE_SCHEMA = "parsing.full_segment_role.v1"
ASSOCIATION_SCHEMA = "parsing.full_segment_association.v1"
BODY_MASK_POLICY_SCHEMA = "parsing.full_segment_body_mask_policy.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SourceViewScope(str, Enum):
    BODY_AVAILABLE = "BODY_AVAILABLE"
    BODY_TRUNCATED = "BODY_TRUNCATED"
    FACE_NATIVE = "FACE_NATIVE"
    HEAD_NATIVE = "HEAD_NATIVE"
    AMBIGUOUS = "AMBIGUOUS"
    UNAVAILABLE = "UNAVAILABLE"


class FullStatus(str, Enum):
    USABLE = "USABLE"
    REVIEW = "REVIEW"
    UNUSABLE = "UNUSABLE"
    AMBIGUOUS = "AMBIGUOUS"


class TerminalObservability(str, Enum):
    NOT_RUN = "NOT_RUN"
    NOT_DETECTED = "NOT_DETECTED"
    REVIEW = "REVIEW"
    USABLE = "USABLE"
    NATIVE = "NATIVE"


class SegmentRole(str, Enum):
    FULL = "FULL"
    FACE = "FACE"
    NOSE = "NOSE"


class ObservationRoute(str, Enum):
    BODY_PARSING = "BODY_PARSING"
    BODY_MASK = "BODY_MASK"
    NATIVE_FACE = "NATIVE_FACE"
    NATIVE_HEAD = "NATIVE_HEAD"
    NONE = "NONE"


class AssociationKind(str, Enum):
    EXACTLY_ONE = "EXACTLY_ONE"
    AUTHORITATIVE = "AUTHORITATIVE"


class BodyMaskPolicyKind(str, Enum):
    OXFORD_IIIT_PET_TRIMAP = "OXFORD_IIIT_PET_TRIMAP"


@dataclass(frozen=True, slots=True)
class BodyMaskPolicy:
    kind: BodyMaskPolicyKind
    permitted_labels: tuple[int, ...] | None = None
    schema_version: str = BODY_MASK_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != BODY_MASK_POLICY_SCHEMA:
            raise ValueError("body mask policy schema differs")
        if self.kind is not BodyMaskPolicyKind.OXFORD_IIIT_PET_TRIMAP:
            raise ValueError("body mask policy kind differs")
        labels = self.permitted_labels
        if labels is not None and (
            not isinstance(labels, tuple)
            or not labels
            or any(isinstance(label, bool) or not isinstance(label, int) for label in labels)
            or tuple(sorted(set(labels))) != labels
            or not set(labels).issubset({1, 2, 3})
            or 1 not in labels
        ):
            raise ValueError("body mask permitted labels differ")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "foreground_labels": [1],
            "excluded_labels": [2, 3],
            "permitted_labels": (
                None if self.permitted_labels is None else list(self.permitted_labels)
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "policy_sha256": self.policy_sha256}

    @classmethod
    def from_dict(cls, value: object) -> BodyMaskPolicy:
        fields = {
            "schema_version",
            "kind",
            "foreground_labels",
            "excluded_labels",
            "permitted_labels",
            "policy_sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("body mask policy record schema differs")
        _require_sha256(value["policy_sha256"], "body mask policy")
        for field, expected in (
            ("foreground_labels", [1]),
            ("excluded_labels", [2, 3]),
        ):
            labels = value[field]
            if (
                not isinstance(labels, list)
                or any(type(label) is not int for label in labels)
                or labels != expected
            ):
                raise ValueError("body mask policy label semantics differ")
        payload = {
            key: item for key, item in value.items() if key != "policy_sha256"
        }
        if content_sha256(payload) != value["policy_sha256"]:
            raise ValueError("body mask policy digest differs")
        permitted = value["permitted_labels"]
        if permitted is not None and not isinstance(permitted, list):
            raise TypeError("body mask permitted labels must be an array or null")
        try:
            policy = cls(
                schema_version=value["schema_version"],
                kind=BodyMaskPolicyKind(value["kind"]),
                permitted_labels=None if permitted is None else tuple(permitted),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("body mask policy values differ") from exc
        if value != policy.to_dict():
            raise ValueError("body mask policy content or digest differs")
        return policy


@dataclass(frozen=True, slots=True)
class AnimalAssociation:
    kind: AssociationKind
    instance_index: int
    authority_sha256: str | None = None
    schema_version: str = ASSOCIATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ASSOCIATION_SCHEMA:
            raise ValueError("Full segment association schema differs")
        if (
            isinstance(self.instance_index, bool)
            or not isinstance(self.instance_index, int)
            or self.instance_index < 0
        ):
            raise ValueError("Full segment association index must be non-negative")
        if self.kind is AssociationKind.EXACTLY_ONE:
            if self.instance_index != 0 or self.authority_sha256 is not None:
                raise ValueError("exactly-one association must select sole index zero")
        elif self.authority_sha256 is None:
            raise ValueError("authoritative association requires an authority digest")
        if self.authority_sha256 is not None:
            _require_sha256(self.authority_sha256, "association authority")

    def validate_instance_count(self, instance_count: int) -> None:
        if (
            isinstance(instance_count, bool)
            or not isinstance(instance_count, int)
            or instance_count <= 0
            or self.instance_index >= instance_count
        ):
            raise ValueError("Full segment association does not select a prediction")
        if self.kind is AssociationKind.EXACTLY_ONE and instance_count != 1:
            raise ValueError("exactly-one association requires exactly one prediction")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "instance_index": self.instance_index,
            "authority_sha256": self.authority_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> AnimalAssociation:
        fields = {
            "schema_version",
            "kind",
            "instance_index",
            "authority_sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("Full segment association record schema differs")
        try:
            kind = AssociationKind(value["kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Full segment association kind differs") from exc
        return cls(
            schema_version=value["schema_version"],
            kind=kind,
            instance_index=value["instance_index"],
            authority_sha256=value["authority_sha256"],
        )


@dataclass(frozen=True, slots=True)
class SegmentRoleRecord:
    role: SegmentRole
    status: str
    route: ObservationRoute
    producer_sha256: str | None
    artifact_sha256: str | None
    schema_version: str = ROLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ROLE_SCHEMA:
            raise ValueError("Full segment role schema differs")
        if self.role is SegmentRole.FULL:
            try:
                status = FullStatus(self.status)
            except ValueError as exc:
                raise ValueError("Full role status differs") from exc
            if status in {FullStatus.USABLE, FullStatus.REVIEW}:
                if self.route is ObservationRoute.BODY_PARSING:
                    if self.producer_sha256 is None:
                        raise ValueError("parsed Full role requires parser provenance")
                elif self.route is ObservationRoute.BODY_MASK:
                    if self.producer_sha256 is None or self.artifact_sha256 is None:
                        raise ValueError(
                            "body-mask Full role requires policy and crop provenance"
                        )
                elif self.route in {
                    ObservationRoute.NATIVE_FACE,
                    ObservationRoute.NATIVE_HEAD,
                }:
                    if self.producer_sha256 is not None or self.artifact_sha256 is None:
                        raise ValueError("native Full role requires only a bound crop artifact")
                else:
                    raise ValueError("observable Full role requires a materialization route")
            elif self.artifact_sha256 is not None:
                raise ValueError("unobservable Full role cannot bind an artifact")
        else:
            try:
                status = TerminalObservability(self.status)
            except ValueError as exc:
                raise ValueError("Face/Nose role observability differs") from exc
            if status is TerminalObservability.NATIVE:
                if self.route not in {
                    ObservationRoute.NATIVE_FACE,
                    ObservationRoute.NATIVE_HEAD,
                }:
                    raise ValueError("native role requires an explicit native route")
                if self.artifact_sha256 is None:
                    raise ValueError("native role requires a content-bound artifact")
            elif self.role is SegmentRole.FACE and self.route in {
                ObservationRoute.NATIVE_FACE,
                ObservationRoute.NATIVE_HEAD,
            }:
                raise ValueError("native route must retain native Face observability")
            elif self.route is ObservationRoute.BODY_MASK and (
                self.producer_sha256 is not None or self.artifact_sha256 is not None
            ):
                raise ValueError("body-mask Face/Nose roles cannot bind region evidence")
        for value, label in (
            (self.producer_sha256, "role producer"),
            (self.artifact_sha256, "role artifact"),
        ):
            if value is not None:
                _require_sha256(value, label)

    @property
    def role_sha256(self) -> str:
        return content_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role.value,
            "status": self.status,
            "route": self.route.value,
            "producer_sha256": self.producer_sha256,
            "artifact_sha256": self.artifact_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "role_sha256": self.role_sha256}

    @classmethod
    def from_dict(cls, value: object) -> SegmentRoleRecord:
        fields = {
            "schema_version",
            "role",
            "status",
            "route",
            "producer_sha256",
            "artifact_sha256",
            "role_sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("Full segment role record schema differs")
        try:
            record = cls(
                schema_version=value["schema_version"],
                role=SegmentRole(value["role"]),
                status=value["status"],
                route=ObservationRoute(value["route"]),
                producer_sha256=value["producer_sha256"],
                artifact_sha256=value["artifact_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Full segment role record values differ") from exc
        if value["role_sha256"] != record.role_sha256:
            raise ValueError("Full segment role record digest differs")
        return record


@dataclass(frozen=True, slots=True)
class FullSegmentObservation:
    source_id: str
    source_sha256: str
    source_width: int
    source_height: int
    source_view_scope: SourceViewScope
    full_status: FullStatus
    face_observability: TerminalObservability
    nose_observability: TerminalObservability
    route: ObservationRoute
    parsing_prediction_sha256: str | None
    association: AnimalAssociation | None
    authoritative_mask_sha256: str | None
    mask_policy_sha256: str | None
    roles: tuple[SegmentRoleRecord, ...]
    schema_version: str = OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != OBSERVATION_SCHEMA:
            raise ValueError("Full segment observation schema differs")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("Full segment source ID must be non-empty")
        _require_sha256(self.source_sha256, "Full segment source")
        for value, label in (
            (self.source_width, "source width"),
            (self.source_height, "source height"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Full segment {label} must be positive")
        if tuple(record.role for record in self.roles) != tuple(SegmentRole):
            raise ValueError("Full segment roles must contain FULL, FACE, NOSE in order")
        expected_states = (
            self.full_status.value,
            self.face_observability.value,
            self.nose_observability.value,
        )
        if tuple(record.status for record in self.roles) != expected_states:
            raise ValueError("Full segment role states differ from observation")
        if any(record.route is not self.route for record in self.roles):
            raise ValueError("Full segment role routes differ from observation")
        self._validate_route()

    def _validate_route(self) -> None:
        body_scopes = {
            SourceViewScope.BODY_AVAILABLE,
            SourceViewScope.BODY_TRUNCATED,
        }
        if self.route is ObservationRoute.BODY_MASK:
            if self.source_view_scope not in body_scopes:
                raise ValueError("body mask route requires body source scope")
            if self.parsing_prediction_sha256 is not None or self.association is not None:
                raise ValueError("body mask route cannot claim whole-body parsing")
            if (
                self.authoritative_mask_sha256 is None
                or self.mask_policy_sha256 is None
            ):
                raise ValueError("body mask route requires mask and policy provenance")
            _require_sha256(self.authoritative_mask_sha256, "authoritative body mask")
            _require_sha256(self.mask_policy_sha256, "body mask policy")
            expected_status = (
                FullStatus.REVIEW
                if self.source_view_scope is SourceViewScope.BODY_TRUNCATED
                else FullStatus.USABLE
            )
            if self.full_status is not expected_status:
                raise ValueError("body mask Full status differs from source scope")
            if TerminalObservability.NATIVE in {
                self.face_observability,
                self.nose_observability,
            }:
                raise ValueError("body mask route cannot claim native Face/Nose evidence")
            if self.roles[0].producer_sha256 != self.mask_policy_sha256:
                raise ValueError("body mask role policy differs from observation")
            return
        if (
            self.authoritative_mask_sha256 is not None
            or self.mask_policy_sha256 is not None
        ):
            raise ValueError("non-mask route cannot claim authoritative mask provenance")
        if self.route is ObservationRoute.BODY_PARSING:
            if self.source_view_scope not in body_scopes:
                raise ValueError("body parsing route requires body source scope")
            if self.parsing_prediction_sha256 is None or self.association is None:
                raise ValueError("body parsing route requires frozen parsing and association")
            _require_sha256(self.parsing_prediction_sha256, "frozen parsing prediction")
            if self.full_status is FullStatus.AMBIGUOUS:
                raise ValueError("body parsing route cannot claim ambiguous Full status")
            if self.face_observability is TerminalObservability.NATIVE or self.nose_observability is TerminalObservability.NATIVE:
                raise ValueError("body parsing route cannot claim native Face/Nose evidence")
            return
        if self.parsing_prediction_sha256 is not None or self.association is not None:
            raise ValueError("non-body route cannot claim whole-body parsing")
        if self.route is ObservationRoute.NATIVE_FACE:
            if self.source_view_scope is not SourceViewScope.FACE_NATIVE:
                raise ValueError("native face route requires FACE_NATIVE scope")
            if self.face_observability is not TerminalObservability.NATIVE:
                raise ValueError("native face route requires native Face observability")
            if self.full_status is not FullStatus.USABLE:
                raise ValueError("native face route requires usable Full appearance")
            return
        if self.route is ObservationRoute.NATIVE_HEAD:
            if self.source_view_scope is not SourceViewScope.HEAD_NATIVE:
                raise ValueError("native head route requires HEAD_NATIVE scope")
            if self.face_observability is not TerminalObservability.NATIVE:
                raise ValueError("native head route requires native Face observability")
            if self.full_status is not FullStatus.USABLE:
                raise ValueError("native head route requires usable Full appearance")
            return
        if self.source_view_scope not in {
            SourceViewScope.AMBIGUOUS,
            SourceViewScope.UNAVAILABLE,
        }:
            raise ValueError("route NONE requires ambiguous or unavailable scope")
        expected = (
            FullStatus.AMBIGUOUS
            if self.source_view_scope is SourceViewScope.AMBIGUOUS
            else FullStatus.UNUSABLE
        )
        if self.full_status is not expected:
            raise ValueError("route NONE Full status differs from source scope")
        if TerminalObservability.NATIVE in {
            self.face_observability,
            self.nose_observability,
        }:
            raise ValueError("route NONE cannot claim native evidence")

    @property
    def observation_sha256(self) -> str:
        return content_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_view_scope": self.source_view_scope.value,
            "full_status": self.full_status.value,
            "face_observability": self.face_observability.value,
            "nose_observability": self.nose_observability.value,
            "route": self.route.value,
            "parsing_prediction_sha256": self.parsing_prediction_sha256,
            "association": None if self.association is None else self.association.to_dict(),
            "authoritative_mask_sha256": self.authoritative_mask_sha256,
            "mask_policy_sha256": self.mask_policy_sha256,
            "roles": [record.to_dict() for record in self.roles],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "observation_sha256": self.observation_sha256}

    @classmethod
    def from_dict(cls, value: object) -> FullSegmentObservation:
        fields = {
            "schema_version",
            "source_id",
            "source_sha256",
            "source_width",
            "source_height",
            "source_view_scope",
            "full_status",
            "face_observability",
            "nose_observability",
            "route",
            "parsing_prediction_sha256",
            "association",
            "authoritative_mask_sha256",
            "mask_policy_sha256",
            "roles",
            "observation_sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("Full segment observation record schema differs")
        raw_roles = value["roles"]
        if not isinstance(raw_roles, list):
            raise TypeError("Full segment roles must be an array")
        try:
            observation = cls(
                schema_version=value["schema_version"],
                source_id=value["source_id"],
                source_sha256=value["source_sha256"],
                source_width=value["source_width"],
                source_height=value["source_height"],
                source_view_scope=SourceViewScope(value["source_view_scope"]),
                full_status=FullStatus(value["full_status"]),
                face_observability=TerminalObservability(value["face_observability"]),
                nose_observability=TerminalObservability(value["nose_observability"]),
                route=ObservationRoute(value["route"]),
                parsing_prediction_sha256=value["parsing_prediction_sha256"],
                association=(
                    None
                    if value["association"] is None
                    else AnimalAssociation.from_dict(value["association"])
                ),
                authoritative_mask_sha256=value["authoritative_mask_sha256"],
                mask_policy_sha256=value["mask_policy_sha256"],
                roles=tuple(SegmentRoleRecord.from_dict(item) for item in raw_roles),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Full segment observation values differ") from exc
        if value["observation_sha256"] != observation.observation_sha256:
            raise ValueError("Full segment observation digest differs")
        return observation


def build_native_observation(
    *,
    source_id: str,
    source_sha256: str,
    source_width: int,
    source_height: int,
    source_view_scope: SourceViewScope,
    native_artifact_sha256: str,
    full_rgb_sha256: str,
    nose_observability: TerminalObservability = TerminalObservability.NOT_RUN,
) -> FullSegmentObservation:
    if source_view_scope is SourceViewScope.FACE_NATIVE:
        route = ObservationRoute.NATIVE_FACE
    elif source_view_scope is SourceViewScope.HEAD_NATIVE:
        route = ObservationRoute.NATIVE_HEAD
    else:
        raise ValueError("native observation requires FACE_NATIVE or HEAD_NATIVE scope")
    roles = (
        SegmentRoleRecord(
            SegmentRole.FULL,
            FullStatus.USABLE.value,
            route,
            None,
            full_rgb_sha256,
        ),
        SegmentRoleRecord(
            SegmentRole.FACE,
            TerminalObservability.NATIVE.value,
            route,
            None,
            native_artifact_sha256,
        ),
        SegmentRoleRecord(
            SegmentRole.NOSE,
            nose_observability.value,
            route,
            None,
            native_artifact_sha256
            if nose_observability is TerminalObservability.NATIVE
            else None,
        ),
    )
    return FullSegmentObservation(
        source_id=source_id,
        source_sha256=source_sha256,
        source_width=source_width,
        source_height=source_height,
        source_view_scope=source_view_scope,
        full_status=FullStatus.USABLE,
        face_observability=TerminalObservability.NATIVE,
        nose_observability=nose_observability,
        route=route,
        parsing_prediction_sha256=None,
        association=None,
        authoritative_mask_sha256=None,
        mask_policy_sha256=None,
        roles=roles,
    )


def build_terminal_observation(
    *,
    source_id: str,
    source_sha256: str,
    source_width: int,
    source_height: int,
    source_view_scope: SourceViewScope,
    face_observability: TerminalObservability,
    nose_observability: TerminalObservability,
) -> FullSegmentObservation:
    if source_view_scope not in {
        SourceViewScope.AMBIGUOUS,
        SourceViewScope.UNAVAILABLE,
    }:
        raise ValueError("terminal observation requires ambiguous or unavailable scope")
    full_status = (
        FullStatus.AMBIGUOUS
        if source_view_scope is SourceViewScope.AMBIGUOUS
        else FullStatus.UNUSABLE
    )
    roles = tuple(
        SegmentRoleRecord(role, status, ObservationRoute.NONE, None, None)
        for role, status in (
            (SegmentRole.FULL, full_status.value),
            (SegmentRole.FACE, face_observability.value),
            (SegmentRole.NOSE, nose_observability.value),
        )
    )
    return FullSegmentObservation(
        source_id=source_id,
        source_sha256=source_sha256,
        source_width=source_width,
        source_height=source_height,
        source_view_scope=source_view_scope,
        full_status=full_status,
        face_observability=face_observability,
        nose_observability=nose_observability,
        route=ObservationRoute.NONE,
        parsing_prediction_sha256=None,
        association=None,
        authoritative_mask_sha256=None,
        mask_policy_sha256=None,
        roles=roles,
    )


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


__all__ = [
    "ASSOCIATION_SCHEMA",
    "BODY_MASK_POLICY_SCHEMA",
    "OBSERVATION_SCHEMA",
    "ROLE_SCHEMA",
    "AnimalAssociation",
    "AssociationKind",
    "BodyMaskPolicy",
    "BodyMaskPolicyKind",
    "FullSegmentObservation",
    "FullStatus",
    "ObservationRoute",
    "SegmentRole",
    "SegmentRoleRecord",
    "SourceViewScope",
    "TerminalObservability",
    "build_native_observation",
    "build_terminal_observation",
]
