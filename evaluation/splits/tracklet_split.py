"""Tracklet metadata and leakage-resistant open-set split contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from shared.contracts.contracts import Modality
from shared.foundation.provenance import content_sha256


class EvaluationStage(StrEnum):
    TRAINING = "TRAINING"
    CALIBRATION = "CALIBRATION"
    TEST = "TEST"


class PresenceState(StrEnum):
    UNKNOWN = "UNKNOWN"
    ABSENT = "ABSENT"
    PRESENT = "PRESENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SplitRole(StrEnum):
    TRAIN = "TRAIN"
    DEVELOPMENT = "DEVELOPMENT"
    CALIBRATION_GALLERY = "CALIBRATION_GALLERY"
    CALIBRATION_KNOWN_QUERY = "CALIBRATION_KNOWN_QUERY"
    CALIBRATION_UNKNOWN_QUERY = "CALIBRATION_UNKNOWN_QUERY"
    TEST_GALLERY = "TEST_GALLERY"
    TEST_KNOWN_QUERY = "TEST_KNOWN_QUERY"
    TEST_UNKNOWN_QUERY = "TEST_UNKNOWN_QUERY"

    @property
    def stage(self) -> EvaluationStage:
        if self in {SplitRole.TRAIN, SplitRole.DEVELOPMENT}:
            return EvaluationStage.TRAINING
        if self in {
            SplitRole.CALIBRATION_GALLERY,
            SplitRole.CALIBRATION_KNOWN_QUERY,
            SplitRole.CALIBRATION_UNKNOWN_QUERY,
        }:
            return EvaluationStage.CALIBRATION
        return EvaluationStage.TEST


@dataclass(frozen=True, slots=True)
class TrackletRecord:
    sample_id: str
    role: SplitRole
    registered_dog_id: str
    identity_verification_source: str
    source_id: str
    site_id: str
    camera_id: str
    cage_id: str
    session_id: str
    occupancy_episode_id: str
    track_id: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    modality: Modality
    collar: PresenceState = PresenceState.UNKNOWN
    harness: PresenceState = PresenceState.UNKNOWN
    clothing: PresenceState = PresenceState.UNKNOWN

    def __post_init__(self) -> None:
        for field_name in (
            "sample_id",
            "registered_dog_id",
            "identity_verification_source",
            "source_id",
            "site_id",
            "camera_id",
            "cage_id",
            "session_id",
            "occupancy_episode_id",
            "track_id",
        ):
            _require_nonempty(field_name, getattr(self, field_name))
        if self.start_timestamp_ns < 0:
            raise ValueError("tracklet start must be non-negative")
        if self.end_timestamp_ns <= self.start_timestamp_ns:
            raise ValueError("tracklet end must be after start")

    @property
    def session_key(self) -> tuple[str, str]:
        return (self.camera_id, self.session_id)

    @property
    def episode_key(self) -> tuple[str, str, str, str]:
        return (
            self.camera_id,
            self.cage_id,
            self.session_id,
            self.occupancy_episode_id,
        )

    @property
    def track_key(self) -> tuple[str, str, str]:
        return (self.camera_id, self.session_id, self.track_id)

    @property
    def accessory_signature(
        self,
    ) -> tuple[PresenceState, PresenceState, PresenceState]:
        return (self.collar, self.harness, self.clothing)

    def to_dict(self) -> dict[str, str | int]:
        return {
            "sample_id": self.sample_id,
            "role": self.role.value,
            "registered_dog_id": self.registered_dog_id,
            "identity_verification_source": self.identity_verification_source,
            "source_id": self.source_id,
            "site_id": self.site_id,
            "camera_id": self.camera_id,
            "cage_id": self.cage_id,
            "session_id": self.session_id,
            "occupancy_episode_id": self.occupancy_episode_id,
            "track_id": self.track_id,
            "start_timestamp_ns": self.start_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "modality": self.modality.value,
            "collar": self.collar.value,
            "harness": self.harness.value,
            "clothing": self.clothing.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrackletRecord:
        allowed = {
            "sample_id",
            "role",
            "registered_dog_id",
            "identity_verification_source",
            "source_id",
            "site_id",
            "camera_id",
            "cage_id",
            "session_id",
            "occupancy_episode_id",
            "track_id",
            "start_timestamp_ns",
            "end_timestamp_ns",
            "modality",
            "collar",
            "harness",
            "clothing",
        }
        _reject_unknown_keys(payload, allowed, "tracklet record")
        kwargs = dict(payload)
        kwargs["role"] = SplitRole(payload["role"])
        kwargs["modality"] = Modality(payload["modality"])
        for field_name in ("collar", "harness", "clothing"):
            kwargs[field_name] = PresenceState(
                payload.get(field_name, PresenceState.UNKNOWN.value)
            )
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class RoleModalityRule:
    role: SplitRole
    allowed_modalities: tuple[Modality, ...]

    def __post_init__(self) -> None:
        if not self.allowed_modalities:
            raise ValueError("allowed_modalities must not be empty")
        if len(self.allowed_modalities) != len(set(self.allowed_modalities)):
            raise ValueError("allowed_modalities must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "allowed_modalities": [
                modality.value for modality in self.allowed_modalities
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RoleModalityRule:
        allowed = {"role", "allowed_modalities"}
        _reject_unknown_keys(payload, allowed, "role modality rule")
        return cls(
            role=SplitRole(payload["role"]),
            allowed_modalities=tuple(
                Modality(modality) for modality in payload["allowed_modalities"]
            ),
        )


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    name: str
    required_roles: tuple[SplitRole, ...] = tuple(SplitRole)
    stage_disjoint_keys: tuple[str, ...] = ()
    require_train_evaluation_identity_disjoint: bool = True
    require_calibration_test_identity_disjoint: bool = True
    require_chronological_test: bool = False
    modality_rules: tuple[RoleModalityRule, ...] = ()
    require_known_query_accessory_change: bool = False
    minimum_gallery_query_gap_seconds: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty("name", self.name)
        if len(self.required_roles) != len(set(self.required_roles)):
            raise ValueError("required_roles must be unique")
        allowed_stage_keys = {"camera_id", "cage_id", "site_id"}
        unknown = set(self.stage_disjoint_keys) - allowed_stage_keys
        if unknown:
            raise ValueError(
                f"unsupported stage-disjoint keys: {', '.join(sorted(unknown))}"
            )
        if len(self.stage_disjoint_keys) != len(set(self.stage_disjoint_keys)):
            raise ValueError("stage_disjoint_keys must be unique")
        modality_roles = tuple(rule.role for rule in self.modality_rules)
        if len(modality_roles) != len(set(modality_roles)):
            raise ValueError("each role may have only one modality rule")
        if self.minimum_gallery_query_gap_seconds is not None:
            if self.minimum_gallery_query_gap_seconds <= 0:
                raise ValueError("minimum gallery/query gap must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required_roles": [role.value for role in self.required_roles],
            "stage_disjoint_keys": list(self.stage_disjoint_keys),
            "require_train_evaluation_identity_disjoint": (
                self.require_train_evaluation_identity_disjoint
            ),
            "require_calibration_test_identity_disjoint": (
                self.require_calibration_test_identity_disjoint
            ),
            "require_chronological_test": self.require_chronological_test,
            "modality_rules": [rule.to_dict() for rule in self.modality_rules],
            "require_known_query_accessory_change": (
                self.require_known_query_accessory_change
            ),
            "minimum_gallery_query_gap_seconds": (
                self.minimum_gallery_query_gap_seconds
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitPolicy:
        allowed = {
            "name",
            "required_roles",
            "stage_disjoint_keys",
            "require_train_evaluation_identity_disjoint",
            "require_calibration_test_identity_disjoint",
            "require_chronological_test",
            "modality_rules",
            "require_known_query_accessory_change",
            "minimum_gallery_query_gap_seconds",
        }
        _reject_unknown_keys(payload, allowed, "split policy")
        return cls(
            name=payload["name"],
            required_roles=tuple(
                SplitRole(role) for role in payload.get("required_roles", SplitRole)
            ),
            stage_disjoint_keys=tuple(payload.get("stage_disjoint_keys", ())),
            require_train_evaluation_identity_disjoint=payload.get(
                "require_train_evaluation_identity_disjoint",
                True,
            ),
            require_calibration_test_identity_disjoint=payload.get(
                "require_calibration_test_identity_disjoint",
                True,
            ),
            require_chronological_test=payload.get(
                "require_chronological_test",
                False,
            ),
            modality_rules=tuple(
                RoleModalityRule.from_dict(rule)
                for rule in payload.get("modality_rules", ())
            ),
            require_known_query_accessory_change=payload.get(
                "require_known_query_accessory_change",
                False,
            ),
            minimum_gallery_query_gap_seconds=payload.get(
                "minimum_gallery_query_gap_seconds"
            ),
        )


@dataclass(frozen=True, slots=True)
class SplitManifest:
    policy: SplitPolicy
    admitted_source_ids: tuple[str, ...]
    records: tuple[TrackletRecord, ...]
    schema_version: str = "evaluation.split.v1"

    def __post_init__(self) -> None:
        if not self.admitted_source_ids:
            raise ValueError("admitted_source_ids must not be empty")
        _require_unique(self.admitted_source_ids, "admitted_source_ids")
        _require_unique(
            tuple(record.sample_id for record in self.records),
            "sample_id",
        )
        unknown_sources = {
            record.source_id for record in self.records
        } - set(self.admitted_source_ids)
        if unknown_sources:
            raise ValueError(
                f"records use unadmitted sources: {', '.join(sorted(unknown_sources))}"
            )

    def gate_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        present_roles = {record.role for record in self.records}
        for role in self.policy.required_roles:
            if role not in present_roles:
                blockers.append(f"missing_role:{role.value}")

        mandatory_groups = {
            "source_id": lambda record: record.source_id,
            "session_key": lambda record: record.session_key,
            "episode_key": lambda record: record.episode_key,
            "track_key": lambda record: record.track_key,
        }
        for group_name, key_function in mandatory_groups.items():
            role_sets: dict[Any, set[SplitRole]] = {}
            for record in self.records:
                role_sets.setdefault(key_function(record), set()).add(record.role)
            for group_value, roles in sorted(
                role_sets.items(),
                key=lambda item: repr(item[0]),
            ):
                if len(roles) > 1:
                    role_names = ",".join(sorted(role.value for role in roles))
                    blockers.append(
                        f"cross_role_leak:{group_name}:{group_value!r}:{role_names}"
                    )

        self._append_open_set_blockers(blockers, EvaluationStage.CALIBRATION)
        self._append_open_set_blockers(blockers, EvaluationStage.TEST)

        identity_by_stage = {
            stage: {
                record.registered_dog_id
                for record in self.records
                if record.role.stage is stage
            }
            for stage in EvaluationStage
        }
        if self.policy.require_train_evaluation_identity_disjoint:
            overlap = identity_by_stage[EvaluationStage.TRAINING] & (
                identity_by_stage[EvaluationStage.CALIBRATION]
                | identity_by_stage[EvaluationStage.TEST]
            )
            blockers.extend(
                f"train_evaluation_identity_leak:{dog_id}"
                for dog_id in sorted(overlap)
            )
        if self.policy.require_calibration_test_identity_disjoint:
            overlap = (
                identity_by_stage[EvaluationStage.CALIBRATION]
                & identity_by_stage[EvaluationStage.TEST]
            )
            blockers.extend(
                f"calibration_test_identity_leak:{dog_id}"
                for dog_id in sorted(overlap)
            )

        for key_name in self.policy.stage_disjoint_keys:
            stage_sets: dict[str, set[EvaluationStage]] = {}
            for record in self.records:
                stage_sets.setdefault(
                    str(getattr(record, key_name)),
                    set(),
                ).add(record.role.stage)
            for value, stages in sorted(stage_sets.items()):
                if len(stages) > 1:
                    stage_names = ",".join(sorted(stage.value for stage in stages))
                    blockers.append(
                        f"cross_stage_leak:{key_name}:{value}:{stage_names}"
                    )

        for rule in self.policy.modality_rules:
            for record in self.records:
                if (
                    record.role is rule.role
                    and record.modality not in rule.allowed_modalities
                ):
                    blockers.append(
                        f"modality_violation:{record.sample_id}:"
                        f"{record.role.value}:{record.modality.value}"
                    )

        if self.policy.require_chronological_test:
            pretest_ends = [
                record.end_timestamp_ns
                for record in self.records
                if record.role.stage is not EvaluationStage.TEST
            ]
            test_starts = [
                record.start_timestamp_ns
                for record in self.records
                if record.role.stage is EvaluationStage.TEST
            ]
            if (
                pretest_ends
                and test_starts
                and min(test_starts) <= max(pretest_ends)
            ):
                blockers.append("chronological_leak:test_not_strictly_after_pretest")
        if self.policy.require_known_query_accessory_change:
            self._append_accessory_change_blockers(
                blockers,
                EvaluationStage.CALIBRATION,
            )
            self._append_accessory_change_blockers(blockers, EvaluationStage.TEST)
        if self.policy.minimum_gallery_query_gap_seconds is not None:
            self._append_longitudinal_blockers(
                blockers,
                EvaluationStage.CALIBRATION,
                self.policy.minimum_gallery_query_gap_seconds,
            )
            self._append_longitudinal_blockers(
                blockers,
                EvaluationStage.TEST,
                self.policy.minimum_gallery_query_gap_seconds,
            )
        return tuple(blockers)

    def _append_open_set_blockers(
        self,
        blockers: list[str],
        stage: EvaluationStage,
    ) -> None:
        gallery_role, known_role, unknown_role = _stage_roles(stage)
        identities = {
            role: {
                record.registered_dog_id
                for record in self.records
                if record.role is role
            }
            for role in (gallery_role, known_role, unknown_role)
        }
        missing_from_gallery = identities[known_role] - identities[gallery_role]
        blockers.extend(
            f"{stage.value.lower()}:known_without_gallery:{dog_id}"
            for dog_id in sorted(missing_from_gallery)
        )
        unknown_overlap = identities[unknown_role] & (
            identities[gallery_role] | identities[known_role]
        )
        blockers.extend(
            f"{stage.value.lower()}:unknown_identity_leak:{dog_id}"
            for dog_id in sorted(unknown_overlap)
        )

    def _append_accessory_change_blockers(
        self,
        blockers: list[str],
        stage: EvaluationStage,
    ) -> None:
        gallery_role, known_role, _ = _stage_roles(stage)
        known_ids = {
            record.registered_dog_id
            for record in self.records
            if record.role is known_role
        }
        for dog_id in sorted(known_ids):
            gallery_signatures = {
                record.accessory_signature
                for record in self.records
                if record.role is gallery_role
                and record.registered_dog_id == dog_id
                and _accessory_signature_is_known(record.accessory_signature)
            }
            query_signatures = {
                record.accessory_signature
                for record in self.records
                if record.role is known_role
                and record.registered_dog_id == dog_id
                and _accessory_signature_is_known(record.accessory_signature)
            }
            if not gallery_signatures or not query_signatures:
                blockers.append(
                    f"{stage.value.lower()}:accessory_state_unresolved:{dog_id}"
                )
            elif not any(
                gallery != query
                for gallery in gallery_signatures
                for query in query_signatures
            ):
                blockers.append(
                    f"{stage.value.lower()}:accessory_unchanged:{dog_id}"
                )

    def _append_longitudinal_blockers(
        self,
        blockers: list[str],
        stage: EvaluationStage,
        minimum_gap_seconds: float,
    ) -> None:
        gallery_role, known_role, _ = _stage_roles(stage)
        minimum_gap_ns = round(minimum_gap_seconds * 1_000_000_000)
        gallery_by_id: dict[str, list[TrackletRecord]] = {}
        for record in self.records:
            if record.role is gallery_role:
                gallery_by_id.setdefault(record.registered_dog_id, []).append(record)
        for query in self.records:
            if query.role is not known_role:
                continue
            gallery_records = gallery_by_id.get(query.registered_dog_id, [])
            if not gallery_records:
                continue
            latest_gallery_end = max(
                record.end_timestamp_ns for record in gallery_records
            )
            if query.start_timestamp_ns - latest_gallery_end < minimum_gap_ns:
                blockers.append(
                    f"{stage.value.lower()}:insufficient_gallery_query_gap:"
                    f"{query.sample_id}"
                )

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy.to_dict(),
            "admitted_source_ids": list(self.admitted_source_ids),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitManifest:
        allowed = {
            "schema_version",
            "policy",
            "admitted_source_ids",
            "records",
        }
        _reject_unknown_keys(payload, allowed, "split manifest")
        schema_version = payload.get("schema_version")
        if schema_version != "evaluation.split.v1":
            raise ValueError(f"unsupported split schema: {schema_version!r}")
        admitted_sources = payload.get("admitted_source_ids")
        records = payload.get("records")
        if not isinstance(admitted_sources, list) or not isinstance(records, list):
            raise ValueError("admitted_source_ids and records must be lists")
        return cls(
            policy=SplitPolicy.from_dict(payload["policy"]),
            admitted_source_ids=tuple(admitted_sources),
            records=tuple(TrackletRecord.from_dict(record) for record in records),
            schema_version=schema_version,
        )


def _require_nonempty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_unique(values: tuple[Any, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} values must be unique")


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


def _stage_roles(
    stage: EvaluationStage,
) -> tuple[SplitRole, SplitRole, SplitRole]:
    if stage is EvaluationStage.CALIBRATION:
        return (
            SplitRole.CALIBRATION_GALLERY,
            SplitRole.CALIBRATION_KNOWN_QUERY,
            SplitRole.CALIBRATION_UNKNOWN_QUERY,
        )
    if stage is EvaluationStage.TEST:
        return (
            SplitRole.TEST_GALLERY,
            SplitRole.TEST_KNOWN_QUERY,
            SplitRole.TEST_UNKNOWN_QUERY,
        )
    raise ValueError("training stage has no gallery/query role triple")


def _accessory_signature_is_known(
    signature: tuple[PresenceState, PresenceState, PresenceState],
) -> bool:
    return PresenceState.UNKNOWN not in signature
