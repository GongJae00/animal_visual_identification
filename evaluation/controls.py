"""Matched, label-blind planning contracts for visual shortcut controls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from data.acquisition import sha256_file
from evaluation.pairing import (
    PairConstructionResult,
    PairScoringRequest,
    PairStratum,
)
from evaluation.scoring import (
    PairArtifactEntry,
    PairArtifactManifest,
    PairArtifactVerification,
    validate_pair_artifact_manifest,
)
from foundation.provenance import content_sha256

if TYPE_CHECKING:
    from evaluation.mask_semantics import MaskSemanticVerification


class MaskRole(StrEnum):
    DOG = "DOG"
    ACCESSORY = "ACCESSORY"


class MaskReviewStatus(StrEnum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    REJECTED = "REJECTED"


class VisualControlKind(StrEnum):
    ORIGINAL = "ORIGINAL"
    DOG_ONLY = "DOG_ONLY"
    BACKGROUND_ONLY = "BACKGROUND_ONLY"
    BODY_BLURRED = "BODY_BLURRED"
    MASK_ONLY = "MASK_ONLY"
    ACCESSORY_ONLY = "ACCESSORY_ONLY"
    ACCESSORY_MASKED = "ACCESSORY_MASKED"

    @property
    def required_mask_roles(self) -> tuple[MaskRole, ...]:
        mapping = {
            VisualControlKind.ORIGINAL: (),
            VisualControlKind.DOG_ONLY: (MaskRole.DOG,),
            VisualControlKind.BACKGROUND_ONLY: (MaskRole.DOG,),
            VisualControlKind.BODY_BLURRED: (MaskRole.DOG,),
            VisualControlKind.MASK_ONLY: (MaskRole.DOG,),
            VisualControlKind.ACCESSORY_ONLY: (
                MaskRole.DOG,
                MaskRole.ACCESSORY,
            ),
            VisualControlKind.ACCESSORY_MASKED: (
                MaskRole.DOG,
                MaskRole.ACCESSORY,
            ),
        }
        return mapping[self]


@dataclass(frozen=True, slots=True)
class MaskEvidence:
    role: MaskRole
    artifact_token: str
    relative_path: str
    content_sha256: str
    byte_size: int
    width: int
    height: int
    annotation_version: str
    provenance_kind: str
    provenance_reference_sha256: str
    review_status: MaskReviewStatus

    def __post_init__(self) -> None:
        _require_nonempty(self.artifact_token, "artifact_token")
        _validate_mask_relative_path(
            self.relative_path,
            self.artifact_token,
        )
        _validate_sha256(self.content_sha256, "content_sha256")
        _require_positive_int(self.byte_size, "byte_size")
        _validate_sha256(
            self.provenance_reference_sha256,
            "provenance_reference_sha256",
        )
        _require_positive_int(self.width, "width")
        _require_positive_int(self.height, "height")
        _require_nonempty(self.annotation_version, "annotation_version")
        _require_nonempty(self.provenance_kind, "provenance_kind")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "role": self.role.value,
            "artifact_token": self.artifact_token,
            "relative_path": self.relative_path,
            "content_sha256": self.content_sha256,
            "byte_size": self.byte_size,
            "width": self.width,
            "height": self.height,
            "annotation_version": self.annotation_version,
            "provenance_kind": self.provenance_kind,
            "provenance_reference_sha256": (
                self.provenance_reference_sha256
            ),
            "review_status": self.review_status.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MaskEvidence:
        _require_exact_keys(
            payload,
            {
                "role",
                "artifact_token",
                "relative_path",
                "content_sha256",
                "byte_size",
                "width",
                "height",
                "annotation_version",
                "provenance_kind",
                "provenance_reference_sha256",
                "review_status",
            },
            "mask evidence",
        )
        return cls(
            role=MaskRole(payload["role"]),
            artifact_token=payload["artifact_token"],
            relative_path=payload["relative_path"],
            content_sha256=payload["content_sha256"],
            byte_size=payload["byte_size"],
            width=payload["width"],
            height=payload["height"],
            annotation_version=payload["annotation_version"],
            provenance_kind=payload["provenance_kind"],
            provenance_reference_sha256=payload[
                "provenance_reference_sha256"
            ],
            review_status=MaskReviewStatus(payload["review_status"]),
        )


@dataclass(frozen=True, slots=True)
class ControlMaskEntry:
    base_artifact_token: str
    masks: tuple[MaskEvidence, ...]

    def __post_init__(self) -> None:
        _require_nonempty(
            self.base_artifact_token,
            "base_artifact_token",
        )
        roles = tuple(mask.role for mask in self.masks)
        if len(roles) != len(set(roles)):
            raise ValueError("mask roles must be unique per base artifact")
        dimensions = {
            (mask.width, mask.height) for mask in self.masks
        }
        if len(dimensions) > 1:
            raise ValueError("masks for one artifact must share dimensions")

    def mask_for(self, role: MaskRole) -> MaskEvidence | None:
        return next(
            (mask for mask in self.masks if mask.role is role),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_artifact_token": self.base_artifact_token,
            "masks": [mask.to_dict() for mask in self.masks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlMaskEntry:
        _require_exact_keys(
            payload,
            {"base_artifact_token", "masks"},
            "control mask entry",
        )
        masks = payload["masks"]
        if not isinstance(masks, list):
            raise TypeError("control entry masks must be a list")
        return cls(
            base_artifact_token=payload["base_artifact_token"],
            masks=tuple(MaskEvidence.from_dict(item) for item in masks),
        )


@dataclass(frozen=True, slots=True)
class ControlMaskManifest:
    base_artifact_manifest_sha256: str
    entries: tuple[ControlMaskEntry, ...]
    schema_version: str = "cvi.control_mask_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.control_mask_manifest.v1":
            raise ValueError("unsupported control mask manifest schema")
        _validate_sha256(
            self.base_artifact_manifest_sha256,
            "base_artifact_manifest_sha256",
        )
        if not self.entries:
            raise ValueError("control mask manifest must not be empty")
        tokens = tuple(entry.base_artifact_token for entry in self.entries)
        if len(tokens) != len(set(tokens)):
            raise ValueError("control mask artifact tokens must be unique")
        mask_tokens = tuple(
            mask.artifact_token
            for entry in self.entries
            for mask in entry.masks
        )
        mask_paths = tuple(
            mask.relative_path
            for entry in self.entries
            for mask in entry.masks
        )
        if len(mask_tokens) != len(set(mask_tokens)):
            raise ValueError("mask artifact tokens must be globally unique")
        if len(mask_paths) != len(set(mask_paths)):
            raise ValueError("mask artifact paths must be globally unique")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_artifact_manifest_sha256": (
                self.base_artifact_manifest_sha256
            ),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlMaskManifest:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "base_artifact_manifest_sha256",
                "entries",
            },
            "control mask manifest",
        )
        entries = payload["entries"]
        if not isinstance(entries, list):
            raise TypeError("control mask entries must be a list")
        return cls(
            schema_version=payload["schema_version"],
            base_artifact_manifest_sha256=payload[
                "base_artifact_manifest_sha256"
            ],
            entries=tuple(
                ControlMaskEntry.from_dict(item) for item in entries
            ),
        )


@dataclass(frozen=True, slots=True)
class ControlMaskVerification:
    mask_manifest_sha256: str
    verified_files: int
    verified_bytes: int

    def __post_init__(self) -> None:
        _validate_sha256(
            self.mask_manifest_sha256,
            "mask_manifest_sha256",
        )
        for name in ("verified_files", "verified_bytes"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "schema_version": "cvi.control_mask_verification.v1",
            "mask_manifest_sha256": self.mask_manifest_sha256,
            "verified_files": self.verified_files,
            "verified_bytes": self.verified_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlMaskVerification:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "mask_manifest_sha256",
                "verified_files",
                "verified_bytes",
            },
            "control mask verification",
        )
        if payload["schema_version"] != "cvi.control_mask_verification.v1":
            raise ValueError("unsupported control mask verification schema")
        return cls(
            mask_manifest_sha256=payload["mask_manifest_sha256"],
            verified_files=payload["verified_files"],
            verified_bytes=payload["verified_bytes"],
        )


@dataclass(frozen=True, slots=True)
class VisualControlRecipe:
    kind: VisualControlKind
    transform_config_sha256: str
    semantics_version: str

    def __post_init__(self) -> None:
        _validate_sha256(
            self.transform_config_sha256,
            "transform_config_sha256",
        )
        _require_nonempty(self.semantics_version, "semantics_version")

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "transform_config_sha256": self.transform_config_sha256,
            "semantics_version": self.semantics_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VisualControlRecipe:
        _require_exact_keys(
            payload,
            {
                "kind",
                "transform_config_sha256",
                "semantics_version",
            },
            "visual control recipe",
        )
        return cls(
            kind=VisualControlKind(payload["kind"]),
            transform_config_sha256=payload[
                "transform_config_sha256"
            ],
            semantics_version=payload["semantics_version"],
        )


@dataclass(frozen=True, slots=True)
class VisualControlPanel:
    panel_id: str
    controls: tuple[VisualControlKind, ...]
    minimum_matched_pairs: int
    maximum_matched_pairs: int

    def __post_init__(self) -> None:
        _require_nonempty(self.panel_id, "panel_id")
        if len(self.controls) < 2:
            raise ValueError("control panel requires at least two controls")
        if len(self.controls) != len(set(self.controls)):
            raise ValueError("control panel kinds must be unique")
        if self.controls[0] is not VisualControlKind.ORIGINAL:
            raise ValueError("ORIGINAL must be the first panel control")
        _require_positive_int(
            self.minimum_matched_pairs,
            "minimum_matched_pairs",
        )
        _require_positive_int(
            self.maximum_matched_pairs,
            "maximum_matched_pairs",
        )
        if self.minimum_matched_pairs > self.maximum_matched_pairs:
            raise ValueError(
                "minimum_matched_pairs exceeds maximum_matched_pairs"
            )

    @property
    def required_mask_roles(self) -> tuple[MaskRole, ...]:
        return tuple(
            sorted(
                {
                    role
                    for kind in self.controls
                    for role in kind.required_mask_roles
                },
                key=lambda role: role.value,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "controls": [control.value for control in self.controls],
            "minimum_matched_pairs": self.minimum_matched_pairs,
            "maximum_matched_pairs": self.maximum_matched_pairs,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VisualControlPanel:
        _require_exact_keys(
            payload,
            {
                "panel_id",
                "controls",
                "minimum_matched_pairs",
                "maximum_matched_pairs",
            },
            "visual control panel",
        )
        controls = payload["controls"]
        if not isinstance(controls, list):
            raise TypeError("visual control panel controls must be a list")
        return cls(
            panel_id=payload["panel_id"],
            controls=tuple(
                VisualControlKind(item) for item in controls
            ),
            minimum_matched_pairs=payload["minimum_matched_pairs"],
            maximum_matched_pairs=payload["maximum_matched_pairs"],
        )


@dataclass(frozen=True, slots=True)
class VisualControlPolicy:
    name: str
    recipes: tuple[VisualControlRecipe, ...]
    panels: tuple[VisualControlPanel, ...]
    seed: int
    schema_version: str = "cvi.visual_control_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.visual_control_policy.v1":
            raise ValueError("unsupported visual control policy schema")
        _require_nonempty(self.name, "name")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        recipe_kinds = tuple(recipe.kind for recipe in self.recipes)
        if len(recipe_kinds) != len(set(recipe_kinds)):
            raise ValueError("visual control recipes must be unique by kind")
        panel_ids = tuple(panel.panel_id for panel in self.panels)
        if not panel_ids or len(panel_ids) != len(set(panel_ids)):
            raise ValueError("visual control panel IDs must be nonempty/unique")
        used_kinds = {
            kind for panel in self.panels for kind in panel.controls
        }
        if used_kinds != set(recipe_kinds):
            raise ValueError(
                "recipes must exactly cover the controls used by panels"
            )

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def recipe_for(self, kind: VisualControlKind) -> VisualControlRecipe:
        return next(recipe for recipe in self.recipes if recipe.kind is kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "recipes": [recipe.to_dict() for recipe in self.recipes],
            "panels": [panel.to_dict() for panel in self.panels],
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VisualControlPolicy:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "name",
                "recipes",
                "panels",
                "seed",
            },
            "visual control policy",
        )
        recipes = payload["recipes"]
        panels = payload["panels"]
        if not isinstance(recipes, list) or not isinstance(panels, list):
            raise TypeError("visual control recipes and panels must be lists")
        return cls(
            schema_version=payload["schema_version"],
            name=payload["name"],
            recipes=tuple(
                VisualControlRecipe.from_dict(item) for item in recipes
            ),
            panels=tuple(
                VisualControlPanel.from_dict(item) for item in panels
            ),
            seed=payload["seed"],
        )


@dataclass(frozen=True, slots=True)
class ControlTransformTask:
    control_artifact_token: str
    base_artifact_token: str
    control_kind: VisualControlKind
    transform_config_sha256: str
    semantics_version: str
    mask_artifacts: tuple[tuple[MaskRole, str, str], ...]

    def __post_init__(self) -> None:
        _require_nonempty(
            self.control_artifact_token,
            "control_artifact_token",
        )
        _require_nonempty(self.base_artifact_token, "base_artifact_token")
        if self.control_kind is VisualControlKind.ORIGINAL:
            raise ValueError("ORIGINAL does not require a transform task")
        _validate_sha256(
            self.transform_config_sha256,
            "transform_config_sha256",
        )
        _require_nonempty(self.semantics_version, "semantics_version")
        roles = tuple(role for role, _, _ in self.mask_artifacts)
        if roles != self.control_kind.required_mask_roles:
            raise ValueError(
                "transform task mask roles must exactly match control kind"
            )
        tokens = tuple(token for _, token, _ in self.mask_artifacts)
        if len(tokens) != len(set(tokens)):
            raise ValueError("transform task mask tokens must be unique")
        for _, token, digest in self.mask_artifacts:
            _require_nonempty(token, "mask artifact token")
            _validate_sha256(digest, "mask artifact content_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_artifact_token": self.control_artifact_token,
            "base_artifact_token": self.base_artifact_token,
            "control_kind": self.control_kind.value,
            "transform_config_sha256": self.transform_config_sha256,
            "semantics_version": self.semantics_version,
            "mask_artifacts": [
                {
                    "role": role.value,
                    "artifact_token": token,
                    "content_sha256": digest,
                }
                for role, token, digest in self.mask_artifacts
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlTransformTask:
        _require_exact_keys(
            payload,
            {
                "control_artifact_token",
                "base_artifact_token",
                "control_kind",
                "transform_config_sha256",
                "semantics_version",
                "mask_artifacts",
            },
            "control transform task",
        )
        masks = payload["mask_artifacts"]
        if not isinstance(masks, list):
            raise TypeError("transform task mask_artifacts must be a list")
        parsed_masks: list[tuple[MaskRole, str, str]] = []
        for item in masks:
            _require_exact_keys(
                item,
                {"role", "artifact_token", "content_sha256"},
                "control transform task mask",
            )
            parsed_masks.append(
                (
                    MaskRole(item["role"]),
                    item["artifact_token"],
                    item["content_sha256"],
                )
            )
        return cls(
            control_artifact_token=payload["control_artifact_token"],
            base_artifact_token=payload["base_artifact_token"],
            control_kind=VisualControlKind(payload["control_kind"]),
            transform_config_sha256=payload[
                "transform_config_sha256"
            ],
            semantics_version=payload["semantics_version"],
            mask_artifacts=tuple(parsed_masks),
        )


@dataclass(frozen=True, slots=True)
class ControlScoringRequest:
    request_id: str
    query_artifact_token: str
    reference_artifact_token: str

    def __post_init__(self) -> None:
        _require_nonempty(self.request_id, "request_id")
        _require_nonempty(
            self.query_artifact_token,
            "query_artifact_token",
        )
        _require_nonempty(
            self.reference_artifact_token,
            "reference_artifact_token",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "query_artifact_token": self.query_artifact_token,
            "reference_artifact_token": self.reference_artifact_token,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlScoringRequest:
        _require_exact_keys(
            payload,
            {
                "request_id",
                "query_artifact_token",
                "reference_artifact_token",
            },
            "control scoring request",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ControlEvaluationBinding:
    request_id: str
    panel_id: str
    control_kind: VisualControlKind
    base_pair_id: str

    def __post_init__(self) -> None:
        _require_nonempty(self.request_id, "request_id")
        _require_nonempty(self.panel_id, "panel_id")
        _require_nonempty(self.base_pair_id, "base_pair_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "panel_id": self.panel_id,
            "control_kind": self.control_kind.value,
            "base_pair_id": self.base_pair_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlEvaluationBinding:
        _require_exact_keys(
            payload,
            {
                "request_id",
                "panel_id",
                "control_kind",
                "base_pair_id",
            },
            "control evaluation binding",
        )
        return cls(
            request_id=payload["request_id"],
            panel_id=payload["panel_id"],
            control_kind=VisualControlKind(payload["control_kind"]),
            base_pair_id=payload["base_pair_id"],
        )


@dataclass(frozen=True, slots=True)
class ControlStratumCount:
    stratum: PairStratum
    eligible_pairs: int
    selected_pairs: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(
            self.eligible_pairs,
            "eligible_pairs",
        )
        _require_nonnegative_int(
            self.selected_pairs,
            "selected_pairs",
        )
        if self.selected_pairs > self.eligible_pairs:
            raise ValueError("selected stratum pairs exceed eligible pairs")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "stratum": self.stratum.value,
            "eligible_pairs": self.eligible_pairs,
            "selected_pairs": self.selected_pairs,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlStratumCount:
        _require_exact_keys(
            payload,
            {"stratum", "eligible_pairs", "selected_pairs"},
            "control stratum count",
        )
        return cls(
            stratum=PairStratum(payload["stratum"]),
            eligible_pairs=payload["eligible_pairs"],
            selected_pairs=payload["selected_pairs"],
        )


@dataclass(frozen=True, slots=True)
class ControlExclusionCount:
    reason: str
    pair_count: int

    def __post_init__(self) -> None:
        _require_nonempty(self.reason, "reason")
        _require_positive_int(self.pair_count, "pair_count")

    def to_dict(self) -> dict[str, str | int]:
        return {"reason": self.reason, "pair_count": self.pair_count}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlExclusionCount:
        _require_exact_keys(
            payload,
            {"reason", "pair_count"},
            "control exclusion count",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ControlPanelSummary:
    panel_id: str
    required_mask_roles: tuple[MaskRole, ...]
    total_pairs: int
    eligible_pairs: int
    selected_pairs: int
    ineligible_pairs: int
    cap_applied: bool
    minimum_met: bool
    exclusions: tuple[ControlExclusionCount, ...]
    strata: tuple[ControlStratumCount, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.panel_id, "panel_id")
        if len(self.required_mask_roles) != len(
            set(self.required_mask_roles)
        ):
            raise ValueError("panel summary mask roles must be unique")
        for name in (
            "total_pairs",
            "eligible_pairs",
            "selected_pairs",
            "ineligible_pairs",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        if self.total_pairs != self.eligible_pairs + self.ineligible_pairs:
            raise ValueError("panel eligibility counts are inconsistent")
        if self.selected_pairs > self.eligible_pairs:
            raise ValueError("panel selected pairs exceed eligible pairs")
        if not isinstance(self.cap_applied, bool) or not isinstance(
            self.minimum_met,
            bool,
        ):
            raise TypeError("panel summary flags must be booleans")
        if self.cap_applied != (self.selected_pairs < self.eligible_pairs):
            raise ValueError("panel cap flag is inconsistent")
        reasons = tuple(item.reason for item in self.exclusions)
        if len(reasons) != len(set(reasons)):
            raise ValueError("panel exclusion reasons must be unique")
        strata = tuple(item.stratum for item in self.strata)
        if len(strata) != len(set(strata)):
            raise ValueError("panel strata must be unique")
        if sum(item.eligible_pairs for item in self.strata) != (
            self.eligible_pairs
        ):
            raise ValueError("panel eligible stratum counts are inconsistent")
        if sum(item.selected_pairs for item in self.strata) != (
            self.selected_pairs
        ):
            raise ValueError("panel selected stratum counts are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "required_mask_roles": [
                role.value for role in self.required_mask_roles
            ],
            "total_pairs": self.total_pairs,
            "eligible_pairs": self.eligible_pairs,
            "selected_pairs": self.selected_pairs,
            "ineligible_pairs": self.ineligible_pairs,
            "cap_applied": self.cap_applied,
            "minimum_met": self.minimum_met,
            "exclusions": [item.to_dict() for item in self.exclusions],
            "strata": [item.to_dict() for item in self.strata],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlPanelSummary:
        _require_exact_keys(
            payload,
            {
                "panel_id",
                "required_mask_roles",
                "total_pairs",
                "eligible_pairs",
                "selected_pairs",
                "ineligible_pairs",
                "cap_applied",
                "minimum_met",
                "exclusions",
                "strata",
            },
            "control panel summary",
        )
        roles = payload["required_mask_roles"]
        exclusions = payload["exclusions"]
        strata = payload["strata"]
        if (
            not isinstance(roles, list)
            or not isinstance(exclusions, list)
            or not isinstance(strata, list)
        ):
            raise TypeError(
                "panel roles, exclusions, and strata must be lists"
            )
        return cls(
            panel_id=payload["panel_id"],
            required_mask_roles=tuple(MaskRole(role) for role in roles),
            total_pairs=payload["total_pairs"],
            eligible_pairs=payload["eligible_pairs"],
            selected_pairs=payload["selected_pairs"],
            ineligible_pairs=payload["ineligible_pairs"],
            cap_applied=payload["cap_applied"],
            minimum_met=payload["minimum_met"],
            exclusions=tuple(
                ControlExclusionCount.from_dict(item) for item in exclusions
            ),
            strata=tuple(
                ControlStratumCount.from_dict(item) for item in strata
            ),
        )


@dataclass(frozen=True, slots=True)
class ControlCostSummary:
    scoring_requests: int
    naive_embedding_calls: int
    unique_embedding_artifacts: int
    reusable_embedding_calls_saved: int
    transform_tasks: int

    def to_dict(self) -> dict[str, int]:
        return {
            "scoring_requests": self.scoring_requests,
            "naive_embedding_calls": self.naive_embedding_calls,
            "unique_embedding_artifacts": self.unique_embedding_artifacts,
            "reusable_embedding_calls_saved": (
                self.reusable_embedding_calls_saved
            ),
            "transform_tasks": self.transform_tasks,
        }


@dataclass(frozen=True, slots=True)
class VisualControlAuditPlan:
    pair_set_sha256: str
    base_artifact_manifest_sha256: str
    base_artifact_verification_sha256: str
    mask_manifest_sha256: str
    mask_verification_sha256: str
    mask_semantic_verification_sha256: str
    policy_sha256: str
    panels: tuple[ControlPanelSummary, ...]
    transform_tasks: tuple[ControlTransformTask, ...]
    scoring_requests: tuple[ControlScoringRequest, ...]
    evaluation_bindings: tuple[ControlEvaluationBinding, ...]
    cost: ControlCostSummary
    schema_version: str = "cvi.visual_control_audit_plan.v1"

    @property
    def plan_sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def gate_blockers(self) -> tuple[str, ...]:
        return tuple(
            f"{panel.panel_id}: insufficient matched pairs "
            f"({panel.selected_pairs})"
            for panel in self.panels
            if not panel.minimum_met
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair_set_sha256": self.pair_set_sha256,
            "base_artifact_manifest_sha256": (
                self.base_artifact_manifest_sha256
            ),
            "base_artifact_verification_sha256": (
                self.base_artifact_verification_sha256
            ),
            "mask_manifest_sha256": self.mask_manifest_sha256,
            "mask_verification_sha256": self.mask_verification_sha256,
            "mask_semantic_verification_sha256": (
                self.mask_semantic_verification_sha256
            ),
            "policy_sha256": self.policy_sha256,
            "panels": [panel.to_dict() for panel in self.panels],
            "transform_tasks": [
                task.to_dict() for task in self.transform_tasks
            ],
            "scoring_requests": [
                request.to_dict() for request in self.scoring_requests
            ],
            "evaluation_bindings": [
                binding.to_dict() for binding in self.evaluation_bindings
            ],
            "cost": self.cost.to_dict(),
        }

    def scoring_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "cvi.visual_control_scoring_requests.v1",
            "plan_sha256": self.plan_sha256,
            "requests": [
                request.to_dict() for request in self.scoring_requests
            ],
        }

    def protected_transform_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "cvi.visual_control_transform_tasks.v1",
            "plan_sha256": self.plan_sha256,
            "scoring_requests_sha256": content_sha256(
                self.scoring_payload()
            ),
            "tasks": [task.to_dict() for task in self.transform_tasks],
        }

    def sealed_evaluation_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "cvi.visual_control_evaluation_bindings.v1",
            "plan_sha256": self.plan_sha256,
            "pair_set_sha256": self.pair_set_sha256,
            "bindings": [
                binding.to_dict() for binding in self.evaluation_bindings
            ],
            "panel_summaries": [
                panel.to_dict() for panel in self.panels
            ],
        }

    def summary_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "cvi.visual_control_audit_summary.v1",
            "plan_sha256": self.plan_sha256,
            "pair_set_sha256": self.pair_set_sha256,
            "base_artifact_manifest_sha256": (
                self.base_artifact_manifest_sha256
            ),
            "base_artifact_verification_sha256": (
                self.base_artifact_verification_sha256
            ),
            "mask_manifest_sha256": self.mask_manifest_sha256,
            "mask_verification_sha256": self.mask_verification_sha256,
            "mask_semantic_verification_sha256": (
                self.mask_semantic_verification_sha256
            ),
            "policy_sha256": self.policy_sha256,
            "gate_blockers": list(self.gate_blockers),
            "panels": [panel.to_dict() for panel in self.panels],
            "cost": self.cost.to_dict(),
        }


def plan_visual_control_audit(
    construction: PairConstructionResult,
    base_artifact_manifest: PairArtifactManifest,
    base_artifact_verification: PairArtifactVerification,
    mask_manifest: ControlMaskManifest,
    mask_verification: ControlMaskVerification,
    mask_semantic_verification: MaskSemanticVerification,
    policy: VisualControlPolicy,
) -> VisualControlAuditPlan:
    """Build matched control requests without score- or identity-based selection."""

    validate_pair_artifact_manifest(
        construction,
        base_artifact_manifest,
    )
    if (
        base_artifact_verification.artifact_manifest_sha256
        != base_artifact_manifest.manifest_sha256
        or base_artifact_verification.verified_files
        != len(base_artifact_manifest.entries)
        or base_artifact_verification.verified_bytes
        != sum(
            entry.byte_size for entry in base_artifact_manifest.entries
        )
    ):
        raise ValueError("base artifact verification receipt mismatch")
    if (
        mask_manifest.base_artifact_manifest_sha256
        != base_artifact_manifest.manifest_sha256
    ):
        raise ValueError("mask manifest base-artifact hash mismatch")
    base_entries = {
        entry.artifact_token: entry
        for entry in base_artifact_manifest.entries
    }
    mask_entries = {
        entry.base_artifact_token: entry
        for entry in mask_manifest.entries
    }
    if set(base_entries) != set(mask_entries):
        raise ValueError("mask manifest must cover every base artifact token")
    masks = tuple(
        mask for entry in mask_manifest.entries for mask in entry.masks
    )
    if (
        mask_verification.mask_manifest_sha256
        != mask_manifest.manifest_sha256
        or mask_verification.verified_files != len(masks)
        or mask_verification.verified_bytes
        != sum(mask.byte_size for mask in masks)
    ):
        raise ValueError("mask verification receipt mismatch")
    if (
        mask_semantic_verification.base_artifact_manifest_sha256
        != base_artifact_manifest.manifest_sha256
        or mask_semantic_verification.base_artifact_verification_sha256
        != content_sha256(base_artifact_verification.to_dict())
        or mask_semantic_verification.mask_manifest_sha256
        != mask_manifest.manifest_sha256
        or mask_semantic_verification.mask_file_verification_sha256
        != content_sha256(mask_verification.to_dict())
    ):
        raise ValueError("mask semantic verification receipt mismatch")
    expected_semantic_tokens = {
        entry.base_artifact_token
        for entry in mask_manifest.entries
        if any(
            mask.review_status is MaskReviewStatus.VERIFIED
            for mask in entry.masks
        )
    }
    actual_semantic_tokens = {
        entry.base_artifact_token
        for entry in mask_semantic_verification.entries
    }
    if expected_semantic_tokens != actual_semantic_tokens:
        raise ValueError(
            "mask semantic verification token coverage mismatch"
        )
    truth_by_pair = {
        truth.pair_id: truth for truth in construction.ground_truth
    }
    transform_tasks: dict[str, ControlTransformTask] = {}
    scoring_requests: list[ControlScoringRequest] = []
    evaluation_bindings: list[ControlEvaluationBinding] = []
    panel_summaries: list[ControlPanelSummary] = []
    for panel in policy.panels:
        eligibility: dict[str, tuple[str, ...]] = {}
        for request in construction.scoring_requests:
            eligibility[request.pair_id] = _exclusion_reasons(
                request,
                panel.required_mask_roles,
                mask_entries,
            )
        eligible = tuple(
            request
            for request in construction.scoring_requests
            if not eligibility[request.pair_id]
        )
        selected = tuple(
            sorted(
                eligible,
                key=lambda request: _stable_rank(
                    policy.seed,
                    panel.panel_id,
                    request.pair_id,
                ),
            )[: panel.maximum_matched_pairs]
        )
        selected_ids = {request.pair_id for request in selected}
        selected_in_source_order = tuple(
            request
            for request in construction.scoring_requests
            if request.pair_id in selected_ids
        )
        for request in selected_in_source_order:
            for kind in panel.controls:
                query_token = _artifact_for_control(
                    request.query_artifact_token,
                    kind,
                    base_entries,
                    mask_entries,
                    policy,
                    transform_tasks,
                )
                reference_token = _artifact_for_control(
                    request.reference_artifact_token,
                    kind,
                    base_entries,
                    mask_entries,
                    policy,
                    transform_tasks,
                )
                request_id = "control-pair-" + content_sha256(
                    {
                        "policy_sha256": policy.policy_sha256,
                        "panel_id": panel.panel_id,
                        "control_kind": kind.value,
                        "base_pair_id": request.pair_id,
                    }
                )[:24]
                scoring_requests.append(
                    ControlScoringRequest(
                        request_id,
                        query_token,
                        reference_token,
                    )
                )
                evaluation_bindings.append(
                    ControlEvaluationBinding(
                        request_id,
                        panel.panel_id,
                        kind,
                        request.pair_id,
                    )
                )
        exclusion_counts: dict[str, int] = {}
        for reasons in eligibility.values():
            for reason in reasons:
                exclusion_counts[reason] = (
                    exclusion_counts.get(reason, 0) + 1
                )
        strata = tuple(
            ControlStratumCount(
                stratum,
                sum(
                    truth_by_pair[item.pair_id].stratum is stratum
                    for item in eligible
                ),
                sum(
                    truth_by_pair[item.pair_id].stratum is stratum
                    for item in selected_in_source_order
                ),
            )
            for stratum in PairStratum
        )
        panel_summaries.append(
            ControlPanelSummary(
                panel_id=panel.panel_id,
                required_mask_roles=panel.required_mask_roles,
                total_pairs=len(construction.scoring_requests),
                eligible_pairs=len(eligible),
                selected_pairs=len(selected_in_source_order),
                ineligible_pairs=(
                    len(construction.scoring_requests) - len(eligible)
                ),
                cap_applied=len(eligible) > len(selected_in_source_order),
                minimum_met=(
                    len(selected_in_source_order)
                    >= panel.minimum_matched_pairs
                ),
                exclusions=tuple(
                    ControlExclusionCount(reason, count)
                    for reason, count in sorted(exclusion_counts.items())
                ),
                strata=strata,
            )
        )
    request_ids = tuple(
        request.request_id for request in scoring_requests
    )
    if len(request_ids) != len(set(request_ids)):
        raise RuntimeError("visual control request ID collision")
    embedding_tokens = {
        token
        for request in scoring_requests
        for token in (
            request.query_artifact_token,
            request.reference_artifact_token,
        )
    }
    naive_calls = 2 * len(scoring_requests)
    cost = ControlCostSummary(
        scoring_requests=len(scoring_requests),
        naive_embedding_calls=naive_calls,
        unique_embedding_artifacts=len(embedding_tokens),
        reusable_embedding_calls_saved=(
            naive_calls - len(embedding_tokens)
        ),
        transform_tasks=len(transform_tasks),
    )
    return VisualControlAuditPlan(
        pair_set_sha256=construction.result_sha256,
        base_artifact_manifest_sha256=(
            base_artifact_manifest.manifest_sha256
        ),
        base_artifact_verification_sha256=content_sha256(
            base_artifact_verification.to_dict()
        ),
        mask_manifest_sha256=mask_manifest.manifest_sha256,
        mask_verification_sha256=content_sha256(
            mask_verification.to_dict()
        ),
        mask_semantic_verification_sha256=(
            mask_semantic_verification.verification_sha256
        ),
        policy_sha256=policy.policy_sha256,
        panels=tuple(panel_summaries),
        transform_tasks=tuple(
            transform_tasks[token] for token in sorted(transform_tasks)
        ),
        scoring_requests=tuple(scoring_requests),
        evaluation_bindings=tuple(evaluation_bindings),
        cost=cost,
    )


def verify_control_mask_files(
    root: Path,
    manifest: ControlMaskManifest,
) -> ControlMaskVerification:
    if root.is_symlink():
        raise ValueError("mask artifact root must not be a symlink")
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(resolved_root)
    directory_entries = tuple(resolved_root.iterdir())
    if any(entry.is_symlink() for entry in directory_entries):
        raise ValueError("mask artifact directory must not contain symlinks")
    if any(not entry.is_file() for entry in directory_entries):
        raise ValueError("mask artifact directory must contain files only")
    masks = tuple(
        mask for entry in manifest.entries for mask in entry.masks
    )
    expected_names = {mask.relative_path for mask in masks}
    actual_names = {entry.name for entry in directory_entries}
    if expected_names != actual_names:
        raise ValueError(
            "mask artifact directory entries mismatch; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    verified_bytes = 0
    for mask in masks:
        path = resolved_root / mask.relative_path
        initial = path.stat()
        if initial.st_size != mask.byte_size:
            raise ValueError(
                f"mask artifact byte-size mismatch: {mask.artifact_token}"
            )
        digest = sha256_file(path)
        final = path.stat()
        if (
            initial.st_size != final.st_size
            or initial.st_mtime_ns != final.st_mtime_ns
        ):
            raise RuntimeError(
                f"mask artifact changed during verification: "
                f"{mask.artifact_token}"
            )
        if digest != mask.content_sha256:
            raise ValueError(
                f"mask artifact content hash mismatch: {mask.artifact_token}"
            )
        verified_bytes += mask.byte_size
    return ControlMaskVerification(
        mask_manifest_sha256=manifest.manifest_sha256,
        verified_files=len(masks),
        verified_bytes=verified_bytes,
    )


def _artifact_for_control(
    base_token: str,
    kind: VisualControlKind,
    base_entries: dict[str, PairArtifactEntry],
    mask_entries: dict[str, ControlMaskEntry],
    policy: VisualControlPolicy,
    tasks: dict[str, ControlTransformTask],
) -> str:
    if kind is VisualControlKind.ORIGINAL:
        return base_token
    recipe = policy.recipe_for(kind)
    masks = tuple(
        (
            role,
            evidence.artifact_token,
            evidence.content_sha256,
        )
        for role in kind.required_mask_roles
        for evidence in (
            _verified_mask(mask_entries[base_token], role),
        )
    )
    token = control_artifact_token(
        base_content_sha256=base_entries[base_token].content_sha256,
        kind=kind,
        transform_config_sha256=recipe.transform_config_sha256,
        semantics_version=recipe.semantics_version,
        mask_artifacts=masks,
    )
    task = ControlTransformTask(
        control_artifact_token=token,
        base_artifact_token=base_token,
        control_kind=kind,
        transform_config_sha256=recipe.transform_config_sha256,
        semantics_version=recipe.semantics_version,
        mask_artifacts=masks,
    )
    tasks.setdefault(token, task)
    return token


def control_artifact_token(
    *,
    base_content_sha256: str,
    kind: VisualControlKind,
    transform_config_sha256: str,
    semantics_version: str,
    mask_artifacts: tuple[tuple[MaskRole, str, str], ...],
) -> str:
    """Return the content-addressed token for one non-original control."""

    if kind is VisualControlKind.ORIGINAL:
        raise ValueError("ORIGINAL does not have a transformed artifact")
    _validate_sha256(base_content_sha256, "base_content_sha256")
    _validate_sha256(
        transform_config_sha256,
        "transform_config_sha256",
    )
    _require_nonempty(semantics_version, "semantics_version")
    roles = tuple(role for role, _, _ in mask_artifacts)
    if roles != kind.required_mask_roles:
        raise ValueError(
            "control token mask roles must exactly match control kind"
        )
    for _, token, digest in mask_artifacts:
        _require_nonempty(token, "mask artifact token")
        _validate_sha256(digest, "mask artifact content_sha256")
    return "control-" + content_sha256(
        {
            "base_content_sha256": base_content_sha256,
            "control_kind": kind.value,
            "transform_config_sha256": transform_config_sha256,
            "semantics_version": semantics_version,
            "masks": [
                {"role": role.value, "content_sha256": digest}
                for role, _, digest in mask_artifacts
            ],
        }
    )[:24]


def _verified_mask(
    entry: ControlMaskEntry,
    role: MaskRole,
) -> MaskEvidence:
    evidence = entry.mask_for(role)
    if evidence is None or evidence.review_status is not MaskReviewStatus.VERIFIED:
        raise RuntimeError("planner selected an artifact without a verified mask")
    return evidence


def _exclusion_reasons(
    request: PairScoringRequest,
    roles: tuple[MaskRole, ...],
    mask_entries: dict[str, ControlMaskEntry],
) -> tuple[str, ...]:
    reasons: set[str] = set()
    for token in (
        request.query_artifact_token,
        request.reference_artifact_token,
    ):
        for role in roles:
            evidence = mask_entries[token].mask_for(role)
            if evidence is None:
                reasons.add(f"MISSING_{role.value}_MASK")
            elif evidence.review_status is not MaskReviewStatus.VERIFIED:
                reasons.add(f"UNVERIFIED_{role.value}_MASK")
    return tuple(sorted(reasons))


def _stable_rank(*parts: Any) -> str:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_mask_relative_path(value: str, artifact_token: str) -> None:
    _require_nonempty(value, "relative_path")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value:
        raise ValueError("mask relative_path must be one filename")
    if path.suffix.casefold() != ".png":
        raise ValueError("mask artifact must use a .png extension")
    if path.stem != artifact_token:
        raise ValueError("mask filename stem must equal artifact token")


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
