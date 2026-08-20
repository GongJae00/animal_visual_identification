"""Score-blind face-eligibility overlay for an immutable Full128 route plan."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from shared.foundation.provenance import content_sha256
from data.full_segment.route_plan import (
    validate_full128_route_plan_bundle,
)

OVERLAY_SCHEMA = "evaluation.face_eligibility_overlay.v1"
RECORD_SCHEMA = "evaluation.face_eligibility_record.v1"
CENSUS_SCHEMA = "evaluation.face_eligibility_census.v1"
BUNDLE_SCHEMA = "evaluation.face_eligibility_overlay_bundle.v1"
POLICY_SCHEMA = "evaluation.face_eligibility_policy.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DATASETS = (
    "ap10k-dog",
    "dogfacenet224",
    "dogflw",
    "mpdd",
    "oxford-pets-dog",
    "sibetan",
    "yt-bb-dog",
)
_AP10K_KEYPOINT_COUNT = 17
_AP10K_HEAD_VISIBILITY_INDEXES = (2, 5, 8)  # left eye, right eye, nose


class FaceEligibilityStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    FRAGMENT_ONLY = "FRAGMENT_ONLY"
    NOT_VISIBLE = "NOT_VISIBLE"
    AMBIGUOUS = "AMBIGUOUS"
    UNAVAILABLE = "UNAVAILABLE"


class FaceEvidenceKind(StrEnum):
    PUBLISHER_NATIVE_FACE_CROP = "PUBLISHER_NATIVE_FACE_CROP"
    PUBLISHER_FACE_LANDMARK_CROP = "PUBLISHER_FACE_LANDMARK_CROP"
    PUBLISHER_IDENTITY_CROP = "PUBLISHER_IDENTITY_CROP"
    PUBLISHER_HEAD_ROI = "PUBLISHER_HEAD_ROI"
    PUBLISHER_KEYPOINT_GEOMETRY_PROXY = "PUBLISHER_KEYPOINT_GEOMETRY_PROXY"
    NONE = "NONE"


class FaceProtocolRole(StrEnum):
    FIT = "FIT"
    EXPOSED_DIAGNOSTIC = "EXPOSED_DIAGNOSTIC"
    AUXILIARY = "AUXILIARY"
    FACE_INELIGIBLE = "FACE_INELIGIBLE"


@dataclass(frozen=True, slots=True)
class DogFaceSplitEvidence:
    train_values: tuple[int, ...]
    test_values: tuple[int, ...]
    train_sha256: str
    test_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.train_sha256, "DogFace train class SHA-256")
        _require_sha256(self.test_sha256, "DogFace test class SHA-256")
        if not self.train_values or not self.test_values:
            raise ValueError(
                "DogFace class evidence must contain both publisher splits"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.train_values + self.test_values
        ):
            raise ValueError("DogFace class evidence must contain nonnegative integers")
        if set(self.train_values) & set(self.test_values):
            raise ValueError("DogFace publisher identity splits overlap")

    @property
    def split_by_identity(self) -> dict[int, str]:
        return {
            **{identity: "train" for identity in self.train_values},
            **{identity: "test" for identity in self.test_values},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_sha256": self.train_sha256,
            "test_sha256": self.test_sha256,
            "train_sample_count": len(self.train_values),
            "test_sample_count": len(self.test_values),
            "train_identity_count": len(set(self.train_values)),
            "test_identity_count": len(set(self.test_values)),
        }


def build_face_eligibility_overlay(
    route_plan_bundle: object,
    *,
    dogface_split: DogFaceSplitEvidence,
    ap10k_annotations_by_sha256: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one observation-complete overlay without visual or score inputs."""

    route = validate_full128_route_plan_bundle(route_plan_bundle, verify_files=False)
    route_records = route["plan"]["records"]
    _validate_route_population(route_records)
    dogface_splits = _validated_dogface_splits(route_records, dogface_split)
    ap10k_annotations = _ap10k_annotation_indexes(
        route_records, ap10k_annotations_by_sha256
    )

    records = tuple(
        _classify_record(
            row,
            dogface_splits=dogface_splits,
            ap10k_annotations=ap10k_annotations,
        )
        for row in route_records
    )
    policy = _policy()
    evidence = {
        "dogface_class_split": dogface_split.to_dict(),
        "ap10k_annotation_artifacts": [
            {
                "sha256": sha256,
                "annotation_count": len(index),
            }
            for sha256, index in sorted(ap10k_annotations.items())
        ],
    }
    overlay = {
        "schema_version": OVERLAY_SCHEMA,
        "source_route_plan_sha256": route["plan_sha256"],
        "source_route_plan_bundle_sha256": route["bundle_sha256"],
        "policy": policy,
        "policy_sha256": content_sha256(policy),
        "evidence": evidence,
        "evidence_sha256": content_sha256(evidence),
        "records": list(records),
    }
    overlay = {**overlay, "overlay_sha256": content_sha256(overlay)}
    census = _build_census(records)
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "overlay": overlay,
        "overlay_sha256": overlay["overlay_sha256"],
        "census": census,
        "census_sha256": content_sha256(census),
    }
    return {**payload, "bundle_sha256": content_sha256(payload)}


def validate_face_eligibility_overlay_bundle(value: object) -> dict[str, Any]:
    """Validate hashes, strict record fields, policy, and the exact census."""

    expected = {
        "schema_version",
        "overlay",
        "overlay_sha256",
        "census",
        "census_sha256",
        "bundle_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face-eligibility bundle fields differ")
    bundle = dict(value)
    if bundle["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError("face-eligibility bundle schema differs")
    payload = {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    if bundle["bundle_sha256"] != content_sha256(payload):
        raise ValueError("face-eligibility bundle digest differs")
    overlay = bundle["overlay"]
    expected_overlay = {
        "schema_version",
        "source_route_plan_sha256",
        "source_route_plan_bundle_sha256",
        "policy",
        "policy_sha256",
        "evidence",
        "evidence_sha256",
        "records",
        "overlay_sha256",
    }
    if not isinstance(overlay, dict) or set(overlay) != expected_overlay:
        raise ValueError("face-eligibility overlay fields differ")
    if overlay["schema_version"] != OVERLAY_SCHEMA:
        raise ValueError("face-eligibility overlay schema differs")
    for field in (
        "source_route_plan_sha256",
        "source_route_plan_bundle_sha256",
        "policy_sha256",
        "evidence_sha256",
        "overlay_sha256",
    ):
        _require_sha256(overlay[field], field)
    if overlay["policy"] != _policy() or overlay["policy_sha256"] != content_sha256(
        overlay["policy"]
    ):
        raise ValueError("face-eligibility policy differs")
    if overlay["evidence_sha256"] != content_sha256(overlay["evidence"]):
        raise ValueError("face-eligibility evidence digest differs")
    overlay_payload = {
        key: item for key, item in overlay.items() if key != "overlay_sha256"
    }
    if overlay["overlay_sha256"] != content_sha256(overlay_payload):
        raise ValueError("face-eligibility overlay digest differs")
    if bundle["overlay_sha256"] != overlay["overlay_sha256"]:
        raise ValueError("face-eligibility overlay binding differs")
    if not isinstance(overlay["records"], list) or not overlay["records"]:
        raise ValueError("face-eligibility records must not be empty")
    records = tuple(_validate_record(record) for record in overlay["records"])
    if [record["sample_token"] for record in records] != sorted(
        record["sample_token"] for record in records
    ):
        raise ValueError("face-eligibility records must be sample-token sorted")
    if len(records) != len({record["sample_token"] for record in records}):
        raise ValueError("face-eligibility sample tokens must be unique")
    census = _build_census(records)
    if bundle["census"] != census or bundle["census_sha256"] != content_sha256(census):
        raise ValueError("face-eligibility census differs")
    return bundle


def _classify_record(
    row: Mapping[str, Any],
    *,
    dogface_splits: Mapping[int, str],
    ap10k_annotations: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    dataset = row["dataset_name"]
    status = FaceEligibilityStatus.UNAVAILABLE
    evidence_kind = FaceEvidenceKind.NONE
    authority_sha256: str | None = None
    split = row["split"]
    reason = "NO_PUBLISHER_FACE_GEOMETRY"

    if dataset == "dogfacenet224":
        identity = int(row["identity_metadata"]["raw_identity_id"])
        split = dogface_splits[identity]
        status = FaceEligibilityStatus.ELIGIBLE
        evidence_kind = FaceEvidenceKind.PUBLISHER_NATIVE_FACE_CROP
        authority_sha256 = row["record_sha256"]
        reason = "PUBLISHER_NATIVE_FACE_CROP"
    elif dataset == "dogflw":
        status = FaceEligibilityStatus.ELIGIBLE
        evidence_kind = FaceEvidenceKind.PUBLISHER_FACE_LANDMARK_CROP
        authority_sha256 = row["record_sha256"]
        reason = "PUBLISHER_FACE46_CROP"
    elif dataset == "mpdd":
        status = FaceEligibilityStatus.ELIGIBLE
        evidence_kind = FaceEvidenceKind.PUBLISHER_IDENTITY_CROP
        authority_sha256 = row["record_sha256"]
        reason = "PUBLISHER_MULTI_POSE_IDENTITY_CROP"
    elif (
        dataset == "oxford-pets-dog"
        and "head_pose" in row["source_metadata"]["adapter_metadata"]
    ):
        status = FaceEligibilityStatus.ELIGIBLE
        evidence_kind = FaceEvidenceKind.PUBLISHER_HEAD_ROI
        authority_sha256 = row["record_sha256"]
        reason = "PUBLISHER_XML_HEAD_ROI"
    elif dataset == "ap10k-dog" and _ap10k_face_proxy(row, ap10k_annotations):
        status = FaceEligibilityStatus.ELIGIBLE
        evidence_kind = FaceEvidenceKind.PUBLISHER_KEYPOINT_GEOMETRY_PROXY
        authority_sha256 = row["route_evidence"]["annotation_artifact"]["sha256"]
        reason = "NOSE_AND_AT_LEAST_ONE_EYE_VISIBLE"

    role, objective_roles = _protocol_use(dataset, split, status, row)
    payload = {
        "schema_version": RECORD_SCHEMA,
        "sample_token": row["sample_token"],
        "dataset_name": dataset,
        "source_record_sha256": row["record_sha256"],
        "publisher_split": split,
        "registered_identity_id": row["identity_metadata"]["registered_identity_id"],
        "status": status.value,
        "evidence_kind": evidence_kind.value,
        "evidence_authority_sha256": authority_sha256,
        "reason": reason,
        "face_protocol_role": role.value,
        "objective_roles": objective_roles,
        "gallery_query_eligible": status is FaceEligibilityStatus.ELIGIBLE
        and row["identity_metadata"]["registered_identity_id"] is not None,
        "score_inputs_used": False,
        "learned_candidate_used": False,
    }
    return {**payload, "record_sha256": content_sha256(payload)}


def _protocol_use(
    dataset: str,
    split: str,
    status: FaceEligibilityStatus,
    row: Mapping[str, Any],
) -> tuple[FaceProtocolRole, list[str]]:
    if status is not FaceEligibilityStatus.ELIGIBLE:
        return FaceProtocolRole.FACE_INELIGIBLE, []
    if dataset == "dogfacenet224" and split == "train":
        return FaceProtocolRole.FIT, ["SELF_SUPERVISION", "SUPERVISED_IDENTITY"]
    if dataset in {"dogfacenet224", "mpdd"}:
        return FaceProtocolRole.EXPOSED_DIAGNOSTIC, [
            "EXPOSED_RETRIEVAL",
            "ROBUSTNESS",
        ]
    roles = ["LOCALIZATION", "SELF_SUPERVISION"]
    if dataset in {"ap10k-dog", "dogflw"}:
        roles.append("PROVISIONAL_IDENTITY_MINING")
    if row["identity_metadata"]["registered_identity_id"] is not None:
        raise ValueError(
            "auxiliary face record unexpectedly carries registered identity"
        )
    return FaceProtocolRole.AUXILIARY, sorted(roles)


def _ap10k_face_proxy(
    row: Mapping[str, Any],
    indexes: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> bool:
    evidence = row["route_evidence"]
    artifact_sha256 = evidence["annotation_artifact"]["sha256"]
    annotation = indexes[artifact_sha256][evidence["annotation_id"]]
    if (
        annotation.get("image_id") != evidence["image_id"]
        or annotation.get("category_id") != 8
    ):
        raise ValueError("AP-10K overlay annotation association differs")
    keypoints = annotation.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) != _AP10K_KEYPOINT_COUNT * 3:
        raise TypeError("AP-10K overlay keypoint schema differs")
    left_eye, right_eye, nose = (
        keypoints[index] for index in _AP10K_HEAD_VISIBILITY_INDEXES
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (left_eye, right_eye, nose)
    ):
        raise TypeError("AP-10K overlay visibility values differ")
    return nose > 0 and (left_eye > 0 or right_eye > 0)


def _ap10k_annotation_indexes(
    route_records: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    required = {
        row["route_evidence"]["annotation_artifact"]["sha256"]
        for row in route_records
        if row["dataset_name"] == "ap10k-dog"
    }
    if set(documents) != required:
        raise ValueError("AP-10K overlay annotation artifact set differs")
    indexes: dict[str, dict[int, Mapping[str, Any]]] = {}
    for sha256, document in documents.items():
        _require_sha256(sha256, "AP-10K annotation SHA-256")
        annotations = document.get("annotations")
        if not isinstance(annotations, list):
            raise TypeError("AP-10K overlay annotations must be an array")
        index: dict[int, Mapping[str, Any]] = {}
        for annotation in annotations:
            if not isinstance(annotation, Mapping):
                raise TypeError("AP-10K overlay annotation must be an object")
            annotation_id = annotation.get("id")
            if isinstance(annotation_id, bool) or not isinstance(annotation_id, int):
                raise TypeError("AP-10K overlay annotation ID differs")
            if annotation_id in index:
                raise ValueError("AP-10K overlay annotation ID is duplicated")
            index[annotation_id] = annotation
        indexes[sha256] = index
    return indexes


def _validated_dogface_splits(
    records: Sequence[Mapping[str, Any]], evidence: DogFaceSplitEvidence
) -> dict[int, str]:
    observed = Counter(
        int(row["identity_metadata"]["raw_identity_id"])
        for row in records
        if row["dataset_name"] == "dogfacenet224"
    )
    expected = Counter(evidence.train_values) + Counter(evidence.test_values)
    if observed != expected:
        raise ValueError("DogFace overlay class multiplicities differ from route plan")
    return evidence.split_by_identity


def _validate_route_population(records: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(row["dataset_name"] for row in records)
    if tuple(sorted(counts)) != _DATASETS:
        raise ValueError("face overlay requires the canonical seven-dataset route plan")
    if [row["sample_token"] for row in records] != sorted(
        row["sample_token"] for row in records
    ):
        raise ValueError("source route records must be sample-token sorted")


def _build_census(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []
    for dataset in _DATASETS:
        rows = [row for row in records if row["dataset_name"] == dataset]
        statuses = Counter(row["status"] for row in rows)
        roles = Counter(row["face_protocol_role"] for row in rows)
        eligible_by_identity: defaultdict[str, int] = defaultdict(int)
        for row in rows:
            identity = row["registered_identity_id"]
            if row["gallery_query_eligible"] and identity is not None:
                eligible_by_identity[identity] += 1
        dataset_rows.append(
            {
                "dataset_name": dataset,
                "observation_count": len(rows),
                "status_counts": {
                    status.value: statuses[status.value]
                    for status in FaceEligibilityStatus
                },
                "face_protocol_role_counts": {
                    role.value: roles[role.value] for role in FaceProtocolRole
                },
                "gallery_query_identity_count": len(eligible_by_identity),
                "retrieval_feasible_identity_count": sum(
                    count >= 2 for count in eligible_by_identity.values()
                ),
            }
        )
    eligible_count = sum(
        row["status"] == FaceEligibilityStatus.ELIGIBLE.value for row in records
    )
    return {
        "schema_version": CENSUS_SCHEMA,
        "observation_count": len(records),
        "eligible_count": eligible_count,
        "ineligible_count": len(records) - eligible_count,
        "datasets": dataset_rows,
        "score_inputs_used": False,
        "learned_candidate_used": False,
    }


def _validate_record(value: object) -> dict[str, Any]:
    expected = {
        "schema_version",
        "sample_token",
        "dataset_name",
        "source_record_sha256",
        "publisher_split",
        "registered_identity_id",
        "status",
        "evidence_kind",
        "evidence_authority_sha256",
        "reason",
        "face_protocol_role",
        "objective_roles",
        "gallery_query_eligible",
        "score_inputs_used",
        "learned_candidate_used",
        "record_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face-eligibility record fields differ")
    record = dict(value)
    if record["schema_version"] != RECORD_SCHEMA:
        raise ValueError("face-eligibility record schema differs")
    for field in ("sample_token", "source_record_sha256", "record_sha256"):
        _require_sha256(record[field], field)
    if record["dataset_name"] not in _DATASETS:
        raise ValueError("face-eligibility record dataset differs")
    status = FaceEligibilityStatus(record["status"])
    evidence_kind = FaceEvidenceKind(record["evidence_kind"])
    role = FaceProtocolRole(record["face_protocol_role"])
    if status is FaceEligibilityStatus.ELIGIBLE:
        if evidence_kind is FaceEvidenceKind.NONE:
            raise ValueError("eligible face record requires publisher evidence")
        _require_sha256(record["evidence_authority_sha256"], "evidence authority")
        if role is FaceProtocolRole.FACE_INELIGIBLE:
            raise ValueError("eligible face record cannot have ineligible role")
    elif (
        evidence_kind is not FaceEvidenceKind.NONE
        or record["evidence_authority_sha256"] is not None
        or role is not FaceProtocolRole.FACE_INELIGIBLE
    ):
        raise ValueError("ineligible face record cannot carry positive evidence")
    if (
        record["score_inputs_used"] is not False
        or record["learned_candidate_used"] is not False
    ):
        raise ValueError("face eligibility must remain score- and model-blind")
    if not isinstance(record["objective_roles"], list) or record[
        "objective_roles"
    ] != sorted(set(record["objective_roles"])):
        raise ValueError("face objective roles must be sorted and unique")
    expected_gallery = (
        status is FaceEligibilityStatus.ELIGIBLE
        and record["registered_identity_id"] is not None
    )
    if record["gallery_query_eligible"] is not expected_gallery:
        raise ValueError("face gallery/query eligibility differs")
    payload = {key: item for key, item in record.items() if key != "record_sha256"}
    if record["record_sha256"] != content_sha256(payload):
        raise ValueError("face-eligibility record digest differs")
    return record


def _policy() -> dict[str, Any]:
    return {
        "schema_version": POLICY_SCHEMA,
        "decision_basis": "PUBLISHER_METADATA_ONLY",
        "ap10k_required_all_keypoints": ["nose_center"],
        "ap10k_required_any_keypoints": ["left_eye", "right_eye"],
        "gallery_query_excluded_statuses": [
            FaceEligibilityStatus.AMBIGUOUS.value,
            FaceEligibilityStatus.FRAGMENT_ONLY.value,
            FaceEligibilityStatus.NOT_VISIBLE.value,
            FaceEligibilityStatus.UNAVAILABLE.value,
        ],
        "score_inputs_used": False,
        "learned_candidate_used": False,
    }


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


__all__ = [
    "DogFaceSplitEvidence",
    "FaceEligibilityStatus",
    "FaceEvidenceKind",
    "FaceProtocolRole",
    "build_face_eligibility_overlay",
    "validate_face_eligibility_overlay_bundle",
]
