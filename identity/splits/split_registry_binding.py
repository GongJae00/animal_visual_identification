"""Bind a protected split assignment to the identity registry.

Joins the score-blind assignment (identity_token-based) with the
registered_dog_id (UUIDv5) for downstream training and evaluation binding.

Validation guarantees
  - Every identity_token in the assignment appears in the registry.
  - No identity_token maps to more than one registered_dog_id.
  - The database exactly matches its source-bound producer manifest.
  - Identity distribution report covers role, access, dataset, and sample
    disposition.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from identity.registry.identity_registry import (
    REGISTERED_DOG_NAMESPACE,
    compute_identity_token,
    compute_registered_dog_id,
    extract_dataset_name,
    load_registry_manifest,
)
from foundation.provenance import content_sha256


_SCHEMA_VERSION = "cvi.split_registry_binding.v2"
_ASSIGNMENT_KEYS = {
    "schema_version",
    "status",
    "seed_commitment",
    "evidence_root_sha256",
    "policy_sha256",
    "strict_external_boundary",
    "score_inputs_used",
    "label_fields_present",
    "capacity",
    "protocol_cohorts",
    "records",
    "interpretation",
}
_RECEIPT_V1_KEYS = {
    "schema_version",
    "status",
    "seed_commitment",
    "evidence_root_sha256",
    "source_bundle_sha256",
    "graph_sha256",
    "policy_sha256",
    "evidence_bindings",
    "input_file_sha256s",
    "assignment_sha256",
    "evaluator_binding_sha256",
    "capacity",
    "protocol_cohorts",
    "quarantine",
    "tool_provenance",
    "tool_provenance_sha256",
    "interpretation",
    "receipt_sha256",
}
_RECEIPT_V2_KEYS = _RECEIPT_V1_KEYS | {
    "role_exposure_ledger_sha256",
    "role_exposure_receipt_sha256",
}
_RECEIPT_V3_KEYS = _RECEIPT_V2_KEYS | {
    "capacity_mode",
    "requested_role_counts",
    "actual_role_counts",
    "quarantined_identity_counts_by_lane",
    "yt_test_unknown_fpir_power",
}
_REVOKED_RECEIPT_SHA256S = {
    "b381813ab2ca4d981cfdb73aa6bc103bcd2e129b58e29af9f3eb9020b3ad2c88",
    "51acf6533e32d6ad69eefee6b3cc5df06ef9f934e979c74116334eace621ab1b",
}
_REVOKED_ASSIGNMENT_SHA256S = {
    "f4d8774906091b8cd477cc4fe984fc16bad732a0fae87d13643db5e814cb1881",
    "27e77203764153b52b2a3a207249970624c41fc3b98342d8f0d63c40d4be164d",
}
_REVOKED_SEED_COMMITMENTS = {
    "024c469615d8f56ea7f99c6b6e87dce3b2df07aeac0ae3c1fac07fbebdc302ef",
}
_HISTORICAL_V1_RECEIPT_SHA256S = _REVOKED_RECEIPT_SHA256S | {
    "3c5ef13a653e976cf4ba85eea17c63e1b7e3ac8a323bf912cca0092a5448ef7d",
}
_HISTORICAL_V2_RECEIPT_SHA256S = {
    "5f05435ecc8378b9f6ede88c188293c657cbb6d8d647ba0df9be12b37cb367b9",
}
_REGISTRY_MANIFEST_KEYS = {
    "schema_version",
    "generated_at",
    "namespace_uuid",
    "registrations",
    "source_bundle_sha256",
    "tool_provenance",
    "manifest_sha256",
}
_ASSIGNMENT_RECORD_KEYS = {
    "sample_token",
    "identity_token",
    "component_token",
    "dataset_name",
    "source_variant",
    "identity_role",
    "model_access",
    "sample_disposition",
    "paired_original_token",
    "uses",
}
_USE_KEYS = {
    "protocol",
    "episode",
    "gallery_size",
    "shot",
    "role",
    "event_token",
    "primary_query_event_token",
    "bootstrap_cluster_token",
}
_ROLE_ACCESS = {
    "YT_FIT": "MODEL_TRAINING",
    "YT_DEVELOPMENT": "MODEL_SELECTION",
    "YT_CALIBRATION_KNOWN": "DECISION_CALIBRATION_ONLY",
    "YT_CALIBRATION_UNKNOWN": "DECISION_CALIBRATION_ONLY",
    "YT_TEST_KNOWN": "SEALED_FINAL_TEST",
    "YT_TEST_UNKNOWN": "SEALED_FINAL_TEST",
    "DOGFACE_FIT": "SEPARATE_FACE_ONLY_LANE",
    "DOGFACE_DEVELOPMENT": "SEPARATE_FACE_ONLY_LANE",
    "DOGFACE_CALIBRATION": "SEPARATE_FACE_ONLY_LANE",
    "DOGFACE_TEST": "SEPARATE_FACE_ONLY_LANE",
    "MPDD_EXTERNAL_KNOWN": "SEALED_EXTERNAL_ZERO_SHOT",
    "MPDD_EXTERNAL_UNKNOWN": "SEALED_EXTERNAL_ZERO_SHOT",
    "MPDD_EXTERNAL_UNUSED_TRAIN_DOMAIN": "SEALED_EXTERNAL_ZERO_SHOT",
    "SIBETAN_EXTERNAL_KNOWN": "SEALED_EXTERNAL_ZERO_SHOT",
    "SIBETAN_EXTERNAL_UNKNOWN": "SEALED_EXTERNAL_ZERO_SHOT",
}
_ROLE_DATASET = {
    role: (
        "yt-bb-dog"
        if role.startswith("YT_")
        else "dogfacenet224"
        if role.startswith("DOGFACE_")
        else "mpdd"
        if role.startswith("MPDD_")
        else "sibetan"
    )
    for role in _ROLE_ACCESS
}


@dataclass(frozen=True, slots=True)
class IdentityRoleSummary:
    role: str
    access: str
    unique_identities: int
    sample_count: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "role": self.role,
            "access": self.access,
            "unique_identities": self.unique_identities,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class IdentityBinding:
    identity_token: str
    registered_dog_id: str
    dataset_name: str
    identity_role: str
    model_access: str
    sample_disposition: str
    sample_count: int
    sample_tokens: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_token": self.identity_token,
            "registered_dog_id": self.registered_dog_id,
            "dataset_name": self.dataset_name,
            "identity_role": self.identity_role,
            "model_access": self.model_access,
            "sample_disposition": self.sample_disposition,
            "sample_count": self.sample_count,
            "sample_tokens": list(self.sample_tokens),
        }


@dataclass(frozen=True, slots=True)
class SplitRegistryBinding:
    bindings: tuple[IdentityBinding, ...]
    identity_summaries: tuple[IdentityRoleSummary, ...]
    total_identities: int
    total_samples: int
    unregistered_tokens: tuple[str, ...]
    registry_manifest_sha256: str
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported split registry binding schema")
        _require_sha256(
            self.registry_manifest_sha256, "registry_manifest_sha256"
        )

    @property
    def is_valid(self) -> bool:
        return (
            len(self.unregistered_tokens) == 0
            and self.total_identities > 0
            and self.total_samples > 0
            and bool(self.bindings)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "is_valid": self.is_valid,
            "total_identities": self.total_identities,
            "total_samples": self.total_samples,
            "unregistered_tokens": sorted(self.unregistered_tokens),
            "registry_manifest_sha256": self.registry_manifest_sha256,
            "identity_summaries": [s.to_dict() for s in self.identity_summaries],
            "bindings": [b.to_dict() for b in self.bindings],
        }


def build_binding(
    assignment_payload: dict[str, Any],
    registry_db: Path,
    receipt_payload: dict[str, Any],
    registry_manifest_payload: dict[str, Any],
    expected_receipt_sha256: str,
) -> SplitRegistryBinding:
    if set(assignment_payload) != _ASSIGNMENT_KEYS:
        raise ValueError("protected split assignment schema keys differ")
    if assignment_payload.get("schema_version") != (
        "cvi.protected_public_split_assignment.v1"
    ):
        raise ValueError("unsupported protected split assignment schema")
    status = assignment_payload.get("status")
    if status != "PASS_PROTECTED_SPLIT_CONSTRUCTION":
        raise ValueError("protected split assignment did not pass")
    if assignment_payload["score_inputs_used"] is not False or (
        assignment_payload["label_fields_present"] is not False
    ):
        raise ValueError("protected split assignment is not score- and label-blind")
    for field in ("seed_commitment", "evidence_root_sha256", "policy_sha256"):
        value = assignment_payload[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"protected split assignment {field} is invalid")
    if assignment_payload["strict_external_boundary"] != (
        "STRICT_EXTERNAL_DOMAIN_ZERO_SHOT"
    ):
        raise ValueError("protected split assignment boundary is not strict")
    if not isinstance(assignment_payload["capacity"], dict) or not (
        assignment_payload["capacity"]
    ) or not isinstance(assignment_payload["protocol_cohorts"], list) or not (
        assignment_payload["protocol_cohorts"]
    ):
        raise ValueError("protected split assignment summaries are invalid")
    if assignment_payload["capacity"].get("status") != (
        "PASS_PROTECTED_SPLIT_CONSTRUCTION"
    ):
        raise ValueError("protected split assignment capacity did not pass")
    _validate_protocol_cohorts(assignment_payload["protocol_cohorts"])
    if assignment_payload["interpretation"] != (
        "OPAQUE_ROLE_ASSIGNMENT_ONLY_NOT_MODEL_OR_ACCURACY_EVIDENCE"
    ):
        raise ValueError("protected split assignment interpretation differs")
    _validate_assignment_receipt(
        assignment_payload, receipt_payload, expected_receipt_sha256
    )
    records = assignment_payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("assignment records must be a non-empty list")
    _validate_assignment_records(
        records,
        assignment_payload["protocol_cohorts"],
        assignment_payload["capacity"],
    )
    if not registry_db.is_file():
        raise FileNotFoundError(f"identity registry not found: {registry_db}")
    registry = load_registry_manifest(registry_db)
    registry_manifest_sha256 = _validate_registry_manifest(
        registry_manifest_payload,
        registry.records,
        receipt_payload["source_bundle_sha256"],
    )
    token_to_rid: dict[str, str] = {}
    token_to_dsn: dict[str, str] = {}
    for record in registry.records:
        if record.identity_token != compute_identity_token(record.dataset_identity_id):
            raise ValueError("registry identity token differs from source label")
        if record.registered_dog_id != compute_registered_dog_id(
            record.dataset_identity_id
        ):
            raise ValueError("registry registered ID differs from source label")
        if record.dataset_name != extract_dataset_name(record.dataset_identity_id):
            raise ValueError("registry dataset namespace differs from source label")
        token_to_rid[record.identity_token] = record.registered_dog_id
        token_to_dsn[record.identity_token] = record.dataset_name

    identity_aggregator: dict[str, dict[str, Any]] = {}
    unregistered: set[str] = set()
    total_samples = 0

    seen_samples: set[str] = set()
    identity_contracts: dict[str, tuple[str, str, str]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            raise TypeError("assignment record must be an object")
        required = {
            "identity_token", "dataset_name", "identity_role", "model_access",
            "sample_disposition", "sample_token",
        }
        if not required.issubset(rec):
            raise ValueError("assignment record is missing required identity fields")
        token = rec["identity_token"]
        sample_token = rec["sample_token"]
        if not isinstance(token, str) or len(token) != 64:
            raise ValueError("assignment identity token is invalid")
        if not isinstance(sample_token, str) or not sample_token:
            raise ValueError("assignment sample token is invalid")
        if sample_token in seen_samples:
            raise ValueError("assignment contains duplicate sample tokens")
        seen_samples.add(sample_token)
        if token not in token_to_rid:
            unregistered.add(token)
            continue
        total_samples += 1
        role = rec["identity_role"]
        access = rec["model_access"]
        disposition = rec["sample_disposition"]
        dsn = rec["dataset_name"]
        if any(not isinstance(value, str) or not value for value in (
            role, access, disposition, dsn
        )):
            raise ValueError("assignment identity fields must be non-empty strings")
        if dsn != token_to_dsn[token]:
            raise ValueError("assignment dataset namespace differs from registry")

        identity_contract = (role, access, dsn)
        previous_contract = identity_contracts.setdefault(token, identity_contract)
        if previous_contract != identity_contract:
            raise ValueError(
                "assignment places one identity in conflicting roles or access lanes"
            )

        key = (token, role, access, disposition, dsn)
        if key not in identity_aggregator:
            identity_aggregator[key] = {
                "identity_token": token,
                "registered_dog_id": token_to_rid[token],
                "dataset_name": dsn,
                "identity_role": role,
                "model_access": access,
                "sample_disposition": disposition,
                "sample_count": 0,
                "sample_tokens": [],
            }
        identity_aggregator[key]["sample_count"] += 1
        identity_aggregator[key]["sample_tokens"].append(sample_token)

    bindings = tuple(
        IdentityBinding(
            identity_token=v["identity_token"],
            registered_dog_id=v["registered_dog_id"],
            dataset_name=v["dataset_name"],
            identity_role=v["identity_role"],
            model_access=v["model_access"],
            sample_disposition=v["sample_disposition"],
            sample_count=v["sample_count"],
            sample_tokens=tuple(sorted(v["sample_tokens"])),
        )
        for v in sorted(
            identity_aggregator.values(),
            key=lambda x: (x["dataset_name"], x["identity_role"], x["identity_token"]),
        )
    )

    summary_agg: dict[tuple[str, str], tuple[set[str], int]] = defaultdict(
        lambda: (set(), 0)
    )
    for b in bindings:
        key = (b.identity_role, b.model_access)
        summary_agg[key][0].add(b.identity_token)
        summary_agg[key] = (
            summary_agg[key][0],
            summary_agg[key][1] + b.sample_count,
        )

    summaries = tuple(
        IdentityRoleSummary(
            role=role,
            access=access,
            unique_identities=len(ids),
            sample_count=count,
        )
        for (role, access), (ids, count) in sorted(summary_agg.items())
    )

    unique_identity_tokens = {b.identity_token for b in bindings}
    return SplitRegistryBinding(
        bindings=bindings,
        identity_summaries=summaries,
        total_identities=len(unique_identity_tokens),
        total_samples=total_samples,
        unregistered_tokens=tuple(sorted(unregistered)),
        registry_manifest_sha256=registry_manifest_sha256,
    )


def validate_assignment_receipt_binding(
    assignment_payload: dict[str, Any],
    receipt_payload: dict[str, Any],
    expected_receipt_sha256: str,
) -> None:
    """Validate a protected assignment and receipt without evaluator labels."""

    _validate_assignment_receipt(
        assignment_payload, receipt_payload, expected_receipt_sha256
    )


def _validate_assignment_receipt(
    assignment_payload: dict[str, Any],
    receipt_payload: dict[str, Any],
    expected_receipt_sha256: str,
) -> None:
    schema_version = receipt_payload.get("schema_version")
    if schema_version == "cvi.protected_public_split_receipt.v3":
        expected_keys = _RECEIPT_V3_KEYS
    elif schema_version == "cvi.protected_public_split_receipt.v2":
        receipt_sha256 = receipt_payload.get("receipt_sha256")
        if receipt_sha256 not in _HISTORICAL_V2_RECEIPT_SHA256S:
            raise ValueError("protected split receipt v2 is not a persisted artifact")
        expected_keys = _RECEIPT_V2_KEYS
    elif schema_version == "cvi.protected_public_split_receipt.v1":
        receipt_sha256 = receipt_payload.get("receipt_sha256")
        if receipt_sha256 not in _HISTORICAL_V1_RECEIPT_SHA256S:
            raise ValueError("protected split receipt v1 is not a persisted artifact")
        expected_keys = _RECEIPT_V1_KEYS
    else:
        raise ValueError("unsupported protected split receipt schema")
    if set(receipt_payload) != expected_keys:
        raise ValueError("protected split receipt schema keys differ")
    if schema_version in {
        "cvi.protected_public_split_receipt.v2",
        "cvi.protected_public_split_receipt.v3",
    }:
        _require_sha256(
            receipt_payload["role_exposure_ledger_sha256"],
            "role_exposure_ledger_sha256",
        )
        _require_sha256(
            receipt_payload["role_exposure_receipt_sha256"],
            "role_exposure_receipt_sha256",
        )
    if schema_version == "cvi.protected_public_split_receipt.v3":
        capacity = assignment_payload.get("capacity")
        if not isinstance(capacity, dict):
            raise ValueError("protected split assignment capacity differs")
        for field in (
            "capacity_mode",
            "requested_role_counts",
            "actual_role_counts",
            "quarantined_identity_counts_by_lane",
            "yt_test_unknown_fpir_power",
        ):
            if receipt_payload[field] != capacity.get(field):
                raise ValueError(f"protected split receipt {field} differs")
        if receipt_payload["capacity_mode"] != (
            "EVIDENCE_CONSTRAINED_MAXIMAL_COVERAGE"
        ):
            raise ValueError("protected split receipt capacity mode differs")
    _require_sha256(expected_receipt_sha256, "expected receipt digest")
    if receipt_payload["receipt_sha256"] != expected_receipt_sha256:
        raise ValueError("protected split receipt differs from the external pin")
    if (
        receipt_payload["receipt_sha256"] in _REVOKED_RECEIPT_SHA256S
        or receipt_payload["assignment_sha256"] in _REVOKED_ASSIGNMENT_SHA256S
        or receipt_payload["seed_commitment"] in _REVOKED_SEED_COMMITMENTS
    ):
        raise ValueError("protected split receipt has been revoked")
    if receipt_payload["status"] != "PASS_PROTECTED_SPLIT_CONSTRUCTION":
        raise ValueError("protected split receipt did not pass")
    expected_receipt_sha = content_sha256({
        key: value
        for key, value in receipt_payload.items()
        if key != "receipt_sha256"
    })
    if receipt_payload["receipt_sha256"] != expected_receipt_sha:
        raise ValueError("protected split receipt digest differs")
    if receipt_payload["assignment_sha256"] != content_sha256(assignment_payload):
        raise ValueError("protected split receipt does not bind the assignment")
    for field in (
        "status",
        "seed_commitment",
        "evidence_root_sha256",
        "policy_sha256",
        "capacity",
        "protocol_cohorts",
    ):
        if receipt_payload[field] != assignment_payload[field]:
            raise ValueError(f"protected split receipt {field} differs")
    if receipt_payload["interpretation"] != (
        "SPLIT_CONTRACT_BEHAVIOR_ONLY_NOT_PERFORMANCE_OR_DATA_ADMISSION"
    ):
        raise ValueError("protected split receipt interpretation differs")


def validate_assignment_and_evaluator_binding(
    assignment_payload: dict[str, Any],
    receipt_payload: dict[str, Any],
    evaluator_binding_payload: dict[str, Any],
    expected_receipt_sha256: str,
) -> None:
    """Authenticate the public assignment and its private label join."""

    if set(assignment_payload) != _ASSIGNMENT_KEYS:
        raise ValueError("protected split assignment schema keys differ")
    _validate_assignment_receipt(
        assignment_payload, receipt_payload, expected_receipt_sha256
    )
    records = assignment_payload["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("assignment records must be a non-empty list")
    _validate_assignment_records(
        records,
        assignment_payload["protocol_cohorts"],
        assignment_payload["capacity"],
    )
    expected_binding_keys = {
        "schema_version",
        "status",
        "seed_commitment",
        "evidence_root_sha256",
        "records",
        "interpretation",
    }
    if (
        not isinstance(evaluator_binding_payload, dict)
        or set(evaluator_binding_payload) != expected_binding_keys
    ):
        raise ValueError("protected split evaluator binding schema keys differ")
    if evaluator_binding_payload["schema_version"] != (
        "cvi.protected_public_split_evaluator_binding.v1"
    ):
        raise ValueError("unsupported protected split evaluator binding schema")
    if evaluator_binding_payload["status"] != assignment_payload["status"] or (
        evaluator_binding_payload["seed_commitment"]
        != assignment_payload["seed_commitment"]
    ) or (
        evaluator_binding_payload["evidence_root_sha256"]
        != assignment_payload["evidence_root_sha256"]
    ):
        raise ValueError("protected split evaluator binding header differs")
    if evaluator_binding_payload["interpretation"] != (
        "PRIVATE_LABEL_JOIN_FOR_SEALED_EVALUATION_ONLY"
    ):
        raise ValueError("protected split evaluator binding interpretation differs")
    if receipt_payload["evaluator_binding_sha256"] != content_sha256(
        evaluator_binding_payload
    ):
        raise ValueError("protected split receipt does not bind evaluator labels")
    label_records = evaluator_binding_payload["records"]
    if not isinstance(label_records, list) or len(label_records) != len(records):
        raise ValueError("protected split evaluator binding coverage differs")
    label_keys = {
        "sample_token",
        "identity_token",
        "source_sample_id",
        "dataset_identity_id",
        "sequence_token",
        "raw_frame_index",
        "original_split",
        "region",
    }
    assignment_by_sample = {record["sample_token"]: record for record in records}
    seen: set[str] = set()
    for label in label_records:
        if not isinstance(label, dict) or set(label) != label_keys:
            raise ValueError("protected split evaluator record schema differs")
        sample_token = label["sample_token"]
        _require_sha256(sample_token, "evaluator sample token")
        if sample_token in seen or sample_token not in assignment_by_sample:
            raise ValueError("protected split evaluator sample coverage differs")
        seen.add(sample_token)
        assignment_record = assignment_by_sample[sample_token]
        if label["identity_token"] != assignment_record["identity_token"] or (
            label["dataset_identity_id"].split(":", 1)[0]
            != assignment_record["dataset_name"]
        ):
            raise ValueError("protected split evaluator identity binding differs")
    if seen != set(assignment_by_sample):
        raise ValueError("protected split evaluator sample coverage differs")


def _validate_assignment_records(
    records: list[Any], protocol_cohorts: list[Any], capacity: dict[str, Any]
) -> None:
    seen_samples: set[str] = set()
    roles_by_identity: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != _ASSIGNMENT_RECORD_KEYS:
            raise ValueError("protected split assignment record schema differs")
        for field in ("sample_token", "identity_token", "component_token"):
            _require_sha256(record[field], f"assignment {field}")
        if record["sample_token"] in seen_samples:
            raise ValueError("assignment contains duplicate sample tokens")
        seen_samples.add(record["sample_token"])
        role = record["identity_role"]
        if role not in _ROLE_ACCESS or record["model_access"] != _ROLE_ACCESS[role]:
            raise ValueError("assignment role and model access differ")
        if record["dataset_name"] != _ROLE_DATASET[role]:
            raise ValueError("assignment role and dataset differ")
        prior_role = roles_by_identity.setdefault(record["identity_token"], role)
        if prior_role != role:
            raise ValueError("assignment identity has conflicting roles")
        variant = record["source_variant"]
        disposition = record["sample_disposition"]
        paired = record["paired_original_token"]
        if variant == "original":
            if disposition != "PRIMARY_ORACLE_CROP" or paired is not None:
                raise ValueError("original sample disposition differs")
        elif variant == "random_background":
            if (
                record["dataset_name"] != "yt-bb-dog"
                or disposition != "PAIRED_CONTROL_ONLY"
            ):
                raise ValueError("random-background sample disposition differs")
            _require_sha256(paired, "paired_original_token")
        else:
            raise ValueError("assignment source variant differs")
        uses = record["uses"]
        if not isinstance(uses, list):
            raise ValueError("assignment uses must be a list")
        for use in uses:
            _validate_use(use)
    if _summarize_protocol_cohorts(records) != protocol_cohorts:
        raise ValueError("protected split protocol cohorts differ from records")
    if capacity.get("capacity_mode") == "EVIDENCE_CONSTRAINED_MAXIMAL_COVERAGE":
        actual = Counter(roles_by_identity.values())
        actual_payload = capacity.get("actual_role_counts")
        requested = capacity.get("requested_role_counts")
        minimum = capacity.get("minimum_role_counts")
        contracted = capacity.get("contracted_role_counts")
        quarantined = capacity.get("quarantined_identity_counts_by_lane")
        if not all(
            isinstance(value, dict)
            for value in (
                actual_payload,
                requested,
                minimum,
                contracted,
                quarantined,
            )
        ):
            raise ValueError("protected split capacity count bindings differ")
        if actual_payload != {
            role: actual.get(role, 0) for role in actual_payload
        }:
            raise ValueError("protected split actual role counts differ from records")
        if (
            set(actual_payload) != set(requested)
            or set(actual_payload) != set(minimum)
            or set(actual_payload) != set(contracted)
        ):
            raise ValueError("protected split role count key sets differ")
        for role, count in actual_payload.items():
            if role not in _ROLE_ACCESS or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (count, requested[role], minimum[role])
            ):
                raise ValueError("protected split role capacity value differs")
            if count > requested[role] or count < minimum[role]:
                raise ValueError("protected split role count violates its floor")
            if contracted[role] != requested[role] - count:
                raise ValueError("protected split contracted role count differs")
        if any(
            not isinstance(lane, str)
            or not lane
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for lane, count in quarantined.items()
        ):
            raise ValueError("protected split quarantined lane counts differ")


def _validate_use(use: Any) -> None:
    if not isinstance(use, dict) or set(use) != _USE_KEYS:
        raise ValueError("protected split use schema differs")
    for field in ("protocol", "episode"):
        if not isinstance(use[field], str) or not use[field]:
            raise ValueError("protected split use name is invalid")
    if use["role"] not in {"GALLERY", "KNOWN_QUERY", "UNKNOWN_QUERY", "PAIRED_CONTROL"}:
        raise ValueError("protected split use role is invalid")
    for field in ("gallery_size", "shot"):
        value = use[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"protected split use {field} is invalid")
    if use["role"] != "PAIRED_CONTROL" and use["shot"] < 1:
        raise ValueError("protected split non-control shot is invalid")
    _require_sha256(use["event_token"], "event_token")
    query = use["role"] in {"KNOWN_QUERY", "UNKNOWN_QUERY"}
    for field in ("primary_query_event_token", "bootstrap_cluster_token"):
        value = use[field]
        if query:
            _require_sha256(value, field)
        elif value is not None:
            raise ValueError(f"protected split non-query {field} must be null")


def _summarize_protocol_cohorts(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cohorts: dict[tuple[str, str, int, int, str], set[str]] = defaultdict(set)
    for record in records:
        for use in record["uses"]:
            if use["role"] in {"KNOWN_QUERY", "UNKNOWN_QUERY"}:
                cohorts[(
                    use["protocol"],
                    use["episode"],
                    use["gallery_size"],
                    use["shot"],
                    use["role"],
                )].add(record["identity_token"])
    return [
        {
            "protocol": protocol,
            "episode": episode,
            "gallery_size": gallery_size,
            "shot": shot,
            "query_role": role,
            "identity_count": len(tokens),
            "opaque_identity_set_sha256": content_sha256(sorted(tokens)),
        }
        for (protocol, episode, gallery_size, shot, role), tokens in sorted(
            cohorts.items()
        )
    ]


def _validate_registry_manifest(
    payload: dict[str, Any],
    registry_records: tuple[Any, ...],
    expected_source_bundle_sha256: str,
) -> str:
    if not isinstance(payload, dict) or set(payload) != _REGISTRY_MANIFEST_KEYS:
        raise ValueError("identity registry producer manifest schema keys differ")
    if payload["schema_version"] != "cvi.identity_registry_manifest.v1":
        raise ValueError("unsupported identity registry producer manifest schema")
    if payload["namespace_uuid"] != str(REGISTERED_DOG_NAMESPACE):
        raise ValueError("identity registry namespace differs")
    _require_sha256(payload["source_bundle_sha256"], "source_bundle_sha256")
    if payload["source_bundle_sha256"] != expected_source_bundle_sha256:
        raise ValueError("identity registry source bundle differs from split receipt")
    if not isinstance(payload["generated_at"], str) or not payload["generated_at"]:
        raise ValueError("identity registry generated_at is invalid")
    if not isinstance(payload["tool_provenance"], dict) or not payload[
        "tool_provenance"
    ]:
        raise ValueError("identity registry tool provenance is invalid")
    expected_registrations = [record.to_dict() for record in registry_records]
    if payload["registrations"] != expected_registrations:
        raise ValueError("identity registry producer manifest differs from database")
    expected_manifest_sha256 = content_sha256({
        key: value for key, value in payload.items() if key != "manifest_sha256"
    })
    _require_sha256(payload["manifest_sha256"], "registry manifest digest")
    if payload["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("identity registry producer manifest digest differs")
    return expected_manifest_sha256


def _validate_protocol_cohorts(cohorts: list[Any]) -> None:
    expected = {
        "episode",
        "gallery_size",
        "identity_count",
        "opaque_identity_set_sha256",
        "protocol",
        "query_role",
        "shot",
    }
    for cohort in cohorts:
        if not isinstance(cohort, dict) or set(cohort) != expected:
            raise ValueError("protected split protocol cohort schema differs")
        if any(
            not isinstance(cohort[field], str) or not cohort[field]
            for field in ("episode", "protocol", "query_role")
        ):
            raise ValueError("protected split protocol cohort name is invalid")
        digest = cohort["opaque_identity_set_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("protected split protocol cohort digest is invalid")
        for field, minimum in (
            ("gallery_size", 0),
            ("identity_count", 1),
            ("shot", 1),
        ):
            value = cohort[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"protected split protocol cohort {field} is invalid"
                )


def _require_sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
