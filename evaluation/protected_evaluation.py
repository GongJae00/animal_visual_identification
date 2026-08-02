"""Receipt-bound protected evaluation preparation, execution, and verification."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, ClassVar

from jsonschema import Draft202012Validator

from foundation.protected_io import (
    StrictJsonDocument,
    json_document_bytes,
    read_strict_json_document,
    write_private_json_directory_bundle,
)
from foundation.provenance import content_sha256
from identity_governance.role_exposure import (
    CandidateRoleAssignment,
    CandidateRoleRecord,
    ExposureDeclarationKind,
    ExposureStage,
    RoleExposureDeclaration,
    RoleExposureDeclarationRecord,
    RoleExposureLedger,
    RoleExposureReceipt,
    validate_candidate_assignment,
    verify_role_exposure_receipt,
)


REPORT_SCHEMA_VERSION = "cvi.evaluation.report.v3"
REPORT_SCHEMA_FILENAME = "cvi.evaluation.report.v3.schema.json"
REPORT_PROTOCOL_STATUS = "RECEIPT_CHAIN_VERIFIED"
REPORT_INTERPRETATION = (
    "RECEIPT_CHAIN_INTEGRITY_ONLY_SCIENTIFIC_VALIDITY_REQUIRES_EXTERNAL_PROTOCOL_MODEL_DATA_REVIEW"
)
PREPARATION_FILENAMES = (
    "policy_receipt.json",
    "input_receipt.json",
    "advanced_exposure_declaration.json",
    "plan_receipt.json",
)


@dataclass(frozen=True, slots=True)
class ProtectedEvaluationRoleBinding:
    name: str
    protocol: str
    episode: str
    gallery_size: int
    shot: int
    role: str
    schema_version: str = "cvi.protected_evaluation_role_binding.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.protected_evaluation_role_binding.v1":
            raise ValueError("unsupported protected evaluation role binding")
        if self.name not in {"gallery", "queries"}:
            raise ValueError("protected evaluation input name differs")
        _text(self.protocol, "role protocol", 128)
        _text(self.episode, "role episode", 128)
        _positive_int(self.gallery_size, "role gallery_size")
        _positive_int(self.shot, "role shot")
        expected_role = "GALLERY" if self.name == "gallery" else "KNOWN_QUERY"
        if self.role != expected_role:
            raise ValueError("protected evaluation role differs from input name")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtectedEvaluationRoleBinding:
        _exact(payload, set(cls.__dataclass_fields__), "protected role binding")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ProtectedEvaluationPolicy:
    role_bindings: tuple[ProtectedEvaluationRoleBinding, ...]
    rank_ks: tuple[int, ...]
    bootstrap_resamples: int
    bootstrap_seed: int
    maximum_json_bytes: int
    maximum_json_depth: int
    maximum_json_nodes: int
    maximum_json_keys: int
    maximum_json_array_length: int
    maximum_json_string_characters: int
    maximum_json_number_characters: int
    maximum_samples_per_input: int
    maximum_embedding_dimension: int
    maximum_total_embedding_values: int
    maximum_score_matrix_elements: int
    score_dtype: str
    metric: str
    self_match_policy: str
    schema_version: str = "cvi.protected_evaluation_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.protected_evaluation_policy.v1":
            raise ValueError("unsupported protected evaluation policy")
        if not isinstance(self.role_bindings, tuple) or tuple(
            item.name for item in self.role_bindings
        ) != ("gallery", "queries"):
            raise ValueError("protected role bindings must be gallery then queries")
        if any(not isinstance(item, ProtectedEvaluationRoleBinding) for item in self.role_bindings):
            raise TypeError("protected role bindings have the wrong type")
        if not isinstance(self.rank_ks, tuple) or self.rank_ks != tuple(sorted(set(self.rank_ks))):
            raise ValueError("rank_ks must be sorted and unique")
        for value in self.rank_ks:
            _positive_int(value, "rank_k")
        for name in (
            "bootstrap_resamples",
            "maximum_json_bytes",
            "maximum_json_depth",
            "maximum_json_nodes",
            "maximum_json_keys",
            "maximum_json_array_length",
            "maximum_json_string_characters",
            "maximum_json_number_characters",
            "maximum_samples_per_input",
            "maximum_embedding_dimension",
            "maximum_total_embedding_values",
            "maximum_score_matrix_elements",
        ):
            _positive_int(getattr(self, name), name)
        if isinstance(self.bootstrap_seed, bool) or not isinstance(self.bootstrap_seed, int) or self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be a nonnegative integer")
        if self.score_dtype != "float64" or self.metric != "cosine":
            raise ValueError("protected score representation or metric differs")
        if self.self_match_policy != "exclude":
            raise ValueError("protected evaluation requires self-match exclusion")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["role_bindings"] = [item.to_dict() for item in self.role_bindings]
        result["rank_ks"] = list(self.rank_ks)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtectedEvaluationPolicy:
        _exact(payload, set(cls.__dataclass_fields__), "protected evaluation policy")
        if not isinstance(payload["role_bindings"], list) or not isinstance(payload["rank_ks"], list):
            raise TypeError("protected policy collections must be lists")
        values = dict(payload)
        values["role_bindings"] = tuple(
            ProtectedEvaluationRoleBinding.from_dict(item)
            for item in values["role_bindings"]
        )
        values["rank_ks"] = tuple(values["rank_ks"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ProtectedEvaluationExternalPins:
    policy_raw_sha256: str
    split_assignment_raw_sha256: str
    split_receipt_raw_sha256: str
    exposure_ledger_raw_sha256: str
    exposure_receipt_raw_sha256: str
    gallery_raw_sha256: str
    gallery_production_receipt_sha256: str
    queries_raw_sha256: str
    queries_production_receipt_sha256: str
    schema_version: str = "cvi.protected_evaluation_external_pins.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.protected_evaluation_external_pins.v1":
            raise ValueError("unsupported protected evaluation external pins")
        for name in self.__dataclass_fields__:
            if name != "schema_version":
                _sha256(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtectedEvaluationExternalPins:
        _exact(payload, set(cls.__dataclass_fields__), "protected external pins")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ProtectedEmbeddingRecord:
    sample_token: str
    identity_token: str
    public_subject_token: str
    template_token: str
    embedding: tuple[float, ...]
    schema_version: str = "cvi.protected_embedding_record.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.protected_embedding_record.v1":
            raise ValueError("unsupported protected embedding record")
        for name in ("sample_token", "identity_token", "public_subject_token", "template_token"):
            _sha256(getattr(self, name), name)
        if len({self.sample_token, self.identity_token, self.public_subject_token, self.template_token}) != 4:
            raise ValueError("protected embedding token namespaces must be distinct")
        if not isinstance(self.embedding, tuple) or not self.embedding:
            raise ValueError("protected embedding must be a non-empty tuple")
        for value in self.embedding:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("protected embedding values must be finite real numbers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_token": self.sample_token,
            "identity_token": self.identity_token,
            "public_subject_token": self.public_subject_token,
            "template_token": self.template_token,
            "embedding": list(self.embedding),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtectedEmbeddingRecord:
        _exact(payload, set(cls.__dataclass_fields__), "protected embedding record")
        if not isinstance(payload["embedding"], list):
            raise TypeError("protected embedding must be a list")
        values = dict(payload)
        values["embedding"] = tuple(values["embedding"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ProtectedEmbeddingManifest:
    input_name: str
    production_receipt_sha256: str
    records: tuple[ProtectedEmbeddingRecord, ...]
    interpretation: str = "RECEIPT_BOUND_EMBEDDINGS_ONLY_NO_INFERENCE"
    schema_version: str = "cvi.protected_embedding_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.protected_embedding_manifest.v1":
            raise ValueError("unsupported protected embedding manifest")
        if self.input_name not in {"gallery", "queries"}:
            raise ValueError("protected embedding input name differs")
        _sha256(self.production_receipt_sha256, "production_receipt_sha256")
        if self.interpretation != "RECEIPT_BOUND_EMBEDDINGS_ONLY_NO_INFERENCE":
            raise ValueError("protected embedding interpretation differs")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("protected embedding records must not be empty")
        if any(not isinstance(item, ProtectedEmbeddingRecord) for item in self.records):
            raise TypeError("protected embedding records have the wrong type")
        if self.records != tuple(sorted(self.records, key=lambda item: item.sample_token)):
            raise ValueError("protected embedding records must be sample-token sorted")
        _unique(tuple(item.sample_token for item in self.records), "sample tokens")
        _unique(tuple(item.template_token for item in self.records), "template tokens")
        dimensions = {len(item.embedding) for item in self.records}
        if len(dimensions) != 1:
            raise ValueError("protected embedding dimensions differ")
        links: dict[str, str] = {}
        reverse: dict[str, str] = {}
        for item in self.records:
            if links.setdefault(item.identity_token, item.public_subject_token) != item.public_subject_token:
                raise ValueError("identity maps to multiple public subjects")
            if reverse.setdefault(item.public_subject_token, item.identity_token) != item.identity_token:
                raise ValueError("public subject maps to multiple identities")

    @property
    def dimension(self) -> int:
        return len(self.records[0].embedding)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_name": self.input_name,
            "production_receipt_sha256": self.production_receipt_sha256,
            "records": [item.to_dict() for item in self.records],
            "interpretation": self.interpretation,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtectedEmbeddingManifest:
        _exact(payload, set(cls.__dataclass_fields__), "protected embedding manifest")
        if not isinstance(payload["records"], list):
            raise TypeError("protected embedding records must be a list")
        return cls(
            input_name=payload["input_name"],
            production_receipt_sha256=payload["production_receipt_sha256"],
            records=tuple(ProtectedEmbeddingRecord.from_dict(item) for item in payload["records"]),
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ProtectedFileReceipt:
    name: str
    raw_sha256: str
    canonical_payload_sha256: str
    byte_size: int
    external_receipt_sha256: str | None
    schema_version: str = "cvi.protected_file_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.protected_file_receipt.v1":
            raise ValueError("unsupported protected file receipt")
        _text(self.name, "protected file name", 128)
        _sha256(self.raw_sha256, "raw_sha256")
        _sha256(self.canonical_payload_sha256, "canonical_payload_sha256")
        _positive_int(self.byte_size, "byte_size")
        if self.external_receipt_sha256 is not None:
            _sha256(self.external_receipt_sha256, "external_receipt_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtectedFileReceipt:
        _exact(payload, set(cls.__dataclass_fields__), "protected file receipt")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ProtectedEvaluationPolicyReceipt:
    policy: ProtectedFileReceipt
    policy_sha256: str
    external_pins: ProtectedFileReceipt
    interpretation: str = "FROZEN_POLICY_AND_EXTERNAL_PINS_BEFORE_SCORING"
    schema_version: str = "cvi.protected_evaluation_policy_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.protected_evaluation_policy_receipt.v1":
            raise ValueError("unsupported protected policy receipt")
        if self.policy.name != "policy" or self.external_pins.name != "external_pins":
            raise ValueError("protected policy receipt names differ")
        _sha256(self.policy_sha256, "policy_sha256")
        if self.policy.canonical_payload_sha256 != self.policy_sha256:
            raise ValueError("protected policy semantic hash differs")
        if self.interpretation != "FROZEN_POLICY_AND_EXTERNAL_PINS_BEFORE_SCORING":
            raise ValueError("protected policy receipt interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "policy_sha256": self.policy_sha256,
            "external_pins": self.external_pins.to_dict(),
            "interpretation": self.interpretation,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtectedEvaluationPolicyReceipt:
        _exact(payload, set(cls.__dataclass_fields__), "protected policy receipt")
        return cls(
            policy=ProtectedFileReceipt.from_dict(payload["policy"]),
            policy_sha256=payload["policy_sha256"],
            external_pins=ProtectedFileReceipt.from_dict(payload["external_pins"]),
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ProtectedEvaluationInputReceipt:
    files: tuple[ProtectedFileReceipt, ...]
    sample_counts: tuple[tuple[str, int], ...]
    embedding_dimension: int
    total_embedding_values: int
    score_matrix_elements: int
    interpretation: str = "ALL_SCORING_INPUTS_HASHED_AND_RESOURCE_CHECKED_BEFORE_ALLOCATION"
    schema_version: str = "cvi.protected_evaluation_input_receipt.v1"

    EXPECTED_NAMES: ClassVar[tuple[str, ...]] = (
        "exposure_ledger",
        "exposure_receipt",
        "gallery",
        "queries",
        "split_assignment",
        "split_receipt",
    )

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.protected_evaluation_input_receipt.v1":
            raise ValueError("unsupported protected input receipt")
        if tuple(item.name for item in self.files) != self.EXPECTED_NAMES:
            raise ValueError("protected input receipt file set differs")
        if self.sample_counts != tuple(sorted(self.sample_counts)) or tuple(name for name, _ in self.sample_counts) != ("gallery", "queries"):
            raise ValueError("protected sample counts differ")
        for _, count in self.sample_counts:
            _positive_int(count, "sample count")
        for name in ("embedding_dimension", "total_embedding_values", "score_matrix_elements"):
            _positive_int(getattr(self, name), name)
        if self.interpretation != "ALL_SCORING_INPUTS_HASHED_AND_RESOURCE_CHECKED_BEFORE_ALLOCATION":
            raise ValueError("protected input receipt interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [item.to_dict() for item in self.files],
            "sample_counts": [list(item) for item in self.sample_counts],
            "embedding_dimension": self.embedding_dimension,
            "total_embedding_values": self.total_embedding_values,
            "score_matrix_elements": self.score_matrix_elements,
            "interpretation": self.interpretation,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtectedEvaluationInputReceipt:
        _exact(
            payload,
            set(cls.__dataclass_fields__) - {"EXPECTED_NAMES"},
            "protected input receipt",
        )
        if not isinstance(payload["files"], list) or not isinstance(payload["sample_counts"], list):
            raise TypeError("protected input receipt collections must be lists")
        counts = _pairs(payload["sample_counts"], "sample counts")
        return cls(
            files=tuple(ProtectedFileReceipt.from_dict(item) for item in payload["files"]),
            sample_counts=tuple((name, count) for name, count in counts),
            embedding_dimension=payload["embedding_dimension"],
            total_embedding_values=payload["total_embedding_values"],
            score_matrix_elements=payload["score_matrix_elements"],
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ProtectedEvaluationPlanReceipt:
    evaluation_token: str
    policy_receipt_sha256: str
    input_receipt_sha256: str
    split_assignment_sha256: str
    prior_exposure_ledger_sha256: str
    prior_exposure_receipt_sha256: str
    advanced_exposure_declaration_sha256: str
    tool_provenance: dict[str, Any]
    tool_provenance_sha256: str
    status: str = "PRE_SCORE_EXPOSURE_PUBLISHED"
    interpretation: str = "PLAN_ONLY_NO_SCORE_OR_PERFORMANCE_EVIDENCE"
    schema_version: str = "cvi.protected_evaluation_plan_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.protected_evaluation_plan_receipt.v1":
            raise ValueError("unsupported protected evaluation plan receipt")
        for name in (
            "evaluation_token",
            "policy_receipt_sha256",
            "input_receipt_sha256",
            "split_assignment_sha256",
            "prior_exposure_ledger_sha256",
            "prior_exposure_receipt_sha256",
            "advanced_exposure_declaration_sha256",
            "tool_provenance_sha256",
        ):
            _sha256(getattr(self, name), name)
        if not isinstance(self.tool_provenance, dict) or content_sha256(self.tool_provenance) != self.tool_provenance_sha256:
            raise ValueError("protected plan tool provenance differs")
        if self.status != "PRE_SCORE_EXPOSURE_PUBLISHED" or self.interpretation != "PLAN_ONLY_NO_SCORE_OR_PERFORMANCE_EVIDENCE":
            raise ValueError("protected plan status or interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtectedEvaluationPlanReceipt:
        _exact(payload, set(cls.__dataclass_fields__), "protected plan receipt")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ProtectedEvaluationOutputReceipt:
    plan_receipt_sha256: str
    policy_receipt_sha256: str
    input_receipt_sha256: str
    advanced_exposure_declaration_sha256: str
    report_raw_sha256: str
    report_canonical_payload_sha256: str
    report_byte_size: int
    report_schema_raw_sha256: str
    report_schema_canonical_payload_sha256: str
    evaluator_provenance: dict[str, Any]
    evaluator_provenance_sha256: str
    status: str = REPORT_PROTOCOL_STATUS
    interpretation: str = REPORT_INTERPRETATION
    schema_version: str = "cvi.protected_evaluation_output_receipt.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.protected_evaluation_output_receipt.v2":
            raise ValueError("unsupported protected output receipt")
        for name in (
            "plan_receipt_sha256",
            "policy_receipt_sha256",
            "input_receipt_sha256",
            "advanced_exposure_declaration_sha256",
            "report_raw_sha256",
            "report_canonical_payload_sha256",
            "report_schema_raw_sha256",
            "report_schema_canonical_payload_sha256",
            "evaluator_provenance_sha256",
        ):
            _sha256(getattr(self, name), name)
        _positive_int(self.report_byte_size, "report_byte_size")
        if not isinstance(self.evaluator_provenance, dict) or content_sha256(self.evaluator_provenance) != self.evaluator_provenance_sha256:
            raise ValueError("protected evaluator provenance differs")
        if self.status != REPORT_PROTOCOL_STATUS or self.interpretation != REPORT_INTERPRETATION:
            raise ValueError("protected output status or interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtectedEvaluationOutputReceipt:
        _exact(payload, set(cls.__dataclass_fields__), "protected output receipt")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ProtectedEvaluationPreparation:
    policy: ProtectedEvaluationPolicy
    policy_receipt: ProtectedEvaluationPolicyReceipt
    input_receipt: ProtectedEvaluationInputReceipt
    plan_receipt: ProtectedEvaluationPlanReceipt
    advanced_exposure_declaration: RoleExposureDeclaration
    gallery: ProtectedEmbeddingManifest
    queries: ProtectedEmbeddingManifest


def prepare_protected_evaluation(
    *,
    policy_path: Path,
    external_pins_path: Path,
    expected_external_pins_raw_sha256: str,
    split_assignment_path: Path,
    split_receipt_path: Path,
    exposure_ledger_path: Path,
    exposure_receipt_path: Path,
    gallery_path: Path,
    queries_path: Path,
    output_directory: Path,
    tool_provenance: dict[str, Any],
) -> ProtectedEvaluationPlanReceipt:
    """Validate actual history and publish the advanced exposure before scoring."""

    _sha256(expected_external_pins_raw_sha256, "expected external pins raw hash")
    pins_document = read_strict_json_document(external_pins_path, maximum_bytes=1_048_576)
    if pins_document.raw_sha256 != expected_external_pins_raw_sha256:
        raise ValueError("external pins file differs from external anchor")
    pins = ProtectedEvaluationExternalPins.from_dict(pins_document.payload)
    policy_document = read_strict_json_document(policy_path, maximum_bytes=1_048_576)
    _raw_pin(policy_document, pins.policy_raw_sha256, "policy")
    policy = ProtectedEvaluationPolicy.from_dict(policy_document.payload)
    documents = _read_source_documents(
        policy,
        split_assignment_path=split_assignment_path,
        split_receipt_path=split_receipt_path,
        exposure_ledger_path=exposure_ledger_path,
        exposure_receipt_path=exposure_receipt_path,
        gallery_path=gallery_path,
        queries_path=queries_path,
    )
    expected_raw = {
        "split_assignment": pins.split_assignment_raw_sha256,
        "split_receipt": pins.split_receipt_raw_sha256,
        "exposure_ledger": pins.exposure_ledger_raw_sha256,
        "exposure_receipt": pins.exposure_receipt_raw_sha256,
        "gallery": pins.gallery_raw_sha256,
        "queries": pins.queries_raw_sha256,
    }
    for name, document in documents.items():
        _raw_pin(document, expected_raw[name], name)
    gallery = ProtectedEmbeddingManifest.from_dict(documents["gallery"].payload)
    queries = ProtectedEmbeddingManifest.from_dict(documents["queries"].payload)
    if gallery.input_name != "gallery" or queries.input_name != "queries":
        raise ValueError("protected embedding manifest input names differ")
    if gallery.production_receipt_sha256 != pins.gallery_production_receipt_sha256 or queries.production_receipt_sha256 != pins.queries_production_receipt_sha256:
        raise ValueError("embedding production receipt differs from external pin")
    _validate_resource_caps(policy, gallery, queries)
    _validate_split_binding(
        documents["split_assignment"].payload,
        documents["split_receipt"].payload,
        documents["split_assignment"].canonical_payload_sha256,
        policy,
        gallery,
        queries,
    )
    ledger = RoleExposureLedger.from_dict(documents["exposure_ledger"].payload)
    exposure_receipt = RoleExposureReceipt.from_dict(documents["exposure_receipt"].payload)
    verify_role_exposure_receipt(ledger, exposure_receipt)
    candidate_records = tuple(
        CandidateRoleRecord(
            sample_token=item.sample_token,
            identity_token=item.identity_token,
            public_subject_token=item.public_subject_token,
            assigned_stage=ExposureStage.FINAL_TEST_SCORED,
        )
        for item in sorted((*gallery.records, *queries.records), key=lambda item: item.sample_token)
    )
    _unique(tuple(item.sample_token for item in candidate_records), "evaluation sample tokens")
    candidate = CandidateRoleAssignment(
        source_artifact_sha256=documents["split_assignment"].canonical_payload_sha256,
        records=candidate_records,
    )
    _validate_prior_final_exposure(ledger, candidate)
    validate_candidate_assignment(ledger, candidate)
    files = tuple(
        _file_receipt(
            name,
            documents[name],
            (
                gallery.production_receipt_sha256
                if name == "gallery"
                else queries.production_receipt_sha256 if name == "queries" else None
            ),
        )
        for name in ProtectedEvaluationInputReceipt.EXPECTED_NAMES
    )
    input_receipt = ProtectedEvaluationInputReceipt(
        files=files,
        sample_counts=(("gallery", len(gallery.records)), ("queries", len(queries.records))),
        embedding_dimension=gallery.dimension,
        total_embedding_values=(len(gallery.records) + len(queries.records)) * gallery.dimension,
        score_matrix_elements=len(gallery.records) * len(queries.records),
    )
    declaration = RoleExposureDeclaration(
        source_artifact_sha256=input_receipt.receipt_sha256,
        kind=ExposureDeclarationKind.PRIOR_EVALUATION,
        revoked=False,
        records=tuple(
            RoleExposureDeclarationRecord(
                sample_token=item.sample_token,
                identity_token=item.identity_token,
                public_subject_token=item.public_subject_token,
                stage=ExposureStage.FINAL_TEST_SCORED,
            )
            for item in candidate_records
        ),
    )
    policy_receipt = ProtectedEvaluationPolicyReceipt(
        policy=_file_receipt("policy", policy_document, None),
        policy_sha256=policy.policy_sha256,
        external_pins=_file_receipt("external_pins", pins_document, None),
    )
    declaration_sha256 = content_sha256(declaration.to_dict())
    evaluation_token = content_sha256({
        "domain": "CVI_PROTECTED_EVALUATION_V1",
        "policy_receipt_sha256": policy_receipt.receipt_sha256,
        "input_receipt_sha256": input_receipt.receipt_sha256,
        "advanced_exposure_declaration_sha256": declaration_sha256,
    })
    plan = ProtectedEvaluationPlanReceipt(
        evaluation_token=evaluation_token,
        policy_receipt_sha256=policy_receipt.receipt_sha256,
        input_receipt_sha256=input_receipt.receipt_sha256,
        split_assignment_sha256=documents["split_assignment"].canonical_payload_sha256,
        prior_exposure_ledger_sha256=ledger.ledger_sha256,
        prior_exposure_receipt_sha256=exposure_receipt.receipt_sha256,
        advanced_exposure_declaration_sha256=declaration_sha256,
        tool_provenance=tool_provenance,
        tool_provenance_sha256=content_sha256(tool_provenance),
    )
    write_private_json_directory_bundle(
        output_directory,
        (
            ("policy_receipt.json", _bundle("cvi.protected_evaluation_policy_bundle.v1", policy_receipt.receipt_sha256, policy_receipt.to_dict())),
            ("input_receipt.json", _bundle("cvi.protected_evaluation_input_bundle.v1", input_receipt.receipt_sha256, input_receipt.to_dict())),
            ("advanced_exposure_declaration.json", declaration.to_dict()),
            ("plan_receipt.json", _bundle("cvi.protected_evaluation_plan_bundle.v1", plan.receipt_sha256, plan.to_dict())),
        ),
    )
    return plan


def load_protected_evaluation(
    *,
    preparation_directory: Path,
    expected_plan_receipt_sha256: str,
    expected_advanced_exposure_declaration_sha256: str,
    policy_path: Path,
    split_assignment_path: Path,
    split_receipt_path: Path,
    exposure_ledger_path: Path,
    exposure_receipt_path: Path,
    gallery_path: Path,
    queries_path: Path,
) -> ProtectedEvaluationPreparation:
    """Re-read and bind every prepared source before any score allocation."""

    _sha256(expected_plan_receipt_sha256, "expected plan receipt")
    _sha256(expected_advanced_exposure_declaration_sha256, "expected exposure declaration")
    policy_receipt = _read_bundle(
        preparation_directory / "policy_receipt.json",
        "cvi.protected_evaluation_policy_bundle.v1",
        ProtectedEvaluationPolicyReceipt,
    )
    input_receipt = _read_bundle(
        preparation_directory / "input_receipt.json",
        "cvi.protected_evaluation_input_bundle.v1",
        ProtectedEvaluationInputReceipt,
    )
    plan = _read_bundle(
        preparation_directory / "plan_receipt.json",
        "cvi.protected_evaluation_plan_bundle.v1",
        ProtectedEvaluationPlanReceipt,
    )
    if plan.receipt_sha256 != expected_plan_receipt_sha256:
        raise ValueError("protected plan differs from external anchor")
    declaration_document = read_strict_json_document(
        preparation_directory / "advanced_exposure_declaration.json",
        maximum_bytes=64 * 1024 * 1024,
    )
    declaration = RoleExposureDeclaration.from_dict(declaration_document.payload)
    if declaration_document.canonical_payload_sha256 != expected_advanced_exposure_declaration_sha256 or plan.advanced_exposure_declaration_sha256 != expected_advanced_exposure_declaration_sha256:
        raise ValueError("advanced exposure declaration differs from external anchor")
    if plan.policy_receipt_sha256 != policy_receipt.receipt_sha256 or plan.input_receipt_sha256 != input_receipt.receipt_sha256:
        raise ValueError("protected preparation receipt chain differs")
    policy_document = read_strict_json_document(policy_path, maximum_bytes=1_048_576)
    _match_file_receipt(policy_receipt.policy, policy_document)
    policy = ProtectedEvaluationPolicy.from_dict(policy_document.payload)
    documents = _read_source_documents(
        policy,
        split_assignment_path=split_assignment_path,
        split_receipt_path=split_receipt_path,
        exposure_ledger_path=exposure_ledger_path,
        exposure_receipt_path=exposure_receipt_path,
        gallery_path=gallery_path,
        queries_path=queries_path,
    )
    by_name = {item.name: item for item in input_receipt.files}
    for name, document in documents.items():
        _match_file_receipt(by_name[name], document)
    gallery = ProtectedEmbeddingManifest.from_dict(documents["gallery"].payload)
    queries = ProtectedEmbeddingManifest.from_dict(documents["queries"].payload)
    if (
        by_name["gallery"].external_receipt_sha256
        != gallery.production_receipt_sha256
        or by_name["queries"].external_receipt_sha256
        != queries.production_receipt_sha256
    ):
        raise ValueError("embedding production receipt differs from input receipt")
    _validate_resource_caps(policy, gallery, queries)
    if plan.split_assignment_sha256 != documents[
        "split_assignment"
    ].canonical_payload_sha256:
        raise ValueError("protected split assignment differs from plan")
    if (
        input_receipt.sample_counts
        != (("gallery", len(gallery.records)), ("queries", len(queries.records)))
        or input_receipt.embedding_dimension != gallery.dimension
        or input_receipt.total_embedding_values != (len(gallery.records) + len(queries.records)) * gallery.dimension
        or input_receipt.score_matrix_elements != len(gallery.records) * len(queries.records)
    ):
        raise ValueError("protected input resource receipt differs")
    _validate_split_binding(
        documents["split_assignment"].payload,
        documents["split_receipt"].payload,
        documents["split_assignment"].canonical_payload_sha256,
        policy,
        gallery,
        queries,
    )
    ledger = RoleExposureLedger.from_dict(documents["exposure_ledger"].payload)
    exposure_receipt = RoleExposureReceipt.from_dict(documents["exposure_receipt"].payload)
    verify_role_exposure_receipt(ledger, exposure_receipt)
    if ledger.ledger_sha256 != plan.prior_exposure_ledger_sha256 or exposure_receipt.receipt_sha256 != plan.prior_exposure_receipt_sha256:
        raise ValueError("prior exposure history differs from protected plan")
    expected_declaration_records = tuple(
        RoleExposureDeclarationRecord(
            sample_token=item.sample_token,
            identity_token=item.identity_token,
            public_subject_token=item.public_subject_token,
            stage=ExposureStage.FINAL_TEST_SCORED,
        )
        for item in sorted((*gallery.records, *queries.records), key=lambda item: item.sample_token)
    )
    if declaration.source_artifact_sha256 != input_receipt.receipt_sha256 or declaration.records != expected_declaration_records:
        raise ValueError("advanced exposure declaration differs from scored samples")
    candidate = CandidateRoleAssignment(
        source_artifact_sha256=documents["split_assignment"].canonical_payload_sha256,
        records=tuple(
            CandidateRoleRecord(
                sample_token=item.sample_token,
                identity_token=item.identity_token,
                public_subject_token=item.public_subject_token,
                assigned_stage=ExposureStage.FINAL_TEST_SCORED,
            )
            for item in sorted(
                (*gallery.records, *queries.records),
                key=lambda item: item.sample_token,
            )
        ),
    )
    _validate_prior_final_exposure(ledger, candidate)
    validate_candidate_assignment(ledger, candidate)
    return ProtectedEvaluationPreparation(
        policy=policy,
        policy_receipt=policy_receipt,
        input_receipt=input_receipt,
        plan_receipt=plan,
        advanced_exposure_declaration=declaration,
        gallery=gallery,
        queries=queries,
    )


def validate_protected_report(report: dict[str, Any]) -> StrictJsonDocument:
    resource = files("artifact_contracts").joinpath("schemas").joinpath(
        REPORT_SCHEMA_FILENAME
    )
    with as_file(resource) as schema_path:
        schema = read_strict_json_document(schema_path, maximum_bytes=1_048_576)
    errors = sorted(Draft202012Validator(schema.payload).iter_errors(report), key=lambda error: list(error.absolute_path))
    if errors:
        raise ValueError("protected report v3 schema validation failed: " + "; ".join(error.message for error in errors))
    return schema


def publish_protected_evaluation_output(
    *,
    output_directory: Path,
    preparation: ProtectedEvaluationPreparation,
    report: dict[str, Any],
    evaluator_provenance: dict[str, Any],
) -> ProtectedEvaluationOutputReceipt:
    schema_document = validate_protected_report(report)
    report_bytes = json_document_bytes(report)
    output_receipt = ProtectedEvaluationOutputReceipt(
        plan_receipt_sha256=preparation.plan_receipt.receipt_sha256,
        policy_receipt_sha256=preparation.policy_receipt.receipt_sha256,
        input_receipt_sha256=preparation.input_receipt.receipt_sha256,
        advanced_exposure_declaration_sha256=preparation.plan_receipt.advanced_exposure_declaration_sha256,
        report_raw_sha256=hashlib.sha256(report_bytes).hexdigest(),
        report_canonical_payload_sha256=content_sha256(report),
        report_byte_size=len(report_bytes),
        report_schema_raw_sha256=schema_document.raw_sha256,
        report_schema_canonical_payload_sha256=schema_document.canonical_payload_sha256,
        evaluator_provenance=evaluator_provenance,
        evaluator_provenance_sha256=content_sha256(evaluator_provenance),
    )
    write_private_json_directory_bundle(
        output_directory,
        (
            ("report.json", report),
            ("output_receipt.json", _bundle("cvi.protected_evaluation_output_bundle.v2", output_receipt.receipt_sha256, output_receipt.to_dict())),
        ),
    )
    return output_receipt


def verify_protected_evaluation_output(
    *,
    preparation_directory: Path,
    output_directory: Path,
    expected_plan_receipt_sha256: str,
    expected_advanced_exposure_declaration_sha256: str,
    expected_output_receipt_sha256: str,
) -> ProtectedEvaluationOutputReceipt:
    """Verify receipt-chain integrity against an out-of-band output anchor."""

    _sha256(expected_plan_receipt_sha256, "expected plan receipt")
    _sha256(
        expected_advanced_exposure_declaration_sha256,
        "expected advanced exposure declaration",
    )
    _sha256(expected_output_receipt_sha256, "expected output receipt")
    policy_receipt = _read_bundle(
        preparation_directory / "policy_receipt.json",
        "cvi.protected_evaluation_policy_bundle.v1",
        ProtectedEvaluationPolicyReceipt,
    )
    input_receipt = _read_bundle(
        preparation_directory / "input_receipt.json",
        "cvi.protected_evaluation_input_bundle.v1",
        ProtectedEvaluationInputReceipt,
    )
    plan = _read_bundle(
        preparation_directory / "plan_receipt.json",
        "cvi.protected_evaluation_plan_bundle.v1",
        ProtectedEvaluationPlanReceipt,
    )
    if plan.receipt_sha256 != expected_plan_receipt_sha256:
        raise ValueError("protected plan differs from external anchor")
    if (
        plan.policy_receipt_sha256 != policy_receipt.receipt_sha256
        or plan.input_receipt_sha256 != input_receipt.receipt_sha256
    ):
        raise ValueError("protected preparation receipt chain differs")
    declaration = read_strict_json_document(
        preparation_directory / "advanced_exposure_declaration.json",
        maximum_bytes=64 * 1024 * 1024,
    )
    RoleExposureDeclaration.from_dict(declaration.payload)
    if declaration.canonical_payload_sha256 != expected_advanced_exposure_declaration_sha256 or plan.advanced_exposure_declaration_sha256 != expected_advanced_exposure_declaration_sha256:
        raise ValueError("advanced exposure declaration differs from external anchor")
    output = _read_bundle(
        output_directory / "output_receipt.json",
        "cvi.protected_evaluation_output_bundle.v2",
        ProtectedEvaluationOutputReceipt,
    )
    if output.receipt_sha256 != expected_output_receipt_sha256:
        raise ValueError("protected output receipt differs from external output anchor")
    report = read_strict_json_document(output_directory / "report.json", maximum_bytes=64 * 1024 * 1024)
    schema = validate_protected_report(report.payload)
    if (
        output.plan_receipt_sha256 != plan.receipt_sha256
        or output.policy_receipt_sha256 != policy_receipt.receipt_sha256
        or output.input_receipt_sha256 != input_receipt.receipt_sha256
        or output.advanced_exposure_declaration_sha256 != declaration.canonical_payload_sha256
        or output.report_raw_sha256 != report.raw_sha256
        or output.report_canonical_payload_sha256 != report.canonical_payload_sha256
        or output.report_byte_size != report.byte_size
        or output.report_schema_raw_sha256 != schema.raw_sha256
        or output.report_schema_canonical_payload_sha256 != schema.canonical_payload_sha256
    ):
        raise ValueError("protected output report receipt chain differs")
    expected_report_chain = {
        "plan_receipt_sha256": plan.receipt_sha256,
        "policy_receipt_sha256": policy_receipt.receipt_sha256,
        "input_receipt_sha256": input_receipt.receipt_sha256,
        "advanced_exposure_declaration_sha256": declaration.canonical_payload_sha256,
        "split_assignment_sha256": plan.split_assignment_sha256,
        "prior_exposure_ledger_sha256": plan.prior_exposure_ledger_sha256,
        "prior_exposure_receipt_sha256": plan.prior_exposure_receipt_sha256,
    }
    if (
        report.payload["evaluation_token"] != plan.evaluation_token
        or report.payload["receipt_chain"] != expected_report_chain
        or report.payload["evaluator_provenance_sha256"]
        != output.evaluator_provenance_sha256
        or report.payload["protocol_status"] != REPORT_PROTOCOL_STATUS
        or report.payload["receipt_chain_verified"] is not True
        or report.payload["valid_for_model_selection"] is not False
        or report.payload["valid_for_final_reporting"] is not False
        or report.payload["interpretation"] != REPORT_INTERPRETATION
    ):
        raise ValueError(
            "protected report receipt bindings or validity semantics differ"
        )
    return output


def _validate_prior_final_exposure(
    ledger: RoleExposureLedger,
    candidate: CandidateRoleAssignment,
) -> None:
    """Require declared byte export, and no prior scoring, for every final row."""

    by_sample = {record.sample_token: record for record in ledger.records}
    by_identity: dict[str, list[Any]] = {}
    by_subject: dict[str, list[Any]] = {}
    for record in ledger.records:
        by_identity.setdefault(record.identity_token, []).append(record)
        by_subject.setdefault(record.public_subject_token, []).append(record)
    for item in candidate.records:
        prior = by_sample.get(item.sample_token)
        if prior is None:
            raise ValueError("protected final sample lacks actual prior exposure history")
        related = (
            prior,
            *by_identity.get(item.identity_token, ()),
            *by_subject.get(item.public_subject_token, ()),
        )
        if any(
            record.maximum_historical_stage is not ExposureStage.BYTES_EXPORTED
            for record in related
        ):
            raise ValueError("protected final sample, identity, or subject was previously advanced")


def _read_source_documents(
    policy: ProtectedEvaluationPolicy,
    *,
    split_assignment_path: Path,
    split_receipt_path: Path,
    exposure_ledger_path: Path,
    exposure_receipt_path: Path,
    gallery_path: Path,
    queries_path: Path,
) -> dict[str, StrictJsonDocument]:
    paths = {
        "split_assignment": split_assignment_path,
        "split_receipt": split_receipt_path,
        "exposure_ledger": exposure_ledger_path,
        "exposure_receipt": exposure_receipt_path,
        "gallery": gallery_path,
        "queries": queries_path,
    }
    return {
        name: read_strict_json_document(
            path,
            maximum_bytes=policy.maximum_json_bytes,
            maximum_depth=policy.maximum_json_depth,
            maximum_nodes=policy.maximum_json_nodes,
            maximum_keys=policy.maximum_json_keys,
            maximum_array_length=policy.maximum_json_array_length,
            maximum_string_characters=policy.maximum_json_string_characters,
            maximum_number_characters=policy.maximum_json_number_characters,
        )
        for name, path in paths.items()
    }


def _validate_resource_caps(policy: ProtectedEvaluationPolicy, gallery: ProtectedEmbeddingManifest, queries: ProtectedEmbeddingManifest) -> None:
    if len(gallery.records) > policy.maximum_samples_per_input or len(queries.records) > policy.maximum_samples_per_input:
        raise ValueError("protected sample count exceeds resource policy")
    if gallery.dimension != queries.dimension or gallery.dimension > policy.maximum_embedding_dimension:
        raise ValueError("protected embedding dimension exceeds or differs from policy")
    total_values = (len(gallery.records) + len(queries.records)) * gallery.dimension
    if total_values > policy.maximum_total_embedding_values:
        raise ValueError("protected embedding value count exceeds resource policy")
    score_elements = len(gallery.records) * len(queries.records)
    if score_elements > policy.maximum_score_matrix_elements:
        raise ValueError("protected score matrix exceeds resource policy")


def _validate_split_binding(
    assignment: dict[str, Any],
    split_receipt: dict[str, Any],
    assignment_sha256: str,
    policy: ProtectedEvaluationPolicy,
    gallery: ProtectedEmbeddingManifest,
    queries: ProtectedEmbeddingManifest,
) -> None:
    required_assignment = {"schema_version", "status", "records"}
    if not required_assignment <= set(assignment) or assignment["schema_version"] != "cvi.protected_public_split_assignment.v1" or assignment["status"] != "PASS_PROTECTED_SPLIT_CONSTRUCTION" or not isinstance(assignment["records"], list):
        raise ValueError("protected split assignment is not an admitted final split")
    if (
        split_receipt.get("schema_version")
        != "cvi.protected_public_split_receipt.v2"
        or split_receipt.get("assignment_sha256") != assignment_sha256
        or not _is_sha256(split_receipt.get("role_exposure_ledger_sha256"))
        or not _is_sha256(split_receipt.get("role_exposure_receipt_sha256"))
    ):
        raise ValueError("protected split receipt does not bind assignment")
    by_sample: dict[str, dict[str, Any]] = {}
    for record in assignment["records"]:
        if not isinstance(record, dict) or not {"sample_token", "identity_token", "uses"} <= set(record) or not isinstance(record["uses"], list):
            raise ValueError("protected split assignment record differs")
        if record["sample_token"] in by_sample:
            raise ValueError("protected split assignment sample is duplicated")
        by_sample[record["sample_token"]] = record
    manifests = {"gallery": gallery, "queries": queries}
    for binding in policy.role_bindings:
        manifest = manifests[binding.name]
        for item in manifest.records:
            assigned = by_sample.get(item.sample_token)
            if assigned is None or assigned["identity_token"] != item.identity_token:
                raise ValueError("embedding sample differs from protected split assignment")
            matches = [
                use
                for use in assigned["uses"]
                if isinstance(use, dict)
                and use.get("protocol") == binding.protocol
                and use.get("episode") == binding.episode
                and use.get("gallery_size") == binding.gallery_size
                and use.get("shot") == binding.shot
                and use.get("role") == binding.role
            ]
            if len(matches) != 1:
                raise ValueError("embedding sample lacks exactly one protected protocol role")
    gallery_identities = {item.identity_token for item in gallery.records}
    if not {item.identity_token for item in queries.records} <= gallery_identities:
        raise ValueError("protected closed-set query has no gallery identity")
    if {item.template_token for item in gallery.records} & {item.template_token for item in queries.records}:
        raise ValueError("protected gallery/query template overlap")
    if {item.sample_token for item in gallery.records} & {item.sample_token for item in queries.records}:
        raise ValueError("protected gallery/query sample overlap")


def _file_receipt(name: str, document: StrictJsonDocument, external_receipt: str | None) -> ProtectedFileReceipt:
    return ProtectedFileReceipt(
        name=name,
        raw_sha256=document.raw_sha256,
        canonical_payload_sha256=document.canonical_payload_sha256,
        byte_size=document.byte_size,
        external_receipt_sha256=external_receipt,
    )


def _match_file_receipt(receipt: ProtectedFileReceipt, document: StrictJsonDocument) -> None:
    if (
        receipt.raw_sha256 != document.raw_sha256
        or receipt.canonical_payload_sha256 != document.canonical_payload_sha256
        or receipt.byte_size != document.byte_size
    ):
        raise ValueError(f"protected {receipt.name} differs from input receipt")


def _raw_pin(document: StrictJsonDocument, expected: str, name: str) -> None:
    if document.raw_sha256 != expected:
        raise ValueError(f"protected {name} differs from external raw-byte pin")


def _bundle(schema_version: str, receipt_sha256: str, receipt: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": schema_version, "receipt_sha256": receipt_sha256, "receipt": receipt}


def _read_bundle(path: Path, schema_version: str, receipt_type: type[Any]) -> Any:
    document = read_strict_json_document(path, maximum_bytes=64 * 1024 * 1024)
    _exact(document.payload, {"schema_version", "receipt_sha256", "receipt"}, "protected receipt bundle")
    if document.payload["schema_version"] != schema_version or not isinstance(document.payload["receipt"], dict):
        raise ValueError("protected receipt bundle schema differs")
    receipt = receipt_type.from_dict(document.payload["receipt"])
    if receipt.receipt_sha256 != document.payload["receipt_sha256"]:
        raise ValueError("protected receipt bundle digest differs")
    return receipt


def _exact(payload: object, expected: set[str], name: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{name} keys differ")


def _pairs(values: list[Any], name: str) -> tuple[tuple[Any, Any], ...]:
    result = []
    for item in values:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{name} entries differ")
        result.append((item[0], item[1]))
    return tuple(result)


def _sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _text(value: object, name: str, maximum: int) -> None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be bounded non-empty text")


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")


__all__ = [
    "PREPARATION_FILENAMES",
    "REPORT_INTERPRETATION",
    "REPORT_PROTOCOL_STATUS",
    "REPORT_SCHEMA_FILENAME",
    "REPORT_SCHEMA_VERSION",
    "ProtectedEmbeddingManifest",
    "ProtectedEmbeddingRecord",
    "ProtectedEvaluationExternalPins",
    "ProtectedEvaluationInputReceipt",
    "ProtectedEvaluationOutputReceipt",
    "ProtectedEvaluationPlanReceipt",
    "ProtectedEvaluationPolicy",
    "ProtectedEvaluationPolicyReceipt",
    "ProtectedEvaluationPreparation",
    "ProtectedEvaluationRoleBinding",
    "load_protected_evaluation",
    "prepare_protected_evaluation",
    "publish_protected_evaluation_output",
    "validate_protected_report",
    "verify_protected_evaluation_output",
]
