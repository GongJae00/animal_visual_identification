"""Receipt-authenticated, resumable Meta PDQ audit for the public canine corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.public.public_canine_manifest import PublicCanineRecord
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from foundation.provenance import content_sha256
from embedding.methods.classical.pdq.contracts import PDQFingerprint, PDQSearchPolicy
from embedding.methods.classical.pdq.mih import find_pdq_near_duplicate_candidates
from embedding.methods.classical.pdq.native import PdqNativeBuildReceipt
from embedding.methods.classical.public_canine_phash_audit import _AuthenticatedSource

MAXIMUM_FINGERPRINT_CHUNK_SAMPLES = 10_000
_PASS = "PASS_BOUNDED_LABEL_BLIND_PDQ_CANDIDATE_GENERATION"
_INTERPRETATION = (
    "PDQ_SIMILARITY_CANDIDATES_ONLY_NOT_DUPLICATE_NONDUPLICATE_"
    "SPLIT_OR_MODEL_ADMISSION"
)


@dataclass(frozen=True, slots=True)
class PDQFingerprintChunk:
    source_spec_sha256: str
    source_receipt_bindings_sha256: str
    native_build_receipt_sha256: str
    native_binary_sha256: str
    official_regression_receipt_sha256: str
    corpus_sample_ids_sha256: str
    corpus_sample_count: int
    start_index: int
    end_index: int
    fingerprints: tuple[PDQFingerprint, ...]
    schema_version: str = "cvi.public_canine_pdq_fingerprint_chunk.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_canine_pdq_fingerprint_chunk.v1":
            raise ValueError("unsupported PDQ fingerprint chunk schema")
        for name in (
            "source_spec_sha256",
            "source_receipt_bindings_sha256",
            "native_build_receipt_sha256",
            "native_binary_sha256",
            "official_regression_receipt_sha256",
            "corpus_sample_ids_sha256",
        ):
            _sha256(getattr(self, name), name)
        _positive_int(self.corpus_sample_count, "corpus_sample_count")
        _nonnegative_int(self.start_index, "start_index")
        _positive_int(self.end_index, "end_index")
        if not self.start_index < self.end_index <= self.corpus_sample_count:
            raise ValueError("PDQ fingerprint chunk range differs")
        if len(self.fingerprints) != self.end_index - self.start_index:
            raise ValueError("PDQ fingerprint chunk cardinality differs")
        ids = tuple(item.opaque_sample_id for item in self.fingerprints)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("PDQ fingerprint chunk IDs must be sorted unique")

    @property
    def chunk_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_spec_sha256": self.source_spec_sha256,
            "source_receipt_bindings_sha256": self.source_receipt_bindings_sha256,
            "native_build_receipt_sha256": self.native_build_receipt_sha256,
            "native_binary_sha256": self.native_binary_sha256,
            "official_regression_receipt_sha256": self.official_regression_receipt_sha256,
            "corpus_sample_ids_sha256": self.corpus_sample_ids_sha256,
            "corpus_sample_count": self.corpus_sample_count,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "fingerprints": [item.to_dict() for item in self.fingerprints],
            "fingerprint_count": len(self.fingerprints),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PDQFingerprintChunk":
        expected = set(cls.__dataclass_fields__) | {"fingerprint_count"}
        _exact(payload, expected, "PDQ fingerprint chunk")
        raw = payload["fingerprints"]
        if not isinstance(raw, list):
            raise TypeError("PDQ fingerprint chunk fingerprints must be an array")
        fingerprints = tuple(PDQFingerprint.from_dict(item) for item in raw)
        if payload["fingerprint_count"] != len(fingerprints):
            raise ValueError("PDQ fingerprint chunk count differs")
        values = {key: value for key, value in payload.items() if key != "fingerprint_count"}
        values["fingerprints"] = fingerprints
        return cls(**values)


@dataclass(frozen=True, slots=True)
class _CorpusItem:
    authenticated: _AuthenticatedSource
    record: PublicCanineRecord
    opaque_id: str


@dataclass(frozen=True, slots=True)
class _AuditContext:
    items: tuple[_CorpusItem, ...]
    source_spec_sha256: str
    source_receipt_bindings: tuple[dict[str, str], ...]
    source_receipt_bindings_sha256: str
    native_build_receipt: PdqNativeBuildReceipt
    native_build_receipt_sha256: str
    native_binary_path: Path
    official_regression_receipt_sha256: str
    corpus_sample_ids_sha256: str


def merge_pdq_fingerprint_chunks(
    *,
    context: _AuditContext,
    chunks: Sequence[PDQFingerprintChunk],
) -> dict[str, Any]:
    """Validate exact contiguous corpus coverage and return one manifest payload."""

    if not chunks:
        raise ValueError("at least one PDQ fingerprint chunk is required")
    ordered = tuple(sorted(chunks, key=lambda item: item.start_index))
    cursor = 0
    fingerprints: list[PDQFingerprint] = []
    chunk_rows: list[dict[str, Any]] = []
    for chunk in ordered:
        _validate_chunk_lineage(chunk, context, chunk.start_index, chunk.end_index)
        if chunk.start_index != cursor:
            raise ValueError("PDQ fingerprint chunks contain a gap or overlap")
        fingerprints.extend(chunk.fingerprints)
        chunk_rows.append({
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "chunk_sha256": chunk.chunk_sha256,
        })
        cursor = chunk.end_index
    if cursor != len(context.items):
        raise ValueError("PDQ fingerprint chunks do not exactly cover the corpus")
    expected_ids = tuple(item.opaque_id for item in context.items)
    observed_ids = tuple(item.opaque_sample_id for item in fingerprints)
    if observed_ids != expected_ids:
        raise ValueError("merged PDQ fingerprint IDs differ from authenticated corpus")
    return {
        "schema_version": "cvi.public_canine_pdq_fingerprint_manifest.v1",
        "source_spec_sha256": context.source_spec_sha256,
        "source_receipt_bindings": list(context.source_receipt_bindings),
        "source_receipt_bindings_sha256": context.source_receipt_bindings_sha256,
        "native_build_receipt_sha256": context.native_build_receipt_sha256,
        "native_binary_sha256": context.native_build_receipt.binary_sha256,
        "official_regression_receipt_sha256": (
            context.official_regression_receipt_sha256
        ),
        "corpus_sample_ids_sha256": context.corpus_sample_ids_sha256,
        "fingerprints": [item.to_dict() for item in fingerprints],
        "fingerprint_count": len(fingerprints),
        "chunks": chunk_rows,
        "chunk_count": len(chunk_rows),
        "decision": "PASS_EXACT_RESUMABLE_PDQ_FINGERPRINT_COVERAGE",
        "interpretation": "LABEL_BLIND_FINGERPRINTS_ONLY_NOT_DUPLICATE_ADJUDICATION",
    }


def build_pdq_evidence_bundle(
    *, fingerprint_manifest: Mapping[str, Any], policy: PDQSearchPolicy
) -> dict[str, Any]:
    """Run bounded exact-complete MIH and build the public v1 evidence bundle."""

    fingerprints = _validate_fingerprint_manifest(fingerprint_manifest)
    if len(fingerprints) > policy.maximum_samples:
        raise ValueError("PDQ fingerprint manifest exceeds search policy")
    search = find_pdq_near_duplicate_candidates(fingerprints, policy=policy)
    evidence = {
        "schema_version": "cvi.public_canine_pdq_evidence.v1",
        "search_result": search.to_dict(),
        "sample_ids_sha256": fingerprint_manifest["corpus_sample_ids_sha256"],
        "fingerprint_count": len(fingerprints),
        "fingerprint_manifest_sha256": content_sha256(fingerprint_manifest),
        "source_spec_sha256": fingerprint_manifest["source_spec_sha256"],
        "source_receipt_bindings_sha256": fingerprint_manifest[
            "source_receipt_bindings_sha256"
        ],
        "native_build_receipt_sha256": fingerprint_manifest[
            "native_build_receipt_sha256"
        ],
        "native_binary_sha256": fingerprint_manifest["native_binary_sha256"],
        "official_regression_receipt_sha256": fingerprint_manifest[
            "official_regression_receipt_sha256"
        ],
        "policy": policy.to_dict(),
        "policy_sha256": policy.policy_sha256,
        "decision": _PASS,
        "interpretation": _INTERPRETATION,
    }
    return {
        "schema_version": "cvi.public_canine_pdq_evidence_bundle.v1",
        "evidence": evidence,
        "evidence_sha256": content_sha256(evidence),
    }


def publish_pdq_fingerprint_chunk(
    path: Path,
    chunk: PDQFingerprintChunk,
    *,
    tool_provenance: Mapping[str, Any],
) -> str:
    bundle = _bundle(
        "cvi.public_canine_pdq_fingerprint_chunk_bundle.v1",
        "chunk",
        chunk.to_dict(),
        "chunk_sha256",
        tool_provenance,
    )
    write_private_json_bundle(((path, bundle),))
    return bundle["bundle_sha256"]


def publish_pdq_evidence_bundle(path: Path, bundle: Mapping[str, Any]) -> str:
    _validate_pdq_evidence_bundle(bundle)
    write_private_json_bundle(((path, dict(bundle)),))
    return content_sha256(bundle)


def read_pdq_fingerprint_chunk(path: Path) -> PDQFingerprintChunk:
    payload = _read_bundle(
        path,
        "cvi.public_canine_pdq_fingerprint_chunk_bundle.v1",
        "chunk",
        "chunk_sha256",
    )
    return PDQFingerprintChunk.from_dict(payload)


def _validate_chunk_lineage(
    chunk: PDQFingerprintChunk,
    context: _AuditContext,
    start: int,
    end: int,
) -> None:
    expected = (
        context.source_spec_sha256,
        context.source_receipt_bindings_sha256,
        context.native_build_receipt_sha256,
        context.native_build_receipt.binary_sha256,
        context.official_regression_receipt_sha256,
        context.corpus_sample_ids_sha256,
        len(context.items),
        start,
        end,
    )
    observed = (
        chunk.source_spec_sha256,
        chunk.source_receipt_bindings_sha256,
        chunk.native_build_receipt_sha256,
        chunk.native_binary_sha256,
        chunk.official_regression_receipt_sha256,
        chunk.corpus_sample_ids_sha256,
        chunk.corpus_sample_count,
        chunk.start_index,
        chunk.end_index,
    )
    if observed != expected:
        raise ValueError("PDQ fingerprint chunk lineage differs")


def _validate_fingerprint_manifest(
    payload: Mapping[str, Any],
) -> tuple[PDQFingerprint, ...]:
    expected = {
        "schema_version",
        "source_spec_sha256",
        "source_receipt_bindings",
        "source_receipt_bindings_sha256",
        "native_build_receipt_sha256",
        "native_binary_sha256",
        "official_regression_receipt_sha256",
        "corpus_sample_ids_sha256",
        "fingerprints",
        "fingerprint_count",
        "chunks",
        "chunk_count",
        "decision",
        "interpretation",
    }
    _exact(payload, expected, "PDQ fingerprint manifest")
    for name in (
        "source_spec_sha256",
        "source_receipt_bindings_sha256",
        "native_build_receipt_sha256",
        "native_binary_sha256",
        "official_regression_receipt_sha256",
        "corpus_sample_ids_sha256",
    ):
        _sha256(payload[name], name)
    if content_sha256(payload["source_receipt_bindings"]) != payload[
        "source_receipt_bindings_sha256"
    ]:
        raise ValueError("PDQ source receipt bindings digest differs")
    if not isinstance(payload["fingerprints"], list) or not isinstance(
        payload["chunks"], list
    ):
        raise TypeError("PDQ fingerprint manifest collections differ")
    fingerprints = tuple(
        PDQFingerprint.from_dict(item) for item in payload["fingerprints"]
    )
    ids = tuple(item.opaque_sample_id for item in fingerprints)
    if (
        payload["schema_version"] != "cvi.public_canine_pdq_fingerprint_manifest.v1"
        or payload["fingerprint_count"] != len(fingerprints)
        or payload["chunk_count"] != len(payload["chunks"])
        or ids != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or content_sha256(list(ids)) != payload["corpus_sample_ids_sha256"]
        or payload["decision"] != "PASS_EXACT_RESUMABLE_PDQ_FINGERPRINT_COVERAGE"
    ):
        raise ValueError("PDQ fingerprint manifest coverage differs")
    cursor = 0
    for row in payload["chunks"]:
        _exact(row, {"start_index", "end_index", "chunk_sha256"}, "PDQ chunk row")
        if row["start_index"] != cursor or not cursor < row["end_index"] <= len(ids):
            raise ValueError("PDQ manifest chunk coverage differs")
        _sha256(row["chunk_sha256"], "chunk_sha256")
        cursor = row["end_index"]
    if cursor != len(ids):
        raise ValueError("PDQ manifest chunks do not exactly cover fingerprints")
    return fingerprints


def _validate_pdq_evidence_bundle(bundle: Mapping[str, Any]) -> None:
    _exact(bundle, {"schema_version", "evidence", "evidence_sha256"}, "PDQ evidence bundle")
    if (
        bundle["schema_version"] != "cvi.public_canine_pdq_evidence_bundle.v1"
        or not isinstance(bundle["evidence"], Mapping)
        or content_sha256(bundle["evidence"]) != bundle["evidence_sha256"]
    ):
        raise ValueError("PDQ evidence bundle binding differs")


def _bundle(
    schema: str,
    payload_name: str,
    payload: dict[str, Any],
    digest_name: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("tool provenance must be nonempty")
    result = {
        "schema_version": schema,
        payload_name: payload,
        digest_name: content_sha256(payload),
        "tool_provenance": dict(provenance),
        "tool_provenance_sha256": content_sha256(provenance),
    }
    result["bundle_sha256"] = content_sha256(result)
    return result


def _read_bundle(
    path: Path, schema: str, payload_name: str, digest_name: str
) -> Mapping[str, Any]:
    bundle = read_strict_json_object(path)
    expected = {
        "schema_version",
        payload_name,
        digest_name,
        "tool_provenance",
        "tool_provenance_sha256",
        "bundle_sha256",
    }
    _exact(bundle, expected, "PDQ protected bundle")
    unsigned = dict(bundle)
    observed = unsigned.pop("bundle_sha256")
    if (
        bundle["schema_version"] != schema
        or content_sha256(unsigned) != observed
        or content_sha256(bundle["tool_provenance"])
        != bundle["tool_provenance_sha256"]
    ):
        raise ValueError("PDQ protected bundle digest differs")
    payload = bundle[payload_name]
    if not isinstance(payload, Mapping) or content_sha256(payload) != bundle[digest_name]:
        raise ValueError("PDQ protected payload digest differs")
    return payload


def _sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{name} must be lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be lowercase SHA-256") from error


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _exact(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"{name} fields differ")
