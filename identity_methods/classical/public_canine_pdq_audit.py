"""Receipt-authenticated, resumable Meta PDQ audit for the public canine corpus."""

from __future__ import annotations

import os
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.public_canine_manifest import PublicCanineRecord
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from foundation.provenance import content_sha256
from identity_methods.classical.pdq_contracts import PDQFingerprint, PDQSearchPolicy
from identity_methods.classical.pdq_mih import find_pdq_near_duplicate_candidates
from identity_methods.classical.pdq_native import (
    MAXIMUM_BATCH_BYTES,
    MAXIMUM_BATCH_REQUESTS,
    CanonicalRGBRequest,
    PdqNativeBuildReceipt,
    hash_rgb_batch,
    verify_native_pdq_build,
)
from identity_methods.classical.phash_mih import opaque_sample_id
from identity_methods.classical.public_canine_phash_audit import (
    PublicCaninePHashPolicy,
    PublicCaninePHashSource,
    _authenticate_source,
    _AuthenticatedSource,
    _bound_container_info,
    _bound_member_info,
    _canonical_rgb_member,
    _open_bound_archive,
    _pillow,
    _sha256_stream,
    _source_spec_sha256,
    _stage_nested_container,
    _unique_info_index,
    _validate_source_set,
    _verify_archive_stability,
)

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


def prepare_pdq_audit_context(
    *,
    sources: tuple[PublicCaninePHashSource, ...],
    decode_policy: PublicCaninePHashPolicy,
    native_worker_directory: Path,
    official_regression_receipt_path: Path,
) -> _AuditContext:
    """Authenticate all fixed inputs and freeze the corpus-wide opaque order."""

    _validate_source_set(sources)
    authenticated = tuple(_authenticate_source(item, decode_policy) for item in sources)
    items = tuple(sorted(
        (
            _CorpusItem(item, record, opaque_sample_id(record.source_sample_id))
            for item in authenticated
            for manifest in item.manifests
            for record in manifest.records
        ),
        key=lambda item: item.opaque_id,
    ))
    ids = tuple(item.opaque_id for item in items)
    if len(ids) != len(set(ids)):
        raise ValueError("PDQ corpus contains an opaque sample ID collision")
    if len(items) > decode_policy.maximum_fingerprints:
        raise ValueError("PDQ corpus exceeds fingerprint cap")
    source_rows = tuple(sorted(
        (
            {
                "archive_receipt_sha256": item.archive_receipt_sha256,
                "semantic_receipt_sha256": item.semantic_receipt_sha256,
                "image_receipt_sha256": item.image_receipt_sha256,
                "image_policy_sha256": item.image_policy_sha256,
            }
            for item in authenticated
        ),
        key=lambda row: tuple(row.values()),
    ))
    build_payload = read_strict_json_object(native_worker_directory / "build-receipt.json")
    build = PdqNativeBuildReceipt.from_dict(build_payload)
    verify_native_pdq_build(native_worker_directory, build)
    official_bundle = read_strict_json_object(official_regression_receipt_path)
    official_sha256 = _validate_official_regression(official_bundle, build)
    return _AuditContext(
        items=items,
        source_spec_sha256=_source_spec_sha256(sources),
        source_receipt_bindings=source_rows,
        source_receipt_bindings_sha256=content_sha256(source_rows),
        native_build_receipt=build,
        native_build_receipt_sha256=build.receipt_sha256,
        native_binary_path=native_worker_directory / build.binary_filename,
        official_regression_receipt_sha256=official_sha256,
        corpus_sample_ids_sha256=content_sha256(list(ids)),
    )


def run_resumable_fingerprint_chunks(
    *,
    context: _AuditContext,
    decode_policy: PublicCaninePHashPolicy,
    output_directory: Path,
    chunk_size: int,
    tool_provenance: Mapping[str, Any],
    maximum_new_chunks: int | None = None,
) -> tuple[int, int]:
    """Create missing deterministic chunks and validate any existing chunks."""

    _positive_int(chunk_size, "chunk_size")
    if chunk_size > MAXIMUM_FINGERPRINT_CHUNK_SAMPLES:
        raise ValueError("PDQ fingerprint chunk size exceeds fixed cap")
    if maximum_new_chunks is not None:
        _positive_int(maximum_new_chunks, "maximum_new_chunks")
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("PDQ chunk output must be an existing real directory")
    created = 0
    reused = 0
    for start in range(0, len(context.items), chunk_size):
        end = min(len(context.items), start + chunk_size)
        path = output_directory / f"pdq-fingerprints-{start:05d}-{end:05d}.json"
        if path.exists():
            observed = read_pdq_fingerprint_chunk(path)
            _validate_chunk_lineage(observed, context, start, end)
            expected_ids = tuple(item.opaque_id for item in context.items[start:end])
            if tuple(item.opaque_sample_id for item in observed.fingerprints) != expected_ids:
                raise ValueError("existing PDQ chunk sample coverage differs")
            reused += 1
            continue
        if maximum_new_chunks is not None and created >= maximum_new_chunks:
            break
        fingerprints = _fingerprint_items(
            context.items[start:end], context=context, policy=decode_policy
        )
        chunk = PDQFingerprintChunk(
            source_spec_sha256=context.source_spec_sha256,
            source_receipt_bindings_sha256=context.source_receipt_bindings_sha256,
            native_build_receipt_sha256=context.native_build_receipt_sha256,
            native_binary_sha256=context.native_build_receipt.binary_sha256,
            official_regression_receipt_sha256=(
                context.official_regression_receipt_sha256
            ),
            corpus_sample_ids_sha256=context.corpus_sample_ids_sha256,
            corpus_sample_count=len(context.items),
            start_index=start,
            end_index=end,
            fingerprints=fingerprints,
        )
        publish_pdq_fingerprint_chunk(path, chunk, tool_provenance=tool_provenance)
        created += 1
    return created, reused


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


def publish_pdq_fingerprint_manifest(
    path: Path, manifest: Mapping[str, Any], *, tool_provenance: Mapping[str, Any]
) -> str:
    _validate_fingerprint_manifest(manifest)
    bundle = _bundle(
        "cvi.public_canine_pdq_fingerprint_manifest_bundle.v1",
        "manifest",
        dict(manifest),
        "manifest_sha256",
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


def read_pdq_fingerprint_manifest(path: Path) -> dict[str, Any]:
    payload = dict(_read_bundle(
        path,
        "cvi.public_canine_pdq_fingerprint_manifest_bundle.v1",
        "manifest",
        "manifest_sha256",
    ))
    _validate_fingerprint_manifest(payload)
    return payload


def _fingerprint_items(
    items: Sequence[_CorpusItem],
    *,
    context: _AuditContext,
    policy: PublicCaninePHashPolicy,
) -> tuple[PDQFingerprint, ...]:
    by_source: dict[PublicCaninePHashSource, list[_CorpusItem]] = defaultdict(list)
    for item in items:
        by_source[item.authenticated.source].append(item)
    output: dict[str, PDQFingerprint] = {}
    for source in sorted(by_source, key=lambda item: item.dataset_name):
        selected = by_source[source]
        authenticated = selected[0].authenticated
        _fingerprint_source_items(
            authenticated,
            selected,
            context=context,
            policy=policy,
            output=output,
        )
    expected = tuple(item.opaque_id for item in items)
    if set(output) != set(expected):
        raise RuntimeError("PDQ chunk fingerprint coverage differs")
    return tuple(output[value] for value in expected)


def _fingerprint_source_items(
    authenticated: _AuthenticatedSource,
    selected: Sequence[_CorpusItem],
    *,
    context: _AuditContext,
    policy: PublicCaninePHashPolicy,
    output: dict[str, PDQFingerprint],
) -> None:
    PIL, Image, ImageFile, ImageOps, UnidentifiedImageError = _pillow()
    if authenticated.image_decoder_name != "Pillow" or authenticated.image_decoder_version != PIL.__version__:
        raise ValueError("current Pillow differs from protected image receipt")
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise RuntimeError("Pillow truncated-image decoding must be disabled")
    descriptor, initial_stat = _open_bound_archive(
        authenticated.source.archive_path, policy.maximum_archive_bytes
    )
    pending: list[CanonicalRGBRequest] = []
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending_bytes
        if not pending:
            return
        results = hash_rgb_batch(
            tuple(pending),
            binary_path=context.native_binary_path,
            expected_binary_sha256=context.native_build_receipt.binary_sha256,
        )
        for result in results:
            if result.request_token in output:
                raise ValueError("duplicate PDQ fingerprint output")
            output[result.request_token] = PDQFingerprint(
                result.request_token, result.d4_hashes, result.quality
            )
        pending.clear()
        pending_bytes = 0

    def append(item: _CorpusItem, archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> None:
        nonlocal pending_bytes
        protected = authenticated.image_records[item.record.source_sample_id]
        rgb, width, height, _ = _canonical_rgb_member(
            archive,
            info,
            protected,
            policy,
            Image,
            ImageOps,
            UnidentifiedImageError,
        )
        if pending and (
            len(pending) >= MAXIMUM_BATCH_REQUESTS
            or pending_bytes + len(rgb) > MAXIMUM_BATCH_BYTES
        ):
            flush()
        pending.append(CanonicalRGBRequest(
            width=width,
            height=height,
            rgb=rgb,
            request_sequence=item_index[item.opaque_id],
            request_token=item.opaque_id,
        ))
        pending_bytes += len(rgb)

    item_index = {item.opaque_id: index for index, item in enumerate(context.items)}
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            digest = _sha256_stream(stream, policy.read_chunk_bytes)
            expected_sha = selected[0].record.source_archive_sha256
            if digest != expected_sha:
                raise ValueError("archive bytes differ from authenticated source")
            stream.seek(0)
            with zipfile.ZipFile(stream) as outer:
                outer_index = _unique_info_index(outer)
                direct = sorted(
                    (item for item in selected if item.record.container_member_path is None),
                    key=lambda item: item.record.source_sample_id,
                )
                for item in direct:
                    append(
                        item,
                        outer,
                        _bound_member_info(outer_index, item.record, policy),
                    )
                nested: dict[str, list[_CorpusItem]] = defaultdict(list)
                for item in selected:
                    if item.record.container_member_path is not None:
                        nested[item.record.container_member_path].append(item)
                for container_path in sorted(nested):
                    group = tuple(value.record for value in nested[container_path])
                    info = _bound_container_info(outer_index, group, policy)
                    with _stage_nested_container(outer, info, policy) as nested_stream:
                        with zipfile.ZipFile(nested_stream) as inner:
                            inner_index = _unique_info_index(inner)
                            for item in sorted(
                                nested[container_path],
                                key=lambda value: value.record.source_sample_id,
                            ):
                                append(
                                    item,
                                    inner,
                                    _bound_member_info(inner_index, item.record, policy),
                                )
            flush()
            _verify_archive_stability(
                authenticated.source.archive_path, stream.fileno(), initial_stat
            )
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _validate_official_regression(
    bundle: Mapping[str, Any], build: PdqNativeBuildReceipt
) -> str:
    expected = {
        "schema_version",
        "receipt",
        "receipt_sha256",
        "tool_provenance",
        "tool_provenance_sha256",
    }
    _exact(bundle, expected, "official PDQ regression bundle")
    receipt = bundle["receipt"]
    if (
        bundle["schema_version"] != "cvi.pdq_official_regression_bundle.v1"
        or not isinstance(receipt, Mapping)
        or content_sha256(receipt) != bundle["receipt_sha256"]
        or content_sha256(bundle["tool_provenance"])
        != bundle["tool_provenance_sha256"]
        or receipt.get("decision") != "PASS_EXACT_FIXED_COMMIT_OFFICIAL_REGRESSION"
        or receipt.get("native_binary_sha256") != build.binary_sha256
        or receipt.get("native_build_receipt_sha256") != build.receipt_sha256
    ):
        raise ValueError("official PDQ regression receipt binding differs")
    return content_sha256(bundle)


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
