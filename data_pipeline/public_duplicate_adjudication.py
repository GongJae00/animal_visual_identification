"""Receipt-bound duplicate adjudication for the public canine corpus.

Candidate generation remains label-blind. This module joins opaque candidates
to split sample tokens only after generation, records one outcome per candidate,
and refuses promotion while any candidate or required evidence channel remains
unresolved. Chunk boundaries make the inexpensive join/adjudication pass
bounded and resumable without permitting partial publication as a frozen graph.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from identity_methods.classical.geometric_verifier import (
    GeometricDecision,
    GeometricVerifierEvidence,
    read_geometric_evidence_bundle,
)
from identity_methods.classical.pdq_contracts import PDQSearchPolicy, PDQSearchResult
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from identity_governance.protected_public_split import (
    EvidenceRelation,
    FrozenPublicSplitEvidenceGraph,
    PublicSplitEvidenceEdge,
    PublicSplitSourceBundle,
)
from foundation.provenance import content_sha256


MAXIMUM_CANDIDATES = 1_000_000
MAXIMUM_CHUNK_CANDIDATES = 100_000
_IMAGE_DECISION = "PASS_IMAGE_CONTENT_AUDIT"
_IMAGE_INTERPRETATION = (
    "DECODE_AND_PIXEL_EXACT_DUPLICATE_EVIDENCE_ONLY_NOT_SPLIT_OR_MODEL_ADMISSION"
)


def validate_dinov2_filter_for_corpus(*_args: Any, **_kwargs: Any) -> Any:
    """Fail closed when an external, project-specific DINO filter is absent."""

    raise RuntimeError(
        "DINOv2 duplicate-filter admission is not part of the public package"
    )


def validate_admission_for_corpus(*_args: Any, **_kwargs: Any) -> Any:
    """Fail closed when an external, project-specific PDQ admission is absent."""

    raise RuntimeError(
        "PDQ transform admission is not part of the public package"
    )


class CandidateOutcome(StrEnum):
    EXACT_CONFIRMED = "EXACT_CONFIRMED"
    DECLARED_IDENTITY_COMPONENT_CLOSED = "DECLARED_IDENTITY_COMPONENT_CLOSED"
    GEOMETRIC_CONFIRMED = "GEOMETRIC_CONFIRMED"
    GEOMETRIC_REJECTED = "GEOMETRIC_REJECTED"
    REVIEW_CONFIRMED = "REVIEW_CONFIRMED"
    REVIEW_REJECTED = "REVIEW_REJECTED"
    CANDIDATE_COMPONENT_DEPENDENCY = "CANDIDATE_COMPONENT_DEPENDENCY"
    PHASH_ONLY_REJECTED_BY_ADMITTED_PDQ = (
        "PHASH_ONLY_REJECTED_BY_ADMITTED_PDQ"
    )
    PHASH_ONLY_REJECTED_BY_ADMITTED_DINOV2 = (
        "PHASH_ONLY_REJECTED_BY_ADMITTED_DINOV2"
    )
    UNRESOLVED = "UNRESOLVED"


class AdjudicationMode(StrEnum):
    STANDARD = "STANDARD"
    LEAKAGE_CONSERVATIVE_CANDIDATE_COMPONENT_CLOSURE = (
        "LEAKAGE_CONSERVATIVE_CANDIDATE_COMPONENT_CLOSURE"
    )
    PDQ_COMPLETE_NEGATIVE_FILTER = "PDQ_COMPLETE_NEGATIVE_FILTER"
    DINOV2_TRANSFORM_FAMILY_FILTER = "DINOV2_TRANSFORM_FAMILY_FILTER"


@dataclass(frozen=True, slots=True)
class ExactDuplicatePair:
    left_sample_token: str
    right_sample_token: str
    pixel_sha256: str
    evidence_token: str
    schema_version: str = "cvi.exact_duplicate_pair.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.exact_duplicate_pair.v1":
            raise ValueError("unsupported exact duplicate pair schema")
        _ordered_pair(self.left_sample_token, self.right_sample_token)
        _sha256(self.pixel_sha256, "pixel SHA-256")
        expected = content_sha256({
            "schema_version": self.schema_version,
            "left_sample_token": self.left_sample_token,
            "right_sample_token": self.right_sample_token,
            "pixel_sha256": self.pixel_sha256,
        })
        if self.evidence_token != expected:
            raise ValueError("exact duplicate evidence token differs")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExactDuplicatePair":
        _exact(payload, set(cls.__dataclass_fields__), "exact duplicate pair")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ExactDuplicateGraph:
    source_corpus_sha256: str
    image_content_receipts_sha256: str
    opaque_binding_sha256: str
    sample_count: int
    pairs: tuple[ExactDuplicatePair, ...]
    decision: str = "PASS_AUTHENTICATED_PIXEL_EXACT_GRAPH"
    interpretation: str = "PIXEL_EXACT_COMPONENTS_ONLY_NOT_NEAR_DUPLICATE_ADJUDICATION"
    schema_version: str = "cvi.exact_duplicate_graph.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.exact_duplicate_graph.v2":
            raise ValueError("unsupported exact duplicate graph schema")
        for name in (
            "source_corpus_sha256",
            "image_content_receipts_sha256",
            "opaque_binding_sha256",
        ):
            _sha256(getattr(self, name), name)
        _positive_int(self.sample_count, "sample_count")
        expected = tuple(sorted(
            self.pairs,
            key=lambda item: (item.left_sample_token, item.right_sample_token),
        ))
        if self.pairs != expected or len({
            (item.left_sample_token, item.right_sample_token) for item in self.pairs
        }) != len(self.pairs):
            raise ValueError("exact duplicate pairs must be sorted and unique")

    @property
    def graph_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_corpus_sha256": self.source_corpus_sha256,
            "image_content_receipts_sha256": self.image_content_receipts_sha256,
            "opaque_binding_sha256": self.opaque_binding_sha256,
            "sample_count": self.sample_count,
            "pairs": [item.to_dict() for item in self.pairs],
            "pair_count": len(self.pairs),
            "decision": self.decision,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExactDuplicateGraph":
        expected = set(cls.__dataclass_fields__) | {"pair_count"}
        _exact(payload, expected, "exact duplicate graph")
        if not isinstance(payload["pairs"], list):
            raise TypeError("exact duplicate graph pairs must be a JSON array")
        pairs = tuple(ExactDuplicatePair.from_dict(item) for item in payload["pairs"])
        if payload["pair_count"] != len(pairs):
            raise ValueError("exact duplicate pair count differs")
        values = {key: value for key, value in payload.items() if key != "pair_count"}
        values["pairs"] = pairs
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CandidateAdjudication:
    left_sample_token: str
    right_sample_token: str
    candidate_channels: tuple[str, ...]
    candidate_evidence_tokens: tuple[str, ...]
    outcome: CandidateOutcome
    reason: str
    decision_evidence_tokens: tuple[str, ...]
    schema_version: str = "cvi.public_duplicate_candidate_adjudication.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_duplicate_candidate_adjudication.v1":
            raise ValueError("unsupported candidate adjudication schema")
        _ordered_pair(self.left_sample_token, self.right_sample_token)
        _canonical_tokens(self.candidate_channels, "candidate channels", digest=False)
        _canonical_tokens(self.candidate_evidence_tokens, "candidate evidence tokens")
        _canonical_tokens(
            self.decision_evidence_tokens, "decision evidence tokens", allow_empty=True
        )
        if not isinstance(self.outcome, CandidateOutcome):
            raise TypeError("candidate outcome must be typed")
        if not isinstance(self.reason, str) or not self.reason or len(self.reason) > 256:
            raise ValueError("candidate reason must be bounded nonempty text")
        if self.outcome is not CandidateOutcome.UNRESOLVED and not (
            self.decision_evidence_tokens
        ):
            raise ValueError("decisive candidate outcome requires evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "left_sample_token": self.left_sample_token,
            "right_sample_token": self.right_sample_token,
            "candidate_channels": list(self.candidate_channels),
            "candidate_evidence_tokens": list(self.candidate_evidence_tokens),
            "outcome": self.outcome.value,
            "reason": self.reason,
            "decision_evidence_tokens": list(self.decision_evidence_tokens),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateAdjudication":
        _exact(payload, set(cls.__dataclass_fields__), "candidate adjudication")
        for name in (
            "candidate_channels",
            "candidate_evidence_tokens",
            "decision_evidence_tokens",
        ):
            if not isinstance(payload[name], list):
                raise TypeError(f"{name} must be a JSON array")
        return cls(
            left_sample_token=payload["left_sample_token"],
            right_sample_token=payload["right_sample_token"],
            candidate_channels=tuple(payload["candidate_channels"]),
            candidate_evidence_tokens=tuple(payload["candidate_evidence_tokens"]),
            outcome=CandidateOutcome(payload["outcome"]),
            reason=payload["reason"],
            decision_evidence_tokens=tuple(payload["decision_evidence_tokens"]),
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class AdjudicationChunk:
    source_bundle_sha256: str
    candidate_set_sha256: str
    evidence_bindings: tuple[tuple[str, str], ...]
    total_candidate_count: int
    start_index: int
    end_index: int
    records: tuple[CandidateAdjudication, ...]
    global_blockers: tuple[str, ...]
    mode: AdjudicationMode = AdjudicationMode.STANDARD
    unbound_candidate_count: int = 0
    schema_version: str = "cvi.public_duplicate_adjudication_chunk.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_duplicate_adjudication_chunk.v2":
            raise ValueError("unsupported adjudication chunk schema")
        _sha256(self.source_bundle_sha256, "source bundle SHA-256")
        _sha256(self.candidate_set_sha256, "candidate set SHA-256")
        _binding_rows(self.evidence_bindings)
        _nonnegative_int(self.total_candidate_count, "total_candidate_count")
        _nonnegative_int(self.start_index, "start_index")
        _nonnegative_int(self.end_index, "end_index")
        if not self.start_index < self.end_index <= self.total_candidate_count:
            raise ValueError("adjudication chunk range differs")
        if len(self.records) != self.end_index - self.start_index:
            raise ValueError("adjudication chunk cardinality differs")
        _canonical_blockers(self.global_blockers)
        if not isinstance(self.mode, AdjudicationMode):
            raise TypeError("adjudication mode must be typed")
        _nonnegative_int(self.unbound_candidate_count, "unbound_candidate_count")
        if (
            self.mode
            is AdjudicationMode.LEAKAGE_CONSERVATIVE_CANDIDATE_COMPONENT_CLOSURE
            and self.unbound_candidate_count != 0
        ):
            raise ValueError("conservative closure requires zero unbound candidates")

    @property
    def chunk_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_bundle_sha256": self.source_bundle_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "evidence_bindings": [list(item) for item in self.evidence_bindings],
            "total_candidate_count": self.total_candidate_count,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "records": [item.to_dict() for item in self.records],
            "global_blockers": list(self.global_blockers),
            "mode": self.mode.value,
            "unbound_candidate_count": self.unbound_candidate_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdjudicationChunk":
        _exact(payload, set(cls.__dataclass_fields__), "adjudication chunk")
        if not isinstance(payload["records"], list):
            raise TypeError("adjudication chunk records must be a JSON array")
        return cls(
            source_bundle_sha256=payload["source_bundle_sha256"],
            candidate_set_sha256=payload["candidate_set_sha256"],
            evidence_bindings=_pairs(payload["evidence_bindings"]),
            total_candidate_count=payload["total_candidate_count"],
            start_index=payload["start_index"],
            end_index=payload["end_index"],
            records=tuple(
                CandidateAdjudication.from_dict(item) for item in payload["records"]
            ),
            global_blockers=tuple(payload["global_blockers"]),
            mode=AdjudicationMode(payload["mode"]),
            unbound_candidate_count=payload["unbound_candidate_count"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class AdjudicationLedger:
    source_bundle_sha256: str
    candidate_set_sha256: str
    evidence_bindings: tuple[tuple[str, str], ...]
    records: tuple[CandidateAdjudication, ...]
    outcome_counts: tuple[tuple[str, int], ...]
    global_blockers: tuple[str, ...]
    promotion_status: str
    mode: AdjudicationMode = AdjudicationMode.STANDARD
    unbound_candidate_count: int = 0
    schema_version: str = "cvi.public_duplicate_adjudication_ledger.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_duplicate_adjudication_ledger.v2":
            raise ValueError("unsupported adjudication ledger schema")
        _sha256(self.source_bundle_sha256, "source bundle SHA-256")
        _sha256(self.candidate_set_sha256, "candidate set SHA-256")
        _binding_rows(self.evidence_bindings)
        pairs = tuple(
            (item.left_sample_token, item.right_sample_token) for item in self.records
        )
        if pairs != tuple(sorted(pairs)) or len(pairs) != len(set(pairs)):
            raise ValueError("adjudication ledger records must be pair-sorted unique")
        expected_counts = tuple(sorted(Counter(
            item.outcome.value for item in self.records
        ).items()))
        if self.outcome_counts != expected_counts:
            raise ValueError("adjudication ledger outcome counts differ")
        _canonical_blockers(self.global_blockers)
        if not isinstance(self.mode, AdjudicationMode):
            raise TypeError("adjudication mode must be typed")
        _nonnegative_int(self.unbound_candidate_count, "unbound_candidate_count")
        unresolved = dict(expected_counts).get(CandidateOutcome.UNRESOLVED.value, 0)
        expected_status = (
            "READY_FOR_GRAPH_PROMOTION"
            if (
                not self.global_blockers
                and unresolved == 0
                and self.unbound_candidate_count == 0
            )
            else "BLOCKED"
        )
        if self.promotion_status != expected_status:
            raise ValueError("adjudication ledger promotion status differs")

    @property
    def ledger_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_bundle_sha256": self.source_bundle_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "evidence_bindings": [list(item) for item in self.evidence_bindings],
            "records": [item.to_dict() for item in self.records],
            "candidate_count": len(self.records),
            "outcome_counts": dict(self.outcome_counts),
            "global_blockers": list(self.global_blockers),
            "promotion_status": self.promotion_status,
            "mode": self.mode.value,
            "unbound_candidate_count": self.unbound_candidate_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdjudicationLedger":
        expected = set(cls.__dataclass_fields__) | {"candidate_count"}
        _exact(payload, expected, "adjudication ledger")
        if not isinstance(payload["records"], list) or not isinstance(
            payload["outcome_counts"], Mapping
        ):
            raise TypeError("adjudication ledger collection fields differ")
        records = tuple(
            CandidateAdjudication.from_dict(item) for item in payload["records"]
        )
        if payload["candidate_count"] != len(records):
            raise ValueError("adjudication ledger candidate count differs")
        return cls(
            source_bundle_sha256=payload["source_bundle_sha256"],
            candidate_set_sha256=payload["candidate_set_sha256"],
            evidence_bindings=_pairs(payload["evidence_bindings"]),
            records=records,
            outcome_counts=tuple(sorted(payload["outcome_counts"].items())),
            global_blockers=tuple(payload["global_blockers"]),
            promotion_status=payload["promotion_status"],
            mode=AdjudicationMode(payload["mode"]),
            unbound_candidate_count=payload["unbound_candidate_count"],
            schema_version=payload["schema_version"],
        )


def build_exact_duplicate_graph(
    *,
    source: PublicSplitSourceBundle,
    image_receipts: Mapping[str, Any],
    opaque_binding_bundle: Mapping[str, Any],
) -> ExactDuplicateGraph:
    """Authenticate exact-pixel groups and map them to opaque split tokens."""

    source_by_id = {item.source_sample_id: item for item in source.samples}
    opaque_to_source, binding_sha256, _ = _validate_opaque_binding(
        opaque_binding_bundle, source_by_id
    )
    if set(opaque_to_source.values()) != set(source_by_id):
        raise ValueError("opaque binding does not exactly cover source samples")
    seen_source_ids: set[str] = set()
    pairs: list[ExactDuplicatePair] = []
    for dataset_key, bundle in sorted(image_receipts.items()):
        if not isinstance(dataset_key, str) or not isinstance(bundle, Mapping):
            raise TypeError("merged image receipt entries must be objects")
        receipt = _validate_image_bundle(bundle)
        records = receipt["records"]
        record_pixels: dict[str, str] = {}
        for record in records:
            source_id = record["source_sample_id"]
            if source_id in seen_source_ids or source_id not in source_by_id:
                raise ValueError("image receipt source coverage differs")
            if source_by_id[source_id].dataset_name != record["dataset_name"]:
                raise ValueError("image receipt dataset binding differs")
            seen_source_ids.add(source_id)
            record_pixels[source_id] = record["pixel_sha256"]
        for group in receipt["exact_duplicate_groups"]:
            ids = group["source_sample_ids"]
            pixel_sha256 = group["pixel_sha256"]
            if any(record_pixels.get(value) != pixel_sha256 for value in ids):
                raise ValueError("exact group pixel binding differs")
            tokens = sorted(source_by_id[value].sample_token for value in ids)
            for left_index, left in enumerate(tokens):
                for right in tokens[left_index + 1:]:
                    payload = {
                        "schema_version": "cvi.exact_duplicate_pair.v1",
                        "left_sample_token": left,
                        "right_sample_token": right,
                        "pixel_sha256": pixel_sha256,
                    }
                    pairs.append(ExactDuplicatePair(
                        left_sample_token=left,
                        right_sample_token=right,
                        pixel_sha256=pixel_sha256,
                        evidence_token=content_sha256(payload),
                    ))
    if seen_source_ids != set(source_by_id):
        raise ValueError("image receipts do not exactly cover source bundle")
    return ExactDuplicateGraph(
        source_corpus_sha256=source_corpus_sha256(source),
        image_content_receipts_sha256=content_sha256(image_receipts),
        opaque_binding_sha256=binding_sha256,
        sample_count=len(source.samples),
        pairs=tuple(sorted(
            pairs, key=lambda item: (item.left_sample_token, item.right_sample_token)
        )),
    )


def source_corpus_sha256(source: PublicSplitSourceBundle) -> str:
    """Hash immutable sample semantics without circular evidence bindings."""

    return content_sha256({
        "schema_version": "cvi.public_split_source_corpus.v1",
        "samples": [
            item.to_dict()
            for item in sorted(source.samples, key=lambda value: value.sample_token)
        ],
    })


def build_duplicate_evidence_source_generation(
    *,
    source: PublicSplitSourceBundle,
    exact_graph: ExactDuplicateGraph,
    exact_graph_bundle: Mapping[str, Any],
    phash_evidence_bundle: Mapping[str, Any],
    pdq_evidence_bundle: Mapping[str, Any],
    opaque_binding_bundle: Mapping[str, Any],
) -> PublicSplitSourceBundle:
    """Return a new immutable source generation bound to complete core evidence."""

    if exact_graph.source_corpus_sha256 != source_corpus_sha256(source):
        raise ValueError("exact graph source corpus binding differs")
    _validate_exact_graph_bundle(exact_graph_bundle, exact_graph)
    source_by_id = {item.source_sample_id: item for item in source.samples}
    opaque_to_source, binding_sha256, _ = _validate_opaque_binding(
        opaque_binding_bundle, source_by_id
    )
    if set(opaque_to_source.values()) != set(source_by_id):
        raise ValueError("opaque binding does not exactly cover source corpus")
    if exact_graph.opaque_binding_sha256 != binding_sha256:
        raise ValueError("exact graph opaque binding differs")
    if dict(source.evidence_bindings)["image_content_receipts_sha256"] != (
        exact_graph.image_content_receipts_sha256
    ):
        raise ValueError("exact graph image receipt binding differs")
    sample_by_opaque = {
        opaque: source_by_id[source_id].sample_token
        for opaque, source_id in opaque_to_source.items()
    }
    candidates: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    phash_sha256 = content_sha256(phash_evidence_bundle)
    pdq_sha256 = content_sha256(pdq_evidence_bundle)
    _add_phash_candidates(
        candidates, phash_evidence_bundle, sample_by_opaque, phash_sha256
    )
    _add_pdq_candidates(
        candidates, pdq_evidence_bundle, sample_by_opaque, pdq_sha256
    )
    exact_bundle_sha256 = content_sha256(exact_graph_bundle)
    bindings = dict(source.evidence_bindings)
    bindings.update({
        "exact_duplicate_graph_sha256": exact_bundle_sha256,
        "phash_candidates_sha256": phash_sha256,
        "pdq_candidates_sha256": pdq_sha256,
    })
    return PublicSplitSourceBundle(
        evidence_bindings=tuple(sorted(bindings.items())),
        samples=source.samples,
    )


def publish_source_generation(
    path: Path, source: PublicSplitSourceBundle
) -> str:
    write_private_json_bundle(((path, source.to_dict()),))
    return source.bundle_sha256


def build_adjudication_chunk(
    *,
    source: PublicSplitSourceBundle,
    exact_graph: ExactDuplicateGraph,
    exact_graph_artifact_sha256: str,
    phash_evidence_bundle: Mapping[str, Any],
    opaque_binding_bundle: Mapping[str, Any],
    start_index: int,
    maximum_candidates: int,
    pdq_evidence_bundle: Mapping[str, Any] | None = None,
    pdq_transform_admission: Mapping[str, Any] | None = None,
    dinov2_filter_evidence: Mapping[str, Any] | None = None,
    geometric_evidence: Sequence[GeometricVerifierEvidence] = (),
    geometric_admission_receipt: Mapping[str, Any] | None = None,
    review_bundle: Mapping[str, Any] | None = None,
    mode: AdjudicationMode = AdjudicationMode.STANDARD,
) -> AdjudicationChunk:
    """Build one deterministic candidate-range chunk with explicit outcomes."""

    _nonnegative_int(start_index, "start_index")
    _positive_int(maximum_candidates, "maximum_candidates")
    _sha256(exact_graph_artifact_sha256, "exact graph artifact SHA-256")
    if not isinstance(mode, AdjudicationMode):
        raise TypeError("adjudication mode must be typed")
    if maximum_candidates > MAXIMUM_CHUNK_CANDIDATES:
        raise ValueError("maximum_candidates exceeds adjudication chunk cap")
    source_by_id = {item.source_sample_id: item for item in source.samples}
    source_by_token = {item.sample_token: item for item in source.samples}
    source_bundle_sha256 = source.bundle_sha256
    opaque_to_source, binding_sha256, binding_artifact_sha256 = (
        _validate_opaque_binding(opaque_binding_bundle, source_by_id)
    )
    sample_by_opaque = {
        opaque: source_by_id[source_id].sample_token
        for opaque, source_id in opaque_to_source.items()
    }
    if exact_graph.source_corpus_sha256 != source_corpus_sha256(source):
        raise ValueError("exact graph source corpus binding differs")
    if exact_graph.opaque_binding_sha256 != binding_sha256:
        raise ValueError("exact graph opaque binding differs")

    pair_channels: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    phash_artifact_sha256 = content_sha256(phash_evidence_bundle)
    _add_phash_candidates(
        pair_channels, phash_evidence_bundle, sample_by_opaque, phash_artifact_sha256
    )
    blockers: set[str] = set()
    source_bindings = dict(source.evidence_bindings)
    if source_bindings["image_content_receipts_sha256"] != (
        exact_graph.image_content_receipts_sha256
    ):
        blockers.add("SOURCE_IMAGE_RECEIPT_BINDING_STALE")
    if source_bindings["exact_duplicate_graph_sha256"] != (
        exact_graph_artifact_sha256
    ):
        blockers.add("SOURCE_EXACT_GRAPH_BINDING_STALE")
    if source_bindings["phash_candidates_sha256"] != phash_artifact_sha256:
        blockers.add("SOURCE_PHASH_BINDING_STALE")
    pdq_artifact_sha256: str | None = None
    pdq_search: PDQSearchResult | None = None
    if pdq_evidence_bundle is None:
        blockers.add("PDQ_CORPUS_EVIDENCE_MISSING")
    else:
        pdq_artifact_sha256 = content_sha256(pdq_evidence_bundle)
        pdq_search = _add_pdq_candidates(
            pair_channels,
            pdq_evidence_bundle,
            sample_by_opaque,
            pdq_artifact_sha256,
        )
        if source_bindings["pdq_candidates_sha256"] != pdq_artifact_sha256:
            blockers.add("SOURCE_PDQ_BINDING_STALE")
    exact_by_pair = {
        (item.left_sample_token, item.right_sample_token): item
        for item in exact_graph.pairs
    }
    for pair, item in exact_by_pair.items():
        pair_channels[pair]["EXACT"] = item.evidence_token

    ordered_pairs = tuple(sorted(pair_channels))
    if len(ordered_pairs) > MAXIMUM_CANDIDATES:
        raise ValueError("candidate union exceeds adjudication cap")
    candidate_set_sha256 = content_sha256([
        {
            "left_sample_token": pair[0],
            "right_sample_token": pair[1],
            "candidate_channels": sorted(pair_channels[pair]),
            "candidate_evidence_tokens": sorted(pair_channels[pair].values()),
        }
        for pair in ordered_pairs
    ])
    if start_index >= len(ordered_pairs):
        raise ValueError("start_index is outside candidate set")
    end_index = min(len(ordered_pairs), start_index + maximum_candidates)

    geometry_by_pair, geometry_artifact_sha256 = _geometry_results(
        geometric_evidence, sample_by_opaque
    )
    conservative = (
        mode
        is AdjudicationMode.LEAKAGE_CONSERVATIVE_CANDIDATE_COMPONENT_CLOSURE
    )
    pdq_negative_filter = mode is AdjudicationMode.PDQ_COMPLETE_NEGATIVE_FILTER
    dinov2_filter = mode is AdjudicationMode.DINOV2_TRANSFORM_FAMILY_FILTER
    if not geometric_evidence:
        if not conservative and not pdq_negative_filter and not dinov2_filter:
            blockers.add("GEOMETRIC_EVIDENCE_MISSING")
    elif source_bindings["geometric_verifier_sha256"] != geometry_artifact_sha256:
        blockers.add("SOURCE_GEOMETRIC_BINDING_STALE")
    geometry_admitted = _geometry_admitted(
        geometric_admission_receipt, geometric_evidence
    )
    reviews, review_artifact_sha256 = _review_results(
        review_bundle, candidate_set_sha256
    )
    if review_bundle is None:
        if not conservative and not pdq_negative_filter and not dinov2_filter:
            blockers.add("REVIEW_ADJUDICATION_RECEIPT_MISSING")
    elif source_bindings["review_adjudication_sha256"] != review_artifact_sha256:
        blockers.add("SOURCE_REVIEW_BINDING_STALE")

    pdq_admission_sha256: str | None = None
    pdq_admitted = False
    if pdq_negative_filter:
        if pdq_transform_admission is None:
            blockers.add("PDQ_TRANSFORM_ADMISSION_MISSING")
        elif pdq_evidence_bundle is None:
            blockers.add("PDQ_CORPUS_EVIDENCE_MISSING")
        else:
            pdq_admission_sha256 = validate_admission_for_corpus(
                pdq_transform_admission, pdq_evidence_bundle
            )
            pdq_admitted = True

    dinov2_filter_sha256: str | None = None
    dinov2_by_pair: dict[tuple[str, str], tuple[str, float | None, str]] = {}
    if dinov2_filter:
        if dinov2_filter_evidence is None:
            blockers.add("DINOV2_TRANSFORM_FILTER_EVIDENCE_MISSING")
        elif pdq_evidence_bundle is None:
            blockers.add("PDQ_CORPUS_EVIDENCE_MISSING")
        else:
            dinov2_filter_sha256, by_opaque = validate_dinov2_filter_for_corpus(
                dinov2_filter_evidence,
                phash_evidence_bundle,
                pdq_evidence_bundle,
            )
            for opaque_pair, result in by_opaque.items():
                try:
                    pair = tuple(sorted((
                        sample_by_opaque[opaque_pair[0]],
                        sample_by_opaque[opaque_pair[1]],
                    )))
                except KeyError as error:
                    raise ValueError(
                        "DINOv2 filter references an unknown corpus sample"
                    ) from error
                if pair in dinov2_by_pair:
                    raise ValueError("duplicate DINOv2 filter pair after source join")
                dinov2_by_pair[pair] = result

    eligible_pdq_samples = (
        set(pdq_search.eligible_sample_ids) if pdq_search is not None else set()
    )
    eligible_pdq_tokens = {
        sample_by_opaque[opaque] for opaque in eligible_pdq_samples
    }
    records: list[CandidateAdjudication] = []
    for pair in ordered_pairs[start_index:end_index]:
        channels = pair_channels[pair]
        exact = exact_by_pair.get(pair)
        review = reviews.get(pair)
        geometry = geometry_by_pair.get(pair)
        if conservative:
            outcome = CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY
            reason = "AUTHENTICATED_CANDIDATE_COMPONENT_LEAKAGE_CLOSURE_ONLY"
            decision_tokens = tuple(sorted(channels.values()))
        elif dinov2_filter:
            dino = dinov2_by_pair.get(pair)
            if exact is not None or "PDQ" in channels or (
                geometry is not None
                and geometry.decision is GeometricDecision.GEOMETRIC_CONFIRMED
            ):
                outcome = CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY
                reason = (
                    "AUTHENTICATED_EXACT_PAIR_DEPENDENCY"
                    if exact is not None
                    else "PDQ_CANDIDATE_DEPENDENCY"
                    if "PDQ" in channels
                    else "GEOMETRIC_CONFIRMATION_DEPENDENCY"
                )
                decision_tokens = tuple(sorted(
                    set(channels.values())
                    | ({geometry.evidence_token} if geometry is not None else set())
                ))
            elif dino is None:
                outcome = CandidateOutcome.UNRESOLVED
                reason = (
                    "DINOV2_FILTER_PAIR_COVERAGE_MISSING"
                    if dinov2_filter_sha256 is not None
                    else "DINOV2_TRANSFORM_FILTER_NOT_ADMITTED"
                )
                decision_tokens = ()
            elif dino[0] == "LEAKAGE_FILTER_REJECTION":
                outcome = CandidateOutcome.PHASH_ONLY_REJECTED_BY_ADMITTED_DINOV2
                reason = "BELOW_ADMITTED_DINOV2_THRESHOLD"
                decision_tokens = (dino[2],)
            else:
                outcome = CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY
                reason = "DINOV2_ABOVE_THRESHOLD_OR_INVALID_DEPENDENCY"
                decision_tokens = tuple(sorted((*channels.values(), dino[2])))
        elif pdq_negative_filter:
            if exact is not None or "PDQ" in channels:
                outcome = CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY
                reason = (
                    "AUTHENTICATED_EXACT_PAIR_DEPENDENCY"
                    if exact is not None
                    else "PDQ_RADIUS_31_CANDIDATE_DEPENDENCY"
                )
                decision_tokens = tuple(sorted(channels.values()))
            elif (
                pdq_admitted
                and set(pair) <= eligible_pdq_tokens
                and tuple(sorted(channels)) == ("PHASH",)
            ):
                outcome = CandidateOutcome.PHASH_ONLY_REJECTED_BY_ADMITTED_PDQ
                reason = "PDQ_COMPLETE_NEGATIVE"
                decision_tokens = (
                    content_sha256({
                        "schema_version": "cvi.pdq_complete_negative_decision.v1",
                        "left_sample_token": pair[0],
                        "right_sample_token": pair[1],
                        "pdq_transform_admission_sha256": pdq_admission_sha256,
                        "pdq_candidates_sha256": pdq_artifact_sha256,
                        "distance_threshold": 31,
                        "interpretation": (
                            "LEAKAGE_CANDIDATE_FILTERING_ONLY_NOT_BIOMETRIC_NONMATCH"
                        ),
                    }),
                )
            else:
                outcome = CandidateOutcome.UNRESOLVED
                reason = (
                    "PDQ_INELIGIBLE_LOW_QUALITY_FAIL_CLOSED"
                    if pdq_admitted
                    else "PDQ_TRANSFORM_POLICY_NOT_ADMITTED"
                )
                decision_tokens = ()
        elif exact is not None:
            if review is not None and review[0] == "REVIEW_REJECTED":
                raise ValueError("review contradicts authenticated exact pixels")
            outcome = CandidateOutcome.EXACT_CONFIRMED
            reason = "AUTHENTICATED_CANONICAL_RGB_DIGEST_EQUAL"
            decision_tokens = (exact.evidence_token,)
        elif source_by_token[pair[0]].identity_token == source_by_token[
            pair[1]
        ].identity_token:
            outcome = CandidateOutcome.DECLARED_IDENTITY_COMPONENT_CLOSED
            reason = "DECLARED_IDENTITY_ALREADY_ROLE_CLOSED_NOT_A_DUPLICATE_DECISION"
            decision_tokens = (content_sha256({
                "source_bundle_sha256": source_bundle_sha256,
                "left_sample_token": pair[0],
                "right_sample_token": pair[1],
                "outcome": outcome.value,
            }),)
        elif review is not None and review[0] != "REVIEW_UNRESOLVED":
            outcome = (
                CandidateOutcome.REVIEW_CONFIRMED
                if review[0] == "REVIEW_CONFIRMED"
                else CandidateOutcome.REVIEW_REJECTED
            )
            reason = review[1]
            decision_tokens = (review[2],)
        elif review is not None:
            outcome = CandidateOutcome.UNRESOLVED
            reason = "HUMAN_REVIEW_UNRESOLVED"
            decision_tokens = (review[2],)
        elif geometry is not None and geometry_admitted:
            if geometry.decision is GeometricDecision.GEOMETRIC_CONFIRMED:
                outcome = CandidateOutcome.GEOMETRIC_CONFIRMED
                reason = geometry.reason.value
            elif geometry.decision is GeometricDecision.GEOMETRIC_REJECTED:
                outcome = CandidateOutcome.GEOMETRIC_REJECTED
                reason = geometry.reason.value
            else:
                outcome = CandidateOutcome.UNRESOLVED
                reason = geometry.reason.value
            decision_tokens = (geometry.evidence_token,)
        elif geometry is not None:
            outcome = CandidateOutcome.UNRESOLVED
            reason = "GEOMETRIC_POLICY_NOT_ADMITTED"
            decision_tokens = (geometry.evidence_token,)
        else:
            outcome = CandidateOutcome.UNRESOLVED
            reason = "NO_DECISIVE_GEOMETRIC_OR_REVIEW_EVIDENCE"
            decision_tokens = ()
        records.append(CandidateAdjudication(
            left_sample_token=pair[0],
            right_sample_token=pair[1],
            candidate_channels=tuple(sorted(channels)),
            candidate_evidence_tokens=tuple(sorted(channels.values())),
            outcome=outcome,
            reason=reason,
            decision_evidence_tokens=tuple(sorted(decision_tokens)),
        ))

    evidence_bindings = {
        "exact_duplicate_graph_sha256": exact_graph_artifact_sha256,
        "image_content_receipts_sha256": exact_graph.image_content_receipts_sha256,
        "opaque_binding_artifact_sha256": binding_artifact_sha256,
        "phash_candidates_sha256": phash_artifact_sha256,
    }
    if pdq_artifact_sha256 is not None:
        evidence_bindings["pdq_candidates_sha256"] = pdq_artifact_sha256
    if pdq_admission_sha256 is not None:
        evidence_bindings["pdq_transform_admission_sha256"] = pdq_admission_sha256
    if dinov2_filter_sha256 is not None:
        evidence_bindings["dinov2_duplicate_filter_sha256"] = dinov2_filter_sha256
    if geometry_artifact_sha256 is not None:
        evidence_bindings["geometric_verifier_sha256"] = geometry_artifact_sha256
    if review_artifact_sha256 is not None:
        evidence_bindings["review_adjudication_sha256"] = review_artifact_sha256
    return AdjudicationChunk(
        source_bundle_sha256=source_bundle_sha256,
        candidate_set_sha256=candidate_set_sha256,
        evidence_bindings=tuple(sorted(evidence_bindings.items())),
        total_candidate_count=len(ordered_pairs),
        start_index=start_index,
        end_index=end_index,
        records=tuple(records),
        global_blockers=tuple(sorted(blockers)),
        mode=mode,
        unbound_candidate_count=0,
    )


def merge_adjudication_chunks(chunks: Sequence[AdjudicationChunk]) -> AdjudicationLedger:
    """Merge an exact contiguous partition; gaps and overlaps fail closed."""

    if not chunks:
        raise ValueError("at least one adjudication chunk is required")
    ordered = tuple(sorted(chunks, key=lambda item: item.start_index))
    first = ordered[0]
    cursor = 0
    records: list[CandidateAdjudication] = []
    for chunk in ordered:
        if (
            chunk.source_bundle_sha256 != first.source_bundle_sha256
            or chunk.candidate_set_sha256 != first.candidate_set_sha256
            or chunk.evidence_bindings != first.evidence_bindings
            or chunk.total_candidate_count != first.total_candidate_count
            or chunk.global_blockers != first.global_blockers
            or chunk.mode is not first.mode
            or chunk.unbound_candidate_count != first.unbound_candidate_count
        ):
            raise ValueError("adjudication chunk lineage differs")
        if chunk.start_index != cursor:
            raise ValueError("adjudication chunks contain a gap or overlap")
        records.extend(chunk.records)
        cursor = chunk.end_index
    if cursor != first.total_candidate_count:
        raise ValueError("adjudication chunks do not cover every candidate")
    counts = tuple(sorted(Counter(item.outcome.value for item in records).items()))
    unresolved = dict(counts).get(CandidateOutcome.UNRESOLVED.value, 0)
    promotion = (
        "READY_FOR_GRAPH_PROMOTION"
        if (
            not first.global_blockers
            and unresolved == 0
            and first.unbound_candidate_count == 0
        )
        else "BLOCKED"
    )
    return AdjudicationLedger(
        source_bundle_sha256=first.source_bundle_sha256,
        candidate_set_sha256=first.candidate_set_sha256,
        evidence_bindings=first.evidence_bindings,
        records=tuple(records),
        outcome_counts=counts,
        global_blockers=first.global_blockers,
        promotion_status=promotion,
        mode=first.mode,
        unbound_candidate_count=first.unbound_candidate_count,
    )


def assemble_frozen_evidence_graph(
    *, source: PublicSplitSourceBundle, ledger: AdjudicationLedger
) -> FrozenPublicSplitEvidenceGraph:
    """Promote only a complete, blocker-free ledger bound to this source."""

    if ledger.source_bundle_sha256 != source.bundle_sha256:
        raise ValueError("adjudication ledger source binding differs")
    if ledger.promotion_status != "READY_FOR_GRAPH_PROMOTION":
        raise RuntimeError(
            "adjudication ledger is blocked; frozen graph requires zero unresolved "
            "candidates and zero global blockers"
        )
    expected_bindings = dict(source.evidence_bindings)
    supplied_bindings = dict(ledger.evidence_bindings)
    required = (
        (
            "exact_duplicate_graph_sha256",
            "phash_candidates_sha256",
            "pdq_candidates_sha256",
        )
        if ledger.mode
        is AdjudicationMode.LEAKAGE_CONSERVATIVE_CANDIDATE_COMPONENT_CLOSURE
        else (
            "exact_duplicate_graph_sha256",
            "phash_candidates_sha256",
            "pdq_candidates_sha256",
        )
        if ledger.mode in {
            AdjudicationMode.PDQ_COMPLETE_NEGATIVE_FILTER,
            AdjudicationMode.DINOV2_TRANSFORM_FAMILY_FILTER,
        }
        else (
            "exact_duplicate_graph_sha256",
            "geometric_verifier_sha256",
            "phash_candidates_sha256",
            "pdq_candidates_sha256",
            "review_adjudication_sha256",
        )
    )
    for name in required:
        if supplied_bindings.get(name) != expected_bindings.get(name):
            raise ValueError(f"source evidence binding differs for {name}")
    if ledger.mode is AdjudicationMode.PDQ_COMPLETE_NEGATIVE_FILTER:
        _sha256(
            supplied_bindings.get("pdq_transform_admission_sha256"),
            "pdq_transform_admission_sha256",
        )
    if ledger.mode is AdjudicationMode.DINOV2_TRANSFORM_FAMILY_FILTER:
        _sha256(
            supplied_bindings.get("dinov2_duplicate_filter_sha256"),
            "dinov2_duplicate_filter_sha256",
        )
    if ledger.unbound_candidate_count != 0:
        raise RuntimeError("frozen graph requires zero unbound candidates")
    if ledger.mode is AdjudicationMode.LEAKAGE_CONSERVATIVE_CANDIDATE_COMPONENT_CLOSURE:
        edges = _conservative_dependency_edges(source, ledger)
    elif ledger.mode is AdjudicationMode.PDQ_COMPLETE_NEGATIVE_FILTER:
        edges = _pdq_admitted_dependency_edges(source, ledger)
    elif ledger.mode is AdjudicationMode.DINOV2_TRANSFORM_FAMILY_FILTER:
        edges = _dinov2_admitted_dependency_edges(source, ledger)
    else:
        edges = _dependency_edges(source)
    for item in ledger.records:
        if ledger.mode in {
            AdjudicationMode.LEAKAGE_CONSERVATIVE_CANDIDATE_COMPONENT_CLOSURE,
            AdjudicationMode.PDQ_COMPLETE_NEGATIVE_FILTER,
            AdjudicationMode.DINOV2_TRANSFORM_FAMILY_FILTER,
        }:
            continue
        relation = {
            CandidateOutcome.EXACT_CONFIRMED: EvidenceRelation.EXACT_CONFIRMED,
            CandidateOutcome.GEOMETRIC_CONFIRMED: EvidenceRelation.GEOMETRIC_CONFIRMED,
            CandidateOutcome.REVIEW_CONFIRMED: EvidenceRelation.REVIEW_CONFIRMED,
            CandidateOutcome.REVIEW_REJECTED: EvidenceRelation.REVIEW_REJECTED,
        }.get(item.outcome)
        if relation is None:
            continue
        evidence_token = content_sha256({
            "ledger_sha256": ledger.ledger_sha256,
            "left_sample_token": item.left_sample_token,
            "right_sample_token": item.right_sample_token,
            "relation": relation.value,
            "decision_evidence_tokens": list(item.decision_evidence_tokens),
        })
        edges.append(PublicSplitEvidenceEdge(
            left_sample_token=item.left_sample_token,
            right_sample_token=item.right_sample_token,
            relation=relation,
            evidence_token=evidence_token,
        ))
    return FrozenPublicSplitEvidenceGraph(
        evidence_bindings=source.evidence_bindings,
        edges=tuple(sorted(edges, key=lambda item: (
            item.left_sample_token,
            item.right_sample_token,
            item.relation.value,
            item.evidence_token,
        ))),
    )


def publish_exact_graph(
    path: Path, graph: ExactDuplicateGraph, *, tool_provenance: Mapping[str, Any]
) -> str:
    bundle = _bundle(
        "cvi.exact_duplicate_graph_bundle.v2",
        "graph",
        graph.to_dict(),
        "graph_sha256",
        tool_provenance,
    )
    write_private_json_bundle(((path, bundle),))
    return bundle["bundle_sha256"]


def publish_adjudication_chunk(
    path: Path, chunk: AdjudicationChunk, *, tool_provenance: Mapping[str, Any]
) -> str:
    bundle = _bundle(
        "cvi.public_duplicate_adjudication_chunk_bundle.v2",
        "chunk",
        chunk.to_dict(),
        "chunk_sha256",
        tool_provenance,
    )
    write_private_json_bundle(((path, bundle),))
    return bundle["bundle_sha256"]


def publish_adjudication_ledger(
    path: Path, ledger: AdjudicationLedger, *, tool_provenance: Mapping[str, Any]
) -> str:
    bundle = _bundle(
        "cvi.public_duplicate_adjudication_ledger_bundle.v2",
        "ledger",
        ledger.to_dict(),
        "ledger_sha256",
        tool_provenance,
    )
    write_private_json_bundle(((path, bundle),))
    return bundle["bundle_sha256"]


def build_review_queue(
    *,
    source: PublicSplitSourceBundle,
    ledger: AdjudicationLedger,
    image_receipts: Mapping[str, Any],
    phash_evidence_bundle: Mapping[str, Any],
    opaque_binding_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve blocked pairs to authenticated source locations without deciding them."""

    if ledger.source_bundle_sha256 != source.bundle_sha256:
        raise ValueError("review queue source binding differs")
    if dict(ledger.evidence_bindings).get("phash_candidates_sha256") != content_sha256(
        phash_evidence_bundle
    ):
        raise ValueError("review queue pHash binding differs")
    source_by_id = {item.source_sample_id: item for item in source.samples}
    source_by_token = {item.sample_token: item for item in source.samples}
    opaque_to_source, _, binding_artifact_sha256 = _validate_opaque_binding(
        opaque_binding_bundle, source_by_id
    )
    if dict(ledger.evidence_bindings).get("opaque_binding_artifact_sha256") != (
        binding_artifact_sha256
    ):
        raise ValueError("review queue opaque binding differs")
    locations: dict[str, dict[str, Any]] = {}
    for bundle in image_receipts.values():
        receipt = _validate_image_bundle(bundle)
        for raw in receipt["records"]:
            source_id = raw["source_sample_id"]
            if source_id in locations:
                raise ValueError("duplicate review source location")
            locations[source_id] = {
                "dataset_name": raw["dataset_name"],
                "source_sample_id": source_id,
                "member_path": raw["member_path"],
                "container_member_path": raw["container_member_path"],
                "pixel_sha256": raw["pixel_sha256"],
            }
    if set(locations) != set(source_by_id):
        raise ValueError("review source locations do not exactly cover source bundle")

    distances: dict[tuple[str, str], int] = {}
    evidence = phash_evidence_bundle.get("evidence")
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get("candidates"), list):
        raise ValueError("review queue pHash candidates differ")
    for raw in evidence["candidates"]:
        left_source = opaque_to_source[raw["left_opaque_sample_id"]]
        right_source = opaque_to_source[raw["right_opaque_sample_id"]]
        pair = tuple(sorted((
            source_by_id[left_source].sample_token,
            source_by_id[right_source].sample_token,
        )))
        if pair in distances:
            raise ValueError("duplicate review queue pHash pair")
        distances[pair] = raw["hamming_distance"]

    rows: list[dict[str, Any]] = []
    for item in ledger.records:
        if item.outcome is not CandidateOutcome.UNRESOLVED:
            continue
        pair = (item.left_sample_token, item.right_sample_token)
        left = source_by_token[pair[0]]
        right = source_by_token[pair[1]]
        rows.append({
            "schema_version": "cvi.public_duplicate_review_queue_record.v1",
            "left_sample_token": pair[0],
            "right_sample_token": pair[1],
            "left_source": locations[left.source_sample_id],
            "right_source": locations[right.source_sample_id],
            "candidate_channels": list(item.candidate_channels),
            "phash_hamming_distance": distances.get(pair),
            "unresolved_reason": item.reason,
        })
    rows.sort(key=lambda row: (
        row["phash_hamming_distance"] is None,
        row["phash_hamming_distance"] if row["phash_hamming_distance"] is not None else 999,
        row["left_sample_token"],
        row["right_sample_token"],
    ))
    return {
        "schema_version": "cvi.public_duplicate_review_queue.v1",
        "source_bundle_sha256": source.bundle_sha256,
        "adjudication_ledger_sha256": ledger.ledger_sha256,
        "candidate_set_sha256": ledger.candidate_set_sha256,
        "records": rows,
        "record_count": len(rows),
        "decision": "REVIEW_REQUIRED_NO_OUTCOMES_ASSIGNED",
        "interpretation": "PROTECTED_SOURCE_LOCATION_QUEUE_NOT_ADJUDICATION_EVIDENCE",
    }


def publish_review_queue(
    path: Path, queue: Mapping[str, Any], *, tool_provenance: Mapping[str, Any]
) -> str:
    bundle = _bundle(
        "cvi.public_duplicate_review_queue_bundle.v1",
        "queue",
        dict(queue),
        "queue_sha256",
        tool_provenance,
    )
    write_private_json_bundle(((path, bundle),))
    return bundle["bundle_sha256"]


def read_source_bundle(path: Path) -> PublicSplitSourceBundle:
    return PublicSplitSourceBundle.from_dict(read_strict_json_object(path))


def read_exact_graph(path: Path) -> ExactDuplicateGraph:
    payload = _read_bundle(
        path,
        "cvi.exact_duplicate_graph_bundle.v2",
        "graph",
        "graph_sha256",
    )
    graph = ExactDuplicateGraph.from_dict(payload)
    return graph


def read_adjudication_chunk(path: Path) -> AdjudicationChunk:
    payload = _read_bundle(
        path,
        "cvi.public_duplicate_adjudication_chunk_bundle.v2",
        "chunk",
        "chunk_sha256",
    )
    return AdjudicationChunk.from_dict(payload)


def read_adjudication_ledger(path: Path) -> AdjudicationLedger:
    payload = _read_bundle(
        path,
        "cvi.public_duplicate_adjudication_ledger_bundle.v2",
        "ledger",
        "ledger_sha256",
    )
    return AdjudicationLedger.from_dict(payload)


def read_geometric_chunks(paths: Iterable[Path]) -> tuple[GeometricVerifierEvidence, ...]:
    return tuple(read_geometric_evidence_bundle(path) for path in paths)


def _add_phash_candidates(
    pairs: dict[tuple[str, str], dict[str, str]],
    bundle: Mapping[str, Any],
    sample_by_opaque: Mapping[str, str],
    artifact_sha256: str,
) -> None:
    _exact(bundle, {"schema_version", "evidence", "evidence_sha256"}, "pHash bundle")
    evidence = bundle["evidence"]
    if bundle["schema_version"] != "cvi.public_canine_phash_evidence_bundle.v1" or not isinstance(
        evidence, Mapping
    ) or content_sha256(evidence) != bundle["evidence_sha256"]:
        raise ValueError("pHash evidence bundle binding differs")
    candidates = evidence.get("candidates")
    if not isinstance(candidates, list) or evidence.get("candidate_count") != len(candidates):
        raise ValueError("pHash candidate cardinality differs")
    if evidence.get("fingerprint_count") != len(sample_by_opaque):
        raise ValueError("pHash fingerprint coverage differs")
    seen: set[tuple[str, str]] = set()
    for raw in candidates:
        if not isinstance(raw, Mapping):
            raise TypeError("pHash candidate must be an object")
        left, right = _mapped_pair(raw, sample_by_opaque)
        if (left, right) in seen:
            raise ValueError("duplicate pHash candidate pair")
        seen.add((left, right))
        token = content_sha256({
            "channel": "PHASH",
            "artifact_sha256": artifact_sha256,
            "candidate": raw,
        })
        pairs[(left, right)]["PHASH"] = token


def _add_pdq_candidates(
    pairs: dict[tuple[str, str], dict[str, str]],
    bundle: Mapping[str, Any],
    sample_by_opaque: Mapping[str, str],
    artifact_sha256: str,
) -> PDQSearchResult:
    _exact(bundle, {"schema_version", "evidence", "evidence_sha256"}, "PDQ bundle")
    evidence = bundle["evidence"]
    if bundle["schema_version"] != "cvi.public_canine_pdq_evidence_bundle.v1" or not isinstance(
        evidence, Mapping
    ) or content_sha256(evidence) != bundle["evidence_sha256"]:
        raise ValueError("PDQ evidence bundle binding differs")
    expected = {
        "schema_version",
        "search_result",
        "sample_ids_sha256",
        "fingerprint_count",
        "fingerprint_manifest_sha256",
        "source_spec_sha256",
        "source_receipt_bindings_sha256",
        "native_build_receipt_sha256",
        "native_binary_sha256",
        "official_regression_receipt_sha256",
        "policy",
        "policy_sha256",
        "decision",
        "interpretation",
    }
    _exact(evidence, expected, "PDQ corpus evidence")
    if evidence["schema_version"] != "cvi.public_canine_pdq_evidence.v1":
        raise ValueError("unsupported PDQ corpus evidence schema")
    for name in (
        "fingerprint_manifest_sha256",
        "source_spec_sha256",
        "source_receipt_bindings_sha256",
        "native_build_receipt_sha256",
        "native_binary_sha256",
        "official_regression_receipt_sha256",
        "policy_sha256",
    ):
        _sha256(evidence[name], name)
    if not isinstance(evidence["policy"], Mapping):
        raise TypeError("PDQ corpus policy must be an object")
    policy = PDQSearchPolicy.from_dict(evidence["policy"])
    search = PDQSearchResult.from_dict(evidence["search_result"])
    if (
        policy.policy_sha256 != evidence["policy_sha256"]
        or policy.distance_threshold != search.distance_threshold
        or policy.quality_threshold != search.quality_threshold
    ):
        raise ValueError("PDQ corpus policy binding differs")
    covered = tuple(sorted(
        search.eligible_sample_ids + search.ineligible_low_quality_sample_ids
    ))
    if (
        len(covered) != len(sample_by_opaque)
        or set(covered) != set(sample_by_opaque)
        or evidence["fingerprint_count"] != len(covered)
        or evidence["sample_ids_sha256"] != content_sha256(list(covered))
        or evidence["decision"] != "PASS_BOUNDED_LABEL_BLIND_PDQ_CANDIDATE_GENERATION"
    ):
        raise ValueError("PDQ corpus coverage differs")
    for item in search.candidates:
        left, right = sorted((
            sample_by_opaque[item.left_opaque_sample_id],
            sample_by_opaque[item.right_opaque_sample_id],
        ))
        token = content_sha256({
            "channel": "PDQ",
            "artifact_sha256": artifact_sha256,
            "candidate": item.to_dict(),
        })
        if "PDQ" in pairs[(left, right)]:
            raise ValueError("duplicate PDQ candidate pair")
        pairs[(left, right)]["PDQ"] = token
    return search


def _geometry_results(
    evidence: Sequence[GeometricVerifierEvidence],
    sample_by_opaque: Mapping[str, str],
) -> tuple[dict[tuple[str, str], Any], str | None]:
    if not evidence:
        return {}, None
    results: dict[tuple[str, str], Any] = {}
    digests: list[str] = []
    for chunk in evidence:
        digests.append(chunk.evidence_sha256)
        for item in chunk.results:
            try:
                pair = tuple(sorted((
                    sample_by_opaque[item.left_opaque_sample_id],
                    sample_by_opaque[item.right_opaque_sample_id],
                )))
            except KeyError as error:
                raise ValueError("geometric result references unknown sample") from error
            if pair in results:
                raise ValueError("duplicate geometric result across chunks")
            results[pair] = item
    return results, content_sha256({
        "schema_version": "cvi.geometric_evidence_set.v1",
        "evidence_sha256s": sorted(digests),
    })


def _geometry_admitted(
    receipt: Mapping[str, Any] | None,
    evidence: Sequence[GeometricVerifierEvidence],
) -> bool:
    if receipt is None:
        return False
    expected = {
        "schema_version",
        "policy_sha256",
        "calibration_receipt_sha256",
        "decision",
        "receipt_sha256",
    }
    _exact(receipt, expected, "geometric admission receipt")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt["schema_version"] != "cvi.geometric_policy_admission_receipt.v1"
        or content_sha256(unsigned) != receipt["receipt_sha256"]
        or receipt["decision"]
        != "ADMIT_GEOMETRIC_POLICY_FOR_PUBLIC_DUPLICATE_ADJUDICATION"
    ):
        raise ValueError("geometric admission receipt differs")
    policy_hashes = {item.policy.policy_sha256 for item in evidence}
    if policy_hashes != {receipt["policy_sha256"]}:
        raise ValueError("geometric admission policy binding differs")
    return True


def _review_results(
    bundle: Mapping[str, Any] | None, candidate_set_sha256: str
) -> tuple[dict[tuple[str, str], tuple[str, str, str]], str | None]:
    if bundle is None:
        return {}, None
    expected = {
        "schema_version",
        "candidate_set_sha256",
        "records",
        "review_protocol_sha256",
        "reviewer_attestation_sha256",
        "bundle_sha256",
    }
    _exact(bundle, expected, "review adjudication bundle")
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if (
        bundle["schema_version"] != "cvi.public_duplicate_review_bundle.v1"
        or bundle["candidate_set_sha256"] != candidate_set_sha256
        or content_sha256(unsigned) != bundle["bundle_sha256"]
    ):
        raise ValueError("review adjudication bundle binding differs")
    for name in ("review_protocol_sha256", "reviewer_attestation_sha256"):
        _sha256(bundle[name], name)
    if not isinstance(bundle["records"], list):
        raise TypeError("review records must be a JSON array")
    output: dict[tuple[str, str], tuple[str, str, str]] = {}
    for raw in bundle["records"]:
        fields = {
            "schema_version",
            "left_sample_token",
            "right_sample_token",
            "decision",
            "reason",
            "evidence_token",
        }
        _exact(raw, fields, "review record")
        pair = (raw["left_sample_token"], raw["right_sample_token"])
        _ordered_pair(*pair)
        if raw["schema_version"] != "cvi.public_duplicate_review_record.v1" or raw[
            "decision"
        ] not in {"REVIEW_CONFIRMED", "REVIEW_REJECTED", "REVIEW_UNRESOLVED"}:
            raise ValueError("review decision differs")
        expected_token = content_sha256({
            key: value for key, value in raw.items() if key != "evidence_token"
        })
        if raw["evidence_token"] != expected_token or pair in output:
            raise ValueError("review record token or uniqueness differs")
        output[pair] = (raw["decision"], raw["reason"], raw["evidence_token"])
    return output, content_sha256(bundle)


def _validate_opaque_binding(
    bundle: Mapping[str, Any], source_by_id: Mapping[str, Any]
) -> tuple[dict[str, str], str, str]:
    _exact(bundle, {"schema_version", "binding", "binding_sha256"}, "pHash binding bundle")
    binding = bundle["binding"]
    if bundle["schema_version"] != "cvi.public_canine_phash_binding_bundle.v1" or not isinstance(
        binding, Mapping
    ) or content_sha256(binding) != bundle["binding_sha256"]:
        raise ValueError("pHash binding bundle digest differs")
    rows = binding.get("bindings")
    if not isinstance(rows, list) or binding.get("binding_count") != len(rows):
        raise ValueError("pHash binding cardinality differs")
    output: dict[str, str] = {}
    sources: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "opaque_sample_id", "dataset_name", "source_sample_id"
        }:
            raise ValueError("pHash binding row fields differ")
        opaque = row["opaque_sample_id"]
        source_id = row["source_sample_id"]
        _sha256(opaque, "opaque sample ID")
        if opaque in output or source_id in sources or source_id not in source_by_id:
            raise ValueError("pHash binding coverage or uniqueness differs")
        if source_by_id[source_id].dataset_name != row["dataset_name"]:
            raise ValueError("pHash binding dataset differs")
        output[opaque] = source_id
        sources.add(source_id)
    return output, bundle["binding_sha256"], content_sha256(bundle)


def _validate_image_bundle(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "schema_version",
        "semantic_receipt_sha256",
        "policy",
        "policy_sha256",
        "receipt",
        "receipt_sha256",
        "tool_provenance",
        "tool_provenance_sha256",
    }
    _exact(bundle, expected, "image-content bundle")
    if bundle["schema_version"] != "cvi.image_content_audit_bundle.v1":
        raise ValueError("image-content bundle schema differs")
    for payload_name, digest_name in (
        ("policy", "policy_sha256"),
        ("receipt", "receipt_sha256"),
        ("tool_provenance", "tool_provenance_sha256"),
    ):
        if content_sha256(bundle[payload_name]) != bundle[digest_name]:
            raise ValueError(f"image-content {payload_name} digest differs")
    receipt = bundle["receipt"]
    if not isinstance(receipt, Mapping) or receipt.get("decision") != _IMAGE_DECISION or receipt.get(
        "interpretation"
    ) != _IMAGE_INTERPRETATION:
        raise ValueError("image-content receipt decision differs")
    records = receipt.get("records")
    groups = receipt.get("exact_duplicate_groups")
    if not isinstance(records, list) or not isinstance(groups, list):
        raise TypeError("image-content receipt collections differ")
    record_ids = [item.get("source_sample_id") for item in records]
    if record_ids != sorted(record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("image-content records must be sorted unique")
    pixels = {item["source_sample_id"]: item["pixel_sha256"] for item in records}
    expected_groups: list[dict[str, Any]] = []
    grouped: dict[str, list[str]] = defaultdict(list)
    for source_id, digest in pixels.items():
        _sha256(digest, "pixel SHA-256")
        grouped[digest].append(source_id)
    for digest, ids in sorted(grouped.items()):
        if len(ids) > 1:
            expected_groups.append({
                "schema_version": "cvi.pixel_exact_duplicate_group.v1",
                "pixel_sha256": digest,
                "source_sample_ids": sorted(ids),
            })
    if groups != expected_groups:
        raise ValueError("image-content exact groups differ from records")
    return receipt


def _validate_exact_graph_bundle(
    bundle: Mapping[str, Any], graph: ExactDuplicateGraph
) -> None:
    expected = {
        "schema_version",
        "graph",
        "graph_sha256",
        "tool_provenance",
        "tool_provenance_sha256",
        "bundle_sha256",
    }
    _exact(bundle, expected, "exact duplicate graph bundle")
    unsigned = dict(bundle)
    observed = unsigned.pop("bundle_sha256")
    if (
        bundle["schema_version"] != "cvi.exact_duplicate_graph_bundle.v2"
        or bundle["graph"] != graph.to_dict()
        or bundle["graph_sha256"] != graph.graph_sha256
        or content_sha256(bundle["tool_provenance"])
        != bundle["tool_provenance_sha256"]
        or content_sha256(unsigned) != observed
    ):
        raise ValueError("exact duplicate graph bundle binding differs")


def _dependency_edges(source: PublicSplitSourceBundle) -> list[PublicSplitEvidenceEdge]:
    by_source = {item.source_sample_id: item for item in source.samples}
    edges: list[PublicSplitEvidenceEdge] = []
    for item in source.samples:
        if item.paired_source_sample_id is None:
            continue
        try:
            paired = by_source[item.paired_source_sample_id]
        except KeyError as error:
            raise ValueError("dependency pair references unknown source sample") from error
        left, right = sorted((item.sample_token, paired.sample_token))
        edges.append(PublicSplitEvidenceEdge(
            left_sample_token=left,
            right_sample_token=right,
            relation=EvidenceRelation.DEPENDENCY,
            evidence_token=content_sha256({
                "relation": "DEPENDENCY",
                "left_sample_token": left,
                "right_sample_token": right,
            }),
        ))
    return edges


def _conservative_dependency_edges(
    source: PublicSplitSourceBundle, ledger: AdjudicationLedger
) -> list[PublicSplitEvidenceEdge]:
    if ledger.mode is not (
        AdjudicationMode.LEAKAGE_CONSERVATIVE_CANDIDATE_COMPONENT_CLOSURE
    ):
        raise ValueError("conservative dependency assembly requires explicit mode")
    evidence_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for edge in _dependency_edges(source):
        evidence_by_pair[(edge.left_sample_token, edge.right_sample_token)].append(
            edge.evidence_token
        )
    candidate_pairs: set[tuple[str, str]] = set()
    for item in ledger.records:
        if item.outcome is not CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY:
            raise ValueError("conservative ledger contains a nondependency outcome")
        pair = (item.left_sample_token, item.right_sample_token)
        candidate_pairs.add(pair)
        evidence_by_pair[pair].extend(item.candidate_evidence_tokens)
    if len(candidate_pairs) != len(ledger.records):
        raise ValueError("conservative ledger candidate coverage differs")
    ledger_sha256 = ledger.ledger_sha256
    edges = [
        PublicSplitEvidenceEdge(
            left_sample_token=pair[0],
            right_sample_token=pair[1],
            relation=EvidenceRelation.DEPENDENCY,
            evidence_token=content_sha256({
                "schema_version": "cvi.conservative_candidate_dependency.v1",
                "mode": ledger.mode.value,
                "ledger_sha256": ledger_sha256,
                "left_sample_token": pair[0],
                "right_sample_token": pair[1],
                "bound_evidence_tokens": sorted(set(tokens)),
                "interpretation": (
                    "LEAKAGE_COMPONENT_CLOSURE_ONLY_NOT_DUPLICATE_OR_"
                    "NONDUPLICATE_ADJUDICATION"
                ),
            }),
        )
        for pair, tokens in sorted(evidence_by_pair.items())
    ]
    observed = {
        (item.left_sample_token, item.right_sample_token) for item in edges
    }
    if not candidate_pairs <= observed:
        raise ValueError("conservative graph leaves candidate pairs unbound")
    return edges


def _pdq_admitted_dependency_edges(
    source: PublicSplitSourceBundle, ledger: AdjudicationLedger
) -> list[PublicSplitEvidenceEdge]:
    if ledger.mode is not AdjudicationMode.PDQ_COMPLETE_NEGATIVE_FILTER:
        raise ValueError("PDQ dependency assembly requires explicit mode")
    evidence_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for edge in _dependency_edges(source):
        evidence_by_pair[(edge.left_sample_token, edge.right_sample_token)].append(
            edge.evidence_token
        )
    for item in ledger.records:
        pair = (item.left_sample_token, item.right_sample_token)
        if item.outcome is CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY:
            if not ({"EXACT", "PDQ"} & set(item.candidate_channels)):
                raise ValueError("PDQ-filtered dependency lacks exact or PDQ evidence")
            evidence_by_pair[pair].extend(item.candidate_evidence_tokens)
        elif item.outcome is CandidateOutcome.PHASH_ONLY_REJECTED_BY_ADMITTED_PDQ:
            if item.candidate_channels != ("PHASH",) or item.reason != "PDQ_COMPLETE_NEGATIVE":
                raise ValueError("PDQ complete-negative outcome differs")
        else:
            raise ValueError("PDQ-filtered ledger contains an unsupported outcome")
    ledger_sha256 = ledger.ledger_sha256
    return [
        PublicSplitEvidenceEdge(
            left_sample_token=pair[0],
            right_sample_token=pair[1],
            relation=EvidenceRelation.DEPENDENCY,
            evidence_token=content_sha256({
                "schema_version": "cvi.pdq_admitted_candidate_dependency.v1",
                "mode": ledger.mode.value,
                "ledger_sha256": ledger_sha256,
                "left_sample_token": pair[0],
                "right_sample_token": pair[1],
                "bound_evidence_tokens": sorted(set(tokens)),
                "interpretation": (
                    "LEAKAGE_COMPONENT_CLOSURE_ONLY_NOT_DUPLICATE_OR_NONMATCH"
                ),
            }),
        )
        for pair, tokens in sorted(evidence_by_pair.items())
    ]


def _dinov2_admitted_dependency_edges(
    source: PublicSplitSourceBundle, ledger: AdjudicationLedger
) -> list[PublicSplitEvidenceEdge]:
    if ledger.mode is not AdjudicationMode.DINOV2_TRANSFORM_FAMILY_FILTER:
        raise ValueError("DINOv2 dependency assembly requires explicit mode")
    evidence_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for edge in _dependency_edges(source):
        evidence_by_pair[(edge.left_sample_token, edge.right_sample_token)].append(
            edge.evidence_token
        )
    for item in ledger.records:
        pair = (item.left_sample_token, item.right_sample_token)
        if item.outcome is CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY:
            evidence_by_pair[pair].extend(
                item.candidate_evidence_tokens + item.decision_evidence_tokens
            )
        elif item.outcome is CandidateOutcome.PHASH_ONLY_REJECTED_BY_ADMITTED_DINOV2:
            if (
                item.candidate_channels != ("PHASH",)
                or item.reason != "BELOW_ADMITTED_DINOV2_THRESHOLD"
            ):
                raise ValueError("DINOv2 leakage-filter rejection outcome differs")
        else:
            raise ValueError("DINOv2-filtered ledger contains an unsupported outcome")
    ledger_sha256 = ledger.ledger_sha256
    return [
        PublicSplitEvidenceEdge(
            left_sample_token=pair[0],
            right_sample_token=pair[1],
            relation=EvidenceRelation.DEPENDENCY,
            evidence_token=content_sha256({
                "schema_version": "cvi.dinov2_admitted_candidate_dependency.v1",
                "mode": ledger.mode.value,
                "ledger_sha256": ledger_sha256,
                "left_sample_token": pair[0],
                "right_sample_token": pair[1],
                "bound_evidence_tokens": sorted(set(tokens)),
                "interpretation": (
                    "LEAKAGE_COMPONENT_CLOSURE_ONLY_NOT_DUPLICATE_OR_NONMATCH"
                ),
            }),
        )
        for pair, tokens in sorted(evidence_by_pair.items())
    ]


def _mapped_pair(
    raw: Mapping[str, Any], sample_by_opaque: Mapping[str, str]
) -> tuple[str, str]:
    try:
        left = sample_by_opaque[raw["left_opaque_sample_id"]]
        right = sample_by_opaque[raw["right_opaque_sample_id"]]
    except (KeyError, TypeError) as error:
        raise ValueError("candidate references unknown opaque sample") from error
    return tuple(sorted((left, right)))


def _bundle(
    schema: str,
    payload_name: str,
    payload: dict[str, Any],
    digest_name: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("tool provenance must be a nonempty object")
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
    _exact(bundle, expected, "protected adjudication bundle")
    unsigned = dict(bundle)
    observed = unsigned.pop("bundle_sha256")
    if bundle["schema_version"] != schema or content_sha256(unsigned) != observed:
        raise ValueError("protected adjudication bundle digest differs")
    if content_sha256(bundle["tool_provenance"]) != bundle[
        "tool_provenance_sha256"
    ]:
        raise ValueError("protected adjudication provenance differs")
    payload = bundle[payload_name]
    if not isinstance(payload, Mapping) or content_sha256(payload) != bundle[digest_name]:
        raise ValueError("protected adjudication payload digest differs")
    return payload


def _ordered_pair(left: str, right: str) -> None:
    _sha256(left, "left sample token")
    _sha256(right, "right sample token")
    if left >= right:
        raise ValueError("candidate endpoints must be distinct and sorted")


def _sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be lowercase SHA-256") from error
    if value != value.lower():
        raise ValueError(f"{name} must be lowercase SHA-256")


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _canonical_tokens(
    values: tuple[str, ...], name: str, *, digest: bool = True, allow_empty: bool = False
) -> None:
    if not isinstance(values, tuple) or (not values and not allow_empty):
        raise ValueError(f"{name} must be a nonempty tuple")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be sorted and unique")
    for value in values:
        if digest:
            _sha256(value, name)
        elif not isinstance(value, str) or not value or len(value) > 64:
            raise ValueError(f"{name} contains invalid text")


def _canonical_blockers(values: tuple[str, ...]) -> None:
    if values != tuple(sorted(set(values))) or any(
        not isinstance(item, str) or not item for item in values
    ):
        raise ValueError("global blockers must be sorted unique text")


def _binding_rows(values: tuple[tuple[str, str], ...]) -> None:
    if not values or values != tuple(sorted(values)) or len(dict(values)) != len(values):
        raise ValueError("evidence bindings must be sorted unique and nonempty")
    for name, digest in values:
        if not isinstance(name, str) or not name:
            raise ValueError("evidence binding name differs")
        _sha256(digest, "evidence binding digest")


def _pairs(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise TypeError("JSON pair collection must be an array")
    rows: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError("JSON pair row must contain two values")
        rows.append((item[0], item[1]))
    return tuple(rows)


def _exact(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"{name} fields differ")
