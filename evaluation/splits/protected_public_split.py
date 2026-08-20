"""Protected, score-blind public canine split construction.

The module consumes a frozen semantic source bundle and a fully adjudicated
duplicate/dependency/review graph.  It produces two deliberately separate
artifacts: an opaque assignment for model/scorer code and a private evaluator
binding containing source labels.  The split secret is never serialized into
either artifact or its receipt.

This is split infrastructure, not a frozen split.  In particular, importing
this module or passing its unit tests does not admit a real evidence graph.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from shared.foundation.binomial import required_zero_event_trials
from shared.foundation.provenance import content_sha256
from evaluation.splits.split_role_exposure import (
    ExposureStage,
    RoleExposureLedger,
    RoleExposureReceipt,
    role_allows_historical_stage,
    validate_split_candidate_assignment,
    verify_split_role_exposure_inputs,
)

_HEX_SHA256_LENGTH = 64
_REQUIRED_EVIDENCE_BINDINGS = (
    "exact_duplicate_graph_sha256",
    "geometric_verifier_sha256",
    "image_content_receipts_sha256",
    "pdq_candidates_sha256",
    "phash_candidates_sha256",
    "review_adjudication_sha256",
    "semantic_receipts_sha256",
)


class EvidenceRelation(StrEnum):
    EXACT_CONFIRMED = "EXACT_CONFIRMED"
    GEOMETRIC_CONFIRMED = "GEOMETRIC_CONFIRMED"
    DEPENDENCY = "DEPENDENCY"
    REVIEW_CONFIRMED = "REVIEW_CONFIRMED"
    REVIEW_REJECTED = "REVIEW_REJECTED"
    REVIEW_UNRESOLVED = "REVIEW_UNRESOLVED"


@dataclass(frozen=True, slots=True)
class PublicSplitSample:
    sample_token: str
    identity_token: str
    sequence_token: str
    source_sample_id: str
    dataset_identity_id: str
    dataset_name: str
    source_variant: str
    original_split: str | None
    raw_frame_index: int
    paired_source_sample_id: str | None
    in_no_mono_subset: bool | None
    region: str
    schema_version: str = "cvi.public_split_sample.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_split_sample.v1":
            raise ValueError("unsupported public split sample schema")
        for name in ("sample_token", "identity_token", "sequence_token"):
            _sha256(getattr(self, name), name)
        for name in (
            "source_sample_id",
            "dataset_identity_id",
            "dataset_name",
            "source_variant",
            "region",
        ):
            _text(getattr(self, name), name, 2048)
        if self.original_split is not None:
            _text(self.original_split, "original_split", 64)
        if self.paired_source_sample_id is not None:
            _text(self.paired_source_sample_id, "paired_source_sample_id", 2048)
        if isinstance(self.raw_frame_index, bool) or not isinstance(
            self.raw_frame_index, int
        ) or self.raw_frame_index < 0:
            raise ValueError("raw_frame_index must be a nonnegative integer")
        if self.in_no_mono_subset is not None and not isinstance(
            self.in_no_mono_subset, bool
        ):
            raise TypeError("in_no_mono_subset must be boolean or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublicSplitSample:
        _exact_keys(payload, set(cls.__dataclass_fields__), "public split sample")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PublicSplitSourceBundle:
    evidence_bindings: tuple[tuple[str, str], ...]
    samples: tuple[PublicSplitSample, ...]
    interpretation: str = "SEMANTIC_LABEL_BINDING_ONLY_NOT_MODEL_INPUT"
    schema_version: str = "cvi.public_split_source_bundle.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_split_source_bundle.v1":
            raise ValueError("unsupported public split source bundle schema")
        if self.interpretation != "SEMANTIC_LABEL_BINDING_ONLY_NOT_MODEL_INPUT":
            raise ValueError("public split source interpretation differs")
        _bindings(self.evidence_bindings, require_all=True)
        if not self.samples:
            raise ValueError("public split source bundle must not be empty")
        sample_tokens: set[str] = set()
        source_ids: set[str] = set()
        identity_by_token: dict[str, str] = {}
        token_by_identity: dict[str, str] = {}
        sequence_by_token: dict[str, tuple[str, str]] = {}
        for sample in self.samples:
            if not isinstance(sample, PublicSplitSample):
                raise TypeError("source bundle samples must be PublicSplitSample")
            if sample.sample_token in sample_tokens:
                raise ValueError("duplicate opaque sample token")
            if sample.source_sample_id in source_ids:
                raise ValueError("duplicate source sample ID")
            allowed_variants = (
                {"original", "random_background"}
                if sample.dataset_name == "yt-bb-dog"
                else {"original"}
            )
            if sample.dataset_name not in {
                "yt-bb-dog",
                "dogfacenet224",
                "mpdd",
                "sibetan",
            } or sample.source_variant not in allowed_variants:
                raise ValueError("unsupported public dataset or source variant")
            sample_tokens.add(sample.sample_token)
            source_ids.add(sample.source_sample_id)
            prior_identity = identity_by_token.setdefault(
                sample.identity_token, sample.dataset_identity_id
            )
            if prior_identity != sample.dataset_identity_id:
                raise ValueError("opaque identity token aliases multiple labels")
            prior_token = token_by_identity.setdefault(
                sample.dataset_identity_id, sample.identity_token
            )
            if prior_token != sample.identity_token:
                raise ValueError("one identity label maps to multiple opaque tokens")
            prior_sequence = sequence_by_token.setdefault(
                sample.sequence_token,
                (sample.dataset_name, sample.dataset_identity_id),
            )
            if prior_sequence != (sample.dataset_name, sample.dataset_identity_id):
                raise ValueError("opaque sequence token crosses identity or dataset")

    @property
    def bundle_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_bindings": [list(item) for item in self.evidence_bindings],
            "samples": [
                sample.to_dict()
                for sample in sorted(
                    self.samples, key=lambda item: item.sample_token
                )
            ],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublicSplitSourceBundle:
        expected = {"schema_version", "evidence_bindings", "samples", "interpretation"}
        _exact_keys(payload, expected, "public split source bundle")
        if not isinstance(payload["evidence_bindings"], list) or not isinstance(
            payload["samples"], list
        ):
            raise TypeError("source bundle collections must be lists")
        return cls(
            evidence_bindings=_pair_tuple(payload["evidence_bindings"]),
            samples=tuple(PublicSplitSample.from_dict(item) for item in payload["samples"]),
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class PublicSplitEvidenceEdge:
    left_sample_token: str
    right_sample_token: str
    relation: EvidenceRelation
    evidence_token: str
    schema_version: str = "cvi.public_split_evidence_edge.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_split_evidence_edge.v1":
            raise ValueError("unsupported split evidence edge schema")
        for name in ("left_sample_token", "right_sample_token", "evidence_token"):
            _sha256(getattr(self, name), name)
        if self.left_sample_token >= self.right_sample_token:
            raise ValueError("edge endpoints must be distinct and sorted")
        if not isinstance(self.relation, EvidenceRelation):
            raise TypeError("edge relation must be EvidenceRelation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "left_sample_token": self.left_sample_token,
            "right_sample_token": self.right_sample_token,
            "relation": self.relation.value,
            "evidence_token": self.evidence_token,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublicSplitEvidenceEdge:
        _exact_keys(payload, set(cls.__dataclass_fields__), "split evidence edge")
        values = dict(payload)
        values["relation"] = EvidenceRelation(values["relation"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class FrozenPublicSplitEvidenceGraph:
    evidence_bindings: tuple[tuple[str, str], ...]
    edges: tuple[PublicSplitEvidenceEdge, ...]
    adjudication_state: str = "FROZEN_DUPLICATE_REVIEW_DEPENDENCY_GRAPH"
    schema_version: str = "cvi.frozen_public_split_evidence_graph.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.frozen_public_split_evidence_graph.v1":
            raise ValueError("unsupported frozen split evidence graph schema")
        if self.adjudication_state != "FROZEN_DUPLICATE_REVIEW_DEPENDENCY_GRAPH":
            raise ValueError("split evidence graph is not frozen")
        _bindings(self.evidence_bindings, require_all=True)
        keys: set[tuple[str, str, EvidenceRelation]] = set()
        nondependency_relations: dict[
            tuple[str, str], set[EvidenceRelation]
        ] = defaultdict(set)
        for edge in self.edges:
            if not isinstance(edge, PublicSplitEvidenceEdge):
                raise TypeError("graph edges must be PublicSplitEvidenceEdge")
            key = (
                edge.left_sample_token,
                edge.right_sample_token,
                edge.relation,
            )
            if key in keys:
                raise ValueError("duplicate relation for one sample pair")
            keys.add(key)
            if edge.relation is not EvidenceRelation.DEPENDENCY:
                pair = (edge.left_sample_token, edge.right_sample_token)
                nondependency_relations[pair].add(edge.relation)
                if len(nondependency_relations[pair]) > 1:
                    raise ValueError(
                        "sample pair has contradictory duplicate adjudications"
                    )
        if tuple(sorted(self.edges, key=_edge_key)) != self.edges:
            raise ValueError("graph edges must be canonically sorted")

    @property
    def graph_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_bindings": [list(item) for item in self.evidence_bindings],
            "edges": [edge.to_dict() for edge in self.edges],
            "adjudication_state": self.adjudication_state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrozenPublicSplitEvidenceGraph:
        expected = {"schema_version", "evidence_bindings", "edges", "adjudication_state"}
        _exact_keys(payload, expected, "frozen split evidence graph")
        if not isinstance(payload["evidence_bindings"], list) or not isinstance(
            payload["edges"], list
        ):
            raise TypeError("evidence graph collections must be lists")
        return cls(
            evidence_bindings=_pair_tuple(payload["evidence_bindings"]),
            edges=tuple(PublicSplitEvidenceEdge.from_dict(item) for item in payload["edges"]),
            adjudication_state=payload["adjudication_state"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ProtectedPublicSplitPolicy:
    capacity_mode: str = "EVIDENCE_CONSTRAINED_MAXIMAL_COVERAGE"
    yt_official_train_identities: int = 2000
    yt_official_test_identities: int = 723
    yt_requested_fit_identities: int = 1200
    yt_minimum_fit_identities: int = 1000
    yt_development_identities: int = 200
    yt_development_episode_identities: int = 100
    yt_calibration_known_identities: int = 300
    yt_calibration_unknown_identities: int = 300
    yt_calibration_gallery_sizes: tuple[int, ...] = (39, 64, 100, 300)
    yt_minimum_eligible_train_identities: int = 1800
    yt_test_known_identities: int = 300
    yt_test_unknown_identities: int = 423
    yt_test_unknown_target_fpir: float = 0.01
    yt_test_unknown_reporting_fpir: float = 0.001
    yt_test_unknown_confidence_level: float = 0.95
    yt_test_unknown_minimum_identities: int = required_zero_event_trials(
        0.01, confidence_level=0.95
    )
    dogface_train_identities: int = 1254
    dogface_test_identities: int = 139
    dogface_fit_identities: int = 1004
    dogface_minimum_fit_identities: int = 1000
    dogface_development_identities: int = 125
    dogface_calibration_identities: int = 125
    dogface_minimum_test_identities: int = 125
    mpdd_train_identities: int = 95
    mpdd_test_identities: int = 96
    mpdd_open_known_identities: int = 64
    mpdd_open_unknown_identities: int = 32
    sibetan_identities: int = 59
    sibetan_cross_sequence_identities: int = 39
    sibetan_unknown_identities: int = 20
    yt_guard_components: int = 1
    yt_minimum_component_gap: int = 2
    yt_minimum_raw_frame_gap: int = 2
    shot_counts: tuple[int, ...] = (1, 3)
    diagnostic_shot_counts: tuple[int, ...] = (5,)
    yt_primary_open_set_gallery_size: int = 300
    yt_primary_open_set_shot: int = 3
    mpdd_external_gallery_size: int = 64
    sibetan_external_gallery_size: int = 39
    external_result_boundary: str = "STRICT_EXTERNAL_DOMAIN_ZERO_SHOT"
    threshold_selection: str = "YT_CALIBRATION_ONLY_EXACT_ORDER_STATISTIC"
    development_reuse_policy: str = "PROHIBITED_AFTER_MARGIN_SELECTION"
    bootstrap_draws: int = 10_000
    schema_version: str = "cvi.protected_public_split_policy.v3"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.protected_public_split_policy.v3":
            raise ValueError("unsupported protected public split policy")
        expected = ProtectedPublicSplitPolicy.__dataclass_fields__
        for name in expected:
            if name in {
                "shot_counts",
                "diagnostic_shot_counts",
                "yt_calibration_gallery_sizes",
                "capacity_mode",
                "external_result_boundary",
                "threshold_selection",
                "development_reuse_policy",
                "yt_test_unknown_target_fpir",
                "yt_test_unknown_reporting_fpir",
                "yt_test_unknown_confidence_level",
                "schema_version",
            }:
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.shot_counts != (1, 3):
            raise ValueError("primary shot counts are fixed to nested 1/3")
        if self.diagnostic_shot_counts != (5,):
            raise ValueError("diagnostic shot counts are fixed to 5")
        if set(self.shot_counts) & set(self.diagnostic_shot_counts) or tuple(
            sorted((*self.shot_counts, *self.diagnostic_shot_counts))
        ) != (1, 3, 5):
            raise ValueError("primary and diagnostic shot partitions differ")
        if self.capacity_mode != "EVIDENCE_CONSTRAINED_MAXIMAL_COVERAGE":
            raise ValueError("protected split capacity mode differs")
        if self.yt_calibration_gallery_sizes != (39, 64, 100, 300):
            raise ValueError("YT calibration gallery sizes are fixed")
        if self.external_result_boundary != "STRICT_EXTERNAL_DOMAIN_ZERO_SHOT":
            raise ValueError("external result boundary differs")
        if self.threshold_selection != "YT_CALIBRATION_ONLY_EXACT_ORDER_STATISTIC":
            raise ValueError("threshold selection boundary differs")
        if self.yt_test_known_identities + self.yt_test_unknown_identities != 723:
            raise ValueError("YT test role counts must preserve all 723 identities")
        if self.yt_test_unknown_target_fpir != 0.01:
            raise ValueError("YT test unknown target FPIR differs")
        if self.yt_test_unknown_reporting_fpir != 0.001:
            raise ValueError("YT test unknown reporting FPIR differs")
        if self.yt_test_unknown_confidence_level != 0.95:
            raise ValueError("YT test unknown confidence level differs")
        if self.yt_test_unknown_minimum_identities != required_zero_event_trials(
            self.yt_test_unknown_target_fpir,
            confidence_level=self.yt_test_unknown_confidence_level,
        ):
            raise ValueError("YT test unknown statistical floor differs")
        if self.yt_primary_open_set_gallery_size != self.yt_test_known_identities:
            raise ValueError("YT primary open-set N differs from known identities")
        if self.yt_primary_open_set_shot != 3:
            raise ValueError("YT primary open-set shot differs")
        if self.mpdd_external_gallery_size != self.mpdd_open_known_identities:
            raise ValueError("MPDD external open-set N differs")
        if self.sibetan_external_gallery_size != self.sibetan_cross_sequence_identities:
            raise ValueError("Sibetan external open-set N differs")
        if self.development_reuse_policy != "PROHIBITED_AFTER_MARGIN_SELECTION":
            raise ValueError("development reuse policy differs")
        if self.yt_minimum_fit_identities + 800 != self.yt_minimum_eligible_train_identities:
            raise ValueError("YT minimum capacity arithmetic differs")
        if (
            self.yt_development_episode_identities * 2
            != self.yt_development_identities
        ):
            raise ValueError("YT development A/B episode arithmetic differs")
        if self.yt_calibration_gallery_sizes[-1] != (
            self.yt_calibration_known_identities
        ):
            raise ValueError("YT calibration maximum N differs from known bank")
        if self.dogface_fit_identities + self.dogface_development_identities + self.dogface_calibration_identities != self.dogface_train_identities:
            raise ValueError("DogFace role counts differ")
        if self.dogface_minimum_fit_identities != 1000:
            raise ValueError("DogFace fit statistical floor differs")
        if self.dogface_minimum_test_identities != 125:
            raise ValueError("DogFace test statistical floor differs")
        if self.mpdd_open_known_identities + self.mpdd_open_unknown_identities != self.mpdd_test_identities:
            raise ValueError("MPDD derivative counts differ")
        if self.sibetan_cross_sequence_identities + self.sibetan_unknown_identities != self.sibetan_identities:
            raise ValueError("Sibetan role counts differ")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        value = {name: getattr(self, name) for name in self.__dataclass_fields__}
        value["shot_counts"] = list(self.shot_counts)
        value["diagnostic_shot_counts"] = list(self.diagnostic_shot_counts)
        value["yt_calibration_gallery_sizes"] = list(
            self.yt_calibration_gallery_sizes
        )
        return value

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProtectedPublicSplitPolicy:
        _exact_keys(payload, set(cls.__dataclass_fields__), "protected split policy")
        values = dict(payload)
        if not isinstance(values["shot_counts"], list):
            raise TypeError("shot_counts must be a list")
        values["shot_counts"] = tuple(values["shot_counts"])
        if not isinstance(values["diagnostic_shot_counts"], list):
            raise TypeError("diagnostic_shot_counts must be a list")
        values["diagnostic_shot_counts"] = tuple(values["diagnostic_shot_counts"])
        if not isinstance(values["yt_calibration_gallery_sizes"], list):
            raise TypeError("yt_calibration_gallery_sizes must be a list")
        values["yt_calibration_gallery_sizes"] = tuple(
            values["yt_calibration_gallery_sizes"]
        )
        candidate = cls(**values)
        if candidate != cls():
            raise ValueError("protected public split policy constants differ")
        return candidate


@dataclass(frozen=True, slots=True)
class _Component:
    token: str
    samples: tuple[PublicSplitSample, ...]


@dataclass(frozen=True, slots=True)
class _AllocationBlock:
    token: str
    identity_tokens: tuple[str, ...]
    component_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublicSplitBuildResult:
    status: str
    assignment: dict[str, Any]
    evaluator_binding: dict[str, Any]
    receipt: dict[str, Any]


class _DisjointSet:
    def __init__(self, tokens: Iterable[str]) -> None:
        self.parent = {token: token for token in tokens}

    def find(self, token: str) -> str:
        root = token
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[token] != token:
            parent = self.parent[token]
            self.parent[token] = root
            token = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def validate_protected_split_output_paths(paths: tuple[Path, ...]) -> None:
    """Fail before seed creation or split work if publication is unsafe."""

    if len(paths) != 3 or len(set(paths)) != 3:
        raise ValueError("protected split requires three distinct JSON outputs")
    resolved: list[Path] = []
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite split output: {path}")
        parent = path.parent.resolve(strict=True)
        if not parent.is_dir():
            raise NotADirectoryError(parent)
        resolved.append(parent / path.name)
    if len({path.parent for path in resolved}) != 1:
        raise ValueError("all protected split JSON outputs must share one directory")
    if len(set(resolved)) != 3:
        raise ValueError("resolved protected split outputs must be distinct")


def seed_commitment(secret: bytes) -> str:
    _secret(secret)
    return hashlib.sha256(b"CVI_PROTECTED_SPLIT_SEED_V1\0" + secret).hexdigest()


def create_split_secret(path: Path) -> bytes:
    """Create a no-replace 32-byte seed with mode 0600."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("protected split secret creation requires O_NOFOLLOW")
    absolute = Path(os.path.abspath(os.fspath(path)))
    parent = absolute.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(parent / absolute.name, flags, 0o600)
    secret = os.urandom(32)
    try:
        offset = 0
        while offset < len(secret):
            written = os.write(descriptor, secret[offset:])
            if written <= 0:
                raise OSError("short write while creating protected split secret")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
    except BaseException:
        os.close(descriptor)
        (parent / absolute.name).unlink(missing_ok=True)
        raise
    os.close(descriptor)
    return secret


def read_split_secret(path: Path) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("protected split secret reading requires O_NOFOLLOW")
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(absolute, flags)
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError("split secret must be a regular file")
        if stat.S_IMODE(initial.st_mode) & 0o077:
            raise PermissionError("split secret must not grant group/other access")
        secret = os.read(descriptor, 33)
        final = os.fstat(descriptor)
        named = os.stat(absolute, follow_symlinks=False)
        if _stat_identity(initial) != _stat_identity(final) or (
            named.st_dev,
            named.st_ino,
        ) != (initial.st_dev, initial.st_ino):
            raise RuntimeError("split secret changed while reading")
    finally:
        os.close(descriptor)
    _secret(secret)
    return secret


def build_protected_public_split(
    *,
    source: PublicSplitSourceBundle,
    graph: FrozenPublicSplitEvidenceGraph,
    policy: ProtectedPublicSplitPolicy,
    secret: bytes,
    input_file_sha256s: tuple[tuple[str, str], ...],
    tool_provenance: dict[str, Any],
    role_exposure_ledger: RoleExposureLedger,
    role_exposure_receipt: RoleExposureReceipt,
) -> PublicSplitBuildResult:
    """Build opaque roles from frozen evidence without any score input."""

    _secret(secret)
    _bindings(input_file_sha256s, require_all=False)
    if source.evidence_bindings != graph.evidence_bindings:
        raise ValueError("source and graph evidence bindings differ")
    if policy != ProtectedPublicSplitPolicy():
        raise ValueError("protected split policy is not the fixed protocol")
    if not isinstance(tool_provenance, dict) or not tool_provenance:
        raise ValueError("tool provenance must be a non-empty object")
    historical_exposure = verify_split_role_exposure_inputs(
        source.samples,
        role_exposure_ledger,
        role_exposure_receipt,
    )
    samples_by_token = {sample.sample_token: sample for sample in source.samples}
    source_id_to_token = {
        sample.source_sample_id: sample.sample_token for sample in source.samples
    }
    _validate_graph_references(graph, samples_by_token)
    _validate_dependency_edges(source.samples, graph.edges, source_id_to_token)
    _validate_official_identity_cardinalities(source.samples, policy)

    components, quarantined_components, quarantine_reasons = _close_components(
        source.samples, graph.edges
    )
    component_by_sample = {
        sample.sample_token: component
        for component in components
        for sample in component.samples
    }
    allocation_blocks, block_by_identity = _build_allocation_blocks(components)
    _quarantine_official_lane_conflicts(allocation_blocks, components, quarantine_reasons)
    _propagate_component_quarantine(
        allocation_blocks, quarantine_reasons
    )
    quarantined_components = set(quarantine_reasons)
    quarantined_identities = {
        identity
        for block in allocation_blocks
        if set(block.component_tokens) & quarantined_components
        for identity in block.identity_tokens
    }
    evidence_root = content_sha256({
        "source_bundle_sha256": source.bundle_sha256,
        "graph_sha256": graph.graph_sha256,
        "policy_sha256": policy.policy_sha256,
        "evidence_bindings": [list(item) for item in source.evidence_bindings],
        "input_file_sha256s": [list(item) for item in input_file_sha256s],
    })
    keys = _derive_keys(secret, evidence_root)
    roles, capacity = _assign_identity_roles(
        source.samples,
        component_by_sample,
        quarantined_identities,
        policy,
        keys["identity_roles"],
        keys["frame_roles"],
        historical_exposure,
        block_by_identity,
        quarantine_reasons,
    )
    _propagate_component_quarantine(allocation_blocks, quarantine_reasons)
    quarantined_components = set(quarantine_reasons)
    quarantined_identities = {
        identity
        for block in allocation_blocks
        if set(block.component_tokens) & quarantined_components
        for identity in block.identity_tokens
    }
    status = capacity["status"]
    if status == "PASS_PROTECTED_SPLIT_CONSTRUCTION":
        uses, evidence_capacity = _build_protocol_uses(
            source.samples,
            component_by_sample,
            quarantined_components,
            roles,
            policy,
            keys,
        )
        capacity["protocol_evidence_capacity"] = evidence_capacity
        if evidence_capacity["status"] != "PASS_PROTOCOL_EVIDENCE_CAPACITY":
            status = evidence_capacity["status"]
            capacity["status"] = status
            roles = {}
            uses = defaultdict(list)
    else:
        roles = {}
        uses = defaultdict(list)

    assignment_records: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []
    for sample in sorted(source.samples, key=lambda item: item.sample_token):
        if sample.identity_token not in roles:
            continue
        component = component_by_sample[sample.sample_token]
        random_control = sample.source_variant == "random_background"
        assignment_records.append({
            "sample_token": sample.sample_token,
            "identity_token": sample.identity_token,
            "component_token": component.token,
            "dataset_name": sample.dataset_name,
            "source_variant": sample.source_variant,
            "identity_role": roles[sample.identity_token],
            "model_access": _model_access(sample.dataset_name, roles[sample.identity_token]),
            "sample_disposition": (
                "PAIRED_CONTROL_ONLY" if random_control else "PRIMARY_ORACLE_CROP"
            ),
            "paired_original_token": (
                source_id_to_token[sample.paired_source_sample_id]
                if random_control and sample.paired_source_sample_id is not None
                else None
            ),
            "uses": sorted(uses[sample.sample_token], key=_use_key),
        })
        label_records.append({
            "sample_token": sample.sample_token,
            "identity_token": sample.identity_token,
            "source_sample_id": sample.source_sample_id,
            "dataset_identity_id": sample.dataset_identity_id,
            "sequence_token": sample.sequence_token,
            "raw_frame_index": sample.raw_frame_index,
            "original_split": sample.original_split,
            "region": sample.region,
        })

    cohort_summary = _protocol_cohort_summary(assignment_records)
    assignment = {
        "schema_version": "cvi.protected_public_split_assignment.v1",
        "status": status,
        "seed_commitment": seed_commitment(secret),
        "evidence_root_sha256": evidence_root,
        "policy_sha256": policy.policy_sha256,
        "strict_external_boundary": policy.external_result_boundary,
        "score_inputs_used": False,
        "label_fields_present": False,
        "capacity": capacity,
        "protocol_cohorts": cohort_summary,
        "records": assignment_records,
        "interpretation": "OPAQUE_ROLE_ASSIGNMENT_ONLY_NOT_MODEL_OR_ACCURACY_EVIDENCE",
    }
    _assert_assignment_is_label_free(assignment)
    evaluator_binding = {
        "schema_version": "cvi.protected_public_split_evaluator_binding.v1",
        "status": status,
        "seed_commitment": seed_commitment(secret),
        "evidence_root_sha256": evidence_root,
        "records": label_records,
        "interpretation": "PRIVATE_LABEL_JOIN_FOR_SEALED_EVALUATION_ONLY",
    }
    assignment_sha = content_sha256(assignment)
    binding_sha = content_sha256(evaluator_binding)
    if status == "PASS_PROTECTED_SPLIT_CONSTRUCTION":
        validate_split_candidate_assignment(
            source_samples=source.samples,
            assignment=assignment,
            ledger=role_exposure_ledger,
        )
    receipt_payload = {
        "schema_version": "cvi.protected_public_split_receipt.v3",
        "status": status,
        "seed_commitment": seed_commitment(secret),
        "evidence_root_sha256": evidence_root,
        "source_bundle_sha256": source.bundle_sha256,
        "graph_sha256": graph.graph_sha256,
        "policy_sha256": policy.policy_sha256,
        "evidence_bindings": [list(item) for item in source.evidence_bindings],
        "input_file_sha256s": [list(item) for item in input_file_sha256s],
        "assignment_sha256": assignment_sha,
        "evaluator_binding_sha256": binding_sha,
        "role_exposure_ledger_sha256": role_exposure_ledger.ledger_sha256,
        "role_exposure_receipt_sha256": role_exposure_receipt.receipt_sha256,
        "capacity_mode": policy.capacity_mode,
        "requested_role_counts": capacity.get("requested_role_counts", {}),
        "actual_role_counts": capacity.get("actual_role_counts", {}),
        "quarantined_identity_counts_by_lane": capacity.get(
            "quarantined_identity_counts_by_lane", {}
        ),
        "yt_test_unknown_fpir_power": capacity.get(
            "yt_test_unknown_fpir_power", {}
        ),
        "capacity": capacity,
        "protocol_cohorts": cohort_summary,
        "quarantine": {
            "component_count": len(quarantined_components),
            "identity_count": len(quarantined_identities),
            "sample_count": sum(
                len(component.samples)
                for component in components
                if component.token in quarantined_components
            ),
            "allocation_block_count": sum(
                bool(set(block.component_tokens) & quarantined_components)
                for block in allocation_blocks
            ),
            "total_component_count": len(components),
            "total_allocation_block_count": len(allocation_blocks),
            "largest_quarantined_block_identity_count": max(
                (
                    len(block.identity_tokens)
                    for block in allocation_blocks
                    if set(block.component_tokens) & quarantined_components
                ),
                default=0,
            ),
            "reason_counts": sorted(
                (reason, sum(reason in values for values in quarantine_reasons.values()))
                for reason in {item for values in quarantine_reasons.values() for item in values}
            ),
        },
        "tool_provenance": tool_provenance,
        "tool_provenance_sha256": content_sha256(tool_provenance),
        "interpretation": "SPLIT_CONTRACT_BEHAVIOR_ONLY_NOT_PERFORMANCE_OR_DATA_ADMISSION",
    }
    receipt_payload["receipt_sha256"] = content_sha256(receipt_payload)
    return PublicSplitBuildResult(status, assignment, evaluator_binding, receipt_payload)


def _validate_graph_references(
    graph: FrozenPublicSplitEvidenceGraph,
    samples: dict[str, PublicSplitSample],
) -> None:
    for edge in graph.edges:
        if edge.left_sample_token not in samples or edge.right_sample_token not in samples:
            raise ValueError("evidence graph references an unknown sample token")


def _validate_dependency_edges(
    samples: tuple[PublicSplitSample, ...],
    edges: tuple[PublicSplitEvidenceEdge, ...],
    source_id_to_token: dict[str, str],
) -> None:
    declared: set[tuple[str, str]] = set()
    by_token = {sample.sample_token: sample for sample in samples}
    for sample in samples:
        if sample.source_variant == "random_background":
            if sample.dataset_name != "yt-bb-dog" or sample.paired_source_sample_id is None:
                raise ValueError("random-background sample lacks a typed YT dependency")
            if sample.paired_source_sample_id not in source_id_to_token:
                raise ValueError("random-background pair references an unknown source sample")
            other = by_token[source_id_to_token[sample.paired_source_sample_id]]
            if (
                other.source_variant != "original"
                or other.identity_token != sample.identity_token
                or other.dataset_identity_id != sample.dataset_identity_id
                or other.original_split != "test"
                or sample.original_split != "test"
                or other.sequence_token != sample.sequence_token
                or other.raw_frame_index != sample.raw_frame_index
            ):
                raise ValueError("random-background dependency differs in identity or variant")
            declared.add(tuple(sorted((sample.sample_token, other.sample_token))))
        elif sample.paired_source_sample_id is not None:
            raise ValueError("only random-background controls may declare a pair")
    observed = {
        (edge.left_sample_token, edge.right_sample_token)
        for edge in edges
        if edge.relation is EvidenceRelation.DEPENDENCY
    }
    if not declared.issubset(observed):
        raise ValueError("typed random-background dependency edge is missing")


def _close_components(
    samples: tuple[PublicSplitSample, ...],
    edges: tuple[PublicSplitEvidenceEdge, ...],
) -> tuple[tuple[_Component, ...], set[str], dict[str, set[str]]]:
    union_relations = {
        EvidenceRelation.EXACT_CONFIRMED,
        EvidenceRelation.GEOMETRIC_CONFIRMED,
        EvidenceRelation.DEPENDENCY,
        EvidenceRelation.REVIEW_CONFIRMED,
    }
    dsu = _DisjointSet(sample.sample_token for sample in samples)
    for edge in edges:
        if edge.relation in union_relations:
            dsu.union(edge.left_sample_token, edge.right_sample_token)
    grouped: dict[str, list[PublicSplitSample]] = defaultdict(list)
    for sample in samples:
        grouped[dsu.find(sample.sample_token)].append(sample)
    components: list[_Component] = []
    by_sample: dict[str, str] = {}
    for members in grouped.values():
        ordered = tuple(sorted(members, key=lambda item: item.sample_token))
        token = content_sha256({"domain": "CVI_PUBLIC_SPLIT_COMPONENT_V1", "members": [item.sample_token for item in ordered]})
        components.append(_Component(token, ordered))
        by_sample.update((item.sample_token, token) for item in ordered)
    components.sort(key=lambda item: item.token)
    reasons: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.relation is EvidenceRelation.REVIEW_UNRESOLVED:
            reasons[by_sample[edge.left_sample_token]].add("UNRESOLVED_REVIEW")
            reasons[by_sample[edge.right_sample_token]].add("UNRESOLVED_REVIEW")
    return tuple(components), set(reasons), reasons


def _build_allocation_blocks(
    components: tuple[_Component, ...],
) -> tuple[tuple[_AllocationBlock, ...], dict[str, _AllocationBlock]]:
    identities = {
        sample.identity_token
        for component in components
        for sample in component.samples
    }
    dsu = _DisjointSet(identities)
    component_identities: dict[str, tuple[str, ...]] = {}
    for component in components:
        members = tuple(sorted({sample.identity_token for sample in component.samples}))
        component_identities[component.token] = members
        for identity in members[1:]:
            dsu.union(members[0], identity)
    grouped_identities: dict[str, set[str]] = defaultdict(set)
    for identity in identities:
        grouped_identities[dsu.find(identity)].add(identity)
    component_tokens_by_root: dict[str, set[str]] = defaultdict(set)
    for component_token, members in component_identities.items():
        component_tokens_by_root[dsu.find(members[0])].add(component_token)
    blocks = tuple(
        sorted(
            (
                _AllocationBlock(
                    token=content_sha256({
                        "domain": "CVI_PUBLIC_SPLIT_ALLOCATION_BLOCK_V1",
                        "identities": sorted(grouped_identities[root]),
                        "components": sorted(component_tokens_by_root[root]),
                    }),
                    identity_tokens=tuple(sorted(grouped_identities[root])),
                    component_tokens=tuple(sorted(component_tokens_by_root[root])),
                )
                for root in grouped_identities
            ),
            key=lambda item: item.token,
        )
    )
    by_identity = {
        identity: block for block in blocks for identity in block.identity_tokens
    }
    return blocks, by_identity


def _quarantine_official_lane_conflicts(
    blocks: tuple[_AllocationBlock, ...],
    components: tuple[_Component, ...],
    reasons: dict[str, set[str]],
) -> None:
    lane_by_identity: dict[str, str] = {}
    for component in components:
        for sample in component.samples:
            lane = _official_lane(sample)
            prior = lane_by_identity.setdefault(sample.identity_token, lane)
            if prior != lane:
                raise ValueError("one identity crosses official dataset lanes")
    for block in blocks:
        lanes = {lane_by_identity[identity] for identity in block.identity_tokens}
        if len(lanes) > 1:
            for component_token in block.component_tokens:
                reasons[component_token].add("OFFICIAL_LANE_CONFLICT")


def _propagate_component_quarantine(
    blocks: tuple[_AllocationBlock, ...],
    reasons: dict[str, set[str]],
) -> None:
    for block in blocks:
        if any(component in reasons for component in block.component_tokens):
            for component in block.component_tokens:
                reasons[component].add("ALLOCATION_BLOCK_QUARANTINE_CLOSURE")


def _validate_official_identity_cardinalities(
    samples: tuple[PublicSplitSample, ...], policy: ProtectedPublicSplitPolicy
) -> None:
    originals = [sample for sample in samples if sample.source_variant == "original"]
    def ids(dataset: str, splits: set[str | None]) -> set[str]:
        return {
            sample.identity_token for sample in originals
            if sample.dataset_name == dataset and sample.original_split in splits
        }
    expected = {
        "YT train": (len(ids("yt-bb-dog", {"train"})), policy.yt_official_train_identities),
        "YT test": (len(ids("yt-bb-dog", {"test"})), policy.yt_official_test_identities),
        "DogFace train": (len(ids("dogfacenet224", {"train"})), policy.dogface_train_identities),
        "DogFace test": (len(ids("dogfacenet224", {"test"})), policy.dogface_test_identities),
        "MPDD train/val": (len(ids("mpdd", {"train", "val"})), policy.mpdd_train_identities),
        "MPDD query/gallery": (len(ids("mpdd", {"query", "gallery"})), policy.mpdd_test_identities),
        "Sibetan": (len(ids("sibetan", {None})), policy.sibetan_identities),
    }
    differences = [f"{name}={actual}, expected={wanted}" for name, (actual, wanted) in expected.items() if actual != wanted]
    if differences:
        raise ValueError("official identity cardinality differs: " + "; ".join(differences))
    if ids("yt-bb-dog", {"train"}) & ids("yt-bb-dog", {"test"}):
        raise ValueError("YT official train/test identity boundary crossed")
    if ids("dogfacenet224", {"train"}) & ids("dogfacenet224", {"test"}):
        raise ValueError("DogFace official train/test identity boundary crossed")
    if ids("mpdd", {"train", "val"}) & ids("mpdd", {"query", "gallery"}):
        raise ValueError("MPDD official train/test identity boundary crossed")
    sibetan_known = {
        sample.identity_token for sample in originals
        if sample.dataset_name == "sibetan" and sample.in_no_mono_subset is True
    }
    if len(sibetan_known) != policy.sibetan_cross_sequence_identities:
        raise ValueError("Sibetan no-mono identity cardinality differs")


def _assign_identity_roles(
    samples: tuple[PublicSplitSample, ...],
    component_by_sample: dict[str, _Component],
    quarantined: set[str],
    policy: ProtectedPublicSplitPolicy,
    key: bytes,
    frame_key: bytes,
    historical_exposure: dict[str, ExposureStage],
    block_by_identity: dict[str, _AllocationBlock],
    quarantine_reasons: dict[str, set[str]],
) -> tuple[dict[str, str], dict[str, Any]]:
    original_by_identity: dict[str, list[PublicSplitSample]] = defaultdict(list)
    for sample in samples:
        if sample.source_variant == "original":
            original_by_identity[sample.identity_token].append(sample)
    eligible = {identity for identity in original_by_identity if identity not in quarantined}
    for split in ("train", "test"):
        domain = {
            identity
            for identity in eligible
            if _identity_matches(
                original_by_identity[identity], "yt-bb-dog", {split}
            )
        }
        individually_eligible = {
            identity
            for identity in domain
            if _complete_yt_temporal_plan(
                original_by_identity[identity],
                component_by_sample,
                policy,
                key,
            )
            is not None
        }
        rejected = _quarantine_incomplete_protocol_blocks(
            domain,
            individually_eligible,
            block_by_identity,
            quarantine_reasons,
        )
        eligible.difference_update(rejected)
    lane_by_identity = {
        identity: _official_lane(identity_samples[0])
        for identity, identity_samples in original_by_identity.items()
    }
    raw_lane_counts = Counter(lane_by_identity.values())
    eligible_lane_counts = Counter(
        lane_by_identity[identity] for identity in eligible
    )
    requested_role_counts = {
        "YT_FIT": policy.yt_requested_fit_identities,
        "YT_DEVELOPMENT": policy.yt_development_identities,
        "YT_CALIBRATION_KNOWN": policy.yt_calibration_known_identities,
        "YT_CALIBRATION_UNKNOWN": policy.yt_calibration_unknown_identities,
        "YT_TEST_KNOWN": policy.yt_test_known_identities,
        "YT_TEST_UNKNOWN": policy.yt_test_unknown_identities,
        "DOGFACE_FIT": policy.dogface_fit_identities,
        "DOGFACE_DEVELOPMENT": policy.dogface_development_identities,
        "DOGFACE_CALIBRATION": policy.dogface_calibration_identities,
        "DOGFACE_TEST": policy.dogface_test_identities,
        "MPDD_EXTERNAL_KNOWN": policy.mpdd_open_known_identities,
        "MPDD_EXTERNAL_UNKNOWN": policy.mpdd_open_unknown_identities,
        "MPDD_EXTERNAL_UNUSED_TRAIN_DOMAIN": policy.mpdd_train_identities,
        "SIBETAN_EXTERNAL_KNOWN": policy.sibetan_cross_sequence_identities,
        "SIBETAN_EXTERNAL_UNKNOWN": policy.sibetan_unknown_identities,
    }
    minimum_role_counts = {
        **requested_role_counts,
        "YT_FIT": policy.yt_minimum_fit_identities,
        "YT_TEST_UNKNOWN": policy.yt_test_unknown_minimum_identities,
        "DOGFACE_FIT": policy.dogface_minimum_fit_identities,
        "DOGFACE_TEST": policy.dogface_minimum_test_identities,
    }
    capacity_contract = {
        "capacity_mode": policy.capacity_mode,
        "raw_official_identity_counts_by_lane": dict(sorted(raw_lane_counts.items())),
        "eligible_identity_counts_by_lane": dict(sorted(eligible_lane_counts.items())),
        "quarantined_identity_counts_by_lane": {
            lane: raw_lane_counts[lane] - eligible_lane_counts[lane]
            for lane in sorted(raw_lane_counts)
        },
        "requested_role_counts": requested_role_counts,
        "minimum_role_counts": minimum_role_counts,
        "yt_test_unknown_fpir_power": _yt_test_unknown_fpir_power(
            0, policy
        ),
    }
    yt_train = _rank_tokens(
        (
            identity
            for identity in eligible
            if _identity_matches(
                original_by_identity[identity], "yt-bb-dog", {"train"}
            )
            and _complete_yt_temporal_plan(
                original_by_identity[identity],
                component_by_sample,
                policy,
                key,
            )
            is not None
        ),
        key,
        b"YT_TRAIN",
    )
    capacity_status = "PASS_PROTECTED_SPLIT_CONSTRUCTION"
    if len(yt_train) < policy.yt_minimum_eligible_train_identities:
        capacity_status = "SPLIT_CAPACITY_FAILED"
    actual_fit = min(
        policy.yt_requested_fit_identities,
        max(0, len(yt_train) - policy.yt_development_identities - policy.yt_calibration_known_identities - policy.yt_calibration_unknown_identities),
    )
    if actual_fit < policy.yt_minimum_fit_identities:
        capacity_status = "SPLIT_CAPACITY_FAILED"
    capacity = {
        **capacity_contract,
        "status": capacity_status,
        "eligible_yt_train_identities": len(yt_train),
        "minimum_eligible_yt_train_identities": policy.yt_minimum_eligible_train_identities,
        "requested_fit_identities": policy.yt_requested_fit_identities,
        "minimum_fit_identities": policy.yt_minimum_fit_identities,
        "actual_fit_identities": actual_fit if capacity_status.startswith("PASS") else 0,
        "protected_role_counts": {
            "development": policy.yt_development_identities,
            "calibration_known": policy.yt_calibration_known_identities,
            "calibration_unknown": policy.yt_calibration_unknown_identities,
        },
    }
    if capacity_status != "PASS_PROTECTED_SPLIT_CONSTRUCTION":
        _mark_capacity_loss(
            {
                identity
                for identity, identity_samples in original_by_identity.items()
                if _identity_matches(identity_samples, "yt-bb-dog", {"train"})
            },
            block_by_identity,
            quarantine_reasons,
        )
        return {}, capacity
    roles = _allocate_exposure_constrained_roles(
        yt_train,
        (
            ("YT_FIT", actual_fit),
            ("YT_DEVELOPMENT", policy.yt_development_identities),
            ("YT_CALIBRATION_KNOWN", policy.yt_calibration_known_identities),
            ("YT_CALIBRATION_UNKNOWN", policy.yt_calibration_unknown_identities),
        ),
        historical_exposure,
        block_by_identity,
    )
    if roles is None:
        _mark_allocation_failure(
            yt_train, block_by_identity, quarantine_reasons
        )
        return {}, {
            **capacity,
            "status": "ROLE_EXPOSURE_CAPACITY_FAILED",
            "exposure_constrained_domain": "YT_TRAIN",
        }
    yt_test = _rank_tokens(
        (
            identity
            for identity in eligible
            if _identity_matches(
                original_by_identity[identity], "yt-bb-dog", {"test"}
            )
            and _complete_yt_temporal_plan(
                original_by_identity[identity],
                component_by_sample,
                policy,
                key,
            )
            is not None
        ),
        key,
        b"YT_TEST",
    )
    actual_yt_test_unknown = len(yt_test) - policy.yt_test_known_identities
    capacity["yt_test_unknown_fpir_power"] = _yt_test_unknown_fpir_power(
        max(0, actual_yt_test_unknown), policy
    )
    if (
        len(yt_test) < policy.yt_test_known_identities
        or actual_yt_test_unknown < policy.yt_test_unknown_minimum_identities
    ):
        _mark_capacity_loss(
            {
                identity
                for identity, identity_samples in original_by_identity.items()
                if _identity_matches(identity_samples, "yt-bb-dog", {"test"})
            },
            block_by_identity,
            quarantine_reasons,
        )
        return {}, {
            **capacity,
            "status": "YT_TEST_PRIMARY_CAPACITY_FAILED",
            "eligible_yt_test_primary_identities": len(yt_test),
            "minimum_yt_test_primary_identities": (
                policy.yt_test_known_identities
                + policy.yt_test_unknown_minimum_identities
            ),
            "primary_shot_counts": list(policy.shot_counts),
            "interpolation_allowed": False,
            "post_score_backfill_allowed": False,
        }
    capacity["eligible_yt_test_primary_identities"] = len(yt_test)
    yt_test_roles = _allocate_exposure_constrained_roles(
        yt_test,
        (
            ("YT_TEST_KNOWN", policy.yt_test_known_identities),
            ("YT_TEST_UNKNOWN", actual_yt_test_unknown),
        ),
        historical_exposure,
        block_by_identity,
    )
    if yt_test_roles is None:
        _mark_allocation_failure(
            yt_test, block_by_identity, quarantine_reasons
        )
        return {}, {
            **capacity,
            "status": "ROLE_EXPOSURE_CAPACITY_FAILED",
            "exposure_constrained_domain": "YT_TEST",
        }
    roles.update(yt_test_roles)

    dog_train = _rank_tokens(
        (identity for identity in eligible if _identity_matches(original_by_identity[identity], "dogfacenet224", {"train"})), key, b"DOGFACE_TRAIN"
    )
    actual_dogface_fit = (
        len(dog_train)
        - policy.dogface_development_identities
        - policy.dogface_calibration_identities
    )
    if actual_dogface_fit < policy.dogface_minimum_fit_identities:
        _mark_capacity_loss(
            {
                identity
                for identity, identity_samples in original_by_identity.items()
                if _identity_matches(identity_samples, "dogfacenet224", {"train"})
            },
            block_by_identity,
            quarantine_reasons,
        )
        return {}, {**capacity, "status": "SPLIT_CAPACITY_FAILED", "eligible_dogface_train_identities": len(dog_train)}
    dog_test = _rank_tokens(
        (
            identity
            for identity in eligible
            if _identity_matches(
                original_by_identity[identity], "dogfacenet224", {"test"}
            )
        ),
        key,
        b"DOGFACE_TEST",
    )
    if len(dog_test) < policy.dogface_minimum_test_identities:
        _mark_capacity_loss(
            {
                identity
                for identity, identity_samples in original_by_identity.items()
                if _identity_matches(identity_samples, "dogfacenet224", {"test"})
            },
            block_by_identity,
            quarantine_reasons,
        )
        return {}, {
            **capacity,
            "status": "SPLIT_CAPACITY_FAILED",
            "eligible_dogface_test_identities": len(dog_test),
        }
    dog_train_roles = _allocate_exposure_constrained_roles(
        dog_train,
        (
            ("DOGFACE_FIT", actual_dogface_fit),
            ("DOGFACE_DEVELOPMENT", policy.dogface_development_identities),
            ("DOGFACE_CALIBRATION", policy.dogface_calibration_identities),
        ),
        historical_exposure,
        block_by_identity,
    )
    dog_test_roles = _allocate_exposure_constrained_roles(
        dog_test,
        (("DOGFACE_TEST", len(dog_test)),),
        historical_exposure,
        block_by_identity,
    )
    if dog_train_roles is None or dog_test_roles is None:
        _mark_allocation_failure(
            (*dog_train, *dog_test), block_by_identity, quarantine_reasons
        )
        return {}, {
            **capacity,
            "status": "ROLE_EXPOSURE_CAPACITY_FAILED",
            "exposure_constrained_domain": "DOGFACE",
        }
    roles.update(dog_train_roles)
    roles.update(dog_test_roles)
    mpdd_train = _rank_tokens(
        (
            identity
            for identity in eligible
            if _identity_matches(
                original_by_identity[identity], "mpdd", {"train", "val"}
            )
        ),
        key,
        b"MPDD_TRAIN_DOMAIN",
    )
    mpdd_test = _rank_tokens(
        (identity for identity in eligible if _identity_matches(original_by_identity[identity], "mpdd", {"query", "gallery"})), key, b"MPDD_TEST"
    )
    if (
        len(mpdd_train) != policy.mpdd_train_identities
        or len(mpdd_test) != policy.mpdd_test_identities
    ):
        _mark_capacity_loss(
            {
                identity
                for identity, identity_samples in original_by_identity.items()
                if _identity_matches(
                    identity_samples,
                    "mpdd",
                    {"train", "val", "query", "gallery"},
                )
            },
            block_by_identity,
            quarantine_reasons,
        )
        return {}, {
            **capacity,
            "status": "SPLIT_CAPACITY_FAILED",
            "eligible_mpdd_train_identities": len(mpdd_train),
            "eligible_mpdd_test_identities": len(mpdd_test),
        }
    mpdd_known_eligible = _rank_tokens(
        (
            identity
            for identity in mpdd_test
            if _mpdd_open_set_plan(
                original_by_identity[identity],
                component_by_sample,
                frame_key,
                policy,
            )
            is not None
        ),
        key,
        b"MPDD_EXTERNAL_KNOWN",
    )
    mpdd_known_eligible_set = set(mpdd_known_eligible)
    mpdd_test_set = set(mpdd_test)
    eligible_known_blocks: list[_AllocationBlock] = []
    seen_known_blocks: set[str] = set()
    for identity in mpdd_known_eligible:
        block = block_by_identity[identity]
        if block.token in seen_known_blocks:
            continue
        seen_known_blocks.add(block.token)
        members = set(block.identity_tokens)
        if members.issubset(mpdd_known_eligible_set) and members.issubset(
            mpdd_test_set
        ):
            eligible_known_blocks.append(block)
    selected_known_blocks = _select_ranked_blocks(
        eligible_known_blocks, policy.mpdd_open_known_identities
    )
    if selected_known_blocks is None:
        return {}, {
            **capacity,
            "status": "EXTERNAL_OPEN_SET_CAPACITY_FAILED",
            "eligible_mpdd_open_known_identities": len(mpdd_known_eligible),
            "required_mpdd_open_known_identities": policy.mpdd_open_known_identities,
        }
    mpdd_known_set = {
        identity
        for block in selected_known_blocks
        for identity in block.identity_tokens
    }
    mpdd_known = [
        identity for identity in mpdd_known_eligible if identity in mpdd_known_set
    ]
    mpdd_unknown = _rank_tokens(
        (identity for identity in mpdd_test if identity not in mpdd_known_set),
        key,
        b"MPDD_EXTERNAL_UNKNOWN",
    )
    if len(mpdd_unknown) != policy.mpdd_open_unknown_identities:
        raise RuntimeError("MPDD known/unknown role arithmetic differs")
    mpdd_roles = _allocate_exposure_constrained_roles(
        (*mpdd_known, *mpdd_unknown, *mpdd_train),
        (
            ("MPDD_EXTERNAL_KNOWN", len(mpdd_known)),
            ("MPDD_EXTERNAL_UNKNOWN", len(mpdd_unknown)),
            ("MPDD_EXTERNAL_UNUSED_TRAIN_DOMAIN", len(mpdd_train)),
        ),
        historical_exposure,
        block_by_identity,
    )
    if mpdd_roles is None:
        _mark_allocation_failure(
            (*mpdd_known, *mpdd_unknown, *mpdd_train),
            block_by_identity,
            quarantine_reasons,
        )
        return {}, {
            **capacity,
            "status": "ROLE_EXPOSURE_CAPACITY_FAILED",
            "exposure_constrained_domain": "MPDD",
        }
    roles.update(mpdd_roles)
    sibetan_known = [
        identity for identity in eligible
        if _identity_matches(original_by_identity[identity], "sibetan", {None})
        and any(sample.in_no_mono_subset is True for sample in original_by_identity[identity])
    ]
    sibetan_unknown = [
        identity for identity in eligible
        if _identity_matches(original_by_identity[identity], "sibetan", {None})
        and all(sample.in_no_mono_subset is False for sample in original_by_identity[identity])
    ]
    if (len(sibetan_known), len(sibetan_unknown)) != (39, 20):
        _mark_capacity_loss(
            {
                identity
                for identity, identity_samples in original_by_identity.items()
                if _identity_matches(identity_samples, "sibetan", {None})
            },
            block_by_identity,
            quarantine_reasons,
        )
        return {}, {**capacity, "status": "SPLIT_CAPACITY_FAILED", "eligible_sibetan_known_unknown": [len(sibetan_known), len(sibetan_unknown)]}
    sibetan_roles = _allocate_exposure_constrained_roles(
        (*sibetan_known, *sibetan_unknown),
        (
            ("SIBETAN_EXTERNAL_KNOWN", len(sibetan_known)),
            ("SIBETAN_EXTERNAL_UNKNOWN", len(sibetan_unknown)),
        ),
        historical_exposure,
        block_by_identity,
    )
    if sibetan_roles is None:
        _mark_allocation_failure(
            (*sibetan_known, *sibetan_unknown),
            block_by_identity,
            quarantine_reasons,
        )
        return {}, {
            **capacity,
            "status": "ROLE_EXPOSURE_CAPACITY_FAILED",
            "exposure_constrained_domain": "SIBETAN",
        }
    roles.update(sibetan_roles)
    capacity["assigned_identity_count"] = len(roles)
    actual_role_counts = Counter(roles.values())
    capacity["actual_role_counts"] = {
        role: actual_role_counts.get(role, 0)
        for role in requested_role_counts
    }
    capacity["contracted_role_counts"] = {
        role: requested - actual_role_counts.get(role, 0)
        for role, requested in requested_role_counts.items()
    }
    return roles, capacity


def _yt_test_unknown_fpir_power(
    actual_trials: int,
    policy: ProtectedPublicSplitPolicy,
) -> dict[str, Any]:
    targets = (
        ("PRIMARY", policy.yt_test_unknown_target_fpir),
        ("REPORTING", policy.yt_test_unknown_reporting_fpir),
    )
    return {
        "confidence_level": policy.yt_test_unknown_confidence_level,
        "actual_unknown_identity_trials": actual_trials,
        "targets": [
            {
                "purpose": purpose,
                "target_fpir": target,
                "required_zero_event_trials": required,
                "status": "POWERED" if actual_trials >= required else "UNDERPOWERED",
            }
            for purpose, target in targets
            for required in (
                required_zero_event_trials(
                    target,
                    confidence_level=policy.yt_test_unknown_confidence_level,
                ),
            )
        ],
    }


def _allocate_exposure_constrained_roles(
    ranked_identities: Iterable[str],
    role_counts: tuple[tuple[str, int], ...],
    historical_exposure: dict[str, ExposureStage],
    block_by_identity: dict[str, _AllocationBlock],
) -> dict[str, str] | None:
    """Fill exact role quotas with indivisible blocks in existing HMAC order."""

    ranked = list(ranked_identities)
    ranked_set = set(ranked)
    ordered_blocks: list[_AllocationBlock] = []
    seen_blocks: set[str] = set()
    for identity in ranked:
        block = block_by_identity[identity]
        if block.token in seen_blocks:
            continue
        if not set(block.identity_tokens).issubset(ranked_set):
            return None
        seen_blocks.add(block.token)
        ordered_blocks.append(block)
    remaining = ordered_blocks
    assigned: dict[str, str] = {}
    for role, count in role_counts:
        eligible = [
            block
            for block in remaining
            if role_allows_historical_stage(
                role,
                _maximum_historical_stage(block, historical_exposure),
            )
        ]
        selected = _select_ranked_blocks(eligible, count)
        if selected is None:
            return None
        selected_tokens = {block.token for block in selected}
        assigned.update(
            (identity, role)
            for block in selected
            for identity in block.identity_tokens
        )
        remaining = [
            block for block in remaining if block.token not in selected_tokens
        ]
    if remaining:
        return None
    return assigned


def _select_ranked_blocks(
    blocks: list[_AllocationBlock], target: int
) -> list[_AllocationBlock] | None:
    suffix = [0] * (len(blocks) + 1)
    suffix[-1] = 1
    mask = (1 << (target + 1)) - 1
    for index in range(len(blocks) - 1, -1, -1):
        size = len(blocks[index].identity_tokens)
        suffix[index] = suffix[index + 1] | (
            suffix[index + 1] << size
        ) & mask
    if not (suffix[0] >> target) & 1:
        return None
    selected: list[_AllocationBlock] = []
    remaining = target
    for index, block in enumerate(blocks):
        size = len(block.identity_tokens)
        if size <= remaining and (suffix[index + 1] >> (remaining - size)) & 1:
            selected.append(block)
            remaining -= size
    if remaining != 0:
        raise RuntimeError("component subset reconstruction differs")
    return selected


def _maximum_historical_stage(
    block: _AllocationBlock,
    historical_exposure: dict[str, ExposureStage],
) -> ExposureStage | None:
    stages = [
        historical_exposure[identity]
        for identity in block.identity_tokens
        if identity in historical_exposure
    ]
    return max(stages, key=lambda stage: tuple(ExposureStage).index(stage)) if stages else None


def _mark_capacity_loss(
    domain_identities: set[str],
    block_by_identity: dict[str, _AllocationBlock],
    reasons: dict[str, set[str]],
) -> None:
    blocks = {block_by_identity[identity] for identity in domain_identities}
    for block in blocks:
        if any(component in reasons for component in block.component_tokens):
            for component in block.component_tokens:
                reasons[component].add("FIXED_QUOTA_CAPACITY_LOSS")


def _quarantine_incomplete_protocol_blocks(
    domain_identities: set[str],
    individually_eligible: set[str],
    block_by_identity: dict[str, _AllocationBlock],
    reasons: dict[str, set[str]],
) -> set[str]:
    rejected: set[str] = set()
    for block in {block_by_identity[identity] for identity in domain_identities}:
        members = set(block.identity_tokens)
        if not members.issubset(individually_eligible):
            rejected.update(members)
            for component in block.component_tokens:
                reasons[component].add("PROTOCOL_EVIDENCE_CAPACITY_CONFLICT")
    return rejected


def _mark_allocation_failure(
    identities: Iterable[str],
    block_by_identity: dict[str, _AllocationBlock],
    reasons: dict[str, set[str]],
) -> None:
    for block in {block_by_identity[identity] for identity in identities}:
        for component in block.component_tokens:
            reasons[component].add("INDIVISIBLE_COMPONENT_QUOTA_CONFLICT")


def _build_protocol_uses(
    samples: tuple[PublicSplitSample, ...],
    component_by_sample: dict[str, _Component],
    quarantined_components: set[str],
    roles: dict[str, str],
    policy: ProtectedPublicSplitPolicy,
    keys: dict[str, bytes],
) -> tuple[defaultdict[str, list[dict[str, Any]]], dict[str, Any]]:
    uses: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    originals: dict[str, list[PublicSplitSample]] = defaultdict(list)
    for sample in samples:
        if (
            sample.source_variant == "original"
            and sample.identity_token in roles
            and component_by_sample[sample.sample_token].token
            not in quarantined_components
        ):
            originals[sample.identity_token].append(sample)

    development = sorted(
        identity
        for identity, role in roles.items()
        if role == "YT_DEVELOPMENT"
    )
    calibration_known = sorted(
        identity
        for identity, role in roles.items()
        if role == "YT_CALIBRATION_KNOWN"
    )
    calibration_unknown = sorted(
        identity
        for identity, role in roles.items()
        if role == "YT_CALIBRATION_UNKNOWN"
    )
    test_known = sorted(
        identity for identity, role in roles.items() if role == "YT_TEST_KNOWN"
    )
    test_unknown = sorted(
        identity for identity, role in roles.items() if role == "YT_TEST_UNKNOWN"
    )
    temporal_plans: dict[
        str, dict[int, tuple[tuple[PublicSplitSample, ...], PublicSplitSample]]
    ] = {}
    missing_development: list[str] = []
    missing_calibration_known: list[str] = []
    missing_calibration_unknown: list[str] = []
    for identity, missing in (
        *((identity, missing_development) for identity in development),
        *((identity, missing_calibration_known) for identity in calibration_known),
        *((identity, missing_calibration_unknown) for identity in calibration_unknown),
    ):
        plan = _complete_yt_temporal_plan(
            originals[identity], component_by_sample, policy, keys["frame_roles"]
        )
        if plan is None:
            missing.append(identity)
        else:
            temporal_plans[identity] = plan
    test_temporal_plans: dict[
        str, dict[int, tuple[tuple[PublicSplitSample, ...], PublicSplitSample]]
    ] = {}
    missing_test_known: list[str] = []
    missing_test_unknown: list[str] = []
    diagnostic_test_plans: dict[
        str, dict[int, tuple[tuple[PublicSplitSample, ...], PublicSplitSample]]
    ] = {}
    for identity, missing in (
        *((identity, missing_test_known) for identity in test_known),
        *((identity, missing_test_unknown) for identity in test_unknown),
    ):
        primary_plan = _complete_yt_temporal_plan(
            originals[identity],
            component_by_sample,
            policy,
            keys["frame_roles"],
        )
        if primary_plan is None:
            missing.append(identity)
            continue
        test_temporal_plans[identity] = primary_plan
        diagnostic_plan = _complete_yt_temporal_plan(
            originals[identity],
            component_by_sample,
            policy,
            keys["frame_roles"],
            shots=policy.diagnostic_shot_counts,
        )
        if diagnostic_plan is not None:
            diagnostic_test_plans[identity] = diagnostic_plan
    evidence_capacity = {
        "status": "PASS_PROTOCOL_EVIDENCE_CAPACITY",
        "development_identity_count": len(development),
        "development_eligible_identity_count": (
            len(development) - len(missing_development)
        ),
        "calibration_known_identity_count": len(calibration_known),
        "calibration_known_eligible_identity_count": (
            len(calibration_known) - len(missing_calibration_known)
        ),
        "calibration_unknown_identity_count": len(calibration_unknown),
        "calibration_unknown_eligible_identity_count": (
            len(calibration_unknown) - len(missing_calibration_unknown)
        ),
        "test_known_identity_count": len(test_known),
        "test_known_primary_eligible_identity_count": (
            len(test_known) - len(missing_test_known)
        ),
        "test_unknown_identity_count": len(test_unknown),
        "test_unknown_primary_eligible_identity_count": (
            len(test_unknown) - len(missing_test_unknown)
        ),
        "test_diagnostic_five_shot_eligible_identity_count": len(
            diagnostic_test_plans
        ),
        "primary_shots": list(policy.shot_counts),
        "diagnostic_shots": list(policy.diagnostic_shot_counts),
        "yt_primary_open_set_gallery_size": (
            policy.yt_primary_open_set_gallery_size
        ),
        "yt_primary_open_set_shot": policy.yt_primary_open_set_shot,
        "mpdd_external_gallery_size": policy.mpdd_external_gallery_size,
        "sibetan_external_gallery_size": policy.sibetan_external_gallery_size,
        "calibration_gallery_sizes": list(
            policy.yt_calibration_gallery_sizes
        ),
        "interpolation_allowed": False,
        "post_score_backfill_allowed": False,
    }
    if missing_development:
        evidence_capacity["status"] = "DEVELOPMENT_CAPACITY_FAILED"
        return defaultdict(list), evidence_capacity
    if missing_calibration_known or missing_calibration_unknown:
        evidence_capacity["status"] = "CALIBRATION_CAPACITY_FAILED"
        return defaultdict(list), evidence_capacity
    if missing_test_known or missing_test_unknown:
        evidence_capacity["status"] = "YT_TEST_PRIMARY_CAPACITY_FAILED"
        return defaultdict(list), evidence_capacity

    _development_open_set_uses(
        uses,
        development,
        temporal_plans,
        policy,
        keys,
    )
    _calibration_open_set_uses(
        uses,
        calibration_known,
        calibration_unknown,
        temporal_plans,
        policy,
        keys,
    )
    diagnostic_gallery_size = len(diagnostic_test_plans)
    for identity, identity_samples in originals.items():
        role = roles[identity]
        if role in {"YT_TEST_KNOWN", "YT_TEST_UNKNOWN"}:
            _yt_uses(
                uses,
                role,
                policy,
                keys,
                test_temporal_plans[identity],
                diagnostic_test_plans.get(identity),
                diagnostic_gallery_size,
            )
        elif role == "DOGFACE_TEST":
            _ranked_image_uses(uses, identity_samples, component_by_sample, "DOGFACE_CLOSED_SET", (1, 3), keys["frame_roles"])
        elif role.startswith("MPDD_EXTERNAL_") and role != "MPDD_EXTERNAL_UNUSED_TRAIN_DOMAIN":
            _mpdd_uses(
                uses,
                identity_samples,
                component_by_sample,
                role,
                keys["frame_roles"],
                policy,
            )
        elif role == "SIBETAN_EXTERNAL_KNOWN":
            _sibetan_known_uses(
                uses, identity_samples, component_by_sample, keys, policy
            )
        elif role == "SIBETAN_EXTERNAL_UNKNOWN":
            chosen = _rank_samples(identity_samples, keys["frame_roles"], b"SIBETAN_UNKNOWN_QUERY")[0]
            for shot in policy.shot_counts:
                _add_use(
                    uses,
                    chosen.sample_token,
                    "SIBETAN_OPEN_SET",
                    shot,
                    "UNKNOWN_QUERY",
                    keys["bootstrap"],
                    identity,
                    gallery_size=policy.sibetan_external_gallery_size,
                )
    for sample in samples:
        if sample.source_variant == "random_background" and sample.identity_token in roles:
            _add_use(uses, sample.sample_token, "YT_RANDOM_BACKGROUND_PAIRED_DELTA", 0, "PAIRED_CONTROL", keys["bootstrap"], sample.identity_token)
    external_capacity = _external_open_set_capacity(uses, originals, roles, policy)
    evidence_capacity["external_open_set"] = external_capacity
    if external_capacity["status"] != "PASS_EXTERNAL_OPEN_SET_CAPACITY":
        evidence_capacity["status"] = external_capacity["status"]
        return defaultdict(list), evidence_capacity
    return uses, evidence_capacity


def _complete_yt_temporal_plan(
    samples: list[PublicSplitSample],
    component_by_sample: dict[str, _Component],
    policy: ProtectedPublicSplitPolicy,
    frame_key: bytes,
    *,
    shots: tuple[int, ...] | None = None,
) -> dict[int, tuple[tuple[PublicSplitSample, ...], PublicSplitSample]] | None:
    components = _unique_components(samples, component_by_sample)
    components.sort(
        key=lambda item: (
            min(
                sample.raw_frame_index
                for sample in item.samples
                if sample.source_variant == "original"
            ),
            item.token,
        )
    )
    representatives = [
        _component_representative(component, frame_key, b"YT_FRAME")
        for component in components
    ]
    plans: dict[
        int, tuple[tuple[PublicSplitSample, ...], PublicSplitSample]
    ] = {}
    selected_shots = policy.shot_counts if shots is None else shots
    if not selected_shots or any(
        isinstance(shot, bool) or not isinstance(shot, int) or shot <= 0
        for shot in selected_shots
    ):
        raise ValueError("YT temporal-plan shots must be positive integers")
    for shot in selected_shots:
        needed = shot + policy.yt_guard_components + 1
        if len(components) < needed:
            return None
        final_gallery_component = components[shot - 1]
        query_component = components[-1]
        ordinal_gap = (len(components) - 1) - (shot - 1)
        raw_gap = min(
            sample.raw_frame_index
            for sample in query_component.samples
            if sample.source_variant == "original"
        ) - max(
            sample.raw_frame_index
            for sample in final_gallery_component.samples
            if sample.source_variant == "original"
        )
        if (
            ordinal_gap < policy.yt_minimum_component_gap
            or raw_gap < policy.yt_minimum_raw_frame_gap
        ):
            return None
        plans[shot] = (
            tuple(representatives[:shot]),
            representatives[-1],
        )
    return plans


def _development_open_set_uses(
    uses: defaultdict[str, list[dict[str, Any]]],
    development: list[str],
    temporal_plans: dict[
        str, dict[int, tuple[tuple[PublicSplitSample, ...], PublicSplitSample]]
    ],
    policy: ProtectedPublicSplitPolicy,
    keys: dict[str, bytes],
) -> None:
    ordered = _rank_tokens(
        development,
        keys["development_episodes"],
        b"YT_DEVELOPMENT_AB",
    )
    size = policy.yt_development_episode_identities
    if len(ordered) != size * 2:
        raise RuntimeError("development role count changed after capacity check")
    group_a, group_b = ordered[:size], ordered[size:]
    for episode, known, unknown in (
        ("A_KNOWN_B_UNKNOWN", group_a, group_b),
        ("B_KNOWN_A_UNKNOWN", group_b, group_a),
    ):
        for shot in policy.shot_counts:
            for identity in known:
                gallery, query = temporal_plans[identity][shot]
                for sample in gallery:
                    _add_use(
                        uses,
                        sample.sample_token,
                        "YT_DEVELOPMENT_OPEN_SET",
                        shot,
                        "GALLERY",
                        keys["bootstrap"],
                        identity,
                        episode=episode,
                        gallery_size=size,
                    )
                _add_use(
                    uses,
                    query.sample_token,
                    "YT_DEVELOPMENT_OPEN_SET",
                    shot,
                    "KNOWN_QUERY",
                    keys["bootstrap"],
                    identity,
                    episode=episode,
                    gallery_size=size,
                )
            for identity in unknown:
                query = temporal_plans[identity][shot][1]
                _add_use(
                    uses,
                    query.sample_token,
                    "YT_DEVELOPMENT_OPEN_SET",
                    shot,
                    "UNKNOWN_QUERY",
                    keys["bootstrap"],
                    identity,
                    episode=episode,
                    gallery_size=size,
                )


def _calibration_open_set_uses(
    uses: defaultdict[str, list[dict[str, Any]]],
    calibration_known: list[str],
    calibration_unknown: list[str],
    temporal_plans: dict[
        str, dict[int, tuple[tuple[PublicSplitSample, ...], PublicSplitSample]]
    ],
    policy: ProtectedPublicSplitPolicy,
    keys: dict[str, bytes],
) -> None:
    known_order = _rank_tokens(
        calibration_known,
        keys["calibration_panels"],
        b"YT_CALIBRATION_NESTED_KNOWN",
    )
    unknown_order = _rank_tokens(
        calibration_unknown,
        keys["calibration_panels"],
        b"YT_CALIBRATION_UNKNOWN_EVENTS",
    )
    if (
        len(known_order) != policy.yt_calibration_known_identities
        or len(unknown_order) != policy.yt_calibration_unknown_identities
    ):
        raise RuntimeError("calibration role count changed after capacity check")
    for gallery_size in policy.yt_calibration_gallery_sizes:
        panel = known_order[:gallery_size]
        episode = f"N_{gallery_size}"
        for shot in policy.shot_counts:
            for identity in panel:
                gallery, query = temporal_plans[identity][shot]
                for sample in gallery:
                    _add_use(
                        uses,
                        sample.sample_token,
                        "YT_CALIBRATION_OPEN_SET",
                        shot,
                        "GALLERY",
                        keys["bootstrap"],
                        identity,
                        episode=episode,
                        gallery_size=gallery_size,
                    )
                _add_use(
                    uses,
                    query.sample_token,
                    "YT_CALIBRATION_OPEN_SET",
                    shot,
                    "KNOWN_QUERY",
                    keys["bootstrap"],
                    identity,
                    episode=episode,
                    gallery_size=gallery_size,
                    primary_query_scope="CALIBRATION_ALL_PANELS",
                )
            for identity in unknown_order:
                query = temporal_plans[identity][shot][1]
                _add_use(
                    uses,
                    query.sample_token,
                    "YT_CALIBRATION_OPEN_SET",
                    shot,
                    "UNKNOWN_QUERY",
                    keys["bootstrap"],
                    identity,
                    episode=episode,
                    gallery_size=gallery_size,
                    primary_query_scope="CALIBRATION_ALL_PANELS",
                )


def _yt_uses(
    uses: defaultdict[str, list[dict[str, Any]]],
    role: str,
    policy: ProtectedPublicSplitPolicy,
    keys: dict[str, bytes],
    primary_plan: dict[
        int, tuple[tuple[PublicSplitSample, ...], PublicSplitSample]
    ],
    diagnostic_plan: dict[
        int, tuple[tuple[PublicSplitSample, ...], PublicSplitSample]
    ] | None,
    diagnostic_gallery_size: int,
) -> None:
    if role not in {"YT_TEST_KNOWN", "YT_TEST_UNKNOWN"}:
        raise ValueError("YT protected-test role differs")
    if set(primary_plan) != set(policy.shot_counts):
        raise RuntimeError("YT primary temporal plan is incomplete")
    identity_token = next(iter(primary_plan.values()))[1].identity_token
    for shot in policy.shot_counts:
        gallery, query = primary_plan[shot]
        for sample in gallery:
            _add_use(
                uses,
                sample.sample_token,
                "YT_CLOSED_SET",
                shot,
                "GALLERY",
                keys["bootstrap"],
                identity_token,
                gallery_size=policy.yt_official_test_identities,
            )
        _add_use(
            uses,
            query.sample_token,
            "YT_CLOSED_SET",
            shot,
            "KNOWN_QUERY",
            keys["bootstrap"],
            identity_token,
            gallery_size=policy.yt_official_test_identities,
        )
        if role == "YT_TEST_KNOWN":
            for sample in gallery:
                _add_use(
                    uses,
                    sample.sample_token,
                    "YT_OPEN_SET",
                    shot,
                    "GALLERY",
                    keys["bootstrap"],
                    identity_token,
                    gallery_size=policy.yt_primary_open_set_gallery_size,
                    primary_query_scope="YT_PRIMARY_N300",
                )
            query_role = "KNOWN_QUERY"
        else:
            query_role = "UNKNOWN_QUERY"
        _add_use(
            uses,
            query.sample_token,
            "YT_OPEN_SET",
            shot,
            query_role,
            keys["bootstrap"],
            identity_token,
            gallery_size=policy.yt_primary_open_set_gallery_size,
            primary_query_scope="YT_PRIMARY_N300",
        )
    if diagnostic_plan is None:
        return
    if set(diagnostic_plan) != set(policy.diagnostic_shot_counts):
        raise RuntimeError("YT diagnostic temporal plan is incomplete")
    for shot in policy.diagnostic_shot_counts:
        gallery, query = diagnostic_plan[shot]
        for sample in gallery:
            _add_use(
                uses,
                sample.sample_token,
                "YT_CLOSED_SET_DIAGNOSTIC",
                shot,
                "GALLERY",
                keys["bootstrap"],
                identity_token,
                gallery_size=diagnostic_gallery_size,
            )
        _add_use(
            uses,
            query.sample_token,
            "YT_CLOSED_SET_DIAGNOSTIC",
            shot,
            "KNOWN_QUERY",
            keys["bootstrap"],
            identity_token,
            gallery_size=diagnostic_gallery_size,
        )


def _ranked_image_uses(uses: defaultdict[str, list[dict[str, Any]]], samples: list[PublicSplitSample], component_by_sample: dict[str, _Component], protocol: str, shots: tuple[int, ...], key: bytes) -> None:
    components = _unique_components(samples, component_by_sample)
    ranked = sorted(components, key=lambda item: (_hmac_digest(key, b"IMAGE_COMPONENT", item.token), item.token))
    reps = [_component_representative(item, key, b"IMAGE_FRAME") for item in ranked]
    for shot in shots:
        if len(reps) < shot + 1:
            continue
        for rep in reps[:shot]:
            _add_use(uses, rep.sample_token, protocol, shot, "GALLERY", key, rep.identity_token)
        _add_use(uses, reps[-1].sample_token, protocol, shot, "KNOWN_QUERY", key, reps[-1].identity_token)


def _mpdd_uses(uses: defaultdict[str, list[dict[str, Any]]], samples: list[PublicSplitSample], component_by_sample: dict[str, _Component], role: str, key: bytes, policy: ProtectedPublicSplitPolicy) -> None:
    gallery = [sample for sample in samples if sample.original_split == "gallery"]
    query = [sample for sample in samples if sample.original_split == "query"]
    chosen_query = _rank_samples(query, key, b"MPDD_QUERY")[0] if query else None
    query_component = (
        component_by_sample[chosen_query.sample_token].token
        if chosen_query is not None
        else None
    )
    gallery_components = [
        component
        for component in _unique_components(gallery, component_by_sample)
        if component.token != query_component
    ]
    ranked = sorted(gallery_components, key=lambda item: (_hmac_digest(key, b"MPDD_GALLERY", item.token), item.token))
    reps = [_component_representative(item, key, b"MPDD_FRAME") for item in ranked]
    if chosen_query is not None:
        if role == "MPDD_EXTERNAL_UNKNOWN":
            for shot in policy.shot_counts:
                _add_use(
                    uses,
                    chosen_query.sample_token,
                    "MPDD_OPEN_SET",
                    shot,
                    "UNKNOWN_QUERY",
                    key,
                    chosen_query.identity_token,
                    gallery_size=policy.mpdd_external_gallery_size,
                )
        for shot in (*policy.shot_counts, *policy.diagnostic_shot_counts):
            if len(reps) < shot:
                continue
            for rep in reps[:shot]:
                _add_use(uses, rep.sample_token, "MPDD_CLOSED_SET", shot, "GALLERY", key, rep.identity_token)
                if role == "MPDD_EXTERNAL_KNOWN" and shot in policy.shot_counts:
                    _add_use(uses, rep.sample_token, "MPDD_OPEN_SET", shot, "GALLERY", key, rep.identity_token, gallery_size=policy.mpdd_external_gallery_size)
            _add_use(uses, chosen_query.sample_token, "MPDD_CLOSED_SET", shot, "KNOWN_QUERY", key, chosen_query.identity_token)
            if role == "MPDD_EXTERNAL_KNOWN" and shot in policy.shot_counts:
                _add_use(uses, chosen_query.sample_token, "MPDD_OPEN_SET", shot, "KNOWN_QUERY", key, chosen_query.identity_token, gallery_size=policy.mpdd_external_gallery_size)


def _mpdd_open_set_plan(
    samples: list[PublicSplitSample],
    component_by_sample: dict[str, _Component],
    key: bytes,
    policy: ProtectedPublicSplitPolicy,
) -> tuple[tuple[_Component, ...], PublicSplitSample] | None:
    query = [sample for sample in samples if sample.original_split == "query"]
    if not query:
        return None
    chosen_query = _rank_samples(query, key, b"MPDD_QUERY")[0]
    query_component = component_by_sample[chosen_query.sample_token].token
    gallery_components = tuple(
        component
        for component in _unique_components(
            [sample for sample in samples if sample.original_split == "gallery"],
            component_by_sample,
        )
        if component.token != query_component
    )
    if len(gallery_components) < max(policy.shot_counts):
        return None
    return gallery_components, chosen_query


def _sibetan_known_uses(uses: defaultdict[str, list[dict[str, Any]]], samples: list[PublicSplitSample], component_by_sample: dict[str, _Component], keys: dict[str, bytes], policy: ProtectedPublicSplitPolicy) -> None:
    by_sequence: dict[str, list[PublicSplitSample]] = defaultdict(list)
    for sample in samples:
        by_sequence[sample.sequence_token].append(sample)
    if len(by_sequence) < 2:
        return
    sequences_by_component: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        sequences_by_component[component_by_sample[sample.sample_token].token].add(
            sample.sequence_token
        )
    by_sequence = {
        sequence: [
            sample
            for sample in sequence_samples
            if len(
                sequences_by_component[
                    component_by_sample[sample.sample_token].token
                ]
            )
            == 1
        ]
        for sequence, sequence_samples in by_sequence.items()
    }
    by_sequence = {
        sequence: sequence_samples
        for sequence, sequence_samples in by_sequence.items()
        if sequence_samples
    }
    if len(by_sequence) < 2:
        return
    minimum_gallery_components = max(policy.shot_counts)
    gallery_eligible_sequences = [
        sequence
        for sequence, sequence_samples in by_sequence.items()
        if len(_unique_components(sequence_samples, component_by_sample))
        >= minimum_gallery_components
    ]
    if not gallery_eligible_sequences:
        return
    gallery_sequence = _rank_tokens(
        gallery_eligible_sequences,
        keys["sequence_roles"],
        b"SIBETAN_SEQUENCE",
    )[0]
    query_sequences = _rank_tokens(
        (sequence for sequence in by_sequence if sequence != gallery_sequence),
        keys["sequence_roles"],
        b"SIBETAN_SEQUENCE",
    )
    gallery_components = _unique_components(by_sequence[gallery_sequence], component_by_sample)
    gallery_components.sort(key=lambda item: (_hmac_digest(keys["frame_roles"], b"SIBETAN_GALLERY", item.token), item.token))
    reps = [_component_representative(item, keys["frame_roles"], b"SIBETAN_FRAME") for item in gallery_components]
    query_reps = [
        _rank_samples(by_sequence[sequence], keys["frame_roles"], b"SIBETAN_QUERY")[0]
        for sequence in query_sequences
    ]
    for shot in (1, 3, 5):
        if len(reps) < shot:
            continue
        for rep in reps[:shot]:
            _add_use(uses, rep.sample_token, "SIBETAN_CROSS_SEQUENCE", shot, "GALLERY", keys["bootstrap"], rep.identity_token)
            if shot in policy.shot_counts:
                _add_use(uses, rep.sample_token, "SIBETAN_OPEN_SET", shot, "GALLERY", keys["bootstrap"], rep.identity_token, gallery_size=policy.sibetan_external_gallery_size)
        for rep in query_reps:
            _add_use(uses, rep.sample_token, "SIBETAN_CROSS_SEQUENCE", shot, "KNOWN_QUERY", keys["bootstrap"], rep.identity_token)
            if shot in policy.shot_counts:
                _add_use(uses, rep.sample_token, "SIBETAN_OPEN_SET", shot, "KNOWN_QUERY", keys["bootstrap"], rep.identity_token, gallery_size=policy.sibetan_external_gallery_size)


def _external_open_set_capacity(
    uses: defaultdict[str, list[dict[str, Any]]],
    originals: dict[str, list[PublicSplitSample]],
    roles: dict[str, str],
    policy: ProtectedPublicSplitPolicy,
) -> dict[str, Any]:
    sample_identity = {
        sample.sample_token: identity
        for identity, samples in originals.items()
        for sample in samples
    }
    specifications = (
        (
            "MPDD_OPEN_SET",
            "MPDD_EXTERNAL_KNOWN",
            "MPDD_EXTERNAL_UNKNOWN",
            policy.mpdd_external_gallery_size,
            policy.mpdd_open_unknown_identities,
        ),
        (
            "SIBETAN_OPEN_SET",
            "SIBETAN_EXTERNAL_KNOWN",
            "SIBETAN_EXTERNAL_UNKNOWN",
            policy.sibetan_external_gallery_size,
            policy.sibetan_unknown_identities,
        ),
    )
    panels: dict[str, Any] = {}
    status = "PASS_EXTERNAL_OPEN_SET_CAPACITY"
    for protocol, known_role, unknown_role, gallery_size, unknown_size in specifications:
        known_ids = {identity for identity, role in roles.items() if role == known_role}
        unknown_ids = {identity for identity, role in roles.items() if role == unknown_role}
        panel_rows: dict[str, Any] = {}
        for shot in policy.shot_counts:
            gallery_counts: dict[str, int] = defaultdict(int)
            known_queries: set[str] = set()
            unknown_queries: set[str] = set()
            for sample_token, sample_uses in uses.items():
                identity = sample_identity.get(sample_token)
                if identity is None:
                    continue
                for use in sample_uses:
                    if use["protocol"] != protocol or use["shot"] != shot:
                        continue
                    if use["gallery_size"] != gallery_size:
                        status = "EXTERNAL_OPEN_SET_CAPACITY_FAILED"
                    if use["role"] == "GALLERY":
                        gallery_counts[identity] += 1
                    elif use["role"] == "KNOWN_QUERY":
                        known_queries.add(identity)
                    elif use["role"] == "UNKNOWN_QUERY":
                        unknown_queries.add(identity)
            complete_gallery = {
                identity for identity, count in gallery_counts.items() if count == shot
            }
            if (
                complete_gallery != known_ids
                or known_queries != known_ids
                or unknown_queries != unknown_ids
                or len(known_ids) != gallery_size
                or len(unknown_ids) != unknown_size
                or set(gallery_counts) & unknown_ids
            ):
                status = "EXTERNAL_OPEN_SET_CAPACITY_FAILED"
            panel_rows[f"N{gallery_size}_K{shot}"] = {
                "gallery_identity_count": len(complete_gallery),
                "known_query_identity_count": len(known_queries),
                "unknown_query_identity_count": len(unknown_queries),
            }
        panels[protocol] = panel_rows
    return {"status": status, "panels": panels}


def _add_use(
    uses: defaultdict[str, list[dict[str, Any]]],
    sample_token: str,
    protocol: str,
    shot: int,
    role: str,
    key: bytes,
    identity_token: str,
    *,
    episode: str = "PRIMARY",
    gallery_size: int = 0,
    primary_query_scope: str | None = None,
) -> None:
    event = hmac.new(
        key,
        b"CVI_EVENT_V2\0"
        + protocol.encode()
        + b"\0"
        + episode.encode()
        + b"\0"
        + str(gallery_size).encode()
        + b"\0"
        + str(shot).encode()
        + b"\0"
        + role.encode()
        + b"\0"
        + identity_token.encode()
        + b"\0"
        + sample_token.encode(),
        hashlib.sha256,
    ).hexdigest()
    primary_query_event = None
    bootstrap_cluster = None
    if role in {"KNOWN_QUERY", "UNKNOWN_QUERY"}:
        query_scope = primary_query_scope or episode
        primary_query_event = hmac.new(
            key,
            b"CVI_PRIMARY_QUERY_EVENT_V1\0"
            + protocol.encode()
            + b"\0"
            + query_scope.encode()
            + b"\0"
            + role.encode()
            + b"\0"
            + identity_token.encode()
            + b"\0"
            + sample_token.encode(),
            hashlib.sha256,
        ).hexdigest()
        bootstrap_cluster = hmac.new(
            key,
            b"CVI_BOOTSTRAP_IDENTITY_CLUSTER_V1\0"
            + protocol.encode()
            + b"\0"
            + identity_token.encode(),
            hashlib.sha256,
        ).hexdigest()
    uses[sample_token].append(
        {
            "protocol": protocol,
            "episode": episode,
            "gallery_size": gallery_size,
            "shot": shot,
            "role": role,
            "event_token": event,
            "primary_query_event_token": primary_query_event,
            "bootstrap_cluster_token": bootstrap_cluster,
        }
    )


def _unique_components(samples: list[PublicSplitSample], component_by_sample: dict[str, _Component]) -> list[_Component]:
    grouped: dict[str, list[PublicSplitSample]] = defaultdict(list)
    for sample in samples:
        grouped[component_by_sample[sample.sample_token].token].append(sample)
    return [
        _Component(token, tuple(sorted(members, key=lambda item: item.sample_token)))
        for token, members in grouped.items()
    ]


def _component_representative(component: _Component, key: bytes, domain: bytes) -> PublicSplitSample:
    originals = [sample for sample in component.samples if sample.source_variant == "original"]
    if not originals:
        raise ValueError("split component has no original sample")
    return _rank_samples(originals, key, domain)[0]


def _rank_samples(samples: Iterable[PublicSplitSample], key: bytes, domain: bytes) -> list[PublicSplitSample]:
    return sorted(samples, key=lambda item: (_hmac_digest(key, domain, item.sample_token), item.sample_token))


def _rank_tokens(tokens: Iterable[str], key: bytes, domain: bytes) -> list[str]:
    return sorted(set(tokens), key=lambda token: (_hmac_digest(key, domain, token), token.encode("utf-8")))


def _hmac_digest(key: bytes, domain: bytes, token: str) -> bytes:
    return hmac.new(key, b"CVI_RANK_V1\0" + domain + b"\0" + token.encode("ascii"), hashlib.sha256).digest()


def _derive_keys(secret: bytes, evidence_root: str) -> dict[str, bytes]:
    master = hmac.new(secret, b"CVI_SPLIT_MASTER_V1\0" + bytes.fromhex(evidence_root), hashlib.sha256).digest()
    return {
        domain: hmac.new(master, b"CVI_SPLIT_KEY_V1\0" + domain.encode("ascii"), hashlib.sha256).digest()
        for domain in (
            "identity_roles",
            "development_episodes",
            "calibration_panels",
            "sequence_roles",
            "frame_roles",
            "bootstrap",
        )
    }


def _identity_matches(samples: list[PublicSplitSample], dataset: str, splits: set[str | None]) -> bool:
    return bool(samples) and all(sample.dataset_name == dataset and sample.original_split in splits for sample in samples)


def _official_lane(sample: PublicSplitSample) -> str:
    if sample.dataset_name == "yt-bb-dog" and sample.original_split in {
        "train",
        "test",
    }:
        return f"YT_{sample.original_split.upper()}"
    if sample.dataset_name == "dogfacenet224" and sample.original_split in {
        "train",
        "test",
    }:
        return f"DOGFACE_{sample.original_split.upper()}"
    if sample.dataset_name == "mpdd":
        if sample.original_split in {"train", "val"}:
            return "MPDD_TRAIN_DOMAIN"
        if sample.original_split in {"query", "gallery"}:
            return "MPDD_EXTERNAL_TEST"
    if sample.dataset_name == "sibetan" and sample.original_split is None:
        return (
            "SIBETAN_EXTERNAL_KNOWN"
            if sample.in_no_mono_subset is True
            else "SIBETAN_EXTERNAL_UNKNOWN"
        )
    raise ValueError("sample has no supported official dataset lane")


def _model_access(dataset: str, role: str) -> str:
    if role == "YT_FIT":
        return "MODEL_TRAINING"
    if role == "YT_DEVELOPMENT":
        return "MODEL_SELECTION"
    if role.startswith("YT_CALIBRATION"):
        return "DECISION_CALIBRATION_ONLY"
    if role.startswith("YT_TEST"):
        return "SEALED_FINAL_TEST"
    if dataset in {"mpdd", "sibetan"}:
        return "SEALED_EXTERNAL_ZERO_SHOT"
    if role.startswith("DOGFACE"):
        return "SEPARATE_FACE_ONLY_LANE"
    raise ValueError("unknown model access role")


def _assert_assignment_is_label_free(payload: Any) -> None:
    forbidden = {"source_sample_id", "dataset_identity_id", "sequence_id", "raw_frame_index", "original_split", "score"}
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            overlap = forbidden & set(value)
            if overlap:
                raise RuntimeError(f"assignment contains forbidden label fields: {sorted(overlap)}")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(payload)


def _protocol_cohort_summary(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cohorts: dict[tuple[str, str, int, int, str], set[str]] = defaultdict(set)
    for record in records:
        for use in record["uses"]:
            if use["role"] in {"KNOWN_QUERY", "UNKNOWN_QUERY"}:
                cohorts[
                    (
                        use["protocol"],
                        use["episode"],
                        use["gallery_size"],
                        use["shot"],
                        use["role"],
                    )
                ].add(record["identity_token"])
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
        for (
            protocol,
            episode,
            gallery_size,
            shot,
            role,
        ), tokens in sorted(cohorts.items())
    ]


def _bindings(values: tuple[tuple[str, str], ...], *, require_all: bool) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError("hash bindings must be a non-empty tuple")
    names = []
    for name, digest in values:
        _text(name, "binding name", 128)
        _sha256(digest, name)
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("hash bindings must be key-sorted and unique")
    if require_all and tuple(names) != _REQUIRED_EVIDENCE_BINDINGS:
        raise ValueError("frozen evidence binding set differs")


def _pair_tuple(values: list[Any]) -> tuple[tuple[str, str], ...]:
    result = []
    for item in values:
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError("hash binding entries must be two-item lists")
        result.append((item[0], item[1]))
    return tuple(result)


def _edge_key(edge: PublicSplitEvidenceEdge) -> tuple[str, str, str, str]:
    return (
        edge.left_sample_token,
        edge.right_sample_token,
        edge.relation.value,
        edge.evidence_token,
    )


def _use_key(value: dict[str, Any]) -> tuple[str, str, int, int, str, str]:
    return (
        value["protocol"],
        value["episode"],
        value["gallery_size"],
        value["shot"],
        value["role"],
        value["event_token"],
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _secret(value: bytes) -> None:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError("protected split secret must be exactly 32 bytes")


def _sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != _HEX_SHA256_LENGTH:
        raise ValueError(f"{name} must be lowercase SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase SHA-256")


def _text(value: object, name: str, maximum: int) -> None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be bounded non-empty text")
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must be canonical text")


def _exact_keys(payload: object, expected: set[str], name: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{name} fields differ")
