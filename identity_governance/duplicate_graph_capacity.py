"""Allocation-free capacity analysis for a frozen duplicate dependency graph."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from identity_governance.protected_public_split import (
    EvidenceRelation,
    FrozenPublicSplitEvidenceGraph,
    ProtectedPublicSplitPolicy,
    PublicSplitSample,
    PublicSplitSourceBundle,
)
from foundation.provenance import content_sha256


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            self.parent[larger] = smaller


def analyze_duplicate_graph_capacity(
    *,
    source: PublicSplitSourceBundle,
    graph: FrozenPublicSplitEvidenceGraph,
    policy: ProtectedPublicSplitPolicy,
) -> dict[str, Any]:
    """Return a quota upper bound without assigning any identity to a role."""

    if source.evidence_bindings != graph.evidence_bindings:
        raise ValueError("capacity source and graph evidence bindings differ")
    if policy != ProtectedPublicSplitPolicy():
        raise ValueError("capacity analysis requires the fixed split policy")
    samples = {item.sample_token: item for item in source.samples}
    dsu = _DisjointSet(samples)
    union_relations = {
        EvidenceRelation.EXACT_CONFIRMED,
        EvidenceRelation.GEOMETRIC_CONFIRMED,
        EvidenceRelation.DEPENDENCY,
        EvidenceRelation.REVIEW_CONFIRMED,
    }
    for edge in graph.edges:
        if edge.left_sample_token not in samples or edge.right_sample_token not in samples:
            raise ValueError("capacity graph references an unknown sample")
        if edge.relation in union_relations:
            dsu.union(edge.left_sample_token, edge.right_sample_token)
    component_samples: dict[str, list[PublicSplitSample]] = defaultdict(list)
    for sample in source.samples:
        component_samples[dsu.find(sample.sample_token)].append(sample)

    identities = {item.identity_token for item in source.samples}
    identity_dsu = _DisjointSet(identities)
    for members in component_samples.values():
        component_identities = sorted({item.identity_token for item in members})
        for identity in component_identities[1:]:
            identity_dsu.union(component_identities[0], identity)
    blocks: dict[str, set[str]] = defaultdict(set)
    for identity in identities:
        blocks[identity_dsu.find(identity)].add(identity)

    samples_by_identity: dict[str, list[PublicSplitSample]] = defaultdict(list)
    for sample in source.samples:
        samples_by_identity[sample.identity_token].append(sample)
    lane_by_identity = {
        identity: _identity_lane(identity_samples)
        for identity, identity_samples in samples_by_identity.items()
    }
    cross_lane_blocks = {
        root: members
        for root, members in blocks.items()
        if len({lane_by_identity[identity] for identity in members}) > 1
    }
    quarantined = {
        identity for members in cross_lane_blocks.values() for identity in members
    }
    total_by_lane = Counter(lane_by_identity.values())
    available_by_lane = Counter(
        lane for identity, lane in lane_by_identity.items() if identity not in quarantined
    )
    required = {
        "DOGFACE_TEST": policy.dogface_test_identities,
        "DOGFACE_TRAIN": policy.dogface_train_identities,
        "MPDD_TEST": policy.mpdd_test_identities,
        "MPDD_TRAIN": policy.mpdd_train_identities,
        "SIBETAN": policy.sibetan_identities,
        "YT_TEST": policy.yt_official_test_identities,
        "YT_TRAIN": policy.yt_minimum_eligible_train_identities,
    }
    checks = {
        lane: {
            "available_identity_upper_bound": available_by_lane[lane],
            "required_identity_count": count,
            "passes_component_capacity_upper_bound": available_by_lane[lane] >= count,
        }
        for lane, count in sorted(required.items())
    }
    failed = sorted(
        lane for lane, item in checks.items()
        if not item["passes_component_capacity_upper_bound"]
    )
    component_sizes = [len(members) for members in component_samples.values()]
    block_sizes = [len(members) for members in blocks.values()]
    report = {
        "schema_version": "cvi.duplicate_graph_component_capacity.v1",
        "source_bundle_sha256": source.bundle_sha256,
        "graph_sha256": graph.graph_sha256,
        "split_policy_sha256": policy.policy_sha256,
        "sample_count": len(source.samples),
        "identity_count": len(identities),
        "dependency_component_count": len(component_samples),
        "largest_dependency_component_sample_count": max(component_sizes),
        "allocation_block_count": len(blocks),
        "largest_allocation_block_identity_count": max(block_sizes),
        "cross_lane_block_count": len(cross_lane_blocks),
        "largest_cross_lane_block_identity_count": max(
            (len(value) for value in cross_lane_blocks.values()), default=0
        ),
        "quarantined_identity_count": len(quarantined),
        "identity_counts_by_lane": dict(sorted(total_by_lane.items())),
        "available_identity_upper_bounds_by_lane": dict(
            sorted(available_by_lane.items())
        ),
        "quota_checks": checks,
        "failed_quota_lanes": failed,
        "status": (
            "COMPONENT_CAPACITY_UPPER_BOUND_FAILED"
            if failed else "COMPONENT_CAPACITY_UPPER_BOUND_PASSED_NOT_ALLOCATION"
        ),
        "interpretation": (
            "PREALLOCATION_COMPONENT_CAPACITY_UPPER_BOUND_ONLY_NOT_SPLIT_"
            "ASSIGNMENT_OR_MODEL_EVIDENCE"
        ),
    }
    report["report_sha256"] = content_sha256(report)
    return report


def _identity_lane(samples: list[PublicSplitSample]) -> str:
    lanes = {_sample_lane(item) for item in samples}
    if len(lanes) != 1:
        raise ValueError("one declared identity crosses official lanes")
    return next(iter(lanes))


def _sample_lane(sample: PublicSplitSample) -> str:
    if sample.dataset_name == "yt-bb-dog":
        return "YT_TRAIN" if sample.original_split == "train" else "YT_TEST"
    if sample.dataset_name == "dogfacenet224":
        return "DOGFACE_TRAIN" if sample.original_split == "train" else "DOGFACE_TEST"
    if sample.dataset_name == "mpdd":
        return "MPDD_TRAIN" if sample.original_split in {"train", "val"} else "MPDD_TEST"
    if sample.dataset_name == "sibetan":
        return "SIBETAN"
    raise ValueError("unsupported capacity-analysis dataset")
