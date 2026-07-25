"""Assemble a FrozenPublicSplitEvidenceGraph from the source bundle and
duplicate evidence artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cvi.provenance import content_sha256
from cvi.protected_public_split import (
    EvidenceRelation,
    FrozenPublicSplitEvidenceGraph,
    PublicSplitEvidenceEdge,
    PublicSplitSourceBundle,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_dependency_edges(
    source: PublicSplitSourceBundle,
) -> list[PublicSplitEvidenceEdge]:
    samples_by_id = {s.source_sample_id: s for s in source.samples}
    edges: list[PublicSplitEvidenceEdge] = []
    for sample in source.samples:
        if sample.paired_source_sample_id is None:
            continue
        original = samples_by_id.get(sample.paired_source_sample_id)
        if original is None:
            raise ValueError(
                f"paired source not found: {sample.paired_source_sample_id}"
            )
        left, right = sorted((sample.sample_token, original.sample_token))
        evidence_token = _sha256(
            "dependency\x00" + left + "\x00" + right
        )
        edges.append(
            PublicSplitEvidenceEdge(
                left_sample_token=left,
                right_sample_token=right,
                relation=EvidenceRelation.DEPENDENCY,
                evidence_token=evidence_token,
            )
        )
    return edges


def _build_exact_duplicate_edges(
    source: PublicSplitSourceBundle,
    phash_binding_path: Path | None,
    phash_evidence_path: Path | None,
) -> list[PublicSplitEvidenceEdge]:
    if phash_binding_path is None or phash_evidence_path is None:
        return []
    binding_payload = json.loads(phash_binding_path.read_text())
    evidence_payload = json.loads(phash_evidence_path.read_text())
    bindings: list[dict] = binding_payload.get("binding", {}).get(
        "bindings", []
    )
    opaque_to_source: dict[str, str] = {
        entry["opaque_sample_id"]: entry["source_sample_id"]
        for entry in bindings
    }
    samples_by_source: dict[str, str] = {
        s.source_sample_id: s.sample_token for s in source.samples
    }

    evidence: dict = evidence_payload.get("evidence", {})
    if not isinstance(evidence, dict):
        return []

    edges: list[PublicSplitEvidenceEdge] = []
    groups = evidence.get("exact_pixel_groups", [])
    for group in groups:
        opaque_ids: list[str] = group.get("opaque_sample_ids", [])
        if len(opaque_ids) < 2:
            continue
        tokens: list[str] = []
        for oid in opaque_ids:
            source_id = opaque_to_source.get(oid)
            if source_id is None:
                continue
            token = samples_by_source.get(source_id)
            if token is None:
                continue
            tokens.append(token)
        if len(tokens) < 2:
            continue
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                left, right = sorted((tokens[i], tokens[j]))
                evidence_token = _sha256(
                    "exact\x00" + left + "\x00" + right
                )
                edges.append(
                    PublicSplitEvidenceEdge(
                        left_sample_token=left,
                        right_sample_token=right,
                        relation=EvidenceRelation.EXACT_CONFIRMED,
                        evidence_token=evidence_token,
                    )
                )
    return edges


def _build_geometric_edges(
    source: PublicSplitSourceBundle,
    phash_binding_path: Path | None,
    phash_evidence_path: Path | None,
) -> list[PublicSplitEvidenceEdge]:
    if phash_binding_path is None or phash_evidence_path is None:
        return []
    binding_payload = json.loads(phash_binding_path.read_text())
    evidence_payload = json.loads(phash_evidence_path.read_text())
    bindings: list[dict] = binding_payload.get("binding", {}).get(
        "bindings", []
    )
    opaque_to_source: dict[str, str] = {
        entry["opaque_sample_id"]: entry["source_sample_id"]
        for entry in bindings
    }
    samples_by_source: dict[str, str] = {
        s.source_sample_id: s.sample_token for s in source.samples
    }
    samples_by_token: dict[str, ...] = {
        s.sample_token: s for s in source.samples
    }

    evidence: dict = evidence_payload.get("evidence", {})
    if not isinstance(evidence, dict):
        return []
    edges: list[PublicSplitEvidenceEdge] = []
    for candidate in evidence.get("candidates", []):
        left_oid = candidate.get("left_opaque_sample_id")
        right_oid = candidate.get("right_opaque_sample_id")
        left_source = opaque_to_source.get(left_oid)
        right_source = opaque_to_source.get(right_oid)
        if left_source is None or right_source is None:
            continue
        left_token = samples_by_source.get(left_source)
        right_token = samples_by_source.get(right_source)
        if left_token is None or right_token is None:
            continue
        left_sample = samples_by_token[left_token]
        right_sample = samples_by_token[right_token]
        if (
            left_sample.identity_token == right_sample.identity_token
        ):
            continue
        left_token, right_token = sorted((left_token, right_token))
        evidence_token = _sha256(
            "geometric\x00" + left_token + "\x00" + right_token
        )
        edges.append(
            PublicSplitEvidenceEdge(
                left_sample_token=left_token,
                right_sample_token=right_token,
                relation=EvidenceRelation.GEOMETRIC_CONFIRMED,
                evidence_token=evidence_token,
            )
        )
    seen: set[tuple[str, str]] = set()
    unique_edges: list[PublicSplitEvidenceEdge] = []
    for edge in edges:
        key = (edge.left_sample_token, edge.right_sample_token)
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)
    return unique_edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument(
        "--phash-evidence", type=Path, default=None
    )
    parser.add_argument(
        "--phash-binding", type=Path, default=None
    )
    parser.add_argument(
        "--include-geometric",
        action="store_true",
        help="include GEOMETRIC_CONFIRMED edges from MIH candidates",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_payload = json.loads(args.source_bundle.read_text())
    source = PublicSplitSourceBundle.from_dict(source_payload)

    edges: list[PublicSplitEvidenceEdge] = []

    dep_edges = _build_dependency_edges(source)
    edges.extend(dep_edges)

    exact_edges = _build_exact_duplicate_edges(
        source, args.phash_binding, args.phash_evidence
    )
    edges.extend(exact_edges)

    geometric_edges = []
    if args.include_geometric:
        geometric_edges = _build_geometric_edges(
            source, args.phash_binding, args.phash_evidence
        )
    edges.extend(geometric_edges)

    sorted_edges = tuple(
        sorted(
            edges,
            key=lambda e: (
                e.left_sample_token,
                e.right_sample_token,
                e.relation.value,
                e.evidence_token,
            ),
        )
    )

    graph = FrozenPublicSplitEvidenceGraph(
        evidence_bindings=source.evidence_bindings,
        edges=sorted_edges,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            graph.to_dict(), ensure_ascii=False, sort_keys=True, indent=2
        )
        + "\n"
    )

    print(
        json.dumps(
            {
                "status": "CREATED",
                "graph_sha256": graph.graph_sha256,
                "edge_count": len(sorted_edges),
                "dependency_edges": len(dep_edges),
                "exact_duplicate_edges": len(exact_edges),
                "geometric_edges": len(geometric_edges),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
