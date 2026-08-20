"""Strict Full128 successor-family normalization into ordered figure chapters."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.full128_successor_reporting import (
    Full128SuccessorEvaluationError,
    validate_public_successor_evaluation_report,
)
from shared.foundation.provenance import content_sha256
from visualization.contracts import FigureContractError, FigureData, SourceBinding
from visualization.privacy import PublicationScope

_SCOPES = ("DEV", "CAL", "EXPOSED_DIAGNOSTIC")
_VARIANT_SPECS = (
    ("B0-FV", "Classical PCA baseline", ("B0-FV",)),
    ("B1", "Scratch masked GAP", ("B1", "B1-FV")),
    ("B2", "ImageNet masked GAP", ("B2", "B2-FV")),
    ("B3", "Frozen DINOv2 occupancy probe", ("B3", "B3-FV")),
    ("B4-U0", "Zero-update adapter control", ("B4-U0", "B4-U0-FV")),
    ("B4-U1", "Identity-blind SSL adapter", ("B4-U1", "B4-U1-FV")),
    (
        "B5-Uniform",
        "Uniform occupancy control",
        ("B5-UNIFORM", "B5-UNIFORM-FV"),
    ),
    (
        "B5-Channel",
        "Channel-gate control",
        ("B5-CHANNEL", "B5-CHANNEL-GATE-FV"),
    ),
    ("B5-Spatial", "Spatial scorer", ("B5-SPATIAL", "B5-FV")),
)
_ALIAS_BY_SUCCESSOR_ID = {
    successor_id: alias
    for alias, _, successor_ids in _VARIANT_SPECS
    for successor_id in successor_ids
}
_DESCRIPTION_BY_ALIAS = {alias: description for alias, description, _ in _VARIANT_SPECS}
_ALIAS_ORDER = {alias: index for index, (alias, _, _) in enumerate(_VARIANT_SPECS)}
_CENTRAL_SPECTRUM_ALIASES = ("B0-FV", "B2", "B3", "B4-U1", "B5-Spatial")
_MODEL_EDGES = (
    ("B0-FV", "B1"),
    ("B0-FV", "B2"),
    ("B1", "B3"),
    ("B2", "B3"),
    ("B3", "B4-U0"),
    ("B3", "B4-U1"),
    ("B4-U0", "B5-Uniform"),
    ("B4-U1", "B5-Channel"),
    ("B4-U1", "B5-Spatial"),
)


def adapt_successor_family(
    public_report: Mapping[str, Any],
    protocol_v2_bundle: Mapping[str, Any],
    gallery_query_panel_bundle: Mapping[str, Any],
    successor_inventory_bundle: Mapping[str, Any],
    *,
    target_scope: PublicationScope,
    private_report: Mapping[str, Any] | None = None,
    cache_descriptors: Sequence[Mapping[str, Any]] = (),
    asset_root: Path | None = None,
) -> tuple[FigureData, ...]:
    """Build a coherent chapter subset from exact successor/governance artifacts."""

    try:
        public = validate_public_successor_evaluation_report(public_report)
    except Full128SuccessorEvaluationError as exc:
        raise FigureContractError(str(exc)) from exc
    from evaluation.splits.face.face_gallery_query_panel import (
        validate_face_gallery_query_panel_bundle,
    )
    from evaluation.splits.face.face_identity_protocol_v2 import (
        validate_face_identity_protocol_v2_bundle,
    )
    from archive.full128.methods.face_visible import (
        validate_face_visible_successor_inventory_bundle,
    )

    protocol = validate_face_identity_protocol_v2_bundle(protocol_v2_bundle)
    panel = validate_face_gallery_query_panel_bundle(gallery_query_panel_bundle)
    inventory = validate_face_visible_successor_inventory_bundle(
        successor_inventory_bundle,
        verify_artifacts=False,
    )
    _validate_governance_closure(protocol, panel, inventory)
    from archive.full128.evaluation.full128_successors import build_authoritative_fixed_evaluation_panel

    effective_panel = build_authoritative_fixed_evaluation_panel(
        inventory, protocol, panel
    )
    if public["evaluation_panel_sha256"] != effective_panel["panel_sha256"]:
        raise FigureContractError(
            "public report and governance evaluation panel differ"
        )
    aliases = _publication_aliases(public["candidates"])
    selected_id = public["dev_selection_receipt"]["selected_successor_id"]
    if selected_id not in aliases:
        raise FigureContractError("selected successor is absent from public candidates")
    bindings = _source_bindings(public, protocol, panel, inventory)
    figures = [
        _evidence_ladder(public, bindings),
        _source_atlas(inventory, bindings),
        _census_availability(protocol, panel, inventory, bindings),
        _role_dependency_closure(protocol, bindings),
        _sample_formation_flow(inventory, bindings),
        _mask_qa_availability(inventory, bindings),
        _model_dataflow(public, bindings, cache_count=len(cache_descriptors)),
        _representation_pipeline(public, bindings),
        _training_evaluation_protocol(public, protocol, panel, bindings),
    ]
    if cache_descriptors:
        figures.append(
            _embedding_diagnostics(
                public,
                inventory,
                effective_panel,
                cache_descriptors,
                aliases,
                bindings,
            )
        )
    else:
        figures.append(_embedding_evidence_status(bindings))
    figures.extend(
        (
            _gallery_composition(public, selected_id, bindings),
            _rank_distributions(public, aliases, selected_id, bindings),
        )
    )
    if private_report is not None:
        if target_scope is not PublicationScope.PRIVATE:
            raise PermissionError(
                "private successor reports require private publication scope"
            )
        if asset_root is None:
            raise ValueError(
                "private Q/K/V contact sheets require an explicit asset root"
            )
        figures.append(
            _private_ranked_qkv(
                public,
                private_report,
                inventory,
                selected_id=selected_id,
                asset_root=asset_root,
                bindings=bindings,
            )
        )
    elif asset_root is not None and target_scope is not PublicationScope.PRIVATE:
        raise ValueError(
            "an asset root is accepted only for private successor publication"
        )
    else:
        figures.append(_private_qkv_evidence_status(bindings))
    figures.extend(
        (
            _primary_results(public, aliases, selected_id, bindings),
            _ablation_control_comparisons(public, aliases, selected_id, bindings),
            _robustness_shortcut_ledger(public, aliases, bindings),
            _runtime_device_assessment(bindings),
            _evidence_release_ledger(
                public,
                bindings,
                cache_count=len(cache_descriptors),
                private_included=private_report is not None,
            ),
        )
    )
    return tuple(figures)


def _validate_governance_closure(
    protocol: Mapping[str, Any],
    panel: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    panel_body = panel["panel"]
    protocol_body = protocol["protocol"]
    binding = inventory["source_binding"]
    if (
        panel_body["source_protocol_sha256"] != protocol["protocol_sha256"]
        or panel_body["source_protocol_bundle_sha256"] != protocol["bundle_sha256"]
        or binding.get("face_protocol_v2_sha256") != protocol["protocol_sha256"]
        or binding.get("face_protocol_v2_bundle_sha256") != protocol["bundle_sha256"]
        or binding.get("gallery_query_panel_sha256") != panel["panel_sha256"]
        or binding.get("gallery_query_panel_bundle_sha256") != panel["bundle_sha256"]
        or protocol_body["score_bearing_bytes_used_for_role_allocation"] is not False
        or panel_body["score_inputs_used"] is not False
    ):
        raise FigureContractError("successor governance-v2 source closure differs")


def _source_bindings(
    public: Mapping[str, Any],
    protocol: Mapping[str, Any],
    panel: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> tuple[SourceBinding, ...]:
    return (
        SourceBinding(
            "successor-public-report",
            public["schema_version"],
            content_sha256(public),
        ),
        SourceBinding(
            "face-protocol-v2",
            protocol["schema_version"],
            content_sha256(protocol),
        ),
        SourceBinding(
            "gallery-query-panel",
            panel["schema_version"],
            content_sha256(panel),
        ),
        SourceBinding(
            "successor-inventory",
            inventory["schema_version"],
            content_sha256(inventory),
        ),
    )


def _evidence_ladder(
    public: Mapping[str, Any], bindings: tuple[SourceBinding, ...]
) -> FigureData:
    aliases = _publication_aliases(public["candidates"])
    selected_alias = aliases[public["dev_selection_receipt"]["selected_successor_id"]]
    reported = set(aliases.values())
    positions = {
        "B0-FV": (0, 0.50),
        "B1": (1, 0.72),
        "B2": (1, 0.28),
        "B3": (2, 0.50),
        "B4-U0": (3, 0.72),
        "B4-U1": (3, 0.28),
        "B5-Uniform": (4, 0.78),
        "B5-Channel": (4, 0.50),
        "B5-Spatial": (4, 0.22),
    }
    variants = []
    for alias, description, _ in _VARIANT_SPECS:
        column, row = positions[alias]
        variants.append(
            {
                "alias": alias,
                "description": description,
                "status": "GO" if alias == selected_alias else "NO_GO",
                "reported": alias in reported,
                "column": column,
                "row": row,
            }
        )
    for alias in sorted(reported - set(positions)):
        variants.append(
            {
                "alias": alias,
                "description": "Report candidate",
                "status": "GO" if alias == selected_alias else "NO_GO",
                "reported": True,
                "column": 5,
                "row": 0.50,
            }
        )
    return _figure(
        "00_evidence_ladder",
        "model_ladder",
        "Full128 successor model ladder",
        "Fixed model-family topology with DEV selection outcomes and explicit evidence boundaries.",
        public["limitations"],
        bindings,
        {
            "variants": variants,
            "edges": [
                {"source": source, "target": target} for source, target in _MODEL_EDGES
            ],
            "boundaries": [
                {"label": "DEV", "detail": "selection", "status": "GO/NO_GO"},
                {
                    "label": "CAL",
                    "detail": "reporting only",
                    "status": "NO_SELECTION",
                },
                {
                    "label": "EXPOSED",
                    "detail": "diagnostic, not final",
                    "status": "BOUNDARY",
                },
            ],
        },
    )


def _source_atlas(
    inventory: Mapping[str, Any], bindings: tuple[SourceBinding, ...]
) -> FigureData:
    populations = _inventory_populations(inventory)
    counts: dict[str, Counter[str]] = {}
    for lane, records in populations.items():
        for record in records:
            dataset = _text(record.get("dataset_name"), "inventory dataset_name")
            counts.setdefault(dataset, Counter())[lane] += 1
    rows = []
    for dataset in sorted(counts):
        lane_counts = counts[dataset]
        for group_index, lane in enumerate(("successor", "auxiliary", "terminal")):
            if lane_counts[lane]:
                rows.append(
                    {
                        "label": f"{dataset} | {lane}",
                        "count": lane_counts[lane],
                        "group_index": group_index,
                    }
                )
    if not rows:
        raise FigureContractError("successor inventory has no persisted source rows")
    population_total = sum(row["count"] for row in rows)
    expected_total = inventory["inventory"]["coverage"]["route_plan_sample_count"]
    if population_total != expected_total:
        raise FigureContractError("source atlas and persisted route count differ")
    return _figure(
        "01_source_provenance",
        "census",
        "Governed dataset and source atlas",
        "Aggregate persisted source counts by successor, identity-free auxiliary, and terminal lanes.",
        (
            "Only aggregate persisted counts are published; no sample paths, labels, tokens, or media are exposed.",
        ),
        bindings,
        {
            "rows": rows,
            "x_label": "Persisted source records",
            "x_max": max(1, math.ceil(max(row["count"] for row in rows) * 1.08)),
        },
    )


def _sample_formation_flow(
    inventory: Mapping[str, Any], bindings: tuple[SourceBinding, ...]
) -> FigureData:
    body = inventory["inventory"]
    coverage = body["coverage"]
    policy = body["crop_policy"]
    populations = _inventory_populations(inventory)
    records = [record for lane in populations.values() for record in lane]
    artifact_count = sum(record.get("artifact") is not None for record in records)
    if (
        policy.get("source") != "EXISTING_FULL128_MATERIALIZATION_ONLY"
        or policy.get("recrop_permitted") is not False
    ):
        raise FigureContractError("successor sample formation crop policy differs")
    nodes = [
        {
            "label": f"Persisted route records\n{coverage['route_plan_sample_count']:,}",
            "layer": 0,
            "group_index": 0,
        },
        {
            "label": "Existing Full128 materialization\nrecrop forbidden",
            "layer": 1,
            "group_index": 1,
        },
        {
            "label": f"RGB + mask bindings\n{artifact_count:,}",
            "layer": 2,
            "group_index": 2,
        },
        {
            "label": "Governance-v2 join\nidentity and dependency roles",
            "layer": 3,
            "group_index": 3,
        },
        {
            "label": f"Successor\n{coverage['successor_sample_count']:,}",
            "layer": 4,
            "group_index": 0,
        },
        {
            "label": f"Auxiliary\n{coverage['identity_free_auxiliary_sample_count']:,}",
            "layer": 4,
            "group_index": 2,
        },
        {
            "label": f"Terminal\n{coverage['terminal_exclusion_count']:,}",
            "layer": 4,
            "group_index": 4,
        },
    ]
    return _figure(
        "04_governance_panel",
        "architecture",
        "Sample formation and preprocessing flow",
        "Persisted sample formation reuses bound Full128 RGB/mask artifacts and then assigns governed lanes.",
        (
            "The inventory proves artifact bindings and recrop policy, not qualitative crop or segmentation quality.",
        ),
        bindings,
        {
            "nodes": nodes,
            "edges": [
                {"source": 0, "target": 1, "label": ""},
                {"source": 1, "target": 2, "label": ""},
                {"source": 2, "target": 3, "label": ""},
                *[{"source": 3, "target": target, "label": ""} for target in (4, 5, 6)],
            ],
        },
    )


def _mask_qa_availability(
    inventory: Mapping[str, Any], bindings: tuple[SourceBinding, ...]
) -> FigureData:
    populations = _inventory_populations(inventory)
    records = [record for lane in populations.values() for record in lane]
    bound = [record for record in records if record.get("artifact") is not None]
    mask_bound = sum(
        isinstance(record["artifact"], Mapping)
        and isinstance(record["artifact"].get("full_mask_sha256"), str)
        for record in bound
    )
    if mask_bound != len(bound):
        raise FigureContractError("materialized inventory record lacks a mask binding")
    usable = sum(record.get("state") == "USABLE" for record in records)
    return _status_figure(
        "05_score_distributions",
        "ladder",
        "Mask and segmentation QA availability",
        "Measured inventory availability with qualitative evidence explicitly withheld as unavailable.",
        ("Artifact counts do not establish segmentation accuracy or visual quality.",),
        bindings,
        headline="PARTIAL",
        rows=(
            _status_row(
                "Inventory records",
                "AVAILABLE",
                "Persisted governed population denominator.",
                count=len(records),
            ),
            _status_row(
                "RGB + mask bindings",
                "AVAILABLE",
                "Both content digests are present in materialized inventory records.",
                count=mask_bound,
            ),
            _status_row(
                "Usable records",
                "AVAILABLE",
                "Inventory state only; not a qualitative mask judgment.",
                count=usable,
            ),
            _status_row(
                "No artifact binding",
                "AVAILABLE",
                "Measured absence in persisted inventory records.",
                count=len(records) - len(bound),
            ),
            _status_row(
                "Qualitative segmentation panel",
                "UNAVAILABLE",
                "UNAVAILABLE_NO_PUBLIC_QUALITATIVE_ARTIFACT",
            ),
            _status_row(
                "Pixel-level mask quality",
                "UNASSESSED",
                "UNASSESSED_NO_MASK_QA_REPORT",
            ),
        ),
    )


def _census_availability(
    protocol: Mapping[str, Any],
    panel: Mapping[str, Any],
    inventory: Mapping[str, Any],
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    rows = []
    census = protocol["census"]
    role_labels = {
        "FIT": "Identity: FIT",
        "DEV": "Identity: DEV",
        "CAL": "Identity: CAL",
        "EXPOSED_DIAGNOSTIC": "Identity: EXPOSED",
        "EXCLUDED_UNSAFE_COMPONENT": "Identity: UNSAFE EXCLUDED",
    }
    for index, (role, count) in enumerate(census["identity_role_counts"].items()):
        rows.append(
            {
                "label": role_labels[role],
                "count": count,
                "group_index": index,
            }
        )
    coverage = inventory["inventory"]["coverage"]
    coverage_labels = (
        ("successor_sample_count", "Successor samples"),
        ("identity_free_auxiliary_sample_count", "Identity-free auxiliary samples"),
        ("terminal_exclusion_count", "Terminal exclusions"),
    )
    for index, (key, label) in enumerate(coverage_labels, start=6):
        rows.append(
            {
                "label": label,
                "count": coverage[key],
                "group_index": index,
            }
        )
    rows.append(
        {
            "label": "Common K5-feasible identities",
            "count": panel["census"]["common_k5_feasible_identity_count"],
            "group_index": 2,
        }
    )
    maximum = max(row["count"] for row in rows)
    return _figure(
        "02_census_availability",
        "census",
        "Population census and availability",
        "Aggregate governance roles, successor availability, and fixed-panel feasibility.",
        (
            "Counts describe the persisted governed population, not a deployment population.",
        ),
        bindings,
        {"rows": rows, "x_label": "Count", "x_max": max(1, math.ceil(maximum * 1.1))},
    )


def _role_dependency_closure(
    protocol: Mapping[str, Any],
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    counts = protocol["census"]["identity_role_counts"]
    nodes = [
        {"label": "Audited public sources", "layer": 0, "group_index": 0},
        {"label": "Dependency closure", "layer": 1, "group_index": 1},
        *[
            {"label": f"{role}: {counts[role]}", "layer": 2, "group_index": index + 2}
            for index, role in enumerate(
                (
                    "FIT",
                    "DEV",
                    "CAL",
                    "EXPOSED_DIAGNOSTIC",
                    "EXCLUDED_UNSAFE_COMPONENT",
                )
            )
        ],
        {"label": "Nested K1/K3/K5 panel", "layer": 3, "group_index": 1},
    ]
    edges = [
        {"source": 0, "target": 1, "label": ""},
        *[{"source": 1, "target": index, "label": ""} for index in range(2, 7)],
        *[{"source": index, "target": 7, "label": ""} for index in range(3, 6)],
    ]
    return _figure(
        "03_role_dependency_closure",
        "architecture",
        "Role and dependency closure",
        "Governance-v2 closes duplicate dependencies before score-blind role and panel allocation.",
        (
            "The panel declares dependency disjointness but does not claim cross-session verification.",
        ),
        bindings,
        {"nodes": nodes, "edges": edges},
    )


def _model_dataflow(
    public: Mapping[str, Any],
    bindings: tuple[SourceBinding, ...],
    *,
    cache_count: int,
) -> FigureData:
    nodes = [
        {"label": "Full crop", "layer": 0, "group_index": 0},
        {
            "label": f"{len(public['candidates'])} variants",
            "layer": 1,
            "group_index": 1,
        },
        {
            "label": f"128D L2 ({cache_count} caches)",
            "layer": 2,
            "group_index": 2,
        },
        {"label": "Exact cosine Q/K", "layer": 3, "group_index": 3},
        {"label": "Ranked identity V", "layer": 4, "group_index": 4},
    ]
    edges = [
        {"source": 0, "target": 1, "label": ""},
        {"source": 1, "target": 2, "label": ""},
        {"source": 2, "target": 3, "label": ""},
        {"source": 3, "target": 4, "label": ""},
    ]
    return _figure(
        "06_model_dataflow",
        "architecture",
        "Successor model dataflow",
        "The report-bound path produces one 128D representation and exact closed-set cosine retrieval.",
        (
            "This dataflow does not imply detection, tracking, open-set rejection, or deployment behavior.",
        ),
        bindings,
        {"nodes": nodes, "edges": edges},
    )


def _representation_pipeline(
    public: Mapping[str, Any], bindings: tuple[SourceBinding, ...]
) -> FigureData:
    aliases = set(_publication_aliases(public["candidates"]).values())
    nodes = [
        {"label": "Full RGB + binary mask", "layer": 0, "group_index": 0},
        {
            "label": "Classical descriptors + PCA\nB0-FV",
            "layer": 1,
            "group_index": 0,
        },
        {
            "label": "Masked ResNet GAP\nB1 / B2",
            "layer": 1,
            "group_index": 1,
        },
        {
            "label": "Frozen DINOv2 patches\n384D + occupancy",
            "layer": 1,
            "group_index": 2,
        },
        {
            "label": "Occupancy pooling\nB3",
            "layer": 2,
            "group_index": 2,
        },
        {
            "label": "Residual token adapter\nB4 U0 / U1",
            "layer": 2,
            "group_index": 3,
        },
        {
            "label": "Uniform / channel / spatial\nB5 controls",
            "layer": 2,
            "group_index": 4,
        },
        {"label": "128D projection", "layer": 3, "group_index": 1},
        {"label": "Finite float32 + L2", "layer": 4, "group_index": 2},
    ]
    recognized = sum(1 for alias in aliases if alias in _DESCRIPTION_BY_ALIAS)
    return _figure(
        "07_evaluation_protocol",
        "architecture",
        "Model internals and representation pipeline",
        "Contract-level branches for the report candidates converge on finite 128D L2 representations.",
        (
            "The diagram describes implemented representation contracts; it does not claim learned semantic regions or deployment behavior.",
        ),
        bindings,
        {
            "nodes": [
                *nodes,
                {
                    "label": f"Report candidates\n{recognized} recognized",
                    "layer": 5,
                    "group_index": 5,
                },
            ],
            "edges": [
                *[{"source": 0, "target": target, "label": ""} for target in (1, 2, 3)],
                {"source": 3, "target": 4, "label": ""},
                {"source": 3, "target": 5, "label": ""},
                {"source": 3, "target": 6, "label": ""},
                *[
                    {"source": source, "target": 7, "label": ""}
                    for source in (1, 2, 4, 5, 6)
                ],
                {"source": 7, "target": 8, "label": ""},
                {"source": 8, "target": 9, "label": ""},
            ],
        },
    )


def _training_evaluation_protocol(
    public: Mapping[str, Any],
    protocol: Mapping[str, Any],
    panel: Mapping[str, Any],
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    available_scopes = sum(
        aggregate["status"] == "AVAILABLE"
        for candidate in public["candidates"]
        for aggregate in candidate["scope_aggregates"]
    )
    return _status_figure(
        "08_cache_bindings",
        "ladder",
        "Training and evaluation protocol",
        "Evaluation boundaries are artifact-backed; training details remain unavailable without a bound training artifact.",
        (
            "No optimizer, schedule, augmentation, device, or training-step value is inferred from the evaluation report.",
        ),
        bindings,
        headline="PARTIAL",
        rows=(
            _status_row(
                "Training protocol",
                "UNAVAILABLE",
                "UNAVAILABLE_NO_TRAINING_ARTIFACT_SUPPLIED",
            ),
            _status_row(
                "Identity role allocation",
                "AVAILABLE",
                "Score-bearing bytes were not used for role allocation.",
                count=sum(protocol["census"]["identity_role_counts"].values()),
            ),
            _status_row(
                "Fixed K5-feasible cohort",
                "AVAILABLE",
                "Shared query with nested K1/K3/K5 galleries.",
                count=panel["census"]["common_k5_feasible_identity_count"],
            ),
            _status_row(
                "Available report aggregates",
                "AVAILABLE",
                "Candidate-scope aggregate records in the public report.",
                count=available_scopes,
            ),
            _status_row(
                "DEV",
                "ASSESSED",
                "MODEL_SELECTION_ONLY",
            ),
            _status_row(
                "CAL / exposed",
                "ASSESSED",
                "REPORTING_ONLY / RETROSPECTIVE_DIAGNOSTIC_NOT_FINAL",
            ),
        ),
    )


def _embedding_evidence_status(
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    return _status_figure(
        "09_embedding_spectrum_pca",
        "ladder",
        "Embedding spectrum and PCA retention",
        "Evidence-status panel emitted because no report-bound embedding cache descriptor was supplied.",
        (
            "No spectrum, PCA retention value, coordinate, or embedding statistic is inferred from aggregate retrieval results.",
        ),
        bindings,
        headline="UNAVAILABLE",
        rows=(
            _status_row(
                "Embedding caches",
                "UNAVAILABLE",
                "UNAVAILABLE_NO_CACHE_DESCRIPTOR_SUPPLIED",
            ),
            _status_row(
                "PCA spectrum",
                "UNASSESSED",
                "UNASSESSED_WITHOUT_EMBEDDINGS",
            ),
        ),
    )


def _embedding_diagnostics(
    public: Mapping[str, Any],
    inventory: Mapping[str, Any],
    effective_panel: Mapping[str, Any],
    descriptors: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str],
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    from archive.full128.evaluation.full128_successors import open_successor_embedding_cache

    candidate_hashes = {
        candidate["successor_id"]: candidate["cache_descriptor_sha256"]
        for candidate in public["candidates"]
    }
    caches = []
    seen: set[str] = set()
    for descriptor in descriptors:
        cache = open_successor_embedding_cache(
            descriptor,
            successor_inventory_bundle=inventory,
            evaluation_panel=effective_panel,
        )
        successor_id = cache.descriptor["successor_id"]
        if successor_id in seen or successor_id not in candidate_hashes:
            raise FigureContractError(
                "cache successor is duplicate or absent from report"
            )
        if (
            cache.descriptor["cache_descriptor_sha256"]
            != candidate_hashes[successor_id]
        ):
            raise FigureContractError(
                "cache descriptor and public report binding differ"
            )
        seen.add(successor_id)
        matrix = cache.load_embeddings(cache.descriptor["sample_tokens"])
        if matrix.shape[0] < 2:
            raise FigureContractError("embedding diagnostics require at least two rows")
        caches.append((successor_id, matrix, dict(descriptor)))
    caches.sort(key=lambda item: _alias_sort_key(aliases[item[0]]))
    component_count = min(32, *(min(matrix.shape) for _, matrix, _ in caches))
    if component_count < 2:
        raise FigureContractError(
            "embedding diagnostics require at least two components"
        )
    all_series = []
    for successor_id, matrix, _ in caches:
        centered = matrix.astype(np.float64) - matrix.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        variance = singular**2
        total = float(variance.sum())
        if not math.isfinite(total) or total <= 0:
            raise FigureContractError("embedding cache has zero centered variance")
        explained = variance[:component_count] / total
        all_series.append(
            {
                "label": aliases[successor_id],
                "style_index": _central_style_index(aliases[successor_id]),
                "sample_count": matrix.shape[0],
                "explained_variance": explained.tolist(),
                "cumulative_variance": np.minimum(1.0, np.cumsum(explained)).tolist(),
            }
        )
    displayed_aliases = set(_CENTRAL_SPECTRUM_ALIASES) & {
        item["label"] for item in all_series
    }
    if not displayed_aliases:
        displayed_aliases = {item["label"] for item in all_series[:5]}
    series = [item for item in all_series if item["label"] in displayed_aliases]
    cache_bindings = tuple(
        SourceBinding(
            f"successor-cache-{index + 1}",
            descriptor["schema_version"],
            content_sha256(dict(descriptor)),
        )
        for index, (_, _, descriptor) in enumerate(caches)
    )
    manifest = [
        {
            "alias": aliases[successor_id],
            "description": _DESCRIPTION_BY_ALIAS.get(
                aliases[successor_id], "Report candidate"
            ),
            "sample_count": matrix.shape[0],
            "cache_descriptor_sha256": descriptor["cache_descriptor_sha256"],
            "displayed": aliases[successor_id] in displayed_aliases,
        }
        for successor_id, matrix, descriptor in caches
    ]
    y_max = min(1.0, max(max(item["explained_variance"]) for item in series) * 1.08)
    return _figure(
        "09_embedding_spectrum_pca",
        "embedding_diagnostics",
        "Embedding spectrum and PCA retention",
        "Aggregate centered PCA spectra from fully rehashed report-bound embedding packs.",
        (
            "PCA components are cache-specific and rotationally non-semantic; no per-sample coordinates are published.",
        ),
        (*bindings, *cache_bindings),
        {
            "series": series,
            "manifest": manifest,
            "component_count": component_count,
            "variance_y_max": y_max,
        },
    )


def _gallery_composition(
    public: Mapping[str, Any],
    selected_id: str,
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    candidate = _candidate(public, selected_id)
    totals: dict[tuple[str, int], int] = {}
    for binding in candidate["gallery_bindings"]:
        key = (binding["scope"], binding["enrollment_k"])
        totals[key] = totals.get(key, 0) + binding["template_count"]
    scope_order = {scope: index for index, scope in enumerate(_SCOPES)}
    rows = [
        {
            "label": f"{_short_scope(scope)} | K{k}",
            "value": value,
            "group_index": index,
        }
        for index, ((scope, k), value) in enumerate(
            sorted(
                totals.items(), key=lambda item: (scope_order[item[0][0]], item[0][1])
            )
        )
    ]
    if not rows:
        raise FigureContractError("selected successor has no public gallery bindings")
    return _figure(
        "10_gallery_composition",
        "gallery_composition",
        "Selected-successor gallery composition",
        "Template counts aggregated by governed scope and nested enrollment rank.",
        (
            "The same governed identities can appear across nested K and scope-specific galleries.",
        ),
        bindings,
        {
            "rows": rows,
            "center_label": f"{sum(row['value'] for row in rows)}\nbound templates",
        },
    )


def _rank_distributions(
    public: Mapping[str, Any],
    aliases: Mapping[str, str],
    selected_id: str,
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    candidate = _candidate(public, selected_id)
    rank_series = []
    for aggregate in candidate["scope_aggregates"]:
        if aggregate["status"] != "AVAILABLE":
            continue
        metrics = aggregate["metrics"]
        rank_series.append(
            {
                "label": f"{aliases[selected_id]} | {aggregate['scope']}",
                "ranks": [1, 5, 10],
                "values": [metrics["Rank-1"], metrics["Rank-5"], metrics["Rank-10"]],
            }
        )
    if not rank_series:
        raise FigureContractError("selected successor has no available rank aggregates")
    return _figure(
        "11_cosine_rank_distributions",
        "score_rank_distributions",
        "Public aggregate rank distributions",
        "Cumulative rank points available in the sanitized public successor report.",
        (
            "The public report contains no cosine-score histogram, so no cosine distribution is inferred or fabricated.",
        ),
        bindings,
        {
            "rank_series": rank_series,
            "rank_ticks": [1, 5, 10],
            "rank_x_max": 10,
            "cosine_distribution": None,
        },
    )


def _private_qkv_evidence_status(
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    return _status_figure(
        "12_private_ranked_qkv",
        "ladder",
        "Ranked Q/K/V qualitative evidence",
        "Public evidence-status panel; governed private retrieval crops are not published.",
        (
            "The absence of a public contact sheet is a privacy boundary, not missing aggregate evaluation evidence.",
        ),
        bindings,
        headline="PRIVATE_NOT_PUBLISHED",
        rows=(
            _status_row(
                "Public aggregate metrics",
                "AVAILABLE",
                "Published separately without sample or identity tokens.",
            ),
            _status_row(
                "Ranked Q/K/V trace",
                "PRIVATE_NOT_PUBLISHED",
                "PRIVATE_GOVERNED_EVIDENCE_NOT_INCLUDED",
            ),
            _status_row(
                "Query / gallery crops",
                "PRIVATE_NOT_PUBLISHED",
                "NO_PUBLIC_MEDIA_DISCLOSURE",
            ),
        ),
    )


def _private_ranked_qkv(
    public: Mapping[str, Any],
    private_report: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    selected_id: str,
    asset_root: Path,
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    from archive.full128.evaluation.full128_successors import sanitize_successor_evaluation_report

    sanitized = sanitize_successor_evaluation_report(private_report)
    sanitized_comparable = {
        key: value
        for key, value in sanitized.items()
        if key not in {"schema_version", "public_report_sha256"}
    }
    public_comparable = {
        key: value
        for key, value in public.items()
        if key not in {"schema_version", "public_report_sha256"}
    }
    if sanitized_comparable != public_comparable:
        raise FigureContractError(
            "private successor report does not sanitize to public report"
        )
    candidate = next(
        item
        for item in private_report["candidates"]
        if item["successor_id"] == selected_id
    )
    traces = candidate["ranked_private_qkv_traces"]
    if not isinstance(traces, list) or not traces:
        raise FigureContractError(
            "selected successor has no private ranked Q/K/V traces"
        )
    trace = _validate_private_trace(traces[0])
    records = {
        row["sample_token"]: row
        for population in (
            inventory["inventory"]["successor_population"],
            inventory["inventory"]["terminal_exclusions"],
        )
        for row in population
    }
    root = asset_root.resolve(strict=True)
    if asset_root.is_symlink() or not root.is_dir():
        raise NotADirectoryError(root)
    query_token = trace["Q"]["sample_token"]
    query_record = records.get(query_token)
    if query_record is None:
        raise FigureContractError("private query is absent from successor inventory")
    query_image = _bound_inventory_image(query_record, root)
    candidates = []
    ranked_rows = trace["ranked_KV"][:8]
    for index, ranked in enumerate(ranked_rows):
        key_token = ranked["K"]["sample_token"]
        key_record = records.get(key_token)
        if key_record is None:
            raise FigureContractError("private key is absent from successor inventory")
        image = _bound_inventory_image(key_record, root)
        candidates.append(
            {
                **image,
                "label": f"Key/value {ranked['rank']}",
                "rank": ranked["rank"],
                "score": ranked["score"],
                "margin": (
                    None
                    if index + 1 == len(ranked_rows)
                    else ranked["score"] - ranked_rows[index + 1]["score"]
                ),
                "outcome": (
                    "relevant"
                    if ranked["V"]["registered_identity_id"]
                    == query_record["registered_identity_id"]
                    else "not_relevant"
                ),
            }
        )
    private_binding = SourceBinding(
        "successor-private-report",
        private_report["schema_version"],
        content_sha256(dict(private_report)),
    )
    return FigureData.create(
        figure_id="12_private_ranked_qkv",
        kind="ranked_retrieval",
        scope=PublicationScope.PRIVATE,
        title="Private ranked Q/K/V retrieval trace",
        caption="One exact-cosine private query and its ranked gallery key/value results.",
        limitations=(
            "This contact sheet contains private governed crop evidence and must not be published publicly.",
        ),
        source_bindings=(*bindings, private_binding),
        payload={
            "query": {**query_image, "label": "Query"},
            "candidates": candidates,
        },
    )


def _validate_private_trace(value: Any) -> dict[str, Any]:
    trace = _exact_object(
        value,
        {"scope", "dataset_name", "enrollment_k", "Q", "ranked_KV", "exact_cosine"},
        "private Q/K/V trace",
    )
    if (
        trace["scope"] not in _SCOPES
        or trace["dataset_name"] not in {"dogfacenet224", "mpdd"}
        or trace["enrollment_k"] not in {1, 3, 5}
        or trace["exact_cosine"] is not True
    ):
        raise FigureContractError("private Q/K/V trace contract differs")
    query_fields = set(trace["Q"]) if isinstance(trace["Q"], Mapping) else set()
    if query_fields not in ({"sample_token"}, {"sample_token", "embedding"}):
        raise FigureContractError("private query fields differ")
    query = dict(trace["Q"])
    _sha(query["sample_token"], "private query token")
    if "embedding" in query:
        _embedding(query["embedding"])
    ranked = _nonempty_array(trace["ranked_KV"], "private ranked K/V")
    scores = []
    for expected_rank, item in enumerate(ranked, 1):
        row = _exact_object(item, {"rank", "score", "K", "V"}, "private ranked row")
        if row["rank"] != expected_rank:
            raise FigureContractError("private ranks must be consecutive")
        scores.append(
            _number(row["score"], "private cosine score", minimum=-1, maximum=1)
        )
        key_fields = set(row["K"]) if isinstance(row["K"], Mapping) else set()
        if key_fields not in (
            {"winning_template_row", "sample_token"},
            {"winning_template_row", "sample_token", "embedding"},
        ):
            raise FigureContractError("private key fields differ")
        key = dict(row["K"])
        _count(key["winning_template_row"], "winning_template_row")
        _sha(key["sample_token"], "private key token")
        if "embedding" in key:
            _embedding(key["embedding"])
        result = _exact_object(
            row["V"],
            {"registered_identity_id", "template_id", "content_sha256"},
            "private value",
        )
        _text(result["registered_identity_id"], "registered identity")
        _text(result["template_id"], "template ID")
        _sha(result["content_sha256"], "value content")
    if any(left < right for left, right in pairwise(scores)):
        raise FigureContractError("private ranked scores must not increase")
    return trace


def _bound_inventory_image(record: Mapping[str, Any], root: Path) -> dict[str, str]:
    artifact = record.get("artifact")
    if not isinstance(artifact, Mapping):
        raise FigureContractError("private contact-sheet record has no crop artifact")
    path = Path(artifact["full_rgb_path"])
    if path.is_symlink():
        raise FigureContractError("private crop artifact must not be a symlink")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise FigureContractError(
            "private crop artifact escapes explicit asset root"
        ) from exc
    _sha(artifact["full_rgb_sha256"], "private crop")
    return {"path": relative, "sha256": artifact["full_rgb_sha256"]}


def _primary_results(
    public: Mapping[str, Any],
    aliases: Mapping[str, str],
    selected_id: str,
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    absolute_rows = []
    for candidate in public["candidates"]:
        alias = aliases[candidate["successor_id"]]
        for aggregate in candidate["scope_aggregates"]:
            if aggregate["status"] != "AVAILABLE":
                continue
            absolute_rows.append(
                {
                    "alias": alias,
                    "scope": aggregate["scope"],
                    "rank1": aggregate["metrics"]["Rank-1"],
                    "mrr": aggregate["metrics"]["MRR"],
                }
            )
    delta_rows = []
    for comparison in public["paired_identity_cluster_bootstrap"]:
        if comparison["scope"] != "DEV":
            continue
        for interval in comparison["intervals"]:
            if interval["metric"] not in {"Rank-1", "MRR"}:
                continue
            delta_rows.append(
                {
                    "left_alias": aliases[comparison["left_successor_id"]],
                    "right_alias": aliases[comparison["right_successor_id"]],
                    "metric": interval["metric"],
                    "estimate": interval["estimate"],
                    "lower": interval["lower_bound"],
                    "upper": interval["upper_bound"],
                }
            )
    maximum_delta = max(
        (abs(value) for row in delta_rows for value in (row["lower"], row["upper"])),
        default=0.1,
    )
    delta_limit = min(1.0, max(0.1, math.ceil(maximum_delta * 10) / 10))
    alias_legend = [
        {
            "alias": alias,
            "description": _DESCRIPTION_BY_ALIAS.get(alias, "Report candidate"),
        }
        for alias in sorted(set(aliases.values()), key=_alias_sort_key)
    ]
    return _figure(
        "13_primary_results_paired_deltas",
        "result_forest",
        "Primary aggregates and paired successor deltas",
        "Absolute Rank-1/MRR aggregates and DEV paired whole-identity bootstrap differences.",
        tuple(public["limitations"]),
        bindings,
        {
            "absolute_rows": absolute_rows,
            "delta_rows": delta_rows,
            "alias_legend": alias_legend,
            "selected_alias": aliases[selected_id],
            "delta_limit": delta_limit,
        },
    )


def _ablation_control_comparisons(
    public: Mapping[str, Any],
    aliases: Mapping[str, str],
    selected_id: str,
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    control_aliases = ("B4-U0", "B4-U1", "B5-Uniform", "B5-Channel")
    candidate_by_alias = {
        alias: successor_id for successor_id, alias in aliases.items()
    }
    paired_by_control: dict[str, tuple[Mapping[str, Any], float]] = {}
    for comparison in public["paired_identity_cluster_bootstrap"]:
        if comparison["scope"] != "DEV" or selected_id not in {
            comparison["left_successor_id"],
            comparison["right_successor_id"],
        }:
            continue
        comparator_id = (
            comparison["right_successor_id"]
            if comparison["left_successor_id"] == selected_id
            else comparison["left_successor_id"]
        )
        comparator_alias = aliases[comparator_id]
        if comparator_alias not in control_aliases:
            continue
        direction = 1.0 if comparison["left_successor_id"] == selected_id else -1.0
        paired_by_control[comparator_alias] = (comparison, direction)
    steps = []
    for alias in control_aliases:
        if alias not in candidate_by_alias:
            steps.append(
                {
                    "label": alias,
                    "detail": "UNAVAILABLE_REPORT_CANDIDATE",
                    "status": "conditional",
                }
            )
            continue
        paired_entry = paired_by_control.get(alias)
        if paired_entry is None:
            steps.append(
                {
                    "label": alias,
                    "detail": "UNAVAILABLE_PAIRED_INTERVAL",
                    "status": "conditional",
                }
            )
            continue
        comparison, direction = paired_entry
        intervals = {
            interval["metric"]: interval for interval in comparison["intervals"]
        }
        parts = []
        for metric in ("Rank-1", "MRR"):
            interval = intervals[metric]
            estimate = direction * interval["estimate"]
            lower = (
                interval["lower_bound"] if direction > 0 else -interval["upper_bound"]
            )
            upper = (
                interval["upper_bound"] if direction > 0 else -interval["lower_bound"]
            )
            parts.append(f"{metric} {estimate:+.4f} [{lower:+.4f}, {upper:+.4f}]")
        reference = intervals["Rank-1"]
        steps.append(
            {
                "label": f"{aliases[selected_id]} - {alias}",
                "detail": (
                    f"{' | '.join(parts)}; clusters={reference['cluster_count']}; "
                    f"paired queries={reference['paired_query_count']}"
                ),
                "status": "established",
            }
        )
    return _figure(
        "14_scope_interpretation",
        "ladder",
        "Ablation and control comparisons",
        "Selected-candidate DEV differences against report-listed controls using paired whole-identity intervals only.",
        (
            "A missing report candidate or paired interval remains explicitly unavailable; no independent ablation value is reconstructed.",
        ),
        bindings,
        {"steps": steps},
    )


def _robustness_shortcut_ledger(
    public: Mapping[str, Any],
    aliases: Mapping[str, str],
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    candidates = {aliases[row["successor_id"]]: row for row in public["candidates"]}
    rows = []
    for alias in ("B5-Uniform", "B5-Channel", "B5-Spatial"):
        candidate = candidates.get(alias)
        aggregate = None
        if candidate is not None:
            aggregate = next(
                (
                    item
                    for item in candidate["scope_aggregates"]
                    if item["scope"] == "DEV" and item["status"] == "AVAILABLE"
                ),
                None,
            )
        if aggregate is None:
            rows.append(
                _status_row(
                    alias,
                    "UNAVAILABLE",
                    "UNAVAILABLE_NO_DEV_REPORT_AGGREGATE",
                )
            )
        else:
            metrics = aggregate["metrics"]
            rows.append(
                _status_row(
                    alias,
                    "ASSESSED",
                    f"DEV Rank-1={metrics['Rank-1']:.4f}; MRR={metrics['MRR']:.4f}",
                    count=aggregate["query_count"],
                )
            )
    rows.extend(
        (
            _status_row(
                "Mask perturbation",
                "UNASSESSED",
                "UNASSESSED_NO_MASK_PERTURBATION_ARTIFACT",
            ),
            _status_row(
                "Background perturbation",
                "UNASSESSED",
                "UNASSESSED_NO_BACKGROUND_PERTURBATION_ARTIFACT",
            ),
        )
    )
    return _status_figure(
        "15_limitations",
        "ladder",
        "Robustness and shortcut-control ledger",
        "B5 report controls are separated from unassessed mask and background perturbations.",
        (
            "Uniform, channel, and spatial candidate results are controls, not proof of robustness to image perturbations.",
        ),
        bindings,
        headline="PARTIAL",
        rows=tuple(rows),
    )


def _runtime_device_assessment(
    bindings: tuple[SourceBinding, ...],
) -> FigureData:
    return _status_figure(
        "16_reproducibility_ledger",
        "ladder",
        "Runtime and device assessment",
        "No target-device runtime artifact is present, so latency and on-device behavior remain unassessed.",
        (
            "Training-device configuration, when available elsewhere, is not an on-device latency measurement.",
        ),
        bindings,
        headline="ON_DEVICE_NOT_ASSESSED_NO_TARGET",
        rows=(
            _status_row(
                "Target device",
                "UNASSESSED",
                "ON_DEVICE_NOT_ASSESSED_NO_TARGET",
            ),
            _status_row(
                "Latency",
                "UNASSESSED",
                "UNASSESSED_NO_LATENCY_ARTIFACT",
            ),
            _status_row(
                "Throughput",
                "UNASSESSED",
                "UNASSESSED_NO_RUNTIME_ARTIFACT",
            ),
            _status_row(
                "Memory / power",
                "UNASSESSED",
                "UNASSESSED_NO_DEVICE_TELEMETRY",
            ),
        ),
    )


def _evidence_release_ledger(
    public: Mapping[str, Any],
    bindings: tuple[SourceBinding, ...],
    *,
    cache_count: int,
    private_included: bool,
) -> FigureData:
    return _figure(
        "17_evidence_release_ledger",
        "ladder",
        "Evidence and release ledger",
        "Auditable release status across governance, model selection, diagnostics, and privacy.",
        tuple(public["limitations"]),
        bindings,
        {
            "steps": [
                {
                    "label": "Protocol v2",
                    "detail": "digest verified",
                    "status": "established",
                },
                {
                    "label": "Dependency panel",
                    "detail": "nested and disjoint",
                    "status": "established",
                },
                {
                    "label": "Inventory",
                    "detail": "population preserved",
                    "status": "established",
                },
                {
                    "label": "Embedding packs",
                    "detail": f"{cache_count} supplied",
                    "status": "established" if cache_count else "conditional",
                },
                {
                    "label": "DEV selection",
                    "detail": "single frozen receipt",
                    "status": "established",
                },
                {"label": "CAL", "detail": "reporting only", "status": "conditional"},
                {
                    "label": "Exposed diagnostic",
                    "detail": "not independent",
                    "status": "conditional",
                },
                {
                    "label": "Public release",
                    "detail": "aggregate only",
                    "status": "established",
                },
                {
                    "label": "Private Q/K/V",
                    "detail": "included" if private_included else "not requested",
                    "status": "conditional",
                },
                {
                    "label": "Final evaluation",
                    "detail": "not permitted",
                    "status": "out_of_scope",
                },
            ]
        },
    )


def _inventory_populations(
    inventory: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    body = inventory["inventory"]
    populations = {
        "successor": body.get("successor_population"),
        "auxiliary": body.get("identity_free_auxiliary_population"),
        "terminal": body.get("terminal_exclusions"),
    }
    if any(not isinstance(records, list) for records in populations.values()):
        raise FigureContractError("successor inventory populations differ")
    return populations


def _status_row(
    label: str,
    status: str,
    detail: str,
    *,
    count: int | None = None,
) -> dict[str, Any]:
    return {"label": label, "status": status, "detail": detail, "count": count}


def _status_figure(
    figure_id: str,
    kind: str,
    title: str,
    caption: str,
    limitations: Sequence[str],
    bindings: tuple[SourceBinding, ...],
    *,
    headline: str,
    rows: Sequence[Mapping[str, Any]],
) -> FigureData:
    steps = [
        {
            "label": "Evidence status",
            "detail": headline,
            "status": "established" if headline == "AVAILABLE" else "conditional",
        }
    ]
    for row in rows:
        count = row.get("count")
        prefix = "" if count is None else f"{count:,} | "
        status = str(row["status"])
        detail = str(row["detail"])
        steps.append(
            {
                "label": row["label"],
                "detail": f"{prefix}{status}: {detail}",
                "status": (
                    "established"
                    if status in {"AVAILABLE", "ASSESSED"}
                    else "conditional"
                ),
            }
        )
    return _figure(
        figure_id,
        kind,
        title,
        caption,
        limitations,
        bindings,
        {"steps": steps},
    )


def _figure(
    figure_id: str,
    kind: str,
    title: str,
    caption: str,
    limitations: Sequence[str],
    bindings: tuple[SourceBinding, ...],
    payload: Mapping[str, Any],
) -> FigureData:
    return FigureData.create(
        figure_id=figure_id,
        kind=kind,
        scope=PublicationScope.PUBLIC,
        title=title,
        caption=caption,
        limitations=tuple(limitations),
        source_bindings=bindings,
        payload=payload,
    )


def _candidate(public: Mapping[str, Any], successor_id: str) -> Mapping[str, Any]:
    return next(
        candidate
        for candidate in public["candidates"]
        if candidate["successor_id"] == successor_id
    )


def _publication_aliases(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    fallback = 1
    used: set[str] = set()
    for candidate in candidates:
        successor_id = candidate["successor_id"]
        alias = _ALIAS_BY_SUCCESSOR_ID.get(successor_id)
        if alias is None:
            while f"Variant {fallback:02d}" in used:
                fallback += 1
            alias = f"Variant {fallback:02d}"
            fallback += 1
        if alias in used:
            raise FigureContractError("successor IDs collapse to one publication alias")
        aliases[successor_id] = alias
        used.add(alias)
    return aliases


def _alias_sort_key(alias: str) -> tuple[int, str]:
    return (_ALIAS_ORDER.get(alias, len(_ALIAS_ORDER)), alias)


def _central_style_index(alias: str) -> int:
    try:
        return _CENTRAL_SPECTRUM_ALIASES.index(alias)
    except ValueError:
        return _ALIAS_ORDER.get(alias, 0)


def _short_scope(scope: str) -> str:
    return "EXPOSED" if scope == "EXPOSED_DIAGNOSTIC" else scope


def _embedding(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 128:
        raise FigureContractError("private embedding must be 128D")
    for item in value:
        _number(item, "private embedding value")


def _exact_object(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FigureContractError(f"{name} fields differ")
    return dict(value)


def _nonempty_array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise FigureContractError(f"{name} must be an array")
    if not value:
        raise FigureContractError(f"{name} must not be empty")
    return value


def _text(value: Any, name: str, *, maximum: int = 160) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise FigureContractError(f"{name} must be non-empty bounded text")
    return value


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FigureContractError(f"{name} must be lowercase SHA-256")
    return value


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FigureContractError(f"{name} must be a non-negative integer")
    return value


def _number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FigureContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FigureContractError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise FigureContractError(f"{name} is below its allowed range")
    if maximum is not None and result > maximum:
        raise FigureContractError(f"{name} is above its allowed range")
    return result
