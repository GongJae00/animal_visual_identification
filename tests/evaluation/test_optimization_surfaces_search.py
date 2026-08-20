from __future__ import annotations

from evaluation.optimization_protocol import (
    OptimizationAxis,
    SurfaceStatus,
    protocol_document,
    visualization_trace,
)
from evaluation.optimization_surfaces.search import SURFACES


def test_search_surfaces_export_nonempty_catalog() -> None:
    assert SURFACES
    ids = [surface.surface_id for surface in SURFACES]
    assert len(ids) == len(set(ids))
    assert all(surface.stage_id in {"search.scoring", "search.matching"} for surface in SURFACES)
    assert all(surface.survives_backbone_swap is True for surface in SURFACES)


def test_cosine_max_template_and_availability_levers_are_wired() -> None:
    by_id = {surface.surface_id: surface for surface in SURFACES}
    cosine = by_id["search.scoring.available_intersection_weighted_cosine"]
    assert cosine.axis is OptimizationAxis.math
    assert cosine.status is SurfaceStatus.wired
    assert "exact_available_intersection_weighted_cosine.v1" in cosine.lever
    assert "not attention" in cosine.constraint
    max_template = by_id["search.matching.identity_max_template"]
    assert max_template.status is SurfaceStatus.wired
    assert "max" in max_template.lever
    fail_closed = by_id["search.scoring.fail_closed_required_channels"]
    assert fail_closed.axis is OptimizationAxis.execution
    assert fail_closed.status is SurfaceStatus.wired


def test_admitted_gpu_gemm_and_forbidden_attention_and_match_changes() -> None:
    by_id = {surface.surface_id: surface for surface in SURFACES}
    gpu = by_id["search.scoring.gpu_gemm_cosine"]
    assert gpu.axis is OptimizationAxis.cuda
    assert gpu.status is SurfaceStatus.admitted
    assert "bit-identical" in gpu.constraint
    assert by_id["search.scoring.attention_softmax_value_mixing"].status is SurfaceStatus.forbidden
    assert by_id["search.scoring.learned_projections"].status is SurfaceStatus.forbidden
    assert (
        by_id["search.matching.identity_engine_match_behavior"].status
        is SurfaceStatus.forbidden
    )
    assert (
        by_id["search.matching.open_set_decision_as_product"].status
        is SurfaceStatus.out_of_product_boundary
    )


def test_measurement_pointers_do_not_let_search_import_evaluation() -> None:
    by_id = {surface.surface_id: surface for surface in SURFACES}
    assert by_id["search.matching.evaluation_search_metrics"].measurement == (
        "evaluation.search_metrics"
    )
    assert "must not import evaluation" in by_id[
        "search.matching.evaluation_search_metrics"
    ].constraint
    assert by_id["search.matching.evaluation_verification"].measurement == (
        "evaluation.verification"
    )


def test_search_surfaces_appear_in_protocol_and_visualization_substages() -> None:
    document = protocol_document()
    catalog_ids = {row["surface_id"] for row in document["surfaces"]}
    assert "search.scoring.available_intersection_weighted_cosine" in catalog_ids
    substages = visualization_trace()["substages"]["search"]
    scoring_ids = {row["surface_id"] for row in substages["00_scoring"]}
    matching_ids = {row["surface_id"] for row in substages["01_matching"]}
    assert scoring_ids
    assert matching_ids
    assert scoring_ids.isdisjoint(matching_ids)
