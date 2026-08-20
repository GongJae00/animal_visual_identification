"""Canonical ordered publication slots and their normalized recipe kinds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FigureSpec:
    figure_id: str
    kind: str
    section: str
    primary_formats: tuple[str, ...] = ("png",)
    alternate_kinds: tuple[str, ...] = ()


FIGURE_REGISTRY = (
    FigureSpec("00_evidence_ladder", "model_ladder", "evidence"),
    FigureSpec("01_source_provenance", "census", "evidence"),
    FigureSpec("02_census_availability", "census", "population"),
    FigureSpec("03_role_dependency_closure", "architecture", "governance"),
    FigureSpec("04_governance_panel", "architecture", "governance"),
    FigureSpec("05_score_distributions", "ladder", "population"),
    FigureSpec("06_model_dataflow", "architecture", "model"),
    FigureSpec("07_evaluation_protocol", "architecture", "model"),
    FigureSpec("08_cache_bindings", "ladder", "evaluation"),
    FigureSpec(
        "09_embedding_spectrum_pca",
        "embedding_diagnostics",
        "embeddings",
        alternate_kinds=("ladder",),
    ),
    FigureSpec("10_gallery_composition", "gallery_composition", "retrieval"),
    FigureSpec("11_cosine_rank_distributions", "score_rank_distributions", "retrieval"),
    FigureSpec(
        "12_private_ranked_qkv",
        "ranked_retrieval",
        "retrieval",
        alternate_kinds=("ladder",),
    ),
    FigureSpec("13_primary_results_paired_deltas", "result_forest", "results"),
    FigureSpec("14_scope_interpretation", "ladder", "interpretation"),
    FigureSpec("15_limitations", "ladder", "interpretation"),
    FigureSpec("16_reproducibility_ledger", "ladder", "release"),
    FigureSpec("17_evidence_release_ledger", "ladder", "release"),
)

if tuple(spec.figure_id[:2] for spec in FIGURE_REGISTRY) != tuple(
    f"{index:02d}" for index in range(18)
):
    raise RuntimeError("figure registry must remain ordered from 00 through 17")

FIGURE_BY_ID = {spec.figure_id: spec for spec in FIGURE_REGISTRY}
