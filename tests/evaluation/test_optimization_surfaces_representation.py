from __future__ import annotations

import ast
from collections import Counter

from evaluation.optimization_protocol import (
    STAGE_IDS,
    OptimizationAxis,
    SurfaceStatus,
    visualization_trace,
)
from evaluation.optimization_surfaces.representation import SURFACES
from tests.repo_root import REPO_ROOT as ROOT

_REPRESENTATION_STAGES = {
    "representation.evidence",
    "representation.channels",
    "representation.quality",
}
_REQUIRED_AXES = {
    OptimizationAxis.math,
    OptimizationAxis.memory,
    OptimizationAxis.batch,
    OptimizationAxis.code,
    OptimizationAxis.latency,
    OptimizationAxis.compute,
    OptimizationAxis.cuda,
    OptimizationAxis.parallel_sequential,
    OptimizationAxis.evaluation,
    OptimizationAxis.io,
}
_FORBIDDEN_LEVERS = {
    "attention pooling over channels or templates",
    "IdentityEngine calling parsing",
    "unavailable optional channels are omitted from embedding dicts, never zero-filled",
}


def test_surfaces_export_nonempty_unique() -> None:
    assert SURFACES
    ids = [surface.surface_id for surface in SURFACES]
    assert len(ids) == len(set(ids))
    for surface in SURFACES:
        assert surface.stage_id in _REPRESENTATION_STAGES
        assert surface.stage_id in STAGE_IDS
        assert surface.survives_backbone_swap is True


def test_visualization_trace_representation_rows() -> None:
    substages = visualization_trace()["substages"]["representation"]
    assert tuple(substages) == ("00_evidence", "01_channels", "02_quality")
    by_stage = Counter(surface.stage_id for surface in SURFACES)
    assert len(substages["00_evidence"]) == by_stage["representation.evidence"]
    assert len(substages["01_channels"]) == by_stage["representation.channels"]
    assert len(substages["02_quality"]) == by_stage["representation.quality"]
    assert by_stage["representation.evidence"] >= 1
    assert by_stage["representation.channels"] >= 1
    assert by_stage["representation.quality"] >= 1


def test_required_axes_and_forbidden_rows() -> None:
    axes = {surface.axis for surface in SURFACES}
    assert _REQUIRED_AXES <= axes
    forbidden = {
        surface.lever
        for surface in SURFACES
        if surface.status is SurfaceStatus.forbidden
    }
    assert _FORBIDDEN_LEVERS <= forbidden
    packed = [
        surface
        for surface in SURFACES
        if surface.measurement == "evaluation.integrity.packed_cache"
    ]
    assert packed
    assert any(surface.axis is OptimizationAxis.memory for surface in packed)
    cuda = [surface for surface in SURFACES if surface.axis is OptimizationAxis.cuda]
    assert cuda
    assert all(surface.status is SurfaceStatus.forbidden for surface in cuda)


def test_owners_are_file_line_and_exist() -> None:
    for surface in SURFACES:
        path_text, _, line_text = surface.owner.partition(":")
        path = ROOT / path_text
        assert path.is_file(), surface.owner
        line = int(line_text)
        n_lines = sum(1 for _ in path.open(encoding="utf-8"))
        assert 1 <= line <= n_lines


def test_catalog_has_no_measured_values() -> None:
    measured = {
        "flops",
        "latency_ms",
        "throughput",
        "samples_per_second",
        "memory_bytes",
        "gpu_utilization",
    }
    for surface in SURFACES:
        assert not (measured & set(surface.to_dict()))
        assert "cvi." not in surface.surface_id


def test_representation_catalog_does_not_import_visualization() -> None:
    path = ROOT / "evaluation" / "optimization_surfaces" / "representation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    assert not any(
        name == "visualization" or name.startswith("visualization.") for name in names
    )
    observer = [
        surface
        for surface in SURFACES
        if surface.measurement.startswith("visualization.")
    ]
    assert observer
    assert all(surface.status is SurfaceStatus.admitted for surface in observer)
