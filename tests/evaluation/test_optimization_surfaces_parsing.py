from __future__ import annotations

import ast
from pathlib import Path

from evaluation.optimization_protocol import (
    OptimizationAxis,
    STAGE_IDS,
    SurfaceStatus,
)
from evaluation.optimization_surfaces.parsing import SURFACES

from tests.repo_root import REPO_ROOT as ROOT

_PARSING_STAGES = (
    "parsing.detection",
    "parsing.segmentation",
    "parsing.regions",
    "parsing.quality",
    "parsing.crops",
)
_MEASURED_FIELD_NAMES = {
    "flops",
    "latency_ms",
    "throughput",
    "samples_per_second",
    "memory_bytes",
    "gpu_utilization",
}


def test_covers_all_five_substages() -> None:
    assert _PARSING_STAGES == STAGE_IDS[:5]
    present = {surface.stage_id for surface in SURFACES}
    assert present == set(_PARSING_STAGES)


def test_surface_ids_unique_and_fields_nonempty() -> None:
    ids = [surface.surface_id for surface in SURFACES]
    assert len(ids) == len(set(ids))
    for surface in SURFACES:
        assert surface.stage_id in _PARSING_STAGES
        assert surface.axis in OptimizationAxis
        assert surface.status in SurfaceStatus
        assert surface.lever
        assert surface.owner
        assert surface.measurement
        assert surface.constraint
        assert isinstance(surface.survives_backbone_swap, bool)


def test_owners_point_at_existing_files() -> None:
    for surface in SURFACES:
        path_text, _, line_text = surface.owner.partition(":")
        path = ROOT / path_text
        assert path.is_file(), surface.owner
        line = int(line_text)
        text = path.read_text(encoding="utf-8").splitlines()
        assert 1 <= line <= len(text), surface.owner


def test_catalog_has_no_measured_values() -> None:
    blob = " ".join(str(surface.to_dict()) for surface in SURFACES)
    for name in _MEASURED_FIELD_NAMES:
        assert name not in blob
    assert "cvi." not in blob


def test_required_statuses_present() -> None:
    statuses = {surface.status for surface in SURFACES}
    assert SurfaceStatus.wired in statuses
    assert SurfaceStatus.admitted in statuses
    assert SurfaceStatus.forbidden in statuses
    assert SurfaceStatus.out_of_product_boundary in statuses


def test_backbone_swap_true_for_batch_cuda_cache_io_order() -> None:
    for surface in SURFACES:
        lever = surface.lever.lower() + surface.constraint.lower()
        if surface.status is SurfaceStatus.forbidden:
            continue
        if any(
            token in lever
            for token in (
                "maximum_batch_size",
                "fail-closed",
                "prediction_cache",
                "full_segment_cache",
                "png",
                "stage order",
                "export order",
            )
        ) and "graph" not in lever:
            assert surface.survives_backbone_swap is True, surface.surface_id


def test_measurement_is_existing_command_or_catalog_only() -> None:
    allowed = {
        "catalog_only",
        "evaluation.commands.evaluate localization-benchmark",
        "evaluation.commands.evaluate oxford-pet",
        "evaluation.commands.evaluate parsing-protocol",
    }
    for surface in SURFACES:
        assert surface.measurement in allowed


def test_evaluation_surfaces_do_not_import_visualization() -> None:
    path = ROOT / "evaluation" / "optimization_surfaces" / "parsing.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        names: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = (node.module,)
        assert not any(
            name == "visualization" or name.startswith("visualization.") for name in names
        )


def test_no_parsing_export_optimization_surfaces_module() -> None:
    assert not Path(ROOT / "parsing" / "export" / "optimization_surfaces.py").exists()
