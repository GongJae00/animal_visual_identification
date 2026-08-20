from __future__ import annotations

import ast

from evaluation.optimization_protocol import OptimizationAxis, SurfaceStatus
from evaluation.optimization_surfaces.gallery import SURFACES

from tests.repo_root import REPO_ROOT as ROOT


def test_surfaces_export_and_unique_ids() -> None:
    assert SURFACES
    ids = [surface.surface_id for surface in SURFACES]
    assert len(ids) == len(set(ids))
    for surface in SURFACES:
        assert surface.stage_id == "gallery.store"
        assert surface.survives_backbone_swap is True
        assert "cvi." not in surface.constraint
        assert "cvi." not in surface.measurement


def test_io_and_max_template_wired_vs_admitted() -> None:
    by_id = {surface.surface_id: surface for surface in SURFACES}
    io_atomic = by_id["gallery.store.io.json_bin_atomic"]
    assert io_atomic.axis is OptimizationAxis.io
    assert io_atomic.status is SurfaceStatus.wired
    assert "DrvFS" in io_atomic.constraint
    assert "renameat2" in io_atomic.constraint

    rename = by_id["gallery.store.io.renameat2_publication"]
    assert rename.status is SurfaceStatus.wired
    assert "DrvFS" in rename.constraint

    load = by_id["gallery.store.memory.load_vs_mmap"]
    assert load.status is SurfaceStatus.wired
    mmap = by_id["gallery.store.memory.mmap"]
    assert mmap.status is SurfaceStatus.admitted

    pointer = by_id["gallery.store.memory.gallery_bytes_pointer"]
    assert pointer.status is SurfaceStatus.admitted
    assert "gallery_bytes" in pointer.measurement
    assert "must not import training" in pointer.measurement

    max_template = by_id["gallery.store.math.identity_max_template"]
    assert max_template.axis is OptimizationAxis.math
    assert max_template.status is SurfaceStatus.wired
    assert "max template" in max_template.constraint
    assert "attention" in max_template.constraint

    attention = by_id["gallery.store.math.attention_aggregation"]
    assert attention.status is SurfaceStatus.forbidden

    cuda = by_id["gallery.store.cuda.enroll"]
    assert cuda.status is SurfaceStatus.forbidden

    batch = by_id["gallery.store.batch.enroll_many"]
    assert batch.status is SurfaceStatus.wired
    assert "dog_id" in by_id["gallery.store.execution.required_channels"].constraint


def test_catalog_does_not_import_training() -> None:
    path = ROOT / "evaluation" / "optimization_surfaces" / "gallery.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        names: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = (node.module,)
        assert not any(
            name == "identification.training" or name.startswith("identification.training.")
            for name in names
        )


def test_gallery_package_does_not_import_evaluation_or_training() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "gallery").rglob("*.py")):
        relative = path.relative_to(ROOT)
        if "commands" in relative.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
            if any(
                name == "evaluation"
                or name.startswith("evaluation.")
                or name == "identification.training"
                or name.startswith("identification.training.")
                for name in names
            ):
                violations.append(str(relative))
    assert not violations
