from __future__ import annotations

import ast
from evaluation.optimization_protocol import OptimizationAxis, SurfaceStatus
from evaluation.optimization_surfaces.enrollment import SURFACES

from tests.repo_root import REPO_ROOT as ROOT

_ENROLLMENT_STAGES = {"enrollment.registry", "enrollment.write"}


def test_surfaces_export_and_unique_ids() -> None:
    assert SURFACES
    ids = [surface.surface_id for surface in SURFACES]
    assert len(ids) == len(set(ids))
    for surface in SURFACES:
        assert surface.stage_id in _ENROLLMENT_STAGES
        assert surface.survives_backbone_swap is True
        assert surface.owner
        assert surface.measurement
        assert surface.constraint
        assert "cvi." not in surface.constraint
        assert "cvi." not in surface.measurement


def test_wired_vs_admitted_and_cuda_forbidden() -> None:
    by_id = {surface.surface_id: surface for surface in SURFACES}
    assert by_id["enrollment.write.execution.required_evidence"].status is SurfaceStatus.wired
    assert by_id["enrollment.registry.parallel_sequential.namespaces"].status is SurfaceStatus.wired
    assert by_id["enrollment.write.cuda.enroll"].status is SurfaceStatus.forbidden
    assert by_id["enrollment.registry.cuda.identity_derivation"].status is SurfaceStatus.forbidden
    assert by_id["enrollment.registry.math.uuidv5"].axis is OptimizationAxis.math
    namespaces = by_id["enrollment.registry.parallel_sequential.namespaces"].constraint
    assert "enrollment.registered_dog.v1" in namespaces
    assert "enrollment.generated_dog.v1" in namespaces
    assert "dog_id" in by_id["enrollment.registry.io.sqlite_manifest"].constraint
    assert "dog_id" in by_id["enrollment.write.io.pixel_digest"].constraint


def test_catalog_does_not_import_training() -> None:
    path = ROOT / "evaluation" / "optimization_surfaces" / "enrollment.py"
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


def test_enrollment_packages_do_not_import_evaluation_or_training() -> None:
    violations: list[str] = []
    for package in ("enrollment",):
        for path in sorted((ROOT / package).rglob("*.py")):
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
