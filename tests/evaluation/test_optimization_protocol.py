from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from evaluation.optimization_protocol import (
    INTERPRETATION,
    PROTOCOL_SCHEMA,
    STAGE_IDS,
    OptimizationAxis,
    protocol_document,
    visualization_trace,
)
from evaluation.optimization_surfaces import SURFACES
from visualization.rendering.pipeline import STAGE_LAYOUT

from tests.repo_root import REPO_ROOT as ROOT

_SCHEMA_PATH = (
    ROOT
    / "shared"
    / "contracts"
    / "schemas"
    / "evaluation.optimization_protocol.v1.schema.json"
)
_SURFACE_KEYS = {
    "surface_id",
    "stage_id",
    "axis",
    "lever",
    "survives_backbone_swap",
    "status",
    "owner",
    "measurement",
    "constraint",
}
_MEASURED_FIELD_NAMES = {
    "flops",
    "latency_ms",
    "throughput",
    "samples_per_second",
    "memory_bytes",
    "gpu_utilization",
}


def test_surface_ids_are_unique() -> None:
    ids = [surface.surface_id for surface in SURFACES]
    assert len(ids) == len(set(ids))


def test_known_stage_and_axis() -> None:
    axes = {axis.value for axis in OptimizationAxis}
    for surface in SURFACES:
        assert surface.stage_id in STAGE_IDS
        assert surface.axis.value in axes
        assert isinstance(surface.survives_backbone_swap, bool)


def test_protocol_document_is_a_catalog_not_results() -> None:
    document = protocol_document()
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)
    assert document["schema_version"] == PROTOCOL_SCHEMA
    assert document["interpretation"] == INTERPRETATION
    blob = json.dumps(document)
    assert "cvi." not in blob
    for row in document["surfaces"]:
        assert set(row) == _SURFACE_KEYS
        assert not (_MEASURED_FIELD_NAMES & set(row))
        assert isinstance(row["survives_backbone_swap"], bool)


def test_visualization_trace_has_protocol_and_substages() -> None:
    trace = visualization_trace()
    assert "protocol" in trace
    assert "substages" in trace
    assert "runtime" in trace
    assert "title" not in trace
    assert "caption" not in trace
    substages = trace["substages"]
    assert tuple(substages) == (
        "parsing",
        "identification",
        "representation",
        "enrollment",
        "gallery",
        "search",
    )
    for stage, (_vis_dir, names) in STAGE_LAYOUT.items():
        assert tuple(substages[stage]) == names
    assert "06_" not in json.dumps(trace)


def test_empty_stage_stubs_are_valid() -> None:
    from evaluation.optimization_surfaces.enrollment import SURFACES as enrollment_surfaces
    from evaluation.optimization_surfaces.gallery import SURFACES as gallery_surfaces
    from evaluation.optimization_surfaces.identification import SURFACES as identification_surfaces
    from evaluation.optimization_surfaces.parsing import SURFACES as parsing_surfaces

    assert parsing_surfaces
    assert identification_surfaces
    assert enrollment_surfaces
    assert gallery_surfaces
    parsing = visualization_trace()["substages"]["parsing"]
    assert parsing["00_detection"]
    identification = visualization_trace()["substages"]["identification"]
    assert identification["00_appearance"]
    assert identification["01_face"]
    assert identification["02_nose"]
    enrollment = visualization_trace()["substages"]["enrollment"]
    assert enrollment["00_registry"]
    assert enrollment["01_write"]
    gallery = visualization_trace()["substages"]["gallery"]
    assert gallery["00_store"]


def test_optimization_protocol_cli_writes_trace(tmp_path: Path) -> None:
    output = tmp_path / "optimization_protocol.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.commands.evaluate",
            "optimization-protocol",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["event"] == "optimization_protocol_written"
    assert payload["schema_version"] == PROTOCOL_SCHEMA
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["protocol"]["schema_version"] == PROTOCOL_SCHEMA


def test_evaluation_does_not_import_visualization() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "evaluation").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
            if any(name == "visualization" or name.startswith("visualization.") for name in names):
                violations.append(str(path.relative_to(ROOT)))
    assert not violations


def test_algorithms_except_commands_do_not_import_evaluation() -> None:
    packages = (
        "parsing",
        "identification",
        "representation",
        "enrollment",
        "gallery",
        "search",
    )
    violations: list[str] = []
    for package in packages:
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
                if any(name == "evaluation" or name.startswith("evaluation.") for name in names):
                    violations.append(str(relative))
    assert not violations
