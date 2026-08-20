from __future__ import annotations

import ast
from pathlib import Path

from evaluation.optimization_protocol import (
    OptimizationAxis,
    SurfaceStatus,
    visualization_trace,
)
from evaluation.optimization_surfaces.runtime import SURFACES

from tests.repo_root import REPO_ROOT as ROOT

_OWNED_STAGES = frozenset({"prototype.runtime", "operations.measurement"})
_REQUIRED_IDS = {
    "prototype.runtime.identity_engine.composition",
    "prototype.export.embedding_producer.pipeline",
    "prototype.export.onnx.graph_optimization_enum",
    "prototype.export.onnx.session_options",
    "prototype.export.onnx.enable_cuda_graph",
    "prototype.export.onnx.io_binding",
    "prototype.export.onnx.cpu_backend",
    "prototype.export.onnx.cuda_backend",
    "prototype.export.onnx.cpu_xor_cuda_extra",
    "prototype.export.onnx.maximum_batch_size",
    "prototype.export.onnx.cpu_mem_arena",
    "prototype.export.onnx.gpu_mem_limit",
    "prototype.export.onnx.backend_synchronize",
    "operations.measurement.onnx_inference_benchmark.policy",
    "operations.measurement.onnx_inference_benchmark.tensor_bytes",
    "operations.measurement.onnx_inference_benchmark.fresh_processes",
    "operations.workers.sanitized_environment",
    "operations.workers.batch_invariance_fresh_workers",
    "operations.measurement.gpu_telemetry",
    "operations.measurement.wsl_driver_projection",
    "operations.measurement.onnx_inference_benchmark.receipt",
    "operations.measurement.onnx_inference_benchmark.promotion",
}


def test_runtime_surfaces_are_catalog_rows_for_owned_stages() -> None:
    assert SURFACES
    ids = [surface.surface_id for surface in SURFACES]
    assert len(ids) == len(set(ids))
    assert _REQUIRED_IDS <= set(ids)
    for surface in SURFACES:
        assert surface.stage_id in _OWNED_STAGES
        assert surface.axis in OptimizationAxis
        assert isinstance(surface.survives_backbone_swap, bool)
        assert surface.owner
        assert ":" in surface.owner or surface.owner.startswith("pyproject.toml")


def test_session_cuda_graph_and_io_binding_are_forbidden() -> None:
    by_id = {surface.surface_id: surface for surface in SURFACES}
    assert by_id["prototype.export.onnx.enable_cuda_graph"].status is SurfaceStatus.forbidden
    assert by_id["prototype.export.onnx.io_binding"].status is SurfaceStatus.forbidden
    assert by_id["prototype.export.onnx.cpu_xor_cuda_extra"].status is SurfaceStatus.forbidden
    assert by_id["operations.measurement.onnx_inference_benchmark.promotion"].status is (
        SurfaceStatus.forbidden
    )
    assert by_id["prototype.export.onnx.cuda_backend"].status is SurfaceStatus.wired
    assert by_id["operations.measurement.wsl_driver_projection"].status is SurfaceStatus.admitted


def test_enable_cuda_graph_is_frozen_off_in_backend() -> None:
    source = (ROOT / "prototype" / "export" / "onnx_backend.py").read_text(encoding="utf-8")
    assert '"enable_cuda_graph": "0"' in source
    assert "IOBinding" not in source
    assert "io_binding" not in source
    assert 'inference_api: str = "session_run_named_output"' in source


def test_identity_engine_must_not_import_forbidden_packages() -> None:
    engine = ROOT / "prototype" / "runtime" / "engine.py"
    tree = ast.parse(engine.read_text(encoding="utf-8"), filename=str(engine))
    forbidden_prefixes = (
        "parsing",
        "identification.training",
        "visualization",
        "evaluation",
        "operations",
    )
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    for name in names:
        for prefix in forbidden_prefixes:
            assert name != prefix
            assert not name.startswith(prefix + ".")


def test_visualization_trace_places_runtime_rows_outside_pipeline_substages() -> None:
    trace = visualization_trace()
    runtime_ids = {row["surface_id"] for row in trace["runtime"]}
    assert _REQUIRED_IDS <= runtime_ids
    blob = str(trace["runtime"])
    for measured in (
        "flops",
        "latency_ms",
        "throughput",
        "samples_per_second",
        "memory_bytes",
        "gpu_utilization",
    ):
        assert measured not in blob


def test_deleted_example_configs_are_not_claimed() -> None:
    examples = Path("operations/configs")
    if examples.exists():
        leftover = list(examples.glob("*.example.json"))
        assert leftover == []
    configuration = ROOT / "docs" / "CONFIGURATION.md"
    text = configuration.read_text(encoding="utf-8")
    assert "onnx_inference_benchmark_cpu.example.json" not in text
    assert "embedding_production_policy.example.json" not in text
