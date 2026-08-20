from __future__ import annotations

from evaluation.optimization_protocol import OptimizationAxis, SurfaceStatus
from evaluation.optimization_surfaces.identification import SURFACES


_CHANNELS = (
    "identification.appearance",
    "identification.face",
    "identification.nose",
)

_FALSE_SWAP_IDS = {
    "identification.appearance.session.miewid",
    "identification.appearance.batch.miewid_python_loop",
}

_FORBIDDEN_LEVER_FRAGMENTS = (
    "legacy_options",
    "SuperAnimal",
    "silent CPU fallback",
    "IdentityEngine class name",
)

_ADMITTED_POINTERS = (
    "OnnxGraphOptimization",
    "IOBinding",
    "onnx_inference_benchmark",
)


def test_surfaces_cover_three_identification_channels() -> None:
    assert SURFACES
    ids = [surface.surface_id for surface in SURFACES]
    assert len(ids) == len(set(ids))
    stages = {surface.stage_id for surface in SURFACES}
    assert stages == set(_CHANNELS)
    for channel in _CHANNELS:
        rows = [surface for surface in SURFACES if surface.stage_id == channel]
        assert rows
        axes = {surface.axis for surface in rows}
        assert OptimizationAxis.session in axes
        assert OptimizationAxis.batch in axes
        assert OptimizationAxis.cuda in axes


def test_backbone_swap_flags() -> None:
    for surface in SURFACES:
        if surface.surface_id in _FALSE_SWAP_IDS:
            assert surface.survives_backbone_swap is False
        else:
            assert surface.survives_backbone_swap is True


def test_wired_session_batch_cuda_facts() -> None:
    appearance = {
        surface.surface_id: surface
        for surface in SURFACES
        if surface.stage_id == "identification.appearance"
    }
    assert (
        appearance["identification.appearance.session.onnx_extractor"].status
        is SurfaceStatus.wired
    )
    assert (
        appearance["identification.appearance.session.disable_fallback"].status
        is SurfaceStatus.wired
    )
    assert (
        appearance["identification.appearance.session.expected_providers"].status
        is SurfaceStatus.wired
    )
    assert (
        appearance["identification.appearance.batch.extract_batch"].status
        is SurfaceStatus.wired
    )
    assert (
        appearance["identification.appearance.batch.max_batch_size"].lever.find("32")
        >= 0
    )
    assert appearance["identification.appearance.cuda.use_cuda"].status is SurfaceStatus.wired
    assert (
        appearance["identification.appearance.execution.onnx_load_del"].status
        is SurfaceStatus.wired
    )
    assert (
        appearance["identification.appearance.execution.inference_mode"].status
        is SurfaceStatus.wired
    )


def test_forbidden_and_boundary_rows() -> None:
    forbidden = [s for s in SURFACES if s.status is SurfaceStatus.forbidden]
    blob = " ".join(s.lever for s in forbidden)
    for fragment in _FORBIDDEN_LEVER_FRAGMENTS:
        assert fragment in blob
    boundary = [s for s in SURFACES if s.status is SurfaceStatus.out_of_product_boundary]
    boundary_levers = " ".join(s.lever for s in boundary)
    assert "open-set" in boundary_levers
    assert "tracking" in boundary_levers


def test_admitted_not_wired_and_no_invented_numbers() -> None:
    admitted = [s for s in SURFACES if s.status is SurfaceStatus.admitted]
    text = " ".join(s.lever + " " + s.constraint + " " + s.measurement for s in admitted)
    for pointer in _ADMITTED_POINTERS:
        assert pointer in text
    assert "CUDA graphs" not in " ".join(s.status.value for s in SURFACES if s.status is SurfaceStatus.wired)
    graphs = next(
        s for s in SURFACES if s.surface_id == "identification.appearance.cuda.graphs"
    )
    assert graphs.status is SurfaceStatus.admitted
    io_binding = next(
        s for s in SURFACES if s.surface_id == "identification.appearance.memory.io_binding"
    )
    assert io_binding.status is SurfaceStatus.admitted
    for surface in SURFACES:
        blob = " ".join(
            [
                surface.lever,
                surface.measurement,
                surface.constraint,
            ]
        )
        assert "latency_ms" not in blob
        assert "GFLOP" not in blob
        assert "samples/s" not in blob


def test_face_nose_product_admission_is_constraint_not_engine_wiring() -> None:
    for channel in ("identification.face", "identification.nose"):
        constraints = " ".join(
            s.constraint for s in SURFACES if s.stage_id == channel
        )
        assert "IdentityEngine" in constraints
        assert "silently" in constraints or "product-admission" in constraints


def test_promotion_stays_in_training() -> None:
    pointer = next(
        s
        for s in SURFACES
        if s.surface_id == "identification.appearance.evaluation.promotion_pointer"
    )
    assert pointer.status is SurfaceStatus.admitted
    assert "identification/training/appearance/optimization.py" in pointer.owner
    assert pointer.measurement.endswith("evaluate_promotion")
