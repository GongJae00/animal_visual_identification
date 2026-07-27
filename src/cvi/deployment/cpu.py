"""Reserved CPU deployment facade.

The strict ONNX CPU backend exists in :mod:`cvi.onnx_backend`, but it is not yet
connected to the canonical identity gallery and calibrated decision contract.
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = []


class CVIDeploymentCPU:
    def __init__(self, config: dict[str, Any]):
        raise RuntimeError(
            "CVIDeploymentCPU is disabled: the former class ignored its ONNX "
            "model and only indexed caller-supplied vectors. Use the strict "
            "ONNX CPU measurement backend for research until canonical image "
            "inference, gallery, and open-set contracts are connected."
        )
