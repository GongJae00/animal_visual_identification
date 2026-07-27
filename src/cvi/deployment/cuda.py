"""Reserved CUDA deployment facade.

The strict ONNX CUDA backend exists in :mod:`cvi.onnx_backend`, but it is not
yet connected to the canonical identity gallery and calibrated decision
contract.
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = []


class CVIDeploymentCUDA:
    def __init__(self, config: dict[str, Any]):
        raise RuntimeError(
            "CVIDeploymentCUDA is disabled: the former class executed CPU DINO "
            "and CPU FAISS despite its name. Use the strict ONNX CUDA measurement "
            "backend for research until canonical image inference, gallery, and "
            "open-set contracts are connected."
        )
