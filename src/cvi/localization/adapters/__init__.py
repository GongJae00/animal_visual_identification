"""Model adapter interface for localization models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cvi.localization.types import LocalizationResult


class AbstractLocalizationAdapter(ABC):
    model_family: str
    model_name: str
    requires_gpu: bool = False

    @abstractmethod
    def detect(self, image: Image.Image, *, image_id: str = "") -> LocalizationResult:
        ...

    @abstractmethod
    def detect_batch(
        self, images: list[Image.Image], *, image_ids: list[str] | None = None
    ) -> list[LocalizationResult]:
        ...

    @property
    @abstractmethod
    def artifact_size_bytes(self) -> int:
        ...

    @property
    @abstractmethod
    def license_id(self) -> str:
        ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "requires_gpu": self.requires_gpu,
            "artifact_size_bytes": self.artifact_size_bytes,
            "license_id": self.license_id,
        }


class OnnxLocalizationAdapter(AbstractLocalizationAdapter):
    """Base class for ONNX-based localizers with strict model/contract validation."""

    def __init__(
        self,
        model_path: Path,
        manifest_path: Path,
        *,
        device: str = "cpu",
        model_family: str,
        model_name: str,
    ) -> None:
        self._model_path = Path(model_path)
        self._manifest_path = Path(manifest_path)
        self._device = device
        self.model_family = model_family
        self.model_name = model_name
        self._session = None
        self._manifest = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        import onnxruntime

        if not self._model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self._model_path}")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._session = onnxruntime.InferenceSession(
            str(self._model_path),
            providers=providers,
            sess_options=options,
        )

    @property
    def artifact_size_bytes(self) -> int:
        return self._model_path.stat().st_size if self._model_path.is_file() else 0

    @property
    def license_id(self) -> str:
        return "MODEL_SPECIFIC"


__all__ = [
    "AbstractLocalizationAdapter",
    "OnnxLocalizationAdapter",
]
