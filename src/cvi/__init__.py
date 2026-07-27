"""Public package surface for Canine Video Identity.

Research and optional runtime modules must be imported from their explicit
submodules so importing :mod:`cvi` does not require training or CUDA extras.
"""

from importlib import import_module
from typing import Any

from cvi.api import CVI, Match

_LEGACY_EXPORTS = {
    "ArcFaceModel": ("cvi.trainer", "ArcFaceModel"),
    "Detection": ("cvi.detection", "Detection"),
    "DogDetector": ("cvi.detection", "DogDetector"),
    "DogDetectorConfig": ("cvi.detection", "DogDetectorConfig"),
    "FrameSelector": ("cvi.detection", "FrameSelector"),
    "IdentityRegistry": ("cvi.identity_registry", "IdentityRegistry"),
    "IdentityRegistryRecord": (
        "cvi.identity_registry",
        "IdentityRegistryRecord",
    ),
    "MODELS_DIR": ("cvi.model_paths", "MODELS_DIR"),
    "OnnxEmbeddingModel": ("cvi.inference", "OnnxEmbeddingModel"),
    "QualityMetrics": ("cvi.detection", "QualityMetrics"),
    "TrainConfig": ("cvi.trainer", "TrainConfig"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LEGACY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LEGACY_EXPORTS))


__all__ = ["CVI", "Match"]
