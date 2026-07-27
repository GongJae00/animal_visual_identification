"""NoseID-v1 research package with lazy optional deep-learning imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "AlignedNose": ("cvi.nose_id.types", "AlignedNose"),
    "NoseDetectionResult": ("cvi.nose_id.types", "NoseDetectionResult"),
    "NoseIDOutput": ("cvi.nose_id.types", "NoseIDOutput"),
    "NoseKeypoints": ("cvi.nose_id.types", "NoseKeypoints"),
    "NoseIDConfig": ("cvi.nose_id.config", "NoseIDConfig"),
    "NoseIDTrainConfig": ("cvi.nose_id.config", "NoseIDTrainConfig"),
    "AlignmentError": ("cvi.nose_id.alignment", "AlignmentError"),
    "align_nose": ("cvi.nose_id.alignment", "align_nose"),
    "estimate_similarity_transform": ("cvi.nose_id.alignment", "estimate_similarity_transform"),
    "FixedFrequencyBank": ("cvi.nose_id.frequency", "FixedFrequencyBank"),
    "NoseIDModel": ("cvi.nose_id.model", "NoseIDModel"),
    "NoseIDObjective": ("cvi.nose_id.losses", "NoseIDObjective"),
    "CrossSessionPKBatchSampler": ("cvi.nose_id.sampler", "CrossSessionPKBatchSampler"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)
