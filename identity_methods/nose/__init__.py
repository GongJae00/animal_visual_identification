"""NoseID-v1 research package with lazy optional deep-learning imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "AlignedNose": ("identity_methods.nose.types", "AlignedNose"),
    "NoseDetectionResult": ("identity_methods.nose.types", "NoseDetectionResult"),
    "NoseIDOutput": ("identity_methods.nose.types", "NoseIDOutput"),
    "NoseKeypoints": ("identity_methods.nose.types", "NoseKeypoints"),
    "NoseIDConfig": ("identity_methods.nose.config", "NoseIDConfig"),
    "NoseIDTrainConfig": ("identity_methods.nose.config", "NoseIDTrainConfig"),
    "AlignmentError": ("identity_methods.nose.alignment", "AlignmentError"),
    "align_nose": ("identity_methods.nose.alignment", "align_nose"),
    "estimate_similarity_transform": ("identity_methods.nose.alignment", "estimate_similarity_transform"),
    "FixedFrequencyBank": ("identity_methods.nose.frequency", "FixedFrequencyBank"),
    "RestorationConfig": ("identity_methods.nose.restoration", "RestorationConfig"),
    "RestorationDiagnostics": ("identity_methods.nose.restoration", "RestorationDiagnostics"),
    "RestorationResult": ("identity_methods.nose.restoration", "RestorationResult"),
    "restore_nose_frames": ("identity_methods.nose.restoration", "restore_nose_frames"),
    "TemporalEmbeddingResult": ("identity_methods.nose.temporal", "TemporalEmbeddingResult"),
    "aggregate_nose_embeddings": ("identity_methods.nose.temporal", "aggregate_nose_embeddings"),
    "NoseIDModel": ("identity_methods.nose.model", "NoseIDModel"),
    "NoseIDObjective": ("identity_methods.nose.losses", "NoseIDObjective"),
    "CrossSessionPKBatchSampler": ("identity_methods.nose.sampler", "CrossSessionPKBatchSampler"),
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
