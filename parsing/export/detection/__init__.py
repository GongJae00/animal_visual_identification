"""Runtime dog detection and localization adapters."""

from parsing.export.detection.adapters import AbstractLocalizationAdapter
from parsing.export.detection.detection import Detection, DogDetector, DogDetectorConfig

__all__ = [
    "AbstractLocalizationAdapter",
    "Detection",
    "DogDetector",
    "DogDetectorConfig",
]
