"""Parsing: detection, segmentation, regions, quality, and crops.

``training/`` holds student/teacher loops. ``export/`` is the runtime path.
``prototype.runtime.IdentityEngine`` does not import this package.
"""

from parsing.export.types import (
    DetectionBox,
    Keypoint,
    KeypointSet,
    LocalizationResult,
)

__all__ = [
    "DetectionBox",
    "Keypoint",
    "KeypointSet",
    "LocalizationResult",
]
