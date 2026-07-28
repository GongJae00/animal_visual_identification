"""Localization types, adapters, metrics, and benchmark harness."""

from cvi.localization.types import (
    DetectionBox,
    Keypoint,
    KeypointSet,
    LocalizationBenchmarkEntry,
    LocalizationResult,
)
from cvi.localization.adapters import (
    AbstractLocalizationAdapter,
    OnnxLocalizationAdapter,
)
from cvi.localization.roi import (
    expand_bbox,
    face_roi_from_dog,
    is_truncated,
    square_padded_crop,
)
from cvi.localization.quality import (
    compute_iou,
    detection_summary,
    greedy_bipartite_match,
    normalized_mean_error,
    pixel_correct_keypoint,
)
from cvi.localization.benchmark import build_contact_sheet, run_benchmark

__all__ = [
    "AbstractLocalizationAdapter",
    "DetectionBox",
    "Keypoint",
    "KeypointSet",
    "LocalizationBenchmarkEntry",
    "LocalizationResult",
    "OnnxLocalizationAdapter",
    "build_contact_sheet",
    "compute_iou",
    "detection_summary",
    "expand_bbox",
    "face_roi_from_dog",
    "greedy_bipartite_match",
    "is_truncated",
    "normalized_mean_error",
    "pixel_correct_keypoint",
    "run_benchmark",
    "square_padded_crop",
]
