"""Localization types, adapters, metrics, consensus, student, and benchmark harness."""

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
    estimate_blur,
    greedy_bipartite_match,
    normalized_mean_error,
    pixel_correct_keypoint,
    score_dog_quality,
    score_face_quality,
    score_nose_quality,
    DogQuality,
    FaceQuality,
    NoseQuality,
)
from cvi.localization.consensus import (
    compute_error_correlation,
    consensus_admission,
    consensus_dog_bbox,
    consensus_keypoint,
    robust_weighted_keypoint,
    weighted_box_fusion,
    FailureVector,
)
from cvi.localization.student import AbstractStudentTrainer, TeacherLabel
from cvi.localization.benchmark import build_contact_sheet, run_benchmark

__all__ = [
    "AbstractLocalizationAdapter",
    "AbstractStudentTrainer",
    "DetectionBox",
    "DogQuality",
    "FaceQuality",
    "FailureVector",
    "Keypoint",
    "KeypointSet",
    "LocalizationBenchmarkEntry",
    "LocalizationResult",
    "NoseQuality",
    "OnnxLocalizationAdapter",
    "TeacherLabel",
    "build_contact_sheet",
    "compute_error_correlation",
    "compute_iou",
    "consensus_admission",
    "consensus_dog_bbox",
    "consensus_keypoint",
    "detection_summary",
    "estimate_blur",
    "expand_bbox",
    "face_roi_from_dog",
    "greedy_bipartite_match",
    "is_truncated",
    "normalized_mean_error",
    "pixel_correct_keypoint",
    "robust_weighted_keypoint",
    "run_benchmark",
    "score_dog_quality",
    "score_face_quality",
    "score_nose_quality",
    "square_padded_crop",
    "weighted_box_fusion",
]
