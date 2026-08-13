"""Unsupervised ROI-level quality scoring.

Quality features are computed from the image and predictions without
ground-truth annotations. Composite ``overall`` scores are normalized to
``[0, 1]`` with higher values preferred; raw diagnostic feature direction is
defined by each dataclass field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from parsing.roi import is_truncated
from parsing.types import DetectionBox, KeypointSet


@dataclass(frozen=True, slots=True)
class DogQuality:
    detector_confidence: float
    model_agreement: float
    truncation: float
    native_resolution: float
    multi_dog_contamination: float
    blur_estimate: float
    overall: float

    def to_list(self) -> list[float]:
        return [
            self.detector_confidence,
            self.model_agreement,
            self.truncation,
            self.native_resolution,
            self.multi_dog_contamination,
            self.blur_estimate,
        ]


@dataclass(frozen=True, slots=True)
class FaceQuality:
    landmark_confidence: float
    anchor_visibility: float
    yaw_roll_proxy: float
    resolution: float
    truncation: float
    blur_estimate: float
    overall: float

    def to_list(self) -> list[float]:
        return [
            self.landmark_confidence,
            self.anchor_visibility,
            self.yaw_roll_proxy,
            self.resolution,
            self.truncation,
            self.blur_estimate,
        ]


@dataclass(frozen=True, slots=True)
class NoseQuality:
    anchor_agreement: float
    native_resolution: float
    blur_estimate: float
    specular_ratio: float
    truncation: float
    muzzle_contamination: float
    support_coverage: float
    overall: float

    def to_list(self) -> list[float]:
        return [
            self.anchor_agreement,
            self.native_resolution,
            self.blur_estimate,
            self.specular_ratio,
            self.truncation,
            self.muzzle_contamination,
            self.support_coverage,
        ]


def estimate_blur(image: Image.Image) -> float:
    import cv2

    gray = np.asarray(image.convert("L"), dtype=np.float64)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return min(1.0, variance / 500.0)


def score_dog_quality(
    bbox: DetectionBox,
    *,
    model_agreement: float = 1.0,
    multi_dog_boxes: int = 1,
    image_width: int = 0,
    image_height: int = 0,
    blur: float = 0.5,
) -> DogQuality:
    trunc = 0.0
    if image_width and image_height:
        trunc = 1.0 - float(
            is_truncated(bbox, image_width=image_width, image_height=image_height)
        )
    native = min(1.0, max(bbox.width / 224.0, bbox.height / 224.0))
    contamination = 1.0 / max(multi_dog_boxes, 1)
    overall = float(
        np.mean([bbox.confidence, model_agreement, trunc, native, contamination, blur])
    )
    return DogQuality(
        detector_confidence=bbox.confidence,
        model_agreement=model_agreement,
        truncation=trunc,
        native_resolution=native,
        multi_dog_contamination=contamination,
        blur_estimate=blur,
        overall=overall,
    )


def score_face_quality(
    landmarks: KeypointSet | None,
    bbox: DetectionBox | None = None,
    *,
    image_width: int = 0,
    image_height: int = 0,
    blur: float = 0.5,
) -> FaceQuality:
    landmark_conf = 0.0
    anchor_vis = 0.0
    if landmarks is not None:
        confidences = [kp.confidence for kp in landmarks.keypoints.values()]
        landmark_conf = float(np.mean(confidences)) if confidences else 0.0
        anchors = ("nose_center", "left_eye", "right_eye")
        anchor_confidences = [
            point.confidence
            for anchor in anchors
            if (point := landmarks.named(anchor)) is not None
        ]
        anchor_vis = float(np.mean(anchor_confidences)) if anchor_confidences else 0.0
    yaw = 1.0
    if landmarks is not None:
        left = landmarks.named("left_eye")
        right = landmarks.named("right_eye")
        nose = landmarks.named("nose_center")
        if left and right and nose:
            eye_mid = ((left.x + right.x) / 2.0, (left.y + right.y) / 2.0)
            offset = abs(nose.x - eye_mid[0]) / max(abs(right.x - left.x), 1.0)
            yaw = max(0.0, 1.0 - offset / 0.5)
    trunc = 1.0
    if bbox is not None and image_width and image_height:
        trunc = 1.0 - float(
            is_truncated(bbox, image_width=image_width, image_height=image_height)
        )
    resolution = 1.0
    if bbox is not None:
        resolution = min(1.0, max(bbox.width, bbox.height) / 112.0)
    overall = float(np.mean([landmark_conf, anchor_vis, yaw, resolution, trunc, blur]))
    return FaceQuality(
        landmark_confidence=landmark_conf,
        anchor_visibility=anchor_vis,
        yaw_roll_proxy=yaw,
        resolution=resolution,
        truncation=trunc,
        blur_estimate=blur,
        overall=overall,
    )


def score_nose_quality(
    nose_bbox: DetectionBox,
    *,
    anchor_agreement: float = 1.0,
    native_short_side: float = 0.0,
    blur: float = 0.5,
    specular_ratio: float = 0.0,
    image_width: int = 0,
    image_height: int = 0,
    muzzle_contamination: float = 0.0,
    support_coverage: float = 1.0,
) -> NoseQuality:
    trunc = 1.0
    if image_width and image_height:
        trunc = 1.0 - float(
            is_truncated(nose_bbox, image_width=image_width, image_height=image_height)
        )
    native = min(1.0, native_short_side / 224.0) if native_short_side > 0 else 0.5
    overall = float(
        np.mean(
            [
                anchor_agreement,
                native,
                blur,
                1.0 - specular_ratio,
                trunc,
                1.0 - muzzle_contamination,
                support_coverage,
            ]
        )
    )
    return NoseQuality(
        anchor_agreement=anchor_agreement,
        native_resolution=native,
        blur_estimate=blur,
        specular_ratio=specular_ratio,
        truncation=trunc,
        muzzle_contamination=muzzle_contamination,
        support_coverage=support_coverage,
        overall=overall,
    )


__all__ = [
    "DogQuality",
    "FaceQuality",
    "NoseQuality",
    "estimate_blur",
    "score_dog_quality",
    "score_face_quality",
    "score_nose_quality",
]
