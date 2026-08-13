"""Robust similarity alignment for the NoseID-v1 master grid."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from embedding.methods.nose.types import AlignedNose, NoseKeypoints


CANONICAL_KEYPOINTS = np.asarray(
    [
        (0.34, 0.50),
        (0.66, 0.50),
        (0.50, 0.24),
        (0.50, 0.76),
        (0.18, 0.50),
        (0.82, 0.50),
    ],
    dtype=np.float64,
)
_ANATOMICAL_WEIGHTS = np.asarray([2.0, 2.0, 1.0, 1.0, 0.5, 0.5])


class AlignmentError(ValueError):
    """Raised when geometric alignment fails conservative admission rules."""


@dataclass(frozen=True, slots=True)
class ResidualRegistration:
    transform: np.ndarray
    shift_xy: tuple[float, float]
    forward_shift_pixels: float
    residual: float
    response: float
    accepted: bool
    reason: str


def register_residual_translation(
    reference_luminance: np.ndarray,
    moving_luminance: np.ndarray,
    reference_valid_mask: np.ndarray | None = None,
    moving_valid_mask: np.ndarray | None = None,
    *,
    max_forward_shift: float = 12.0,
    max_residual: float = 0.25,
    min_response: float = 0.05,
) -> ResidualRegistration:
    """Estimate a residual translation and conservatively reject poor registrations."""
    reference = np.asarray(reference_luminance, dtype=np.float32)
    moving = np.asarray(moving_luminance, dtype=np.float32)
    if (
        reference.ndim != 2
        or moving.shape != reference.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(moving).all()
    ):
        raise AlignmentError("registration luminance must be finite same-shaped 2D arrays")
    if max_forward_shift <= 0.0 or max_residual <= 0.0 or not 0.0 <= min_response <= 1.0:
        raise AlignmentError("invalid residual registration thresholds")

    def mask_or_ones(value: np.ndarray | None, name: str) -> np.ndarray:
        if value is None:
            return np.ones(reference.shape, dtype=bool)
        result = np.asarray(value)
        if result.shape != reference.shape or not np.isfinite(result).all():
            raise AlignmentError(f"{name} mask must be finite and match luminance")
        return result > 0.5

    reference_mask = mask_or_ones(reference_valid_mask, "reference")
    moving_mask = mask_or_ones(moving_valid_mask, "moving")
    if int(reference_mask.sum()) < 16 or int(moving_mask.sum()) < 16:
        return ResidualRegistration(
            transform=np.eye(2, 3, dtype=np.float32),
            shift_xy=(0.0, 0.0),
            forward_shift_pixels=0.0,
            residual=1.0,
            response=0.0,
            accepted=False,
            reason="insufficient_valid_pixels",
        )

    reference_center = float(np.median(reference[reference_mask]))
    moving_center = float(np.median(moving[moving_mask]))
    prepared_reference = np.where(reference_mask, reference - reference_center, 0.0)
    prepared_moving = np.where(moving_mask, moving - moving_center, 0.0)
    window = cv2.createHanningWindow(
        (reference.shape[1], reference.shape[0]), cv2.CV_32F
    )
    shift, response = cv2.phaseCorrelate(
        prepared_reference.astype(np.float32),
        prepared_moving.astype(np.float32),
        window,
    )
    shift_x, shift_y = (float(shift[0]), float(shift[1]))
    forward = math.hypot(shift_x, shift_y)
    transform = np.asarray(((1.0, 0.0, -shift_x), (0.0, 1.0, -shift_y)), dtype=np.float32)
    warped = cv2.warpAffine(
        moving,
        transform,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask = cv2.warpAffine(
        moving_mask.astype(np.uint8),
        transform,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    overlap = reference_mask & warped_mask
    if int(overlap.sum()) < 16:
        residual = 1.0
        reason = "insufficient_overlap"
    else:
        residual = float(np.median(np.abs(reference[overlap] - warped[overlap])))
        if not np.isfinite(response) or response < min_response:
            reason = "low_phase_response"
        elif forward > max_forward_shift:
            reason = "forward_shift_exceeded"
        elif residual > max_residual:
            reason = "residual_exceeded"
        else:
            reason = "accepted"
    return ResidualRegistration(
        transform=transform,
        shift_xy=(shift_x, shift_y),
        forward_shift_pixels=float(forward),
        residual=float(residual),
        response=float(response) if np.isfinite(response) else 0.0,
        accepted=reason == "accepted",
        reason=reason,
    )


def _weighted_procrustes(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0.0:
        raise AlignmentError("alignment requires positive keypoint weight")
    source_mean = np.sum(source * weights[:, None], axis=0) / total
    target_mean = np.sum(target * weights[:, None], axis=0) / total
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (source_centered * weights[:, None]).T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) <= 0.0:
        raise AlignmentError("reflection is forbidden")
    denominator = float(np.sum(weights * np.sum(source_centered**2, axis=1)))
    if denominator <= 1e-12:
        raise AlignmentError("keypoint geometry is degenerate")
    scale = float(
        np.sum(weights * np.sum((source_centered @ rotation) * target_centered, axis=1))
        / denominator
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise AlignmentError("alignment scale must be positive")
    translation = target_mean - scale * (source_mean @ rotation)
    matrix = np.concatenate(
        [(scale * rotation).T, translation[:, None]], axis=1
    )
    return matrix


def estimate_similarity_transform(
    keypoints: NoseKeypoints | np.ndarray,
    *,
    output_size: int = 448,
) -> tuple[np.ndarray, float]:
    xyc = keypoints.xyc if isinstance(keypoints, NoseKeypoints) else np.asarray(keypoints)
    if xyc.shape != (6, 3) or not np.isfinite(xyc).all():
        raise AlignmentError("keypoints must have finite shape [6,3]")
    confidence = np.clip(xyc[:, 2].astype(np.float64), 0.0, 1.0)
    valid = confidence > 0.0
    if valid[:2].sum() < 2 or valid[2:4].sum() < 1:
        raise AlignmentError("both nostrils and one midline point are required")
    source = xyc[:, :2].astype(np.float64)[valid]
    target = CANONICAL_KEYPOINTS[valid] * float(output_size - 1)
    base_weights = _ANATOMICAL_WEIGHTS[valid] * confidence[valid]
    weights = base_weights.copy()
    matrix = np.empty((2, 3), dtype=np.float64)
    for _ in range(3):
        matrix = _weighted_procrustes(source, target, weights)
        predicted = source @ matrix[:, :2].T + matrix[:, 2]
        residuals = np.linalg.norm(predicted - target, axis=1)
        scale = max(float(np.median(residuals)), 1.0)
        huber = np.minimum(1.0, (1.5 * scale) / np.maximum(residuals, 1e-8))
        weights = base_weights * huber
    rotation_degrees = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    if abs(rotation_degrees) > 35.0:
        raise AlignmentError("estimated nose roll exceeds 35 degrees")
    predicted = source @ matrix[:, :2].T + matrix[:, 2]
    rms = math.sqrt(float(np.sum(weights * np.sum((predicted - target) ** 2, axis=1)) / weights.sum()))
    nostril_distance = float(
        np.linalg.norm(CANONICAL_KEYPOINTS[0] - CANONICAL_KEYPOINTS[1])
        * (output_size - 1)
    )
    normalized_residual = rms / nostril_distance
    if normalized_residual > 0.18:
        raise AlignmentError("normalized alignment residual exceeds 0.18")
    return matrix.astype(np.float32), float(normalized_residual)


def align_nose(
    image: np.ndarray,
    keypoints: NoseKeypoints,
    *,
    native_short_side: float,
    output_size: int = 448,
) -> AlignedNose:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or not np.isfinite(array).all():
        raise AlignmentError("source RGB must have finite shape [H,W,3]")
    rgb = array.astype(np.float32)
    if rgb.max(initial=0.0) > 1.0:
        rgb /= 255.0
    if np.any((rgb < 0.0) | (rgb > 1.0)):
        raise AlignmentError("source RGB must be in [0,1] or uint8")
    matrix, residual = estimate_similarity_transform(keypoints, output_size=output_size)
    local_scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
    interpolation = cv2.INTER_CUBIC if local_scale >= 1.0 else cv2.INTER_AREA
    warped = cv2.warpAffine(
        rgb,
        matrix,
        (output_size, output_size),
        flags=interpolation,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    xy1 = np.concatenate(
        [keypoints.xyc[:, :2], np.ones((6, 1), dtype=np.float32)], axis=1
    )
    aligned_xy = xy1 @ matrix.T
    aligned_keypoints = np.concatenate(
        [aligned_xy, keypoints.xyc[:, 2:3]], axis=1
    ).astype(np.float32)
    return AlignedNose(
        rgb=warped.transpose(2, 0, 1).astype(np.float32),
        keypoints_xyc=aligned_keypoints,
        transform=matrix,
        normalized_residual=residual,
        native_short_side=float(native_short_side),
    )


__all__ = [
    "AlignmentError",
    "CANONICAL_KEYPOINTS",
    "ResidualRegistration",
    "align_nose",
    "estimate_similarity_transform",
    "register_residual_translation",
]
