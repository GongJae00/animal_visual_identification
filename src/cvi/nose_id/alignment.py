"""Robust similarity alignment for the NoseID-v1 master grid."""

from __future__ import annotations

import math

import cv2
import numpy as np

from cvi.nose_id.types import AlignedNose, NoseKeypoints


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
    pass


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
    "align_nose",
    "estimate_similarity_transform",
]
