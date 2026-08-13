"""Geometry-seeded binary region candidates from dense foundation features."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from parsing.regions.foundation_dense_runtime import FoundationImageTransform


@dataclass(frozen=True, slots=True)
class FoundationBinaryCandidate:
    source_probability: np.ndarray
    source_hard_mask: np.ndarray
    source_geometry_support: np.ndarray
    patch_probability: np.ndarray
    confidence: float
    algorithm: str = (
        "DENSE_COSINE_POSITIVE_KEYPOINT_PROTOTYPE_MINUS_OUTSIDE_ROI_BACKGROUND_"
        "WITH_ROBUST_RANK_THRESHOLD"
    )


def derive_binary_foundation_candidate(
    features: np.ndarray,
    *,
    transform: FoundationImageTransform,
    source_validity: np.ndarray,
    box_xyxy: Sequence[float],
    positive_points_xy: Sequence[Sequence[float]],
    retained_fraction: float = 0.65,
) -> FoundationBinaryCandidate:
    """Create a candidate, never a verified semantic mask."""

    values = np.asarray(features, dtype=np.float32)
    validity = np.asarray(source_validity, dtype=bool)
    if values.ndim != 3 or validity.shape != values.shape[:2]:
        raise ValueError("foundation candidate features and validity shapes differ")
    if not np.isfinite(values).all():
        raise ValueError("foundation candidate features must be finite")
    if not 0.0 < retained_fraction < 1.0:
        raise ValueError("foundation retained fraction must be in (0,1)")
    box = _box(box_xyxy, width=transform.source_width, height=transform.source_height)
    points = tuple(
        _point(item, width=transform.source_width, height=transform.source_height)
        for item in positive_points_xy
    )
    if not points:
        raise ValueError("foundation candidate requires positive points")
    grid_height, grid_width = values.shape[:2]
    patch_y = (np.arange(grid_height, dtype=np.float32) + 0.5) * (
        transform.canvas_size / grid_height
    )
    patch_x = (np.arange(grid_width, dtype=np.float32) + 0.5) * (
        transform.canvas_size / grid_width
    )
    yy, xx = np.meshgrid(patch_y, patch_x, indexing="ij")
    x1, y1 = transform.source_to_canvas(box[0], box[1])
    x2, y2 = transform.source_to_canvas(box[2], box[3])
    geometry = validity & (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
    if not np.any(geometry):
        raise ValueError("foundation candidate geometry contains no patch centers")
    patch_radius = 1.6 * transform.canvas_size / grid_width
    seeds = np.zeros_like(validity)
    for source_x, source_y in points:
        canvas_x, canvas_y = transform.source_to_canvas(source_x, source_y)
        seeds |= (xx - canvas_x) ** 2 + (yy - canvas_y) ** 2 <= patch_radius**2
    seeds &= geometry
    if not np.any(seeds):
        raise ValueError("foundation candidate positive points contain no valid patches")
    background = validity & ~geometry
    if not np.any(background):
        background = validity & _grid_border(validity.shape) & ~seeds
    if not np.any(background):
        raise ValueError("foundation candidate contains no background support")
    foreground_prototype = _normalized_mean(values[seeds])
    background_prototype = _normalized_mean(values[background])
    score = values @ foreground_prototype - values @ background_prototype
    threshold = float(np.quantile(score[geometry], 1.0 - retained_fraction))
    centered = score[geometry] - threshold
    scale = max(float(np.quantile(centered, 0.75) - np.quantile(centered, 0.25)), 0.05)
    probability = 1.0 / (1.0 + np.exp(-np.clip((score - threshold) / scale, -30.0, 30.0)))
    probability[~geometry] = 0.0
    patch_mask = probability >= 0.5
    patch_mask |= seeds
    patch_mask = _seed_component(patch_mask, seeds)
    probability[~patch_mask] *= 0.25
    source_probability = _to_source(probability, transform, interpolation=cv2.INTER_CUBIC)
    source_geometry = _source_box_mask(box, transform.source_width, transform.source_height)
    source_probability = np.clip(source_probability, 0.0, 1.0)
    source_probability[~source_geometry] = 0.0
    source_mask = source_probability >= 0.5
    confidence = float(np.mean(np.maximum(source_probability, 1.0 - source_probability)))
    return FoundationBinaryCandidate(
        source_probability=source_probability.astype(np.float32, copy=False),
        source_hard_mask=source_mask,
        source_geometry_support=source_geometry,
        patch_probability=probability.astype(np.float32, copy=False),
        confidence=confidence,
    )


def _to_source(
    values: np.ndarray,
    transform: FoundationImageTransform,
    *,
    interpolation: int,
) -> np.ndarray:
    canvas = cv2.resize(
        values,
        (transform.canvas_size, transform.canvas_size),
        interpolation=interpolation,
    )
    cropped = canvas[
        transform.pad_top : transform.pad_top + transform.resized_height,
        transform.pad_left : transform.pad_left + transform.resized_width,
    ]
    return cv2.resize(
        cropped,
        (transform.source_width, transform.source_height),
        interpolation=interpolation,
    )


def _seed_component(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    seed_labels = labels[seeds]
    seed_labels = seed_labels[seed_labels > 0]
    if not seed_labels.size:
        return np.zeros_like(mask)
    selected = int(np.bincount(seed_labels).argmax())
    return labels == selected


def _normalized_mean(values: np.ndarray) -> np.ndarray:
    result = values.mean(axis=0)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("foundation candidate prototype has zero norm")
    return result / norm


def _grid_border(shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    result[[0, -1], :] = True
    result[:, [0, -1]] = True
    return result


def _source_box_mask(
    box: tuple[float, float, float, float], width: int, height: int
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return (
        (xx + 0.5 >= box[0])
        & (xx + 0.5 <= box[2])
        & (yy + 0.5 >= box[1])
        & (yy + 0.5 <= box[3])
    )


def _box(
    value: Sequence[float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    if len(value) != 4 or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in value
    ):
        raise ValueError("foundation candidate box differs")
    x1, y1, x2, y2 = (float(item) for item in value)
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError("foundation candidate box lies outside source image")
    return x1, y1, x2, y2


def _point(value: Sequence[float], *, width: int, height: int) -> tuple[float, float]:
    if len(value) != 2 or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in value
    ):
        raise ValueError("foundation candidate point differs")
    x, y = (float(item) for item in value)
    if not 0.0 <= x <= width or not 0.0 <= y <= height:
        raise ValueError("foundation candidate point lies outside source image")
    return x, y


__all__ = ["FoundationBinaryCandidate", "derive_binary_foundation_candidate"]
