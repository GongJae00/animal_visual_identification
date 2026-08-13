"""Conservative multi-frame nose restoration from observed image evidence only."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
import torch

from embedding.methods.nose.signal.alignment import ResidualRegistration, register_residual_translation
from embedding.methods.nose.signal.frequency import classical_texture_descriptors
from embedding.methods.nose.signal.photometric import (
    glare_saturation_invalid_mask,
    linear_rgb_luminance,
    linear_to_srgb,
    masked_illumination_normalize,
    srgb_to_linear,
)


class RestorationError(ValueError):
    """Raised when observed frames cannot satisfy restoration admission rules."""


@dataclass(frozen=True, slots=True)
class RestorationConfig:
    registration_mode: str = "phase_translation"
    illumination_normalization: bool = True
    percentile_low: float = 0.02
    percentile_high: float = 0.98
    illumination_kernel_size: int = 31
    illumination_sigma: float = 9.0
    max_illumination_gain: float = 4.0
    glare_luminance: float = 0.90
    clipped_channel: float = 0.995
    dark_clip: float = 0.002
    clahe_clip_limit: float | None = None
    clahe_grid_size: tuple[int, int] = (8, 8)
    denoise_strength: float = 0.0
    deblock_strength: float = 0.0
    max_filter_delta: float = 0.03
    minimum_valid_fraction: float = 0.10
    max_forward_shift: float = 12.0
    max_registration_residual: float = 0.25
    minimum_phase_response: float = 0.05
    compute_descriptors: bool = True

    def __post_init__(self) -> None:
        if self.registration_mode not in {
            "phase_translation",
            "canonical_crop_identity",
        }:
            raise ValueError("unsupported restoration registration mode")
        if not isinstance(self.illumination_normalization, bool):
            raise ValueError("illumination_normalization must be boolean")
        numeric_values = (
            self.percentile_low,
            self.percentile_high,
            self.illumination_sigma,
            self.max_illumination_gain,
            self.glare_luminance,
            self.clipped_channel,
            self.dark_clip,
            self.denoise_strength,
            self.deblock_strength,
            self.max_filter_delta,
            self.minimum_valid_fraction,
            self.max_forward_shift,
            self.max_registration_residual,
            self.minimum_phase_response,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("restoration parameters must be finite")
        if not 0.0 <= self.percentile_low < self.percentile_high <= 1.0:
            raise ValueError("invalid restoration percentiles")
        if self.illumination_kernel_size < 3 or self.illumination_kernel_size % 2 != 1:
            raise ValueError("illumination kernel size must be odd and at least three")
        if self.illumination_sigma <= 0.0 or self.max_illumination_gain < 1.0:
            raise ValueError("invalid illumination parameters")
        if not 0.0 <= self.dark_clip < self.glare_luminance < self.clipped_channel <= 1.0:
            raise ValueError("invalid glare or clipping thresholds")
        if self.clahe_clip_limit is not None and (
            not math.isfinite(self.clahe_clip_limit) or self.clahe_clip_limit <= 0.0
        ):
            raise ValueError("CLAHE clip limit must be finite and positive")
        if len(self.clahe_grid_size) != 2 or any(value < 1 for value in self.clahe_grid_size):
            raise ValueError("CLAHE grid size must contain two positive integers")
        if not 0.0 <= self.denoise_strength <= 1.0:
            raise ValueError("denoise strength must be in [0,1]")
        if not 0.0 <= self.deblock_strength <= 1.0:
            raise ValueError("deblock strength must be in [0,1]")
        if not 0.0 <= self.max_filter_delta <= 0.25:
            raise ValueError("maximum filter delta must be in [0,0.25]")
        if not 0.0 < self.minimum_valid_fraction <= 1.0:
            raise ValueError("minimum valid fraction must be in (0,1]")
        if self.max_forward_shift <= 0.0 or self.max_registration_residual <= 0.0:
            raise ValueError("registration bounds must be positive")
        if not 0.0 <= self.minimum_phase_response <= 1.0:
            raise ValueError("minimum phase response must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrameRestorationDiagnostic:
    index: int
    accepted: bool
    reason: str
    valid_fraction: float
    transform: tuple[tuple[float, float, float], tuple[float, float, float]]
    shift_xy: tuple[float, float]
    forward_shift_pixels: float
    residual: float
    response: float
    fusion_weight: float
    input_sha256: str
    source_mask_sha256: str


@dataclass(frozen=True, slots=True)
class RestorationDiagnostics:
    schema_version: str
    image_shape: tuple[int, int, int]
    reference_index: int
    config: RestorationConfig
    frames: tuple[FrameRestorationDiagnostic, ...]
    accepted_indices: tuple[int, ...]
    leave_one_out_mean: float
    leave_one_out_max: float
    restored_sha256: str
    valid_mask_sha256: str
    observation_count_sha256: str
    temporal_variance_sha256: str
    descriptor_sha256: dict[str, str]
    implementations: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "image_shape": list(self.image_shape),
            "reference_index": self.reference_index,
            "config": self.config.to_dict(),
            "frames": [asdict(frame) for frame in self.frames],
            "accepted_indices": list(self.accepted_indices),
            "leave_one_out_mean": self.leave_one_out_mean,
            "leave_one_out_max": self.leave_one_out_max,
            "restored_sha256": self.restored_sha256,
            "valid_mask_sha256": self.valid_mask_sha256,
            "observation_count_sha256": self.observation_count_sha256,
            "temporal_variance_sha256": self.temporal_variance_sha256,
            "descriptor_sha256": dict(sorted(self.descriptor_sha256.items())),
            "implementations": dict(sorted(self.implementations.items())),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def provenance_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RestorationResult:
    restored_rgb: np.ndarray
    valid_mask: np.ndarray
    observation_count: np.ndarray
    temporal_variance: np.ndarray
    descriptors: dict[str, np.ndarray]
    diagnostics: RestorationDiagnostics


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_frames(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(frames)
    if source.ndim != 4 or source.shape[-1] != 3 or source.shape[0] < 1:
        raise RestorationError("frames must have shape [N,H,W,3]")
    if source.shape[1] < 8 or source.shape[2] < 8:
        raise RestorationError("restoration frames must be at least 8x8")
    if not np.issubdtype(source.dtype, np.number) or not np.isfinite(source).all():
        raise RestorationError("frames must contain finite numeric RGB values")
    if np.issubdtype(source.dtype, np.integer):
        if source.dtype != np.uint8:
            raise RestorationError("integer RGB frames must be uint8")
        rgb = source.astype(np.float32) / 255.0
    else:
        rgb = source.astype(np.float32, copy=True)
        if np.any((rgb < 0.0) | (rgb > 1.0)):
            raise RestorationError("floating RGB frames must be in [0,1]")
    return source, np.ascontiguousarray(rgb)


def _bounded_filter(
    image: np.ndarray, valid_mask: np.ndarray, config: RestorationConfig
) -> np.ndarray:
    output = image.copy()
    support = cv2.erode(
        valid_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    if config.denoise_strength > 0.0 and np.any(support):
        sigma_color = 0.02 + 0.08 * config.denoise_strength
        filtered = cv2.bilateralFilter(
            output, d=3, sigmaColor=sigma_color, sigmaSpace=1.0
        )
        delta = np.clip(
            filtered - output, -config.max_filter_delta, config.max_filter_delta
        )
        output[support] += config.denoise_strength * delta[support]
    if config.deblock_strength > 0.0 and np.any(support):
        blurred = cv2.GaussianBlur(output, (3, 3), 0.65)
        boundary = np.zeros(valid_mask.shape, dtype=bool)
        boundary[7::8, :] = True
        boundary[8::8, :] = True
        boundary[:, 7::8] = True
        boundary[:, 8::8] = True
        boundary &= support
        delta = np.clip(
            blurred - output, -config.max_filter_delta, config.max_filter_delta
        )
        output[boundary] += config.deblock_strength * delta[boundary]
    output[~valid_mask] = 0.0
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def _apply_clahe(
    linear_rgb: np.ndarray, valid_mask: np.ndarray, config: RestorationConfig
) -> np.ndarray:
    if config.clahe_clip_limit is None:
        return linear_rgb
    luminance = np.sum(
        linear_rgb * np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32), axis=2
    )
    clahe = cv2.createCLAHE(
        clipLimit=float(config.clahe_clip_limit),
        tileGridSize=config.clahe_grid_size,
    )
    enhanced = clahe.apply(np.rint(luminance * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
    gain = np.clip(enhanced / np.maximum(luminance, 1e-6), 0.5, 2.0)
    result = np.clip(linear_rgb * gain[..., None], 0.0, 1.0)
    result[~valid_mask] = 0.0
    return result.astype(np.float32)


def robust_linear_fusion(
    linear_frames: np.ndarray,
    valid_masks: np.ndarray,
    frame_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse observed linear-light samples with a per-channel weighted median."""
    frames = np.asarray(linear_frames, dtype=np.float32)
    masks = np.asarray(valid_masks) > 0.5
    if (
        frames.ndim != 4
        or frames.shape[-1] != 3
        or masks.shape != frames.shape[:3]
        or frames.shape[0] < 1
        or not np.isfinite(frames).all()
    ):
        raise RestorationError("fusion inputs must be finite [N,H,W,3] and [N,H,W]")
    if frame_weights is None:
        scalar_weights = np.ones(frames.shape[0], dtype=np.float32)
    else:
        scalar_weights = np.asarray(frame_weights, dtype=np.float32)
        if (
            scalar_weights.shape != (frames.shape[0],)
            or not np.isfinite(scalar_weights).all()
            or np.any(scalar_weights < 0.0)
            or not np.any(scalar_weights > 0.0)
        ):
            raise RestorationError("fusion weights must be finite non-negative [N]")
    pixel_weights = masks.astype(np.float32) * scalar_weights[:, None, None]
    total_weight = pixel_weights.sum(axis=0)
    fused = np.zeros(frames.shape[1:], dtype=np.float32)
    for channel in range(3):
        values = frames[..., channel]
        order = np.argsort(values, axis=0, kind="stable")
        sorted_values = np.take_along_axis(values, order, axis=0)
        sorted_weights = np.take_along_axis(pixel_weights, order, axis=0)
        cumulative = np.cumsum(sorted_weights, axis=0)
        median_index = np.argmax(cumulative >= (0.5 * total_weight)[None], axis=0)
        fused[..., channel] = np.take_along_axis(
            sorted_values, median_index[None], axis=0
        )[0]
    valid = total_weight > 0.0
    fused[~valid] = 0.0
    squared_error = np.mean((frames - fused[None]) ** 2, axis=3)
    variance = np.divide(
        np.sum(pixel_weights * squared_error, axis=0),
        total_weight,
        out=np.zeros_like(total_weight),
        where=valid,
    ).astype(np.float32)
    count = masks.sum(axis=0).astype(np.uint16)
    return fused, count, variance


def leave_one_out_stability(
    linear_frames: np.ndarray,
    valid_masks: np.ndarray,
    frame_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return maximum leave-one-frame-out deviation where alternatives exist."""
    frames = np.asarray(linear_frames, dtype=np.float32)
    masks = np.asarray(valid_masks) > 0.5
    if frames.shape[0] < 2:
        return np.zeros(frames.shape[1:3], dtype=np.float32)
    weights = (
        np.ones(frames.shape[0], dtype=np.float32)
        if frame_weights is None
        else np.asarray(frame_weights, dtype=np.float32)
    )
    full, count, _ = robust_linear_fusion(frames, masks, weights)
    stability = np.zeros(frames.shape[1:3], dtype=np.float32)
    for index in range(frames.shape[0]):
        keep = np.arange(frames.shape[0]) != index
        candidate, candidate_count, _ = robust_linear_fusion(
            frames[keep], masks[keep], weights[keep]
        )
        comparable = (count > 1) & (candidate_count > 0)
        difference = np.max(np.abs(candidate - full), axis=2)
        stability[comparable] = np.maximum(
            stability[comparable], difference[comparable]
        )
    return stability


def redegradation_consistency(
    restored_linear: np.ndarray,
    observed_linear: np.ndarray,
    observed_mask: np.ndarray,
    *,
    downsample_factor: int = 1,
    blur_sigma: float = 0.0,
) -> float:
    """Measure observed-pixel RMSE after an explicit deterministic re-degradation."""
    restored = np.asarray(restored_linear, dtype=np.float32)
    observed = np.asarray(observed_linear, dtype=np.float32)
    mask = np.asarray(observed_mask) > 0.5
    if restored.shape != observed.shape or restored.ndim != 3 or restored.shape[2] != 3:
        raise RestorationError("consistency images must be same-shaped HxWx3 arrays")
    if mask.shape != restored.shape[:2] or downsample_factor < 1 or blur_sigma < 0.0:
        raise RestorationError("invalid consistency mask or degradation parameters")
    if not np.isfinite(restored).all() or not np.isfinite(observed).all():
        raise RestorationError("consistency images must be finite")
    degraded = restored.copy()
    if blur_sigma > 0.0:
        degraded = cv2.GaussianBlur(degraded, (0, 0), blur_sigma)
    if downsample_factor > 1:
        height, width = restored.shape[:2]
        small = cv2.resize(
            degraded,
            (max(1, width // downsample_factor), max(1, height // downsample_factor)),
            interpolation=cv2.INTER_AREA,
        )
        degraded = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean((degraded[mask] - observed[mask]) ** 2)))


def restore_nose_frames(
    frames: np.ndarray,
    source_valid_masks: np.ndarray | None = None,
    *,
    config: RestorationConfig | None = None,
) -> RestorationResult:
    """Restore a frame stack without synthesizing any unobserved pixel."""
    settings = config or RestorationConfig()
    source, rgb = _validate_frames(frames)
    frame_count, height, width, _ = rgb.shape
    if source_valid_masks is None:
        source_masks = np.ones((frame_count, height, width), dtype=bool)
    else:
        raw_masks = np.asarray(source_valid_masks)
        if raw_masks.shape != (frame_count, height, width) or not np.isfinite(raw_masks).all():
            raise RestorationError("source valid masks must be finite [N,H,W]")
        source_masks = raw_masks > 0.5

    tensor = torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy())
    source_mask_tensor = torch.from_numpy(source_masks[:, None].copy())
    invalid = glare_saturation_invalid_mask(
        tensor,
        source_mask_tensor,
        glare_luminance=settings.glare_luminance,
        clipped_channel=settings.clipped_channel,
        dark_clip=settings.dark_clip,
    )
    valid_masks = (~invalid[:, 0]).cpu().numpy()
    if settings.illumination_normalization:
        normalized, _ = masked_illumination_normalize(
            tensor,
            (~invalid).to(dtype=tensor.dtype),
            low=settings.percentile_low,
            high=settings.percentile_high,
            kernel_size=settings.illumination_kernel_size,
            sigma=settings.illumination_sigma,
            max_gain=settings.max_illumination_gain,
        )
    else:
        normalized = srgb_to_linear(tensor).float()
        normalized *= (~invalid).to(dtype=normalized.dtype)
    linear_frames = normalized.cpu().numpy().transpose(0, 2, 3, 1)
    for index in range(frame_count):
        linear_frames[index] = _bounded_filter(
            _apply_clahe(linear_frames[index], valid_masks[index], settings),
            valid_masks[index],
            settings,
        )

    valid_fractions = valid_masks.reshape(frame_count, -1).mean(axis=1)
    eligible = np.flatnonzero(valid_fractions >= settings.minimum_valid_fraction)
    if eligible.size == 0:
        raise RestorationError("no frame has sufficient observed, non-glare pixels")
    reference_index = int(eligible[np.argmax(valid_fractions[eligible])])
    luminance_tensor = linear_rgb_luminance(
        torch.from_numpy(linear_frames.transpose(0, 3, 1, 2).copy())
    )
    luminance = luminance_tensor[:, 0].numpy()

    registrations: list[ResidualRegistration] = []
    identity = np.eye(2, 3, dtype=np.float32)
    for index in range(frame_count):
        if valid_fractions[index] < settings.minimum_valid_fraction:
            registrations.append(
                ResidualRegistration(identity.copy(), (0.0, 0.0), 0.0, 1.0, 0.0, False, "insufficient_valid_fraction")
            )
        elif index == reference_index:
            registrations.append(
                ResidualRegistration(identity.copy(), (0.0, 0.0), 0.0, 0.0, 1.0, True, "reference")
            )
        elif settings.registration_mode == "canonical_crop_identity":
            registrations.append(
                ResidualRegistration(
                    identity.copy(),
                    (0.0, 0.0),
                    0.0,
                    0.0,
                    1.0,
                    True,
                    "canonical_crop_identity",
                )
            )
        else:
            registrations.append(
                register_residual_translation(
                    luminance[reference_index],
                    luminance[index],
                    valid_masks[reference_index],
                    valid_masks[index],
                    max_forward_shift=settings.max_forward_shift,
                    max_residual=settings.max_registration_residual,
                    min_response=settings.minimum_phase_response,
                )
            )

    aligned_frames: list[np.ndarray] = []
    aligned_masks: list[np.ndarray] = []
    accepted_indices: list[int] = []
    weights: list[float] = []
    for index, registration in enumerate(registrations):
        if not registration.accepted:
            continue
        aligned = cv2.warpAffine(
            linear_frames[index], registration.transform, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        aligned_mask = cv2.warpAffine(
            valid_masks[index].astype(np.uint8), registration.transform, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(bool)
        aligned[~aligned_mask] = 0.0
        weight = 1.0 if index == reference_index else max(0.05, min(1.0, registration.response)) / (1.0 + 4.0 * registration.residual)
        aligned_frames.append(aligned.astype(np.float32))
        aligned_masks.append(aligned_mask)
        accepted_indices.append(index)
        weights.append(float(weight))

    aligned_array = np.stack(aligned_frames)
    aligned_mask_array = np.stack(aligned_masks)
    weight_array = np.asarray(weights, dtype=np.float32)
    fused_linear, observation_count, temporal_variance = robust_linear_fusion(
        aligned_array, aligned_mask_array, weight_array
    )
    fused_valid = observation_count > 0
    stability = leave_one_out_stability(
        aligned_array, aligned_mask_array, weight_array
    )
    comparable = observation_count > 1
    stability_mean = float(stability[comparable].mean()) if np.any(comparable) else 0.0
    stability_max = float(stability[comparable].max()) if np.any(comparable) else 0.0
    fused_luminance = np.sum(
        fused_linear * np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32), axis=2
    )
    descriptors = (
        classical_texture_descriptors(fused_luminance, fused_valid)
        if settings.compute_descriptors
        else {}
    )
    restored_tensor = linear_to_srgb(
        torch.from_numpy(fused_linear.transpose(2, 0, 1)[None].copy())
    )
    restored_rgb = restored_tensor[0].numpy().transpose(1, 2, 0).astype(np.float32)
    restored_rgb[~fused_valid] = 0.0

    frame_diagnostics: list[FrameRestorationDiagnostic] = []
    accepted_weight = dict(zip(accepted_indices, weights, strict=True))
    for index, registration in enumerate(registrations):
        frame_diagnostics.append(
            FrameRestorationDiagnostic(
                index=index,
                accepted=registration.accepted,
                reason=registration.reason,
                valid_fraction=float(valid_fractions[index]),
                transform=tuple(tuple(float(value) for value in row) for row in registration.transform),  # type: ignore[arg-type]
                shift_xy=registration.shift_xy,
                forward_shift_pixels=registration.forward_shift_pixels,
                residual=registration.residual,
                response=registration.response,
                fusion_weight=float(accepted_weight.get(index, 0.0)),
                input_sha256=_array_sha256(source[index]),
                source_mask_sha256=_array_sha256(source_masks[index]),
            )
        )
    diagnostics = RestorationDiagnostics(
        schema_version="cvi.nose_restoration.v1",
        image_shape=(height, width, 3),
        reference_index=reference_index,
        config=settings,
        frames=tuple(frame_diagnostics),
        accepted_indices=tuple(accepted_indices),
        leave_one_out_mean=stability_mean,
        leave_one_out_max=stability_max,
        restored_sha256=_array_sha256(restored_rgb),
        valid_mask_sha256=_array_sha256(fused_valid),
        observation_count_sha256=_array_sha256(observation_count),
        temporal_variance_sha256=_array_sha256(temporal_variance),
        descriptor_sha256={name: _array_sha256(value) for name, value in descriptors.items()},
        implementations={
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "registration": settings.registration_mode,
            "fusion": "per_channel_weighted_median_linear_srgb",
        },
    )
    if not (
        np.isfinite(restored_rgb).all()
        and np.isfinite(temporal_variance).all()
        and all(np.isfinite(value).all() for value in descriptors.values())
    ):
        raise RuntimeError("restoration produced non-finite output")
    return RestorationResult(
        restored_rgb=restored_rgb,
        valid_mask=fused_valid,
        observation_count=observation_count,
        temporal_variance=temporal_variance,
        descriptors=descriptors,
        diagnostics=diagnostics,
    )


__all__ = [
    "FrameRestorationDiagnostic",
    "RestorationConfig",
    "RestorationDiagnostics",
    "RestorationError",
    "RestorationResult",
    "leave_one_out_stability",
    "redegradation_consistency",
    "restore_nose_frames",
    "robust_linear_fusion",
]
