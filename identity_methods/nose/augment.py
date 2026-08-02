"""Paired NoseID-v1 augmentation with explicit degradation supervision."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class NoseAugmentedView:
    rgb: torch.Tensor
    keypoints: torch.Tensor
    semantic_probability: torch.Tensor
    invalid_probability: torch.Tensor
    source_valid_probability: torch.Tensor
    degradation_target: torch.Tensor


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3.0 * sigma))
    coordinate = torch.arange(-radius, radius + 1, dtype=image.dtype)
    kernel = torch.exp(-(coordinate**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    channels = image.shape[0]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = horizontal.transpose(-1, -2)
    value = F.conv2d(
        F.pad(image[None], (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=channels,
    )
    return F.conv2d(
        F.pad(value, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=channels,
    )[0]


class NoseIdentityAugment:
    def __init__(self, seed: int = 0) -> None:
        self.generator = torch.Generator().manual_seed(seed)

    def _rand(self) -> float:
        return float(torch.rand((), generator=self.generator))

    def _ellipse_mask(
        self,
        support: torch.Tensor,
        *,
        area_fraction: float,
        soft: bool,
    ) -> torch.Tensor:
        indices = torch.nonzero(support > 0.5, as_tuple=False)
        if len(indices) == 0:
            return torch.zeros_like(support)
        center = indices[
            int(torch.randint(len(indices), (1,), generator=self.generator))
        ]
        target_area = max(float(len(indices)) * area_fraction, 1.0)
        aspect = 0.5 + self._rand() * 1.5
        radius_x = math.sqrt(target_area * aspect / math.pi)
        radius_y = target_area / (math.pi * max(radius_x, 1e-6))
        y, x = torch.meshgrid(
            torch.arange(448, dtype=support.dtype),
            torch.arange(448, dtype=support.dtype),
            indexing="ij",
        )
        distance = (
            ((x - center[1]) / max(radius_x, 1.0)).square()
            + ((y - center[0]) / max(radius_y, 1.0)).square()
        )
        mask = torch.exp(-2.0 * distance) if soft else (distance <= 1.0).to(support.dtype)
        return mask * support

    @staticmethod
    def _normalize_semantic(foreground: torch.Tensor) -> torch.Tensor:
        foreground = foreground.clamp(0.0, 1.0)
        context = (1.0 - foreground.sum(dim=0, keepdim=True)).clamp(0.0, 1.0)
        semantic = torch.cat([context, foreground], dim=0)
        return semantic / semantic.sum(dim=0, keepdim=True).clamp_min(1e-6)

    def __call__(
        self,
        image: torch.Tensor,
        keypoints: torch.Tensor,
        semantic_probability: torch.Tensor,
        invalid_probability: torch.Tensor,
        source_valid_probability: torch.Tensor,
    ) -> NoseAugmentedView:
        if image.shape != (3, 448, 448) or not torch.isfinite(image).all():
            raise ValueError("NoseID augmentation expects finite [3,448,448] RGB")
        if keypoints.shape != (6, 3):
            raise ValueError("NoseID augmentation expects keypoints [6,3]")
        if (
            semantic_probability.shape != (3, 448, 448)
            or invalid_probability.shape != (1, 448, 448)
            or source_valid_probability.shape != (1, 448, 448)
        ):
            raise ValueError("NoseID augmentation mask shape differs")
        if image.device.type != "cpu":
            raise ValueError("NoseID augmentation is a CPU dataloader transform")

        angle_degrees = self._rand() * 24.0 - 12.0
        angle = math.radians(angle_degrees)
        scale = 0.92 + self._rand() * 0.16
        tx = self._rand() * 0.08 - 0.04
        ty = self._rand() * 0.08 - 0.04
        cosine, sine = math.cos(angle), math.sin(angle)
        theta = image.new_tensor(
            ((scale * cosine, -scale * sine, tx), (scale * sine, scale * cosine, ty))
        )[None]
        grid = F.affine_grid(theta, (1, 3, 448, 448), align_corners=True)
        rgb = F.grid_sample(
            image[None], grid, mode="bilinear", padding_mode="reflection", align_corners=True
        )[0]
        foreground = F.grid_sample(
            semantic_probability[None, 1:],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0]
        semantic = self._normalize_semantic(foreground)
        source_valid = F.grid_sample(
            source_valid_probability[None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0].clamp(0, 1)
        invalid = F.grid_sample(
            invalid_probability[None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0].clamp(0, 1)
        invalid = torch.maximum(invalid, 1.0 - source_valid)

        transform = torch.eye(3, dtype=image.dtype)
        transform[:2] = theta[0]
        inverse = torch.linalg.inv(transform)
        normalized_xy = keypoints[:, :2] / 447.0 * 2.0 - 1.0
        homogeneous = torch.cat([normalized_xy, torch.ones((6, 1))], dim=1)
        transformed_xy = homogeneous @ inverse.T
        transformed_xy = (transformed_xy[:, :2] + 1.0) * 0.5 * 447.0
        inside = (
            (transformed_xy[:, 0] >= 0.0)
            & (transformed_xy[:, 0] <= 447.0)
            & (transformed_xy[:, 1] >= 0.0)
            & (transformed_xy[:, 1] <= 447.0)
        )
        confidence = keypoints[:, 2] * inside.to(keypoints.dtype)
        transformed_keypoints = torch.cat([transformed_xy, confidence[:, None]], dim=1)
        pose_severity = max(
            abs(angle_degrees) / 12.0,
            abs(math.log(scale)) / math.log(1.08),
            abs(tx) / 0.04,
            abs(ty) / 0.04,
        )

        downsample_severity = 0.0
        if self._rand() < 0.25:
            downsample_scale = 0.55 + self._rand() * 0.35
            size = max(1, round(448 * downsample_scale))
            rgb = F.interpolate(rgb[None], size=(size, size), mode="area")
            rgb = F.interpolate(
                rgb, size=(448, 448), mode="bicubic", align_corners=False, antialias=True
            )[0]
            downsample_severity = (1.0 - downsample_scale) / 0.45

        blur_severity = 0.0
        if self._rand() < 0.35:
            sigma = 0.2 + self._rand() * 0.8
            rgb = _gaussian_blur(rgb, sigma)
            blur_severity = sigma

        surface = semantic[1] * source_valid[0]
        specular_severity = 0.0
        if self._rand() < 0.20:
            requested_fraction = 0.01 + self._rand() * 0.05
            blob_count = 1 + int(self._rand() * 3.0)
            specular = torch.zeros_like(surface)
            for _ in range(blob_count):
                specular = torch.maximum(
                    specular,
                    self._ellipse_mask(
                        surface,
                        area_fraction=requested_fraction / blob_count,
                        soft=True,
                    ),
                )
            rgb = rgb * (1.0 - specular[None]) + specular[None]
            invalid = torch.maximum(invalid, specular[None])
            actual = float(specular.sum() / surface.sum().clamp_min(1.0))
            specular_severity = min(actual / 0.06, 1.0)

        occlusion_severity = 0.0
        if self._rand() < 0.10:
            requested_fraction = 0.01 + self._rand() * 0.07
            occlusion = self._ellipse_mask(
                surface, area_fraction=requested_fraction, soft=False
            )
            valid_rgb = rgb[:, surface > 0.5]
            fill = (
                valid_rgb.mean(dim=1, keepdim=True).view(3, 1, 1)
                if valid_rgb.numel()
                else rgb.mean(dim=(1, 2), keepdim=True)
            )
            noise = torch.randn(rgb.shape, generator=self.generator) * 0.005
            replacement = (fill + noise).clamp(0, 1)
            rgb = rgb * (1.0 - occlusion[None]) + replacement * occlusion[None]
            invalid = torch.maximum(invalid, occlusion[None])
            actual = float(occlusion.sum() / surface.sum().clamp_min(1.0))
            occlusion_severity = min(actual / 0.08, 1.0)

        mask_severity = 0.0
        if self._rand() < 0.20:
            radius = 1 + int(self._rand() >= 0.5)
            kernel = 2 * radius + 1
            nostril = semantic[2:3]
            nasal_surface = semantic[1:2]
            if self._rand() < 0.5:
                nostril = F.max_pool2d(nostril[None], kernel, stride=1, padding=radius)[0]
                nasal_surface = F.max_pool2d(
                    nasal_surface[None], kernel, stride=1, padding=radius
                )[0]
            else:
                nostril = -F.max_pool2d(
                    -nostril[None], kernel, stride=1, padding=radius
                )[0]
                nasal_surface = -F.max_pool2d(
                    -nasal_surface[None], kernel, stride=1, padding=radius
                )[0]
            nostril = nostril * source_valid
            nasal_surface = nasal_surface * source_valid * (1.0 - nostril)
            semantic = self._normalize_semantic(torch.cat([nasal_surface, nostril], dim=0))
            mask_severity = radius / 2.0

        if self._rand() < 0.15:
            sigma = self._rand() * 0.01
            rgb = rgb + torch.randn(rgb.shape, generator=self.generator) * sigma

        exposure = 2.0 ** (self._rand() * 0.6 - 0.3)
        gamma = 0.9 + self._rand() * 0.2
        gains = 0.9 + torch.rand((3, 1, 1), generator=self.generator) * 0.2
        rgb = (rgb.clamp(0, 1).pow(gamma) * exposure * gains).clamp(0, 1)
        degradation = torch.tensor(
            (
                blur_severity,
                downsample_severity,
                specular_severity,
                occlusion_severity,
                min(pose_severity, 1.0),
                mask_severity,
            ),
            dtype=torch.float32,
        ).clamp(0, 1)
        return NoseAugmentedView(
            rgb=rgb,
            keypoints=transformed_keypoints,
            semantic_probability=semantic,
            invalid_probability=torch.maximum(invalid, 1.0 - source_valid),
            source_valid_probability=source_valid,
            degradation_target=degradation,
        )

    def pair(
        self,
        image: torch.Tensor,
        keypoints: torch.Tensor,
        semantic_probability: torch.Tensor,
        invalid_probability: torch.Tensor,
        source_valid_probability: torch.Tensor,
    ) -> tuple[NoseAugmentedView, NoseAugmentedView]:
        return (
            self(
                image,
                keypoints,
                semantic_probability,
                invalid_probability,
                source_valid_probability,
            ),
            self(
                image,
                keypoints,
                semantic_probability,
                invalid_probability,
                source_valid_probability,
            ),
        )


__all__ = ["NoseAugmentedView", "NoseIdentityAugment"]
