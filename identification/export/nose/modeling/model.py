"""NoseID-v1 frozen-DINO dual-stream recognition model."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F

from identification.export.nose.signal.frequency import FixedFrequencyBank
from identification.export.nose.modeling.pooling import AnatomicalPartPool
from identification.export.nose.modeling.segmentation import FactorizedNoseSegmenter


class _LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = _LayerNorm2d(dim)
        self.expand = nn.Conv2d(dim, 4 * dim, 1)
        self.project = nn.Conv2d(4 * dim, dim, 1)
        self.layer_scale = nn.Parameter(torch.full((1, dim, 1, 1), 1e-6))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(value)
        value = self.norm(value)
        value = self.project(F.gelu(self.expand(value)))
        return residual + self.layer_scale * value


class TextureConvNeXtS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dims = (64, 128, 256, 384)
        depths = (2, 2, 6, 2)
        self.stem = nn.Sequential(nn.Conv2d(11, dims[0], 4, stride=4), _LayerNorm2d(dims[0]))
        self.stages = nn.ModuleList(
            [nn.Sequential(*[_ConvNeXtBlock(dim) for _ in range(depth)]) for dim, depth in zip(dims, depths, strict=True)]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(_LayerNorm2d(dims[index]), nn.Conv2d(dims[index], dims[index + 1], 2, stride=2))
                for index in range(3)
            ]
        )
        self.pool = AnatomicalPartPool(384, 256)

    def forward(self, value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        value = self.stem(value)
        for index, stage in enumerate(self.stages):
            mask = F.interpolate(valid_mask, size=value.shape[-2:], mode="area")
            value = stage(value) * mask
            if index < len(self.downsamples):
                value = self.downsamples[index](value)
        mask = F.interpolate(valid_mask, size=value.shape[-2:], mode="area")
        return self.pool(value, mask)


class RGBLocalSemanticStream(nn.Module):
    hidden_indices = (8, 9, 10, 11)

    def __init__(self) -> None:
        super().__init__()
        self.layer_mix_logits = nn.Parameter(torch.zeros(4))
        self.hidden_norms = nn.ModuleList([nn.LayerNorm(384) for _ in range(4)])
        self.side_projection = nn.Sequential(
            nn.Linear(396, 384), nn.GELU(), nn.LayerNorm(384)
        )
        self.pool = AnatomicalPartPool(384, 256)

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        semantic_probability: torch.Tensor,
        invalid_probability: torch.Tensor,
        aligned_keypoints: torch.Tensor,
        source_valid_probability: torch.Tensor,
    ) -> torch.Tensor:
        mixture = torch.softmax(self.layer_mix_logits, dim=0)
        tokens = sum(
            mixture[index] * norm(hidden_states[layer + 1][:, 1:])
            for index, (layer, norm) in enumerate(zip(self.hidden_indices, self.hidden_norms, strict=True))
        )
        if tokens.shape[1:] != (576, 384):
            raise ValueError("DINO patch tokens must have shape [B,576,384]")
        semantic_patch = F.adaptive_avg_pool2d(semantic_probability, (24, 24))
        invalid_patch = F.adaptive_avg_pool2d(invalid_probability, (24, 24))
        coordinate = torch.linspace(0.0, 1.0, 24, device=tokens.device, dtype=tokens.dtype)
        y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
        xy = torch.stack([x, y], dim=-1).reshape(1, 576, 2).expand(tokens.shape[0], -1, -1)
        keypoint_xy = aligned_keypoints[:, :, :2] / 447.0
        distances = torch.linalg.vector_norm(xy[:, :, None] - keypoint_xy[:, None], dim=-1)
        side = torch.cat(
            [
                xy,
                semantic_patch.flatten(2).transpose(1, 2),
                invalid_patch.flatten(2).transpose(1, 2),
                distances,
            ],
            dim=-1,
        )
        features = self.side_projection(torch.cat([tokens, side], dim=-1))
        features = features.transpose(1, 2).reshape(-1, 384, 24, 24)
        rgb_mask = (
            semantic_probability[:, 1:2] + 0.35 * semantic_probability[:, 2:3]
        ) * (1.0 - 0.5 * invalid_probability) * source_valid_probability
        return self.pool(features, F.adaptive_avg_pool2d(rgb_mask, (24, 24)))


def _shape_features(
    keypoints: torch.Tensor,
    semantic_probability: torch.Tensor,
) -> torch.Tensor:
    xy = keypoints[:, :, :2] / 447.0
    confidence = keypoints[:, :, 2].clamp(0.0, 1.0)
    left, right, root, inferior, left_alar, right_alar = xy.unbind(dim=1)
    distance = lambda a, b: torch.linalg.vector_norm(a - b, dim=1)
    nostril_vector = right - left
    geometry = torch.stack(
        [
            distance(left, right),
            distance(left_alar, right_alar),
            distance(root, inferior),
            distance(left, left_alar),
            distance(right, right_alar),
            distance(left, inferior),
            distance(right, inferior),
            nostril_vector[:, 1] / distance(left, right).clamp_min(1e-6),
            nostril_vector[:, 0] / distance(left, right).clamp_min(1e-6),
            semantic_probability[:, 2, :, :224].mean(dim=(1, 2))
            / semantic_probability[:, 2, :, 224:].mean(dim=(1, 2)).clamp_min(1e-6),
        ],
        dim=1,
    ).clamp(-4.0, 4.0)
    surface = semantic_probability[:, 1]
    dx = F.pad((surface[:, :, 1:] - surface[:, :, :-1]).abs(), (0, 1))
    dy = F.pad((surface[:, 1:, :] - surface[:, :-1, :]).abs(), (0, 0, 0, 1))
    boundary = (dx + dy).clamp(0.0, 1.0)
    coordinate = torch.linspace(-1.0, 1.0, 448, device=surface.device, dtype=surface.dtype)
    grid_y, grid_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    theta = torch.atan2(grid_y, grid_x)
    denominator = boundary.sum(dim=(1, 2)).clamp_min(1e-6)
    fourier = []
    for harmonic in range(1, 5):
        cosine = torch.cos(harmonic * theta)
        sine = torch.sin(harmonic * theta)
        for coordinate_grid in (grid_x, grid_y):
            fourier.append(torch.sum(boundary * coordinate_grid * cosine, dim=(1, 2)) / denominator)
            fourier.append(torch.sum(boundary * coordinate_grid * sine, dim=(1, 2)) / denominator)
    fourier_tensor = torch.stack(fourier, dim=1)
    area = surface.mean(dim=(1, 2))
    x_mass = torch.sum(surface, dim=1)
    y_mass = torch.sum(surface, dim=2)
    x_support = (x_mass > 0.1).float().sum(dim=1) / 448.0
    y_support = (y_mass > 0.1).float().sum(dim=1) / 448.0
    aspect = x_support / y_support.clamp_min(1e-6)
    symmetry = 1.0 - torch.mean((surface - torch.flip(surface, dims=(2,))).abs(), dim=(1, 2))
    nostril_ratio = semantic_probability[:, 2].mean(dim=(1, 2)) / area.clamp_min(1e-6)
    stats = torch.stack([area, aspect, symmetry, nostril_ratio], dim=1).clamp(-4.0, 4.0)
    result = torch.cat([xy.flatten(1), confidence, geometry, fourier_tensor, stats], dim=1)
    if result.shape[1] != 48:
        raise RuntimeError("shape feature dimension differs")
    return result


class ShapeStream(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(48, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.1),
            nn.Linear(128, 64),
        )

    def forward(self, keypoints: torch.Tensor, semantic_probability: torch.Tensor) -> torch.Tensor:
        return self.mlp(_shape_features(keypoints, semantic_probability))


def _quality_vector(
    aligned_rgb: torch.Tensor,
    keypoints: torch.Tensor,
    semantic_probability: torch.Tensor,
    invalid_probability: torch.Tensor,
    source_valid_probability: torch.Tensor,
    runtime_quality: torch.Tensor,
) -> torch.Tensor:
    detector_confidence = runtime_quality[:, 0].clamp(0, 1)
    native_ratio = runtime_quality[:, 1].clamp(0, 1)
    alignment_quality = (1.0 - runtime_quality[:, 3] / 0.18).clamp(0, 1)
    confidence = keypoints[:, :, 2].clamp(0, 1)
    surface = semantic_probability[:, 1]
    nostril = semantic_probability[:, 2]
    max_probability = semantic_probability.max(dim=1).values.mean(dim=(1, 2))
    entropy = -torch.sum(semantic_probability * torch.log(semantic_probability.clamp_min(1e-6)), dim=1).mean(dim=(1, 2))
    entropy_quality = (1.0 - entropy / torch.log(aligned_rgb.new_tensor(3.0))).clamp(0, 1)
    luminance = aligned_rgb.mean(dim=1, keepdim=True)
    gx = F.pad(luminance[:, :, :, 1:] - luminance[:, :, :, :-1], (0, 1))
    gy = F.pad(luminance[:, :, 1:, :] - luminance[:, :, :-1, :], (0, 0, 0, 1))
    sharpness = ((gx.square() + gy.square()) * surface[:, None]).mean(dim=(1, 2, 3))
    sharpness = (sharpness / 0.02).clamp(0, 1)
    symmetry = 1.0 - torch.mean((surface - torch.flip(surface, dims=(2,))).abs(), dim=(1, 2))
    return torch.stack(
        [
            detector_confidence,
            confidence.min(dim=1).values,
            confidence.mean(dim=1),
            alignment_quality,
            (runtime_quality[:, 1] * 2.0).clamp(0, 1),
            native_ratio,
            surface.mean(dim=(1, 2)),
            nostril.mean(dim=(1, 2)),
            1.0 - invalid_probability.mean(dim=(1, 2, 3)),
            max_probability,
            entropy_quality,
            sharpness,
            symmetry.clamp(0, 1),
            source_valid_probability.mean(dim=(1, 2, 3)),
        ],
        dim=1,
    ).clamp(0, 1)


class NoseIDModel(nn.Module):
    def __init__(
        self,
        dino_backbone: nn.Module,
        *,
        image_mean: tuple[float, float, float],
        image_std: tuple[float, float, float],
        rescale_factor: float,
    ) -> None:
        super().__init__()
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("DINO preprocessing mean/std must contain three values")
        values = (*image_mean, *image_std, rescale_factor)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("DINO preprocessing values must be finite")
        if any(value <= 0.0 for value in image_std) or rescale_factor <= 0.0:
            raise ValueError("DINO preprocessing std and rescale factor must be positive")
        self.dino = dino_backbone
        self.register_buffer(
            "image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_scale", torch.tensor(255.0 * rescale_factor, dtype=torch.float32)
        )
        for parameter in self.dino.parameters():
            parameter.requires_grad = False
        self.segmenter = FactorizedNoseSegmenter()
        self.rgb_stream = RGBLocalSemanticStream()
        self.frequency_bank = FixedFrequencyBank()
        self.texture_stream = TextureConvNeXtS()
        self.shape_stream = ShapeStream()
        self.quality_head = nn.Sequential(
            nn.Linear(78, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU()
        )
        self.utility_head = nn.Linear(32, 3)
        self.degradation_head = nn.Linear(32, 6)
        self.shape_projection = nn.Linear(64, 256)
        self.gate_head = nn.Sequential(nn.Linear(78, 64), nn.GELU(), nn.Linear(64, 3))
        self.fusion = nn.Sequential(
            nn.Linear(782, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 512)
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        with torch.no_grad():
            self.gate_head[-1].bias.copy_(
                torch.log(torch.tensor((0.45, 0.40, 0.15)))
            )
        nn.init.zeros_(self.utility_head.weight)
        with torch.no_grad():
            self.utility_head.bias.copy_(
                torch.logit(torch.tensor((0.80, 0.70, 0.70)))
            )

    def _dino_forward(self, rgb_336: torch.Tensor) -> Sequence[torch.Tensor]:
        with torch.set_grad_enabled(any(parameter.requires_grad for parameter in self.dino.parameters())):
            output = self.dino(pixel_values=rgb_336, output_hidden_states=True)
        hidden_states = getattr(output, "hidden_states", None)
        if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) < 13:
            raise RuntimeError("DINO backbone must return 13 hidden states")
        return hidden_states

    def forward(
        self,
        aligned_rgb: torch.Tensor,
        aligned_keypoints: torch.Tensor,
        runtime_quality: torch.Tensor,
        *,
        semantic_probability: torch.Tensor | None = None,
        invalid_probability: torch.Tensor | None = None,
        source_valid_probability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if aligned_rgb.ndim != 4 or aligned_rgb.shape[1:] != (3, 448, 448):
            raise ValueError("aligned_rgb must have shape [B,3,448,448]")
        batch = aligned_rgb.shape[0]
        if aligned_keypoints.shape != (batch, 6, 3) or runtime_quality.shape != (batch, 4):
            raise ValueError("aligned keypoint or runtime quality shape differs")
        rgb_336 = F.interpolate(aligned_rgb, size=(336, 336), mode="bicubic", align_corners=False, antialias=True)
        mean = self.image_mean.to(dtype=aligned_rgb.dtype)
        std = self.image_std.to(dtype=aligned_rgb.dtype)
        pixel_scale = self.pixel_scale.to(dtype=aligned_rgb.dtype)
        hidden_states = self._dino_forward((rgb_336 * pixel_scale - mean) / std)
        if (
            semantic_probability is None
            or invalid_probability is None
            or source_valid_probability is None
        ):
            raise RuntimeError(
                "oracle segmentation and source-valid masks are required"
            )
        if (
            semantic_probability.shape != (batch, 3, 448, 448)
            or invalid_probability.shape != (batch, 1, 448, 448)
            or source_valid_probability.shape != (batch, 1, 448, 448)
        ):
            raise ValueError("factorized segmentation probability shape differs")
        semantic_probability = semantic_probability.clamp(0, 1)
        semantic_probability = semantic_probability / semantic_probability.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        invalid_probability = invalid_probability.clamp(0, 1)
        source_valid_probability = source_valid_probability.clamp(0, 1)
        maximum = aligned_rgb.max(dim=1, keepdim=True).values
        minimum = aligned_rgb.min(dim=1, keepdim=True).values
        saturation = (maximum - minimum) / maximum.clamp_min(1e-6)
        luminance = aligned_rgb.mean(dim=1, keepdim=True)
        highlight = ((luminance > 0.92) & (saturation < 0.12)).to(aligned_rgb.dtype)
        highlight = highlight * semantic_probability[:, 1:2]
        effective_invalid = torch.maximum(invalid_probability, 0.8 * highlight)
        effective_invalid = torch.maximum(
            effective_invalid, 1.0 - source_valid_probability
        )
        texture_mask = (
            semantic_probability[:, 1:2]
            * (1.0 - semantic_probability[:, 2:3])
            * (1.0 - effective_invalid)
            * source_valid_probability
        )
        z_rgb = self.rgb_stream(
            hidden_states,
            semantic_probability,
            effective_invalid,
            aligned_keypoints,
            source_valid_probability,
        )
        z_texture = self.texture_stream(self.frequency_bank(aligned_rgb, texture_mask), texture_mask)
        z_shape = self.shape_stream(aligned_keypoints, semantic_probability)
        quality = _quality_vector(
            aligned_rgb,
            aligned_keypoints,
            semantic_probability,
            effective_invalid,
            source_valid_probability,
            runtime_quality,
        )
        quality_state = self.quality_head(torch.cat([quality, z_shape], dim=1))
        utilities = torch.sigmoid(self.utility_head(quality_state))
        native_short_side = runtime_quality[:, 1] * 448.0
        texture_prior = torch.where(
            native_short_side < 160.0,
            torch.zeros_like(native_short_side),
            torch.where(
                native_short_side < 224.0,
                0.7 * (native_short_side - 160.0) / 64.0,
                torch.ones_like(native_short_side),
            ),
        ).clamp(0, 1)
        utilities = torch.stack([utilities[:, 0], utilities[:, 1] * texture_prior, utilities[:, 2]], dim=1)
        gate_logits = self.gate_head(torch.cat([quality, z_shape], dim=1)) + torch.log(utilities + 1e-4)
        if self.training:
            drop_probabilities = gate_logits.new_tensor((0.10, 0.15, 0.10))
            dropped = torch.rand_like(gate_logits) < drop_probabilities
            all_dropped = dropped.all(dim=1)
            dropped[all_dropped, 0] = False
            gate_logits = gate_logits.masked_fill(dropped, -1e4)
        gates = torch.softmax(gate_logits, dim=1)
        shape_256 = F.normalize(self.shape_projection(z_shape), dim=1)
        fused = torch.cat(
            [gates[:, 0:1] * z_rgb, gates[:, 1:2] * z_texture, gates[:, 2:3] * shape_256, quality],
            dim=1,
        )
        embedding = F.normalize(self.fusion(fused), dim=1)
        return {
            "embedding": embedding,
            "z_rgb": z_rgb,
            "z_texture": z_texture,
            "z_shape": z_shape,
            "quality_vector": quality,
            "branch_utilities": utilities,
            "branch_gates": gates,
            "nose_utility": torch.sum(gates * utilities, dim=1, keepdim=True),
            "degradation_predictions": torch.sigmoid(
                self.degradation_head(quality_state)
            ),
            "semantic_probability": semantic_probability,
            "invalid_probability": effective_invalid,
            "source_valid_probability": source_valid_probability,
        }


__all__ = ["NoseIDModel", "RGBLocalSemanticStream", "ShapeStream", "TextureConvNeXtS"]
