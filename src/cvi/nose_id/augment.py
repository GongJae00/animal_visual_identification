"""Texture-preserving paired augmentation without horizontal reflection."""

from __future__ import annotations

import torch
from torch.nn import functional as F


class NoseIdentityAugment:
    def __init__(self, seed: int = 0) -> None:
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(
        self,
        image: torch.Tensor,
        keypoints: torch.Tensor,
        semantic_probability: torch.Tensor,
        invalid_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.shape != (3, 448, 448) or not torch.isfinite(image).all():
            raise ValueError("NoseID augmentation expects finite [3,448,448] RGB")
        if keypoints.shape != (6, 3):
            raise ValueError("NoseID augmentation expects keypoints [6,3]")
        if semantic_probability.shape != (3, 448, 448) or invalid_probability.shape != (1, 448, 448):
            raise ValueError("NoseID augmentation mask shape differs")
        angle = (float(torch.rand((), generator=self.generator)) * 24.0 - 12.0) * torch.pi / 180.0
        scale = 0.92 + float(torch.rand((), generator=self.generator)) * 0.16
        tx = float(torch.rand((), generator=self.generator)) * 0.08 - 0.04
        ty = float(torch.rand((), generator=self.generator)) * 0.08 - 0.04
        cosine, sine = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        theta = image.new_tensor(((scale * cosine, -scale * sine, tx), (scale * sine, scale * cosine, ty)))[None]
        grid = F.affine_grid(theta, (1, 3, 448, 448), align_corners=True)
        result = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="reflection", align_corners=True)[0]
        semantic = F.grid_sample(
            semantic_probability[None], grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0]
        semantic = semantic / semantic.sum(dim=0, keepdim=True).clamp_min(1e-6)
        invalid = F.grid_sample(
            invalid_probability[None], grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0].clamp(0, 1)
        transform = torch.eye(3, dtype=image.dtype, device=image.device)
        transform[:2] = theta[0]
        inverse = torch.linalg.inv(transform)
        normalized_xy = keypoints[:, :2] / 447.0 * 2.0 - 1.0
        homogeneous = torch.cat([normalized_xy, torch.ones((6, 1), device=image.device, dtype=image.dtype)], dim=1)
        transformed_xy = homogeneous @ inverse.T
        transformed_xy = (transformed_xy[:, :2] + 1.0) * 0.5 * 447.0
        transformed_keypoints = torch.cat([transformed_xy, keypoints[:, 2:3]], dim=1)
        exposure = 2.0 ** (float(torch.rand((), generator=self.generator)) * 0.6 - 0.3)
        gamma = 0.9 + float(torch.rand((), generator=self.generator)) * 0.2
        gains = 0.9 + torch.rand((3, 1, 1), generator=self.generator, device="cpu") * 0.2
        gains = gains.to(device=image.device, dtype=image.dtype)
        result = (result.clamp(0, 1).pow(gamma) * exposure * gains).clamp(0, 1)
        return result, transformed_keypoints, semantic, invalid

    def pair(
        self,
        image: torch.Tensor,
        keypoints: torch.Tensor,
        semantic_probability: torch.Tensor,
        invalid_probability: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        return (
            self(image, keypoints, semantic_probability, invalid_probability),
            self(image, keypoints, semantic_probability, invalid_probability),
        )


__all__ = ["NoseIdentityAugment"]
