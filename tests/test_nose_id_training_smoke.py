from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from embedding.methods.nose.training.losses import NoseIDObjective


class _ToyNoseBranches(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rgb = nn.Linear(12, 256)
        self.texture = nn.Linear(12, 256)
        self.shape = nn.Linear(12, 64)
        self.quality = nn.Linear(12, 6)
        self.fusion = nn.Linear(256 + 256 + 64, 512)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb = F.normalize(self.rgb(value), dim=1)
        texture = F.normalize(self.texture(value), dim=1)
        shape = self.shape(value)
        embedding = F.normalize(self.fusion(torch.cat([rgb, texture, shape], dim=1)), dim=1)
        return {
            "embedding": embedding,
            "z_rgb": rgb,
            "z_texture": texture,
            "degradation_predictions": torch.sigmoid(self.quality(value)),
        }


class NoseIDTrainingSmokeTests(unittest.TestCase):
    def test_quality_auxiliary_and_all_identity_branches_receive_gradients(self) -> None:
        model = _ToyNoseBranches()
        objective = NoseIDObjective(512, 16)
        values = torch.randn((64, 12))
        second_values = values + 0.01 * torch.randn_like(values)
        labels = torch.arange(16).repeat_interleave(4)
        sessions = torch.tensor([0, 0, 1, 1] * 16)
        first_target = torch.rand((64, 6))
        second_target = torch.rand((64, 6))
        losses = objective(
            model(values),
            labels,
            sessions,
            second_view_output=model(second_values),
            first_degradation_target=first_target,
            second_degradation_target=second_target,
            margin_scale=0.5,
        )
        self.assertGreater(float(losses["quality_auxiliary"]), 0.0)
        losses["total"].backward()
        for module in (model.rgb, model.texture, model.shape, model.quality, model.fusion):
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(float(module.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(objective.arcface.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
