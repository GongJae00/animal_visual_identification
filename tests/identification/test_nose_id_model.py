from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from identification.export.nose.modeling.model import NoseIDModel
from identification.export.nose.modeling.pooling import signed_gem

class _DummyDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, *, pixel_values: torch.Tensor, output_hidden_states: bool):
        self.assert_hidden = output_hidden_states
        batch = pixel_values.shape[0]
        base = pixel_values.mean(dim=(1, 2, 3)).view(batch, 1, 1) + self.anchor
        return SimpleNamespace(
            hidden_states=tuple(base.expand(batch, 577, 384) for _ in range(13))
        )

class NoseIDModelTests(unittest.TestCase):
    @staticmethod
    def _model() -> NoseIDModel:
        return NoseIDModel(
            _DummyDino(),
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
            rescale_factor=1.0 / 255.0,
        )

    def test_oracle_mask_path_produces_fixed_normalized_outputs(self) -> None:
        model = self._model().eval()
        rgb = torch.rand((1, 3, 448, 448))
        keypoints = torch.tensor(
            [[
                [152.0, 224.0, 1.0], [295.0, 224.0, 1.0],
                [224.0, 107.0, 1.0], [224.0, 340.0, 1.0],
                [80.0, 224.0, 1.0], [367.0, 224.0, 1.0],
            ]]
        )
        runtime_quality = torch.tensor([[1.0, 0.6, 1.0, 0.0]])
        semantic = torch.zeros((1, 3, 448, 448))
        semantic[:, 1] = 1.0
        invalid = torch.zeros((1, 1, 448, 448))
        with torch.no_grad():
            output = model(
                rgb,
                keypoints,
                runtime_quality,
                semantic_probability=semantic,
                invalid_probability=invalid,
                source_valid_probability=torch.ones((1, 1, 448, 448)),
            )
        self.assertEqual(output["z_rgb"].shape, (1, 256))
        self.assertEqual(output["z_texture"].shape, (1, 256))
        self.assertEqual(output["z_shape"].shape, (1, 64))
        self.assertEqual(output["embedding"].shape, (1, 512))
        self.assertEqual(output["quality_vector"].shape, (1, 14))
        self.assertTrue(torch.isfinite(output["embedding"]).all())
        torch.testing.assert_close(
            torch.linalg.vector_norm(output["embedding"], dim=1), torch.ones(1), atol=1e-5, rtol=1e-5
        )

    def test_low_resolution_disables_texture_utility(self) -> None:
        model = self._model().eval()
        rgb = torch.rand((1, 3, 448, 448))
        keypoints = torch.zeros((1, 6, 3))
        keypoints[:, :, 2] = 1.0
        semantic = torch.zeros((1, 3, 448, 448))
        semantic[:, 1] = 1.0
        with torch.no_grad():
            output = model(
                rgb,
                keypoints,
                torch.tensor([[1.0, 95.0 / 448.0, 1.0, 0.0]]),
                semantic_probability=semantic,
                invalid_probability=torch.zeros((1, 1, 448, 448)),
                source_valid_probability=torch.ones((1, 1, 448, 448)),
            )
        self.assertEqual(float(output["branch_utilities"][0, 1]), 0.0)

class NoseIDSignedGeMTests(unittest.TestCase):
    def test_positive_and_negative_features_both_contribute_gradients(self) -> None:
        features = torch.tensor([[[[2.0, -3.0]]]], requires_grad=True)
        weights = torch.ones((1, 1, 1, 2))
        output = signed_gem(features, weights, torch.tensor(3.0))
        self.assertLess(float(output), 0.0)
        output.sum().backward()
        self.assertGreater(float(features.grad[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(features.grad[0, 0, 0, 1]), 0.0)

if __name__ == "__main__":
    unittest.main()
