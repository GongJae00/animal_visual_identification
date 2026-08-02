from __future__ import annotations

import unittest

import torch

from identity_methods.nose.pooling import signed_gem


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
