from __future__ import annotations

import unittest

import torch

from cvi.nose_id.frequency import FixedFrequencyBank


class NoseIDFrequencyTests(unittest.TestCase):
    def test_constant_image_is_finite_with_fixed_channel_count(self) -> None:
        bank = FixedFrequencyBank()
        image = torch.full((1, 3, 448, 448), 0.5)
        mask = torch.ones((1, 1, 448, 448))
        result = bank(image, mask)
        self.assertEqual(result.shape, (1, 11, 448, 448))
        self.assertTrue(torch.isfinite(result).all())

    def test_masked_pixels_are_excluded(self) -> None:
        bank = FixedFrequencyBank()
        image = torch.rand((1, 3, 448, 448))
        mask = torch.zeros((1, 1, 448, 448))
        mask[:, :, 96:352, 96:352] = 1.0
        result = bank(image, mask)
        self.assertTrue(torch.equal(result[:, :10, :80, :80], torch.zeros_like(result[:, :10, :80, :80])))
        self.assertTrue(torch.equal(result[:, 10:11], mask))


if __name__ == "__main__":
    unittest.main()
