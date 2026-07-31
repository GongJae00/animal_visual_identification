from __future__ import annotations

import unittest

import cv2
import numpy as np
import torch

from cvi.nose_id.alignment import register_residual_translation
from cvi.nose_id.frequency import classical_texture_descriptors
from cvi.nose_id.photometric import glare_saturation_invalid_mask
from cvi.nose_id.restoration import (
    RestorationConfig,
    leave_one_out_stability,
    redegradation_consistency,
    restore_nose_frames,
)


def _texture(size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(41)
    noise = rng.random((size, size), dtype=np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 1.0)
    texture = 0.15 + 0.60 * (texture - texture.min()) / (texture.max() - texture.min())
    return np.repeat(texture[..., None], 3, axis=2).astype(np.float32)


class NoseRestorationTests(unittest.TestCase):
    def test_glare_mask_marks_clipped_region(self) -> None:
        image = torch.full((1, 3, 32, 32), 0.45)
        image[:, :, 9:17, 11:20] = 1.0
        invalid = glare_saturation_invalid_mask(image)
        self.assertTrue(invalid[:, :, 9:17, 11:20].all())
        self.assertFalse(invalid[:, :, :8, :8].any())

    def test_restoration_is_deterministic_and_does_not_modify_inputs(self) -> None:
        base = _texture(64)
        frames = np.stack([base, np.clip(base * 0.92 + 0.02, 0.0, 1.0)])
        masks = np.ones(frames.shape[:3], dtype=np.uint8)
        frames_before = frames.copy()
        masks_before = masks.copy()
        config = RestorationConfig(
            denoise_strength=0.4,
            deblock_strength=0.25,
            clahe_clip_limit=1.5,
            compute_descriptors=True,
        )

        first = restore_nose_frames(frames, masks, config=config)
        second = restore_nose_frames(frames, masks, config=config)

        np.testing.assert_array_equal(frames, frames_before)
        np.testing.assert_array_equal(masks, masks_before)
        np.testing.assert_array_equal(first.restored_rgb, second.restored_rgb)
        np.testing.assert_array_equal(first.observation_count, second.observation_count)
        self.assertEqual(
            first.diagnostics.canonical_json(), second.diagnostics.canonical_json()
        )
        self.assertEqual(
            first.diagnostics.provenance_sha256,
            second.diagnostics.provenance_sha256,
        )

    def test_glare_is_never_filled(self) -> None:
        frames = np.stack([_texture(64), _texture(64)])
        frames[:, 20:30, 18:29] = 1.0
        result = restore_nose_frames(
            frames, config=RestorationConfig(compute_descriptors=False)
        )
        self.assertFalse(result.valid_mask[20:30, 18:29].any())
        self.assertTrue(np.equal(result.observation_count[20:30, 18:29], 0).all())
        self.assertTrue(np.equal(result.restored_rgb[20:30, 18:29], 0.0).all())

    def test_phase_registration_and_fusion_improve_shifted_noisy_frames(self) -> None:
        clean = _texture()
        clean_luminance = clean[..., 0]
        transform = np.asarray(((1.0, 0.0, 4.0), (0.0, 1.0, -3.0)), dtype=np.float32)
        shifted = cv2.warpAffine(
            clean_luminance,
            transform,
            (96, 96),
            borderMode=cv2.BORDER_CONSTANT,
        )
        shifted_mask = cv2.warpAffine(
            np.ones((96, 96), dtype=np.uint8), transform, (96, 96)
        )
        registration = register_residual_translation(
            clean_luminance, shifted, None, shifted_mask
        )
        self.assertTrue(registration.accepted)
        self.assertAlmostEqual(registration.shift_xy[0], 4.0, delta=0.15)
        self.assertAlmostEqual(registration.shift_xy[1], -3.0, delta=0.15)

        rng = np.random.default_rng(7)
        noisy_frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        shifts = ((0.0, 0.0), (3.0, -2.0), (-2.0, 2.0))
        for shift_x, shift_y in shifts:
            noisy = np.clip(
                clean + rng.normal(0.0, 0.035, clean.shape).astype(np.float32),
                0.01,
                0.98,
            )
            matrix = np.asarray(
                ((1.0, 0.0, shift_x), (0.0, 1.0, shift_y)), dtype=np.float32
            )
            noisy_frames.append(
                cv2.warpAffine(
                    noisy, matrix, (96, 96), borderMode=cv2.BORDER_REFLECT_101
                )
            )
            masks.append(
                cv2.warpAffine(
                    np.ones((96, 96), dtype=np.uint8), matrix, (96, 96)
                )
            )
        config = RestorationConfig(
            max_registration_residual=0.15, compute_descriptors=False
        )
        target = restore_nose_frames(clean[None], config=config)
        single = restore_nose_frames(np.stack(noisy_frames[:1]), np.stack(masks[:1]), config=config)
        fused = restore_nose_frames(np.stack(noisy_frames), np.stack(masks), config=config)
        common = target.valid_mask & single.valid_mask & fused.valid_mask
        single_error = np.mean((single.restored_rgb[common] - target.restored_rgb[common]) ** 2)
        fused_error = np.mean((fused.restored_rgb[common] - target.restored_rgb[common]) ** 2)
        self.assertEqual(fused.diagnostics.accepted_indices, (0, 1, 2))
        self.assertLess(fused_error, single_error)
        self.assertGreater(int(fused.observation_count.max()), 1)

    def test_unrelated_frame_is_rejected_and_outputs_are_finite(self) -> None:
        base = _texture(64)
        rng = np.random.default_rng(19)
        unrelated = rng.random(base.shape, dtype=np.float32) * 0.7 + 0.1
        frames = np.stack([base, base * 0.97 + 0.01, unrelated])
        result = restore_nose_frames(
            frames,
            config=RestorationConfig(
                max_registration_residual=0.08,
                minimum_phase_response=0.10,
                compute_descriptors=True,
            ),
        )
        self.assertEqual(result.diagnostics.accepted_indices, (0, 1))
        self.assertFalse(result.diagnostics.frames[2].accepted)
        self.assertTrue(np.isfinite(result.restored_rgb).all())
        self.assertTrue(np.isfinite(result.temporal_variance).all())
        self.assertTrue(
            all(np.isfinite(value).all() for value in result.descriptors.values())
        )

    def test_canonical_crop_identity_accepts_all_observed_frames(self) -> None:
        base = _texture(32)
        frames = np.stack(
            [base, np.roll(base, 5, axis=0), np.roll(base, -4, axis=1)]
        )
        result = restore_nose_frames(
            frames,
            config=RestorationConfig(
                registration_mode="canonical_crop_identity",
                illumination_normalization=False,
                compute_descriptors=False,
            ),
        )
        self.assertEqual(result.diagnostics.accepted_indices, (0, 1, 2))
        self.assertTrue(
            all(
                frame.reason in {"reference", "canonical_crop_identity"}
                for frame in result.diagnostics.frames
            )
        )

    def test_configuration_rejects_unknown_registration_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "registration mode"):
            RestorationConfig(registration_mode="invented")

    def test_classical_descriptors_and_consistency_helpers_are_finite(self) -> None:
        image = _texture(32)
        luminance = image[..., 0]
        mask = np.ones((32, 32), dtype=bool)
        descriptors = classical_texture_descriptors(luminance, mask)
        self.assertEqual(descriptors["gabor"].shape, (8,))
        self.assertEqual(descriptors["lbp"].shape, (256,))
        self.assertEqual(descriptors["radial_frequency"].shape, (8,))
        self.assertTrue(all(np.isfinite(value).all() for value in descriptors.values()))
        stack = np.stack([image, image])
        stack_mask = np.stack([mask, mask])
        stability = leave_one_out_stability(stack, stack_mask)
        np.testing.assert_array_equal(stability, np.zeros_like(stability))
        self.assertEqual(redegradation_consistency(image, image, mask), 0.0)


if __name__ == "__main__":
    unittest.main()
