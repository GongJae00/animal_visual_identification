from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from archive.nose.experiments.nose_texture import (
    _apply_whitener,
    _fit_weight,
    _fit_whitener,
    _identity_partition,
    texture_descriptor,
)

def test_texture_descriptor_is_finite_normalized_and_mask_aware(tmp_path: Path) -> None:
    y, x = np.indices((48, 48))
    pattern = ((x // 4 + y // 4) % 2 * 180 + 40).astype(np.uint8)
    rgb = np.stack((pattern, np.roll(pattern, 1, axis=0), pattern), axis=2)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[8:40, 10:38] = 255
    mask[2, 2] = 255
    Image.fromarray(rgb, mode="RGB").save(tmp_path / "crop.png")
    Image.fromarray(mask, mode="L").save(tmp_path / "mask.png")

    descriptor = texture_descriptor(
        tmp_path,
        {"crop_path": "crop.png", "binary_mask_path": "mask.png"},
    )

    assert descriptor.shape == (816,)
    assert np.isfinite(descriptor).all()
    assert np.linalg.norm(descriptor) == pytest.approx(1.0)

def test_weight_fit_prefers_zero_for_a_harmful_texture_branch() -> None:
    raw = np.eye(4, dtype=np.float64)
    texture = np.fliplr(raw)

    weight, grid = _fit_weight(raw, texture, ["a", "b", "c", "d"])

    assert weight == 0.0
    assert len(grid) == 7

def test_identity_partition_is_deterministic_and_nontrivial() -> None:
    first = [_identity_partition(f"identity-{index}") for index in range(100)]
    second = [_identity_partition(f"identity-{index}") for index in range(100)]

    assert first == second
    assert set(first) == {"DEVELOPMENT", "EVALUATION"}

def test_label_blind_whitening_removes_common_direction_and_is_finite() -> None:
    rng = np.random.default_rng(0)
    matrix = np.ones((40, 12)) * 10.0 + rng.normal(0.0, 0.2, size=(40, 12))
    whitener = _fit_whitener(matrix[:20])

    transformed = _apply_whitener(matrix[20:], whitener)

    assert transformed.shape[0] == 20
    assert 1 <= transformed.shape[1] <= 12
    assert np.isfinite(transformed).all()
    assert np.linalg.norm(transformed, axis=1) == pytest.approx(np.ones(20))
