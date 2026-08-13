from __future__ import annotations

import math

import numpy as np
import torch

from representation_learning.masked_afn import (
    _evaluate_partition,
    _fit_fusion_weights,
    _resolve_candidate_source,
    _train_region_adapter,
)


def _unit(index: int) -> np.ndarray:
    value = np.zeros(384, dtype=np.float32)
    value[index] = 1.0
    return value


def test_region_residual_adapter_trains_with_finite_loss() -> None:
    rows = [
        {"identity_token": "dog-a", "embeddings": {"A": _unit(0)}},
        {"identity_token": "dog-a", "embeddings": {"A": _unit(1)}},
        {"identity_token": "dog-b", "embeddings": {"A": _unit(10)}},
        {"identity_token": "dog-b", "embeddings": {"A": _unit(11)}},
    ]
    adapter, summary = _train_region_adapter(
        rows,
        region="A",
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        residual_scale=0.25,
        device=torch.device("cpu"),
        seed=7,
    )
    assert summary["sample_count"] == 4
    assert summary["identity_count"] == 2
    assert all(math.isfinite(value) for value in summary["epoch_losses"])
    with torch.inference_mode():
        output = adapter(torch.from_numpy(np.stack([_unit(0), _unit(10)])))
    assert output.shape == (2, 384)
    assert torch.allclose(torch.linalg.vector_norm(output, dim=1), torch.ones(2))


def test_availability_fusion_renormalizes_over_query_and_gallery_intersection() -> None:
    partition = {
        "gallery": {
            "dog-a": {"A": _unit(0), "F": _unit(1)},
            "dog-b": {"A": _unit(10)},
        },
        "queries": [
            {
                "sample_token": "q-a",
                "identity_token": "dog-a",
                "dataset_name": "fixture",
                "availability": "AF",
                "embeddings": {"A": _unit(0), "F": _unit(1)},
            },
            {
                "sample_token": "q-b",
                "identity_token": "dog-b",
                "dataset_name": "fixture",
                "availability": "A",
                "embeddings": {"A": _unit(10)},
            },
        ],
    }
    weights, dev = _fit_fusion_weights(partition, resolution=4)
    result = _evaluate_partition(partition, weights)
    assert set(weights) == {"A", "F", "N"}
    assert dev["overall"]["query_count"] == 2
    assert result["overall"]["Rank-1"] == 1.0
    assert set(result["by_availability"]) == {"A", "AF"}


def test_candidate_source_resolution_uses_member_path_for_duplicate_bytes() -> None:
    digest = "a" * 64
    resolver = {
        ("sibetan", digest): (
            {
                "sample_token": "first",
                "dataset_identity_id": "sibetan:v1:gt-json:1",
                "member_path": "Sibetan/1/site_C1_0001_1.jpg",
            },
            {
                "sample_token": "second",
                "dataset_identity_id": "sibetan:v1:gt-json:2",
                "member_path": "Sibetan/2/site_C1_0001_1.jpg",
            },
        )
    }
    source = _resolve_candidate_source(
        {
            "dataset_name": "sibetan",
            "image_sha256": digest,
            "image_path": "Sibetan/2/site_C1_0001_1.jpg",
        },
        resolver,
    )
    assert source["sample_token"] == "second"


def test_candidate_source_resolution_fails_closed_when_unbound() -> None:
    with np.testing.assert_raises_regex(ValueError, "one audited source"):
        _resolve_candidate_source(
            {
                "dataset_name": "mpdd",
                "image_sha256": "b" * 64,
                "image_path": "missing.jpg",
            },
            {},
        )


def test_fusion_weight_selection_rejects_empty_dev_partition() -> None:
    with np.testing.assert_raises_regex(ValueError, "no scoreable"):
        _fit_fusion_weights({"gallery": {}, "queries": []}, resolution=4)
