from __future__ import annotations

import pytest

from localization.foundation_region_student import (
    RegionDecoderConfig,
    build_region_decoder,
    region_distillation_loss,
)
from localization.region_teacher_consensus import ConsensusState


def test_region_decoders_have_task_specific_class_counts() -> None:
    import torch

    features = torch.randn(2, 8, 8, 32)
    expected = {"A": 2, "F": 4, "N": 3}
    for region, classes in expected.items():
        decoder = build_region_decoder(
            RegionDecoderConfig(region, input_dimension=32, hidden_dimension=16, depth=1)
        )
        assert decoder(features, output_size=(32, 24)).shape == (2, classes, 32, 24)


def test_soft_consensus_uses_uncertainty_weighted_distillation() -> None:
    import torch

    logits = torch.randn(1, 2, 8, 8, requires_grad=True)
    targets = torch.softmax(torch.randn(1, 2, 8, 8), dim=1)
    loss = region_distillation_loss(
        logits,
        state=ConsensusState.SOFT_CANDIDATE,
        soft_probabilities=targets,
        hard_mask=torch.argmax(targets, dim=1),
        uncertainty=torch.full((1, 8, 8), 0.2),
        validity=torch.ones((1, 8, 8), dtype=torch.bool),
    )
    assert loss.isfinite()
    loss.backward()
    assert logits.grad is not None


def test_abstention_cannot_supervise_student() -> None:
    import torch

    logits = torch.zeros(1, 2, 4, 4)
    with pytest.raises(ValueError, match="cannot supervise"):
        region_distillation_loss(
            logits,
            state=ConsensusState.ABSTAIN,
            soft_probabilities=torch.full((1, 2, 4, 4), 0.5),
            hard_mask=torch.zeros((1, 4, 4)),
            uncertainty=torch.ones((1, 4, 4)),
            validity=torch.ones((1, 4, 4), dtype=torch.bool),
        )
