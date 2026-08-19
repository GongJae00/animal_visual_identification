from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from foundation.provenance import content_sha256
from embedding.methods.full_segment.models.classical import Classical128
from embedding.methods.full_segment.models.model import MaskedGAP128
from embedding.methods.full_segment.models.successor_models import (
    ClassicalFV128,
    Dinov2OccupancyProbe128,
    IdentityBlindResidualTokenAdapter128,
    SpatialScorer128,
    build_b1_fv,
    build_b2_fv,
    occupancy_pool,
    parameter_partition,
)
from embedding.learning.full_segment.full128_successors import (
    SUCCESSOR_CANDIDATES,
    build_successor_family_manifest,
    default_successor_training_config,
    load_dinov2_token_cache,
    load_successor_checkpoint,
    make_identity_blind_views,
    run_dinov2_successor_stages,
    smoke_successor_execution,
    train_identity_blind_fixed_steps,
    train_supervised_fixed_steps,
    validate_successor_training_config,
    write_dinov2_token_cache,
    write_successor_checkpoint,
)
from legacy.version.full128.workflows.run_full128_successors import main as successor_workflow


class _PatchBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.training_modes: list[bool] = []

    def forward(
        self, *, pixel_values: torch.Tensor, interpolate_pos_encoding: bool
    ) -> SimpleNamespace:
        assert interpolate_pos_encoding is True
        self.training_modes.append(self.training)
        patches = F.avg_pool2d(pixel_values, 14, 14)
        patches = patches.flatten(2).transpose(1, 2).mean(2, keepdim=True)
        patches = patches.expand(-1, -1, 384) * self.scale
        cls = torch.zeros(len(pixel_values), 1, 384, device=pixel_values.device)
        return SimpleNamespace(last_hidden_state=torch.cat((cls, patches), dim=1))


def _tokens() -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randn(4, 16, 384, generator=torch.Generator().manual_seed(17))
    occupancy = torch.ones(4, 16)
    occupancy[:, ::3] = 0.25
    return tokens, occupancy


def _config(*, supervised_steps: int = 1, ssl_steps: int = 1) -> dict[str, object]:
    config = default_successor_training_config(smoke=True)
    config["supervised_steps"] = supervised_steps
    config["ssl_steps"] = ssl_steps
    return config


def _assert_embedding(values: torch.Tensor) -> None:
    assert values.shape == (4, 128)
    assert values.dtype == torch.float32
    assert torch.isfinite(values).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(values, dim=1), torch.ones(4), atol=1e-6, rtol=1e-6
    )


def test_family_manifest_and_fixed_step_config_cover_every_control() -> None:
    manifest = build_successor_family_manifest()
    assert manifest["output"] == {
        "dimension": 128,
        "dtype": "float32",
        "normalization": "L2",
    }
    assert [row["candidate_id"] for row in manifest["candidates"]] == [
        row[0] for row in SUCCESSOR_CANDIDATES
    ]
    assert manifest["training_contracts"]["b2_reuse"].startswith("FORBIDDEN")
    config = default_successor_training_config(smoke=True)
    assert validate_successor_training_config(config) == config
    invalid = deepcopy(config)
    invalid["triplet_margin"] = 0.3
    with pytest.raises(ValueError, match="fixed at 0.2"):
        validate_successor_training_config(invalid)


def test_b0_wrapper_and_fresh_b1_b2_source_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classical = Classical128()
    expected = np.zeros(128, dtype=np.float32)
    expected[7] = 1
    monkeypatch.setattr(classical, "transform", lambda rgb, mask: expected.copy())
    np.testing.assert_array_equal(
        ClassicalFV128(classical).transform(np.zeros((3, 3, 3)), np.ones((3, 3))),
        expected,
    )

    torch.manual_seed(3)
    first = build_b1_fv()
    torch.manual_seed(4)
    second = build_b1_fv()
    assert isinstance(first, MaskedGAP128)
    assert not torch.equal(first.projection.weight, second.projection.weight)
    sentinel = MaskedGAP128()
    observed: list[tuple[Path, Path]] = []

    def fake_source(
        cls: type[MaskedGAP128], checkpoint: Path, *, intake_bundle_path: Path
    ) -> MaskedGAP128:
        del cls
        observed.append((checkpoint, intake_bundle_path))
        return sentinel

    monkeypatch.setattr(MaskedGAP128, "from_supervised_imagenet", classmethod(fake_source))
    assert build_b2_fv(Path("source.pth"), intake_bundle_path=Path("receipt.json")) is sentinel
    assert observed == [(Path("source.pth"), Path("receipt.json"))]


def test_b3_frozen_patch_occupancy_probe_has_strict_output_and_boundary() -> None:
    backbone = _PatchBackbone()
    model = Dinov2OccupancyProbe128(backbone).train()
    rgb = torch.rand(4, 3, 28, 28)
    mask = torch.ones(4, 1, 28, 28)
    mask[:, :, :, :14] = 0
    tokens, occupancy = model.extract_tokens(rgb, mask)
    output = model.forward_from_tokens(tokens, occupancy)

    assert tokens.shape == (4, 4, 384)
    assert occupancy.shape == (4, 4)
    assert backbone.training_modes == [False]
    assert parameter_partition(model)["trainable"] == (
        "projection.weight",
        "projection.bias",
    )
    _assert_embedding(output)


def test_occupancy_pool_fails_closed_for_invalid_masks() -> None:
    tokens, occupancy = _tokens()
    expected = (tokens * (occupancy / occupancy.sum(1, keepdim=True)).unsqueeze(-1)).sum(1)
    torch.testing.assert_close(occupancy_pool(tokens, occupancy), expected)
    with pytest.raises(ValueError, match="at least one"):
        occupancy_pool(tokens, torch.zeros_like(occupancy))
    invalid = occupancy.clone()
    invalid[0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        occupancy_pool(tokens, invalid)


def test_b4_zero_init_matches_b3_and_only_adapter_is_trainable() -> None:
    tokens, occupancy = _tokens()
    projection = nn.Linear(384, 128)
    b3 = Dinov2OccupancyProbe128(_PatchBackbone(), projection=deepcopy(projection))
    b4 = IdentityBlindResidualTokenAdapter128(_PatchBackbone(), deepcopy(projection))
    torch.testing.assert_close(
        b4.forward_from_tokens(tokens, occupancy),
        b3.forward_from_tokens(tokens, occupancy),
        rtol=0,
        atol=0,
    )
    partition = parameter_partition(b4)
    assert all(name.startswith("adapter.") for name in partition["trainable"])
    assert "projection.weight" in partition["frozen"]
    assert "tokens.backbone.scale" in partition["frozen"]


def test_b4_rejects_identity_metadata_and_u0_u1_share_step_semantics() -> None:
    tokens, occupancy = _tokens()
    clean = make_identity_blind_views(tokens, occupancy, phase=0)
    invalid = {**clean, "identity_id": torch.arange(4)}
    probe = IdentityBlindResidualTokenAdapter128(_PatchBackbone(), nn.Linear(384, 128))
    with pytest.raises(ValueError, match="reject identity metadata"):
        train_identity_blind_fixed_steps(probe, (invalid,), _config(), update_enabled=False)

    projection = nn.Linear(384, 128)
    u0 = IdentityBlindResidualTokenAdapter128(_PatchBackbone(), deepcopy(projection))
    u1 = IdentityBlindResidualTokenAdapter128(_PatchBackbone(), deepcopy(projection))
    u1.adapter.load_state_dict(u0.adapter.state_dict())
    initial = deepcopy(u0.adapter.state_dict())
    config = _config(ssl_steps=2)
    receipt0 = train_identity_blind_fixed_steps(u0, (clean,), config, update_enabled=False)
    receipt1 = train_identity_blind_fixed_steps(u1, (clean,), config, update_enabled=True)
    assert receipt0["attempted_steps"] == receipt1["attempted_steps"] == 2
    assert receipt0["update_steps"] == 0 and receipt1["update_steps"] == 2
    for name, value in u0.adapter.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    assert any(
        not torch.equal(value, initial[name])
        for name, value in u1.adapter.state_dict().items()
    )


@pytest.mark.parametrize(
    ("uniform", "channel"), ((False, False), (True, False), (True, True))
)
def test_b5_decomposition_and_zero_init_controls(uniform: bool, channel: bool) -> None:
    tokens, occupancy = _tokens()
    projection = nn.Linear(384, 128)
    baseline = Dinov2OccupancyProbe128(_PatchBackbone(), projection=deepcopy(projection))
    model = SpatialScorer128(
        _PatchBackbone(), deepcopy(projection), uniform_spatial=uniform, channel_gate=channel
    )
    output, parts = model.forward_from_tokens(tokens, occupancy, return_decomposition=True)
    torch.testing.assert_close(
        output, baseline.forward_from_tokens(tokens, occupancy), rtol=0, atol=0
    )
    torch.testing.assert_close(parts.occupancy, occupancy)
    torch.testing.assert_close(parts.logits, torch.zeros_like(occupancy))
    torch.testing.assert_close(parts.weights, occupancy / occupancy.sum(1, keepdim=True))
    _assert_embedding(output)


def test_b5_trainable_partitions_distinguish_spatial_uniform_and_channel_controls() -> None:
    spatial = SpatialScorer128(_PatchBackbone(), nn.Linear(384, 128))
    uniform = SpatialScorer128(
        _PatchBackbone(), nn.Linear(384, 128), uniform_spatial=True
    )
    channel = SpatialScorer128(
        _PatchBackbone(), nn.Linear(384, 128), uniform_spatial=True, channel_gate=True
    )
    assert parameter_partition(spatial)["trainable"] == ("scorer.weight", "scorer.bias")
    assert parameter_partition(uniform)["trainable"] == ()
    assert parameter_partition(channel)["trainable"] == ("channel_gate",)


def test_b5_uses_and_freezes_the_explicit_parent_token_adapter() -> None:
    class _OffsetAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.offset = nn.Parameter(torch.linspace(-0.2, 0.2, 384))

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return tokens + self.offset

    tokens, occupancy = _tokens()
    projection = nn.Linear(384, 128)
    baseline = SpatialScorer128(
        _PatchBackbone(), deepcopy(projection), uniform_spatial=True
    )
    parent_bound = SpatialScorer128(
        _PatchBackbone(),
        deepcopy(projection),
        token_adapter=_OffsetAdapter(),
        uniform_spatial=True,
    )

    baseline_output = baseline.forward_from_tokens(tokens, occupancy)
    parent_output = parent_bound.forward_from_tokens(tokens, occupancy)

    assert not torch.equal(baseline_output, parent_output)
    assert "token_adapter.offset" in parameter_partition(parent_bound)["frozen"]
    assert not parent_bound.token_adapter.offset.requires_grad


def test_supervised_fixed_steps_update_only_the_b3_probe() -> None:
    tokens, occupancy = _tokens()
    model = Dinov2OccupancyProbe128(_PatchBackbone())
    frozen_before = model.tokens.backbone.scale.detach().clone()
    probe_before = model.projection.weight.detach().clone()
    receipt = train_supervised_fixed_steps(
        model,
        ({"tokens": tokens, "occupancy": occupancy, "label": torch.tensor([0, 0, 1, 1])},),
        _config(supervised_steps=2),
    )
    assert receipt["attempted_steps"] == receipt["update_steps"] == 2
    torch.testing.assert_close(model.tokens.backbone.scale, frozen_before, rtol=0, atol=0)
    assert not torch.equal(model.projection.weight, probe_before)


def test_successor_training_reads_lazy_batches_without_materializing_sequence() -> None:
    tokens, occupancy = _tokens()
    batch = {
        "tokens": tokens,
        "occupancy": occupancy,
        "label": torch.tensor([0, 0, 1, 1]),
    }

    class LazyBatches(Sequence[Mapping[str, torch.Tensor]]):
        def __init__(self) -> None:
            self.reads = 0

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
            if index != 0:
                raise IndexError(index)
            self.reads += 1
            if self.reads > 2:
                raise AssertionError("batch sequence was eagerly materialized")
            return batch

    batches = LazyBatches()
    train_supervised_fixed_steps(
        Dinov2OccupancyProbe128(_PatchBackbone()),
        batches,
        _config(supervised_steps=2),
    )
    assert batches.reads == 2


def test_dino_stage_orchestrator_runs_all_update_and_control_lanes() -> None:
    tokens, occupancy = _tokens()
    supervised = {
        "tokens": tokens,
        "occupancy": occupancy,
        "label": torch.tensor([0, 0, 1, 1]),
    }
    result = run_dinov2_successor_stages(
        backbone=_PatchBackbone(),
        supervised_batches=(supervised,),
        identity_blind_batches=(make_identity_blind_views(tokens, occupancy, phase=0),),
        config=_config(),
    )
    assert set(result["models"]) == {
        "B3-FV",
        "B4-U0-FV",
        "B4-U1-FV",
        "B5-FV",
        "B5-UNIFORM-FV",
        "B5-CHANNEL-GATE-FV",
    }
    receipts = result["training_receipts"]
    assert receipts["B4-U0-FV"]["update_steps"] == 0
    assert receipts["B4-U1-FV"]["update_steps"] == 1
    assert receipts["B5-UNIFORM-FV"]["update_steps"] == 0
    assert receipts["B5-FV"]["update_steps"] == 1


def test_token_cache_round_trip_is_deterministic_and_tamper_evident(tmp_path: Path) -> None:
    ids = tuple(sorted(hashlib.sha256(f"sample-{i}".encode()).hexdigest() for i in range(3)))
    rng = np.random.default_rng(8)
    tokens = rng.normal(size=(3, 4, 384)).astype(np.float32)
    occupancy = rng.uniform(size=(3, 4)).astype(np.float32)
    bindings = {"receipt": "a" * 64}
    first = write_dinov2_token_cache(
        tmp_path / "first", sample_ids=ids, tokens=tokens, occupancy=occupancy, bindings=bindings
    )
    second = write_dinov2_token_cache(
        tmp_path / "second", sample_ids=ids, tokens=tokens, occupancy=occupancy, bindings=bindings
    )
    restored_ids, restored_tokens, restored_occupancy, restored = load_dinov2_token_cache(
        tmp_path / "first", bindings=bindings
    )
    assert first == second == restored and restored_ids == ids
    np.testing.assert_array_equal(restored_tokens, tokens)
    np.testing.assert_array_equal(restored_occupancy, occupancy)
    path = tmp_path / "first" / "tokens.f32le"
    payload = bytearray(path.read_bytes())
    payload[0] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="file digest"):
        load_dinov2_token_cache(tmp_path / "first", bindings=bindings)


def test_trainable_only_and_empty_control_checkpoints_round_trip(tmp_path: Path) -> None:
    config = _config()
    bindings = {"dinov2_receipt": "b" * 64}
    b3 = Dinov2OccupancyProbe128(_PatchBackbone())
    manifest = write_successor_checkpoint(
        tmp_path / "b3", candidate_id="B3-FV", model=b3, config=config,
        bindings=bindings, training_receipt={"update_steps": 1},
    )
    restored = Dinov2OccupancyProbe128(_PatchBackbone())
    assert load_successor_checkpoint(
        tmp_path / "b3", candidate_id="B3-FV", model=restored,
        config=config, bindings=bindings,
    ) == manifest
    torch.testing.assert_close(restored.projection.weight, b3.projection.weight)

    uniform = SpatialScorer128(
        _PatchBackbone(), nn.Linear(384, 128), uniform_spatial=True
    )
    empty = write_successor_checkpoint(
        tmp_path / "uniform", candidate_id="B5-UNIFORM-FV", model=uniform,
        config=config, bindings=bindings, training_receipt={"update_steps": 0},
    )
    assert empty["parameter_partition"]["trainable"] == []


def test_smoke_executes_models_cache_checkpoint_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = smoke_successor_execution(tmp_path / "direct")
    assert receipt["smoke_receipt_sha256"] == content_sha256(
        {key: value for key, value in receipt.items() if key != "smoke_receipt_sha256"}
    )
    assert (tmp_path / "direct" / "token-cache" / "cache-manifest.json").is_file()
    assert successor_workflow(["--smoke-output", str(tmp_path / "workflow")]) == 0
    assert "FULL128_SUCCESSOR_SYNTHETIC_SMOKE_COMPLETE" in capsys.readouterr().out
