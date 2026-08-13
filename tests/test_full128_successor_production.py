from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

import embedding.learning.full_segment.full128_successor_production as production
from evaluation.full_segment.full128_successors import build_successor_embedding_cache_descriptor
from embedding.methods.full_segment.preparation.data import Full128Sample
from embedding.methods.full_segment.models.successor_models import Dinov2OccupancyProbe128
from embedding.learning.full_segment.full128_successor_production import (
    PRODUCTION_CANDIDATES,
    build_balanced_pk_schedule,
    default_production_config,
    run_successor_production,
    validate_production_config,
)
from workflows.generate_full128_representation_traces import _new_external_output
from workflows.run_full128_successors import main as successor_workflow


class _PatchBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(
        self, *, pixel_values: torch.Tensor, interpolate_pos_encoding: bool
    ) -> SimpleNamespace:
        assert interpolate_pos_encoding
        patches = F.avg_pool2d(pixel_values, 14, 14)
        patches = patches.flatten(2).transpose(1, 2).mean(2, keepdim=True)
        patches = patches.expand(-1, -1, 384) * self.scale
        cls = torch.zeros(len(pixel_values), 1, 384, device=pixel_values.device)
        return SimpleNamespace(last_hidden_state=torch.cat((cls, patches), dim=1))


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _config() -> dict[str, object]:
    config = default_production_config()
    config.update(
        {
            "supervised_steps": 1,
            "ssl_steps": 1,
            "workers": 0,
            "identities_per_batch": 2,
            "extraction_batch_size": 2,
            "precision": {"device": "cpu", "amp": False, "amp_dtype": "float32"},
        }
    )
    return config


def test_identity_blind_batch_sequence_stops_at_declared_length() -> None:
    cache = {
        "tokens": np.zeros((3, 4, 384), dtype=np.float32),
        "occupancy": np.ones((3, 4), dtype=np.float32),
    }
    batches = production._IdentityBlindBatchSequence(cache, batch_size=2)

    assert len(tuple(batches)) == 2
    with pytest.raises(IndexError):
        batches[2]


def _sample(index: int, *, dataset: str = "dogfacenet224") -> Full128Sample:
    return Full128Sample(
        sample_id=_sha(f"sample-{index}"),
        identity_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"identity-{index // 2}")),
        dataset_name=dataset,
        view="face",
        role="FIT",
        rgb_path=Path(f"/external/rgb-{index}.png"),
        rgb_sha256="a" * 64,
        mask_path=Path(f"/external/mask-{index}.png"),
        mask_sha256="b" * 64,
        crop_record_sha256="c" * 64,
    )


def test_production_config_has_no_implicit_test_shortening() -> None:
    config = default_production_config()
    assert validate_production_config(config) == config
    assert config["supervised_steps"] == 2_000
    assert config["ssl_steps"] == 2_000
    assert set(PRODUCTION_CANDIDATES) == {
        "B0-FV",
        "B1-FV",
        "B2-FV",
        "B3",
        "B4-U0",
        "B4-U1",
        "B5-UNIFORM",
        "B5-CHANNEL",
        "B5-SPATIAL",
    }

    bounded = _config()
    assert validate_production_config(bounded)["supervised_steps"] == 1


def test_balanced_pk_schedule_is_deterministic_and_dataset_balanced() -> None:
    samples = tuple(
        [_sample(index) for index in range(8)]
        + [replace(_sample(index + 100), dataset_name="mpdd") for index in range(8)]
    )

    first = build_balanced_pk_schedule(
        samples, identities_per_batch=4, samples_per_identity=2, seed=19
    )
    second = build_balanced_pk_schedule(
        samples, identities_per_batch=4, samples_per_identity=2, seed=19
    )

    assert first == second
    for batch in first:
        identities = {samples[index].identity_id for index in batch}
        datasets = {samples[index].dataset_name for index in batch}
        assert len(batch) == 4 * 2
        assert len(identities) == 4
        assert datasets == {"dogfacenet224", "mpdd"}
        assert all(
            sum(samples[index].identity_id == identity for index in batch) == 2
            for identity in identities
        )


def test_balanced_pk_schedule_rejects_more_datasets_than_identities_per_batch() -> None:
    samples = tuple(
        _sample(index, dataset=dataset)
        for index, dataset in enumerate(
            ("dataset-a",) * 2 + ("dataset-b",) * 2 + ("dataset-c",) * 2
        )
    )

    with pytest.raises(ValueError, match="cannot represent every dataset within P"):
        build_balanced_pk_schedule(
            samples, identities_per_batch=2, samples_per_identity=2, seed=19
        )


def test_production_rejects_unpaired_b4_and_uncommitted_b5_parent(
    tmp_path: Path,
) -> None:
    b2 = tmp_path / "b2.pth"
    receipt = tmp_path / "receipt.json"
    b2.write_bytes(b"b2")
    receipt.write_bytes(b"receipt")
    inventory = {"unused": True}

    with pytest.raises(ValueError, match="must run together"):
        run_successor_production(
            successor_inventory_bundle=inventory,
            required_evaluation_tokens=(_sha("eval"),),
            evaluation_panel_sha256=_sha("panel"),
            output_dir=tmp_path / "output-a",
            candidates=("B3", "B4-U0"),
            b5_parent_id="B3",
            config=_config(),
            b2_checkpoint_path=b2,
            b2_intake_bundle_path=receipt,
            dinov2_backbone=_PatchBackbone(),
            dinov2_contract=object(),
            descriptor_builder=build_successor_embedding_cache_descriptor,
        )
    with pytest.raises(ValueError, match="parent must be included"):
        run_successor_production(
            successor_inventory_bundle=inventory,
            required_evaluation_tokens=(_sha("eval"),),
            evaluation_panel_sha256=_sha("panel"),
            output_dir=tmp_path / "output-b",
            candidates=("B3", "B5-SPATIAL"),
            b5_parent_id="B4-U1",
            config=_config(),
            b2_checkpoint_path=b2,
            b2_intake_bundle_path=receipt,
            dinov2_backbone=_PatchBackbone(),
            dinov2_contract=object(),
            descriptor_builder=build_successor_embedding_cache_descriptor,
        )


def test_b5_bindings_include_precommitted_parent_checkpoint(tmp_path: Path) -> None:
    parent = tmp_path / "B4-U1"
    parent.mkdir()
    parent_run = {
        "candidate_run_sha256": _sha("parent-run"),
        "checkpoint_sha256": _sha("parent-checkpoint"),
    }
    (parent / "candidate-run.json").write_text(json.dumps(parent_run), encoding="utf-8")

    bindings = production._candidate_bindings(
        tmp_path,
        {"run_manifest_sha256": _sha("run")},
        "B5-SPATIAL",
        "B4-U1",
    )

    assert bindings["precommitted_parent_id"] == "B4-U1"
    assert bindings["precommitted_parent_candidate_run_sha256"] == _sha("parent-run")
    assert bindings["precommitted_parent_checkpoint_sha256"] == _sha(
        "parent-checkpoint"
    )


def _materialized_sample(root: Path, index: int) -> Full128Sample:
    rgb = np.full((224, 224, 3), 20 + index, dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[14:210, 14:210] = 255
    rgb_path = root / f"rgb-{index}.png"
    mask_path = root / f"mask-{index}.png"
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    return Full128Sample(
        sample_id=_sha(f"materialized-{index}"),
        identity_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"materialized-{index // 2}")),
        dataset_name="dogfacenet224",
        view="face",
        role="FIT",
        rgb_path=rgb_path,
        rgb_sha256=hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
        mask_path=mask_path,
        mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        crop_record_sha256=_sha(f"crop-{index}"),
    )


def test_real_png_token_extraction_is_streamed_and_resumable(tmp_path: Path) -> None:
    samples = tuple(_materialized_sample(tmp_path, index) for index in range(2))
    config = _config()
    bindings = {"receipt": _sha("receipt")}
    target = tmp_path / "token-cache"

    first = production._materialize_or_open_token_cache(
        target,
        model=Dinov2OccupancyProbe128(_PatchBackbone()),
        samples=samples,
        config=config,
        bindings=bindings,
    )
    second = production._materialize_or_open_token_cache(
        target,
        model=Dinov2OccupancyProbe128(_PatchBackbone()),
        samples=samples,
        config=config,
        bindings=bindings,
    )

    assert first["manifest"] == second["manifest"]
    assert first["tokens"].shape == (2, 256, 384)
    assert first["occupancy"].shape == (2, 256)
    assert np.isfinite(first["tokens"]).all()
    assert np.all((first["occupancy"] >= 0) & (first["occupancy"] <= 1))


def _inventory_rows(tmp_path: Path) -> tuple[dict[str, object], tuple[str, ...]]:
    rows = []
    for index in range(4):
        sample = _materialized_sample(tmp_path, index)
        rows.append(
            {
                "sample_token": sample.sample_id,
                "registered_identity_id": sample.identity_id,
                "dataset_name": sample.dataset_name,
                "protocol_scope": "FIT" if index < 4 else "DEV",
                "gradient_eligible": True,
                "state": "USABLE",
                "record_sha256": _sha(f"record-{index}"),
                "artifact": {
                    "full_rgb_path": str(sample.rgb_path),
                    "full_rgb_sha256": sample.rgb_sha256,
                    "full_mask_path": str(sample.mask_path),
                    "full_mask_sha256": sample.mask_sha256,
                    "crop_record_sha256": sample.crop_record_sha256,
                },
            }
        )
    rows.sort(key=lambda row: row["sample_token"])
    inventory = {
        "bundle_sha256": _sha("inventory-bundle"),
        "inventory_sha256": _sha("inventory"),
        "inventory": {"successor_population": rows},
    }
    evaluation = tuple(sorted(row["sample_token"] for row in rows[:2]))
    return inventory, evaluation


def test_production_publishes_evaluation_descriptor_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory, evaluation = _inventory_rows(tmp_path)
    monkeypatch.setattr(
        production,
        "validate_face_visible_successor_inventory_bundle",
        lambda value, *, verify_artifacts: value,
    )
    monkeypatch.setattr(
        production,
        "dinov2_contract_bindings",
        lambda contract: {"model_sha256": _sha("dino")},
    )
    population_tokens = tuple(
        row["sample_token"] for row in inventory["inventory"]["successor_population"]
    )
    token_values = np.ones((4, 256, 384), dtype=np.float32)
    occupancy = np.ones((4, 256), dtype=np.float32)
    monkeypatch.setattr(
        production,
        "_materialize_or_open_token_cache",
        lambda *args, **kwargs: {
            "sample_tokens": population_tokens,
            "tokens": token_values,
            "occupancy": occupancy,
        },
    )
    train_count = 0

    def fake_train(*args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
        nonlocal train_count
        train_count += 1
        return Dinov2OccupancyProbe128(_PatchBackbone()), {
            "kind": "TEST_FIXED_STEP",
            "attempted_steps": 1,
            "update_steps": 1,
        }

    monkeypatch.setattr(production, "_train_candidate", fake_train)
    embeddings = np.zeros((4, 128), dtype=np.float32)
    embeddings[:, 0] = 1
    monkeypatch.setattr(
        production, "_extract_candidate_embeddings", lambda *args, **kwargs: embeddings
    )
    b2 = tmp_path / "b2.pth"
    receipt = tmp_path / "receipt.json"
    b2.write_bytes(b"b2")
    receipt.write_bytes(b"receipt")
    output = tmp_path / "run"
    arguments = {
        "successor_inventory_bundle": inventory,
        "required_evaluation_tokens": evaluation,
        "evaluation_panel_sha256": _sha("panel"),
        "output_dir": output,
        "candidates": ("B3",),
        "b5_parent_id": "B3",
        "config": _config(),
        "b2_checkpoint_path": b2,
        "b2_intake_bundle_path": receipt,
        "dinov2_backbone": _PatchBackbone(),
        "dinov2_contract": object(),
        "descriptor_builder": build_successor_embedding_cache_descriptor,
        "real_smoke_fit_limit": 4,
    }

    created = run_successor_production(**arguments)
    resumed = run_successor_production(**arguments)

    assert created["candidate_statuses"] == {"B3": "CREATED"}
    assert resumed["candidate_statuses"] == {"B3": "VALIDATED_EXISTING"}
    assert train_count == 1
    descriptor = production._read_json(
        output / "B3" / "evaluation-cache-descriptor.json"
    )
    assert descriptor["sample_tokens"] == list(evaluation)
    assert descriptor["evaluation_panel_sha256"] == _sha("panel")
    assert (output / "B3" / "population-embeddings.f32le").stat().st_size == 4 * 512

    changed = deepcopy(arguments)
    changed["config"] = {**_config(), "seed": 99}
    with pytest.raises(FileExistsError, match="immutable bindings"):
        run_successor_production(**changed)

    pack = output / "B3" / "population-embeddings.f32le"
    payload = bytearray(pack.read_bytes())
    payload[0] ^= 1
    pack.write_bytes(payload)
    with pytest.raises(ValueError, match="population embedding pack"):
        run_successor_production(**arguments)


def test_workflow_prints_production_config_without_running_training(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert successor_workflow(["--print-production-config"]) == 0
    output = capsys.readouterr().out
    assert PRODUCTION_CONFIG_SCHEMA_FRAGMENT in output


PRODUCTION_CONFIG_SCHEMA_FRAGMENT = "cvi.full128_successor_production_config.v1"


def test_trace_workflow_rejects_repository_and_existing_outputs(tmp_path: Path) -> None:
    repository_output = Path(__file__).resolve().parents[1] / "private-traces"
    with pytest.raises(ValueError, match="outside repository"):
        _new_external_output(repository_output)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        _new_external_output(existing)
