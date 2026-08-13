from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from artifact_contracts.pretrained_supporting_asset_intake import (
    PretrainedSupportingAssetKind,
    PretrainedSupportingAssetSourceContract,
    audit_pretrained_supporting_asset,
)
from artifact_contracts.pretrained_weight_intake import (
    PretrainedWeightChecksumAuthority,
    PretrainedWeightFileFormat,
    PretrainedWeightSourceContract,
    PretrainedWeightUsageLane,
    audit_pretrained_weight_file,
)
from foundation.provenance import content_sha256
from identity_methods.appearance import ReceiptBoundDinov2Small


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _admitted_preprocessor() -> dict:
    return {
        "crop_size": {"height": 224, "width": 224},
        "do_center_crop": True,
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_processor_type": "BitImageProcessor",
        "image_std": [0.229, 0.224, 0.225],
        "resample": 3,
        "rescale_factor": 1 / 255,
        "size": {"shortest_edge": 256},
    }


def _dinov2_config() -> dict:
    return {
        "architectures": ["Dinov2Model"],
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 384,
        "image_size": 518,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-6,
        "layerscale_value": 1.0,
        "mlp_ratio": 4,
        "model_type": "dinov2",
        "num_attention_heads": 6,
        "num_channels": 3,
        "num_hidden_layers": 12,
        "patch_size": 14,
        "qkv_bias": True,
        "use_swiglu_ffn": False,
    }


def _intake_bundle(source, receipt, schema_version: str) -> dict:
    provenance = {"fixture": "synthetic non-model bytes"}
    return {
        "schema_version": schema_version,
        "source_contract_sha256": source.contract_sha256,
        "source_contract": source.to_dict(),
        "receipt_sha256": receipt.receipt_sha256,
        "receipt": receipt.to_dict(),
        "tool_provenance": provenance,
        "tool_provenance_sha256": content_sha256(provenance),
    }


def _local_fixture(
    root: Path,
    *,
    preprocessor: dict | None = None,
    config: dict | None = None,
) -> dict[str, Path]:
    model_directory = root / "local-model"
    model_directory.mkdir()
    weight = model_directory / "model.safetensors"
    processor_path = model_directory / "preprocessor_config.json"
    config_path = model_directory / "config.json"
    license_path = root / "LICENSE.snapshot"
    training_path = root / "TRAINING.snapshot"
    weight.write_bytes(b"synthetic safetensors fixture")
    _write_json(processor_path, preprocessor or _admitted_preprocessor())
    _write_json(config_path, config or _dinov2_config())
    license_path.write_text("fixture license only", encoding="utf-8")
    training_path.write_text("fixture lineage only", encoding="utf-8")

    lane = PretrainedWeightUsageLane.RESEARCH_ONLY
    weight_source = PretrainedWeightSourceContract(
        source_model_id="facebook/dinov2-small",
        source_revision="0123456789abcdef",
        source_model_page_url="https://example.org/facebook/dinov2-small",
        source_file_url="https://example.org/model.safetensors",
        weight_filename="model.safetensors",
        license_id="FIXTURE-TEST-ONLY",
        license_url="https://example.org/license",
        license_snapshot_sha256=_sha256(license_path),
        license_usage_lane=lane,
        training_description="Synthetic bytes; no model admission claim.",
        training_description_url="https://example.org/training",
        training_description_snapshot_sha256=_sha256(training_path),
        expected_file_bytes=weight.stat().st_size,
        expected_sha256=_sha256(weight),
        checksum_authority=(
            PretrainedWeightChecksumAuthority.UNVERIFIED_SHA256
        ),
        target_lane=lane,
        file_format=PretrainedWeightFileFormat.SAFETENSORS,
    )
    weight_receipt = audit_pretrained_weight_file(
        weight_path=weight,
        license_snapshot_path=license_path,
        training_description_snapshot_path=training_path,
        source=weight_source,
    )
    preprocessor_source = PretrainedSupportingAssetSourceContract(
        source_model_id=weight_source.source_model_id,
        source_revision=weight_source.source_revision,
        source_model_page_url=weight_source.source_model_page_url,
        source_file_url="https://example.org/preprocessor_config.json",
        asset_filename="preprocessor_config.json",
        asset_kind=PretrainedSupportingAssetKind.PREPROCESSOR_CONFIG,
        expected_file_bytes=processor_path.stat().st_size,
        expected_sha256=_sha256(processor_path),
        license_id="FIXTURE-TEST-ONLY",
        license_url="https://example.org/license",
        license_snapshot_sha256=_sha256(license_path),
        license_usage_lane=lane,
        associated_pretrained_weight_receipt_sha256=(
            weight_receipt.receipt_sha256
        ),
        target_lane=lane,
    )
    preprocessor_receipt = audit_pretrained_supporting_asset(
        asset_path=processor_path,
        license_snapshot_path=license_path,
        source=preprocessor_source,
        associated_weight_source=weight_source,
        associated_weight_receipt=weight_receipt,
    )
    weight_bundle = root / "weight-intake.json"
    processor_bundle = root / "preprocessor-intake.json"
    _write_json(
        weight_bundle,
        _intake_bundle(
            weight_source,
            weight_receipt,
            "cvi.pretrained_weight_intake_bundle.v1",
        ),
    )
    _write_json(
        processor_bundle,
        _intake_bundle(
            preprocessor_source,
            preprocessor_receipt,
            "cvi.pretrained_supporting_asset_intake_bundle.v1",
        ),
    )
    return {
        "model_directory": model_directory,
        "weight": weight,
        "preprocessor": processor_path,
        "config": config_path,
        "weight_bundle": weight_bundle,
        "preprocessor_bundle": processor_bundle,
    }


def _receipt_bound(paths: dict[str, Path]) -> ReceiptBoundDinov2Small:
    return ReceiptBoundDinov2Small(
        model_directory=paths["model_directory"],
        weight_intake_bundle=paths["weight_bundle"],
        preprocessor_intake_bundle=paths["preprocessor_bundle"],
    )


class _DummyHfDino(torch.nn.Module):
    def __init__(self, *, nonfinite: bool = False) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.arange(1, 385, dtype=torch.float32))
        self.nonfinite = nonfinite

    def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        output = self.anchor.unsqueeze(0).expand(pixel_values.shape[0], -1)
        if self.nonfinite:
            output = output.clone()
            output[:, 0] = torch.inf
        return SimpleNamespace(pooler_output=output)


def _fake_transformers(model: torch.nn.Module) -> tuple[ModuleType, list]:
    calls: list[tuple[tuple, dict]] = []

    class FakeDinov2Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return model

    module = ModuleType("transformers")
    module.Dinov2Model = FakeDinov2Model
    return module, calls


class ReceiptBoundAppearanceTests(unittest.TestCase):
    def test_local_only_load_normalizes_and_exposes_gallery_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = _local_fixture(Path(temporary))
            evidencer = _receipt_bound(paths)
            fake_transformers, calls = _fake_transformers(_DummyHfDino())
            with patch.dict(sys.modules, {"transformers": fake_transformers}), patch(
                "torch.hub.load", side_effect=AssertionError("Torch Hub used")
            ):
                embeddings = evidencer.extract_batch([
                    Image.new("L", (320, 160), color=128),
                    Image.new("RGB", (8, 9)),
                ])
            self.assertEqual(embeddings.shape, (2, 384))
            self.assertEqual(embeddings.dtype, np.float32)
            self.assertTrue(np.isfinite(embeddings).all())
            np.testing.assert_allclose(
                np.linalg.norm(embeddings, axis=1), np.ones(2), atol=1e-6
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], (str(paths["model_directory"]),))
            self.assertEqual(
                calls[0][1],
                {
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "use_safetensors": True,
                },
            )
            fields = evidencer.gallery_contract_fields
            self.assertEqual(fields["model_sha256"], _sha256(paths["weight"]))
            self.assertEqual(
                fields["model_config_sha256"], _sha256(paths["config"])
            )
            self.assertEqual(
                fields["preprocessor_sha256"], _sha256(paths["preprocessor"])
            )
            self.assertEqual(len(fields["weight_intake_receipt_sha256"]), 64)
            self.assertEqual(
                len(fields["preprocessor_intake_receipt_sha256"]), 64
            )

    def test_admitted_hf_preprocessing_is_shortest_bicubic_center_crop(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = _local_fixture(Path(temporary))
            evidencer = _receipt_bound(paths)
            source = np.arange(160 * 320 * 3, dtype=np.uint32)
            image = Image.fromarray(
                (source % 256).astype(np.uint8).reshape(160, 320, 3)
            )
            observed = evidencer._preprocess([image]).cpu().numpy()[0]

            resized = image.convert("RGB").resize(
                (512, 256), Image.Resampling.BICUBIC
            )
            cropped = resized.crop((144, 16, 368, 240))
            expected = np.asarray(cropped, dtype=np.float32).transpose(2, 0, 1)
            expected *= 1 / 255
            expected -= np.asarray([0.485, 0.456, 0.406])[:, None, None]
            expected /= np.asarray([0.229, 0.224, 0.225])[:, None, None]
            np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)

    def test_bundle_self_hash_and_source_binding_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = _local_fixture(Path(temporary))
            bundle = json.loads(paths["weight_bundle"].read_text(encoding="utf-8"))
            bundle["receipt_sha256"] = "0" * 64
            _write_json(paths["weight_bundle"], bundle)
            with self.assertRaisesRegex(ValueError, "bundle receipt hash"):
                _receipt_bound(paths)

        with TemporaryDirectory() as temporary:
            paths = _local_fixture(Path(temporary))
            bundle = json.loads(
                paths["preprocessor_bundle"].read_text(encoding="utf-8")
            )
            bundle["source_contract"][
                "associated_pretrained_weight_receipt_sha256"
            ] = "0" * 64
            bundle["source_contract_sha256"] = content_sha256(
                bundle["source_contract"]
            )
            _write_json(paths["preprocessor_bundle"], bundle)
            with self.assertRaisesRegex(ValueError, "source_contract_sha256"):
                _receipt_bound(paths)

    def test_exact_weight_preprocessor_bytes_and_structure_are_required(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = _local_fixture(Path(temporary))
            paths["weight"].write_bytes(b"substituted weight bytes")
            with self.assertRaisesRegex(ValueError, "byte size|SHA-256"):
                _receipt_bound(paths)

        with TemporaryDirectory() as temporary:
            paths = _local_fixture(Path(temporary))
            payload = bytearray(paths["preprocessor"].read_bytes())
            payload[-1] = ord(" ")
            paths["preprocessor"].write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _receipt_bound(paths)

        with TemporaryDirectory() as temporary:
            paths = _local_fixture(Path(temporary))
            bundle = json.loads(
                paths["preprocessor_bundle"].read_text(encoding="utf-8")
            )
            bundle["receipt"]["json_structure_sha256"] = "0" * 64
            bundle["receipt_sha256"] = content_sha256(bundle["receipt"])
            _write_json(paths["preprocessor_bundle"], bundle)
            with self.assertRaisesRegex(ValueError, "JSON structure hash"):
                _receipt_bound(paths)

    def test_preprocessor_semantics_and_small_architecture_fail_closed(self) -> None:
        processor = _admitted_preprocessor()
        processor["resample"] = 2
        with TemporaryDirectory() as temporary:
            paths = _local_fixture(Path(temporary), preprocessor=processor)
            with self.assertRaisesRegex(ValueError, "preprocessor structure"):
                _receipt_bound(paths)

        config = _dinov2_config()
        config["hidden_size"] = 768
        with TemporaryDirectory() as temporary:
            paths = _local_fixture(Path(temporary), config=config)
            with self.assertRaisesRegex(ValueError, "hidden_size"):
                _receipt_bound(paths)

    def test_nonfinite_pooler_output_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = _local_fixture(Path(temporary))
            evidencer = _receipt_bound(paths)
            fake_transformers, _ = _fake_transformers(
                _DummyHfDino(nonfinite=True)
            )
            with patch.dict(
                sys.modules, {"transformers": fake_transformers}
            ), self.assertRaisesRegex(RuntimeError, "non-finite"):
                evidencer.extract(Image.new("RGB", (224, 224)))


if __name__ == "__main__":
    unittest.main()
