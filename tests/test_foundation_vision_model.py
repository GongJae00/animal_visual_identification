from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from artifact_contracts.foundation_vision_model import (
    FoundationFileBinding,
    FoundationModelFamily,
    FoundationModelUsageLane,
    FoundationVisionArtifact,
    FoundationVisionModelManifest,
    foundation_model_bundle,
)
from foundation.provenance import content_sha256
from localization.foundation_dense_runtime import _prepare_images


def _binding(root: Path, name: str, payload: bytes) -> FoundationFileBinding:
    path = root / name
    path.write_bytes(payload)
    return FoundationFileBinding(name, len(payload), hashlib.sha256(payload).hexdigest())


def test_foundation_artifact_binds_weight_config_preprocessor_and_code(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    manifest = FoundationVisionModelManifest(
        model_id="fixture/model",
        source_revision="a" * 40,
        family=FoundationModelFamily.CRADIO_V4,
        license_id="fixture-license",
        license_url="https://example.org/license",
        usage_lane=FoundationModelUsageLane.RESEARCH_ONLY,
        patch_size=16,
        dense_feature_dimension=32,
        summary_dimension=64,
        preferred_resolution=512,
        maximum_resolution=1024,
        requires_local_code=True,
        weight=_binding(model, "model.safetensors", b"weights"),
        config=_binding(model, "config.json", b"{}"),
        preprocessor=_binding(model, "preprocessor_config.json", b"{}"),
        executable_sources=(_binding(model, "modeling.py", b"pass\n"),),
    )
    bundle = foundation_model_bundle(manifest)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    artifact = FoundationVisionArtifact.load(
        model_directory=model, manifest_bundle_path=bundle_path
    )
    assert artifact.manifest.manifest_sha256 == content_sha256(bundle["manifest"])
    (model / "modeling.py").write_bytes(b"raise RuntimeError\n")
    with pytest.raises(ValueError, match="changed|SHA-256|byte (count|size)"):
        artifact.revalidate_local_files()


def test_foundation_preprocessing_preserves_aspect_and_patch_validity() -> None:
    image = Image.new("RGB", (640, 320), color=(255, 0, 0))
    values, validity, transforms = _prepare_images(
        (image,),
        resolution=512,
        patch_size=16,
        family=FoundationModelFamily.CRADIO_V4,
    )
    assert values.shape == (1, 3, 512, 512)
    assert validity[0].shape == (32, 32)
    assert validity[0].sum() == 32 * 16
    assert transforms[0].resized_width == 512
    assert transforms[0].resized_height == 256
    assert transforms[0].pad_top == 128


def test_foundation_manifest_rejects_unbound_local_code() -> None:
    binding = FoundationFileBinding("model.safetensors", 1, "a" * 64)
    with pytest.raises(ValueError, match="local-code flag"):
        FoundationVisionModelManifest(
            model_id="fixture/model",
            source_revision="b" * 40,
            family=FoundationModelFamily.CRADIO_V4,
            license_id="fixture",
            license_url="https://example.org/license",
            usage_lane=FoundationModelUsageLane.RESEARCH_ONLY,
            patch_size=16,
            dense_feature_dimension=32,
            summary_dimension=64,
            preferred_resolution=512,
            maximum_resolution=1024,
            requires_local_code=True,
            weight=binding,
            config=FoundationFileBinding("config.json", 1, "b" * 64),
            preprocessor=FoundationFileBinding("preprocessor.json", 1, "c" * 64),
            executable_sources=(),
        )
