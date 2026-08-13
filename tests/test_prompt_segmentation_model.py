from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_contracts.foundation_vision_model import FoundationFileBinding
from artifact_contracts.prompt_segmentation_model import (
    PromptSegmentationArtifact,
    PromptSegmentationModelManifest,
    prompt_segmentation_model_bundle,
)
from localization.sam2_prompt_runtime import _validate_box, _validate_point


def _binding(root: Path, name: str, payload: bytes) -> FoundationFileBinding:
    path = root / name
    path.write_bytes(payload)
    return FoundationFileBinding(name, len(payload), hashlib.sha256(payload).hexdigest())


def test_prompt_model_artifact_revalidates_all_bound_files(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    manifest = PromptSegmentationModelManifest(
        model_id="fixture/sam2",
        source_revision="a" * 40,
        model_family="SAM2_1_HIERA_LARGE",
        license_id="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        runtime_conversion="FIXTURE_ZERO_MISSING_KEYS",
        files=tuple(
            sorted(
                (
                    _binding(root, "model.safetensors", b"weight"),
                    _binding(root, "config.json", b"{}"),
                    _binding(root, "preprocessor_config.json", b"{}"),
                ),
                key=lambda item: item.relative_path,
            )
        ),
    )
    bundle_path = tmp_path / "manifest.json"
    bundle_path.write_text(
        json.dumps(prompt_segmentation_model_bundle(manifest)), encoding="utf-8"
    )
    artifact = PromptSegmentationArtifact.load(
        model_directory=root, manifest_bundle_path=bundle_path
    )
    assert artifact.manifest.manifest_sha256 == manifest.manifest_sha256
    (root / "config.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte size|SHA-256"):
        artifact.revalidate_local_files()


def test_sam2_prompt_geometry_fails_closed() -> None:
    _validate_box((0.0, 0.0, 100.0, 50.0), width=100, height=50)
    _validate_point((100.0, 50.0), width=100, height=50)
    with pytest.raises(ValueError, match="outside"):
        _validate_box((-1.0, 0.0, 20.0, 20.0), width=100, height=50)
    with pytest.raises(ValueError, match="outside"):
        _validate_point((101.0, 25.0), width=100, height=50)
