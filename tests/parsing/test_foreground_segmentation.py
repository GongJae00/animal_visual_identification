from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shared.contracts.foreground_segmentation_model import (
    ForegroundSegmentationArtifact,
    ForegroundSegmentationModelManifest,
    foreground_segmentation_model_bundle,
)
from shared.contracts.model_file_binding import ModelFileBinding
from parsing.export.segmentation.foreground_segmentation import (
    _compute_inference_size,
    _validate_target_box,
)

def _binding(root: Path, name: str, payload: bytes) -> ModelFileBinding:
    path = root / name
    path.write_bytes(payload)
    return ModelFileBinding(name, len(payload), hashlib.sha256(payload).hexdigest())

def _manifest(root: Path) -> ForegroundSegmentationModelManifest:
    return ForegroundSegmentationModelManifest(
        model_id="fixture/birefnet",
        source_revision="a" * 40,
        model_family="BIREFNET_DYNAMIC_SWIN_V1_LARGE",
        task="HIGH_RESOLUTION_DICHOTOMOUS_IMAGE_SEGMENTATION",
        license_id="MIT",
        license_url="https://example.org/license",
        input_multiple=32,
        minimum_inference_side=256,
        maximum_inference_side=2304,
        files=tuple(
            sorted(
                (
                    _binding(root, "BiRefNet_config.py", b"pass\n"),
                    _binding(root, "birefnet.py", b"pass\n"),
                    _binding(root, "config.json", b"{}"),
                    _binding(root, "model.safetensors", b"weights"),
                ),
                key=lambda item: item.relative_path,
            )
        ),
    )

def test_foreground_model_artifact_binds_weight_config_and_executable_code(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    root.mkdir()
    manifest = _manifest(root)
    bundle_path = tmp_path / "manifest.json"
    bundle_path.write_text(
        json.dumps(foreground_segmentation_model_bundle(manifest)), encoding="utf-8"
    )
    artifact = ForegroundSegmentationArtifact.load(
        model_directory=root, manifest_bundle_path=bundle_path
    )
    assert artifact.manifest.manifest_sha256 == manifest.manifest_sha256
    (root / "birefnet.py").write_bytes(b"raise RuntimeError\n")
    with pytest.raises(ValueError, match="byte size|SHA-256"):
        artifact.revalidate_local_files()

def test_foreground_inference_size_preserves_dynamic_bounds() -> None:
    assert _compute_inference_size(
        1024, 688, multiple=32, minimum_side=256, maximum_side=2304
    ) == (1024, 704)
    assert _compute_inference_size(
        51, 75, multiple=32, minimum_side=256, maximum_side=2304
    ) == (256, 384)
    width, height = _compute_inference_size(
        8000, 2000, multiple=32, minimum_side=256, maximum_side=2304
    )
    assert (width, height) == (2304, 576)

def test_foreground_target_box_fails_closed_and_aligns_outward() -> None:
    assert _validate_target_box(None, width=100, height=50) == (0, 0, 100, 50)
    assert _validate_target_box(
        (1.2, 2.8, 90.1, 40.2), width=100, height=50
    ) == (1, 2, 91, 41)
    with pytest.raises(ValueError, match="outside"):
        _validate_target_box((-1.0, 0.0, 10.0, 10.0), width=100, height=50)
    with pytest.raises(ValueError, match="four"):
        _validate_target_box((0.0, 1.0, 2.0), width=100, height=50)  # type: ignore[arg-type]

def test_foreground_manifest_rejects_unaligned_dynamic_bounds(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    manifest = _manifest(root)
    fields = manifest.to_dict()
    fields["minimum_inference_side"] = 255
    with pytest.raises(ValueError, match="bounds"):
        ForegroundSegmentationModelManifest.from_dict(fields)
