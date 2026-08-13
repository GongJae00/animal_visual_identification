"""Build an exact local foundation-model artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from contracts.foundation_vision_model import (
    FoundationFileBinding,
    FoundationModelFamily,
    FoundationModelUsageLane,
    FoundationVisionModelManifest,
    foundation_model_bundle,
)
from foundation.protected_io import write_private_json_bundle

_SPECS = {
    "cradio-v4-so400m": {
        "model_id": "nvidia/C-RADIOv4-SO400M",
        "family": FoundationModelFamily.CRADIO_V4,
        "license_id": "NVIDIA-Open-Model-License",
        "license_url": "https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf",
        "usage_lane": FoundationModelUsageLane.DEPLOYMENT_CANDIDATE,
        "patch_size": 16,
        "dense_feature_dimension": 1152,
        "summary_dimension": 2304,
        "preferred_resolution": 512,
        "maximum_resolution": 2048,
        "requires_local_code": True,
    },
    "dinov3-vitb16": {
        "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "family": FoundationModelFamily.DINOV3_VIT,
        "license_id": "DINOv3-License",
        "license_url": "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
        "usage_lane": FoundationModelUsageLane.RESEARCH_ONLY,
        "patch_size": 16,
        "dense_feature_dimension": 768,
        "summary_dimension": 768,
        "preferred_resolution": 512,
        "maximum_resolution": 1024,
        "requires_local_code": False,
    },
    "dinov3-vitl16": {
        "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "family": FoundationModelFamily.DINOV3_VIT,
        "license_id": "DINOv3-License",
        "license_url": "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
        "usage_lane": FoundationModelUsageLane.RESEARCH_ONLY,
        "patch_size": 16,
        "dense_feature_dimension": 1024,
        "summary_dimension": 1024,
        "preferred_resolution": 512,
        "maximum_resolution": 1024,
        "requires_local_code": False,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(_SPECS), required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite foundation model manifest")
    root = args.model_directory.resolve(strict=True)
    spec = _SPECS[args.profile]
    sources = tuple(
        _binding(path, root)
        for path in sorted(root.glob("*.py"), key=lambda value: value.name)
    )
    manifest = FoundationVisionModelManifest(
        **spec,
        source_revision=args.source_revision,
        weight=_binding(root / "model.safetensors", root),
        config=_binding(root / "config.json", root),
        preprocessor=_binding(root / "preprocessor_config.json", root),
        executable_sources=sources,
    )
    write_private_json_bundle(((args.output, foundation_model_bundle(manifest)),))
    print(
        json.dumps(
            {
                "status": "CREATED_FOUNDATION_VISION_MODEL_MANIFEST",
                "model_id": manifest.model_id,
                "manifest_sha256": manifest.manifest_sha256,
                "executable_source_count": len(sources),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _binding(path: Path, root: Path) -> FoundationFileBinding:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"foundation model input is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return FoundationFileBinding(path.relative_to(root).as_posix(), size, digest.hexdigest())


if __name__ == "__main__":
    raise SystemExit(main())
