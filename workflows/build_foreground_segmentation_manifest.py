"""Build an exact local BiRefNet foreground-model manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts.foreground_segmentation_model import (
    ForegroundSegmentationModelManifest,
    foreground_segmentation_model_bundle,
)
from contracts.model_file_binding import ModelFileBinding
from foundation.protected_io import write_private_json_bundle
from foundation.retained_file import read_retained_regular_file

_REQUIRED_FILES = (
    "BiRefNet_config.py",
    "birefnet.py",
    "config.json",
    "model.safetensors",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite foreground model manifest")
    root = args.model_directory.resolve(strict=True)
    manifest = ForegroundSegmentationModelManifest(
        model_id="ZhengPeng7/BiRefNet_dynamic",
        source_revision=args.source_revision,
        model_family="BIREFNET_DYNAMIC_SWIN_V1_LARGE",
        task="HIGH_RESOLUTION_DICHOTOMOUS_IMAGE_SEGMENTATION",
        license_id="MIT",
        license_url="https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE",
        input_multiple=32,
        minimum_inference_side=256,
        maximum_inference_side=2304,
        files=tuple(_binding(root / name, root) for name in _REQUIRED_FILES),
    )
    write_private_json_bundle(
        ((args.output, foreground_segmentation_model_bundle(manifest)),)
    )
    print(
        json.dumps(
            {
                "status": "CREATED_FOREGROUND_SEGMENTATION_MODEL_MANIFEST",
                "manifest_sha256": manifest.manifest_sha256,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _binding(path: Path, root: Path) -> ModelFileBinding:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"foreground model input is not a regular file: {path}")
    retained = read_retained_regular_file(path, subject="foreground model input")
    return ModelFileBinding(
        path.relative_to(root).as_posix(), retained.byte_count, retained.sha256
    )


if __name__ == "__main__":
    raise SystemExit(main())
