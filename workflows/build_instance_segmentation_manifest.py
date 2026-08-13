"""Build an exact local RF-DETR instance-model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from contracts.instance_segmentation_model import (
    InstanceSegmentationModelManifest,
    instance_segmentation_model_bundle,
)
from contracts.model_file_binding import ModelFileBinding
from foundation.protected_io import write_private_json_bundle

_REQUIRED_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite instance model manifest")
    root = args.model_directory.resolve(strict=True)
    manifest = InstanceSegmentationModelManifest(
        model_id="Roboflow/rf-detr-segmentation",
        source_revision=args.source_revision,
        model_family="RF_DETR_SEGMENTATION_COCO",
        training_label_space="COCO_2017_INSTANCE_91_CATEGORY_IDS",
        license_id="Apache-2.0",
        license_url="https://github.com/roboflow/rf-detr/blob/develop/LICENSE",
        files=tuple(_binding(root / name, root) for name in _REQUIRED_FILES),
    )
    write_private_json_bundle(((args.output, instance_segmentation_model_bundle(manifest)),))
    print(
        json.dumps(
            {
                "status": "CREATED_INSTANCE_SEGMENTATION_MODEL_MANIFEST",
                "manifest_sha256": manifest.manifest_sha256,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _binding(path: Path, root: Path) -> ModelFileBinding:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"instance model input is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return ModelFileBinding(path.relative_to(root).as_posix(), size, digest.hexdigest())


if __name__ == "__main__":
    raise SystemExit(main())
