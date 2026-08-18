"""Build an exact local prompt-segmentation model manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts.foundation_vision_model import FoundationFileBinding
from contracts.prompt_segmentation_model import (
    PromptSegmentationModelManifest,
    prompt_segmentation_model_bundle,
)
from foundation.protected_io import write_private_json_bundle
from foundation.retained_file import read_retained_regular_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite prompt model manifest")
    root = args.model_directory.resolve(strict=True)
    names = (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    )
    manifest = PromptSegmentationModelManifest(
        model_id="facebook/sam2.1-hiera-large",
        source_revision=args.source_revision,
        model_family="SAM2_1_HIERA_LARGE",
        license_id="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        runtime_conversion="SAM2_VIDEO_CHECKPOINT_TO_IMAGE_MODEL_ZERO_MISSING_UNEXPECTED_MISMATCHED_KEYS",
        files=tuple(sorted((_binding(root / name, root) for name in names), key=lambda item: item.relative_path)),
    )
    write_private_json_bundle(((args.output, prompt_segmentation_model_bundle(manifest)),))
    print(
        json.dumps(
            {
                "status": "CREATED_PROMPT_SEGMENTATION_MODEL_MANIFEST",
                "manifest_sha256": manifest.manifest_sha256,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _binding(path: Path, root: Path) -> FoundationFileBinding:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"prompt model input is not a regular file: {path}")
    retained = read_retained_regular_file(path, subject="prompt model input")
    return FoundationFileBinding(
        path.relative_to(root).as_posix(), retained.byte_count, retained.sha256
    )


if __name__ == "__main__":
    raise SystemExit(main())
