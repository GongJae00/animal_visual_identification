"""Bind an exact candidate animal parser to source, models, policy, and reports.

Kinds (default animal_parsing_runtime):
  animal_parsing_runtime, foreground, foundation, instance, prompt
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

from contracts.animal_parsing_runtime import (
    QUALIFICATION,
    AnimalParsingRuntimeManifest,
    ParsingEvaluationBinding,
    animal_parsing_runtime_bundle,
)
from contracts.foreground_segmentation_model import (
    ForegroundSegmentationArtifact,
    ForegroundSegmentationModelManifest,
    foreground_segmentation_model_bundle,
)
from contracts.foundation_vision_model import (
    FoundationFileBinding,
    FoundationModelFamily,
    FoundationModelUsageLane,
    FoundationVisionModelManifest,
    foundation_model_bundle,
)
from contracts.instance_segmentation_model import (
    InstanceSegmentationArtifact,
    InstanceSegmentationModelManifest,
    instance_segmentation_model_bundle,
)
from contracts.model_file_binding import ModelFileBinding
from contracts.prompt_segmentation_model import (
    PromptSegmentationModelManifest,
    prompt_segmentation_model_bundle,
)
from contracts.source_provenance import build_source_provenance
from foundation.protected_io import (
    json_document_bytes,
    read_strict_json_document,
    write_private_json_bundle,
)
from foundation.protected_publication import fsync_directory
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file
from parsing.full_segment.animal_parsing import (
    PARSING_ONTOLOGY,
    PARSING_ONTOLOGY_DESCRIPTION,
    AnimalParsingPolicy,
)

_SOURCE_PATHS = (
    "embedding/methods/full_segment/preparation/materialization.py",
    "parsing/full_segment/animal_instance_segmentation.py",
    "parsing/full_segment/animal_parsing.py",
    "parsing/full_segment/foreground_segmentation.py",
    "parsing/full_segment/full_segment_cache.py",
)


def _run_animal_parsing_runtime(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--instance-model-directory", type=Path, required=True)
    parser.add_argument("--instance-model-manifest", type=Path, required=True)
    parser.add_argument(
        "--evaluation-report", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8, 16), default=4)
    parser.add_argument("--publication-workers", type=int, choices=(1, 4), default=4)
    args = parser.parse_args(argv)
    bundle = build_manifest(
        repository_root=args.repository_root,
        model_directory=args.model_directory,
        model_manifest=args.model_manifest,
        instance_model_directory=args.instance_model_directory,
        instance_model_manifest=args.instance_model_manifest,
        evaluation_reports=tuple(args.evaluation_report),
        batch_size=args.batch_size,
        publication_workers=args.publication_workers,
    )
    _write_output(args.output, json_document_bytes(bundle))
    print(
        json.dumps(
            {
                "status": "CREATED_ANIMAL_PARSING_RUNTIME_CANDIDATE_MANIFEST",
                "qualification": QUALIFICATION,
                "output": str(args.output),
                "manifest_sha256": bundle["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_manifest(
    *,
    repository_root: Path,
    model_directory: Path,
    model_manifest: Path,
    instance_model_directory: Path,
    instance_model_manifest: Path,
    evaluation_reports: tuple[Path, ...],
    batch_size: int = 4,
    publication_workers: int = 4,
) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    if repository_root.is_symlink() or not root.is_dir():
        raise ValueError("animal parsing repository root must be a regular directory")
    foreground = ForegroundSegmentationArtifact.load(
        model_directory=model_directory,
        manifest_bundle_path=model_manifest,
    )
    instance = InstanceSegmentationArtifact.load(
        model_directory=instance_model_directory,
        manifest_bundle_path=instance_model_manifest,
    )
    policy = AnimalParsingPolicy()
    reports = tuple(
        sorted(
            (_evaluation_binding(path, policy=policy) for path in evaluation_reports),
            key=lambda item: item.name,
        )
    )
    source_provenance = build_source_provenance(
        (root.joinpath(*relative_path.split("/")) for relative_path in _SOURCE_PATHS),
        logical_component="cvi.animal_parsing_runtime.v2",
    )
    sources = tuple(
        ModelFileBinding(
            relative_path=row["relative_path"],
            byte_size=row["byte_size"],
            sha256=row["content_sha256"],
        )
        for row in source_provenance["code_source_files"]
    )
    manifest = AnimalParsingRuntimeManifest(
        parser_family="RF_DETR_BIREFNET_BATCHED_SEEDED_EXPANSION_EXCLUSIVE_OWNERSHIP_V2",
        qualification=QUALIFICATION,
        ontology=PARSING_ONTOLOGY,
        ontology_description=PARSING_ONTOLOGY_DESCRIPTION,
        supported_classes=tuple(sorted(policy.class_names)),
        policy=policy.to_dict(),
        policy_sha256=policy.policy_sha256,
        foreground_model_manifest_sha256=foreground.manifest.manifest_sha256,
        foreground_model_bundle_raw_sha256=foreground.bundle_sha256,
        instance_model_manifest_sha256=instance.manifest.manifest_sha256,
        instance_model_bundle_raw_sha256=instance.bundle_sha256,
        inference_batching={
            "job_batch_size": batch_size,
            "instance_batch_size": batch_size,
            "foreground_batch_size": batch_size,
            "job_ordering": "SOURCE_SHA256_ASC",
            "publication_workers": publication_workers,
            "shape_policy": "EXACT_PREPROCESSED_SHAPE_BUCKETS",
            "oom_policy": "FAIL_CLOSED_NO_RETRY",
        },
        frozen_cache={
            "array_encoding": "BASE64_ZLIB_C_ORDER",
            "zlib_level": 1,
            "retained_arrays": [
                "instance_probability",
                "foreground_probability",
                "ownership_probability",
                "hard_mask",
            ],
        },
        runtime_libraries={
            name: importlib.metadata.version(distribution)
            for name, distribution in {
                "numpy": "numpy",
                "pillow": "Pillow",
                "torch": "torch",
                "torchvision": "torchvision",
                "transformers": "transformers",
            }.items()
        },
        source_files=sources,
        evaluation_reports=reports,
    )
    return animal_parsing_runtime_bundle(manifest)


def _evaluation_binding(
    path: Path, *, policy: AnimalParsingPolicy
) -> ParsingEvaluationBinding:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError("animal parsing evaluation report must be a regular file")
    document = read_strict_json_document(resolved, maximum_bytes=134_217_728)
    payload = document.payload
    if (
        payload.get("parsing_policy_sha256") != policy.policy_sha256
        or payload.get("parsing_policy") != policy.to_dict()
        or not isinstance(payload.get("schema_version"), str)
        or not isinstance(payload.get("interpretation"), str)
    ):
        raise ValueError("animal parsing evaluation policy or contract differs")
    return ParsingEvaluationBinding(
        name=resolved.parent.name,
        schema_version=payload["schema_version"],
        interpretation=payload["interpretation"],
        byte_size=resolved.stat().st_size,
        raw_sha256=document.raw_sha256,
        content_sha256=content_sha256(payload),
    )


def _write_output(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    parent = path.parent.resolve(strict=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("animal parsing runtime manifest write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(parent)


_FOREGROUND_FILES = (
    "BiRefNet_config.py",
    "birefnet.py",
    "config.json",
    "model.safetensors",
)
_INSTANCE_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")
_PROMPT_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)
_FOUNDATION_SPECS = {
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
_KINDS = (
    "animal_parsing_runtime",
    "foreground",
    "foundation",
    "instance",
    "prompt",
)


def _model_file_binding(path: Path, root: Path, subject: str) -> ModelFileBinding:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{subject} is not a regular file: {path}")
    retained = read_retained_regular_file(path, subject=subject)
    return ModelFileBinding(
        path.relative_to(root).as_posix(), retained.byte_count, retained.sha256
    )


def _foundation_file_binding(path: Path, root: Path, subject: str) -> FoundationFileBinding:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{subject} is not a regular file: {path}")
    retained = read_retained_regular_file(path, subject=subject)
    return FoundationFileBinding(
        path.relative_to(root).as_posix(), retained.byte_count, retained.sha256
    )


def _run_foreground(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Build an exact local BiRefNet foreground-model manifest."
    )
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
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
        files=tuple(
            _model_file_binding(root / name, root, "foreground model input")
            for name in _FOREGROUND_FILES
        ),
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


def _run_foundation(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Build an exact local foundation-model artifact manifest."
    )
    parser.add_argument("--profile", choices=sorted(_FOUNDATION_SPECS), required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite foundation model manifest")
    root = args.model_directory.resolve(strict=True)
    spec = _FOUNDATION_SPECS[args.profile]
    sources = tuple(
        _foundation_file_binding(path, root, "foundation model input")
        for path in sorted(root.glob("*.py"), key=lambda value: value.name)
    )
    manifest = FoundationVisionModelManifest(
        **spec,
        source_revision=args.source_revision,
        weight=_foundation_file_binding(root / "model.safetensors", root, "foundation model input"),
        config=_foundation_file_binding(root / "config.json", root, "foundation model input"),
        preprocessor=_foundation_file_binding(
            root / "preprocessor_config.json", root, "foundation model input"
        ),
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


def _run_instance(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Build an exact local RF-DETR instance-model manifest."
    )
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
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
        files=tuple(
            _model_file_binding(root / name, root, "instance model input")
            for name in _INSTANCE_FILES
        ),
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


def _run_prompt(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Build an exact local prompt-segmentation model manifest."
    )
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite prompt model manifest")
    root = args.model_directory.resolve(strict=True)
    manifest = PromptSegmentationModelManifest(
        model_id="facebook/sam2.1-hiera-large",
        source_revision=args.source_revision,
        model_family="SAM2_1_HIERA_LARGE",
        license_id="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        runtime_conversion="SAM2_VIDEO_CHECKPOINT_TO_IMAGE_MODEL_ZERO_MISSING_UNEXPECTED_MISMATCHED_KEYS",
        files=tuple(
            sorted(
                (
                    _foundation_file_binding(root / name, root, "prompt model input")
                    for name in _PROMPT_FILES
                ),
                key=lambda item: item.relative_path,
            )
        ),
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    kind = "animal_parsing_runtime"
    if argv and argv[0] in _KINDS:
        kind = argv[0]
        argv = argv[1:]
    elif "--kind" in argv:
        index = argv.index("--kind")
        if index + 1 >= len(argv):
            raise SystemExit("error: argument --kind: expected one argument")
        kind = argv[index + 1]
        if kind not in _KINDS:
            raise SystemExit(f"error: argument --kind: invalid choice: {kind!r}")
        argv = argv[:index] + argv[index + 2 :]
    runners = {
        "animal_parsing_runtime": _run_animal_parsing_runtime,
        "foreground": _run_foreground,
        "foundation": _run_foundation,
        "instance": _run_instance,
        "prompt": _run_prompt,
    }
    return runners[kind](argv)


if __name__ == "__main__":
    raise SystemExit(main())
