"""Bind an exact candidate animal parser to source, models, policy, and reports."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
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
)
from contracts.instance_segmentation_model import (
    InstanceSegmentationArtifact,
)
from contracts.model_file_binding import ModelFileBinding
from contracts.source_provenance import build_source_provenance
from foundation.protected_io import json_document_bytes, read_strict_json_document
from foundation.protected_publication import fsync_directory
from foundation.provenance import content_sha256
from localization.animal_parsing import (
    PARSING_ONTOLOGY,
    PARSING_ONTOLOGY_DESCRIPTION,
    AnimalParsingPolicy,
)

_SOURCE_PATHS = (
    "identity_methods/full_segment/materialization.py",
    "localization/animal_instance_segmentation.py",
    "localization/animal_parsing.py",
    "localization/foreground_segmentation.py",
    "localization/full_segment_cache.py",
)


def main() -> int:
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
    args = parser.parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
