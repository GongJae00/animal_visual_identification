"""Run a frozen-feature parsed whole-body ReID diagnostic on YT-BB-Dog tracks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

from shared.contracts.animal_parsing_runtime import (
    SUPPORTED_BUNDLE_SCHEMAS as SUPPORTED_PARSING_BUNDLE_SCHEMAS,
)
from shared.contracts.animal_parsing_runtime import (
    AnimalParsingRuntimeManifest,
)
from shared.contracts.foreground_segmentation_model import (
    ForegroundSegmentationArtifact,
)
from shared.contracts.instance_segmentation_model import (
    InstanceSegmentationArtifact,
)
from data.adapters import adapt_yt_bb_dog
from data.source_lock import get_record
from data.types import UnifiedCanidSample
from evaluation.search_metrics.metrics import (
    compute_cosine_score_matrix,
    evaluate_multi_template_closed_set,
    identity_clustered_bootstrap_ci,
)
from shared.foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from shared.foundation.protected_publication import (
    admit_new_external_output,
    fsync_directory,
    rename_directory_noreplace,
)
from shared.foundation.provenance import content_sha256
from enrollment.registry.generated_identity_registry import (
    GeneratedIdentityRecord,
    GeneratedIdentityRegistry,
    create_provisional_identity,
)
from identification.export.appearance import ReceiptBoundDinov2Small
from parsing.export.segmentation.animal_instance_segmentation import (
    AnimalInstanceSegmentationRuntime,
)
from parsing.export.segmentation.animal_parsing import (
    AnimalIdentityCrop,
    AnimalParsingPolicy,
    AnimalParsingRuntime,
    ParsedAnimalInstance,
    materialize_identity_crop,
)
from parsing.export.segmentation.foreground_segmentation import ForegroundSegmentationRuntime

REPORT_SCHEMA = "cvi.parsed_body_reid_diagnostic.v2"
GENERATOR_ID = "cvi.yt-bb-dog.video-track:v1"
INTERPRETATION = (
    "WITHIN_VIDEO_TRACK_CLOSED_SET_FROZEN_FEATURE_DIAGNOSTIC_NOT_LIFELONG_"
    "IDENTITY_OR_CROSS_SESSION_VALIDATION"
)
_VIEWS = ("masked_rgb", "box_rgb", "background_plus_silhouette")


@dataclass(frozen=True, slots=True)
class _AcceptedFrame:
    sample: UnifiedCanidSample
    generated_identity_id: str
    crop: AnimalIdentityCrop
    quality: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evenly_spaced(
    samples: Sequence[UnifiedCanidSample], limit: int
) -> tuple[UnifiedCanidSample, ...]:
    def frame_order(sample: UnifiedCanidSample) -> tuple[int, str]:
        suffix = Path(sample.image_path).stem.rpartition("_")[2]
        if not suffix.isdigit():
            raise ValueError("YT-BB-Dog frame filename lacks a numeric index")
        return int(suffix), sample.sample_id

    ordered = tuple(sorted(samples, key=frame_order))
    if len(ordered) <= limit:
        return ordered
    indices = tuple(index * (len(ordered) - 1) // (limit - 1) for index in range(limit))
    return tuple(ordered[index] for index in indices)


def _candidate_tracks(
    samples: Sequence[UnifiedCanidSample],
    *,
    candidate_limit: int,
    frames_per_identity: int,
) -> tuple[tuple[str, tuple[UnifiedCanidSample, ...]], ...]:
    if candidate_limit <= 0 or frames_per_identity < 2:
        raise ValueError("candidate limit must be positive and frames per identity at least two")
    grouped: dict[str, list[UnifiedCanidSample]] = defaultdict(list)
    for sample in samples:
        if (
            sample.dataset_name == "yt-bb-dog"
            and sample.split_role == "train"
            and sample.raw_identity_id is not None
        ):
            grouped[sample.raw_identity_id].append(sample)
    eligible = [
        (identity, _evenly_spaced(grouped[identity], frames_per_identity))
        for identity in sorted(grouped)
        if len(grouped[identity]) >= 2
    ]
    return tuple(eligible[:candidate_limit])


def _identity_instance(
    instances: Sequence[ParsedAnimalInstance],
) -> tuple[ParsedAnimalInstance | None, str | None]:
    dogs = tuple(instance for instance in instances if instance.class_name == "dog")
    if not dogs:
        return None, "NO_DOG_INSTANCE"
    if len(dogs) != 1:
        return None, "MULTIPLE_DOG_INSTANCES"
    if dogs[0].quality.state != "USABLE":
        return None, f"DOG_{dogs[0].quality.state}"
    return dogs[0], None


def _background_plus_silhouette(crop: AnimalIdentityCrop) -> Image.Image:
    box = np.asarray(crop.box_rgb, dtype=np.uint8).copy()
    foreground = np.asarray(crop.mask, dtype=np.uint8) > 0
    box[foreground] = (127, 127, 127)
    return Image.fromarray(box, mode="RGB")


def _view_image(frame: _AcceptedFrame, view: str) -> Image.Image:
    if view == "masked_rgb":
        return frame.crop.masked_rgb
    if view == "box_rgb":
        return frame.crop.box_rgb
    if view == "background_plus_silhouette":
        return _background_plus_silhouette(frame.crop)
    raise ValueError(f"unsupported parsed-body view: {view!r}")


def _evaluate_view(
    frames_by_identity: Sequence[tuple[str, Sequence[_AcceptedFrame]]],
    embeddings: np.ndarray,
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    gallery_indices: list[int] = []
    query_indices: list[int] = []
    identity_ids: list[str] = []
    sample_ids: list[str] = []
    offset = 0
    for identity_id, frames in frames_by_identity:
        if len(frames) < 2:
            raise ValueError("each parsed-body identity requires gallery and query frames")
        identity_ids.extend([identity_id] * len(frames))
        sample_ids.extend(frame.sample.sample_id for frame in frames)
        gallery_indices.append(offset)
        query_indices.extend(range(offset + 1, offset + len(frames)))
        offset += len(frames)
    if embeddings.shape != (offset, 384):
        raise ValueError("parsed-body embedding matrix shape differs")
    scores = compute_cosine_score_matrix(
        embeddings[query_indices], embeddings[gallery_indices]
    )
    metrics = evaluate_multi_template_closed_set(
        scores,
        query_identity_ids=np.asarray([identity_ids[index] for index in query_indices]),
        gallery_template_identity_ids=np.asarray(
            [identity_ids[index] for index in gallery_indices]
        ),
        self_match_policy="exclude",
        query_template_ids=np.asarray([sample_ids[index] for index in query_indices]),
        gallery_template_ids=np.asarray(
            [sample_ids[index] for index in gallery_indices]
        ),
        rank_ks=(1, 5, 10),
    )
    query_rows = []
    for row, frame_index in zip(metrics["query_rows"], query_indices, strict=True):
        query_rows.append(
            {
                "sample_id": sample_ids[frame_index],
                "generated_identity_id": identity_ids[frame_index],
                "relevant_rank": row["relevant_rank"],
                "reciprocal_rank": row["reciprocal_rank"],
            }
        )
    return {
        "gallery_identities": metrics["num_gallery_identities"],
        "gallery_templates": metrics["num_gallery_templates"],
        "queries": metrics["num_queries"],
        "Rank-1": metrics["Rank-1"],
        "Rank-5": metrics["Rank-5"],
        "Rank-10": metrics["Rank-10"],
        "MRR": metrics["MRR"],
        "identity_clustered_rank1_ci": identity_clustered_bootstrap_ci(
            metrics["query_rows"],
            metric="Rank-1",
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "query_rows": query_rows,
    }


def _load_parsing_contract(
    path: Path, *, repository_root: Path
) -> tuple[AnimalParsingRuntimeManifest, str]:
    document = read_strict_json_document(path, maximum_bytes=16_777_216)
    bundle = document.payload
    if (
        set(bundle) != {"schema_version", "manifest_sha256", "manifest"}
        or bundle["schema_version"] not in SUPPORTED_PARSING_BUNDLE_SCHEMAS
        or not isinstance(bundle["manifest"], dict)
        or content_sha256(bundle["manifest"]) != bundle["manifest_sha256"]
    ):
        raise ValueError("animal parsing runtime bundle differs")
    manifest = AnimalParsingRuntimeManifest.from_dict(bundle["manifest"])
    if manifest.manifest_sha256 != bundle["manifest_sha256"]:
        raise ValueError("animal parsing runtime manifest digest differs")
    root = repository_root.resolve(strict=True)
    for binding in manifest.source_files:
        source = root.joinpath(*binding.relative_path.split("/"))
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != binding.byte_size
            or _sha256(source) != binding.sha256
        ):
            raise ValueError(
                f"frozen animal parsing source differs: {binding.relative_path}"
            )
    return manifest, document.raw_sha256


def _require_external_output(output_directory: Path, *, repository_root: Path) -> Path:
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(output_directory)
    return admit_new_external_output(
        output_directory,
        repository_root=repository_root,
        repository_error="parsed-body artifacts must remain outside Git",
        overwrite_error=output_directory,
    )


def _validate_archive_topology(
    samples: Sequence[UnifiedCanidSample], source_archive: zipfile.ZipFile
) -> None:
    observed = {
        Path(*Path(sample.image_path).parts[1:]).as_posix() for sample in samples
    }
    archive_images = [
        info.filename
        for info in source_archive.infolist()
        if not info.is_dir()
        and info.filename.startswith(("YT-BB-Dog/train/", "YT-BB-Dog/test/"))
        and Path(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if len(archive_images) != len(set(archive_images)):
        raise ValueError("YT-BB-Dog admitted archive repeats an image member")
    if observed != set(archive_images):
        raise ValueError(
            "YT-BB-Dog extracted image topology differs from admitted archive"
        )


def _source_file_hashes(repository_root: Path) -> dict[str, str]:
    return {
        relative: _sha256(repository_root / relative)
        for relative in (
            "shared/contracts/animal_parsing_runtime.py",
            "shared/contracts/dinov2_contract.py",
            "shared/contracts/foreground_segmentation_model.py",
            "shared/contracts/instance_segmentation_model.py",
            "data/adapters.py",
            "data/source_lock.py",
            "data/types.py",
            "evaluation/search_metrics/metrics.py",
            "shared/foundation/protected_io.py",
            "shared/foundation/protected_publication.py",
            "shared/foundation/provenance.py",
            "enrollment/registry/generated_identity_registry.py",
            "identification/export/appearance/evidencer.py",
            "parsing/export/segmentation/animal_instance_segmentation.py",
            "parsing/export/segmentation/animal_parsing.py",
            "parsing/export/segmentation/foreground_segmentation.py",
            "evaluation/parsed_body.py",
        )
    }


def _read_source_image(
    sample: UnifiedCanidSample,
    *,
    dataset_root: Path,
    source_archive: zipfile.ZipFile,
) -> Image.Image:
    source_path = dataset_root.joinpath(*Path(sample.image_path).parts)
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("YT-BB-Dog source image must be a regular non-symlink file")
    payload = source_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sample.image_sha256:
        raise ValueError("YT-BB-Dog source image SHA-256 differs")
    relative_parts = Path(sample.image_path).parts
    if not relative_parts or relative_parts[0] != "YT-BB-dog":
        raise ValueError("YT-BB-Dog extracted image path differs")
    archive_member = Path(*relative_parts[1:]).as_posix()
    try:
        archived_payload = source_archive.read(archive_member)
    except KeyError as exc:
        raise ValueError("YT-BB-Dog source image is absent from admitted archive") from exc
    if archived_payload != payload:
        raise ValueError("YT-BB-Dog extracted image differs from admitted archive")
    with Image.open(io.BytesIO(payload)) as opened:
        image = opened.convert("RGB")
        image.load()
    if image.size != (sample.width, sample.height):
        raise ValueError("YT-BB-Dog source image dimensions differ")
    return image


def _build_runtime(args: argparse.Namespace) -> tuple[AnimalParsingRuntime, dict[str, Any]]:
    repository_root = Path(__file__).resolve().parents[1]
    parsing_manifest, parsing_bundle_sha256 = _load_parsing_contract(
        args.parsing_runtime_manifest, repository_root=repository_root
    )
    foreground = ForegroundSegmentationArtifact.load(
        model_directory=args.foreground_model_dir,
        manifest_bundle_path=args.foreground_model_manifest,
    )
    instance = InstanceSegmentationArtifact.load(
        model_directory=args.instance_model_dir,
        manifest_bundle_path=args.instance_model_manifest,
    )
    if (
        foreground.manifest.manifest_sha256
        != parsing_manifest.foreground_model_manifest_sha256
        or foreground.bundle_sha256
        != parsing_manifest.foreground_model_bundle_raw_sha256
    ):
        raise ValueError("foreground model differs from frozen parser")
    if (
        instance.manifest.manifest_sha256
        != parsing_manifest.instance_model_manifest_sha256
        or instance.bundle_sha256 != parsing_manifest.instance_model_bundle_raw_sha256
    ):
        raise ValueError("instance model differs from frozen parser")
    policy = AnimalParsingPolicy.from_dict(parsing_manifest.policy)
    if policy.policy_sha256 != parsing_manifest.policy_sha256:
        raise ValueError("animal parsing policy differs from frozen parser")
    runtime = AnimalParsingRuntime(
        instance_runtime=AnimalInstanceSegmentationRuntime(
            artifact=instance,
            device=args.device,
            mask_threshold=policy.foreground_threshold,
        ),
        foreground_runtime=ForegroundSegmentationRuntime(
            artifact=foreground,
            device=args.device,
            threshold=policy.foreground_threshold,
        ),
        policy=policy,
    )
    return runtime, {
        "animal_parsing_runtime_manifest_sha256": parsing_manifest.manifest_sha256,
        "animal_parsing_runtime_bundle_raw_sha256": parsing_bundle_sha256,
        "foreground_model_manifest_sha256": foreground.manifest.manifest_sha256,
        "instance_model_manifest_sha256": instance.manifest.manifest_sha256,
    }


def _extract_embeddings(
    evidencer: ReceiptBoundDinov2Small,
    frames: Sequence[_AcceptedFrame],
    view: str,
    *,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    result = np.empty((len(frames), 384), dtype=np.float32)
    started = time.perf_counter()
    for offset in range(0, len(frames), batch_size):
        batch = frames[offset : offset + batch_size]
        images = [_view_image(frame, view) for frame in batch]
        result[offset : offset + len(batch)] = evidencer.extract_batch(images)
    return result, time.perf_counter() - started


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    output_target = _require_external_output(
        args.output_directory, repository_root=repository_root
    )
    output_parent = output_target.parent
    code_sources = _source_file_hashes(repository_root)
    source_record = get_record("yt-bb-dog")
    expected_archive_sha256 = source_record.sha256_checksums["YT-BB-Dog.zip"]
    if (
        args.source_archive.is_symlink()
        or not args.source_archive.is_file()
        or _sha256(args.source_archive) != expected_archive_sha256
    ):
        raise ValueError("YT-BB-Dog source archive differs from admitted SHA-256")
    samples = adapt_yt_bb_dog(args.dataset_root)
    tracks = _candidate_tracks(
        samples,
        candidate_limit=args.maximum_identities * args.candidate_multiplier,
        frames_per_identity=args.frames_per_identity,
    )
    if not tracks:
        raise RuntimeError("YT-BB-Dog train split has no eligible video tracks")
    parsing_runtime, parser_provenance = _build_runtime(args)
    root = args.dataset_root.resolve(strict=True)
    exclusion_counts: Counter[str] = Counter()
    track_evidence_counts = Counter(
        sample.raw_identity_id
        for sample in samples
        if sample.split_role == "train" and sample.raw_identity_id is not None
    )
    generated_records: list[GeneratedIdentityRecord] = []
    frames_by_identity: list[tuple[str, list[_AcceptedFrame]]] = []
    parsed_frames = 0
    tracks_inspected = 0
    with TemporaryDirectory(prefix=".parsed-body-reid-", dir=output_parent) as temporary:
        staging = Path(temporary) / "diagnostic"
        staging.mkdir(mode=0o700)
        for view in (*_VIEWS, "mask"):
            (staging / view).mkdir(mode=0o700)
        artifact_rows: list[dict[str, Any]] = []
        with zipfile.ZipFile(args.source_archive, mode="r") as source_archive:
            _validate_archive_topology(samples, source_archive)
            for raw_identity_id, track_samples in tracks:
                tracks_inspected += 1
                provisional = create_provisional_identity(
                    GENERATOR_ID,
                    f"yt-bb-dog\0publisher-v1-2025-10-27\0train\0{raw_identity_id}",
                    evidence_count=track_evidence_counts[raw_identity_id],
                )
                accepted: list[_AcceptedFrame] = []
                for sample in track_samples:
                    source = _read_source_image(
                        sample, dataset_root=root, source_archive=source_archive
                    )
                    prediction = parsing_runtime.predict(source)
                    parsed_frames += 1
                    instance, exclusion = _identity_instance(prediction.instances)
                    if instance is None:
                        exclusion_counts[exclusion or "UNKNOWN"] += 1
                        continue
                    crop = materialize_identity_crop(source, instance)
                    quality = {
                        "state": instance.quality.state,
                        "semantic_shape_iou": instance.quality.semantic_shape_iou,
                        "ownership_retention": instance.quality.ownership_retention,
                        "foreground_pixels": instance.quality.foreground_pixels,
                        "component_count": instance.quality.component_count,
                    }
                    accepted.append(
                        _AcceptedFrame(
                            sample, provisional.generated_identity_id, crop, quality
                        )
                    )
                if len(accepted) < 2:
                    exclusion_counts["TRACK_FEWER_THAN_TWO_USABLE_FRAMES"] += 1
                    continue
                for accepted_frame in accepted:
                    sample = accepted_frame.sample
                    crop = accepted_frame.crop
                    row: dict[str, Any] = {
                        "sample_id": sample.sample_id,
                        "source_image_path": sample.image_path,
                        "source_image_sha256": sample.image_sha256,
                        "generated_identity_id": provisional.generated_identity_id,
                        "source_box_xyxy": list(crop.source_box_xyxy),
                        "parsing_quality": accepted_frame.quality,
                        "artifacts": {},
                    }
                    for view, image in {
                        "masked_rgb": crop.masked_rgb,
                        "box_rgb": crop.box_rgb,
                        "background_plus_silhouette": (
                            _background_plus_silhouette(crop)
                        ),
                        "mask": crop.mask,
                    }.items():
                        relative = Path(view) / f"{sample.sample_id}.png"
                        target = staging / relative
                        image.save(target, format="PNG", optimize=False)
                        row["artifacts"][view] = {
                            "relative_path": relative.as_posix(),
                            "sha256": _sha256(target),
                        }
                    artifact_rows.append(row)
                frames_by_identity.append((provisional.generated_identity_id, accepted))
                generated_records.append(provisional)
                if len(frames_by_identity) == args.maximum_identities:
                    break
        if len(frames_by_identity) < 2:
            raise RuntimeError(
                "parsed-body diagnostic requires at least two identities with two "
                "usable frames; "
                f"tracks_inspected={tracks_inspected}, parsed_frames={parsed_frames}, "
                f"exclusions={dict(sorted(exclusion_counts.items()))}"
            )
        flattened = [
            frame for _, identity_frames in frames_by_identity for frame in identity_frames
        ]
        del parsing_runtime
        if args.device == "cuda":
            import torch

            torch.cuda.empty_cache()
        evidencer = ReceiptBoundDinov2Small(
            model_directory=args.dinov2_model_dir,
            weight_intake_bundle=args.dinov2_weight_intake,
            preprocessor_intake_bundle=args.dinov2_preprocessor_intake,
            device=args.device,
            max_batch_size=args.batch_size,
        )
        view_results: dict[str, Any] = {}
        for view in _VIEWS:
            embeddings, elapsed = _extract_embeddings(
                evidencer, flattened, view, batch_size=args.batch_size
            )
            result = _evaluate_view(
                frames_by_identity,
                embeddings,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
            result["embedding_seconds"] = elapsed
            result["images_per_second"] = len(flattened) / elapsed
            view_results[view] = result
        generated_registry = GeneratedIdentityRegistry(tuple(generated_records)).to_dict()
        if _source_file_hashes(repository_root) != code_sources:
            raise RuntimeError("parsed-body workflow source changed during execution")
        report = {
            "schema_version": REPORT_SCHEMA,
            "interpretation": INTERPRETATION,
            "dataset": {
                "canonical_name": "yt-bb-dog",
                "version": "publisher-v1-2025-10-27",
                "source_archive_sha256": expected_archive_sha256,
                "publisher_split": "train",
                "identity_semantics": "provisional generated video-track proxy",
                "registered_identity_claim": False,
            },
            "protocol": {
                "name": "one-earliest-selected-usable-gallery-remaining-selected-query.v1",
                "selection": "deterministic lexicographic tracks and evenly spaced frames",
                "frames_per_identity_cap": args.frames_per_identity,
                "required_parsing_quality": "USABLE",
                "required_detected_dog_instances": 1,
                "embedding": "frozen DINOv2-small pooler, 384D L2-normalized",
                "scoring": "exact cosine",
                "identity_aggregation": "one gallery template per track",
                "views": list(_VIEWS),
            },
            "cohort": {
                "maximum_identities": args.maximum_identities,
                "candidate_multiplier": args.candidate_multiplier,
                "candidate_track_limit": len(tracks),
                "candidate_tracks_inspected": tracks_inspected,
                "parsed_frames": parsed_frames,
                "accepted_identities": len(frames_by_identity),
                "accepted_frames": len(flattened),
                "exclusions": dict(sorted(exclusion_counts.items())),
            },
            "generated_identity_registry": generated_registry,
            "generated_identity_registry_sha256": content_sha256(generated_registry),
            "artifacts": artifact_rows,
            "results": view_results,
            "provenance": {
                **parser_provenance,
                **evidencer.gallery_contract_fields,
                "config_sha256": evidencer.config_sha256,
                "source_files": code_sources,
                "source_files_sha256": content_sha256(code_sources),
                "dependency_lock_sha256": _sha256(repository_root / "uv.lock"),
                "code_commit": subprocess.check_output(
                    ("git", "rev-parse", "HEAD"), cwd=repository_root, text=True
                ).strip(),
                "worktree_dirty": bool(
                    subprocess.check_output(
                        ("git", "status", "--porcelain"),
                        cwd=repository_root,
                        text=True,
                    ).strip()
                ),
                "device": args.device,
                "batch_size": args.batch_size,
            },
        }
        write_private_json_bundle(((staging / "report.json", report),))
        for path in (
            staging / "masked_rgb",
            staging / "box_rgb",
            staging / "background_plus_silhouette",
            staging / "mask",
            staging,
        ):
            fsync_directory(path)
        rename_directory_noreplace(staging, output_target)
        fsync_directory(output_parent)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--parsing-runtime-manifest", required=True, type=Path)
    parser.add_argument("--foreground-model-dir", required=True, type=Path)
    parser.add_argument("--foreground-model-manifest", required=True, type=Path)
    parser.add_argument("--instance-model-dir", required=True, type=Path)
    parser.add_argument("--instance-model-manifest", required=True, type=Path)
    parser.add_argument("--dinov2-model-dir", required=True, type=Path)
    parser.add_argument("--dinov2-weight-intake", required=True, type=Path)
    parser.add_argument("--dinov2-preprocessor-intake", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument(
        "--maximum-identities",
        type=int,
        default=32,
        help="maximum accepted identities; the strict cohort may underfill",
    )
    parser.add_argument("--frames-per-identity", type=int, default=4)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.maximum_identities < 2:
        parser.error("--maximum-identities must be at least two")
    if args.frames_per_identity < 2:
        parser.error("--frames-per-identity must be at least two")
    if args.candidate_multiplier <= 0 or args.batch_size <= 0:
        parser.error("candidate multiplier and batch size must be positive")
    if args.bootstrap_resamples <= 0 or args.bootstrap_seed < 0:
        parser.error("bootstrap configuration differs")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "status": "CREATED_PARSED_BODY_REID_DIAGNOSTIC",
                "output": str(args.output_directory),
                "identities": report["cohort"]["accepted_identities"],
                "frames": report["cohort"]["accepted_frames"],
                "masked_rank1": report["results"]["masked_rgb"]["Rank-1"],
                "box_rank1": report["results"]["box_rgb"]["Rank-1"],
                "background_plus_silhouette_rank1": report["results"]
                ["background_plus_silhouette"]["Rank-1"],
                "report_sha256": content_sha256(report),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
