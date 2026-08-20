"""Run an unassisted multi-instance animal parsing panel on AP-10K images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from shared.contracts.foreground_segmentation_model import (
    ForegroundSegmentationArtifact,
)
from shared.contracts.instance_segmentation_model import (
    InstanceSegmentationArtifact,
)
from data.acquisition import sha256_file
from data.adapters import adapt_ap10k_dog
from data.types import UnifiedCanidSample
from shared.foundation.protected_io import json_document_bytes
from shared.foundation.protected_publication import (
    fsync_directory,
    rename_directory_noreplace,
)
from shared.foundation.provenance import content_sha256
from parsing.export.segmentation.animal_instance_segmentation import (
    AnimalInstanceSegmentationRuntime,
)
from parsing.export.segmentation.animal_parsing import (
    AnimalParsingPolicy,
    AnimalParsingRuntime,
    ParsedAnimalInstance,
    materialize_identity_crop,
)
from parsing.export.segmentation.foreground_segmentation import ForegroundSegmentationRuntime

REPORT_SCHEMA = "cvi.animal_parsing_panel.v1"
INTERPRETATION = (
    "UNASSISTED_MULTI_INSTANCE_VISIBLE_ANIMAL_CANDIDATES_NOT_HUMAN_MASK_VERIFIED"
)
_TILE_WIDTH = 420
_TILE_HEIGHT = 300
_COLORS = (
    (46, 204, 113),
    (241, 196, 15),
    (52, 152, 219),
    (231, 76, 60),
    (155, 89, 182),
    (26, 188, 156),
)


@dataclass(frozen=True, slots=True)
class AP10KSourceGroup:
    source_group_id: str
    annotations: tuple[UnifiedCanidSample, ...]

    @property
    def source(self) -> UnifiedCanidSample:
        return self.annotations[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--instance-model-directory", type=Path, required=True)
    parser.add_argument("--instance-model-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--multi-source-count", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    report = run_panel(
        dataset_root=args.dataset_root,
        model_directory=args.model_directory,
        model_manifest=args.model_manifest,
        instance_model_directory=args.instance_model_directory,
        instance_model_manifest=args.instance_model_manifest,
        output_directory=args.output_directory,
        sample_count=args.sample_count,
        multi_source_count=args.multi_source_count,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": "CREATED_ANIMAL_PARSING_PANEL",
                "output": str(args.output_directory),
                "source_count": len(report["records"]),
                "predicted_instance_count": sum(
                    len(record["predictions"]) for record in report["records"]
                ),
                "report_sha256": content_sha256(report),
            },
            sort_keys=True,
        )
    )
    return 0


def run_panel(
    *,
    dataset_root: Path,
    model_directory: Path,
    model_manifest: Path,
    instance_model_directory: Path,
    instance_model_manifest: Path,
    output_directory: Path,
    sample_count: int,
    multi_source_count: int,
    threshold: float,
    device: str,
    batch_size: int = 4,
) -> dict[str, Any]:
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(output_directory)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("animal parsing panel batch size must be positive")
    output_parent = output_directory.parent.resolve(strict=True)
    groups = _select_ap10k_source_groups(
        adapt_ap10k_dog(dataset_root),
        sample_count=sample_count,
        multi_source_count=multi_source_count,
    )
    foreground_artifact = ForegroundSegmentationArtifact.load(
        model_directory=model_directory,
        manifest_bundle_path=model_manifest,
    )
    instance_artifact = InstanceSegmentationArtifact.load(
        model_directory=instance_model_directory,
        manifest_bundle_path=instance_model_manifest,
    )
    policy = AnimalParsingPolicy(foreground_threshold=threshold)
    runtime = AnimalParsingRuntime(
        instance_runtime=AnimalInstanceSegmentationRuntime(
            artifact=instance_artifact,
            device=device,
            mask_threshold=threshold,
        ),
        foreground_runtime=ForegroundSegmentationRuntime(
            artifact=foreground_artifact,
            device=device,
            threshold=threshold,
        ),
        policy=policy,
    )
    root = dataset_root.resolve(strict=True)
    with TemporaryDirectory(prefix=".animal-parsing-panel-", dir=output_parent) as tmp:
        staging = Path(tmp) / "panel"
        staging.mkdir(mode=0o700)
        for name in ("masks", "masked_crops", "overlays"):
            (staging / name).mkdir(mode=0o700)
        records: list[dict[str, Any]] = []
        rows: list[tuple[Image.Image, Image.Image, Image.Image, str]] = []
        loaded = []
        for group in groups:
            sample = group.source
            source_path = root.joinpath(*Path(sample.image_path).parts)
            if sha256_file(source_path) != sample.image_sha256:
                raise ValueError("animal parsing panel source SHA-256 differs")
            with Image.open(source_path) as opened:
                source = opened.convert("RGB")
            if source.size != (sample.width, sample.height):
                raise ValueError("animal parsing panel source dimensions differ")
            loaded.append((group, source))
        parsed_values = []
        for start in range(0, len(loaded), batch_size):
            chunk = loaded[start : start + batch_size]
            parsed_values.extend(
                runtime.predict_batch(
                    tuple(source for _group, source in chunk),
                    instance_batch_size=batch_size,
                    foreground_batch_size=batch_size,
                )
            )
        for (group, source), parsed in zip(loaded, parsed_values, strict=True):
            sample = group.source
            matches = _match_predictions_to_annotations(
                parsed.instances, group.annotations, minimum_box_iou=0.5
            )
            token = hashlib.sha256(
                b"cvi.animal_parsing_panel.source.v1\0"
                + group.source_group_id.encode("utf-8")
            ).hexdigest()[:20]
            overlay = _instance_overlay(source, parsed.instances)
            overlay_path = staging / "overlays" / f"{token}.png"
            overlay.save(overlay_path, format="PNG", optimize=False)
            combined = _combined_masked_appearance(source, parsed.instances)
            rows.append(
                (
                    _draw_ground_truth_boxes(source, group.annotations),
                    overlay,
                    combined,
                    f"{token}  GT={len(group.annotations)} P={len(parsed.instances)}",
                )
            )
            prediction_rows: list[dict[str, Any]] = []
            for instance in parsed.instances:
                stem = f"{token}-{instance.instance_index:02d}"
                mask_path = staging / "masks" / f"{stem}.png"
                Image.fromarray(instance.hard_mask * 255, mode="L").save(
                    mask_path, format="PNG", optimize=False
                )
                crop = materialize_identity_crop(source, instance, require_usable=False)
                crop_path = staging / "masked_crops" / f"{stem}.png"
                crop.masked_rgb.save(crop_path, format="PNG", optimize=False)
                prediction_rows.append(
                    {
                        "instance_index": instance.instance_index,
                        "query_index": instance.query_index,
                        "class_id": instance.class_id,
                        "class_name": instance.class_name,
                        "class_score": instance.class_score,
                        "detector_box_xyxy": list(instance.detector_box_xyxy),
                        "refinement_box_xyxy": list(instance.refinement_box_xyxy),
                        "mask_box_xyxy": (
                            list(instance.mask_box_xyxy)
                            if instance.mask_box_xyxy is not None
                            else None
                        ),
                        "quality": {
                            "state": instance.quality.state,
                            "reasons": list(instance.quality.reasons),
                            "flags": list(instance.quality.flags),
                            "semantic_shape_iou": (instance.quality.semantic_shape_iou),
                            "ownership_retention": (
                                instance.quality.ownership_retention
                            ),
                            "foreground_pixels": (instance.quality.foreground_pixels),
                            "component_count": instance.quality.component_count,
                            "touches_source_border": (
                                instance.quality.touches_source_border
                            ),
                            "automatic_identity_input_eligible": (
                                instance.quality.state == "USABLE"
                            ),
                        },
                        "mask": _file_binding(mask_path, staging),
                        "masked_identity_crop": _file_binding(crop_path, staging),
                    }
                )
            records.append(
                {
                    "source_group_id": group.source_group_id,
                    "source_image_path": sample.image_path,
                    "source_image_sha256": sample.image_sha256,
                    "source_width": sample.width,
                    "source_height": sample.height,
                    "annotation_assistance_during_inference": False,
                    "ground_truth_annotations": [
                        {
                            "sample_id": annotation.sample_id,
                            "annotation_id": annotation.metadata.get("annotation_id"),
                            "box_xyxy": list(annotation.dog_boxes_xyxy or ()),
                        }
                        for annotation in group.annotations
                    ],
                    "predictions": prediction_rows,
                    "posthoc_box_matches": matches,
                    "false_negative_sample_ids": [
                        annotation.sample_id
                        for annotation in group.annotations
                        if annotation.sample_id
                        not in {match["sample_id"] for match in matches}
                    ],
                    "overlay": _file_binding(overlay_path, staging),
                }
            )
        contact_sheet_path = staging / "contact_sheet.png"
        _contact_sheet(rows).save(contact_sheet_path, format="PNG", optimize=False)
        report = {
            "schema_version": REPORT_SCHEMA,
            "interpretation": INTERPRETATION,
            "ontology": parsed.ontology,
            "ontology_description": parsed.ontology_description,
            "parsing_policy": policy.to_dict(),
            "parsing_policy_sha256": policy.policy_sha256,
            "foreground_model": _artifact_binding(foreground_artifact),
            "instance_model": _artifact_binding(instance_artifact),
            "dataset": {
                "name": "ap10k-dog",
                "version": "official-split1-2021-11-01",
                "split": "test",
                "annotation_use": "POST_INFERENCE_BOX_DIAGNOSTICS_ONLY",
                "pixel_ground_truth_available": False,
            },
            "selection": {
                "algorithm": "SHA256_SOURCE_GROUP_STRATIFIED_MULTI_DOG_V1",
                "requested_source_count": sample_count,
                "requested_multi_dog_source_count": multi_source_count,
                "inference_batch_size": batch_size,
            },
            "records": records,
            "contact_sheet": _file_binding(contact_sheet_path, staging),
        }
        _write_regular_file(staging / "report.json", json_document_bytes(report))
        for directory in (
            staging / "masks",
            staging / "masked_crops",
            staging / "overlays",
            staging,
        ):
            fsync_directory(directory)
        rename_directory_noreplace(staging, output_parent / output_directory.name)
    fsync_directory(output_parent / output_directory.name)
    fsync_directory(output_parent)
    return report


def _select_ap10k_source_groups(
    samples: Iterable[UnifiedCanidSample],
    *,
    sample_count: int,
    multi_source_count: int,
) -> tuple[AP10KSourceGroup, ...]:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or isinstance(multi_source_count, bool)
        or not isinstance(multi_source_count, int)
        or not 0 <= multi_source_count <= sample_count
    ):
        raise ValueError("animal parsing panel selection counts differ")
    grouped: dict[str, list[UnifiedCanidSample]] = defaultdict(list)
    for sample in samples:
        if sample.split_role == "test" and sample.dog_boxes_xyxy is not None:
            grouped[sample.source_group_id].append(sample)
    groups: list[AP10KSourceGroup] = []
    for source_group_id, values in grouped.items():
        values.sort(
            key=lambda item: (
                int(item.metadata.get("annotation_id", -1)),
                item.sample_id,
            )
        )
        reference = values[0]
        if any(
            (
                item.image_path,
                item.image_sha256,
                item.width,
                item.height,
                item.metadata.get("image_id"),
            )
            != (
                reference.image_path,
                reference.image_sha256,
                reference.width,
                reference.height,
                reference.metadata.get("image_id"),
            )
            for item in values[1:]
        ):
            raise ValueError("AP-10K source group image contracts differ")
        groups.append(AP10KSourceGroup(source_group_id, tuple(values)))
    groups.sort(key=_group_order)
    multi = [group for group in groups if len(group.annotations) > 1]
    if len(groups) < sample_count or len(multi) < multi_source_count:
        raise ValueError("insufficient AP-10K source groups for parsing panel")
    selected = multi[:multi_source_count]
    selected_ids = {group.source_group_id for group in selected}
    selected.extend(
        group
        for group in groups
        if group.source_group_id not in selected_ids and len(selected) < sample_count
    )
    return tuple(sorted(selected, key=_group_order))


def _group_order(group: AP10KSourceGroup) -> tuple[bytes, str]:
    return (
        hashlib.sha256(
            b"cvi.animal_parsing_panel.ap10k.v1\0"
            + group.source_group_id.encode("utf-8")
        ).digest(),
        group.source_group_id,
    )


def _match_predictions_to_annotations(
    predictions: tuple[ParsedAnimalInstance, ...],
    annotations: tuple[UnifiedCanidSample, ...],
    *,
    minimum_box_iou: float,
) -> list[dict[str, Any]]:
    if not 0.0 < minimum_box_iou < 1.0:
        raise ValueError("animal parsing panel match IoU differs")
    ordered = sorted(
        predictions,
        key=lambda item: (-item.class_score, item.query_index, item.instance_index),
    )
    unmatched = set(range(len(annotations)))
    matches: list[dict[str, Any]] = []
    for prediction in ordered:
        candidates = [
            (
                _box_iou(
                    prediction.detector_box_xyxy,
                    annotations[index].dog_boxes_xyxy or (),
                ),
                index,
            )
            for index in unmatched
        ]
        if not candidates:
            continue
        iou, selected = max(
            candidates,
            key=lambda item: (
                item[0],
                -int(annotations[item[1]].metadata.get("annotation_id", -1)),
                annotations[item[1]].sample_id,
            ),
        )
        if iou < minimum_box_iou:
            continue
        unmatched.remove(selected)
        matches.append(
            {
                "instance_index": prediction.instance_index,
                "sample_id": annotations[selected].sample_id,
                "annotation_id": annotations[selected].metadata.get("annotation_id"),
                "box_iou": iou,
            }
        )
    matches.sort(key=lambda item: item["instance_index"])
    return matches


def _box_iou(
    first: tuple[int, int, int, int] | tuple[float, float, float, float],
    second: tuple[int, int, int, int] | tuple[float, float, float, float] | tuple[()],
) -> float:
    if len(first) != 4 or len(second) != 4:
        raise ValueError("animal parsing panel box differs")
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _draw_ground_truth_boxes(
    source: Image.Image, annotations: tuple[UnifiedCanidSample, ...]
) -> Image.Image:
    result = source.copy()
    draw = ImageDraw.Draw(result)
    for index, annotation in enumerate(annotations):
        if annotation.dog_boxes_xyxy is None:
            continue
        draw.rectangle(annotation.dog_boxes_xyxy, outline=(0, 120, 255), width=4)
        draw.text(
            (annotation.dog_boxes_xyxy[0] + 4, annotation.dog_boxes_xyxy[1] + 4),
            f"GT {index}",
            fill=(0, 70, 210),
        )
    return result


def _instance_overlay(
    source: Image.Image, instances: tuple[ParsedAnimalInstance, ...]
) -> Image.Image:
    values = np.asarray(source, dtype=np.uint8).copy()
    for instance in instances:
        color = np.asarray(
            _COLORS[instance.instance_index % len(_COLORS)], dtype=np.float32
        )
        foreground = instance.hard_mask.astype(bool)
        values[foreground] = np.rint(
            values[foreground].astype(np.float32) * 0.5 + color * 0.5
        ).astype(np.uint8)
    result = Image.fromarray(values, mode="RGB")
    draw = ImageDraw.Draw(result)
    for instance in instances:
        color = _COLORS[instance.instance_index % len(_COLORS)]
        draw.rectangle(instance.detector_box_xyxy, outline=color, width=4)
        draw.text(
            (instance.detector_box_xyxy[0] + 4, instance.detector_box_xyxy[1] + 4),
            f"P{instance.instance_index} {instance.quality.state}",
            fill=color,
        )
    return result


def _combined_masked_appearance(
    source: Image.Image, instances: tuple[ParsedAnimalInstance, ...]
) -> Image.Image:
    source_values = np.asarray(source, dtype=np.uint8)
    result = np.full_like(source_values, 127)
    if instances:
        mask = np.logical_or.reduce([item.hard_mask.astype(bool) for item in instances])
        result[mask] = source_values[mask]
    return Image.fromarray(result, mode="RGB")


def _contact_sheet(
    rows: list[tuple[Image.Image, Image.Image, Image.Image, str]],
) -> Image.Image:
    if not rows:
        raise ValueError("animal parsing panel cannot render an empty contact sheet")
    header = 44
    row_height = _TILE_HEIGHT + 28
    sheet = Image.new(
        "RGB", (_TILE_WIDTH * 3, header + row_height * len(rows)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, label in enumerate(
        ("SOURCE + AP10K GT BOXES", "UNASSISTED PARSED INSTANCES", "MASKED APPEARANCE")
    ):
        draw.text((index * _TILE_WIDTH + 8, 14), label, fill="black")
    for row_index, (*images, label) in enumerate(rows):
        top = header + row_index * row_height
        for column, image in enumerate(images):
            sheet.paste(_contain(image), (column * _TILE_WIDTH, top))
        draw.text((8, top + _TILE_HEIGHT + 7), label, fill="black")
    return sheet


def _contain(image: Image.Image) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail((_TILE_WIDTH, _TILE_HEIGHT), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (_TILE_WIDTH, _TILE_HEIGHT), (238, 238, 238))
    tile.paste(
        copy, ((_TILE_WIDTH - copy.width) // 2, (_TILE_HEIGHT - copy.height) // 2)
    )
    return tile


def _artifact_binding(artifact: Any) -> dict[str, Any]:
    return {
        "model_id": artifact.manifest.model_id,
        "source_revision": artifact.manifest.source_revision,
        "manifest_sha256": artifact.manifest.manifest_sha256,
        "bundle_raw_sha256": artifact.bundle_sha256,
    }


def _file_binding(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def _write_regular_file(path: Path, payload: bytes) -> None:
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
                raise OSError("animal parsing panel write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
