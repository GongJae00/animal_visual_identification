"""Render deterministic parser-failure contact sheets for human review."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.source_lock import get_record
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file
from data.full_segment.route_plan import validate_full128_route_plan_bundle
from parsing.full_segment.full_segment_cache import (
    thaw_animal_parsing_prediction,
    validate_frozen_animal_parsing,
)

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_REASONS = (
    "NO_PARSED_DOG_INSTANCE",
    "PARSER_INSTANCE_CARDINALITY_AMBIGUOUS",
    "PARSER_DISTINCT_DOG_CARDINALITY_AMBIGUOUS",
    "NO_VALID_PARSED_DOG_INSTANCE",
    "SELECTED_DOG_PARSING_UNUSABLE",
    "AP10K_GLOBAL_BBOX_ASSOCIATION_AMBIGUOUS",
)
_LIMITS = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}
_TILE_SIZE = (300, 230)
_COLUMNS = 4
_ROWS = 6
_PAGE_SIZE = _COLUMNS * _ROWS


def render_parser_failure_review(
    *,
    route_plan: Path,
    materialization_root: Path,
    output_dir: Path,
    samples_per_reason: int = 48,
) -> dict[str, Any]:
    """Render a balanced, token-hash-selected review set without rerunning models."""

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite parser review: {output_dir}")
    if isinstance(samples_per_reason, bool) or not isinstance(samples_per_reason, int):
        raise TypeError("samples_per_reason must be an integer")
    if samples_per_reason <= 0:
        raise ValueError("samples_per_reason must be positive")
    plan_bundle = validate_full128_route_plan_bundle(
        read_strict_json_document(route_plan, **_LIMITS).payload,
        verify_files=False,
    )
    records = plan_bundle["plan"]["records"]
    if materialization_root.is_symlink():
        raise ValueError("materialization root must not be a symlink")
    root = materialization_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("materialization root must be a regular directory")
    rows_by_token = {row["sample_token"]: row for row in records}
    if len(rows_by_token) != len(records):
        raise ValueError("route plan repeats sample tokens")

    candidates: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    dataset_reason_counts: Counter[tuple[str, str]] = Counter()
    samples_root = root / "samples"
    for sample_dir in sorted(samples_root.iterdir(), key=lambda path: path.name):
        receipt_path = sample_dir / "execution-receipt.json"
        if not receipt_path.is_file() or receipt_path.is_symlink():
            continue
        receipt = read_strict_json_document(receipt_path, maximum_bytes=262_144).payload
        reason = receipt.get("terminal_reason")
        if reason not in _REASONS:
            continue
        token = receipt.get("sample_token")
        route_row = rows_by_token.get(token)
        if route_row is None or receipt.get("plan_record_sha256") != route_row.get(
            "record_sha256"
        ):
            raise ValueError("parser review receipt and route-plan binding differ")
        candidates[reason].append(route_row)
        reason_counts[reason] += 1
        dataset_reason_counts[(reason, route_row["dataset_name"])] += 1

    selected: dict[str, list[Mapping[str, Any]]] = {}
    observed_reasons = tuple(reason for reason in _REASONS if candidates[reason])
    if not observed_reasons:
        raise ValueError("parser review has no supported failure candidates")
    for reason in observed_reasons:
        available = candidates[reason]
        selected[reason] = _balanced_select(
            available, min(samples_per_reason, len(available)), reason
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        selected_manifest: list[dict[str, Any]] = []
        pages: list[dict[str, Any]] = []
        for reason in observed_reasons:
            reason_rows = selected[reason]
            for page_index, start in enumerate(
                range(0, len(reason_rows), _PAGE_SIZE), start=1
            ):
                page_rows = reason_rows[start : start + _PAGE_SIZE]
                rendered = [
                    _load_review_sample(
                        row,
                        root,
                        parser_policy_sha256=plan_bundle["parser_policy_sha256"],
                    )
                    for row in page_rows
                ]
                filename = f"{_reason_slug(reason)}_{page_index:02d}.png"
                image_path = staging / filename
                _render_page(image_path, reason, page_index, rendered)
                image_bytes = image_path.read_bytes()
                pages.append(
                    {
                        "relative_path": filename,
                        "reason": reason,
                        "page_index": page_index,
                        "sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "byte_size": len(image_bytes),
                        "sample_tokens": [
                            item["row"]["sample_token"] for item in rendered
                        ],
                    }
                )
                selected_manifest.extend(
                    _manifest_row(reason, item) for item in rendered
                )
        body = {
            "schema_version": "cvi.parser_failure_visual_review.v1",
            "source_route_plan_sha256": plan_bundle["plan_sha256"],
            "source_route_plan_bundle_sha256": plan_bundle["bundle_sha256"],
            "materialization_root": str(root),
            "selection_method": (
                "BALANCED_BY_DATASET_WITH_UNIQUE_SOURCE_PREFERENCE_THEN_"
                "SHA256_OF_REASON_DATASET_AND_SAMPLE_TOKEN"
            ),
            "samples_per_reason": samples_per_reason,
            "observed_reasons": list(observed_reasons),
            "population_counts_by_reason": dict(sorted(reason_counts.items())),
            "population_counts_by_reason_and_dataset": {
                f"{reason}:{dataset}": count
                for (reason, dataset), count in sorted(dataset_reason_counts.items())
            },
            "selected": selected_manifest,
            "pages": pages,
            "interpretation": (
                "HUMAN_VISUAL_REVIEW_AID_ONLY;DETECTOR_BOXES_AND_PARSER_MASKS_ARE_"
                "MODEL_OUTPUTS_NOT_GROUND_TRUTH"
            ),
        }
        manifest = {**body, "review_sha256": content_sha256(body)}
        write_private_json_bundle(((staging / "review-manifest.json", manifest),))
        fsync_directory(staging)
        rename_directory_noreplace(staging, output_dir)
        fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _balanced_select(
    rows: Sequence[Mapping[str, Any]], count: int, reason: str
) -> list[Mapping[str, Any]]:
    by_dataset: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset_name"]].append(row)
    for dataset, values in by_dataset.items():
        values.sort(
            key=lambda row: _selection_key(reason, dataset, row["sample_token"])
        )
    selected: list[Mapping[str, Any]] = []
    deferred: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_sources: set[str] = set()
    datasets = sorted(by_dataset)
    while len(selected) < count:
        progressed = False
        for dataset in datasets:
            values = by_dataset[dataset]
            while values:
                candidate = values.pop(0)
                if candidate["source_sha256"] in seen_sources:
                    deferred[dataset].append(candidate)
                    continue
                selected.append(candidate)
                seen_sources.add(candidate["source_sha256"])
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            break
    while len(selected) < count:
        progressed = False
        for dataset in datasets:
            if deferred[dataset]:
                selected.append(deferred[dataset].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def _selection_key(reason: str, dataset: str, token: str) -> str:
    return hashlib.sha256(f"{reason}\0{dataset}\0{token}".encode("ascii")).hexdigest()


def _load_review_sample(
    row: Mapping[str, Any],
    root: Path,
    *,
    parser_policy_sha256: str,
) -> dict[str, Any]:
    dataset_root = Path(get_record(row["dataset_name"]).data_root).resolve(strict=True)
    relative = Path(row["source_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("parser review source path is unsafe")
    source_candidate = dataset_root / relative
    if source_candidate.is_symlink():
        raise ValueError("parser review source must not be a symlink")
    source_path = source_candidate.resolve(strict=True)
    if (
        not source_path.is_relative_to(dataset_root)
        or source_path.is_symlink()
        or not source_path.is_file()
    ):
        raise ValueError("parser review source is unsafe")
    retained = read_retained_regular_file(
        source_path,
        expected_sha256=row["source_sha256"],
        expected_bytes=row["source_byte_size"],
        maximum_bytes=67_108_864,
        capture_payload=True,
        subject="parser review source image",
    )
    assert retained.payload is not None
    with Image.open(io.BytesIO(retained.payload)) as opened:
        opened.load()
        source = opened.convert("RGB")
    if source.size != (row["source_width"], row["source_height"]):
        raise ValueError("parser review source dimensions differ")

    receipt_path = root / "samples" / row["sample_token"] / "execution-receipt.json"
    receipt = read_strict_json_document(receipt_path, maximum_bytes=262_144).payload
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if content_sha256(receipt_body) != receipt.get("receipt_sha256"):
        raise ValueError("parser review execution receipt digest differs")
    parser_key = receipt["parser_lineage"]["parser_cache_key"]
    if parser_key != row["parser_cache_key"]:
        raise ValueError("parser review cache key differs from route plan")
    cache_dir = root / "parser-cache" / parser_key
    cache_receipt = read_strict_json_document(
        cache_dir / "receipt.json", maximum_bytes=1_048_576
    ).payload
    cache_receipt_body = {
        key: value for key, value in cache_receipt.items() if key != "receipt_sha256"
    }
    if (
        content_sha256(cache_receipt_body) != cache_receipt.get("receipt_sha256")
        or cache_receipt.get("receipt_sha256")
        != receipt["parser_lineage"]["parser_cache_receipt_sha256"]
        or cache_receipt.get("parser_cache_key") != parser_key
        or cache_receipt.get("source_sha256") != row["source_sha256"]
        or cache_receipt.get("source_width") != row["source_width"]
        or cache_receipt.get("source_height") != row["source_height"]
        or cache_receipt.get("runtime", {}).get("parser_policy_sha256")
        != parser_policy_sha256
    ):
        raise ValueError("parser review cache receipt binding differs")
    frozen_document = read_strict_json_document(cache_dir / "frozen.json", **_LIMITS)
    frozen = validate_frozen_animal_parsing(frozen_document.payload)
    if (
        frozen_document.raw_sha256 != cache_receipt.get("frozen_json_sha256")
        or frozen["prediction_sha256"] != cache_receipt.get("prediction_sha256")
        or frozen["prediction_sha256"] != receipt["parser_lineage"]["prediction_sha256"]
    ):
        raise ValueError("parser review frozen prediction binding differs")
    prediction = thaw_animal_parsing_prediction(frozen)
    if prediction.policy_sha256 != parser_policy_sha256:
        raise ValueError("parser review policy binding differs")
    if (
        prediction.source_width != source.width
        or prediction.source_height != source.height
    ):
        raise ValueError("parser review prediction dimensions differ")
    return {
        "row": row,
        "source": source,
        "prediction": prediction,
        "receipt": receipt,
    }


def _render_page(
    path: Path, reason: str, page_index: int, samples: Sequence[Mapping[str, Any]]
) -> None:
    card_width = _TILE_SIZE[0] + 24
    card_height = _TILE_SIZE[1] + 106
    width = _COLUMNS * card_width + 24
    height = 112 + _ROWS * card_height + 24
    canvas = Image.new("RGB", (width, height), (10, 16, 28))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(_FONT_BOLD, 25)
    label_font = ImageFont.truetype(_FONT_BOLD, 14)
    small_font = ImageFont.truetype(_FONT, 12)
    draw.text((24, 18), reason, font=title_font, fill=(244, 247, 252))
    draw.text(
        (24, 55),
        f"page {page_index} | red=dog candidate, orange=other candidate, "
        "cyan=mask, green=AP10K dog annotation",
        font=small_font,
        fill=(148, 163, 184),
    )
    draw.text(
        (24, 76),
        "Review labels: TRUE_MISS / WRONG_CLASS / MULTI_DOG_USE_LARGEST / BAD_MASK / DATA_ERROR",
        font=small_font,
        fill=(251, 191, 36),
    )
    for index, item in enumerate(samples):
        row_index, column = divmod(index, _COLUMNS)
        x = 24 + column * card_width
        y = 108 + row_index * card_height
        visual = _annotated_tile(
            item["source"], item["prediction"], item["row"].get("route_evidence")
        )
        canvas.paste(visual, (x, y))
        row = item["row"]
        prediction = item["prediction"]
        dog_instances = [
            instance
            for instance in prediction.instances
            if instance.class_name == "dog"
        ]
        other_count = len(prediction.instances) - len(dog_instances)
        draw.text(
            (x, y + _TILE_SIZE[1] + 7),
            f"{row['dataset_name']} | {row['sample_token'][:12]}",
            font=label_font,
            fill=(226, 232, 240),
        )
        draw.text(
            (x, y + _TILE_SIZE[1] + 29),
            f"candidates {len(prediction.instances)} | dogs {len(dog_instances)} | "
            f"other {other_count}",
            font=small_font,
            fill=(148, 163, 184),
        )
        details = _dog_details(dog_instances)
        draw.text(
            (x, y + _TILE_SIZE[1] + 48),
            details,
            font=small_font,
            fill=(148, 163, 184),
        )
        draw.text(
            (x, y + _TILE_SIZE[1] + 68),
            "review: __________________________",
            font=small_font,
            fill=(251, 191, 36),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=9, optimize=False)


def _annotated_tile(
    source: Image.Image,
    prediction: Any,
    route_evidence: Mapping[str, Any] | None = None,
) -> Image.Image:
    contained = _contain(source, _TILE_SIZE)
    tile, offset, scale = contained
    draw = ImageDraw.Draw(tile)
    annotation_box = (
        route_evidence.get("bbox_xyxy")
        if route_evidence is not None
        and route_evidence.get("kind") == "AP10K_AUTHORITATIVE_BBOX_ASSOCIATION"
        else None
    )
    if (
        isinstance(annotation_box, list)
        and len(annotation_box) == 4
        and all(
            not isinstance(value, bool) and isinstance(value, (int, float))
            for value in annotation_box
        )
    ):
        draw.rectangle(
            _scale_box(annotation_box, offset, scale),
            outline=(74, 222, 128),
            width=3,
        )
    for instance in prediction.instances:
        color = (248, 113, 113) if instance.class_name == "dog" else (251, 146, 60)
        box = _scale_box(instance.detector_box_xyxy, offset, scale)
        draw.rectangle(box, outline=color, width=3)
        draw.text(
            (box[0] + 3, box[1] + 3),
            f"{instance.class_name} {instance.class_score:.2f}",
            font=ImageFont.truetype(_FONT_BOLD, 12),
            fill=color,
            stroke_width=2,
            stroke_fill=(10, 16, 28),
        )
        if instance.class_name == "dog":
            mask = Image.fromarray((instance.hard_mask > 0).astype(np.uint8) * 255)
            mask = mask.resize(
                (round(source.width * scale), round(source.height * scale)),
                Image.Resampling.NEAREST,
            )
            mask_canvas = Image.new("L", _TILE_SIZE, 0)
            mask_canvas.paste(mask, offset)
            outline = _mask_outline(np.asarray(mask_canvas, dtype=np.uint8) > 0)
            overlay = np.asarray(tile).copy()
            overlay[outline] = (34, 211, 238)
            tile = Image.fromarray(overlay, mode="RGB")
            draw = ImageDraw.Draw(tile)
    return tile


def _contain(
    image: Image.Image, size: tuple[int, int]
) -> tuple[Image.Image, tuple[int, int], float]:
    scale = min(size[0] / image.width, size[1] / image.height)
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.BICUBIC)
    offset = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
    tile = Image.new("RGB", size, (30, 41, 59))
    tile.paste(resized, offset)
    return tile, offset, scale


def _scale_box(
    box: Sequence[int | float], offset: tuple[int, int], scale: float
) -> tuple[int, int, int, int]:
    return tuple(
        round(value * scale) + offset[index % 2] for index, value in enumerate(box)
    )  # type: ignore[return-value]


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def _dog_details(instances: Sequence[Any]) -> str:
    if not instances:
        return "largest dog: none"
    largest = max(instances, key=_box_area)
    return (
        f"largest dog #{largest.instance_index}: score {largest.class_score:.2f} | "
        f"area {_compact_area(_box_area(largest))} | {largest.quality.state}"
    )


def _box_area(instance: Any) -> int:
    return max(0, instance.detector_box_xyxy[2] - instance.detector_box_xyxy[0]) * max(
        0, instance.detector_box_xyxy[3] - instance.detector_box_xyxy[1]
    )


def _compact_area(area: int) -> str:
    if area < 1_000:
        return str(area)
    return f"{area / 1_000:.1f}k"


def _manifest_row(reason: str, item: Mapping[str, Any]) -> dict[str, Any]:
    row = item["row"]
    prediction = item["prediction"]
    instances = [
        {
            "instance_index": instance.instance_index,
            "class_name": instance.class_name,
            "class_score": instance.class_score,
            "detector_box_xyxy": list(instance.detector_box_xyxy),
            "mask_box_xyxy": (
                None if instance.mask_box_xyxy is None else list(instance.mask_box_xyxy)
            ),
            "quality_state": instance.quality.state,
            "quality_reasons": list(instance.quality.reasons),
            "foreground_pixels": instance.quality.foreground_pixels,
        }
        for instance in prediction.instances
    ]
    payload = {
        "reason": reason,
        "sample_token": row["sample_token"],
        "dataset_name": row["dataset_name"],
        "source_path": row["source_path"],
        "source_sha256": row["source_sha256"],
        "source_width": row["source_width"],
        "source_height": row["source_height"],
        "parser_prediction_sha256": item["receipt"]["parser_lineage"][
            "prediction_sha256"
        ],
        "instances": instances,
    }
    return {**payload, "record_sha256": content_sha256(payload)}


def _reason_slug(reason: str) -> str:
    return {
        "NO_PARSED_DOG_INSTANCE": "01_no_parsed_dog",
        "PARSER_INSTANCE_CARDINALITY_AMBIGUOUS": "02_multiple_instances",
        "PARSER_DISTINCT_DOG_CARDINALITY_AMBIGUOUS": "02_distinct_dogs",
        "NO_VALID_PARSED_DOG_INSTANCE": "03_no_valid_dog",
        "SELECTED_DOG_PARSING_UNUSABLE": "04_unusable_mask",
        "AP10K_GLOBAL_BBOX_ASSOCIATION_AMBIGUOUS": "05_ap10k_bbox_ambiguous",
    }[reason]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-plan", required=True, type=Path)
    parser.add_argument("--materialization-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples-per-reason", type=int, default=48)
    args = parser.parse_args(argv)
    manifest = render_parser_failure_review(
        route_plan=args.route_plan,
        materialization_root=args.materialization_root,
        output_dir=args.output_dir,
        samples_per_reason=args.samples_per_reason,
    )
    print(
        json.dumps(
            {
                "status": "CREATED_PARSER_FAILURE_VISUAL_REVIEW",
                "review_sha256": manifest["review_sha256"],
                "selected_sample_count": len(manifest["selected"]),
                "page_count": len(manifest["pages"]),
                "output": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
