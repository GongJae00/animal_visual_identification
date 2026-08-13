"""Render deterministic source/A/F/N contact sheets from region candidates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from foundation.protected_io import write_private_json_bundle
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file
from localization.dinov2_region_segmentation import read_region_candidates

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_PALETTES = {
    "A": {1: (52, 211, 153)},
    "F": {1: (250, 204, 21), 2: (56, 189, 248), 3: (167, 139, 250)},
    "N": {1: (251, 146, 60), 2: (244, 63, 94)},
}


def render_region_qa(
    *,
    candidate_manifest: Path,
    data_root: Path,
    output_dir: Path,
    sample_count: int,
) -> dict[str, object]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite region QA: {output_dir}")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be positive")
    manifest, arrays = read_region_candidates(candidate_manifest)
    root = data_root.resolve(strict=True)
    records = sorted(
        manifest["records"],
        key=lambda row: (
            -sum(row["regions"][region]["state"] == "AVAILABLE" for region in ("A", "F", "N")),
            hashlib.sha256(row["sample_id"].encode("ascii")).hexdigest(),
        ),
    )[:sample_count]
    width = 4 * 300 + 5 * 24
    height = 105 + len(records) * 285 + 30
    canvas = Image.new("RGB", (width, height), (9, 15, 26))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(_FONT_BOLD, 28)
    label_font = ImageFont.truetype(_FONT_BOLD, 17)
    small_font = ImageFont.truetype(_FONT, 13)
    draw.text(
        (24, 20),
        f"{manifest['dataset_name']} | DINOv2 patch-token region candidates",
        font=title_font,
        fill=(244, 247, 252),
    )
    draw.text(
        (24, 60),
        "MODEL_GENERATED_CANDIDATE only; not reviewed semantic segmentation",
        font=small_font,
        fill=(251, 146, 60),
    )
    for column, label in enumerate(("SOURCE", "A FULL BODY", "F EARS/FACE/NECK", "N NOSE")):
        draw.text((24 + column * 324, 84), label, font=label_font, fill=(148, 163, 184))
    selected: list[dict[str, object]] = []
    for row_index, record in enumerate(records):
        y = 112 + row_index * 285
        source = _read_source(root, record)
        source = _cover(source, (300, 240))
        canvas.paste(source, (24, y))
        draw.text(
            (24, y + 246),
            f"sample {row_index + 1:02d} | {record['image_width']}x{record['image_height']}",
            font=small_font,
            fill=(203, 213, 225),
        )
        selected_row = {
            "source_sha256": record["image_sha256"],
            "regions": {},
        }
        for column, region in enumerate(("A", "F", "N"), start=1):
            candidate = record["regions"][region]
            x = 24 + column * 324
            if candidate["state"] == "AVAILABLE":
                mask = arrays[f"{region}_masks"][record["array_index"]]
                overlay = _overlay(source, mask, _PALETTES[region])
                canvas.paste(overlay, (x, y))
                caption = (
                    f"support {candidate['support_fraction']:.3f} | "
                    f"confidence {candidate['confidence']:.3f}"
                )
                selected_row["regions"][region] = {
                    "state": "AVAILABLE",
                    "support_fraction": candidate["support_fraction"],
                    "confidence": candidate["confidence"],
                }
            else:
                tile = Image.new("RGB", (300, 240), (30, 41, 59))
                tile_draw = ImageDraw.Draw(tile)
                tile_draw.text((28, 100), "UNAVAILABLE", font=label_font, fill=(148, 163, 184))
                canvas.paste(tile, (x, y))
                caption = candidate["reason"]
                selected_row["regions"][region] = candidate
            draw.text((x, y + 246), caption, font=small_font, fill=(203, 213, 225))
        selected.append(selected_row)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        image_path = staging / "contact_sheet.png"
        canvas.save(image_path, format="PNG", compress_level=9, optimize=False)
        image_bytes = image_path.read_bytes()
        body = {
            "schema_version": "cvi.dinov2_region_qa.v1",
            "dataset_name": manifest["dataset_name"],
            "candidate_manifest_sha256": content_sha256(manifest),
            "selected": selected,
            "contact_sheet": {
                "relative_path": image_path.name,
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "byte_size": len(image_bytes),
                "width": width,
                "height": height,
            },
            "interpretation": (
                "MODEL_GENERATED_CANDIDATE_VISUAL_QA_NOT_SEMANTIC_OR_BIOMETRIC_VALIDATION"
            ),
        }
        qa = {**body, "qa_sha256": content_sha256(body)}
        write_private_json_bundle(((staging / "qa.json", qa),))
        fsync_directory(staging)
        rename_directory_noreplace(staging, output_dir)
        fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return qa


def _read_source(root: Path, record) -> Image.Image:
    relative = Path(record["image_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("region QA source path is unsafe")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError("region QA source is unsafe")
    retained = read_retained_regular_file(
        path,
        expected_sha256=record["image_sha256"],
        maximum_bytes=67_108_864,
        capture_payload=True,
        subject="region QA source image",
    )
    assert retained.payload is not None
    with Image.open(io.BytesIO(retained.payload)) as opened:
        image = opened.convert("RGB")
        image.load()
    return image


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = max(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def _overlay(
    source: Image.Image, mask: np.ndarray, palette: dict[int, tuple[int, int, int]]
) -> Image.Image:
    resized = np.asarray(
        Image.fromarray(mask.astype(np.uint8), mode="L").resize(
            source.size, Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )
    base = np.asarray(source, dtype=np.float32).copy()
    for value, color in palette.items():
        selected = resized == value
        base[selected] = 0.42 * base[selected] + 0.58 * np.asarray(color)
    return Image.fromarray(np.clip(np.rint(base), 0, 255).astype(np.uint8), mode="RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-candidates", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    args = parser.parse_args()
    qa = render_region_qa(
        candidate_manifest=args.region_candidates,
        data_root=args.data_root,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
    )
    print(
        json.dumps(
            {
                "status": "CREATED_DINOV2_REGION_QA",
                "dataset_name": qa["dataset_name"],
                "qa_sha256": qa["qa_sha256"],
                "output": str(args.output_dir),
                "interpretation": qa["interpretation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
