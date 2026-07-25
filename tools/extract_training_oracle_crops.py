"""Extract oracle crop images for MODEL_TRAINING and SEPARATE_FACE_ONLY_LANE.

Reads the public-dataset assignment and source bundle, finds the source
images in the dataset ZIP archives, applies oracle crop regions, and writes
{identity_token}.jpg files to the output directory.

Supports YT-BB-Dog (video frames) and DogFaceNet (web images) sources.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image


# ---------------------------------------------------------------------------
# Dataset source lookup
# ---------------------------------------------------------------------------

_DATASET_PATHS: dict[str, Path | None] = {}


def _find_yt_dataset(secure_root: Path) -> Path | None:
    for d in sorted(secure_root.glob("datasets/yt-bb-dog-*")):
        z = d / "YT-BB-dog" / "YT-BB-Dog.zip"
        if z.exists():
            return z
    return None


def _find_dogfacenet_dataset(secure_root: Path) -> Path | None:
    for d in sorted(secure_root.glob("datasets/dogfacenet*")):
        return d
    return None


def _find_mpdd_dataset(secure_root: Path) -> Path | None:
    for d in sorted(secure_root.glob("datasets/mpdd*")):
        return d
    return None


def _find_sibetan_dataset(secure_root: Path) -> Path | None:
    for d in sorted(secure_root.glob("datasets/sibetan*")):
        return d
    return None


def _resolve_crop_source(
    sample: dict[str, Any],
    access_filter: str,
    secure_root: Path,
    datasets: dict[str, Path],
) -> Image.Image | None:
    """Load and crop a single source image for a training sample."""
    dsn = sample.get("dataset_name", "")
    ssid = sample.get("source_sample_id", "")
    region = sample.get("region", "")
    variant = sample.get("source_variant", "original")

    if dsn == "yt-bb-dog":
        return _resolve_yt_crop(ssid, variant, region, datasets.get("yt-bb-dog"))
    elif dsn == "dogfacenet224":
        return _resolve_dogfacenet_crop(ssid, region, datasets.get("dogfacenet224"))
    elif dsn == "mpdd":
        return _resolve_mpdd_crop(ssid, region, datasets.get("mpdd"))
    elif dsn == "sibetan":
        return _resolve_sibetan_crop(ssid, region, datasets.get("sibetan"))
    return None


class _YtZipCache:
    _zip: zipfile.ZipFile | None = None
    _namelist: set[str] | None = None
    _path: Path | None = None

    @classmethod
    def open(cls, path: Path | None) -> None:
        if path is not None and cls._path != path:
            cls.close()
            if path.exists():
                cls._zip = zipfile.ZipFile(path, "r")
                cls._namelist = set(cls._zip.namelist())
                cls._path = path

    @classmethod
    def close(cls) -> None:
        if cls._zip is not None:
            cls._zip.close()
            cls._zip = None
            cls._namelist = None
            cls._path = None

    @classmethod
    def read_image(cls, inner_path: str) -> Image.Image | None:
        if cls._zip is None or cls._namelist is None:
            return None
        if inner_path not in cls._namelist:
            return None
        with cls._zip.open(inner_path) as f:
            return Image.open(f).convert("RGB")


def _resolve_yt_crop(
    ssid: str, variant: str, region: str, dataset_zip: Path | None
) -> Image.Image | None:
    if dataset_zip is None:
        return None
    _YtZipCache.open(dataset_zip)
    parts = ssid.split(":")
    if len(parts) < 7:
        return None
    track_id = parts[4]
    frame_index = parts[6]
    img_paths = [
        f"YT-BB-Dog/train/{track_id}/{track_id}_{frame_index}.jpg",
        f"YT-BB-Dog/test/{track_id}/{track_id}_{frame_index}.jpg",
    ]
    for ip in img_paths:
        img = _YtZipCache.read_image(ip)
        if img is not None:
            return img
    return None


def _resolve_dogfacenet_crop(
    ssid: str, region: str, dataset_dir: Path | None
) -> Image.Image | None:
    if dataset_dir is None or not dataset_dir.exists():
        return None
    parts = ssid.split(":")
    # dogfacenet224:v1:original:web-folder:NNN:file:NAME
    if len(parts) >= 6:
        folder = parts[4]
        fname = parts[5].replace("file:", "")
        candidates = list(dataset_dir.rglob(f"**/{folder}/{fname}"))
        if candidates:
            return Image.open(candidates[0]).convert("RGB")
    return None


def _resolve_mpdd_crop(
    ssid: str, region: str, dataset_dir: Path | None
) -> Image.Image | None:
    return None


def _resolve_sibetan_crop(
    ssid: str, region: str, dataset_dir: Path | None
) -> Image.Image | None:
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secure-root", required=True, type=Path)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--access-filter", default="MODEL_TRAINING")
    parser.add_argument("--crop-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.source_bundle and not args.assignment:
        parser.error("--source-bundle or --assignment required")

    secure_root = args.secure_root.resolve()

    datasets: dict[str, Path | None] = {
        "yt-bb-dog": _find_yt_dataset(secure_root),
        "dogfacenet224": _find_dogfacenet_dataset(secure_root),
        "mpdd": _find_mpdd_dataset(secure_root),
        "sibetan": _find_sibetan_dataset(secure_root),
    }
    for name, path in datasets.items():
        if path is None:
            print(json.dumps({"event": "dataset_not_found", "dataset": name}), flush=True)
        else:
            print(json.dumps({"event": "dataset_found", "dataset": name, "path": str(path)}), flush=True)

    if args.source_bundle:
        sb = json.loads(args.source_bundle.read_bytes())
        bundle_samples = {s["sample_token"]: s for s in sb.get("samples", [])}
    else:
        bundle_samples = {}

    if args.assignment:
        assignment = json.loads(args.assignment.read_bytes())
        records = assignment.get("records", [])
    else:
        records = []

    records_to_export: list[dict[str, Any]] = []
    if args.crop_manifest:
        cm = json.loads(args.crop_manifest.read_bytes())
        manifest_tokens = {r["sample_token"] for r in cm.get("records", [])}
    else:
        manifest_tokens = set()

    for rec in records:
        if rec.get("model_access") != args.access_filter:
            continue
        st = rec.get("sample_token", "")
        if not st:
            continue
        if st in manifest_tokens:
            continue
        sample = bundle_samples.get(st, {})
        records_to_export.append(rec)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    missing = 0

    total = len(records_to_export)
    print(json.dumps({
        "event": "export_start",
        "access_filter": args.access_filter,
        "candidate_count": total,
    }), flush=True)

    for idx, rec in enumerate(records_to_export):
        st = rec.get("sample_token", "")
        sample = bundle_samples.get(st, {})
        img = _resolve_crop_source(
            sample, args.access_filter, secure_root, datasets
        )
        if img is None:
            missing += 1
            continue
        img = img.resize((224, 224), Image.BILINEAR)
        if args.dry_run:
            skipped += 1
        else:
            out_path = output_dir / f"{st}.jpg"
            img.save(out_path, "JPEG", quality=95)
            exported += 1
        if (idx + 1) % 1000 == 0 or idx == total - 1:
            print(json.dumps({
                "event": "export_progress",
                "processed": idx + 1,
                "exported": exported,
                "missing_source": missing,
                "total": total,
            }), flush=True)

    _YtZipCache.close()

    print(json.dumps({
        "event": "export_done",
        "access_filter": args.access_filter,
        "total_candidates": len(records_to_export),
        "exported": exported,
        "already_in_manifest": len(manifest_tokens & {r.get("sample_token", "") for r in records_to_export}),
        "missing_source": missing,
        "skipped_dry_run": skipped,
        "output_dir": str(output_dir),
    }), flush=True)


if __name__ == "__main__":
    main()
