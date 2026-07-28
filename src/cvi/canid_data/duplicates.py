"""Per-pixel duplicate detection across canid datasets.

Uses decoded-pixel SHA-256 (canonical RGB pixel hash, not file hash).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cvi.canid_data.types import UnifiedCanidSample


def canonical_pixel_hash(image_path: Path) -> str:
    """SHA-256 of the decoded (width, height, RGB pixels) domain."""
    import hashlib

    with Image.open(image_path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(rgb.shape[0].to_bytes(4, "big"))
    digest.update(rgb.shape[1].to_bytes(4, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def find_exact_duplicates(
    samples: tuple[UnifiedCanidSample, ...],
    data_root: Path,
) -> dict[str, list[str]]:
    """Return {pixel_hash: [sample_id, ...]} for duplicates only."""

    root = Path(data_root)
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for sample in samples:
        path = root / sample.image_path
        if not path.is_file():
            continue
        pixel_hash = canonical_pixel_hash(path)
        grouped[pixel_hash].append(sample.sample_id)
    return {hash_val: ids for hash_val, ids in grouped.items() if len(ids) > 1}


def find_cross_dataset_duplicates(
    samples_by_dataset: dict[str, tuple[UnifiedCanidSample, ...]],
    data_roots: dict[str, Path],
) -> dict[str, list[tuple[str, str, str]]]:
    """Return {pixel_hash: [(dataset, sample_id, image_path), ...]} cross-dataset only."""

    pixel_map: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for dataset_name, samples in samples_by_dataset.items():
        root = data_roots.get(dataset_name, Path())
        for sample in samples:
            path = root / sample.image_path
            if not path.is_file():
                continue
            pixel_hash = canonical_pixel_hash(path)
            pixel_map[pixel_hash].append(
                (dataset_name, sample.sample_id, sample.image_path)
            )
    return {
        hash_val: entries
        for hash_val, entries in pixel_map.items()
        if len({entry[0] for entry in entries}) > 1
    }


def summarize_duplicates(
    samples: tuple[UnifiedCanidSample, ...],
    data_root: Path,
) -> dict[str, Any]:
    duplicates = find_exact_duplicates(samples, data_root)
    return {
        "total_samples": len(samples),
        "duplicate_groups": len(duplicates),
        "duplicate_samples": sum(len(group) for group in duplicates.values()),
        "largest_group": max((len(g) for g in duplicates.values()), default=0),
    }
