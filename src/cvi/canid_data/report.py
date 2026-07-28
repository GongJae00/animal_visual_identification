"""Dataset statistics and admission report generation."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from cvi.canid_data.types import UnifiedCanidSample


def compute_dataset_statistics(
    samples: tuple[UnifiedCanidSample, ...],
) -> dict[str, Any]:
    if not samples:
        return {"error": "no samples"}

    identity_counts = Counter(s.registered_identity_id for s in samples if s.registered_identity_id)
    native_widths = np.asarray([s.width for s in samples])
    native_heights = np.asarray([s.height for s in samples])
    species_dist = Counter(s.species for s in samples)
    breed_dist = Counter(s.breed for s in samples if s.breed)
    label_availability = {
        label: sum(1 for s in samples if s.label_availability.get(label))
        / len(samples)
        for label in (
            "identity", "breed", "dog_bbox", "face_bbox",
            "face_landmarks", "body_keypoints", "nose_mask",
            "capture_group", "camera",
        )
    }
    split_dist = Counter(s.split_role for s in samples)

    return {
        "total_images": len(samples),
        "total_identities": len(identity_counts),
        "images_per_identity": {
            "min": min(identity_counts.values()),
            "max": max(identity_counts.values()),
            "mean": float(np.mean(list(identity_counts.values()))),
            "median": float(np.median(list(identity_counts.values()))),
        },
        "resolution": {
            "width": {
                "min": int(native_widths.min()),
                "max": int(native_widths.max()),
                "mean": float(native_widths.mean()),
            },
            "height": {
                "min": int(native_heights.min()),
                "max": int(native_heights.max()),
                "mean": float(native_heights.mean()),
            },
        },
        "species_distribution": dict(species_dist),
        "breed_count": len(breed_dist),
        "label_availability": label_availability,
        "split_distribution": dict(split_dist),
    }
