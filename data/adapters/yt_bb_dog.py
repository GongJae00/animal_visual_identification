"""YT-BB-Dog layout adapter. Canonical name: ``yt-bb-dog``.

Publisher folders under the canonical dataset root:

    <data_root>/YT-BB-dog/YT-BB-Dog/{train,test}/<id>/*.jpg

Admitted disk: ``.../yt-bb-dog/YT-BB-dog/YT-BB-Dog/{train,test}/<id>/*.jpg``.
Official split folders are ``train`` and ``test`` only. Identity is the ``<id>``
directory: a YouTube-BB video-track label, not a lifelong dog. Frame files
``{id}_{frame}.jpg`` are samples of that track.

Not an identity:
- ``YT-BB-Dog.zip`` and ``YT-BB-Dog_random_bckg.zip`` beside the extracted tree
- the random-background test variant (``YT-BB-Dog_random_bckg/``)
- outer wrappers ``yt-bb-dog/`` and ``YT-BB-dog/``, split folder names, zip members
- ``val`` or any directory other than official ``train`` / ``test``

``evaluation.parsed_body`` and ``evaluation.comparable_transfer`` call
``adapt_yt_bb_dog``. Comparable-transfer TRAIN keeps ``split_role == "train"``
only; this adapter still emits the official test split.
"""

from __future__ import annotations

import os
from pathlib import Path

from data.adapters.io import (
    _file_sha256,
    _image_dims,
    _verified_path,
)
from data.types import (
    CaptureGroupKind,
    UnifiedCanidSample,
)
from shared.contracts.identity_ids import (
    compute_registered_dog_id,
    compute_sample_token,
)


def adapt_yt_bb_dog(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """YT-BB-Dog: video-track identities with train/test split."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "YT-BB-dog" / "YT-BB-Dog"
    if not base.is_dir():
        raise FileNotFoundError(f"YT-BB-Dog base not found: {base}")
    result: list[UnifiedCanidSample] = []
    for split_role in ("train", "test"):
        split_dir = base / split_role
        if not split_dir.is_dir():
            continue
        for identity_dir in sorted(split_dir.iterdir(), key=lambda p: p.name):
            if not identity_dir.is_dir():
                continue
            identity_str = identity_dir.name
            dataset_identity = f"yt-bb-dog:v1:video-track:{identity_str}"
            reg_id = compute_registered_dog_id(dataset_identity)
            for image_file in sorted(identity_dir.iterdir()):
                if image_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                image_file = _verified_path(
                    root, image_file.relative_to(root).as_posix()
                )
                stem = image_file.stem
                sample_id = compute_sample_token(
                    f"yt-bb-dog:{identity_str}:{split_role}:{stem}"
                )
                width, height = _image_dims(image_file)
                sha = _file_sha256(image_file)
                relative = str(image_file.relative_to(root))
                result.append(
                    UnifiedCanidSample(
                        sample_id=sample_id,
                        dataset_name="yt-bb-dog",
                        dataset_version="publisher-v1-2025-10-27",
                        source_group_id=identity_str,
                        image_path=relative,
                        image_sha256=sha,
                        width=width,
                        height=height,
                        registered_identity_id=reg_id,
                        raw_identity_id=identity_str,
                        capture_group_id=identity_str,
                        capture_group_kind=CaptureGroupKind.VIDEO_TRACK,
                        split_role=split_role,
                    )
                )
    return tuple(result)


__all__ = ["adapt_yt_bb_dog"]
