"""MPDD layout adapter.

Canonical name: ``mpdd``. Publisher tree under the dataset root:

```text
MPDD/pytorch/{train,val,query,gallery}/*.jpg
```

On disk that is the only tree: ``MPDD/`` contains ``pytorch/`` only, and
``pytorch/`` contains those four split directories. Images are ``.jpg``.
Observed counts: train 921, val 111, query 104, gallery 521 (1657 files).

Identity unit: the numeric filename prefix. Filenames are
``{identity}_c{camera}s{sequence}_{frame}.jpg`` (compact) or the audited
legacy form ``{identity}_c{camera}_{sequence}_{frame}.jpg``
(``query/146_c1_s3_1.jpg``). Identities are 0..190 (191 dogs). Train and val
share 95 identities; query and gallery share a disjoint bank of 96.

Not an identity: split folder names, camera token ``cN``, sequence token
``sN``, frame index, or the rest of the stem. Camera and sequence tokens are
unverified filename fields; they stay in metadata and do not set ``camera_id``.
Validation/DEV only, not FIT. Missing known split directories are skipped.
"""

from __future__ import annotations

import os
import re
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


def adapt_mpdd(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """MPDD: pose captures with explicit splits and unverified filename tokens."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "MPDD" / "pytorch"
    if not base.is_dir():
        raise FileNotFoundError(f"MPDD base not found: {base}")
    result: list[UnifiedCanidSample] = []
    for split_role in ("train", "val", "query", "gallery"):
        split_dir = base / split_role
        if not split_dir.is_dir():
            continue
        for image_file in sorted(split_dir.iterdir()):
            if image_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            image_file = _verified_path(root, image_file.relative_to(root).as_posix())
            stem = image_file.stem
            match = re.fullmatch(
                r"(?P<identity>\d+)_c(?P<camera>\d+)_?s(?P<sequence>\d+)_(?P<frame>\d+)",
                stem,
            )
            if match is None:
                raise ValueError(f"MPDD image filename schema differs: {image_file.name}")
            identity_str = match.group("identity")
            camera_str = f"c{match.group('camera')}"
            sequence_str = f"s{match.group('sequence')}"
            dataset_identity = f"mpdd:v1:device-capture:{identity_str}"
            reg_id = compute_registered_dog_id(dataset_identity)
            sample_id = compute_sample_token(f"mpdd:{stem}:{split_role}")
            width, height = _image_dims(image_file)
            sha = _file_sha256(image_file)
            relative = str(image_file.relative_to(root))
            capture_id = (
                f"{identity_str}{chr(0)}{camera_str}{chr(0)}{sequence_str}"
            )
            result.append(
                UnifiedCanidSample(
                    sample_id=sample_id,
                    dataset_name="mpdd",
                    dataset_version="mendeley-v5j6m8dzhv-v1",
                    source_group_id=identity_str,
                    image_path=relative,
                    image_sha256=sha,
                    width=width,
                    height=height,
                    registered_identity_id=reg_id,
                    raw_identity_id=identity_str,
                    capture_group_id=capture_id,
                    capture_group_kind=CaptureGroupKind.POSE_VIEW_CLUSTER,
                    split_role=split_role,
                    metadata={
                        "unverified_camera_token": camera_str,
                        "unverified_sequence_token": sequence_str,
                    },
                )
            )
    return tuple(result)
