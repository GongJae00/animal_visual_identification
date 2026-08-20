"""Sibetan layout adapter. Canonical name: ``sibetan``.

Publisher tree under the canonical dataset root
(``/mnt/r/Dataset/Animals_Dataset/sibetan``):

```text
Sibetan/gt_sibetan.json
Sibetan/gt_sibetan_no_mono_cluster.json
Sibetan/<cluster>/*.jpg
```

On disk: 223 digit cluster folders (``0``–``222``), 1,755 ``.jpg`` frames,
``gt_sibetan.json`` (59 dog keys → 223 clusters), and
``gt_sibetan_no_mono_cluster.json`` (39 multi-cluster dogs → 203 clusters).
Frames are ``Indonesia_C<nn>[-_]<date>_<media>_DSCF*.jpg``.

Identity unit: the GT JSON dog key in ``gt_sibetan.json``. Cluster folders
join to that key; ``capture_group_id`` is the cluster (sequence).

Not an identity: cluster/sequence folders, filename camera tokens ``C…``,
``gt_sibetan_no_mono_cluster.json``, non-digit siblings, or non-image files.
Camera tokens stay in ``metadata["unverified_camera_token"]``; ``camera_id``
stays unset. This adapter assigns ``split_role="UNASSIGNED"``.
Comparable-transfer EVAL is identity-disjoint from ``yt-bb-dog`` by dataset
namespace.
"""

from __future__ import annotations

import json
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


def adapt_sibetan(data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """Sibetan: camera-trap sequences joined to publisher GT dog identities."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "Sibetan"
    if not base.is_dir():
        raise FileNotFoundError(f"Sibetan base not found: {base}")
    gt_path = base / "gt_sibetan.json"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Sibetan identity GT not found: {gt_path}")
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise ValueError("Sibetan identity GT must be a non-empty object")
    identity_by_cluster: dict[str, str] = {}
    for dog_identity, raw_clusters in ground_truth.items():
        if (
            not isinstance(dog_identity, str)
            or not dog_identity.isdigit()
            or not isinstance(raw_clusters, list)
            or not raw_clusters
        ):
            raise ValueError("Sibetan identity GT schema differs")
        for cluster in raw_clusters:
            if isinstance(cluster, bool) or not isinstance(cluster, int) or cluster < 0:
                raise ValueError("Sibetan identity GT cluster differs")
            cluster_id = str(cluster)
            if cluster_id in identity_by_cluster:
                raise ValueError("Sibetan identity GT repeats a sequence cluster")
            identity_by_cluster[cluster_id] = dog_identity
    result: list[UnifiedCanidSample] = []
    observed_clusters: set[str] = set()
    for cluster_dir in sorted(base.iterdir()):
        if not cluster_dir.is_dir() or not cluster_dir.name.isdigit():
            continue
        cluster_id = cluster_dir.name
        dog_identity = identity_by_cluster.get(cluster_id)
        if dog_identity is None:
            raise ValueError("Sibetan image cluster is absent from identity GT")
        observed_clusters.add(cluster_id)
        dataset_identity = f"sibetan:v1:gt-json:{dog_identity}"
        reg_id = compute_registered_dog_id(dataset_identity)
        for image_file in sorted(cluster_dir.iterdir()):
            if image_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            image_file = _verified_path(root, image_file.relative_to(root).as_posix())
            stem = image_file.stem
            parts = stem.split("_")
            camera_part = next((p for p in parts if p.startswith("C")), "")
            sample_id = compute_sample_token(f"sibetan:{cluster_id}:{stem}")
            width, height = _image_dims(image_file)
            sha = _file_sha256(image_file)
            relative = str(image_file.relative_to(root))
            result.append(
                UnifiedCanidSample(
                    sample_id=sample_id,
                    dataset_name="sibetan",
                    dataset_version="publisher-v1-2025-10-27",
                    source_group_id=cluster_id,
                    image_path=relative,
                    image_sha256=sha,
                    width=width,
                    height=height,
                    registered_identity_id=reg_id,
                    raw_identity_id=dog_identity,
                    capture_group_id=cluster_id,
                    capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
                    split_role="UNASSIGNED",
                    metadata={"unverified_camera_token": camera_part or None},
                )
            )
    if observed_clusters != set(identity_by_cluster):
        raise ValueError("Sibetan identity GT and sequence directories differ")
    return tuple(result)
