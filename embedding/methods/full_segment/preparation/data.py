"""Secure, lazy data access for the assembled Full128 experiment inventory."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

from foundation.protected_io import read_strict_json_document
from foundation.provenance import content_sha256
from embedding.methods.full_segment.preparation.inventory import (
    BUNDLE_SCHEMA,
    validate_full128_experiment_inventory_bundle,
)
from embedding.methods.full_segment.preparation.materialization import ASSEMBLY_SCHEMA

_ASSEMBLY_FIELDS = {
    "schema_version",
    "plan_sha256",
    "sample_count",
    "allocation_name",
    "topology_report",
    "unified_full_split",
    "inventory_request",
    "inventory_bundle",
    "assembly_sha256",
}
_ELIGIBLE_STATUSES = {"USABLE", "REVIEW"}
_CACHE_ROLES = {"FIT", "DEV", "CAL", "EVAL"}
_MAX_ASSEMBLY_BYTES = 2_147_483_648
_MAX_CROP_BYTES = 67_108_864


@dataclass(frozen=True, slots=True)
class Full128Sample:
    """One content-bound materialized crop without preloaded image bytes."""

    sample_id: str
    identity_id: str
    dataset_name: str
    view: str
    role: str
    rgb_path: Path
    rgb_sha256: str
    mask_path: Path
    mask_sha256: str
    crop_record_sha256: str


@dataclass(frozen=True, slots=True)
class Full128Inventory:
    """Validated assembly bindings and lazily readable eligible samples."""

    assembly_sha256: str
    inventory_bundle_sha256: str
    inventory_sha256: str
    split_manifest_sha256: str
    split_census_sha256: str
    baseline_family_sha256: str
    artifact_root: Path
    samples: tuple[Full128Sample, ...]

    @property
    def fit_samples(self) -> tuple[Full128Sample, ...]:
        return tuple(sample for sample in self.samples if sample.role == "FIT")


def load_full128_assembly(
    path: Path,
    *,
    validation_workers: int = 1,
) -> tuple[Full128Inventory, dict[str, Any]]:
    """Read an assembly once and return its typed inventory and validated bundle."""

    document = read_strict_json_document(
        path,
        maximum_bytes=_MAX_ASSEMBLY_BYTES,
        maximum_nodes=25_000_000,
        maximum_keys=10_000_000,
        maximum_array_length=1_000_000,
    )
    assembly = document.payload
    if not isinstance(assembly, Mapping) or set(assembly) != _ASSEMBLY_FIELDS:
        raise ValueError("Full128 materialization assembly fields differ")
    if assembly["schema_version"] != ASSEMBLY_SCHEMA:
        raise ValueError("Full128 materialization assembly schema differs")
    payload = {
        key: value for key, value in assembly.items() if key != "assembly_sha256"
    }
    if assembly["assembly_sha256"] != content_sha256(payload):
        raise ValueError("Full128 materialization assembly digest differs")
    bundle_value = assembly["inventory_bundle"]
    if (
        not isinstance(bundle_value, dict)
        or bundle_value.get("schema_version") != BUNDLE_SCHEMA
    ):
        raise ValueError("assembly does not contain a Full128 v2 inventory bundle")
    bundle = validate_full128_experiment_inventory_bundle(
        bundle_value,
        validation_workers=validation_workers,
    )
    if assembly["sample_count"] != len(bundle["inventory"]["records"]):
        raise ValueError("Full128 assembly and inventory sample counts differ")
    if assembly["unified_full_split"] != bundle["split_bundle"]:
        raise ValueError("Full128 assembly and inventory split bundles differ")

    root = Path(bundle["artifact_root"])
    samples: list[Full128Sample] = []
    for record in bundle["inventory"]["records"]:
        if record["dataset_name"] == "petface-dog":
            raise ValueError("PetFace is blocked and cannot enter Full128 training")
        if record["full_status"] not in _ELIGIBLE_STATUSES:
            continue
        if not record["crop_artifacts_present"]:
            raise ValueError("eligible Full128 sample is missing crop artifacts")
        identity_id = record["identity_token"]
        if identity_id is None or record["identity_evidence_kind"] == "NONE":
            continue
        role = record["terminal_role"]
        if role not in _CACHE_ROLES:
            continue
        if role == "FIT" and not record["gradient_eligible"]:
            raise ValueError("Full128 FIT sample is not gradient eligible")
        view = (
            "face" if record["view_scope"] in {"FACE_NATIVE", "HEAD_NATIVE"} else "body"
        )
        samples.append(
            Full128Sample(
                sample_id=record["sample_token"],
                identity_id=identity_id,
                dataset_name=record["dataset_name"],
                view=view,
                role=role,
                rgb_path=root / record["full_rgb_path"],
                rgb_sha256=record["full_rgb_sha256"],
                mask_path=root / record["full_mask_path"],
                mask_sha256=record["full_mask_sha256"],
                crop_record_sha256=record["crop_record_sha256"],
            )
        )
    samples.sort(key=lambda sample: sample.sample_id)
    if not samples:
        raise ValueError("Full128 inventory has no materialized eligible samples")
    return (
        Full128Inventory(
            assembly_sha256=assembly["assembly_sha256"],
            inventory_bundle_sha256=bundle["bundle_sha256"],
            inventory_sha256=bundle["inventory_sha256"],
            split_manifest_sha256=bundle["split_manifest_sha256"],
            split_census_sha256=bundle["split_census_sha256"],
            baseline_family_sha256=bundle["baseline_family_sha256"],
            artifact_root=root,
            samples=tuple(samples),
        ),
        bundle,
    )


def read_full128_crop(sample: Full128Sample) -> tuple[np.ndarray, np.ndarray]:
    """Securely open and verify one exact 224x224 RGB and binary-mask pair."""

    rgb_bytes = _read_bound_file(
        sample.rgb_path, sample.rgb_sha256, label="Full128 RGB crop"
    )
    mask_bytes = _read_bound_file(
        sample.mask_path, sample.mask_sha256, label="Full128 mask crop"
    )
    try:
        with Image.open(io.BytesIO(rgb_bytes)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (224, 224):
                raise ValueError("Full128 RGB crop must be a 224x224 RGB PNG")
            image.load()
            rgb = np.asarray(image, dtype=np.uint8).copy()
        with Image.open(io.BytesIO(mask_bytes)) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (224, 224):
                raise ValueError("Full128 mask crop must be a 224x224 grayscale PNG")
            image.load()
            mask_values = np.asarray(image, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Full128 crop is not a supported image") from exc
    if not np.all((mask_values == 0) | (mask_values == 255)):
        raise ValueError("Full128 mask crop must contain only binary 0/255 values")
    mask = mask_values == 255
    if not mask.any():
        raise ValueError("Full128 mask crop must contain foreground")
    return rgb, mask


def read_full128_mask(sample: Full128Sample) -> np.ndarray:
    """Securely open one exact binary mask without decoding its RGB crop."""

    mask_bytes = _read_bound_file(
        sample.mask_path, sample.mask_sha256, label="Full128 mask crop"
    )
    try:
        with Image.open(io.BytesIO(mask_bytes)) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (224, 224):
                raise ValueError("Full128 mask crop must be a 224x224 grayscale PNG")
            image.load()
            values = np.asarray(image, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Full128 mask crop is not a supported image") from exc
    if not np.all((values == 0) | (values == 255)):
        raise ValueError("Full128 mask crop must contain only binary 0/255 values")
    mask = values == 255
    if not mask.any():
        raise ValueError("Full128 mask crop must contain foreground")
    return mask


class Full128TorchDataset:
    """Lazy torch dataset that never preloads the Full128 image population."""

    def __init__(
        self,
        samples: Sequence[Full128Sample],
        *,
        identity_to_label: Mapping[str, int] | None = None,
        payload_mode: Literal["public", "compact"] = "public",
    ) -> None:
        self.samples = tuple(samples)
        if not self.samples:
            raise ValueError("Full128 dataset must be non-empty")
        if payload_mode not in {"public", "compact"}:
            raise ValueError("Full128 dataset payload mode differs")
        self.payload_mode = payload_mode
        self.identity_to_label = (
            None if identity_to_label is None else dict(identity_to_label)
        )
        if self.identity_to_label is not None and any(
            sample.identity_id not in self.identity_to_label for sample in self.samples
        ):
            raise ValueError("Full128 identity label mapping is incomplete")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        sample = self.samples[index]
        rgb, mask = read_full128_crop(sample)
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask[None, ...].copy())
        if self.payload_mode == "compact":
            item: dict[str, Any] = {
                "rgb": rgb_tensor,
                "mask": mask_tensor,
            }
            if self.identity_to_label is not None:
                item["label"] = self.identity_to_label[sample.identity_id]
            return item
        item: dict[str, Any] = {
            "rgb": rgb_tensor.float().div_(255.0),
            "mask": mask_tensor.float(),
            "sample_index": index,
            "sample_id": sample.sample_id,
        }
        if self.identity_to_label is not None:
            item["label"] = self.identity_to_label[sample.identity_id]
        return item


def _read_bound_file(path: Path, expected_sha256: str, *, label: str) -> bytes:
    _require_sha256(expected_sha256, f"{label} SHA-256")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FileNotFoundError(f"{label} cannot be opened: {path}") from exc
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    observed = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= _MAX_CROP_BYTES
        ):
            raise ValueError(f"{label} size or file type differs")
        while chunk := os.read(
            descriptor, min(1_048_576, _MAX_CROP_BYTES + 1 - observed)
        ):
            observed += len(chunk)
            if observed > _MAX_CROP_BYTES:
                raise ValueError(f"{label} exceeds byte limit")
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        raise RuntimeError(f"{label} changed while being read")
    if observed != before.st_size or digest.hexdigest() != expected_sha256:
        raise ValueError(f"{label} content binding differs")
    return b"".join(chunks)


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "Full128Inventory",
    "Full128Sample",
    "Full128TorchDataset",
    "load_full128_assembly",
    "read_full128_crop",
    "read_full128_mask",
]
