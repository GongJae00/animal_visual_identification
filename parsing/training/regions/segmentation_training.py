"""Receipt-bound binary nose-mask student training and static ONNX export."""

from __future__ import annotations

import hashlib
import io
import math
import os
import random
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from shared.contracts.artifact_manifest import (
    ArtifactLicense,
    ImagePreprocessing,
    NoseMaskManifest,
    UsageLane,
)
from shared.foundation.protected_io import (
    read_strict_json_document,
    read_strict_json_object,
    write_private_json_bundle,
)
from shared.foundation.protected_publication import fsync_directory, rename_directory_noreplace
from shared.foundation.provenance import content_sha256
from parsing.export.regions.localizer import (
    IMAGE_MEAN,
    IMAGE_STD,
    MOBILENETV4_MODEL_NAME,
    MOBILENETV4_WEIGHTS_SHA256,
)
from parsing.export.regions.native_yt import validate_manifest_bundle
from parsing.training.regions.sam2_teacher import (
    validate_source_image_manifest,
    validate_teacher_manifest,
)

IMAGE_SIZE = 224
MODEL_ID = "identification.nose.nose_mask.mobilenetv4-conv-small.v1"
LICENSE_ID = "CC-BY-NC-4.0-derived"
CHECKPOINT_SCHEMA = "identification.nose.nose_segmentation_student.checkpoint.v1"
LINEAGE_SCHEMA = "identification.nose.nose_segmentation_student.artifact_bundle.v1"
PARITY_SCHEMA = "identification.nose.nose_segmentation_student.cpu_ort_parity.v1"
DEV_REPORT_SCHEMA = "identification.nose.nose_segmentation_student.dev_selection.v1"
SELECTION_METRICS = ("DEV_teacher_Dice", "DEV_teacher_IoU")
INTERPRETATION = "RESEARCH_ONLY_TEACHER_AGREEMENT_NOT_BIOMETRIC_VALIDATION"


@dataclass(frozen=True, slots=True)
class SegmentationRecord:
    sample_token: str
    registered_dog_id: str
    source_path: Path
    source_sha256: str
    mask_path: Path
    mask_sha256: str
    source_size: tuple[int, int]
    nose_box_xyxy: tuple[int, int, int, int]


def deterministic_identity_split(
    records: Sequence[SegmentationRecord], *, dev_fraction: float, seed: int
) -> tuple[tuple[SegmentationRecord, ...], tuple[SegmentationRecord, ...], dict[str, Any]]:
    """Split accepted masks by registered dog ID with no identity overlap."""

    if not records:
        raise ValueError("segmentation split requires accepted teacher masks")
    if (
        isinstance(dev_fraction, bool)
        or not isinstance(dev_fraction, (int, float))
        or not math.isfinite(dev_fraction)
        or not 0.0 < dev_fraction < 1.0
    ):
        raise ValueError("dev_fraction must be finite and in (0,1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    identities = sorted({record.registered_dog_id for record in records})
    if len(identities) < 2:
        raise ValueError("segmentation split requires at least two identities")
    ordered = sorted(
        identities,
        key=lambda identity: (
            hashlib.sha256(f"{seed}:{identity}".encode("ascii")).hexdigest(),
            identity,
        ),
    )
    dev_count = max(1, min(len(ordered) - 1, int(len(ordered) * dev_fraction + 0.5)))
    dev_ids = set(ordered[:dev_count])
    train = tuple(
        sorted(
            (record for record in records if record.registered_dog_id not in dev_ids),
            key=lambda record: record.sample_token,
        )
    )
    dev = tuple(
        sorted(
            (record for record in records if record.registered_dog_id in dev_ids),
            key=lambda record: record.sample_token,
        )
    )
    train_ids = sorted({record.registered_dog_id for record in train})
    selected_dev_ids = sorted(dev_ids)
    if not train or not dev or set(train_ids) & set(selected_dev_ids):
        raise RuntimeError("deterministic identity split is invalid")
    binding = {
        "algorithm": "SHA256_ASCENDING_SEED_COLON_REGISTERED_DOG_ID",
        "seed": seed,
        "dev_fraction": float(dev_fraction),
        "train_registered_dog_ids": train_ids,
        "dev_registered_dog_ids": selected_dev_ids,
        "train_sample_tokens": [record.sample_token for record in train],
        "dev_sample_tokens": [record.sample_token for record in dev],
    }
    return train, dev, binding


def load_training_records(
    *,
    teacher_manifest_path: Path,
    source_manifest_path: Path,
    native_manifest_path: Path,
    dev_fraction: float,
    seed: int,
) -> tuple[
    tuple[SegmentationRecord, ...],
    tuple[SegmentationRecord, ...],
    dict[str, Any],
]:
    """Validate and cross-bind rich teacher, source, and native manifests."""

    large_manifest_limits = {
        "maximum_bytes": 536_870_912,
        "maximum_nodes": 20_000_000,
        "maximum_keys": 10_000_000,
        "maximum_array_length": 2_000_000,
    }
    teacher_document = read_strict_json_document(
        teacher_manifest_path, **large_manifest_limits
    )
    source_document = read_strict_json_document(
        source_manifest_path, **large_manifest_limits
    )
    native_document = read_strict_json_document(
        native_manifest_path, **large_manifest_limits
    )
    teacher = validate_teacher_manifest(
        teacher_document.payload, root=teacher_manifest_path.parent
    )
    source_binding = teacher["source_binding"]
    if (
        source_binding["source_manifest_schema"]
        != source_document.payload.get("schema_version")
        or source_binding["source_manifest_file_sha256"] != source_document.raw_sha256
        or source_binding["source_manifest_payload_sha256"]
        != source_document.canonical_payload_sha256
        or source_binding["source_receipt_file_sha256"]
        != source_document.payload.get("source_receipt_file_sha256")
    ):
        raise ValueError("teacher/source manifest binding differs")
    sources = validate_source_image_manifest(
        source_document.payload,
        root=source_manifest_path.parent,
        source_receipt_file_sha256=source_binding["source_receipt_file_sha256"],
    )
    native = validate_manifest_bundle(
        native_document.payload, root=native_manifest_path.parent
    )
    if native_document.payload["manifest_sha256"] != content_sha256(native):
        raise ValueError("native manifest payload binding differs")

    source_rows = {
        row["sample_token"]: row for row in source_document.payload["records"]
    }
    teacher_rows = {row["sample_token"]: row for row in teacher["records"]}
    native_rows = {
        row["sample_token"]: row
        for row in native["records"]
        if row["record_state"] != "NO_ROI"
    }
    expected_tokens = set(source_rows)
    if set(teacher_rows) != expected_tokens or set(native_rows) != expected_tokens:
        raise ValueError("teacher/source/native sample coverage differs")
    source_objects = {source.sample_token: source for source in sources}
    accepted: list[SegmentationRecord] = []
    for token in sorted(expected_tokens):
        source_row = source_rows[token]
        teacher_row = teacher_rows[token]
        native_row = native_rows[token]
        common = (
            "sample_token",
            "sequence_token",
            "track_token",
            "frame_index",
            "source_sha256",
            "source_width",
            "source_height",
            "nose_box_xyxy",
        )
        if any(
            source_row[name] != teacher_row[name]
            or source_row[name] != native_row[name]
            for name in common
        ) or source_row["keypoints"] != native_row["keypoints"]:
            raise ValueError(f"teacher/source/native record binding differs for {token}")
        expected_positive = [
            [float(point["source_x"]), float(point["source_y"])]
            for point in source_row["keypoints"][2:]
            if point["confidence"] > 0.0
        ]
        if teacher_row["positive_keypoints_xy"] != expected_positive:
            raise ValueError(f"teacher/source keypoint binding differs for {token}")
        if teacher_row["status"] != "ACCEPTED":
            continue
        source = source_objects[token]
        mask_path = _bound_relative_file(
            teacher_manifest_path.parent,
            teacher_row["mask_path"],
            teacher_row["mask_sha256"],
            "teacher mask",
        )
        source_path = _source_path(source_manifest_path.parent, source_row)
        accepted.append(
            SegmentationRecord(
                sample_token=token,
                registered_dog_id=native_row["registered_dog_id"],
                source_path=source_path,
                source_sha256=source.source_sha256,
                mask_path=mask_path,
                mask_sha256=teacher_row["mask_sha256"],
                source_size=(source.source_width, source.source_height),
                nose_box_xyxy=source.nose_box_xyxy,
            )
        )
    if not accepted:
        raise ValueError("teacher manifest contains no accepted masks")
    train, dev, split = deterministic_identity_split(
        accepted, dev_fraction=dev_fraction, seed=seed
    )
    bindings = {
        "manifests": {
            "teacher": {
                "file_sha256": teacher_document.raw_sha256,
                "payload_sha256": teacher["manifest_sha256"],
            },
            "source": {
                "file_sha256": source_document.raw_sha256,
                "payload_sha256": source_document.canonical_payload_sha256,
                "source_receipt_file_sha256": source_binding[
                    "source_receipt_file_sha256"
                ],
            },
            "native": {
                "file_sha256": native_document.raw_sha256,
                "payload_sha256": native_document.payload["manifest_sha256"],
            },
        },
        "split": split,
        "accepted_mask_count": len(accepted),
        "rejected_mask_count": len(teacher_rows) - len(accepted),
        "license": {
            "license_id": LICENSE_ID,
            "usage_lane": UsageLane.RESEARCH_ONLY.value,
            "reason": "Derived from the research-only native nose localization lane.",
        },
    }
    _validate_input_bindings(bindings)
    return train, dev, bindings


class NoseSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Hash-checked exact nose-box crops with paired 224-pixel transforms."""

    def __init__(
        self,
        records: Sequence[SegmentationRecord],
        *,
        training: bool,
        seed: int,
    ) -> None:
        if not records:
            raise ValueError("segmentation dataset requires records")
        self.records = tuple(records)
        self.training = training
        self.seed = seed
        self.epoch = 0
        self._mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32)[:, None, None]
        self._std = torch.tensor(IMAGE_STD, dtype=torch.float32)[:, None, None]

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("dataset epoch must be nonnegative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        record = self.records[index]
        source_bytes = _read_bound_file(
            record.source_path, record.source_sha256, "source image"
        )
        mask_bytes = _read_bound_file(record.mask_path, record.mask_sha256, "teacher mask")
        with Image.open(io.BytesIO(source_bytes)) as opened:
            source = opened.convert("RGB")
            source.load()
        with Image.open(io.BytesIO(mask_bytes)) as opened:
            if opened.mode != "L":
                raise ValueError("teacher mask must remain an L image")
            mask = opened.copy()
        if source.size != record.source_size or mask.size != record.source_size:
            raise ValueError("source and teacher mask geometry changed")
        image_crop = source.crop(record.nose_box_xyxy)
        mask_crop = mask.crop(record.nose_box_xyxy)
        if not np.isin(np.asarray(mask_crop), (0, 255)).all() or not np.any(
            np.asarray(mask_crop) == 255
        ):
            raise ValueError("cropped teacher mask must be non-empty binary pixels")
        if self.training:
            rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
            if rng.random() < 0.5:
                image_crop = ImageOps.mirror(image_crop)
                mask_crop = ImageOps.mirror(mask_crop)
        image_crop = image_crop.resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        mask_crop = mask_crop.resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
        )
        image_array = np.asarray(image_crop, dtype=np.uint8).copy()
        image_tensor = (
            torch.from_numpy(image_array).permute(2, 0, 1).float().mul_(1.0 / 255.0)
        )
        image_tensor = (image_tensor - self._mean) / self._std
        mask_array = (np.asarray(mask_crop, dtype=np.uint8) == 255).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array.copy()).unsqueeze(0)
        return image_tensor, mask_tensor, index


class NoseSegmentationStudent(nn.Module):
    """Fuse MobileNet semantics with a stride-four RGB stem for mask detail."""

    def __init__(
        self, backbone: nn.Module, feature_channels: int, *, decoder_channels: int = 64
    ) -> None:
        super().__init__()
        if feature_channels <= 0 or decoder_channels <= 0:
            raise ValueError("segmentation decoder channels must be positive")
        self.backbone = backbone
        stem_channels = max(4, decoder_channels // 2)
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                stem_channels,
                decoder_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.backbone_projection = nn.Conv2d(
            feature_channels, decoder_channels, kernel_size=1
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(
                2 * decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                decoder_channels,
                stem_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, 1, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 4:
            raise ValueError("MobileNetV4 forward_features must return a spatial tensor")
        detail = self.rgb_stem(images)
        semantic = F.interpolate(
            self.backbone_projection(features),
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        logits = self.decoder(torch.cat((detail, semantic), dim=1))
        return F.interpolate(
            logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )


def load_mobilenetv4_student(
    weights_path: Path, *, decoder_channels: int = 64
) -> NoseSegmentationStudent:
    """Load the repository's exact pretrained MobileNetV4 safetensors."""

    path = Path(weights_path)
    if path.suffix != ".safetensors" or path.is_symlink() or not path.is_file():
        raise ValueError("backbone weights must be a regular .safetensors file")
    digest = _sha256_file(path)
    if digest != MOBILENETV4_WEIGHTS_SHA256:
        raise ValueError(
            "MobileNetV4 safetensors SHA256 differs: "
            f"expected {MOBILENETV4_WEIGHTS_SHA256}, got {digest}"
        )
    import timm
    from safetensors.torch import load_file

    backbone = timm.create_model(MOBILENETV4_MODEL_NAME, pretrained=False)
    backbone.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    feature_channels = getattr(backbone, "num_features", None)
    if not isinstance(feature_channels, int) or feature_channels <= 0:
        raise ValueError("MobileNetV4 spatial feature width is unavailable")
    return NoseSegmentationStudent(
        backbone, feature_channels, decoder_channels=decoder_channels
    )


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return BCE-with-logits plus soft Dice loss."""

    if logits.ndim != 4 or logits.shape != target.shape or logits.shape[1] != 1:
        raise ValueError("segmentation logits and targets must match [B,1,H,W]")
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise ValueError("segmentation tensors must be finite")
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    axes = (1, 2, 3)
    intersection = (probability * target).sum(dim=axes)
    dice = (2.0 * intersection + 1.0) / (
        probability.sum(dim=axes) + target.sum(dim=axes) + 1.0
    )
    dice_loss = 1.0 - dice.mean()
    return {"total": bce + dice_loss, "bce": bce, "dice": dice_loss}


def teacher_agreement_metrics(
    probability: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5
) -> dict[str, float]:
    """Compute mean per-image binary IoU and Dice against accepted teachers."""

    if probability.shape != target.shape or probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError("teacher agreement tensors must match [B,1,H,W]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("mask threshold must be in [0,1]")
    if not torch.isfinite(probability).all() or torch.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("mask probabilities must be finite and in [0,1]")
    prediction = probability >= threshold
    truth = target >= 0.5
    axes = (1, 2, 3)
    intersection = (prediction & truth).sum(dim=axes).float()
    predicted = prediction.sum(dim=axes).float()
    expected = truth.sum(dim=axes).float()
    union = predicted + expected - intersection
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    denominator = predicted + expected
    dice = torch.where(
        denominator > 0, 2.0 * intersection / denominator, torch.ones_like(denominator)
    )
    return {"IoU": float(iou.mean()), "Dice": float(dice.mean())}


def build_training_config(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    decoder_channels: int,
    num_workers: int,
    seed: int,
    dev_fraction: float,
    threshold: float,
    device_name: str,
    parity_max_absolute_error: float,
) -> dict[str, Any]:
    config = {
        "schema_version": "identification.nose.nose_segmentation_student.training_config.v1",
        "model": {
            "backbone": MOBILENETV4_MODEL_NAME,
            "spatial_feature_channels": 960,
            "decoder_channels": decoder_channels,
            "decoder_fusion": "STRIDE4_RGB_STEM_PLUS_UPSAMPLED_BACKBONE_FEATURES",
        },
        "input": {
            "shape": [3, IMAGE_SIZE, IMAGE_SIZE],
            "crop": "EXACT_SOURCE_RGB_AND_BINARY_MASK_BY_NOSE_BOX_XYXY",
            "resize": "JOINT_DIRECT_BILINEAR_RGB_NEAREST_BINARY_MASK",
            "scale": 1.0 / 255.0,
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
        "augmentation": {"joint_horizontal_flip_probability": 0.5},
        "loss": {"bce_with_logits_weight": 1.0, "soft_dice_weight": 1.0},
        "optimizer": {
            "type": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "mask_threshold": threshold,
        "device": device_name,
        "selection_metric_order": list(SELECTION_METRICS),
        "parity_max_absolute_error": parity_max_absolute_error,
    }
    _validate_training_config(config)
    return config


def build_segmentation_checkpoint(
    *,
    model: nn.Module,
    epoch: int,
    selection: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "selection": dict(selection),
        "input_bindings": dict(input_bindings),
        "input_bindings_sha256": content_sha256(dict(input_bindings)),
        "training_config": dict(training_config),
        "training_config_sha256": content_sha256(dict(training_config)),
        "model_state_dict": state,
        "model_state_sha256": _state_dict_sha256(state),
    }
    payload["checkpoint_payload_sha256"] = _checkpoint_metadata_sha256(payload)
    validate_segmentation_checkpoint(payload)
    return payload


def validate_segmentation_checkpoint(payload: object) -> None:
    expected = {
        "schema_version",
        "epoch",
        "selection",
        "input_bindings",
        "input_bindings_sha256",
        "training_config",
        "training_config_sha256",
        "model_state_dict",
        "model_state_sha256",
        "checkpoint_payload_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("nose segmentation checkpoint keys differ")
    if payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported nose segmentation checkpoint schema")
    if isinstance(payload["epoch"], bool) or not isinstance(payload["epoch"], int) or payload["epoch"] <= 0:
        raise ValueError("checkpoint epoch must be positive")
    selection = payload["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "metric_order",
        "Dice",
        "IoU",
        "interpretation",
    }:
        raise ValueError("checkpoint selection keys differ")
    if selection["metric_order"] != list(SELECTION_METRICS) or selection["interpretation"] != INTERPRETATION:
        raise ValueError("checkpoint selection contract differs")
    for name in ("Dice", "IoU"):
        value = selection[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"checkpoint selection {name} differs")
    _validate_input_bindings(payload["input_bindings"], require_backbone=True)
    _validate_training_config(payload["training_config"])
    if content_sha256(payload["input_bindings"]) != payload["input_bindings_sha256"]:
        raise ValueError("checkpoint input bindings digest differs")
    if content_sha256(payload["training_config"]) != payload["training_config_sha256"]:
        raise ValueError("checkpoint training config digest differs")
    if _state_dict_sha256(payload["model_state_dict"]) != payload["model_state_sha256"]:
        raise ValueError("checkpoint model state digest differs")
    if _checkpoint_metadata_sha256(payload) != payload["checkpoint_payload_sha256"]:
        raise ValueError("checkpoint payload digest differs")


def save_segmentation_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    validate_segmentation_checkpoint(dict(payload))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    with path.open("xb") as stream:
        torch.save(dict(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    validate_segmentation_checkpoint(
        torch.load(path, map_location="cpu", weights_only=True)
    )


def load_segmentation_checkpoint(
    path: Path,
    *,
    expected_input_bindings: Mapping[str, Any] | None = None,
    expected_training_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_segmentation_checkpoint(payload)
    if expected_input_bindings is not None and payload["input_bindings"] != dict(
        expected_input_bindings
    ):
        raise ValueError("checkpoint input bindings differ from expected bindings")
    if expected_training_config is not None and payload["training_config"] != dict(
        expected_training_config
    ):
        raise ValueError("checkpoint training config differs from expected config")
    return payload


class _SigmoidExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(images)).reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)


def export_static_onnx(model: nn.Module, path: Path) -> tuple[str, int]:
    """Export a self-contained static batch-one graph with sigmoid output."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite ONNX artifact: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        wrapper = _SigmoidExport(model.to(torch.device("cpu")).eval()).eval()
        dummy = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (dummy,),
                temporary,
                input_names=["images"],
                output_names=["mask"],
                opset_version=18,
                external_data=False,
                dynamo=False,
            )
        os.chmod(temporary, 0o600)
        digest, byte_count = validate_static_onnx(temporary)
        os.link(temporary, path)
        fsync_directory(path.parent)
        return digest, byte_count
    finally:
        temporary.unlink(missing_ok=True)


def validate_static_onnx(path: Path) -> tuple[str, int]:
    import onnx
    import onnxruntime as ort

    payload = _read_bound_file(path, None, "nose segmentation ONNX")
    model = onnx.load_model_from_string(payload)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("nose segmentation ONNX must have one input and output")
    model_input, model_output = model.graph.input[0], model.graph.output[0]
    if model_input.name != "images" or model_output.name != "mask":
        raise ValueError("nose segmentation ONNX tensor names differ")
    if _onnx_shape(model_input) != (1, 3, IMAGE_SIZE, IMAGE_SIZE) or _onnx_shape(
        model_output
    ) != (1, 1, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("nose segmentation ONNX tensor shapes differ")
    if any(initializer.external_data for initializer in model.graph.initializer):
        raise ValueError("nose segmentation ONNX must not use external tensor data")
    if not any(node.op_type == "Sigmoid" for node in model.graph.node):
        raise ValueError("nose segmentation ONNX must contain sigmoid output")
    session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    if tuple(session.get_inputs()[0].shape) != (1, 3, IMAGE_SIZE, IMAGE_SIZE) or tuple(
        session.get_outputs()[0].shape
    ) != (1, 1, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("CPU ONNX Runtime tensor contract differs")
    output = session.run(
        ["mask"],
        {"images": np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)},
    )[0]
    _validate_probability_output(output, "CPU ONNX Runtime smoke")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def build_runtime_manifest(onnx_path: Path, *, threshold: float) -> NoseMaskManifest:
    return NoseMaskManifest(
        artifact_id=MODEL_ID,
        artifact_sha256=_sha256_file(onnx_path),
        input_name="images",
        input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        output_name="mask",
        output_shape=(1, 1, IMAGE_SIZE, IMAGE_SIZE),
        license=ArtifactLicense(LICENSE_ID, UsageLane.RESEARCH_ONLY),
        preprocessing=ImagePreprocessing(
            color_mode="RGB",
            layout="NCHW",
            dtype="float32",
            resize="bilinear",
            scale=1.0 / 255.0,
            mean=IMAGE_MEAN,
            std=IMAGE_STD,
            clahe=None,
        ),
        threshold=threshold,
    )


def produce_cpu_ort_parity_receipt(
    *,
    model: nn.Module,
    onnx_path: Path,
    checkpoint_path: Path,
    record: SegmentationRecord,
    maximum_absolute_error: float,
) -> dict[str, Any]:
    """Compare selected PyTorch probabilities with CPU ORT on a bound crop."""

    import onnxruntime as ort

    if maximum_absolute_error <= 0.0 or not math.isfinite(maximum_absolute_error):
        raise ValueError("parity maximum absolute error must be finite and positive")
    dataset = NoseSegmentationDataset((record,), training=False, seed=0)
    image, _, _ = dataset[0]
    tensor = image.unsqueeze(0).numpy().astype(np.float32, copy=False)
    model = model.to(torch.device("cpu")).eval()
    with torch.inference_mode():
        reference = torch.sigmoid(model(torch.from_numpy(tensor))).numpy()
    session = ort.InferenceSession(
        onnx_path.read_bytes(), providers=["CPUExecutionProvider"]
    )
    candidate = session.run(["mask"], {"images": tensor})[0]
    _validate_probability_output(reference, "PyTorch parity")
    _validate_probability_output(candidate, "CPU ONNX Runtime parity")
    maximum_error = float(np.max(np.abs(reference - candidate)))
    if maximum_error > maximum_absolute_error:
        raise RuntimeError(
            f"nose segmentation CPU ORT parity failed: {maximum_error} > "
            f"{maximum_absolute_error}"
        )
    receipt = {
        "schema_version": PARITY_SCHEMA,
        "model_id": MODEL_ID,
        "artifact_sha256": _sha256_file(onnx_path),
        "selected_checkpoint_sha256": _sha256_file(checkpoint_path),
        "fixture": {
            "sample_token": record.sample_token,
            "source_sha256": record.source_sha256,
            "teacher_mask_sha256": record.mask_sha256,
            "nose_box_xyxy": list(record.nose_box_xyxy),
        },
        "reference_backend": f"torch={torch.__version__};selected-checkpoint",
        "candidate_backend": f"onnxruntime-cpu={ort.__version__}",
        "reference_output_sha256": hashlib.sha256(
            np.ascontiguousarray(reference).tobytes()
        ).hexdigest(),
        "candidate_output_sha256": hashlib.sha256(
            np.ascontiguousarray(candidate).tobytes()
        ).hexdigest(),
        "maximum_absolute_error": maximum_error,
        "threshold": maximum_absolute_error,
        "finite": True,
        "range": [0.0, 1.0],
        "decision": "PASS",
    }
    receipt["receipt_sha256"] = content_sha256(receipt)
    _validate_parity_receipt(receipt)
    return receipt


def train_and_export(
    *,
    teacher_manifest_path: Path,
    source_manifest_path: Path,
    native_manifest_path: Path,
    backbone_weights_path: Path,
    output_dir: Path,
    device_name: str = "cpu",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    decoder_channels: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    dev_fraction: float = 0.2,
    threshold: float = 0.5,
    parity_max_absolute_error: float = 1e-4,
) -> dict[str, Any]:
    """Train on accepted masks, select best DEV Dice/IoU, and publish once."""

    _validate_output_destination(output_dir)
    config = build_training_config(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        decoder_channels=decoder_channels,
        num_workers=num_workers,
        seed=seed,
        dev_fraction=dev_fraction,
        threshold=threshold,
        device_name=device_name,
        parity_max_absolute_error=parity_max_absolute_error,
    )
    train_records, dev_records, input_bindings = load_training_records(
        teacher_manifest_path=teacher_manifest_path,
        source_manifest_path=source_manifest_path,
        native_manifest_path=native_manifest_path,
        dev_fraction=dev_fraction,
        seed=seed,
    )
    input_bindings = {
        **input_bindings,
        "backbone": {
            "model_name": MOBILENETV4_MODEL_NAME,
            "source_revision": "c9f31ac64483d7f0590db9edccb4418392a96eea",
            "safetensors_sha256": _sha256_file(backbone_weights_path),
        },
    }
    _validate_input_bindings(input_bindings, require_backbone=True)
    if input_bindings["backbone"]["safetensors_sha256"] != MOBILENETV4_WEIGHTS_SHA256:
        raise ValueError("MobileNetV4 backbone digest differs from repository contract")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_name == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(device_name)
    model = load_mobilenetv4_student(
        backbone_weights_path, decoder_channels=decoder_channels
    ).to(device)
    train_dataset = NoseSegmentationDataset(train_records, training=True, seed=seed)
    dev_dataset = NoseSegmentationDataset(dev_records, training=False, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, Any]] = []
    best_key = (-1.0, -1.0)
    best_checkpoint: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch - 1)
        train_metrics = _train_epoch(model, train_loader, optimizer, device)
        dev_metrics = _evaluate(model, dev_loader, device, threshold=threshold)
        history.append({"epoch": epoch, "train": train_metrics, "dev": dev_metrics})
        selection = {
            "metric_order": list(SELECTION_METRICS),
            "Dice": dev_metrics["Dice"],
            "IoU": dev_metrics["IoU"],
            "interpretation": INTERPRETATION,
        }
        key = (dev_metrics["Dice"], dev_metrics["IoU"])
        if key > best_key:
            best_key = key
            best_checkpoint = build_segmentation_checkpoint(
                model=model,
                epoch=epoch,
                selection=selection,
                input_bindings=input_bindings,
                training_config=config,
            )
    if best_checkpoint is None:
        raise RuntimeError("training did not produce a selected checkpoint")
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model.to(torch.device("cpu")).eval()

    output_parent = output_dir.parent.resolve(strict=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_parent)
    )
    try:
        checkpoint_path = staging / "selected_checkpoint.pt"
        save_segmentation_checkpoint(checkpoint_path, best_checkpoint)
        onnx_path = staging / "nose_mask.onnx"
        onnx_sha256, onnx_bytes = export_static_onnx(model, onnx_path)
        runtime = build_runtime_manifest(onnx_path, threshold=threshold)
        if runtime.artifact_sha256 != onnx_sha256:
            raise RuntimeError("runtime manifest ONNX digest differs")
        runtime_path = staging / "nose_mask.runtime.json"
        write_private_json_bundle(((runtime_path, runtime.to_dict()),))
        parity = produce_cpu_ort_parity_receipt(
            model=model,
            onnx_path=onnx_path,
            checkpoint_path=checkpoint_path,
            record=dev_records[0],
            maximum_absolute_error=parity_max_absolute_error,
        )
        parity_path = staging / "cpu_ort_parity.json"
        dev_report = {
            "schema_version": DEV_REPORT_SCHEMA,
            "interpretation": INTERPRETATION,
            "selection_metric_order": list(SELECTION_METRICS),
            "selected_epoch": best_checkpoint["epoch"],
            "best_Dice": best_key[0],
            "best_IoU": best_key[1],
            "history": history,
        }
        dev_path = staging / "dev_selection.json"
        write_private_json_bundle(
            ((parity_path, parity), (dev_path, dev_report))
        )
        lineage = _build_lineage(
            root=staging,
            input_bindings=input_bindings,
            training_config=config,
            onnx_bytes=onnx_bytes,
            parity=parity,
            dev_report=dev_report,
        )
        lineage_path = staging / "artifact_lineage.json"
        write_private_json_bundle(((lineage_path, lineage),))
        validate_lineage_manifest(lineage, staging)
        fsync_directory(staging)
        rename_directory_noreplace(staging, output_dir)
        fsync_directory(output_parent)
        return lineage
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_lineage_manifest(payload: object, root: Path) -> None:
    expected = {
        "schema_version",
        "artifacts",
        "onnx_contract",
        "input_bindings",
        "input_bindings_sha256",
        "training_config",
        "training_config_sha256",
        "dev_selection_payload_sha256",
        "parity_payload_sha256",
        "license",
        "interpretation",
        "lineage_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("nose segmentation lineage keys differ")
    body = {key: value for key, value in payload.items() if key != "lineage_sha256"}
    if payload["schema_version"] != LINEAGE_SCHEMA or content_sha256(body) != payload["lineage_sha256"]:
        raise ValueError("nose segmentation lineage digest differs")
    _validate_input_bindings(payload["input_bindings"], require_backbone=True)
    _validate_training_config(payload["training_config"])
    if content_sha256(payload["input_bindings"]) != payload["input_bindings_sha256"]:
        raise ValueError("lineage input bindings digest differs")
    if content_sha256(payload["training_config"]) != payload["training_config_sha256"]:
        raise ValueError("lineage training config digest differs")
    if payload["license"] != {
        "license_id": LICENSE_ID,
        "usage_lane": UsageLane.RESEARCH_ONLY.value,
    } or payload["interpretation"] != INTERPRETATION:
        raise ValueError("lineage research-only contract differs")
    artifacts = payload["artifacts"]
    expected_paths = {
        "selected_checkpoint": "selected_checkpoint.pt",
        "onnx": "nose_mask.onnx",
        "runtime_manifest": "nose_mask.runtime.json",
        "parity_receipt": "cpu_ort_parity.json",
        "dev_selection": "dev_selection.json",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_paths):
        raise ValueError("lineage artifact set differs")
    resolved_root = root.resolve(strict=True)
    for name, expected_path in expected_paths.items():
        artifact = artifacts[name]
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "bytes"} or artifact["path"] != expected_path:
            raise ValueError("lineage artifact contract differs")
        path = _bound_relative_file(
            resolved_root, artifact["path"], artifact["sha256"], "lineage artifact"
        )
        if path.stat().st_size != artifact["bytes"]:
            raise ValueError("lineage artifact byte count differs")
    runtime = NoseMaskManifest.from_dict(
        read_strict_json_object(resolved_root / expected_paths["runtime_manifest"])
    )
    expected_runtime = build_runtime_manifest(
        resolved_root / expected_paths["onnx"],
        threshold=payload["training_config"]["mask_threshold"],
    )
    if runtime != expected_runtime:
        raise ValueError("lineage runtime manifest differs")
    if payload["onnx_contract"] != {
        "input_name": "images",
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "output_name": "mask",
        "output_shape": [1, 1, IMAGE_SIZE, IMAGE_SIZE],
        "output_semantics": "SIGMOID_PROBABILITY",
        "opset": 18,
        "external_data": False,
        "onnx_bytes": artifacts["onnx"]["bytes"],
    }:
        raise ValueError("lineage ONNX contract differs")
    validate_static_onnx(resolved_root / expected_paths["onnx"])
    parity = read_strict_json_object(resolved_root / expected_paths["parity_receipt"])
    _validate_parity_receipt(parity)
    if (
        parity["receipt_sha256"] != payload["parity_payload_sha256"]
        or parity["artifact_sha256"] != artifacts["onnx"]["sha256"]
        or parity["selected_checkpoint_sha256"]
        != artifacts["selected_checkpoint"]["sha256"]
    ):
        raise ValueError("lineage CPU ORT parity receipt differs")
    dev = read_strict_json_object(resolved_root / expected_paths["dev_selection"])
    if content_sha256(dev) != payload["dev_selection_payload_sha256"]:
        raise ValueError("lineage DEV report differs")
    checkpoint = load_segmentation_checkpoint(
        resolved_root / expected_paths["selected_checkpoint"],
        expected_input_bindings=payload["input_bindings"],
        expected_training_config=payload["training_config"],
    )
    if (
        dev.get("schema_version") != DEV_REPORT_SCHEMA
        or dev.get("interpretation") != INTERPRETATION
        or dev.get("selection_metric_order") != list(SELECTION_METRICS)
        or checkpoint["epoch"] != dev.get("selected_epoch")
        or checkpoint["selection"]["Dice"] != dev.get("best_Dice")
        or checkpoint["selection"]["IoU"] != dev.get("best_IoU")
    ):
        raise ValueError("lineage selected DEV checkpoint differs")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "bce": 0.0, "dice": 0.0}
    samples = 0
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        losses = segmentation_loss(model(images), masks)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError("nose segmentation loss became non-finite")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        count = int(images.shape[0])
        samples += count
        for name in totals:
            totals[name] += float(losses[name].detach()) * count
    if samples <= 0:
        raise RuntimeError("nose segmentation training loader was empty")
    return {name: value / samples for name, value in totals.items()}


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    weighted = {"IoU": 0.0, "Dice": 0.0}
    samples = 0
    with torch.inference_mode():
        for images, masks, _ in loader:
            masks = masks.to(device, non_blocking=device.type == "cuda")
            probability = torch.sigmoid(
                model(images.to(device, non_blocking=device.type == "cuda"))
            )
            metrics = teacher_agreement_metrics(
                probability, masks, threshold=threshold
            )
            count = int(images.shape[0])
            samples += count
            for name in weighted:
                weighted[name] += metrics[name] * count
    if samples <= 0:
        raise RuntimeError("nose segmentation DEV loader was empty")
    return {name: value / samples for name, value in weighted.items()}


def _build_lineage(
    *,
    root: Path,
    input_bindings: Mapping[str, Any],
    training_config: Mapping[str, Any],
    onnx_bytes: int,
    parity: Mapping[str, Any],
    dev_report: Mapping[str, Any],
) -> dict[str, Any]:
    def artifact(name: str) -> dict[str, Any]:
        path = root / name
        return {"path": name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}

    lineage = {
        "schema_version": LINEAGE_SCHEMA,
        "artifacts": {
            "selected_checkpoint": artifact("selected_checkpoint.pt"),
            "onnx": artifact("nose_mask.onnx"),
            "runtime_manifest": artifact("nose_mask.runtime.json"),
            "parity_receipt": artifact("cpu_ort_parity.json"),
            "dev_selection": artifact("dev_selection.json"),
        },
        "onnx_contract": {
            "input_name": "images",
            "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
            "output_name": "mask",
            "output_shape": [1, 1, IMAGE_SIZE, IMAGE_SIZE],
            "output_semantics": "SIGMOID_PROBABILITY",
            "opset": 18,
            "external_data": False,
            "onnx_bytes": onnx_bytes,
        },
        "input_bindings": dict(input_bindings),
        "input_bindings_sha256": content_sha256(dict(input_bindings)),
        "training_config": dict(training_config),
        "training_config_sha256": content_sha256(dict(training_config)),
        "dev_selection_payload_sha256": content_sha256(dict(dev_report)),
        "parity_payload_sha256": parity["receipt_sha256"],
        "license": {
            "license_id": LICENSE_ID,
            "usage_lane": UsageLane.RESEARCH_ONLY.value,
        },
        "interpretation": INTERPRETATION,
    }
    lineage["lineage_sha256"] = content_sha256(lineage)
    return lineage


def _validate_input_bindings(payload: object, *, require_backbone: bool = False) -> None:
    expected = {
        "manifests",
        "split",
        "accepted_mask_count",
        "rejected_mask_count",
        "license",
    }
    if require_backbone:
        expected.add("backbone")
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("segmentation input binding keys differ")
    manifests = payload["manifests"]
    if not isinstance(manifests, dict) or set(manifests) != {"teacher", "source", "native"}:
        raise ValueError("segmentation manifest bindings differ")
    expected_manifest_keys = {
        "teacher": {"file_sha256", "payload_sha256"},
        "source": {
            "file_sha256",
            "payload_sha256",
            "source_receipt_file_sha256",
        },
        "native": {"file_sha256", "payload_sha256"},
    }
    for name, keys in expected_manifest_keys.items():
        if not isinstance(manifests[name], dict) or set(manifests[name]) != keys:
            raise ValueError(f"segmentation {name} manifest binding differs")
        for field, digest in manifests[name].items():
            _require_sha256(digest, f"{name}.{field}")
    split = payload["split"]
    split_keys = {
        "algorithm",
        "seed",
        "dev_fraction",
        "train_registered_dog_ids",
        "dev_registered_dog_ids",
        "train_sample_tokens",
        "dev_sample_tokens",
    }
    if not isinstance(split, dict) or set(split) != split_keys or split["algorithm"] != "SHA256_ASCENDING_SEED_COLON_REGISTERED_DOG_ID":
        raise ValueError("segmentation split binding differs")
    if isinstance(split["seed"], bool) or not isinstance(split["seed"], int) or not 0.0 < split["dev_fraction"] < 1.0:
        raise ValueError("segmentation split parameters differ")
    for name in ("train_registered_dog_ids", "dev_registered_dog_ids"):
        values = split[name]
        if not isinstance(values, list) or not values or values != sorted(set(values)):
            raise ValueError("segmentation split identity population differs")
        for value in values:
            parsed = uuid.UUID(value)
            if parsed.version != 5 or str(parsed) != value:
                raise ValueError("segmentation split identity must be canonical UUIDv5")
    if set(split["train_registered_dog_ids"]) & set(split["dev_registered_dog_ids"]):
        raise ValueError("segmentation TRAIN and DEV identities overlap")
    for name in ("train_sample_tokens", "dev_sample_tokens"):
        values = split[name]
        if not isinstance(values, list) or not values or values != sorted(set(values)):
            raise ValueError("segmentation split sample population differs")
        for value in values:
            _require_sha256(value, "segmentation sample token")
    accepted = payload["accepted_mask_count"]
    rejected = payload["rejected_mask_count"]
    if (
        isinstance(accepted, bool)
        or not isinstance(accepted, int)
        or accepted != len(split["train_sample_tokens"]) + len(split["dev_sample_tokens"])
        or isinstance(rejected, bool)
        or not isinstance(rejected, int)
        or rejected < 0
    ):
        raise ValueError("segmentation accepted/rejected counts differ")
    if payload["license"] != {
        "license_id": LICENSE_ID,
        "usage_lane": UsageLane.RESEARCH_ONLY.value,
        "reason": "Derived from the research-only native nose localization lane.",
    }:
        raise ValueError("segmentation input license binding differs")
    if require_backbone:
        backbone = payload["backbone"]
        if not isinstance(backbone, dict) or set(backbone) != {
            "model_name",
            "source_revision",
            "safetensors_sha256",
        } or backbone["model_name"] != MOBILENETV4_MODEL_NAME or backbone["source_revision"] != "c9f31ac64483d7f0590db9edccb4418392a96eea":
            raise ValueError("segmentation backbone binding differs")
        _require_sha256(backbone["safetensors_sha256"], "backbone safetensors")
        if backbone["safetensors_sha256"] != MOBILENETV4_WEIGHTS_SHA256:
            raise ValueError("segmentation backbone digest differs")


def _validate_training_config(config: object) -> None:
    expected = {
        "schema_version",
        "model",
        "input",
        "augmentation",
        "loss",
        "optimizer",
        "epochs",
        "batch_size",
        "num_workers",
        "seed",
        "dev_fraction",
        "mask_threshold",
        "device",
        "selection_metric_order",
        "parity_max_absolute_error",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("segmentation training config keys differ")
    if config["schema_version"] != "identification.nose.nose_segmentation_student.training_config.v1" or config["selection_metric_order"] != list(SELECTION_METRICS) or config["device"] not in {"cpu", "cuda"}:
        raise ValueError("segmentation training config contract differs")
    if (
        not isinstance(config["model"], dict)
        or set(config["model"])
        != {
            "backbone",
            "spatial_feature_channels",
            "decoder_channels",
            "decoder_fusion",
        }
        or config["model"].get("backbone") != MOBILENETV4_MODEL_NAME
        or config["model"].get("spatial_feature_channels") != 960
        or config["model"].get("decoder_fusion")
        != "STRIDE4_RGB_STEM_PLUS_UPSAMPLED_BACKBONE_FEATURES"
    ):
        raise ValueError("segmentation model config differs")
    if config["input"] != {
        "shape": [3, IMAGE_SIZE, IMAGE_SIZE],
        "crop": "EXACT_SOURCE_RGB_AND_BINARY_MASK_BY_NOSE_BOX_XYXY",
        "resize": "JOINT_DIRECT_BILINEAR_RGB_NEAREST_BINARY_MASK",
        "scale": 1.0 / 255.0,
        "mean": list(IMAGE_MEAN),
        "std": list(IMAGE_STD),
    } or config["augmentation"] != {"joint_horizontal_flip_probability": 0.5} or config["loss"] != {"bce_with_logits_weight": 1.0, "soft_dice_weight": 1.0}:
        raise ValueError("segmentation transform or loss config differs")
    model_channels = config["model"].get("decoder_channels")
    if isinstance(model_channels, bool) or not isinstance(model_channels, int) or model_channels <= 0:
        raise ValueError("segmentation decoder config differs")
    optimizer = config["optimizer"]
    if not isinstance(optimizer, dict) or set(optimizer) != {"type", "learning_rate", "weight_decay"} or optimizer["type"] != "AdamW":
        raise ValueError("segmentation optimizer config differs")
    for name in ("learning_rate", "weight_decay"):
        value = optimizer[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if name == "learning_rate" else value < 0):
            raise ValueError("segmentation optimizer value differs")
    for name in ("epochs", "batch_size"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"segmentation {name} must be positive")
    if isinstance(config["num_workers"], bool) or not isinstance(config["num_workers"], int) or config["num_workers"] < 0 or isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise ValueError("segmentation worker or seed config differs")
    for name in ("dev_fraction", "mask_threshold"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"segmentation {name} must be in (0,1)")
    parity = config["parity_max_absolute_error"]
    if isinstance(parity, bool) or not isinstance(parity, (int, float)) or not math.isfinite(parity) or parity <= 0:
        raise ValueError("segmentation parity threshold differs")


def _checkpoint_metadata_sha256(payload: Mapping[str, Any]) -> str:
    return content_sha256(
        {
            key: value
            for key, value in payload.items()
            if key not in {"model_state_dict", "checkpoint_payload_sha256"}
        }
    )


def _validate_parity_receipt(payload: object) -> None:
    expected = {
        "schema_version",
        "model_id",
        "artifact_sha256",
        "selected_checkpoint_sha256",
        "fixture",
        "reference_backend",
        "candidate_backend",
        "reference_output_sha256",
        "candidate_output_sha256",
        "maximum_absolute_error",
        "threshold",
        "finite",
        "range",
        "decision",
        "receipt_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("CPU ORT parity receipt keys differ")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload["schema_version"] != PARITY_SCHEMA
        or payload["model_id"] != MODEL_ID
        or payload["decision"] != "PASS"
        or payload["finite"] is not True
        or payload["range"] != [0.0, 1.0]
        or content_sha256(body) != payload["receipt_sha256"]
    ):
        raise ValueError("CPU ORT parity receipt contract differs")
    for name in (
        "artifact_sha256",
        "selected_checkpoint_sha256",
        "reference_output_sha256",
        "candidate_output_sha256",
        "receipt_sha256",
    ):
        _require_sha256(payload[name], f"parity {name}")
    fixture = payload["fixture"]
    if not isinstance(fixture, dict) or set(fixture) != {
        "sample_token",
        "source_sha256",
        "teacher_mask_sha256",
        "nose_box_xyxy",
    }:
        raise ValueError("CPU ORT parity fixture differs")
    for name in ("sample_token", "source_sha256", "teacher_mask_sha256"):
        _require_sha256(fixture[name], f"parity fixture {name}")
    box = fixture["nose_box_xyxy"]
    if not isinstance(box, list) or len(box) != 4 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in box
    ) or not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError("CPU ORT parity fixture box differs")
    for name in ("reference_backend", "candidate_backend"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise ValueError("CPU ORT parity backend differs")
    maximum = payload["maximum_absolute_error"]
    threshold = payload["threshold"]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
        for value in (maximum, threshold)
    ) or threshold <= 0.0 or maximum > threshold:
        raise ValueError("CPU ORT parity error threshold differs")


def _state_dict_sha256(state: object) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("model state must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise ValueError("model state must contain named tensors only")
        value = tensor.detach().cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError("model state contains non-finite tensors")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(",".join(str(item) for item in value.shape).encode("ascii") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _source_path(root: Path, row: Mapping[str, Any]) -> Path:
    return _bound_relative_file(
        root, row["source_image_path"], row["source_sha256"], "source image"
    )


def _bound_relative_file(
    root: Path, relative_value: object, expected_sha256: str, subject: str
) -> Path:
    if not isinstance(relative_value, str):
        raise ValueError(f"{subject} path differs")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or relative.as_posix() != relative_value or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{subject} path is unsafe")
    resolved_root = root.resolve(strict=True)
    path = resolved_root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{subject} path is unsafe")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{subject} path is unsafe")
    _read_bound_file(resolved, expected_sha256, subject)
    return resolved


def _read_bound_file(path: Path, expected_sha256: str | None, subject: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{subject} must be a regular non-symlink file")
    payload = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{subject} SHA-256 differs")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_read_bound_file(path, None, "hash input")).hexdigest()


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")


def _validate_probability_output(output: np.ndarray, context: str) -> None:
    if (
        not isinstance(output, np.ndarray)
        or output.dtype != np.float32
        or output.shape != (1, 1, IMAGE_SIZE, IMAGE_SIZE)
        or not np.isfinite(output).all()
        or np.any((output < 0.0) | (output > 1.0))
    ):
        raise ValueError(
            f"{context} output must be finite float32 [1,1,224,224] in [0,1]"
        )


def _onnx_shape(value_info: object) -> tuple[int, ...]:
    dimensions = []
    for dimension in value_info.type.tensor_type.shape.dim:  # type: ignore[attr-defined]
        if dimension.dim_value <= 0:
            raise ValueError("nose segmentation ONNX dimensions must be static")
        dimensions.append(int(dimension.dim_value))
    return tuple(dimensions)


def _validate_output_destination(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("output directory must be absolute")
    if os.path.lexists(path):
        raise FileExistsError("nose segmentation output directory must not exist")
    parent = path.parent.resolve(strict=True)
    repository_root = Path(__file__).resolve().parents[2]
    if parent.is_relative_to(repository_root):
        raise ValueError("nose segmentation output must be outside the Git worktree")


__all__ = [
    "CHECKPOINT_SCHEMA",
    "IMAGE_SIZE",
    "INTERPRETATION",
    "LICENSE_ID",
    "MODEL_ID",
    "NoseSegmentationDataset",
    "NoseSegmentationStudent",
    "SegmentationRecord",
    "build_runtime_manifest",
    "build_segmentation_checkpoint",
    "build_training_config",
    "deterministic_identity_split",
    "export_static_onnx",
    "load_mobilenetv4_student",
    "load_segmentation_checkpoint",
    "load_training_records",
    "produce_cpu_ort_parity_receipt",
    "save_segmentation_checkpoint",
    "segmentation_loss",
    "teacher_agreement_metrics",
    "train_and_export",
    "validate_lineage_manifest",
    "validate_segmentation_checkpoint",
    "validate_static_onnx",
]
