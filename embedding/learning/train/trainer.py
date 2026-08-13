from __future__ import annotations

import copy
import io
import json
import math
import os
import random
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from data.public.public_crop_manifest import (
    PublicCropArtifact,
    PublicCropManifest,
    read_verified_crop_artifact,
)
from identity.exposure.role_exposure import RoleExposureLedger, RoleExposureReceipt
from identity.admission.training_admission import (
    TrainingAdmissionManifest,
    TrainingAdmissionReceipt,
    TrainingCropRow,
    verify_training_admission_receipt,
)
from embedding.learning.train.augment import RandAugment
from embedding.learning.train.config import (
    TrainConfig,  # noqa: F401 — canonical, backward compat
)

# ---------------------------------------------------------------------------
# Image cache — preload all crops into RAM
# ---------------------------------------------------------------------------


class ImageCache:
    """Preload admitted crop images into a contiguous CHW uint8 array.

    Eliminates 9P filesystem bottleneck during training by decoding and
    resizing all crops once at init.  Normalization is deferred to GPU
    transfer time.  Memory: N x 3 x 224 x 224 bytes (≈ 4.2 GiB for 28K crops).
    """

    _NORM_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _NORM_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        crop_root: Path,
        samples: list[tuple[PublicCropArtifact, int]],
        size: tuple[int, int] = (224, 224),
    ) -> None:
        from PIL import Image
        n = len(samples)
        self._data = np.zeros((n, 3, size[0], size[1]), dtype=np.uint8)
        self._labels = np.array([label for _, label in samples], dtype=np.int64)
        for i, (artifact, _) in enumerate(samples):
            payload = read_verified_crop_artifact(crop_root, artifact)
            with Image.open(io.BytesIO(payload)) as image:
                with image.convert("RGB") as rgb:
                    resized = rgb.resize(size, Image.BILINEAR)
                    arr = np.array(resized, dtype=np.uint8)
            self._data[i] = np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self._data[index]), int(self._labels[index])

    @staticmethod
    def gpu_normalize(images: torch.Tensor,
                       mean: torch.Tensor | None = None,
                       std: torch.Tensor | None = None) -> torch.Tensor:
        if mean is None:
            mean = torch.tensor(ImageCache._NORM_MEAN, device=images.device).view(1, 3, 1, 1)
        if std is None:
            std = torch.tensor(ImageCache._NORM_STD, device=images.device).view(1, 3, 1, 1)
        return images.float().div_(255.0).sub_(mean).div_(std)


# ---------------------------------------------------------------------------
# Crop dataset
# ---------------------------------------------------------------------------


class AdmittedCropDataset(Dataset):
    """Read exactly the immutable crop artifacts named by admitted rows.

    Each sample is a (224x224 RGB tensor, label_index) pair.
    When *use_cache* is True (default), all images are preloaded into a
    contiguous uint8 RAM array at init, eliminating filesystem I/O during
    training.
    """

    def __init__(
        self,
        crop_root: Path,
        rows: tuple[TrainingCropRow, ...],
        crop_manifest: PublicCropManifest,
        label_to_index: dict[str, int],
        *,
        use_cache: bool = True,
    ) -> None:
        artifacts_by_sample = {
            artifact.sample_token: artifact for artifact in crop_manifest.artifacts
        }
        self._crop_root = crop_root
        self._samples: list[tuple[PublicCropArtifact, int]] = []
        for row in rows:
            try:
                artifact = artifacts_by_sample[row.sample_token]
                label_idx = label_to_index[row.public_subject_token]
            except KeyError as error:
                raise ValueError("admitted dataset row is not exactly bound") from error
            if (
                artifact.public_subject_token != row.public_subject_token
                or artifact.relative_path != row.crop_relative_path
                or artifact.artifact_sha256 != row.crop_artifact_sha256
            ):
                raise ValueError("admitted dataset row crop binding differs")
            self._samples.append((artifact, label_idx))

        self._cache: ImageCache | None = None
        if use_cache and self._samples:
            self._cache = ImageCache(crop_root, self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def labels(self) -> tuple[int, ...]:
        """Return sampler labels without reading or decoding crop bytes."""

        return tuple(label for _, label in self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self._cache is not None:
            return self._cache[index]
        artifact, label = self._samples[index]
        from PIL import Image
        payload = read_verified_crop_artifact(self._crop_root, artifact)
        with Image.open(io.BytesIO(payload)) as image:
            with image.convert("RGB") as rgb:
                resized = rgb.resize((224, 224), Image.BILINEAR)
                arr = np.array(resized, dtype=np.uint8)
        tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)))
        return tensor, label


# ---------------------------------------------------------------------------
# Balanced sampler
# ---------------------------------------------------------------------------


class IdentityBalancedSampler(Sampler):
    """Yield balanced batches: one sample per identity per batch step.

    Shuffles identity order and sample order within each identity each epoch.
    """

    def __init__(
        self,
        labels: list[int],
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> None:
        if isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not labels:
            raise ValueError("sampler requires at least one sample")
        self._batch_size = batch_size
        self._generator = generator
        label_to_indices: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            label_to_indices.setdefault(lab, []).append(idx)
        self._label_to_indices = label_to_indices
        self._identity_ids = sorted(label_to_indices.keys())
        self._num_identities = len(self._identity_ids)

    def __iter__(self):
        g = self._generator if self._generator is not None else torch.default_generator
        identity_ties = torch.randperm(self._num_identities, generator=g).tolist()
        tie_rank = {self._identity_ids[pos]: rank for rank, pos in enumerate(identity_ties)}
        identities = sorted(
            self._identity_ids,
            key=lambda identity: (
                -len(self._label_to_indices[identity]),
                tie_rank[identity],
            ),
        )
        batches: list[list[int]] = [[] for _ in range(len(self))]
        batch_ties = torch.randperm(len(batches), generator=g).tolist()
        batch_rank = {batch: rank for rank, batch in enumerate(batch_ties)}
        for identity in identities:
            candidates = self._label_to_indices[identity]
            permutation = torch.randperm(len(candidates), generator=g).tolist()
            available = sorted(
                range(len(batches)),
                key=lambda batch: (len(batches[batch]), batch_rank[batch]),
            )
            selected = available[:len(candidates)]
            if any(len(batches[batch]) >= self._batch_size for batch in selected):
                raise RuntimeError("identity-balanced batch schedule is infeasible")
            for sample_position, batch in zip(permutation, selected):
                batches[batch].append(candidates[sample_position])
        order = torch.randperm(len(batches), generator=g).tolist()
        for batch in order:
            if not batches[batch]:
                raise RuntimeError("identity-balanced sampler created an empty batch")
            yield batches[batch]

    def __len__(self) -> int:
        total = sum(len(indices) for indices in self._label_to_indices.values())
        largest_identity = max(len(indices) for indices in self._label_to_indices.values())
        return max(largest_identity, math.ceil(total / self._batch_size))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Dinov2Embedding(nn.Module):
    """DINOv2-small backbone producing an L2-normalized embedding.

    Supports gradient checkpointing via *use_gradient_checkpointing*.
    """

    def __init__(self, embedding_dim: int = 384,
                 use_gradient_checkpointing: bool = False,
                 model_directory: Path | None = None) -> None:
        super().__init__()
        from transformers import AutoModel
        source = (
            str(model_directory)
            if model_directory is not None
            else "facebook/dinov2-small"
        )
        self._backbone = AutoModel.from_pretrained(
            source,
            attn_implementation="sdpa",
            local_files_only=model_directory is not None,
        )
        self._embedding_dim = embedding_dim
        if use_gradient_checkpointing:
            self._backbone.gradient_checkpointing_enable()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._backbone(x)
        emb = out.last_hidden_state[:, 0, :]
        return F.normalize(emb, p=2, dim=1)


class ConvNeXtEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 768,
                 use_gradient_checkpointing: bool = False,
                 model_directory: Path | None = None) -> None:
        super().__init__()
        from transformers import AutoModel
        source = (
            str(model_directory)
            if model_directory is not None
            else "facebook/convnext-base-224"
        )
        self._backbone = AutoModel.from_pretrained(
            source, local_files_only=model_directory is not None
        )
        hidden = self._backbone.config.hidden_sizes[-1]
        if embedding_dim != hidden:
            self._project = nn.Linear(hidden, embedding_dim)
        else:
            self._project = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._backbone(x)
        convnext_out = out.last_hidden_state
        emb = convnext_out.mean(dim=[-1, -2])
        emb = self._project(emb)
        return F.normalize(emb, p=2, dim=1)


class AppearanceBoundedResidual(nn.Module):
    """Frozen A0 embedding plus a hard norm-bounded trainable residual."""

    def __init__(self, baseline: nn.Module, dimension: int, scale: float) -> None:
        super().__init__()
        self.baseline = baseline
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.baseline.eval()
        self.adapter = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.scale = scale

    def train(self, mode: bool = True) -> AppearanceBoundedResidual:
        super().train(mode)
        self.baseline.eval()
        return self

    def baseline_embedding(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.baseline(images)

    def embedding_from_baseline(self, baseline: torch.Tensor) -> torch.Tensor:
        residual = self.adapter(baseline)
        residual = residual / torch.linalg.vector_norm(
            residual, dim=1, keepdim=True
        ).clamp_min(1.0)
        return F.normalize(baseline + self.scale * residual, p=2, dim=1)

    def forward_with_baseline(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        baseline = self.baseline_embedding(images)
        return self.embedding_from_baseline(baseline), baseline

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.forward_with_baseline(images)
        return embedding


class ArcFaceHead(nn.Module):
    """ArcFace classification head with cosine margin.

    W is L2-normalized; input features are L2-normalized before the dot product.
    """

    def __init__(
        self, embedding_dim: int, num_classes: int, scale: float, margin: float
    ) -> None:
        super().__init__()
        self._scale = scale
        self._margin = margin
        self._w = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_normal_(self._w)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self._w, p=2, dim=1)
        cos_theta = F.linear(features, w).clamp(-1, 1)
        sin_theta = torch.sqrt((1.0 - cos_theta.pow(2)).clamp(min=1e-12))
        cos_m = math.cos(self._margin)
        sin_m = math.sin(self._margin)
        phi = cos_theta * cos_m - sin_theta * sin_m
        one_hot = F.one_hot(labels, num_classes=self._w.size(0)).float()
        logits = one_hot * phi + (1.0 - one_hot) * cos_theta
        return logits * self._scale


class ArcFaceModel(nn.Module):
    """Generic backbone + ArcFace head.  Backbone is a callable factory."""

    def __init__(self, config: TrainConfig,
                  backbone_factory: Callable[..., nn.Module] | None = None) -> None:
        super().__init__()
        self._config = config
        if backbone_factory is None:
            factories = {
                "dinov2-small": Dinov2Embedding,
                "convnext-base": ConvNeXtEmbedding,
            }
            try:
                backbone_factory = factories[config.model_name]
            except KeyError as exc:
                raise ValueError(
                    f"unsupported training backbone {config.model_name!r}"
                ) from exc
        backbone = backbone_factory(
            config.embedding_dim,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
        self._backbone = (
            AppearanceBoundedResidual(
                backbone, config.embedding_dim, config.residual_scale
            )
            if config.architecture == "appearance_bounded_residual_v4"
            else backbone
        )
        self._head = ArcFaceHead(
            config.embedding_dim,
            config.num_classes,
            config.arcface_scale,
            config.arcface_margin,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings for inference and retrieval."""
        return self._backbone(images)

    def forward_train(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Return metric-learning class logits regardless of module mode."""
        return self._head(self.encode(images), labels)

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        if labels is not None and self.training:
            return self.forward_train(images, labels)
        return self.encode(images)

    @torch.no_grad()
    def extract_embedding(self, images: torch.Tensor) -> np.ndarray:
        self.eval()
        emb = self.encode(images)
        return emb.cpu().numpy()

    def export_to_onnx(self, output_path: Path) -> None:
        self.eval()
        device = next(self._backbone.parameters()).device
        dummy = torch.randn(3, 3, 224, 224, device=device)
        batch = torch.export.Dim("batch")
        torch.onnx.export(
            self._backbone,
            (dummy,),
            str(output_path),
            input_names=["images"],
            output_names=["embedding"],
            dynamo=True,
            dynamic_shapes=({0: batch},),
            opset_version=18,
            external_data=False,
        )


class Dinov2ArcFaceModel(ArcFaceModel):
    """DINOv2-small backbone + ArcFace head."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__(config, backbone_factory=Dinov2Embedding)


# ---------------------------------------------------------------------------
# FLOPs / throughput estimation
# ---------------------------------------------------------------------------


_FLOPS_BY_MODEL: dict[str, int] = {
    "dinov2-small": 4_600_000_000,
    "convnext-base": 15_000_000_000,
}


def estimate_flops(config: TrainConfig, num_samples: int) -> dict[str, Any]:
    flops_per_sample = _FLOPS_BY_MODEL.get(
        config.model_name, 4_600_000_000
    )
    steps_per_epoch = math.ceil(num_samples / config.batch_size)
    flops_per_step = config.batch_size * flops_per_sample * 3
    total_flops = flops_per_step * steps_per_epoch * config.epochs
    return {
        "model": config.model_name,
        "flops_per_sample_fwd": flops_per_sample,
        "batch_size": config.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "flops_per_step_fwd_bwd": flops_per_step,
        "total_flops_estimate": total_flops,
        "note": "FLOPs are approximate; real throughput depends on data I/O, driver, and GPU load.",
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _build_label_index(rows: tuple[TrainingCropRow, ...]) -> dict[str, int]:
    unique_labels = sorted({row.public_subject_token for row in rows})
    return {label: idx for idx, label in enumerate(unique_labels)}


def _unwrap_model(model: nn.Module) -> ArcFaceModel:
    candidate = getattr(model, "_orig_mod", model)
    if not isinstance(candidate, ArcFaceModel):
        raise TypeError("training model is not an ArcFaceModel")
    return candidate


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    config: TrainConfig,
    label_to_index: dict[str, int],
    epoch: int,
    global_step: int,
    selection_metric: dict[str, float] | None,
    admission_receipt: TrainingAdmissionReceipt | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "cvi.training_checkpoint.v1",
        "architecture": {
            "architecture": config.architecture,
            "model_name": config.model_name,
            "embedding_dim": config.embedding_dim,
            "num_classes": config.num_classes,
            "loss_type": config.loss_type,
        },
        "config": config.to_dict(),
        "label_to_index": dict(sorted(label_to_index.items())),
        "epoch": epoch,
        "global_step": global_step,
        "selection_metric": selection_metric,
        "training_admission": (
            None
            if admission_receipt is None
            else {
                "receipt_sha256": admission_receipt.receipt_sha256,
                "receipt": admission_receipt.to_dict(),
            }
        ),
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": np.random.get_state()[0],
                "keys": [int(value) for value in np.random.get_state()[1]],
                "position": int(np.random.get_state()[2]),
                "has_gauss": int(np.random.get_state()[3]),
                "cached_gaussian": float(np.random.get_state()[4]),
            },
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "preprocessing": {
            "input_shape": ["batch", 3, 224, 224],
            "color_mode": "RGB",
            "resize": "bilinear_stretch_224x224",
            "scale": 1.0 / 255.0,
            "mean": ImageCache._NORM_MEAN.tolist(),
            "std": ImageCache._NORM_STD.tolist(),
            "dtype": "float32",
        },
    }


@torch.no_grad()
def _evaluate_development_retrieval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm_mean: torch.Tensor | None,
    norm_std: torch.Tensor | None,
) -> dict[str, float]:
    model.eval()
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if images.dtype == torch.uint8:
            images = ImageCache.gpu_normalize(images, norm_mean, norm_std)
        else:
            images = images.to(dtype=torch.float32)
        embeddings = _unwrap_model(model).encode(images)
        if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
            raise RuntimeError("development embeddings are invalid")
        all_embeddings.append(embeddings.detach().cpu())
        all_labels.append(labels.detach().cpu())
    if not all_embeddings:
        raise RuntimeError("development retrieval loader is empty")
    embeddings = F.normalize(torch.cat(all_embeddings, dim=0), p=2, dim=1)
    labels = torch.cat(all_labels, dim=0)
    unique, counts = torch.unique(labels, return_counts=True)
    if unique.numel() < 2 or torch.any(counts < 2):
        raise RuntimeError(
            "development retrieval requires at least two identities and two "
            "samples per identity"
        )
    rank1_hits = 0
    average_precisions: list[float] = []
    for query_idx in range(embeddings.shape[0]):
        scores = embeddings @ embeddings[query_idx]
        candidate_mask = torch.arange(embeddings.shape[0]) != query_idx
        order = torch.argsort(
            scores[candidate_mask], descending=True, stable=True
        )
        candidate_labels = labels[candidate_mask]
        relevant = candidate_labels[order] == labels[query_idx]
        rank1_hits += int(relevant[0].item())
        ranks = torch.nonzero(relevant, as_tuple=False).flatten() + 1
        precision = torch.arange(1, ranks.numel() + 1, dtype=torch.float64) / ranks
        average_precisions.append(float(precision.mean().item()))
    count = embeddings.shape[0]
    return {
        "rank1": rank1_hits / count,
        "map": float(np.mean(average_precisions)),
        "queries": float(count),
        "identities": float(unique.numel()),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _warmup_cosine_schedule(
    epoch: int, total_epochs: int, warmup: int, lr_max: float, lr_min: float
) -> float:
    if epoch < warmup:
        return lr_max * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def _selection_improves(
    candidate: dict[str, float], incumbent: dict[str, float]
) -> bool:
    """Require a strict Pareto improvement without regressing Rank-1 or mAP."""

    return (
        candidate["rank1"] >= incumbent["rank1"]
        and candidate["map"] >= incumbent["map"]
        and (
            candidate["rank1"] > incumbent["rank1"]
            or candidate["map"] > incumbent["map"]
        )
    )


def _metric_learning_loss(
    model: nn.Module,
    unwrapped_model: ArcFaceModel,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float,
    frozen_backbone: nn.Module | None,
    consistency_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = model(images)
    logits = unwrapped_model._head(features, labels)
    classification = F.cross_entropy(
        logits, labels, label_smoothing=label_smoothing
    )
    consistency = classification.new_zeros(())
    if frozen_backbone is not None:
        with torch.no_grad():
            reference = frozen_backbone(images)
        consistency = 1.0 - F.cosine_similarity(
            features.float(), reference.float(), dim=1
        ).mean()
    return classification + consistency_weight * consistency, classification, consistency


def _border_challenge(images: torch.Tensor) -> torch.Tensor:
    """Corrupt the crop border while preserving its central region."""

    if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
        raise ValueError("A4 border challenge requires [B,3,224,224]")
    blurred = F.avg_pool2d(images, kernel_size=15, stride=1, padding=7)
    median = images.flatten(2).median(dim=2).values[:, :, None, None]
    noise = torch.randn_like(images) * 0.03
    challenged = (0.55 * blurred + 0.45 * median + noise).clamp(0.0, 1.0)
    ring = torch.ones_like(images[:, :1], dtype=torch.bool)
    ring[:, :, 40:-40, 40:-40] = False
    return torch.where(ring, challenged, images)


def _a4_metric_learning_loss(
    model: nn.Module,
    unwrapped_model: ArcFaceModel,
    clean_images: torch.Tensor,
    challenged_images: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float,
    consistency_weight: float,
    anchor_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(unwrapped_model._backbone, AppearanceBoundedResidual):
        raise TypeError("A4 loss requires AppearanceBoundedResidual")
    backbone = unwrapped_model._backbone
    clean, clean_baseline = backbone.forward_with_baseline(clean_images)
    challenged, challenged_baseline = backbone.forward_with_baseline(
        challenged_images
    )
    clean_logits = unwrapped_model._head(clean, labels)
    challenged_logits = unwrapped_model._head(challenged, labels)
    classification = 0.5 * (
        F.cross_entropy(clean_logits, labels, label_smoothing=label_smoothing)
        + F.cross_entropy(challenged_logits, labels, label_smoothing=label_smoothing)
    )
    consistency = 1.0 - F.cosine_similarity(clean, challenged, dim=1).mean()
    anchor = 0.5 * (
        1.0 - F.cosine_similarity(clean, clean_baseline, dim=1).mean()
        + 1.0
        - F.cosine_similarity(challenged, challenged_baseline, dim=1).mean()
    )
    total = (
        classification + consistency_weight * consistency + anchor_weight * anchor
    )
    return total, classification, consistency, anchor


def _build_dataloader(
    dataset: Dataset,
    sampler: Sampler | None,
    config: TrainConfig,
    device: torch.device,
    shuffle: bool = False,
) -> DataLoader:
    num_workers = config.num_workers
    if config.preload_images and hasattr(dataset, "_cache") and dataset._cache is not None:
        num_workers = 0
    return DataLoader(
        dataset,
        batch_sampler=sampler if sampler is not None else None,
        batch_size=config.batch_size if sampler is None else 1,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def _prepare_training_images(
    images: torch.Tensor,
    augment: RandAugment,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
) -> torch.Tensor:
    if images.dtype != torch.uint8:
        raise TypeError("training dataset images must use uint8 CHW tensors")
    images = images.float().div_(255.0)
    for index in range(images.shape[0]):
        images[index] = augment(images[index])
    return images.sub_(norm_mean).div_(norm_std)


def _prepare_a4_training_images(
    images: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.dtype != torch.uint8:
        raise TypeError("A4 training dataset images must use uint8 CHW tensors")
    clean = images.float().div_(255.0)
    challenged = _border_challenge(clean)
    return (
        (clean - norm_mean) / norm_std,
        (challenged - norm_mean) / norm_std,
    )


def evaluate_pretrained_development(
    config: TrainConfig,
    crop_root: Path,
    admission_manifest: TrainingAdmissionManifest,
    crop_manifest: PublicCropManifest,
    admission_receipt: TrainingAdmissionReceipt,
    *,
    exposure_ledger: RoleExposureLedger,
    exposure_receipt: RoleExposureReceipt,
    expected_admission_manifest_sha256: str,
    expected_admission_receipt_sha256: str,
    expected_split_receipt_sha256: str,
    expected_crop_receipt_sha256: str,
    expected_exposure_receipt_sha256: str,
    expected_model_receipt_sha256: str,
    model_artifact_verifier: Callable[[], None],
    device: torch.device = torch.device("cpu"),
    backbone_factory: Callable[..., nn.Module] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen pretrained backbone on the admitted development set."""
    verify_training_admission_receipt(
        admission_manifest,
        crop_manifest,
        admission_receipt,
        crop_root=crop_root,
        exposure_ledger=exposure_ledger,
        exposure_receipt=exposure_receipt,
        expected_admission_manifest_sha256=expected_admission_manifest_sha256,
        expected_admission_receipt_sha256=expected_admission_receipt_sha256,
        expected_split_receipt_sha256=expected_split_receipt_sha256,
        expected_crop_receipt_sha256=expected_crop_receipt_sha256,
        expected_exposure_receipt_sha256=expected_exposure_receipt_sha256,
        expected_model_receipt_sha256=expected_model_receipt_sha256,
    )
    if not callable(model_artifact_verifier):
        raise TypeError("model_artifact_verifier must be callable")

    train_rows = tuple(
        row for row in admission_manifest.rows if row.lane == "MODEL_TRAINING"
    )
    development_rows = tuple(
        row for row in admission_manifest.rows if row.lane == "MODEL_SELECTION"
    )
    if not train_rows or not development_rows:
        raise ValueError("exact train and development rows are required")
    train_labels = _build_label_index(train_rows)
    development_labels = _build_label_index(development_rows)
    if set(train_labels) & set(development_labels):
        raise ValueError("training and development public subjects must be disjoint")
    development_counts = Counter(
        row.public_subject_token for row in development_rows
    )
    if len(development_counts) < 2 or min(development_counts.values()) < 2:
        raise ValueError(
            "development rows require at least two public subjects and two "
            "samples per subject"
        )

    model_artifact_verifier()
    config = TrainConfig(**{**config.to_dict(), "num_classes": len(train_labels)})
    dataset = AdmittedCropDataset(
        crop_root,
        development_rows,
        crop_manifest,
        development_labels,
        use_cache=config.preload_images,
    )
    loader = _build_dataloader(dataset, None, config, device, shuffle=False)
    model = ArcFaceModel(config, backbone_factory=backbone_factory).to(device)
    norm_mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)
    norm_std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)
    metrics = _evaluate_development_retrieval(
        model, loader, device, norm_mean, norm_std
    )
    return {
        "config": config.to_dict(),
        "development_metrics": metrics,
        "parameters": _count_parameters(model),
        "training_admission": {
            "manifest_sha256": admission_manifest.manifest_sha256,
            "receipt_sha256": admission_receipt.receipt_sha256,
        },
        "interpretation": "FROZEN_PRETRAINED_DEVELOPMENT_BASELINE_ONLY",
    }


def train_model(
    config: TrainConfig,
    crop_root: Path,
    admission_manifest: TrainingAdmissionManifest,
    crop_manifest: PublicCropManifest,
    admission_receipt: TrainingAdmissionReceipt,
    *,
    exposure_ledger: RoleExposureLedger,
    exposure_receipt: RoleExposureReceipt,
    output_directory: Path,
    expected_admission_manifest_sha256: str,
    expected_admission_receipt_sha256: str,
    expected_split_receipt_sha256: str,
    expected_crop_receipt_sha256: str,
    expected_exposure_receipt_sha256: str,
    expected_model_receipt_sha256: str,
    model_artifact_verifier: Callable[[], None],
    device: torch.device = torch.device("cpu"),
    backbone_factory: Callable[..., nn.Module] | None = None,
) -> dict[str, Any]:
    """Run receipt-bound ArcFace training on immutable public crops.

    *backbone_factory* is a callable(embedding_dim, use_gradient_checkpointing) → nn.Module.
    Defaults to Dinov2Embedding.
    """
    if output_directory.is_symlink() or os.path.lexists(output_directory):
        raise FileExistsError("training output directory must not exist")
    checkpoint_dir = output_directory / "checkpoints"
    if Path(config.checkpoint_dir) != checkpoint_dir:
        raise ValueError("checkpoint_dir must be the output directory checkpoints path")
    verified_admission = verify_training_admission_receipt(
        admission_manifest,
        crop_manifest,
        admission_receipt,
        crop_root=crop_root,
        exposure_ledger=exposure_ledger,
        exposure_receipt=exposure_receipt,
        expected_admission_manifest_sha256=expected_admission_manifest_sha256,
        expected_admission_receipt_sha256=expected_admission_receipt_sha256,
        expected_split_receipt_sha256=expected_split_receipt_sha256,
        expected_crop_receipt_sha256=expected_crop_receipt_sha256,
        expected_exposure_receipt_sha256=expected_exposure_receipt_sha256,
        expected_model_receipt_sha256=expected_model_receipt_sha256,
    )
    if not callable(model_artifact_verifier):
        raise TypeError("model_artifact_verifier must be callable")

    train_rows = tuple(
        row for row in admission_manifest.rows if row.lane == "MODEL_TRAINING"
    )
    development_rows = tuple(
        row for row in admission_manifest.rows if row.lane == "MODEL_SELECTION"
    )
    if not train_rows or not development_rows:
        raise ValueError("exact train and development rows are required")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    label_to_index = _build_label_index(train_rows)
    development_label_to_index = _build_label_index(development_rows)
    overlap = set(label_to_index) & set(development_label_to_index)
    if overlap:
        raise ValueError("training and development public subjects must be disjoint")
    development_counts = Counter(
        row.public_subject_token for row in development_rows
    )
    if len(development_counts) < 2 or min(development_counts.values()) < 2:
        raise ValueError(
            "development rows require at least two public subjects and two "
            "samples per subject"
        )
    config = TrainConfig(
        **{**config.to_dict(), "num_classes": len(label_to_index)}
    )

    train_dataset = AdmittedCropDataset(
        crop_root, train_rows, crop_manifest, label_to_index,
        use_cache=config.preload_images,
    )
    if len(train_dataset) == 0:
        raise ValueError("training crop dataset is empty")
    train_labels = list(train_dataset.labels)
    sampler = IdentityBalancedSampler(
        train_labels, config.batch_size,
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = _build_dataloader(train_dataset, sampler, config, device)

    val_dataset = AdmittedCropDataset(
        crop_root, development_rows, crop_manifest, development_label_to_index,
        use_cache=config.preload_images,
    )
    if len(val_dataset) == 0:
        raise ValueError("development crop dataset is empty")
    val_loader = _build_dataloader(
        val_dataset, None, config, device, shuffle=False
    )

    model_artifact_verifier()
    output_directory.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir()
    model = ArcFaceModel(config, backbone_factory=backbone_factory).to(device)
    frozen_backbone = None
    if (
        config.embedding_consistency_weight > 0.0
        and config.architecture == "standard_arcface"
    ):
        frozen_backbone = copy.deepcopy(model._backbone).to(device).eval()
        for parameter in frozen_backbone.parameters():
            parameter.requires_grad_(False)

    augment = RandAugment(n=2, m=9)

    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")

    unwrapped_model = _unwrap_model(model)
    if isinstance(unwrapped_model._backbone, AppearanceBoundedResidual):
        parameter_groups = (
            {
                "params": unwrapped_model._backbone.adapter.parameters(),
                "lr": config.lr,
                "lr_scale": 1.0,
            },
            {
                "params": unwrapped_model._head.parameters(),
                "lr": config.lr,
                "lr_scale": 1.0,
            },
        )
    else:
        parameter_groups = (
            {
                "params": unwrapped_model._backbone.parameters(),
                "lr": config.lr * config.backbone_lr_scale,
                "lr_scale": config.backbone_lr_scale,
            },
            {
                "params": unwrapped_model._head.parameters(),
                "lr": config.lr,
                "lr_scale": 1.0,
            },
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=config.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and config.mixed_precision else None
    history: list[dict[str, Any]] = []
    global_step = 0
    processed_samples = 0
    t0 = time.time()
    norm_mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)
    norm_std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)
    pretrained_metrics = _evaluate_development_retrieval(
        model, val_loader, device, norm_mean, norm_std
    )
    best_metrics = pretrained_metrics
    best_checkpoint = str(checkpoint_dir / "best_model.pt")
    torch.save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            label_to_index=label_to_index,
            epoch=0,
            global_step=0,
            selection_metric=pretrained_metrics,
            admission_receipt=verified_admission,
        ),
        best_checkpoint,
    )
    selected_epoch = 0
    epochs_without_improvement = 0
    completed_epochs = 0
    early_stopped = False
    val_metrics = pretrained_metrics

    for epoch in range(config.epochs):
        lr = _warmup_cosine_schedule(
            epoch, config.epochs, config.warmup_epochs, config.lr, config.lr_min
        )
        if isinstance(unwrapped_model._backbone, AppearanceBoundedResidual):
            backbone_trainable = False
            for parameter in unwrapped_model._backbone.baseline.parameters():
                parameter.requires_grad_(False)
            for parameter in unwrapped_model._backbone.adapter.parameters():
                parameter.requires_grad_(True)
        else:
            backbone_trainable = epoch >= config.freeze_backbone_epochs
            for parameter in unwrapped_model._backbone.parameters():
                parameter.requires_grad_(backbone_trainable)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * param_group["lr_scale"]

        model.train()
        if isinstance(unwrapped_model._backbone, AppearanceBoundedResidual):
            unwrapped_model._backbone.baseline.eval()
        elif not backbone_trainable:
            unwrapped_model._backbone.eval()
        epoch_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_consistency_loss = 0.0
        epoch_anchor_loss = 0.0
        epoch_steps = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            challenged_images = None
            if config.architecture == "appearance_bounded_residual_v4":
                images, challenged_images = _prepare_a4_training_images(
                    images, norm_mean, norm_std
                )
            else:
                images = _prepare_training_images(
                    images, augment, norm_mean, norm_std
                )

            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    if challenged_images is not None:
                        loss, classification_loss, consistency_loss, anchor_loss = (
                            _a4_metric_learning_loss(
                                model,
                                unwrapped_model,
                                images,
                                challenged_images,
                                labels,
                                label_smoothing=config.label_smoothing,
                                consistency_weight=config.border_consistency_weight,
                                anchor_weight=config.baseline_anchor_weight,
                            )
                        )
                    else:
                        loss, classification_loss, consistency_loss = (
                            _metric_learning_loss(
                                model,
                                unwrapped_model,
                                images,
                                labels,
                                label_smoothing=config.label_smoothing,
                                frozen_backbone=frozen_backbone,
                                consistency_weight=(
                                    config.embedding_consistency_weight
                                ),
                            )
                        )
                        anchor_loss = loss.new_zeros(())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if challenged_images is not None:
                    loss, classification_loss, consistency_loss, anchor_loss = (
                        _a4_metric_learning_loss(
                            model,
                            unwrapped_model,
                            images,
                            challenged_images,
                            labels,
                            label_smoothing=config.label_smoothing,
                            consistency_weight=config.border_consistency_weight,
                            anchor_weight=config.baseline_anchor_weight,
                        )
                    )
                else:
                    loss, classification_loss, consistency_loss = (
                        _metric_learning_loss(
                            model,
                            unwrapped_model,
                            images,
                            labels,
                            label_smoothing=config.label_smoothing,
                            frozen_backbone=frozen_backbone,
                            consistency_weight=config.embedding_consistency_weight,
                        )
                    )
                    anchor_loss = loss.new_zeros(())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()

            epoch_loss += loss.item()
            epoch_classification_loss += classification_loss.item()
            epoch_consistency_loss += consistency_loss.item()
            epoch_anchor_loss += anchor_loss.item()
            epoch_steps += 1
            global_step += 1
            processed_samples += int(labels.shape[0])

            if global_step % config.log_interval == 0:
                used = torch.cuda.max_memory_allocated(device) // 2**20 if device.type == "cuda" else 0
                print(
                    json.dumps({
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": round(loss.item(), 4),
                        "lr": round(lr, 8),
                        "elapsed_s": round(time.time() - t0, 1),
                        "vram_mib": used,
                    }),
                    flush=True,
                )

        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        avg_classification_loss = epoch_classification_loss / max(epoch_steps, 1)
        avg_consistency_loss = epoch_consistency_loss / max(epoch_steps, 1)
        avg_anchor_loss = epoch_anchor_loss / max(epoch_steps, 1)

        val_metrics = _evaluate_development_retrieval(
            model, val_loader, device, norm_mean, norm_std
        )
        completed_epochs = epoch + 1
        if _selection_improves(val_metrics, best_metrics):
            best_metrics = val_metrics
            selected_epoch = completed_epochs
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    label_to_index=label_to_index,
                    epoch=epoch + 1,
                    global_step=global_step,
                    selection_metric=val_metrics,
                    admission_receipt=verified_admission,
                ),
                best_checkpoint,
            )
        else:
            if backbone_trainable or isinstance(
                unwrapped_model._backbone, AppearanceBoundedResidual
            ):
                epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "classification_loss": round(avg_classification_loss, 4),
            "embedding_consistency_loss": round(avg_consistency_loss, 6),
            "baseline_anchor_loss": round(avg_anchor_loss, 6),
            "development_rank1": round(val_metrics["rank1"], 6),
            "development_map": round(val_metrics["map"], 6),
            "lr": round(lr, 8),
            "backbone_lr": round(
                0.0
                if isinstance(unwrapped_model._backbone, AppearanceBoundedResidual)
                else lr * config.backbone_lr_scale,
                8,
            ),
            "backbone_trainable": backbone_trainable,
        }
        history.append(epoch_record)
        print(json.dumps({"event": "train_epoch", **epoch_record}), flush=True)

        if (epoch + 1) % config.save_every_n_epochs == 0:
            ckpt = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    label_to_index=label_to_index,
                    epoch=epoch + 1,
                    global_step=global_step,
                    selection_metric=val_metrics,
                    admission_receipt=verified_admission,
                ),
                ckpt,
            )

        if epochs_without_improvement >= config.early_stop_patience:
            early_stopped = True
            break

    last_checkpoint = str(checkpoint_dir / "last_model.pt")
    torch.save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            label_to_index=label_to_index,
            epoch=completed_epochs,
            global_step=global_step,
            selection_metric=val_metrics,
            admission_receipt=verified_admission,
        ),
        last_checkpoint,
    )

    elapsed = time.time() - t0
    steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0
    total_samples = len(train_dataset)
    samples_per_sec = processed_samples / elapsed if elapsed > 0 else 0.0

    summary = {
        "config": config.to_dict(),
        "parameters": _count_parameters(model),
        "throughput": {
            "steps_per_second": round(steps_per_sec, 1),
            "samples_per_second": round(samples_per_sec, 1),
            "elapsed_seconds": round(elapsed, 1),
        },
        "flops_estimate": estimate_flops(config, total_samples),
        "history": history,
        "best_checkpoint": best_checkpoint,
        "pretrained_development_metrics": pretrained_metrics,
        "best_development_metrics": best_metrics,
        "selected_epoch": selected_epoch,
        "completed_epochs": completed_epochs,
        "early_stopped": early_stopped,
        "last_checkpoint": last_checkpoint,
        "total_steps": global_step,
        "processed_samples": processed_samples,
        "elapsed_seconds": round(elapsed, 1),
        "training_admission": {
            "manifest_sha256": admission_manifest.manifest_sha256,
            "receipt_sha256": verified_admission.receipt_sha256,
        },
    }

    summary_path = checkpoint_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    return summary


# ---------------------------------------------------------------------------
# Inference-time embedding extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_embeddings(
    model: ArcFaceModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device)
        if images.dtype == torch.uint8:
            images = ImageCache.gpu_normalize(images)
        else:
            images = images.to(dtype=torch.float32)
        if not torch.isfinite(images).all():
            raise RuntimeError("embedding input contains non-finite values")
        emb = model.extract_embedding(images)
        all_embs.append(emb)
        all_labels.append(labels.numpy())
    return np.concatenate(all_embs, axis=0), np.concatenate(all_labels, axis=0)
