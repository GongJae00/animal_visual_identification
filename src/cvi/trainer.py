from __future__ import annotations

import json
import math
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


from cvi.train.config import TrainConfig  # noqa: F401 — canonical, backward compat
from cvi.train.augment import RandAugment


# ---------------------------------------------------------------------------
# Image cache — preload all crops into RAM
# ---------------------------------------------------------------------------


class ImageCache:
    """Preload all oracle crop images into a contiguous CHW uint8 array.

    Eliminates 9P filesystem bottleneck during training by decoding and
    resizing all crops once at init.  Normalization is deferred to GPU
    transfer time.  Memory: N x 3 x 224 x 224 bytes (≈ 4.2 GiB for 28K crops).
    """

    _NORM_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _NORM_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, samples: list[tuple[Path, int]],
                 size: tuple[int, int] = (224, 224)) -> None:
        from PIL import Image
        n = len(samples)
        self._data = np.zeros((n, 3, size[0], size[1]), dtype=np.uint8)
        self._labels = np.array([label for _, label in samples], dtype=np.int64)
        for i, (path, _) in enumerate(samples):
            img = Image.open(path).convert("RGB")
            img = img.resize(size, Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8)
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


class OracleCropDataset(Dataset):
    """Read oracle crop images from the exported crop directory.

    Each sample is a (224x224 RGB tensor, label_index) pair.
    When *use_cache* is True (default), all images are preloaded into a
    contiguous uint8 RAM array at init, eliminating filesystem I/O during
    training.
    """

    def __init__(
        self,
        crop_root: Path,
        binding_records: list[dict],
        label_to_index: dict[str, int],
        *,
        use_cache: bool = True,
    ) -> None:
        self._samples: list[tuple[Path, int]] = []
        for rec in binding_records:
            label = rec["registered_dog_id"]
            if label not in label_to_index:
                continue
            label_idx = label_to_index[label]
            sample_tokens = rec.get("sample_tokens", [rec.get("identity_token", "")])
            for sample_token in sample_tokens:
                paths = sorted(crop_root.rglob(f"**/{sample_token}.jpg"))
                for p in paths:
                    self._samples.append((p, label_idx))

        self._cache: ImageCache | None = None
        if use_cache and self._samples:
            self._cache = ImageCache(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self._cache is not None:
            return self._cache[index]
        path, label = self._samples[index]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
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
        order = torch.randperm(self._num_identities, generator=g).tolist()
        indices: list[int] = []
        g2 = torch.Generator()
        for identity_id in order:
            candidates = self._label_to_indices[self._identity_ids[identity_id]]
            g2.manual_seed(int(g.seed()) + identity_id)
            perm = torch.randperm(len(candidates), generator=g2).tolist()
            pick = [candidates[i] for i in perm]
            indices.extend(pick)
            if len(indices) >= self._batch_size:
                yield indices[: self._batch_size]
                indices = indices[self._batch_size:]

    def __len__(self) -> int:
        return math.ceil(len(self._label_to_indices) / self._batch_size)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Dinov2Embedding(nn.Module):
    """DINOv2-small backbone producing an L2-normalized embedding.

    Supports gradient checkpointing via *use_gradient_checkpointing*.
    """

    def __init__(self, embedding_dim: int = 384,
                 use_gradient_checkpointing: bool = False) -> None:
        super().__init__()
        from transformers import AutoModel
        self._backbone = AutoModel.from_pretrained(
            "facebook/dinov2-small", attn_implementation="sdpa"
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
                 use_gradient_checkpointing: bool = False) -> None:
        super().__init__()
        from transformers import AutoModel
        self._backbone = AutoModel.from_pretrained("facebook/convnext-base-224")
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
                 backbone_factory: type[nn.Module] | None = None) -> None:
        super().__init__()
        self._config = config
        if backbone_factory is None:
            backbone_factory = Dinov2Embedding
        self._backbone = backbone_factory(
            config.embedding_dim,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
        self._head = ArcFaceHead(
            config.embedding_dim,
            config.num_classes,
            config.arcface_scale,
            config.arcface_margin,
        )

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        emb = self._backbone(images)
        if labels is not None and self.training:
            return self._head(emb, labels)
        return emb

    @torch.no_grad()
    def extract_embedding(self, images: torch.Tensor) -> np.ndarray:
        self.eval()
        emb = self._backbone(images)
        return emb.cpu().numpy()

    def export_to_onnx(self, output_path: Path) -> None:
        self.eval()
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            self._backbone,
            dummy,
            str(output_path),
            input_names=["images"],
            output_names=["embedding"],
            dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=18,
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


def _build_label_index(binding_records: list[dict]) -> dict[str, int]:
    unique_labels = sorted(set(rec["registered_dog_id"] for rec in binding_records))
    return {label: idx for idx, label in enumerate(unique_labels)}


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
        pin_memory=(device.type == "cuda" and num_workers == 0),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def train_model(
    config: TrainConfig,
    crop_root: Path,
    train_binding: list[dict],
    val_binding: list[dict] | None = None,
    device: torch.device = torch.device("cpu"),
    backbone_factory: type[nn.Module] | None = None,
) -> dict[str, Any]:
    """Run supervised ArcFace training on oracle crops.

    *backbone_factory* is a callable(embedding_dim, use_gradient_checkpointing) → nn.Module.
    Defaults to Dinov2Embedding.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    label_to_index = _build_label_index(train_binding)
    config = TrainConfig(
        **{**config.to_dict(), "num_classes": len(label_to_index)}
    )

    train_dataset = OracleCropDataset(
        crop_root, train_binding, label_to_index,
        use_cache=config.preload_images,
    )
    train_labels = [lab for _, lab in train_dataset]
    sampler = IdentityBalancedSampler(
        train_labels, config.batch_size,
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = _build_dataloader(train_dataset, sampler, config, device)

    val_loader: DataLoader | None = None
    if val_binding:
        val_dataset = OracleCropDataset(
            crop_root, val_binding, label_to_index,
            use_cache=config.preload_images,
        )
        val_loader = _build_dataloader(val_dataset, None, config, device, shuffle=False)

    model = ArcFaceModel(config, backbone_factory=backbone_factory).to(device)

    augment = RandAugment(n=2, m=9)

    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and config.mixed_precision else None
    best_val_loss = float("inf")
    best_checkpoint: str | None = None
    history: list[dict[str, Any]] = []
    global_step = 0
    t0 = time.time()
    norm_mean: torch.Tensor | None = None
    norm_std: torch.Tensor | None = None

    for epoch in range(config.epochs):
        lr = _warmup_cosine_schedule(
            epoch, config.epochs, config.warmup_epochs, config.lr, config.lr_min
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if images.dtype == torch.uint8:
                if norm_mean is None:
                    norm_mean = torch.tensor(
                        [0.485, 0.456, 0.406], device=device
                    ).view(1, 3, 1, 1)
                    norm_std = torch.tensor(
                        [0.229, 0.224, 0.225], device=device
                    ).view(1, 3, 1, 1)
                images = images.float().div_(255.0)
                if augment is not None:
                    for i in range(images.shape[0]):
                        images[i] = augment(images[i])
                images = images.sub_(norm_mean).div_(norm_std)

            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(images, labels)
                    loss = F.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images, labels)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

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

        val_loss: float | None = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    if images.dtype == torch.uint8:
                        if norm_mean is None:
                            norm_mean = torch.tensor(
                                [0.485, 0.456, 0.406], device=device
                            ).view(1, 3, 1, 1)
                            norm_std = torch.tensor(
                                [0.229, 0.224, 0.225], device=device
                            ).view(1, 3, 1, 1)
                        images = ImageCache.gpu_normalize(images, norm_mean, norm_std)
                    logits = model(images, labels)
                    loss = F.cross_entropy(logits, labels)
                    total_val_loss += loss.item()
                    val_steps += 1
            val_loss = total_val_loss / max(val_steps, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = str(checkpoint_dir / "best_model.pt")
                torch.save(model.state_dict(), best_checkpoint)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "lr": round(lr, 8),
        }
        history.append(epoch_record)
        print(json.dumps({"event": "train_epoch", **epoch_record}), flush=True)

        if (epoch + 1) % config.save_every_n_epochs == 0:
            ckpt = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            torch.save(model.state_dict(), ckpt)

    last_checkpoint = str(checkpoint_dir / "last_model.pt")
    torch.save(model.state_dict(), last_checkpoint)

    elapsed = time.time() - t0
    steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0
    total_samples = len(train_dataset)
    samples_per_sec = total_samples * config.epochs / elapsed if elapsed > 0 else 0.0

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
        "last_checkpoint": last_checkpoint,
        "total_steps": global_step,
        "elapsed_seconds": round(elapsed, 1),
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
        emb = model.extract_embedding(images)
        all_embs.append(emb)
        all_labels.append(labels.numpy())
    return np.concatenate(all_embs, axis=0), np.concatenate(all_labels, axis=0)
