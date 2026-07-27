"""Train a multi-head DINOv2 embedding model with visual, texture, and structural heads.

Produces a 640-d L2-normalized embedding [visual(384) | texture(128) | structural(128)]
and exports to ONNX for deployment.

Usage:
  uv run python tools/train_multi_head_model.py \\
      --assignment ... --registry-db ... --crop-root ... --output-dir ... \\
      --epochs 50 --batch-size 128 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from cvi.multi_head import MultiHeadConfig, MultiHeadModel
from cvi.protected_io import read_strict_json_object


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OracleCropMultiDataset(Dataset):
    """Training dataset loading oracle crop images with multi-head support."""

    def __init__(self, crop_root: Path, records: list[dict],
                 label_to_index: dict[str, int],
                 use_cache: bool = True) -> None:
        self._samples: list[tuple[Path, int]] = []
        for rec in records:
            label = rec["registered_dog_id"]
            if label not in label_to_index:
                continue
            label_idx = label_to_index[label]
            for st in rec.get("sample_tokens", [rec.get("identity_token", "")]):
                paths = sorted(crop_root.rglob(f"**/{st}.jpg"))
                for p in paths:
                    self._samples.append((p, label_idx))
        self._cache: list[tuple[torch.Tensor, int]] | None = None
        if use_cache and self._samples:
            self._cache = []
            from PIL import Image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            for path, label in self._samples:
                img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = (arr - mean.numpy()) / std.numpy()
                tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)))
                self._cache.append((tensor, label))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self._cache is not None:
            return self._cache[index]
        path, label = self._samples[index]
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(np.transpose(arr, (2, 0, 1))), label


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class IdentityBalancedSamplerMulti(Sampler):
    """Yield balanced batches with one sample per identity per step."""

    def __init__(self, labels: list[int], batch_size: int,
                 generator: torch.Generator | None = None) -> None:
        self._batch_size = batch_size
        self._generator = generator
        label_to_indices: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            label_to_indices.setdefault(lab, []).append(idx)
        self._label_to_indices = label_to_indices
        self._identity_ids = sorted(label_to_indices.keys())
        self._num_identities = len(self._identity_ids)

    def __iter__(self):
        g = self._generator or torch.default_generator
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
                yield indices[:self._batch_size]
                indices = indices[self._batch_size:]

    def __len__(self) -> int:
        return max(1, math.ceil(len(self._label_to_indices) / self._batch_size))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _warmup_cosine(epoch: int, total: int, warmup: int,
                   lr_max: float, lr_min: float) -> float:
    if epoch < warmup:
        return lr_max * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def train_multi_head(config: MultiHeadConfig, crop_root: Path,
                     train_records: list[dict], val_records: list[dict] | None,
                     device: torch.device) -> dict:
    unique_labels = sorted(set(r["registered_dog_id"] for r in train_records))
    label_to_index = {lbl: i for i, lbl in enumerate(unique_labels)}
    config = MultiHeadConfig(**{**config.to_dict(), "num_classes": len(unique_labels)})
    print(json.dumps({"event": "train_start", "config": config.to_dict(),
                       "num_samples": len(train_records),
                       "num_classes": len(unique_labels)}))

    train_ds = OracleCropMultiDataset(crop_root, train_records, label_to_index,
                                       use_cache=config.preload_images)
    train_labels = [l for _, l in train_ds]
    sampler = IdentityBalancedSamplerMulti(
        train_labels, config.batch_size,
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=0 if config.preload_images else 2,
        pin_memory=(device.type == "cuda"),
    )

    val_loader: DataLoader | None = None
    if val_records:
        val_ds = OracleCropMultiDataset(crop_root, val_records, label_to_index,
                                         use_cache=config.preload_images)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                shuffle=False, num_workers=0)

    model = MultiHeadModel(config).to(device)

    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and config.mixed_precision else None
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_checkpoint: str | None = None
    history: list[dict] = []
    global_step = 0
    t0 = time.time()

    for epoch in range(config.epochs):
        lr = _warmup_cosine(epoch, config.epochs, config.warmup_epochs,
                            config.lr, config.lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    lv, lt, ls = model(images, labels)
                    loss = (F.cross_entropy(lv, labels) +
                            config.fusion_weight_t * F.cross_entropy(lt, labels) +
                            config.fusion_weight_s * F.cross_entropy(ls, labels))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                lv, lt, ls = model(images, labels)
                loss = (F.cross_entropy(lv, labels) +
                        config.fusion_weight_t * F.cross_entropy(lt, labels) +
                        config.fusion_weight_s * F.cross_entropy(ls, labels))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            if global_step % config.log_interval == 0:
                used = torch.cuda.max_memory_allocated(device) // 2**20 if device.type == "cuda" else 0
                print(json.dumps({"event": "step", "epoch": epoch + 1,
                                   "step": global_step, "loss": round(loss.item(), 4),
                                   "lr": lr, "vram_mib": used}))

        avg_loss = epoch_loss / max(epoch_steps, 1)
        val_loss_epoch: float | None = None
        if val_loader:
            model.eval()
            vl = 0.0
            vs = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    lv, lt, ls = model(images, labels)
                    vl += (F.cross_entropy(lv, labels).item() +
                           config.fusion_weight_t * F.cross_entropy(lt, labels).item() +
                           config.fusion_weight_s * F.cross_entropy(ls, labels).item())
                    vs += 1
            val_loss_epoch = vl / max(vs, 1)
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_checkpoint = str(checkpoint_dir / "best_model.pt")
                torch.save(model.state_dict(), best_checkpoint)

        epoch_rec = {"epoch": epoch + 1, "train_loss": round(avg_loss, 4),
                     "val_loss": round(val_loss_epoch, 4) if val_loss_epoch else None,
                     "lr": lr}
        history.append(epoch_rec)
        print(json.dumps({"event": "epoch", **epoch_rec}))

        if (epoch + 1) % config.save_every_n_epochs == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch+1:03d}.pt")

    last_checkpoint = str(checkpoint_dir / "last_model.pt")
    torch.save(model.state_dict(), last_checkpoint)

    onnx_path = str(checkpoint_dir / "model.onnx")
    model.export_to_onnx(onnx_path, device)

    elapsed = time.time() - t0
    summary = {
        "config": config.to_dict(),
        "history": history,
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
        "onnx_export": onnx_path,
        "total_steps": global_step,
        "elapsed_seconds": round(elapsed, 1),
    }
    (checkpoint_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def _select_records(binding: dict, access: str) -> list[dict]:
    return [b for b in binding.get("bindings", []) if b.get("model_access") == access]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--split-receipt", required=True, type=Path)
    parser.add_argument("--registry-db", required=True, type=Path)
    parser.add_argument("--registry-manifest", type=Path)
    parser.add_argument("--expected-split-receipt-sha256")
    parser.add_argument("--crop-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--w-t", type=float, default=0.5)
    parser.add_argument("--w-s", type=float, default=0.5)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    parser.exit(
        status=2,
        message=(
            "Multi-head training is disabled: its sampler, development "
            "selection, checkpoint, and authenticated crop-manifest contracts "
            "are not validated.\n"
        ),
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    from cvi.split_registry_binding import build_binding
    assignment = read_strict_json_object(args.assignment)
    split_receipt = read_strict_json_object(args.split_receipt)
    registry_manifest = read_strict_json_object(args.registry_manifest)
    binding = build_binding(
        assignment,
        args.registry_db,
        split_receipt,
        registry_manifest,
        args.expected_split_receipt_sha256,
    )
    if not binding.is_valid:
        print(json.dumps({"event": "binding_invalid",
                           "unregistered": len(binding.unregistered_tokens)}))
        raise SystemExit(1)

    train_records = _select_records(binding.to_dict(), "MODEL_TRAINING")
    val_records = _select_records(binding.to_dict(), "MODEL_SELECTION")

    config = MultiHeadConfig(
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        fusion_weight_t=args.w_t, fusion_weight_s=args.w_s,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=args.compile,
        preload_images=not args.no_preload,
        checkpoint_dir=str(args.output_dir / "checkpoints"),
    )

    summary = train_multi_head(config, args.crop_root, train_records,
                                val_records, device)
    print(json.dumps({"status": "DONE", **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
