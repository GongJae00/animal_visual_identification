"""Face ReID trainer — frozen DINOv2 + trainable regional encoder."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cvi.evidence.dinov2_contract import Dinov2LocalArtifactContract
from cvi.face_id.losses import FaceIDObjective
from cvi.face_id.model import FaceIDModel


def load_receipt_bound_frozen_dino(
    *,
    model_directory: Path,
    weight_intake_bundle: Path,
    preprocessor_intake_bundle: Path,
) -> tuple[torch.nn.Module, Dinov2LocalArtifactContract]:
    """Reused from cvi.nose_id.trainer — identical contract."""
    from cvi.nose_id.trainer import load_receipt_bound_frozen_dino as _loader
    return _loader(
        model_directory=model_directory,
        weight_intake_bundle=weight_intake_bundle,
        preprocessor_intake_bundle=preprocessor_intake_bundle,
    )


def build_faceid_model(
    backbone: torch.nn.Module,
    contract: Dinov2LocalArtifactContract,
) -> FaceIDModel:
    processor = contract.preprocessor
    return FaceIDModel(
        backbone,
        image_mean=tuple(processor["image_mean"]),
        image_std=tuple(processor["image_std"]),
        rescale_factor=float(processor["rescale_factor"]),
    )


def build_faceid_optimizer(
    model: FaceIDModel,
    objective: FaceIDObjective,
) -> torch.optim.AdamW:
    for param in model.dino.parameters():
        param.requires_grad = False
    params = [
        p for p in model.parameters() if p.requires_grad
    ] + list(objective.parameters())
    return torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)


def train_faceid_epoch(
    model: FaceIDModel,
    objective: FaceIDObjective,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    model.dino.eval()
    objective.train()
    margin_scale = min(1.0, max(0.0, epoch / 5.0))
    totals: dict[str, float] = {}
    steps = 0
    use_amp = device.type == "cuda"
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=False)

    for batch in loader:
        rgb = batch["rgb"].to(device=device, dtype=torch.float32)
        labels = torch.as_tensor(batch["identity_index"], device=device, dtype=torch.long)
        sessions = torch.as_tensor(
            [hash(s) % 10000 for s in batch["session_id"]], device=device
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(rgb)
            losses = objective(output, labels, sessions, margin_scale=margin_scale)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad]
            + list(objective.parameters()),
            5.0,
        )
        scaler.step(optimizer)
        scaler.update()

        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        steps += 1

    if steps == 0:
        raise ValueError("FaceID training loader is empty")
    return {name: value / steps for name, value in totals.items()}


__all__ = [
    "build_faceid_model",
    "build_faceid_optimizer",
    "load_receipt_bound_frozen_dino",
    "train_faceid_epoch",
]
