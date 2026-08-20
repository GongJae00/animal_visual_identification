"""Face identity trainer — frozen DINOv2 + trainable regional encoder."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from shared.contracts.dinov2_contract import Dinov2LocalArtifactContract
from identification.training.face.losses import FaceIDObjective, FaceResidualObjective
from identification.export.face.model import FaceIDModel
from identification.export.face.residual_model import FaceIDResidualModel


def load_receipt_bound_frozen_dino(
    *,
    model_directory: Path,
    weight_intake_bundle: Path,
    preprocessor_intake_bundle: Path,
) -> tuple[torch.nn.Module, Dinov2LocalArtifactContract]:
    """Reused from identification.training.nose.trainer — identical contract."""
    from identification.training.nose.trainer import load_receipt_bound_frozen_dino as _loader

    return _loader(
        model_directory=model_directory,
        weight_intake_bundle=weight_intake_bundle,
        preprocessor_intake_bundle=preprocessor_intake_bundle,
    )


def build_faceid_model(
    backbone: torch.nn.Module,
    contract: Dinov2LocalArtifactContract,
    *,
    architecture: str = "regional_v4",
) -> FaceIDModel | FaceIDResidualModel:
    processor = contract.preprocessor
    model_type = {
        "regional_v4": FaceIDModel,
        "cls_residual_v5": FaceIDResidualModel,
        "aligned_cls_residual_v5": FaceIDResidualModel,
    }.get(architecture)
    if model_type is None:
        raise ValueError("unsupported FaceID architecture")
    return model_type(
        backbone,
        image_mean=tuple(processor["image_mean"]),
        image_std=tuple(processor["image_std"]),
        rescale_factor=float(processor["rescale_factor"]),
    )


def build_faceid_optimizer(
    model: FaceIDModel | FaceIDResidualModel,
    objective: FaceIDObjective | FaceResidualObjective,
) -> torch.optim.AdamW:
    for param in model.dino.parameters():
        param.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad] + list(
        objective.parameters()
    )
    return torch.optim.AdamW(
        params, lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8
    )


def train_faceid_epoch(
    model: FaceIDModel | FaceIDResidualModel,
    objective: FaceIDObjective | FaceResidualObjective,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    model.train()
    model.dino.eval()
    objective.train()
    margin_scale = (
        1.0
        if isinstance(objective, FaceResidualObjective)
        else min(1.0, max(0.0, epoch / 5.0))
    )
    totals: dict[str, float] = {}
    steps = 0
    aligned_samples = 0
    observed_samples = 0
    use_amp = device.type == "cuda"
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=False)

    for batch in loader:
        rgb = batch["rgb"].to(device=device, dtype=torch.float32)
        landmarks = batch.get("landmarks")
        if landmarks is not None:
            landmarks = landmarks.to(device=device, dtype=torch.float32)
        quality_target = batch.get("quality_target")
        if quality_target is not None:
            quality_target = quality_target.to(device=device, dtype=torch.float32)
        second_rgb = batch.get("second_rgb")
        if second_rgb is not None:
            second_rgb = second_rgb.to(device=device, dtype=torch.float32)
        labels = torch.as_tensor(
            batch["identity_index"], device=device, dtype=torch.long
        )
        sessions = torch.as_tensor(
            [
                int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")
                & ((1 << 63) - 1)
                for s in batch["session_id"]
            ],
            device=device,
        )
        alignment = batch.get("alignment_applied")
        if alignment is not None:
            aligned_samples += int(torch.as_tensor(alignment).sum())
            observed_samples += len(labels)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            output = model(rgb, landmarks)
            second_output = (
                None if second_rgb is None else model(second_rgb, landmarks)
            )
            losses = objective(
                output,
                labels,
                sessions,
                quality_target=quality_target,
                second_view_embedding=(
                    None if second_output is None else second_output["embedding"]
                ),
                curriculum_stage=min(epoch, 3),
                margin_scale=margin_scale,
            )

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
    metrics = {name: value / steps for name, value in totals.items()}
    metrics["alignment_coverage"] = (
        aligned_samples / observed_samples if observed_samples else 0.0
    )
    metrics["objective_activation"] = {
        "supervised_contrastive": (
            "ACTIVE" if metrics["supcon_valid_anchor_fraction"] > 0.0 else "NO_VALID_ANCHORS"
        ),
        "batch_hard_triplet": (
            "ACTIVE"
            if metrics["cross_session_triplet_valid_anchor_fraction"] > 0.0
            else "NO_CROSS_SESSION_POSITIVES"
        ),
        "view_consistency": (
            "ACTIVE" if metrics["second_view_coverage"] > 0.0 else "NO_SECOND_VIEW"
        ),
    }
    return metrics


__all__ = [
    "build_faceid_model",
    "build_faceid_optimizer",
    "load_receipt_bound_frozen_dino",
    "train_faceid_epoch",
]
