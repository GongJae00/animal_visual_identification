"""Stage-C frozen-DINO trainer for the NoseID-v1 oracle-ROI slice."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from cvi.evidence.dinov2_contract import Dinov2LocalArtifactContract
from cvi.nose_id.augment import NoseIdentityAugment
from cvi.nose_id.losses import NoseIDObjective
from cvi.nose_id.model import NoseIDModel


def load_receipt_bound_frozen_dino(
    *,
    model_directory: Path,
    weight_intake_bundle: Path,
    preprocessor_intake_bundle: Path,
) -> tuple[torch.nn.Module, Dinov2LocalArtifactContract]:
    contract = Dinov2LocalArtifactContract.load(
        model_directory=model_directory,
        weight_intake_bundle=weight_intake_bundle,
        preprocessor_intake_bundle=preprocessor_intake_bundle,
    )
    contract.revalidate_local_files()
    from transformers import Dinov2Model

    backbone = Dinov2Model.from_pretrained(
        str(contract.model_directory),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    if not isinstance(backbone, torch.nn.Module):
        raise TypeError("local DINOv2 loader must return torch.nn.Module")
    contract.revalidate_local_files()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone.eval(), contract


def _runtime_quality(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    native = torch.as_tensor(batch["native_short_side"], device=device, dtype=torch.float32)
    alignment = torch.as_tensor(batch["alignment_rms"], device=device, dtype=torch.float32)
    keypoints = batch["aligned_kp"].to(device=device, dtype=torch.float32)
    return torch.stack(
        [
            torch.ones_like(native),
            (native / 448.0).clamp(0, 1),
            keypoints[:, :, 2].mean(dim=1),
            alignment,
        ],
        dim=1,
    )


def _semantic_probabilities(
    semantic_mask: torch.Tensor, invalid_mask: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic = semantic_mask.to(device=device, dtype=torch.long)
    if semantic.ndim != 3 or torch.any((semantic < 0) | (semantic > 2)):
        raise ValueError("semantic training mask must have shape [B,448,448]")
    semantic_probability = F.one_hot(semantic, num_classes=3).permute(0, 3, 1, 2).float()
    invalid = invalid_mask.to(device=device, dtype=torch.float32)
    if invalid.shape != semantic_probability[:, :1].shape:
        raise ValueError("invalid training mask shape differs")
    return semantic_probability, invalid.clamp(0, 1)


def _session_tensor(values: Iterable[str], device: torch.device) -> torch.Tensor:
    mapping = {value: index for index, value in enumerate(sorted(set(values)))}
    return torch.tensor([mapping[value] for value in values], device=device)


def build_frozen_stage_optimizer(
    model: NoseIDModel,
    objective: NoseIDObjective,
) -> torch.optim.AdamW:
    for parameter in model.dino.parameters():
        parameter.requires_grad = False
    for parameter in model.segmenter.parameters():
        parameter.requires_grad = False
    texture_parameters = list(model.texture_stream.parameters())
    texture_ids = {id(parameter) for parameter in texture_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in texture_ids
    ] + list(objective.parameters())
    return torch.optim.AdamW(
        [
            {"params": texture_parameters, "lr": 3e-4},
            {"params": head_parameters, "lr": 1e-3},
        ],
        weight_decay=0.05,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def train_frozen_stage_epoch(
    model: NoseIDModel,
    objective: NoseIDObjective,
    batches: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    micro_batch_size: int = 8,
    gradient_accumulation_steps: int = 8,
    use_bfloat16: bool = True,
    augment: NoseIdentityAugment | None = None,
) -> dict[str, float]:
    if micro_batch_size * gradient_accumulation_steps != 64:
        raise ValueError("NoseID-v1 effective batch must be 64")
    model.train()
    model.dino.eval()
    model.segmenter.eval()
    objective.train()
    margin_scale = min(1.0, max(0.0, (epoch + 1) / 5.0))
    totals: dict[str, float] = {}
    steps = 0
    use_fp16_scaler = device.type == "cuda" and not (
        use_bfloat16 and torch.cuda.is_bf16_supported()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    augment = augment or NoseIdentityAugment(seed=epoch)
    for batch in batches:
        logical_size = len(batch["identity_index"])
        if logical_size != 64:
            raise ValueError("NoseID-v1 trainer requires logical batches of 64")
        identity_values = [int(value) for value in batch["identity_index"]]
        unique_identities = set(identity_values)
        if len(unique_identities) != 16 or any(
            identity_values.count(identity) != 4 for identity in unique_identities
        ):
            raise ValueError("NoseID-v1 batch must contain 16 identities x 4 samples")
        for identity in unique_identities:
            sessions = {
                batch["session_id"][index]
                for index, value in enumerate(identity_values)
                if value == identity
            }
            if len(sessions) < 2:
                raise ValueError("every NoseID-v1 batch identity needs cross-session positives")
        optimizer.zero_grad(set_to_none=True)
        prepared: list[dict[str, Any]] = []
        cached: dict[str, list[torch.Tensor]] = {
            "embedding": [],
            "z_rgb": [],
            "z_texture": [],
        }
        cached_second: list[torch.Tensor] = []
        for start in range(0, logical_size, micro_batch_size):
            stop = start + micro_batch_size
            rgb = batch["aligned_rgb"][start:stop].to(dtype=torch.float32)
            keypoints = batch["aligned_kp"][start:stop].to(dtype=torch.float32)
            semantic, invalid = _semantic_probabilities(
                batch["semantic_mask"][start:stop],
                batch["invalid_mask"][start:stop],
                torch.device("cpu"),
            )
            pairs = [
                augment.pair(rgb[index], keypoints[index], semantic[index], invalid[index])
                for index in range(len(rgb))
            ]
            first = {
                "rgb": torch.stack([pair[0][0] for pair in pairs]),
                "keypoints": torch.stack([pair[0][1] for pair in pairs]),
                "semantic": torch.stack([pair[0][2] for pair in pairs]),
                "invalid": torch.stack([pair[0][3] for pair in pairs]),
            }
            second = {
                "rgb": torch.stack([pair[1][0] for pair in pairs]),
                "keypoints": torch.stack([pair[1][1] for pair in pairs]),
                "semantic": torch.stack([pair[1][2] for pair in pairs]),
                "invalid": torch.stack([pair[1][3] for pair in pairs]),
            }
            quality_batch = {
                "native_short_side": batch["native_short_side"][start:stop],
                "alignment_rms": batch["alignment_rms"][start:stop],
                "aligned_kp": first["keypoints"],
            }
            runtime_quality = _runtime_quality(quality_batch, torch.device("cpu"))
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            dtype = torch.bfloat16 if use_bfloat16 and device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            enabled = device.type == "cuda"
            device_first = {
                name: value.to(device=device, non_blocking=device.type == "cuda")
                for name, value in first.items()
            }
            device_second = {
                name: value.to(device=device, non_blocking=device.type == "cuda")
                for name, value in second.items()
            }
            device_quality = runtime_quality.to(device=device)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                output = model(
                    device_first["rgb"],
                    device_first["keypoints"],
                    device_quality,
                    semantic_probability=device_first["semantic"],
                    invalid_probability=device_first["invalid"],
                )
                second_embedding = model(
                    device_second["rgb"],
                    device_second["keypoints"],
                    device_quality,
                    semantic_probability=device_second["semantic"],
                    invalid_probability=device_second["invalid"],
                )["embedding"]
            for name in cached:
                cached[name].append(output[name].detach().float().requires_grad_(True))
            cached_second.append(
                second_embedding.detach().float().requires_grad_(True)
            )
            prepared.append(
                {
                    "first": first,
                    "second": second,
                    "quality": runtime_quality,
                    "cpu_rng": cpu_rng,
                    "cuda_rng": cuda_rng,
                }
            )
        logical_output = {
            name: torch.cat(values, dim=0) for name, values in cached.items()
        }
        labels = torch.as_tensor(batch["identity_index"], device=device, dtype=torch.long)
        sessions = _session_tensor(batch["session_id"], device)
        losses = objective(
            logical_output,
            labels,
            sessions,
            second_view_embedding=torch.cat(cached_second, dim=0),
            margin_scale=margin_scale,
        )
        scaler.scale(losses["total"]).backward()
        for micro_index, inputs in enumerate(prepared):
            torch.set_rng_state(inputs["cpu_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(inputs["cuda_rng"], device)
            first = {
                name: value.to(device=device, non_blocking=True)
                for name, value in inputs["first"].items()
            }
            second = {
                name: value.to(device=device, non_blocking=True)
                for name, value in inputs["second"].items()
            }
            quality = inputs["quality"].to(device=device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                output = model(
                    first["rgb"],
                    first["keypoints"],
                    quality,
                    semantic_probability=first["semantic"],
                    invalid_probability=first["invalid"],
                )
                second_embedding = model(
                    second["rgb"],
                    second["keypoints"],
                    quality,
                    semantic_probability=second["semantic"],
                    invalid_probability=second["invalid"],
                )["embedding"]
            outputs = [output[name] for name in cached] + [second_embedding]
            gradients = [cached[name][micro_index].grad for name in cached] + [
                cached_second[micro_index].grad
            ]
            if any(gradient is None for gradient in gradients):
                raise RuntimeError("NoseID gradient cache is incomplete")
            torch.autograd.backward(outputs, gradients)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for module in (model, objective)
                for parameter in module.parameters()
                if parameter.requires_grad
            ],
            5.0,
        )
        scaler.step(optimizer)
        scaler.update()
        steps += 1
    if steps == 0:
        raise ValueError("NoseID-v1 training loader is empty")
    return {name: value / steps for name, value in totals.items()}


__all__ = [
    "build_frozen_stage_optimizer",
    "load_receipt_bound_frozen_dino",
    "train_frozen_stage_epoch",
]
