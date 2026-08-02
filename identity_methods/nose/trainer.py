"""Stage-C frozen-DINO trainer for the NoseID-v1 oracle-ROI slice."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn import functional as F

from artifact_contracts.dinov2_contract import Dinov2LocalArtifactContract
from identity_methods.nose.augment import NoseAugmentedView, NoseIdentityAugment
from identity_methods.nose.losses import NoseIDObjective
from identity_methods.nose.model import NoseIDModel


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


def build_noseid_model(
    backbone: torch.nn.Module,
    contract: Dinov2LocalArtifactContract,
) -> NoseIDModel:
    processor = contract.preprocessor
    return NoseIDModel(
        backbone,
        image_mean=tuple(processor["image_mean"]),
        image_std=tuple(processor["image_std"]),
        rescale_factor=float(processor["rescale_factor"]),
    )


def _runtime_quality(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    native = torch.as_tensor(
        batch["native_short_side"], device=device, dtype=torch.float32
    )
    alignment = torch.as_tensor(
        batch["alignment_rms"], device=device, dtype=torch.float32
    )
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
    semantic_mask: torch.Tensor,
    invalid_mask: torch.Tensor,
    source_valid_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    semantic = semantic_mask.to(device=device, dtype=torch.long)
    if semantic.ndim != 3 or torch.any((semantic < 0) | (semantic > 2)):
        raise ValueError("semantic training mask must have shape [B,448,448]")
    semantic_probability = (
        F.one_hot(semantic, num_classes=3).permute(0, 3, 1, 2).float()
    )
    invalid = invalid_mask.to(device=device, dtype=torch.float32)
    source_valid = source_valid_mask.to(device=device, dtype=torch.float32)
    if (
        invalid.shape != semantic_probability[:, :1].shape
        or source_valid.shape != invalid.shape
    ):
        raise ValueError("invalid/source-valid training mask shape differs")
    source_valid = source_valid.clamp(0, 1)
    invalid = torch.maximum(invalid.clamp(0, 1), 1.0 - source_valid)
    return semantic_probability, invalid, source_valid


def _session_tensor(values: Iterable[str], device: torch.device) -> torch.Tensor:
    mapping = {value: index for index, value in enumerate(sorted(set(values)))}
    return torch.tensor([mapping[value] for value in values], device=device)


def _use_weight_decay(name: str, parameter: torch.Tensor) -> bool:
    lowered = name.lower()
    return parameter.ndim > 1 and not any(
        token in lowered for token in ("bias", "norm", "layer_scale")
    )


def build_frozen_stage_optimizer(
    model: NoseIDModel,
    objective: NoseIDObjective,
) -> torch.optim.AdamW:
    for parameter in model.dino.parameters():
        parameter.requires_grad = False
    for parameter in model.segmenter.parameters():
        parameter.requires_grad = False
    grouped: dict[tuple[float, float], list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        learning_rate = 3e-4 if name.startswith("texture_stream.") else 1e-3
        weight_decay = 0.05 if _use_weight_decay(name, parameter) else 0.0
        grouped.setdefault((learning_rate, weight_decay), []).append(parameter)
    for name, parameter in objective.named_parameters():
        weight_decay = 0.05 if _use_weight_decay(name, parameter) else 0.0
        grouped.setdefault((1e-3, weight_decay), []).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": parameters, "lr": learning_rate, "weight_decay": decay}
            for (learning_rate, decay), parameters in sorted(grouped.items())
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_stage_c_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
    epochs: int = 15,
    minimum_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    if steps_per_epoch <= 0 or epochs <= 0:
        raise ValueError("scheduler steps and epochs must be positive")
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch
    functions = []
    for group in optimizer.param_groups:
        base_lr = float(group["lr"])
        minimum_ratio = min(1.0, minimum_lr / base_lr)

        def multiplier(
            step: int,
            *,
            base_minimum: float = minimum_ratio,
        ) -> float:
            if step < warmup_steps:
                return 0.1 + 0.9 * step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return base_minimum + (1.0 - base_minimum) * cosine

        functions.append(multiplier)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, functions)


def _view_to_mapping(view: NoseAugmentedView) -> dict[str, torch.Tensor]:
    return {
        "rgb": view.rgb,
        "keypoints": view.keypoints,
        "semantic": view.semantic_probability,
        "invalid": view.invalid_probability,
        "source_valid": view.source_valid_probability,
        "degradation": view.degradation_target,
    }


def _device_view(
    value: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device=device, non_blocking=device.type == "cuda")
        for name, tensor in value.items()
    }


def _model_view(
    model: NoseIDModel,
    view: Mapping[str, torch.Tensor],
    quality: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return model(
        view["rgb"],
        view["keypoints"],
        quality,
        semantic_probability=view["semantic"],
        invalid_probability=view["invalid"],
        source_valid_probability=view["source_valid"],
    )


def _backward_cached_output(
    output: Mapping[str, torch.Tensor],
    cached: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> None:
    gradients = [cached[name].grad for name in names]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("NoseID gradient cache is incomplete")
    torch.autograd.backward([output[name] for name in names], gradients)


def _validate_logical_batch(batch: dict[str, Any]) -> None:
    if len(batch["identity_index"]) != 64:
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
            raise ValueError(
                "every NoseID-v1 batch identity needs cross-session positives"
            )


def train_frozen_stage_epoch(
    model: NoseIDModel,
    objective: NoseIDObjective,
    batches: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    micro_batch_size: int = 8,
    use_bfloat16: bool = True,
    augment: NoseIdentityAugment | None = None,
) -> dict[str, float]:
    if micro_batch_size not in {4, 8} or 64 % micro_batch_size:
        raise ValueError("NoseID-v1 micro batch must be 8 or OOM fallback 4")
    model.train()
    model.dino.eval()
    model.segmenter.eval()
    objective.train()
    margin_scale = min(1.0, max(0.0, epoch / 5.0))
    totals: dict[str, float] = {}
    steps = 0
    processed_images = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    augment = augment or NoseIdentityAugment(seed=epoch)
    first_names = ("embedding", "z_rgb", "z_texture", "degradation_predictions")
    second_names = ("embedding", "degradation_predictions")
    for batch in batches:
        _validate_logical_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        prepared: list[dict[str, Any]] = []
        cached_first: dict[str, list[torch.Tensor]] = {
            name: [] for name in first_names
        }
        cached_second: dict[str, list[torch.Tensor]] = {
            name: [] for name in second_names
        }
        for start in range(0, 64, micro_batch_size):
            stop = start + micro_batch_size
            rgb = batch["aligned_rgb"][start:stop].float()
            keypoints = batch["aligned_kp"][start:stop].float()
            semantic, invalid, source_valid = _semantic_probabilities(
                batch["semantic_mask"][start:stop],
                batch["invalid_mask"][start:stop],
                batch["source_valid_mask"][start:stop],
                torch.device("cpu"),
            )
            pairs = [
                augment.pair(
                    rgb[index],
                    keypoints[index],
                    semantic[index],
                    invalid[index],
                    source_valid[index],
                )
                for index in range(len(rgb))
            ]
            first = {
                name: torch.stack([_view_to_mapping(pair[0])[name] for pair in pairs])
                for name in _view_to_mapping(pairs[0][0])
            }
            second = {
                name: torch.stack([_view_to_mapping(pair[1])[name] for pair in pairs])
                for name in _view_to_mapping(pairs[0][1])
            }
            first_quality_batch = {
                "native_short_side": batch["native_short_side"][start:stop],
                "alignment_rms": batch["alignment_rms"][start:stop],
                "aligned_kp": first["keypoints"],
            }
            second_quality_batch = {
                "native_short_side": batch["native_short_side"][start:stop],
                "alignment_rms": batch["alignment_rms"][start:stop],
                "aligned_kp": second["keypoints"],
            }
            first_quality = _runtime_quality(
                first_quality_batch, torch.device("cpu")
            )
            second_quality = _runtime_quality(
                second_quality_batch, torch.device("cpu")
            )
            cpu_rng = torch.get_rng_state()
            cuda_rng = (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            )
            dtype = (
                torch.bfloat16
                if use_bfloat16
                and device.type == "cuda"
                and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            enabled = device.type == "cuda"
            device_first = _device_view(first, device)
            device_second = _device_view(second, device)
            device_first_quality = first_quality.to(device=device)
            device_second_quality = second_quality.to(device=device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=dtype, enabled=enabled
            ):
                first_output = _model_view(model, device_first, device_first_quality)
            for name in first_names:
                cached_first[name].append(
                    first_output[name].detach().float().requires_grad_(True)
                )
            del first_output, device_first
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=dtype, enabled=enabled
            ):
                second_output = _model_view(model, device_second, device_second_quality)
            for name in second_names:
                cached_second[name].append(
                    second_output[name].detach().float().requires_grad_(True)
                )
            del second_output, device_second
            prepared.append(
                {
                    "first": first,
                    "second": second,
                    "first_quality": first_quality,
                    "second_quality": second_quality,
                    "cpu_rng": cpu_rng,
                    "cuda_rng": cuda_rng,
                }
            )
        logical_first = {
            name: torch.cat(values, dim=0)
            for name, values in cached_first.items()
        }
        logical_second = {
            name: torch.cat(values, dim=0)
            for name, values in cached_second.items()
        }
        first_targets = torch.cat(
            [inputs["first"]["degradation"] for inputs in prepared]
        ).to(device)
        second_targets = torch.cat(
            [inputs["second"]["degradation"] for inputs in prepared]
        ).to(device)
        labels = torch.as_tensor(batch["identity_index"], device=device, dtype=torch.long)
        sessions = _session_tensor(batch["session_id"], device)
        losses = objective(
            logical_first,
            labels,
            sessions,
            second_view_output=logical_second,
            first_degradation_target=first_targets,
            second_degradation_target=second_targets,
            margin_scale=margin_scale,
        )
        scaler.scale(losses["total"]).backward()
        for micro_index, inputs in enumerate(prepared):
            torch.set_rng_state(inputs["cpu_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(inputs["cuda_rng"], device)
            first_quality = inputs["first_quality"].to(
                device=device, non_blocking=True
            )
            first = _device_view(inputs["first"], device)
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=enabled
            ):
                first_output = _model_view(model, first, first_quality)
            _backward_cached_output(
                first_output,
                {name: cached_first[name][micro_index] for name in first_names},
                first_names,
            )
            del first_output, first

            second = _device_view(inputs["second"], device)
            second_quality = inputs["second_quality"].to(
                device=device, non_blocking=True
            )
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=enabled
            ):
                second_output = _model_view(model, second, second_quality)
            _backward_cached_output(
                second_output,
                {name: cached_second[name][micro_index] for name in second_names},
                second_names,
            )
            del second_output, second
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
        if scheduler is not None:
            scheduler.step()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        steps += 1
        processed_images += 64
    if steps == 0:
        raise ValueError("NoseID-v1 training loader is empty")
    elapsed = time.perf_counter() - started
    result = {name: value / steps for name, value in totals.items()}
    result["images_per_second"] = processed_images / max(elapsed, 1e-9)
    result["wall_seconds"] = elapsed
    result["peak_cuda_memory_bytes"] = float(
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    return result


__all__ = [
    "_backward_cached_output",
    "build_frozen_stage_optimizer",
    "build_noseid_model",
    "build_stage_c_scheduler",
    "load_receipt_bound_frozen_dino",
    "train_frozen_stage_epoch",
]
