from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainConfig:
    architecture: str = "standard_arcface"
    model_name: str = "dinov2-small"
    embedding_dim: int = 384
    num_classes: int = 0
    loss_type: str = "arcface"
    arcface_scale: float = 30.0
    arcface_margin: float = 0.50
    magface_margin: float = 0.45
    magface_scale: float = 64.0
    batch_size: int = 32
    epochs: int = 100
    lr: float = 3e-4
    lr_min: float = 1e-6
    backbone_lr_scale: float = 0.1
    freeze_backbone_epochs: int = 1
    embedding_consistency_weight: float = 0.0
    border_consistency_weight: float = 0.0
    baseline_anchor_weight: float = 0.0
    residual_scale: float = 0.1
    warmup_epochs: int = 10
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    grad_clip_norm: float = 5.0
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50
    save_every_n_epochs: int = 5
    mixed_precision: bool = True
    num_workers: int = 4
    compile_model: bool = False
    gradient_checkpointing: bool = False
    preload_images: bool = False
    use_amp: bool = True
    val_split: float = 0.1
    early_stop_patience: int = 10

    def __post_init__(self) -> None:
        if self.architecture not in {
            "standard_arcface",
            "appearance_bounded_residual_v4",
        }:
            raise ValueError("unsupported training architecture")
        if self.loss_type != "arcface":
            raise ValueError("only the implemented arcface loss is supported")
        if self.embedding_dim <= 0 or self.num_classes < 0:
            raise ValueError("embedding_dim must be positive and num_classes non-negative")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.lr <= 0.0 or self.lr_min < 0.0 or self.lr_min > self.lr:
            raise ValueError("learning-rate bounds are invalid")
        if not 0.0 < self.backbone_lr_scale <= 1.0:
            raise ValueError("backbone_lr_scale must be in (0, 1]")
        if self.freeze_backbone_epochs < 0 or self.freeze_backbone_epochs > self.epochs:
            raise ValueError("freeze_backbone_epochs must be in [0, epochs]")
        if self.embedding_consistency_weight < 0.0:
            raise ValueError("embedding_consistency_weight must be non-negative")
        for name in ("border_consistency_weight", "baseline_anchor_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < self.residual_scale <= 0.25:
            raise ValueError("residual_scale must be in (0, 0.25]")
        if self.architecture == "appearance_bounded_residual_v4" and (
            self.border_consistency_weight <= 0.0
            or self.baseline_anchor_weight <= 0.0
        ):
            raise ValueError(
                "A4 requires positive border consistency and baseline anchor weights"
            )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.early_stop_patience <= 0:
            raise ValueError("early_stop_patience must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(d) - fields
        if unknown:
            raise ValueError(f"unknown training configuration keys: {sorted(unknown)}")
        return cls(**d)
