from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainConfig:
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

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(d) - fields
        if unknown:
            raise ValueError(f"unknown training configuration keys: {sorted(unknown)}")
        return cls(**d)
