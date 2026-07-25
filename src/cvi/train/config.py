from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainConfig:
    model_name: str = "convnext-base"
    embedding_dim: int = 384
    num_classes: int = 0
    loss_type: str = "magface"
    arcface_scale: float = 30.0
    arcface_margin: float = 0.50
    magface_margin: float = 0.45
    magface_scale: float = 64.0
    batch_size: int = 128
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
    preload_images: bool = True
    use_amp: bool = True
    val_split: float = 0.1
    early_stop_patience: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in fields}
        return cls(**filtered)
