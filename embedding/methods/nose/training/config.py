"""Fixed configuration contracts for the NoseID-v1 vertical slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NoseIDConfig:
    master_size: int = 448
    dino_size: int = 336
    patch_size: int = 14
    rgb_dim: int = 256
    texture_dim: int = 256
    shape_dim: int = 64
    embedding_dim: int = 512
    quality_dim: int = 14
    minimum_native_short_side: int = 96
    texture_start_short_side: int = 160
    texture_full_short_side: int = 224
    query_min_utility: float = 0.20
    enrollment_min_utility: float = 0.55

    def __post_init__(self) -> None:
        expected = {
            "master_size": 448,
            "dino_size": 336,
            "patch_size": 14,
            "rgb_dim": 256,
            "texture_dim": 256,
            "shape_dim": 64,
            "embedding_dim": 512,
            "quality_dim": 14,
            "minimum_native_short_side": 96,
            "texture_start_short_side": 160,
            "texture_full_short_side": 224,
            "query_min_utility": 0.20,
            "enrollment_min_utility": 0.55,
        }
        for field, required in expected.items():
            if getattr(self, field) != required:
                raise ValueError(f"NoseID-v1 fixed field differs: {field}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NoseIDConfig":
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("NoseID-v1 config keys differ")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class NoseIDTrainConfig:
    identities_per_batch: int = 16
    samples_per_identity: int = 4
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    subcenters: int = 3
    arcface_scale: float = 32.0
    arcface_margin: float = 0.30
    supcon_temperature: float = 0.07
    triplet_margin: float = 0.20
    frozen_epochs: int = 15
    seed: int = 0

    @property
    def logical_batch_size(self) -> int:
        return self.identities_per_batch * self.samples_per_identity

    def __post_init__(self) -> None:
        if self.identities_per_batch != 16 or self.samples_per_identity != 4:
            raise ValueError("NoseID-v1 requires P=16 and K=4")
        if self.micro_batch_size * self.gradient_accumulation_steps != 64:
            raise ValueError("micro batch and accumulation must produce batch 64")
        if self.subcenters != 3:
            raise ValueError("NoseID-v1 requires three ArcFace sub-centers")


__all__ = ["NoseIDConfig", "NoseIDTrainConfig"]
