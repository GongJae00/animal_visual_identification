"""Face ReID fixed configuration and types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FaceIDConfig:
    input_size: int = 224
    patch_size: int = 14
    embedding_dim: int = 256
    num_regions: int = 5
    region_names: tuple[str, ...] = (
        "global", "eyes", "muzzle", "forehead", "ears_outer",
    )
    quality_dim: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FaceIDConfig":
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("FaceID config keys differ")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class FaceIDTrainConfig:
    identities_per_batch: int = 16
    samples_per_identity: int = 4
    micro_batch_size: int = 8
    subcenters: int = 3
    arcface_scale: float = 32.0
    arcface_margin: float = 0.30
    supcon_temperature: float = 0.07
    triplet_margin: float = 0.20
    seed: int = 0

    @property
    def logical_batch_size(self) -> int:
        return self.identities_per_batch * self.samples_per_identity

    def __post_init__(self) -> None:
        if self.identities_per_batch != 16 or self.samples_per_identity != 4:
            raise ValueError("FaceID requires P=16, K=4")
        if self.micro_batch_size not in {4, 8} or 64 % self.micro_batch_size:
            raise ValueError("FaceID micro batch must be 8 or 4")


__all__ = ["FaceIDConfig", "FaceIDTrainConfig"]
