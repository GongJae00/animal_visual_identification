"""ResNet18 full-segment baseline with mask-weighted feature pooling."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from contracts.pretrained_weight_intake import (
    PretrainedWeightIntakeReceipt,
    PretrainedWeightSourceContract,
    validate_pretrained_weight_receipt_binding,
)
from foundation.protected_io import read_strict_json_object
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file

_B2_SOURCE_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "contracts"
    / "configs"
    / "pretrained-weights"
    / "torchvision-resnet18-imagenet1k-v1-336d36e8.json"
)
_B2_SOURCE_CONTRACT_SHA256 = (
    "d6b36cb256ab2ecf1b16dd13ec8f929ad707c83439146e6313ee201faef04aa6"
)
_INTAKE_BUNDLE_KEYS = {
    "schema_version",
    "source_contract_sha256",
    "source_contract",
    "receipt_sha256",
    "receipt",
    "tool_provenance",
    "tool_provenance_sha256",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


class AreaMaskedGlobalAveragePool(nn.Module):
    """Pool feature channels using area-downsampled foreground occupancy."""

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = F.interpolate(mask.float(), size=features.shape[-2:], mode="area")
        denominator = weights.sum(dim=(2, 3))
        if torch.any(denominator <= 0):
            raise ValueError("every mask must cover at least one pooled feature location")
        return (features * weights.to(dtype=features.dtype)).sum(dim=(2, 3)) / denominator


class MaskedGAP128(nn.Module):
    """Three-channel ResNet18 with binary-mask weighted 512-to-128 pooling."""

    output_dim = 128
    feature_channels = 512

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = AreaMaskedGlobalAveragePool()
        self.projection = nn.Linear(self.feature_channels, self.output_dim)
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )
        self.initialization = "RANDOM_SCRATCH"
        self.initialization_sha256: str | None = None
        self.initialization_source_contract_sha256: str | None = None
        self.initialization_intake_receipt_sha256: str | None = None
        self.initialization_usage_lane: str | None = None

    @classmethod
    def from_supervised_imagenet(
        cls,
        checkpoint_path: str | Path,
        *,
        intake_bundle_path: str | Path,
    ) -> MaskedGAP128:
        """Load the canonical B2 weights through their source and intake binding."""

        path = Path(checkpoint_path)
        source, receipt = _load_b2_intake_bundle(Path(intake_bundle_path))
        if path.name != source.weight_filename:
            raise ValueError("supervised ImageNet checkpoint filename differs")
        retained = read_retained_regular_file(
            path,
            expected_bytes=source.expected_file_bytes,
            expected_sha256=source.expected_sha256,
            maximum_bytes=source.expected_file_bytes,
            capture_payload=True,
            subject="supervised ImageNet checkpoint",
        )
        if (
            retained.sha256 != receipt.weight_sha256
            or retained.byte_count != receipt.weight_bytes
        ):
            raise ValueError("supervised ImageNet checkpoint receipt binding differs")
        if retained.payload is None:  # pragma: no cover - guaranteed by capture_payload
            raise RuntimeError("supervised ImageNet checkpoint payload was not retained")
        state_dict = torch.load(
            io.BytesIO(retained.payload), map_location="cpu", weights_only=True
        )
        model = cls._from_backbone_state_dict(state_dict)
        model.initialization = "SUPERVISED_IMAGENET"
        model.initialization_sha256 = retained.sha256
        model.initialization_source_contract_sha256 = source.contract_sha256
        model.initialization_intake_receipt_sha256 = receipt.receipt_sha256
        model.initialization_usage_lane = receipt.admitted_lane.value
        return model

    @classmethod
    def from_backbone_state_dict_for_testing(
        cls, state_dict: Mapping[str, torch.Tensor]
    ) -> MaskedGAP128:
        """Load structural test weights without granting pretrained provenance."""

        model = cls._from_backbone_state_dict(state_dict)
        model.initialization = "STRUCTURAL_TEST_ONLY"
        return model

    @classmethod
    def _from_backbone_state_dict(cls, state_dict: object) -> MaskedGAP128:
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("ResNet18 checkpoint must be a state dict")
        if any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state_dict.items()
        ):
            raise ValueError("ResNet18 checkpoint state dict is invalid")
        model = cls()
        backbone = resnet18(weights=None)
        backbone.load_state_dict(state_dict, strict=True)
        model.features = nn.Sequential(*list(backbone.children())[:-2])
        return model

    def forward(self, rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("MaskedGAP128 RGB must have shape [B,3,H,W]")
        if mask.shape != (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]):
            raise ValueError("MaskedGAP128 mask must have shape [B,1,H,W]")
        if min(rgb.shape[-2:]) < 32:
            raise ValueError("MaskedGAP128 spatial dimensions must be at least 32")
        if not torch.isfinite(rgb).all() or not torch.isfinite(mask).all():
            raise ValueError("MaskedGAP128 inputs must be finite")
        if torch.any((rgb < 0) | (rgb > 1)):
            raise ValueError("MaskedGAP128 RGB must be in [0,1]")
        if torch.any((mask != 0) & (mask != 1)):
            raise ValueError("MaskedGAP128 mask must be binary")
        if torch.any(mask.sum(dim=(1, 2, 3)) <= 0):
            raise ValueError("MaskedGAP128 masks must contain foreground")

        mask = mask.to(device=rgb.device, dtype=rgb.dtype)
        mean = self.image_mean.to(dtype=rgb.dtype)
        std = self.image_std.to(dtype=rgb.dtype)
        neutral_rgb = rgb * mask + mean * (1.0 - mask)
        features = self.features((neutral_rgb - mean) / std)
        pooled = self.pool(features, mask)
        embedding = self.projection(pooled)
        norms = torch.linalg.vector_norm(embedding.float(), dim=1, keepdim=True)
        if torch.any(norms <= 1e-12) or not torch.isfinite(norms).all():
            raise RuntimeError("MaskedGAP128 projection produced an invalid embedding")
        return embedding / norms.to(dtype=embedding.dtype)


def _load_b2_intake_bundle(
    path: Path,
) -> tuple[PretrainedWeightSourceContract, PretrainedWeightIntakeReceipt]:
    source = PretrainedWeightSourceContract.from_dict(
        read_strict_json_object(_B2_SOURCE_CONTRACT_PATH)
    )
    if source.contract_sha256 != _B2_SOURCE_CONTRACT_SHA256:
        raise ValueError("B2 repository source contract hash differs")

    bundle = read_strict_json_object(path)
    if set(bundle) != _INTAKE_BUNDLE_KEYS or bundle["schema_version"] != (
        "cvi.pretrained_weight_intake_bundle.v1"
    ):
        raise ValueError("B2 pretrained weight intake bundle schema differs")
    bundled_source = PretrainedWeightSourceContract.from_dict(bundle["source_contract"])
    receipt = PretrainedWeightIntakeReceipt.from_dict(bundle["receipt"])
    if bundled_source != source or bundle["source_contract_sha256"] != (
        source.contract_sha256
    ):
        raise ValueError("B2 pretrained weight source contract binding differs")
    if bundle["receipt_sha256"] != receipt.receipt_sha256:
        raise ValueError("B2 pretrained weight receipt hash differs")
    tool_provenance = bundle["tool_provenance"]
    if not isinstance(tool_provenance, dict) or content_sha256(tool_provenance) != (
        bundle["tool_provenance_sha256"]
    ):
        raise ValueError("B2 pretrained weight tool provenance hash differs")
    validate_pretrained_weight_receipt_binding(receipt, source)
    return source, receipt


__all__ = ["AreaMaskedGlobalAveragePool", "MaskedGAP128", "file_sha256"]
