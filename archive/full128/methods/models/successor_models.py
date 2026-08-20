"""Full-view successor models grounded in the executable Full128 baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from shared.contracts.dinov2_contract import Dinov2LocalArtifactContract
from archive.full128.methods.models.classical import Classical128
from archive.full128.methods.models.model import MaskedGAP128

DINOV2_PATCH_DIMENSION = 384
DINOV2_PATCH_SIZE = 14
FULL_VIEW_OUTPUT_DIMENSION = 128


def _finite_l2(values: torch.Tensor, *, subject: str) -> torch.Tensor:
    values = values.float()
    if values.ndim != 2 or values.shape[1] != FULL_VIEW_OUTPUT_DIMENSION:
        raise RuntimeError(f"{subject} must produce [B,128]")
    if not torch.isfinite(values).all():
        raise RuntimeError(f"{subject} produced non-finite values")
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if torch.any(norms <= 1e-12):
        raise RuntimeError(f"{subject} produced a zero embedding")
    return values / norms


def _validate_tokens(
    tokens: torch.Tensor, occupancy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        tokens.ndim != 3
        or tokens.shape[2] != DINOV2_PATCH_DIMENSION
        or occupancy.shape != tokens.shape[:2]
    ):
        raise ValueError("DINOv2 tokens and occupancy must be [B,N,384] and [B,N]")
    if not torch.isfinite(tokens).all() or not torch.isfinite(occupancy).all():
        raise ValueError("DINOv2 tokens and occupancy must be finite")
    occupancy = occupancy.float()
    if torch.any((occupancy < 0) | (occupancy > 1)):
        raise ValueError("DINOv2 patch occupancy must be in [0,1]")
    if torch.any(occupancy.sum(dim=1) <= 0):
        raise ValueError("every sample must occupy at least one DINOv2 patch")
    return tokens, occupancy


def occupancy_pool(tokens: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
    """Pool patch tokens using fractional foreground area as exact weights."""

    tokens, occupancy = _validate_tokens(tokens, occupancy)
    weights = occupancy / occupancy.sum(dim=1, keepdim=True)
    return (tokens * weights.to(tokens.dtype).unsqueeze(-1)).sum(dim=1)


class ClassicalFV128:
    """B0-FV compatibility wrapper around the fitted Full128 classical model."""

    output_dim = FULL_VIEW_OUTPUT_DIMENSION

    def __init__(self, model: Classical128) -> None:
        if not isinstance(model, Classical128):
            raise TypeError("B0-FV requires a Classical128 instance")
        self.model = model

    @classmethod
    def load_state(cls, path: Path) -> ClassicalFV128:
        return cls(Classical128.load_state(path))

    def transform(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._validate(self.model.transform(rgb, mask)[None, :])[0]

    def transform_batch(
        self, rgbs: Sequence[np.ndarray], masks: Sequence[np.ndarray]
    ) -> np.ndarray:
        return self._validate(self.model.transform_batch(rgbs, masks))

    @staticmethod
    def _validate(values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values)
        if matrix.ndim != 2 or matrix.shape[1] != 128 or not np.isfinite(matrix).all():
            raise RuntimeError("B0-FV must produce finite [N,128] embeddings")
        matrix = matrix.astype(np.float32, copy=False)
        if not np.allclose(
            np.linalg.norm(matrix.astype(np.float64), axis=1),
            1.0,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError("B0-FV embeddings must be L2 normalized")
        return matrix


class MaskedGAP128FV(MaskedGAP128):
    """Full-view MaskedGAP128 with an explicit float32 output boundary."""

    def forward(self, rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _finite_l2(super().forward(rgb, mask), subject="B1/B2-FV")


def build_b1_fv() -> MaskedGAP128FV:
    """Build a fresh random B1-FV MaskedGAP128."""

    return MaskedGAP128FV()


def build_b2_fv(checkpoint_path: Path, *, intake_bundle_path: Path) -> MaskedGAP128FV:
    """Build B2-FV only from the receipt-bound ImageNet source state.

    This deliberately accepts no learned B1/B2 checkpoint or state dictionary.
    """

    return MaskedGAP128FV.from_supervised_imagenet(
        checkpoint_path, intake_bundle_path=intake_bundle_path
    )


def load_receipt_bound_dinov2_patch_backbone(
    *,
    model_directory: Path,
    weight_intake_bundle: Path,
    preprocessor_intake_bundle: Path,
) -> tuple[nn.Module, Dinov2LocalArtifactContract]:
    """Load the existing exact local DINOv2-small receipt contract."""

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
    if not isinstance(backbone, nn.Module):
        raise TypeError("local DINOv2 loader must return torch.nn.Module")
    contract.revalidate_local_files()
    return backbone, contract


class _FrozenDinov2Tokens(nn.Module):
    """Shared frozen DINOv2 patch extraction and mask occupancy calculation."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("DINOv2 backbone must be torch.nn.Module")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> _FrozenDinov2Tokens:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self, rgb: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("DINOv2 full-view RGB must have shape [B,3,H,W]")
        if mask.shape != (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]):
            raise ValueError("DINOv2 full-view mask must have shape [B,1,H,W]")
        height, width = rgb.shape[-2:]
        if height % DINOV2_PATCH_SIZE or width % DINOV2_PATCH_SIZE:
            raise ValueError("DINOv2 full-view dimensions must be divisible by 14")
        if not torch.isfinite(rgb).all() or not torch.isfinite(mask).all():
            raise ValueError("DINOv2 full-view inputs must be finite")
        if torch.any((rgb < 0) | (rgb > 1)):
            raise ValueError("DINOv2 full-view RGB must be in [0,1]")
        if torch.any((mask != 0) & (mask != 1)):
            raise ValueError("DINOv2 full-view masks must be binary")
        if torch.any(mask.sum(dim=(1, 2, 3)) <= 0):
            raise ValueError("DINOv2 full-view masks must contain foreground")

        mask = mask.to(device=rgb.device, dtype=rgb.dtype)
        mean = self.image_mean.to(dtype=rgb.dtype)
        std = self.image_std.to(dtype=rgb.dtype)
        pixels = (rgb * mask + mean * (1.0 - mask) - mean) / std
        with torch.no_grad():
            output = self.backbone(pixel_values=pixels, interpolate_pos_encoding=True)
        hidden = (
            output
            if isinstance(output, torch.Tensor)
            else getattr(output, "last_hidden_state", None)
        )
        expected_patches = (height // DINOV2_PATCH_SIZE) * (width // DINOV2_PATCH_SIZE)
        if not isinstance(hidden, torch.Tensor) or hidden.shape != (
            rgb.shape[0],
            expected_patches + 1,
            DINOV2_PATCH_DIMENSION,
        ):
            raise RuntimeError(
                "DINOv2-small must return one CLS plus 384D patch tokens"
            )
        occupancy = F.interpolate(
            mask.float(),
            size=(height // DINOV2_PATCH_SIZE, width // DINOV2_PATCH_SIZE),
            mode="area",
        ).flatten(1)
        return hidden[:, 1:, :].detach(), occupancy.detach()


@dataclass(frozen=True, slots=True)
class PatchRepresentationDecomposition:
    """Executable patch-to-embedding components used by production traces."""

    effective_tokens: torch.Tensor
    occupancy: torch.Tensor
    logits: torch.Tensor | None
    weights: torch.Tensor
    pooled: torch.Tensor
    projected: torch.Tensor
    embedding: torch.Tensor


class Dinov2OccupancyProbe128(nn.Module):
    """B3: frozen DINOv2-small occupancy pooling and trainable 384-to-128 probe."""

    output_dim = FULL_VIEW_OUTPUT_DIMENSION

    def __init__(
        self, backbone: nn.Module, *, projection: nn.Linear | None = None
    ) -> None:
        super().__init__()
        self.tokens = _FrozenDinov2Tokens(backbone)
        self.projection = projection or nn.Linear(
            DINOV2_PATCH_DIMENSION, FULL_VIEW_OUTPUT_DIMENSION
        )
        if self.projection.in_features != 384 or self.projection.out_features != 128:
            raise ValueError("B3 projection must be Linear(384,128)")

    def extract_tokens(
        self, rgb: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens(rgb, mask)

    def forward_from_tokens(
        self, tokens: torch.Tensor, occupancy: torch.Tensor
    ) -> torch.Tensor:
        return self.decompose_representation(tokens, occupancy).embedding

    def decompose_representation(
        self, tokens: torch.Tensor, occupancy: torch.Tensor
    ) -> PatchRepresentationDecomposition:
        """Return the exact tensors consumed by the B3 embedding path."""

        tokens, occupancy = _validate_tokens(tokens, occupancy)
        weights = occupancy / occupancy.sum(dim=1, keepdim=True)
        pooled = (tokens * weights.to(tokens.dtype).unsqueeze(-1)).sum(dim=1)
        projected = self.projection(pooled)
        embedding = _finite_l2(projected, subject="B3-FV")
        return PatchRepresentationDecomposition(
            effective_tokens=tokens,
            occupancy=occupancy,
            logits=None,
            weights=weights,
            pooled=pooled,
            projected=projected,
            embedding=embedding,
        )

    def forward(self, rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward_from_tokens(*self.extract_tokens(rgb, mask))


class _ResidualTokenAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(DINOV2_PATCH_DIMENSION)
        self.down = nn.Linear(DINOV2_PATCH_DIMENSION, 128)
        self.up = nn.Linear(128, DINOV2_PATCH_DIMENSION)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.up(F.gelu(self.down(self.norm(tokens))))


class IdentityBlindResidualTokenAdapter128(nn.Module):
    """B4: zero-init token adapter with frozen DINOv2 and B3 projection."""

    output_dim = FULL_VIEW_OUTPUT_DIMENSION

    def __init__(self, backbone: nn.Module, projection: nn.Linear) -> None:
        super().__init__()
        self.tokens = _FrozenDinov2Tokens(backbone)
        self.projection = projection
        if self.projection.in_features != 384 or self.projection.out_features != 128:
            raise ValueError("B4 projection must be Linear(384,128)")
        for parameter in self.projection.parameters():
            parameter.requires_grad_(False)
        self.projection.eval()
        self.adapter = _ResidualTokenAdapter()

    def train(self, mode: bool = True) -> IdentityBlindResidualTokenAdapter128:
        super().train(mode)
        self.tokens.backbone.eval()
        self.projection.eval()
        return self

    def forward_from_tokens(
        self, tokens: torch.Tensor, occupancy: torch.Tensor
    ) -> torch.Tensor:
        tokens, occupancy = _validate_tokens(tokens, occupancy)
        pooled = occupancy_pool(self.adapter(tokens), occupancy)
        return _finite_l2(self.projection(pooled), subject="B4-FV")

    def forward(self, rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward_from_tokens(*self.tokens(rgb, mask))


@dataclass(frozen=True, slots=True)
class SpatialWeightDecomposition:
    """Auditable B5 occupancy, scorer-logit, and normalized-weight components."""

    occupancy: torch.Tensor
    logits: torch.Tensor
    weights: torch.Tensor


class SpatialScorer128(nn.Module):
    """B5: zero-init patch scorer with uniform and channel-gate controls."""

    output_dim = FULL_VIEW_OUTPUT_DIMENSION

    def __init__(
        self,
        backbone: nn.Module,
        projection: nn.Linear,
        *,
        token_adapter: nn.Module | None = None,
        uniform_spatial: bool = False,
        channel_gate: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(uniform_spatial, bool) or not isinstance(channel_gate, bool):
            raise TypeError("B5 controls must be boolean")
        self.tokens = _FrozenDinov2Tokens(backbone)
        self.projection = projection
        if self.projection.in_features != 384 or self.projection.out_features != 128:
            raise ValueError("B5 projection must be Linear(384,128)")
        for parameter in self.projection.parameters():
            parameter.requires_grad_(False)
        self.projection.eval()
        self.token_adapter = token_adapter or nn.Identity()
        for parameter in self.token_adapter.parameters():
            parameter.requires_grad_(False)
        self.token_adapter.eval()
        self.uniform_spatial = uniform_spatial
        self.scorer = nn.Linear(DINOV2_PATCH_DIMENSION, 1)
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)
        if uniform_spatial:
            for parameter in self.scorer.parameters():
                parameter.requires_grad_(False)
        if channel_gate:
            self.channel_gate = nn.Parameter(torch.zeros(DINOV2_PATCH_DIMENSION))
        else:
            self.register_parameter("channel_gate", None)

    def train(self, mode: bool = True) -> SpatialScorer128:
        super().train(mode)
        self.tokens.backbone.eval()
        self.projection.eval()
        self.token_adapter.eval()
        return self

    def decompose_weights(
        self, tokens: torch.Tensor, occupancy: torch.Tensor
    ) -> SpatialWeightDecomposition:
        tokens, occupancy = _validate_tokens(tokens, occupancy)
        parent_tokens = self.token_adapter(tokens)
        logits = (
            torch.zeros_like(occupancy)
            if self.uniform_spatial
            else self.scorer(parent_tokens).squeeze(-1).float()
        )
        if not torch.isfinite(logits).all():
            raise RuntimeError("B5-FV scorer produced non-finite logits")
        shifted = logits - logits.max(dim=1, keepdim=True).values
        unnormalized = occupancy * torch.exp(shifted)
        weights = unnormalized / unnormalized.sum(dim=1, keepdim=True)
        return SpatialWeightDecomposition(occupancy, logits, weights)

    def forward_from_tokens(
        self,
        tokens: torch.Tensor,
        occupancy: torch.Tensor,
        *,
        return_decomposition: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SpatialWeightDecomposition]:
        representation = self.decompose_representation(tokens, occupancy)
        if return_decomposition:
            return representation.embedding, SpatialWeightDecomposition(
                representation.occupancy,
                representation.logits,
                representation.weights,
            )
        return representation.embedding

    def decompose_representation(
        self, tokens: torch.Tensor, occupancy: torch.Tensor
    ) -> PatchRepresentationDecomposition:
        """Return the exact tensors consumed by the B5 embedding path."""

        decomposition = self.decompose_weights(tokens, occupancy)
        effective_tokens = self.token_adapter(tokens)
        if self.channel_gate is not None:
            effective_tokens = effective_tokens * (
                1.0 + torch.tanh(self.channel_gate)
            ).to(tokens.dtype)
        pooled = (
            effective_tokens * decomposition.weights.to(tokens.dtype).unsqueeze(-1)
        ).sum(dim=1)
        projected = self.projection(pooled)
        embedding = _finite_l2(projected, subject="B5-FV")
        return PatchRepresentationDecomposition(
            effective_tokens=effective_tokens,
            occupancy=decomposition.occupancy,
            logits=decomposition.logits,
            weights=decomposition.weights,
            pooled=pooled,
            projected=projected,
            embedding=embedding,
        )

    def forward(self, rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens, occupancy = self.tokens(rgb, mask)
        output = self.forward_from_tokens(tokens, occupancy)
        assert isinstance(output, torch.Tensor)
        return output


def parameter_partition(model: nn.Module) -> dict[str, tuple[str, ...]]:
    """Return stable trainable/frozen parameter names for receipts and tests."""

    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        (trainable if parameter.requires_grad else frozen).append(name)
    return {"trainable": tuple(trainable), "frozen": tuple(frozen)}


def dinov2_contract_bindings(
    contract: Dinov2LocalArtifactContract,
) -> dict[str, str]:
    """Return the complete DINOv2 receipt binding used by successor artifacts."""

    return {
        "model_sha256": contract.model_sha256,
        "weight_source_contract_sha256": contract.weight_source.contract_sha256,
        "weight_intake_receipt_sha256": contract.weight_receipt_sha256,
        "preprocessor_sha256": contract.preprocessor_sha256,
        "preprocessor_source_contract_sha256": (
            contract.preprocessor_source.contract_sha256
        ),
        "preprocessor_intake_receipt_sha256": (contract.preprocessor_receipt_sha256),
        "config_sha256": contract.config_sha256,
        "usage_lane": contract.weight_receipt.admitted_lane.value,
    }


def reject_identity_metadata(batch: Mapping[str, Any]) -> None:
    """Fail closed if identity-bearing fields enter the B4 SSL boundary."""

    forbidden = {
        "identity",
        "identity_id",
        "identity_ids",
        "label",
        "labels",
        "class_id",
        "track_id",
        "source_label",
    }
    observed = {str(key).casefold() for key in batch}
    if observed & forbidden:
        raise ValueError("B4 identity-blind batches reject identity metadata")


__all__ = [
    "DINOV2_PATCH_DIMENSION",
    "DINOV2_PATCH_SIZE",
    "FULL_VIEW_OUTPUT_DIMENSION",
    "ClassicalFV128",
    "Dinov2OccupancyProbe128",
    "IdentityBlindResidualTokenAdapter128",
    "MaskedGAP128FV",
    "PatchRepresentationDecomposition",
    "SpatialScorer128",
    "SpatialWeightDecomposition",
    "build_b1_fv",
    "build_b2_fv",
    "dinov2_contract_bindings",
    "load_receipt_bound_dinov2_patch_backbone",
    "occupancy_pool",
    "parameter_partition",
    "reject_identity_metadata",
]
