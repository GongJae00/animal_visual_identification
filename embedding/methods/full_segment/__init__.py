"""Full128 baseline and successor embedding infrastructure."""

from embedding.methods.full_segment.models.classical import Classical128, ClassicalFitInput
from embedding.methods.full_segment.preparation.data import (
    Full128Inventory,
    Full128Sample,
    Full128TorchDataset,
    load_full128_assembly,
    read_full128_crop,
)
from embedding.methods.full_segment.training.losses import batch_hard_triplet_loss
from embedding.methods.full_segment.training.manifests import (
    BASELINE_FAMILY_SCHEMA,
    BASELINE_VARIANTS,
    build_baseline_family_manifest,
    build_checkpoint_manifest,
    build_embedding_manifest,
    build_model_manifest,
    build_preprocessing_manifest,
    manifest_sha256,
)
from embedding.methods.full_segment.models.model import MaskedGAP128
from embedding.methods.full_segment.training.sampler import (
    DEFAULT_GROUP_QUOTAS,
    DatasetViewBalancedPKSampler,
    SampleProvenance,
)
from embedding.methods.full_segment.training.statistics import FeatureOutputStatisticsHooks
from embedding.methods.full_segment.models.successor_models import (
    ClassicalFV128,
    Dinov2OccupancyProbe128,
    IdentityBlindResidualTokenAdapter128,
    MaskedGAP128FV,
    SpatialScorer128,
    build_b1_fv,
    build_b2_fv,
)

__all__ = [
    "BASELINE_FAMILY_SCHEMA",
    "BASELINE_VARIANTS",
    "DEFAULT_GROUP_QUOTAS",
    "Classical128",
    "ClassicalFV128",
    "ClassicalFitInput",
    "DatasetViewBalancedPKSampler",
    "Dinov2OccupancyProbe128",
    "FeatureOutputStatisticsHooks",
    "Full128Inventory",
    "Full128Sample",
    "Full128TorchDataset",
    "IdentityBlindResidualTokenAdapter128",
    "MaskedGAP128",
    "MaskedGAP128FV",
    "SampleProvenance",
    "SpatialScorer128",
    "batch_hard_triplet_loss",
    "build_b1_fv",
    "build_b2_fv",
    "build_baseline_family_manifest",
    "build_checkpoint_manifest",
    "build_embedding_manifest",
    "build_model_manifest",
    "build_preprocessing_manifest",
    "load_full128_assembly",
    "manifest_sha256",
    "read_full128_crop",
]
