"""Evidence contracts and lazily loaded optional runtime adapters."""

from importlib import import_module
from typing import Any

from evidence_fusion.base import (
    AbstractEvidencer,
    EvidenceAvailability,
    EvidenceInsufficiency,
    EvidenceObservation,
    EvidenceUnavailableReason,
    RequiredEvidenceUnavailableError,
)
from artifact_contracts.model_contract import (
    ConvNeXtModelManifest,
    DogFaceNetModelManifest,
    OnnxEvidenceContractError,
    OnnxEvidenceModelManifest,
    OnnxModelLicenseState,
    OnnxModelUsageLane,
    OnnxPreprocessingContract,
    PetReIDModelManifest,
)
from evidence_fusion.quality import (
    QualityDiagnostics,
    QualityLimits,
    QualityMapping,
    QualityObservation,
    QualityReason,
    QualityReasonCode,
    QualityState,
    estimate_blur,
    estimate_brightness,
    estimate_contrast,
    observe_quality,
    overall_quality,
    validate_roi_box,
)
from evidence_fusion.calibrator import PerChannelCalibrator
from evidence_fusion.fuser import LearnedWeightFuser
from evidence_fusion.open_set import EvidentialOpenSet
from evidence_fusion.oof_simplex import (
    OOF_SIMPLEX_SCHEMA_VERSION,
    OOFSimplexConfig,
    OOFSimplexError,
    OOFSimplexModel,
    fit_oof_simplex,
)
from evidence_fusion.temporal import TemporalAggregator

_LAZY_EXPORTS = {
    "Dinov2WithUncertainty": ("identity_methods.appearance", "Dinov2WithUncertainty"),
    "ReceiptBoundDinov2Small": (
        "identity_methods.appearance",
        "ReceiptBoundDinov2Small",
    ),
    "MiewIDArtifactManifest": ("identity_methods.backbones.miewid", "MiewIDArtifactManifest"),
    "MiewIDModelContractError": (
        "identity_methods.backbones.miewid",
        "MiewIDModelContractError",
    ),
    "MiewIDPreprocessingManifest": (
        "identity_methods.backbones.miewid",
        "MiewIDPreprocessingManifest",
    ),
    "MiewIDReIDExtractor": ("identity_methods.backbones.miewid", "MiewIDReIDExtractor"),
    "LandmarkEvidencer": ("localization.landmark_graph", "LandmarkEvidencer"),
    "HRNetHeatmap": ("localization.landmark_graph", "HRNetHeatmap"),
    "LandmarkGraphEmbedder": (
        "localization.landmark_graph",
        "LandmarkGraphEmbedder",
    ),
    "DNPMask": ("identity_methods.nose.extractor", "DNPMask"),
    "NoseDetection": ("identity_methods.nose.extractor", "NoseDetection"),
    "NosePrintExtractor": ("identity_methods.nose.extractor", "NosePrintExtractor"),
    "NoseRoiPolicy": ("identity_methods.nose.extractor", "NoseRoiPolicy"),
    "YoloNoseDetector": ("identity_methods.nose.extractor", "YoloNoseDetector"),
}

# These names were previously public but can never construct usable objects.
# Keep direct imports fail-closed without advertising them as supported API.
_DISABLED_COMPAT_EXPORTS = {
    "MiewIDNoseExtractor": ("identity_methods.nose.extractor", "MiewIDNoseExtractor"),
    "TinyViTBackbone": ("identity_methods.nose.extractor", "TinyViTBackbone"),
    "MagFaceNoseHead": ("identity_methods.nose.extractor", "MagFaceNoseHead"),
    "NoseEnhancer": ("identity_methods.nose.extractor", "NoseEnhancer"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = (_LAZY_EXPORTS | _DISABLED_COMPAT_EXPORTS)[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "AbstractEvidencer", "EvidenceAvailability", "EvidenceInsufficiency",
    "EvidenceObservation", "EvidenceUnavailableReason",
    "RequiredEvidenceUnavailableError",
    "MiewIDArtifactManifest", "MiewIDModelContractError",
    "MiewIDPreprocessingManifest", "MiewIDReIDExtractor",
    "ConvNeXtModelManifest", "DogFaceNetModelManifest",
    "OnnxEvidenceContractError", "OnnxEvidenceModelManifest",
    "OnnxModelLicenseState", "OnnxModelUsageLane",
    "OnnxPreprocessingContract", "PetReIDModelManifest",
    "DNPMask", "NoseDetection", "NosePrintExtractor", "NoseRoiPolicy",
    "YoloNoseDetector",
    "LandmarkEvidencer", "HRNetHeatmap", "LandmarkGraphEmbedder",
    "Dinov2WithUncertainty", "ReceiptBoundDinov2Small",
    "QualityDiagnostics", "QualityLimits", "QualityMapping",
    "QualityObservation", "QualityReason", "QualityReasonCode", "QualityState",
    "estimate_blur", "estimate_brightness", "estimate_contrast", "overall_quality",
    "observe_quality", "validate_roi_box",
    "PerChannelCalibrator", "LearnedWeightFuser", "EvidentialOpenSet",
    "OOF_SIMPLEX_SCHEMA_VERSION", "OOFSimplexConfig", "OOFSimplexError",
    "OOFSimplexModel", "fit_oof_simplex", "TemporalAggregator",
]
