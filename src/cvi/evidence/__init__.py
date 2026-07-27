"""Evidence contracts and lazily loaded optional runtime adapters."""

from importlib import import_module
from typing import Any

from cvi.evidence.base import (
    AbstractEvidencer,
    EvidenceAvailability,
    EvidenceInsufficiency,
    EvidenceObservation,
    EvidenceUnavailableReason,
    RequiredEvidenceUnavailableError,
)
from cvi.evidence.model_contract import (
    ConvNeXtModelManifest,
    DogFaceNetModelManifest,
    OnnxEvidenceContractError,
    OnnxEvidenceModelManifest,
    OnnxModelLicenseState,
    OnnxModelUsageLane,
    OnnxPreprocessingContract,
    PetReIDModelManifest,
)
from cvi.evidence.quality import (
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

_LAZY_EXPORTS = {
    "Dinov2WithUncertainty": ("cvi.evidence.appearance", "Dinov2WithUncertainty"),
    "ReceiptBoundDinov2Small": (
        "cvi.evidence.appearance",
        "ReceiptBoundDinov2Small",
    ),
    "MiewIDArtifactManifest": ("cvi.evidence.miewid", "MiewIDArtifactManifest"),
    "MiewIDModelContractError": (
        "cvi.evidence.miewid",
        "MiewIDModelContractError",
    ),
    "MiewIDPreprocessingManifest": (
        "cvi.evidence.miewid",
        "MiewIDPreprocessingManifest",
    ),
    "MiewIDReIDExtractor": ("cvi.evidence.miewid", "MiewIDReIDExtractor"),
    "LandmarkEvidencer": ("cvi.evidence.landmark_graph", "LandmarkEvidencer"),
    "HRNetHeatmap": ("cvi.evidence.landmark_graph", "HRNetHeatmap"),
    "LandmarkGraphEmbedder": (
        "cvi.evidence.landmark_graph",
        "LandmarkGraphEmbedder",
    ),
    "DNPMask": ("cvi.evidence.nose_print", "DNPMask"),
    "NoseDetection": ("cvi.evidence.nose_print", "NoseDetection"),
    "NosePrintExtractor": ("cvi.evidence.nose_print", "NosePrintExtractor"),
    "NoseRoiPolicy": ("cvi.evidence.nose_print", "NoseRoiPolicy"),
    "YoloNoseDetector": ("cvi.evidence.nose_print", "YoloNoseDetector"),
}

# These names were previously public but can never construct usable objects.
# Keep direct imports fail-closed without advertising them as supported API.
_DISABLED_COMPAT_EXPORTS = {
    "MiewIDNoseExtractor": ("cvi.evidence.nose_print", "MiewIDNoseExtractor"),
    "TinyViTBackbone": ("cvi.evidence.nose_print", "TinyViTBackbone"),
    "MagFaceNoseHead": ("cvi.evidence.nose_print", "MagFaceNoseHead"),
    "NoseEnhancer": ("cvi.evidence.nose_print", "NoseEnhancer"),
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
]
