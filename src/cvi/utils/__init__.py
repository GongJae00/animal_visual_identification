"""Shared utilities: model paths, ONNX helpers, metrics.

Re-exports from canonical cvi.model_paths for discoverability.
"""

from cvi.model_paths import MODELS_DIR, MIEWID_NOSE_ONNX_PATH, SUPERANIMAL_QUADRUPED_PATH, DOGFLW_LANDMARK_PATH
from cvi.utils.metrics import cosine_similarity, l2_normalize

__all__ = [
    "MODELS_DIR", "MIEWID_NOSE_ONNX_PATH", "SUPERANIMAL_QUADRUPED_PATH",
    "DOGFLW_LANDMARK_PATH",
    "cosine_similarity", "l2_normalize",
]
