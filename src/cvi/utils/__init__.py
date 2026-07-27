"""Shared utilities: model paths, ONNX helpers, metrics.

Re-exports from canonical cvi.model_paths for discoverability.
"""

from cvi.utils.model_paths import (
    DOGFLW_LANDMARK_PATH,
    MODELS_DIR,
    SUPERANIMAL_QUADRUPED_PATH,
)
from cvi.utils.metrics import cosine_similarity, l2_normalize

__all__ = [
    "MODELS_DIR", "SUPERANIMAL_QUADRUPED_PATH",
    "DOGFLW_LANDMARK_PATH",
    "cosine_similarity", "l2_normalize",
]
