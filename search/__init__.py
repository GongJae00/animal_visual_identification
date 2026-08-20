"""Search: availability-aware weighted cosine over gallery keys.

Query, gallery-key, and gallery-value names are retrieval roles, not attention.
"""

from search.matching.pipeline import RetrievalResult, SearchPipeline
from search.scoring.roles import (
    AvailableIntersectionScorer,
    GalleryKey,
    GalleryValue,
    RetrievalQuery,
)

__all__ = [
    "AvailableIntersectionScorer",
    "GalleryKey",
    "GalleryValue",
    "SearchPipeline",
    "RetrievalQuery",
    "RetrievalResult",
]
