"""Search: availability-aware weighted cosine over gallery keys.

Query, gallery-key, and gallery-value names are retrieval roles, not attention.
"""

from search.matching.pipeline import RetrievalResult, SearchPipeline

__all__ = ["RetrievalResult", "SearchPipeline"]
