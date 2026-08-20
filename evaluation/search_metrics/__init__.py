"""Closed-set search metrics. Not IdentityEngine search."""

from evaluation.search_metrics.metrics import (
    ClosedSetViolation, EmbeddingNormError, MetricInvariantError,
    MissingGalleryIdentityError, NonFiniteEmbeddingError, RetrievalError,
    SampleIdValidationError, SelfMatchPolicy, TemplateAggregation,
    compute_cosine_score_matrix, compute_retrieval_metrics,
    evaluate_multi_template_closed_set, identity_clustered_bootstrap_ci,
)
