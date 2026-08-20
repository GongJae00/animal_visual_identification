"""Pair verification curves, operating thresholds, and summary metrics."""

from evaluation.verification.metrics import (
    EmptyInputError,
    EvaluationError,
    InvalidLabelError,
    LengthMismatchError,
    NonFiniteScoreError,
    OperatingThreshold,
    SingleClassError,
    VerificationCurve,
    compute_verification_curve,
    compute_verification_metrics,
    evaluate_at_threshold,
    select_threshold_at_far,
)
