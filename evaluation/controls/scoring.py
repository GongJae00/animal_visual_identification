"""Blind score receipts and sealed-label verification evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from data.pair_artifacts import (
    PairArtifactEntry,
    PairArtifactManifest,
    PairArtifactVerification,
    validate_pair_artifact_manifest,
    verify_pair_artifact_files,
)
from evaluation import (
    ClusterBootstrapConfig,
    FrozenVerificationThreshold,
    ScoredVerificationPair,
    VerificationEvaluation,
    evaluate_frozen_verification_threshold,
)
from evaluation.controls.pairing import PairConstructionResult, PairingPolicy
from foundation.provenance import content_sha256


@dataclass(frozen=True, slots=True)
class BlindPairScore:
    pair_id: str
    score: float

    def __post_init__(self) -> None:
        _require_nonempty(self.pair_id, "pair_id")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not isfinite(self.score)
        ):
            raise ValueError("score must be finite")

    def to_dict(self) -> dict[str, str | float]:
        return {"pair_id": self.pair_id, "score": self.score}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BlindPairScore:
        _require_exact_keys(payload, {"pair_id", "score"}, "blind pair score")
        return cls(pair_id=payload["pair_id"], score=payload["score"])


@dataclass(frozen=True, slots=True)
class BlindScoreReceipt:
    pair_set_sha256: str
    scoring_requests_sha256: str
    artifact_manifest_sha256: str
    model_sha256: str
    gallery_sha256: str
    inference_config_sha256: str
    dependency_lock_sha256: str
    code_revision: str
    scorer_version: str
    precision: str
    device: str
    scores: tuple[BlindPairScore, ...]
    schema_version: str = "cvi.blind_score_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.blind_score_receipt.v1":
            raise ValueError("unsupported blind score receipt schema")
        for name in (
            "pair_set_sha256",
            "scoring_requests_sha256",
            "artifact_manifest_sha256",
            "model_sha256",
            "gallery_sha256",
            "inference_config_sha256",
            "dependency_lock_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in (
            "code_revision",
            "scorer_version",
            "precision",
            "device",
        ):
            _require_nonempty(getattr(self, name), name)
        if not self.scores:
            raise ValueError("blind score receipt must contain scores")
        pair_ids = tuple(score.pair_id for score in self.scores)
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("blind score pair IDs must be unique")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair_set_sha256": self.pair_set_sha256,
            "scoring_requests_sha256": self.scoring_requests_sha256,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "model_sha256": self.model_sha256,
            "gallery_sha256": self.gallery_sha256,
            "inference_config_sha256": self.inference_config_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "code_revision": self.code_revision,
            "scorer_version": self.scorer_version,
            "precision": self.precision,
            "device": self.device,
            "scores": [score.to_dict() for score in self.scores],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BlindScoreReceipt:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "pair_set_sha256",
                "scoring_requests_sha256",
                "artifact_manifest_sha256",
                "model_sha256",
                "gallery_sha256",
                "inference_config_sha256",
                "dependency_lock_sha256",
                "code_revision",
                "scorer_version",
                "precision",
                "device",
                "scores",
            },
            "blind score receipt",
        )
        scores = payload["scores"]
        if not isinstance(scores, list):
            raise TypeError("scores must be a list")
        return cls(
            schema_version=payload["schema_version"],
            pair_set_sha256=payload["pair_set_sha256"],
            scoring_requests_sha256=payload["scoring_requests_sha256"],
            artifact_manifest_sha256=payload["artifact_manifest_sha256"],
            model_sha256=payload["model_sha256"],
            gallery_sha256=payload["gallery_sha256"],
            inference_config_sha256=payload["inference_config_sha256"],
            dependency_lock_sha256=payload["dependency_lock_sha256"],
            code_revision=payload["code_revision"],
            scorer_version=payload["scorer_version"],
            precision=payload["precision"],
            device=payload["device"],
            scores=tuple(BlindPairScore.from_dict(item) for item in scores),
        )


@dataclass(frozen=True, slots=True)
class BlindVerificationEvaluation:
    score_receipt_sha256: str
    verification: VerificationEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cvi.blind_verification_evaluation.v1",
            "score_receipt_sha256": self.score_receipt_sha256,
            "verification": self.verification.to_dict(),
        }


def join_blind_scores(
    construction: PairConstructionResult,
    artifact_manifest: PairArtifactManifest,
    receipt: BlindScoreReceipt,
) -> tuple[ScoredVerificationPair, ...]:
    """Join only after exact request/pair identity has been verified."""

    if receipt.pair_set_sha256 != construction.result_sha256:
        raise ValueError("blind score receipt pair-set hash mismatch")
    validate_pair_artifact_manifest(construction, artifact_manifest)
    if receipt.artifact_manifest_sha256 != artifact_manifest.manifest_sha256:
        raise ValueError("blind score receipt artifact manifest hash mismatch")
    expected_request_hash = content_sha256(construction.scoring_payload())
    if receipt.scoring_requests_sha256 != expected_request_hash:
        raise ValueError("blind score receipt request hash mismatch")
    request_ids = tuple(
        request.pair_id for request in construction.scoring_requests
    )
    truth_by_id = {
        truth.pair_id: truth for truth in construction.ground_truth
    }
    score_by_id = {score.pair_id: score.score for score in receipt.scores}
    if set(request_ids) != set(truth_by_id):
        raise RuntimeError("construction request and truth IDs differ")
    missing = set(request_ids) - set(score_by_id)
    extra = set(score_by_id) - set(request_ids)
    if missing or extra:
        raise ValueError(
            "blind score IDs mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return tuple(
        ScoredVerificationPair(
            pair_id=request.pair_id,
            query_track_id=request.query_artifact_token,
            reference_template_id=request.reference_artifact_token,
            query_dog_id=truth_by_id[request.pair_id].query_dog_id,
            reference_dog_id=truth_by_id[
                request.pair_id
            ].reference_dog_id,
            query_session_id=truth_by_id[
                request.pair_id
            ].query_session_id,
            reference_session_id=truth_by_id[
                request.pair_id
            ].reference_session_id,
            score=score_by_id[request.pair_id],
        )
        for request in construction.scoring_requests
    )


def evaluate_blind_score_receipt(
    construction: PairConstructionResult,
    artifact_manifest: PairArtifactManifest,
    receipt: BlindScoreReceipt,
    *,
    pairing_policy: PairingPolicy,
    threshold: FrozenVerificationThreshold,
    test_manifest_sha256: str,
    bootstrap: ClusterBootstrapConfig,
) -> BlindVerificationEvaluation:
    if pairing_policy.policy_sha256 != construction.pairing_policy_sha256:
        raise ValueError(
            "pairing policy does not match pair construction"
        )
    if threshold.direction is not pairing_policy.direction:
        raise ValueError(
            "threshold direction does not match pairing policy"
        )
    if receipt.model_sha256 != threshold.model_sha256:
        raise ValueError("score receipt model does not match threshold")
    if receipt.gallery_sha256 != threshold.gallery_sha256:
        raise ValueError("score receipt gallery does not match threshold")
    verification = evaluate_frozen_verification_threshold(
        join_blind_scores(construction, artifact_manifest, receipt),
        threshold=threshold,
        test_manifest_sha256=test_manifest_sha256,
        bootstrap=bootstrap,
    )
    return BlindVerificationEvaluation(
        score_receipt_sha256=receipt.receipt_sha256,
        verification=verification,
    )


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


__all__ = [
    "BlindPairScore",
    "BlindScoreReceipt",
    "BlindVerificationEvaluation",
    "PairArtifactEntry",
    "PairArtifactManifest",
    "PairArtifactVerification",
    "evaluate_blind_score_receipt",
    "join_blind_scores",
    "validate_pair_artifact_manifest",
    "verify_pair_artifact_files",
]
