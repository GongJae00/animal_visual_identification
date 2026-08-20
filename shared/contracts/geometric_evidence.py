"""Structural contracts consumed from geometric verifier evidence."""

from __future__ import annotations

from typing import Protocol


class GeometricDecisionEvidence(Protocol):
    @property
    def value(self) -> str: ...


class GeometricPairEvidence(Protocol):
    @property
    def left_opaque_sample_id(self) -> str: ...

    @property
    def right_opaque_sample_id(self) -> str: ...

    @property
    def decision(self) -> GeometricDecisionEvidence: ...

    @property
    def reason(self) -> GeometricDecisionEvidence: ...

    @property
    def evidence_token(self) -> str: ...


class GeometricPolicyEvidence(Protocol):
    @property
    def policy_sha256(self) -> str: ...


class GeometricVerifierEvidenceContract(Protocol):
    @property
    def evidence_sha256(self) -> str: ...

    @property
    def policy(self) -> GeometricPolicyEvidence: ...

    @property
    def results(self) -> tuple[GeometricPairEvidence, ...]: ...


__all__ = ["GeometricVerifierEvidenceContract"]
