"""Episode-weighted dog/domain association diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
from typing import Literal

from cvi.dataset import TrackletRecord

DomainKey = Literal["camera_id", "cage_id", "site_id"]


@dataclass(frozen=True, slots=True)
class AssociationAudit:
    domain_key: DomainKey
    episode_identity_events: int
    identities: int
    domains: int
    mutual_information_nats: float
    normalized_mutual_information: float | None
    domain_to_identity_majority_accuracy: float
    global_identity_majority_accuracy: float
    identity_to_domain_concentration: float

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "domain_key": self.domain_key,
            "episode_identity_events": self.episode_identity_events,
            "identities": self.identities,
            "domains": self.domains,
            "mutual_information_nats": self.mutual_information_nats,
            "normalized_mutual_information": self.normalized_mutual_information,
            "domain_to_identity_majority_accuracy": (
                self.domain_to_identity_majority_accuracy
            ),
            "global_identity_majority_accuracy": (
                self.global_identity_majority_accuracy
            ),
            "identity_to_domain_concentration": (
                self.identity_to_domain_concentration
            ),
        }


def association_audit(
    records: tuple[TrackletRecord, ...],
    domain_key: DomainKey,
) -> AssociationAudit:
    """Audit dog/domain association with one vote per episode and dog."""

    if domain_key not in {"camera_id", "cage_id", "site_id"}:
        raise ValueError("domain_key must be camera_id, cage_id, or site_id")
    track_assignments: dict[tuple[str, str, str], tuple[str, str]] = {}
    event_assignments: dict[
        tuple[tuple[str, str, str, str], str],
        str,
    ] = {}
    for record in records:
        domain = str(getattr(record, domain_key))
        assignment = (record.registered_dog_id, domain)
        prior_track = track_assignments.setdefault(record.track_key, assignment)
        if prior_track != assignment:
            raise ValueError(
                f"track {record.track_key!r} has conflicting identity/domain labels"
            )
        event_key = (record.episode_key, record.registered_dog_id)
        prior_domain = event_assignments.setdefault(event_key, domain)
        if prior_domain != domain:
            raise ValueError(
                f"episode identity event {event_key!r} has conflicting domains"
            )
    if not event_assignments:
        raise ValueError("at least one episode identity event is required")

    pairs = tuple(
        (event_key[1], domain)
        for event_key, domain in event_assignments.items()
    )
    pair_counts = Counter(pairs)
    identity_counts = Counter(identity for identity, _ in pairs)
    domain_counts = Counter(domain for _, domain in pairs)
    identities_by_domain: dict[str, Counter[str]] = {}
    domains_by_identity: dict[str, Counter[str]] = {}
    for (identity, domain), count in pair_counts.items():
        identities_by_domain.setdefault(domain, Counter())[identity] = count
        domains_by_identity.setdefault(identity, Counter())[domain] = count
    total = len(pairs)
    mutual_information = 0.0
    for (identity, domain), count in pair_counts.items():
        probability = count / total
        mutual_information += probability * log(
            count * total / (identity_counts[identity] * domain_counts[domain])
        )
    identity_entropy = _entropy(identity_counts, total)
    domain_entropy = _entropy(domain_counts, total)
    normalized = (
        mutual_information / sqrt(identity_entropy * domain_entropy)
        if identity_entropy > 0 and domain_entropy > 0
        else None
    )
    domain_majority_correct = sum(
        max(counts.values()) for counts in identities_by_domain.values()
    )
    identity_domain_majority = sum(
        max(counts.values()) for counts in domains_by_identity.values()
    )
    return AssociationAudit(
        domain_key=domain_key,
        episode_identity_events=total,
        identities=len(identity_counts),
        domains=len(domain_counts),
        mutual_information_nats=mutual_information,
        normalized_mutual_information=normalized,
        domain_to_identity_majority_accuracy=domain_majority_correct / total,
        global_identity_majority_accuracy=max(identity_counts.values()) / total,
        identity_to_domain_concentration=identity_domain_majority / total,
    )


def _entropy(counts: Counter[str], total: int) -> float:
    return -sum(
        (count / total) * log(count / total)
        for count in counts.values()
        if count
    )
