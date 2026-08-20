"""Fail-closed admission policy for registered-only gallery identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from uuid import UUID

from search.scoring.roles import IdentityEvidenceKind

_IDENTITY_POLICY_SCHEMA = "cvi.gallery_identity_policy.v1"
_REGISTERED_ONLY = "REGISTERED_ONLY"


def _is_canonical_uuid5(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return parsed.version == 5 and str(parsed) == value


@dataclass(frozen=True, slots=True)
class IdentityRegistryPolicy:
    """Fail-closed admission policy for registered-only gallery identities."""

    registered_identity_ids: frozenset[str] | None = None
    provisional_generated_identity_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        registered = self.registered_identity_ids
        if registered is not None and not isinstance(registered, frozenset):
            registered = frozenset(registered)
            object.__setattr__(self, "registered_identity_ids", registered)
        provisional = self.provisional_generated_identity_ids
        if not isinstance(provisional, frozenset):
            provisional = frozenset(provisional)
            object.__setattr__(self, "provisional_generated_identity_ids", provisional)
        for values in (registered, provisional):
            if values is not None and any(not _is_canonical_uuid5(value) for value in values):
                raise ValueError("identity registry policy IDs must be canonical UUIDv5")
        if registered is not None and registered & provisional:
            raise ValueError("identity registry policy namespaces overlap")

    @property
    def descriptor(self) -> dict[str, str | None]:
        digest: str | None = None
        if self.registered_identity_ids is not None or self.provisional_generated_identity_ids:
            payload = {
                "registered": sorted(self.registered_identity_ids or ()),
                "provisional_genid": sorted(self.provisional_generated_identity_ids),
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
            ).hexdigest()
        return {
            "schema_version": _IDENTITY_POLICY_SCHEMA,
            "mode": _REGISTERED_ONLY,
            "registry_sha256": digest,
        }

    def validate(self, identity_id: str, kind: IdentityEvidenceKind) -> None:
        if kind is not IdentityEvidenceKind.REGISTERED:
            raise ValueError("registered-only gallery rejects provisional GenID evidence")
        if identity_id in self.provisional_generated_identity_ids:
            raise ValueError("registered-only gallery rejects a registry-known provisional GenID")
        if (
            self.registered_identity_ids is not None
            and identity_id not in self.registered_identity_ids
        ):
            raise ValueError("registered identity is absent from the configured registry")
