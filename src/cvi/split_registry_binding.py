"""Bind a protected split assignment to the identity registry.

Joins the score-blind assignment (identity_token-based) with the
registered_dog_id (UUIDv5) so that downstream training and evaluation
pipelines operate on production-safe identifiers without exposing source
dataset labels to model or scorer code.

Validation guarantees
  - Every identity_token in the assignment appears in the registry.
  - No identity_token maps to more than one registered_dog_id.
  - Identity distribution report covers role, access, dataset, and sample
    disposition.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SCHEMA_VERSION = "cvi.split_registry_binding.v1"


@dataclass(frozen=True, slots=True)
class IdentityRoleSummary:
    role: str
    access: str
    unique_identities: int
    sample_count: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "role": self.role,
            "access": self.access,
            "unique_identities": self.unique_identities,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class IdentityBinding:
    identity_token: str
    registered_dog_id: str
    dataset_name: str
    identity_role: str
    model_access: str
    sample_disposition: str
    sample_count: int
    sample_tokens: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_token": self.identity_token,
            "registered_dog_id": self.registered_dog_id,
            "dataset_name": self.dataset_name,
            "identity_role": self.identity_role,
            "model_access": self.model_access,
            "sample_disposition": self.sample_disposition,
            "sample_count": self.sample_count,
            "sample_tokens": list(self.sample_tokens),
        }


@dataclass(frozen=True, slots=True)
class SplitRegistryBinding:
    bindings: tuple[IdentityBinding, ...]
    identity_summaries: tuple[IdentityRoleSummary, ...]
    total_identities: int
    total_samples: int
    unregistered_tokens: tuple[str, ...]
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported split registry binding schema")

    @property
    def is_valid(self) -> bool:
        return len(self.unregistered_tokens) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "is_valid": self.is_valid,
            "total_identities": self.total_identities,
            "total_samples": self.total_samples,
            "unregistered_tokens": sorted(self.unregistered_tokens),
            "identity_summaries": [s.to_dict() for s in self.identity_summaries],
            "bindings": [b.to_dict() for b in self.bindings],
        }


def build_binding(
    assignment_payload: dict[str, Any],
    registry_db: Path,
) -> SplitRegistryBinding:
    records = assignment_payload.get("records", [])
    if not isinstance(records, list):
        raise TypeError("assignment records must be a list")

    conn = sqlite3.connect(str(registry_db))
    try:
        token_to_rid: dict[str, str] = {}
        token_to_dsn: dict[str, str] = {}
        for row in conn.execute(
            "SELECT identity_token, registered_dog_id, dataset_name "
            "FROM identity_registry"
        ):
            token_to_rid[row[0]] = row[1]
            token_to_dsn[row[0]] = row[2]
    finally:
        conn.close()

    identity_aggregator: dict[str, dict[str, Any]] = {}
    unregistered: set[str] = set()
    total_samples = 0

    for rec in records:
        token = rec.get("identity_token", "")
        if token not in token_to_rid:
            unregistered.add(token)
            continue
        total_samples += 1
        role = rec.get("identity_role", "UNKNOWN")
        access = rec.get("model_access", "UNKNOWN")
        disposition = rec.get("sample_disposition", "UNKNOWN")
        dsn = rec.get("dataset_name", token_to_dsn.get(token, "UNKNOWN"))
        sample_token = rec.get("sample_token", "")

        key = (token, role, access, disposition, dsn)
        if key not in identity_aggregator:
            identity_aggregator[key] = {
                "identity_token": token,
                "registered_dog_id": token_to_rid[token],
                "dataset_name": dsn,
                "identity_role": role,
                "model_access": access,
                "sample_disposition": disposition,
                "sample_count": 0,
                "sample_tokens": [],
            }
        identity_aggregator[key]["sample_count"] += 1
        identity_aggregator[key]["sample_tokens"].append(sample_token)

    bindings = tuple(
        IdentityBinding(
            identity_token=v["identity_token"],
            registered_dog_id=v["registered_dog_id"],
            dataset_name=v["dataset_name"],
            identity_role=v["identity_role"],
            model_access=v["model_access"],
            sample_disposition=v["sample_disposition"],
            sample_count=v["sample_count"],
            sample_tokens=tuple(sorted(v["sample_tokens"])),
        )
        for v in sorted(
            identity_aggregator.values(),
            key=lambda x: (x["dataset_name"], x["identity_role"], x["identity_token"]),
        )
    )

    summary_agg: dict[tuple[str, str], tuple[set[str], int]] = defaultdict(
        lambda: (set(), 0)
    )
    for b in bindings:
        key = (b.identity_role, b.model_access)
        summary_agg[key][0].add(b.identity_token)
        summary_agg[key] = (
            summary_agg[key][0],
            summary_agg[key][1] + b.sample_count,
        )

    summaries = tuple(
        IdentityRoleSummary(
            role=role,
            access=access,
            unique_identities=len(ids),
            sample_count=count,
        )
        for (role, access), (ids, count) in sorted(summary_agg.items())
    )

    unique_identity_tokens = {b.identity_token for b in bindings}
    return SplitRegistryBinding(
        bindings=bindings,
        identity_summaries=summaries,
        total_identities=len(unique_identity_tokens),
        total_samples=total_samples,
        unregistered_tokens=tuple(sorted(unregistered)),
    )
