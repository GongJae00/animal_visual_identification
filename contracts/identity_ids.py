"""Pure deterministic identity and sample identifier derivation."""

from __future__ import annotations

import hashlib
import re
import uuid

REGISTERED_DOG_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "cvi.registered_dog.v1")
GENERATED_DOG_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "cvi.generated_dog.v1")

_IDENTITY_TOKEN_PREFIX = b"identity\x00"
_SAMPLE_TOKEN_PREFIX = b"sample\x00"
_SEQUENCE_TOKEN_PREFIX = b"sequence\x00"
_PUBLIC_SUBJECT_TOKEN_PREFIX = b"public-subject\x00"
_CLUSTER_TOKEN_PREFIX = b"generated-source-cluster\x00"
_GENERATOR_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compute_identity_token(dataset_identity_id: str) -> str:
    return _sha256(_IDENTITY_TOKEN_PREFIX + dataset_identity_id.encode("utf-8"))


def compute_public_subject_token(dataset_identity_id: str) -> str:
    return _sha256(
        _PUBLIC_SUBJECT_TOKEN_PREFIX + dataset_identity_id.encode("utf-8")
    )


def compute_sample_token(source_sample_id: str) -> str:
    return _sha256(_SAMPLE_TOKEN_PREFIX + source_sample_id.encode("utf-8"))


def compute_sequence_token(sequence_id: str | None, identity_token: str) -> str:
    payload = sequence_id if sequence_id is not None else identity_token
    return _sha256(_SEQUENCE_TOKEN_PREFIX + payload.encode("utf-8"))


def compute_registered_dog_id(dataset_identity_id: str) -> str:
    return str(uuid.uuid5(REGISTERED_DOG_NAMESPACE, dataset_identity_id))


def extract_dataset_name(dataset_identity_id: str) -> str:
    return dataset_identity_id.split(":", 1)[0]


def compute_source_cluster_token(source_cluster_id: str) -> str:
    if not isinstance(source_cluster_id, str) or not source_cluster_id:
        raise ValueError("source_cluster_id must be a non-empty string")
    return _sha256(_CLUSTER_TOKEN_PREFIX + source_cluster_id.encode("utf-8"))


def compute_generated_identity_id(generator_id: str, source_cluster_token: str) -> str:
    if not isinstance(generator_id, str) or _GENERATOR_ID.fullmatch(generator_id) is None:
        raise ValueError("generator_id is not canonical")
    if not isinstance(source_cluster_token, str) or _SHA256.fullmatch(
        source_cluster_token
    ) is None:
        raise ValueError("source_cluster_token must be a lowercase SHA256 digest")
    return str(
        uuid.uuid5(
            GENERATED_DOG_NAMESPACE,
            f"{generator_id}\x00{source_cluster_token}",
        )
    )


__all__ = [
    "GENERATED_DOG_NAMESPACE",
    "REGISTERED_DOG_NAMESPACE",
    "compute_generated_identity_id",
    "compute_identity_token",
    "compute_public_subject_token",
    "compute_registered_dog_id",
    "compute_sample_token",
    "compute_sequence_token",
    "compute_source_cluster_token",
    "extract_dataset_name",
]
