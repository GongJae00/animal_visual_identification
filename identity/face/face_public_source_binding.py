"""Strict token bridge from Full128 face rows to the public source bundle."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from foundation.provenance import content_sha256
from identity.face.face_eligibility import (
    validate_face_eligibility_overlay_bundle,
)
from identity.registry.identity_registry import (
    compute_identity_token,
    compute_public_subject_token,
    compute_registered_dog_id,
    compute_sample_token,
)
from identity.splits.protected_public_split import PublicSplitSourceBundle
from data.full_segment.route_plan import (
    validate_full128_route_plan_bundle,
)

BINDING_SCHEMA = "cvi.face_public_source_binding.v1"
RECORD_SCHEMA = "cvi.face_public_source_binding_record.v1"
BUNDLE_SCHEMA = "cvi.face_public_source_binding_bundle.v1"

_FACE_DATASETS = frozenset({"dogfacenet224", "mpdd"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INTERPRETATION = (
    "EXACT_PUBLIC_TOKEN_BRIDGE_BY_DATASET_AND_ENCODED_SHA256_WITH_MEMBER_PATH_"
    "DISAMBIGUATION;NO_SCORE_OR_ROLE_ALLOCATION_INPUT"
)


def build_face_public_source_binding(
    route_plan_bundle: object,
    face_overlay_bundle: object,
    public_source_bundle: object,
    image_content_receipts: object,
) -> dict[str, Any]:
    """Map every eligible route face to exactly one authenticated public sample."""

    route = validate_full128_route_plan_bundle(route_plan_bundle, verify_files=False)
    overlay = validate_face_eligibility_overlay_bundle(face_overlay_bundle)
    source = _source_bundle(public_source_bundle)
    if (
        overlay["overlay"]["source_route_plan_sha256"] != route["plan_sha256"]
        or overlay["overlay"]["source_route_plan_bundle_sha256"]
        != route["bundle_sha256"]
    ):
        raise ValueError("face public binding route and overlay bindings differ")

    receipt_sha256 = content_sha256(image_content_receipts)
    if dict(source.evidence_bindings).get("image_content_receipts_sha256") != (
        receipt_sha256
    ):
        raise ValueError(
            "image-content receipts are not bound by the public source bundle"
        )
    resolver = _receipt_resolver(source, image_content_receipts)

    route_by_token = {row["sample_token"]: row for row in route["plan"]["records"]}
    if len(route_by_token) != len(route["plan"]["records"]):
        raise ValueError("Full128 route repeats a sample token")
    records: list[dict[str, Any]] = []
    seen_public: set[str] = set()
    for face in overlay["overlay"]["records"]:
        if not face["gallery_query_eligible"] or face["dataset_name"] not in (
            _FACE_DATASETS
        ):
            continue
        route_row = route_by_token.get(face["sample_token"])
        if (
            route_row is None
            or face["source_record_sha256"] != route_row["record_sha256"]
        ):
            raise ValueError("face public binding overlay source record differs")
        source_row = _resolve_source(route_row, resolver)
        if source_row.sample_token in seen_public:
            raise ValueError("multiple route faces map to one public source sample")
        seen_public.add(source_row.sample_token)
        registered_identity = route_row["identity_metadata"]["registered_identity_id"]
        if face[
            "registered_identity_id"
        ] != registered_identity or registered_identity != compute_registered_dog_id(
            source_row.dataset_identity_id
        ):
            raise ValueError("face public binding identity conflict")
        if (
            face["publisher_split"] != source_row.original_split
            or route_row["dataset_name"] != source_row.dataset_name
        ):
            raise ValueError("face public binding dataset or publisher split conflict")
        payload = {
            "schema_version": RECORD_SCHEMA,
            "route_sample_token": route_row["sample_token"],
            "public_sample_token": source_row.sample_token,
            "public_identity_token": source_row.identity_token,
            "public_subject_token": compute_public_subject_token(
                source_row.dataset_identity_id
            ),
            "registered_identity_id": registered_identity,
            "dataset_name": source_row.dataset_name,
            "publisher_split": source_row.original_split,
            "source_sample_id": source_row.source_sample_id,
            "source_variant": source_row.source_variant,
            "encoded_sha256": route_row["source_sha256"],
            "member_path": route_row["source_path"],
            "route_record_sha256": route_row["record_sha256"],
            "face_record_sha256": face["record_sha256"],
        }
        records.append({**payload, "record_sha256": content_sha256(payload)})
    if not records:
        raise ValueError("face public binding has no eligible public face records")
    records.sort(key=lambda row: row["route_sample_token"])
    binding = {
        "schema_version": BINDING_SCHEMA,
        "source_route_plan_sha256": route["plan_sha256"],
        "source_route_plan_bundle_sha256": route["bundle_sha256"],
        "source_face_overlay_sha256": overlay["overlay_sha256"],
        "source_face_overlay_bundle_sha256": overlay["bundle_sha256"],
        "public_source_bundle_sha256": source.bundle_sha256,
        "image_content_receipts_sha256": receipt_sha256,
        "resolution_method": (
            "DATASET_AND_ENCODED_SHA256_WITH_EXACT_MEMBER_PATH_DISAMBIGUATION"
        ),
        "records": records,
        "score_inputs_used": False,
        "interpretation": _INTERPRETATION,
    }
    binding = {**binding, "binding_sha256": content_sha256(binding)}
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "binding": binding,
        "binding_sha256": binding["binding_sha256"],
    }
    return {**payload, "bundle_sha256": content_sha256(payload)}


def validate_face_public_source_binding_bundle(value: object) -> dict[str, Any]:
    """Validate strict fields, content hashes, token derivations, and uniqueness."""

    expected = {"schema_version", "binding", "binding_sha256", "bundle_sha256"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face public source binding bundle fields differ")
    bundle = dict(value)
    if bundle["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError("face public source binding bundle schema differs")
    payload = {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    if bundle["bundle_sha256"] != content_sha256(payload):
        raise ValueError("face public source binding bundle digest differs")
    binding = bundle["binding"]
    expected_binding = {
        "schema_version",
        "source_route_plan_sha256",
        "source_route_plan_bundle_sha256",
        "source_face_overlay_sha256",
        "source_face_overlay_bundle_sha256",
        "public_source_bundle_sha256",
        "image_content_receipts_sha256",
        "resolution_method",
        "records",
        "score_inputs_used",
        "interpretation",
        "binding_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != expected_binding:
        raise ValueError("face public source binding fields differ")
    if (
        binding["schema_version"] != BINDING_SCHEMA
        or binding["resolution_method"]
        != "DATASET_AND_ENCODED_SHA256_WITH_EXACT_MEMBER_PATH_DISAMBIGUATION"
        or binding["score_inputs_used"] is not False
        or binding["interpretation"] != _INTERPRETATION
    ):
        raise ValueError("face public source binding policy differs")
    for field in (
        "source_route_plan_sha256",
        "source_route_plan_bundle_sha256",
        "source_face_overlay_sha256",
        "source_face_overlay_bundle_sha256",
        "public_source_bundle_sha256",
        "image_content_receipts_sha256",
        "binding_sha256",
    ):
        _require_sha256(binding[field], field)
    binding_payload = {
        key: item for key, item in binding.items() if key != "binding_sha256"
    }
    if (
        binding["binding_sha256"] != content_sha256(binding_payload)
        or bundle["binding_sha256"] != binding["binding_sha256"]
    ):
        raise ValueError("face public source binding digest differs")
    raw_records = binding["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("face public source binding records must not be empty")
    records = tuple(_validate_record(row) for row in raw_records)
    route_tokens = [row["route_sample_token"] for row in records]
    public_tokens = [row["public_sample_token"] for row in records]
    if route_tokens != sorted(route_tokens) or len(route_tokens) != len(
        set(route_tokens)
    ):
        raise ValueError("face public route tokens must be sorted and unique")
    if len(public_tokens) != len(set(public_tokens)):
        raise ValueError("face public sample tokens must be unique")
    return bundle


def _source_bundle(value: object) -> PublicSplitSourceBundle:
    if not isinstance(value, dict):
        raise TypeError("public source bundle must be an object")
    return PublicSplitSourceBundle.from_dict(value)


def _receipt_resolver(
    source: PublicSplitSourceBundle, receipts: object
) -> dict[tuple[str, str], tuple[tuple[Any, str], ...]]:
    if not isinstance(receipts, Mapping) or not receipts:
        raise TypeError("image-content receipts must be a non-empty object")
    originals = {
        row.source_sample_id: row
        for row in source.samples
        if row.source_variant == "original"
    }
    resolver: defaultdict[tuple[str, str], list[tuple[Any, str]]] = defaultdict(list)
    seen: set[str] = set()
    for value in receipts.values():
        if not isinstance(value, Mapping) or not isinstance(
            value.get("receipt"), Mapping
        ):
            raise TypeError("merged image-content receipt schema differs")
        receipt = value["receipt"]
        if receipt.get("decision") != "PASS_IMAGE_CONTENT_AUDIT" or not isinstance(
            receipt.get("records"), list
        ):
            raise ValueError("image-content receipt is not an audited record set")
        for row in receipt["records"]:
            if not isinstance(row, Mapping) or row.get("source_variant") != "original":
                continue
            source_id = row.get("source_sample_id")
            public = originals.get(source_id)
            encoded = row.get("encoded_sha256")
            member_path = row.get("member_path")
            if (
                public is None
                or source_id in seen
                or row.get("dataset_name") != public.dataset_name
                or compute_sample_token(public.source_sample_id) != public.sample_token
                or compute_identity_token(public.dataset_identity_id)
                != public.identity_token
                or not isinstance(encoded, str)
                or _SHA256.fullmatch(encoded) is None
                or not isinstance(member_path, str)
                or not member_path
            ):
                raise ValueError("image-content receipt source binding differs")
            seen.add(source_id)
            resolver[(public.dataset_name, encoded)].append((public, member_path))
    if seen != set(originals):
        raise ValueError("image-content receipts do not cover source-bundle originals")
    return {key: tuple(value) for key, value in resolver.items()}


def _resolve_source(
    route: Mapping[str, Any],
    resolver: Mapping[tuple[str, str], Sequence[tuple[Any, str]]],
) -> Any:
    candidates = tuple(
        resolver.get((route["dataset_name"], route["source_sha256"]), ())
    )
    if len(candidates) > 1:
        candidates = tuple(
            candidate
            for candidate in candidates
            if candidate[1] == route["source_path"]
        )
    if len(candidates) != 1:
        raise ValueError(
            "route face does not resolve to exactly one audited public source"
        )
    return candidates[0][0]


def _validate_record(value: object) -> dict[str, Any]:
    expected = {
        "schema_version",
        "route_sample_token",
        "public_sample_token",
        "public_identity_token",
        "public_subject_token",
        "registered_identity_id",
        "dataset_name",
        "publisher_split",
        "source_sample_id",
        "source_variant",
        "encoded_sha256",
        "member_path",
        "route_record_sha256",
        "face_record_sha256",
        "record_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face public source binding record fields differ")
    row = dict(value)
    if (
        row["schema_version"] != RECORD_SCHEMA
        or row["dataset_name"] not in _FACE_DATASETS
        or row["source_variant"] != "original"
        or not isinstance(row["source_sample_id"], str)
        or not row["source_sample_id"]
        or not isinstance(row["member_path"], str)
        or not row["member_path"]
    ):
        raise ValueError("face public source binding record policy differs")
    for field in (
        "route_sample_token",
        "public_sample_token",
        "public_identity_token",
        "public_subject_token",
        "encoded_sha256",
        "route_record_sha256",
        "face_record_sha256",
        "record_sha256",
    ):
        _require_sha256(row[field], field)
    if (
        row["public_sample_token"] != compute_sample_token(row["source_sample_id"])
        or row["registered_identity_id"]
        != compute_registered_dog_id(_dataset_identity_id(row))
        or row["public_identity_token"]
        != compute_identity_token(_dataset_identity_id(row))
        or row["public_subject_token"]
        != compute_public_subject_token(_dataset_identity_id(row))
    ):
        raise ValueError("face public source binding token derivation differs")
    payload = {key: item for key, item in row.items() if key != "record_sha256"}
    if row["record_sha256"] != content_sha256(payload):
        raise ValueError("face public source binding record digest differs")
    return row


def _dataset_identity_id(row: Mapping[str, Any]) -> str:
    source_id = row["source_sample_id"]
    if row["dataset_name"] == "dogfacenet224":
        match = re.fullmatch(
            r"dogfacenet224:v1:web-folder:([^:]+):image:[^:]+", source_id
        )
        if match is not None:
            return f"dogfacenet224:v1:web-folder:{match.group(1)}"
    elif row["dataset_name"] == "mpdd":
        match = re.fullmatch(
            r"mpdd:v1:device-capture:([^:]+):(?:train|val|query|gallery):.+",
            source_id,
        )
        if match is not None:
            return f"mpdd:v1:device-capture:{match.group(1)}"
    raise ValueError("face public source sample ID schema differs")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


__all__ = [
    "build_face_public_source_binding",
    "validate_face_public_source_binding_bundle",
]
