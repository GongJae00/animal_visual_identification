"""Public crop-level closed-set API: IdentityEngine and Match.

Enrollment takes a registry-issued UUIDv5 in the public field ``dog_id``.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from PIL import Image

_RETRIEVAL_CONFIG_SCHEMA_V1 = "search.config.v1"
_RETRIEVAL_CONFIG_SCHEMA_V2 = "search.config.v2"
_MAXIMUM_CONFIG_BYTES = 1_048_576
_MAXIMUM_MANIFEST_BYTES = 65_536
_MAXIMUM_METADATA_BYTES = 65_536
_MAXIMUM_IDEMPOTENCY_KEY_BYTES = _MAXIMUM_METADATA_BYTES
_MAXIMUM_TOP_K = 1_000
_MAXIMUM_BREED_FILTERS = 256
_MAXIMUM_BREED_BYTES = 256
_RESERVED_METADATA_KEYS = {
    "template_id", "content_sha256", "idempotency_key", "template_schema"
}


@dataclass
class Match:
    dog_id: str
    similarity: float
    evidence: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_availability: dict[str, bool] = field(default_factory=dict)
    scorer_hash: str = ""
    exact: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = {
            "dog_id": self.dog_id,
            "similarity": round(self.similarity, 4),
            "evidence": {k: round(v, 4) for k, v in self.evidence.items()},
        }
        if self.evidence_availability:
            value["evidence_availability"] = dict(self.evidence_availability)
        if self.scorer_hash:
            value["scorer_hash"] = self.scorer_hash
        if self.exact:
            value["exact"] = True
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value


class IdentityEngine:
    def __init__(self, config: dict[str, Any] | str | Path | None = None):
        if config is None:
            raise ValueError("an explicit IdentityEngine configuration is required")
        elif isinstance(config, (str, Path)):
            raw = str(config) if isinstance(config, Path) else config
            if isinstance(config, str) and raw.strip().startswith("{"):
                config = _parse_strict_json_object(
                    raw.encode("utf-8"),
                    label="IdentityEngine configuration",
                    maximum_bytes=_MAXIMUM_CONFIG_BYTES,
                )
            else:
                config = _read_strict_json_object(
                    Path(raw),
                    label="IdentityEngine configuration",
                    maximum_bytes=_MAXIMUM_CONFIG_BYTES,
                )
        if not isinstance(config, dict):
            raise ValueError(  # noqa: TRY004 - public input-validation contract
                "IdentityEngine configuration must be a JSON object"
            )
        config = _canonical_json_object(
            config,
            "IdentityEngine configuration",
            maximum_bytes=_MAXIMUM_CONFIG_BYTES,
        )
        allowed_keys = {
            "schema_version", "mode", "index_dir", "channels",
            "fusion_weights", "fused_dim", "open_set", "optional_channels",
        }
        unknown_keys = set(config) - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"unknown IdentityEngine configuration keys: {sorted(unknown_keys)}"
            )
        schema_version = config.get("schema_version")
        if schema_version not in {
            _RETRIEVAL_CONFIG_SCHEMA_V1, _RETRIEVAL_CONFIG_SCHEMA_V2
        }:
            raise ValueError(
                f"schema_version must be {_RETRIEVAL_CONFIG_SCHEMA_V2!r} "
                f"(or legacy {_RETRIEVAL_CONFIG_SCHEMA_V1!r})"
            )
        if schema_version == _RETRIEVAL_CONFIG_SCHEMA_V1:
            if "optional_channels" in config:
                raise ValueError(
                    "retrieval config v1 is an all-required migration format"
                )
            optional_channels: list[str] = []
        else:
            optional_channels = config.get("optional_channels")
            if not isinstance(optional_channels, list):
                raise ValueError("config v2 requires explicit optional_channels")
            if any(
                not isinstance(name, str) or not name for name in optional_channels
            ) or len(optional_channels) != len(set(optional_channels)):
                raise ValueError(
                    "optional_channels must contain unique non-empty channel names"
                )
        if config.get("mode") != "closed_set_retrieval":
            raise ValueError("mode must be 'closed_set_retrieval'")
        open_set = config.get("open_set")
        if open_set not in (None, {"enabled": False}):
            raise ValueError(
                "open-set identity decisions are disabled until a frozen "
                "calibration boundary is connected"
            )
        index_dir = config.get("index_dir")
        if not isinstance(index_dir, str) or not index_dir:
            raise ValueError("an explicit index_dir must be a non-empty JSON string")
        self._config = config
        self._optional_channels = frozenset(optional_channels)
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        from enrollment.write.writer import EnrollmentWriter
        from gallery.store.gallery import IdentityGallery
        from representation.channels.extraction import EvidenceExtractionPipeline
        from search.matching.pipeline import SearchPipeline
        from search.scoring.roles import (
            AvailableIntersectionScorer,
            EvidenceChannelSpec,
            canonical_channel_weights,
        )

        evidence = self._build_evidence()
        if not self._optional_channels <= set(evidence):
            raise ValueError("optional_channels contains an unknown channel")
        if not set(evidence) - self._optional_channels:
            raise ValueError("at least one configured channel must be required")
        extraction = EvidenceExtractionPipeline(evidence, self._optional_channels)

        index_dir = Path(self._config["index_dir"])
        total_dimension = self._compute_total_embedding_dimension(evidence)
        configured_dimension = self._config.get("fused_dim", total_dimension)
        if configured_dimension != total_dimension:
            raise ValueError(
                "fused_dim must equal active channel dimensions "
                f"({total_dimension})"
            )
        channels = list(evidence.keys())
        weights = canonical_channel_weights(
            len(channels), self._config.get("fusion_weights")
        )
        scoring_policy = AvailableIntersectionScorer(
            tuple(
                EvidenceChannelSpec(
                    name=name,
                    dimension=int(evidence[name].output_dim),
                    optional=name in self._optional_channels,
                    weight=float(weights[index]),
                )
                for index, name in enumerate(channels)
            )
        )
        embedding_contract = {
            "schema_version": "gallery.embedding_contract.v1",
            "dimension": total_dimension,
            "channels": [
                {
                    "name": name,
                    "dimension": int(evidence[name].output_dim),
                    "optional": name in self._optional_channels,
                    "configuration": self._config["channels"][name],
                    **evidence[name].gallery_contract_fields,
                }
                for name in channels
            ],
            "fusion": {
                "type": "exact_available_intersection_weighted_cosine.v1",
                "weights": weights.astype(float).tolist(),
            },
        }
        self._gallery = IdentityGallery(
            index_dir,
            dim=total_dimension,
            embedding_contract=embedding_contract,
        )
        if self._gallery.scorer_hash != scoring_policy.scorer_hash:
            raise RuntimeError(
                "gallery query/gallery-key scorer contract differs from configuration"
            )
        self._writer = EnrollmentWriter(extraction, self._gallery)
        self._retrieval = SearchPipeline(
            extraction, self._gallery
        )

    def _build_evidence(self) -> dict[str, Any]:
        channels = self._config.get("channels")
        if not isinstance(channels, dict) or not channels:
            raise ValueError("channels must be a non-empty object")
        evidence: dict[str, Any] = {}
        for name, spec in channels.items():
            if not isinstance(name, str) or not name:
                raise ValueError("channel names must be non-empty strings")
            if not isinstance(spec, dict):
                raise ValueError(  # noqa: TRY004 - public input-validation contract
                    f"channel {name!r} must be an object"
                )
            kind = spec.get("type", "")
            if kind == "landmark":
                raise ValueError(
                    "legacy landmark configuration is disabled; use the exact "
                    "landmark_onnx artifact bundle"
                )
            try:
                builder = _CHANNEL_BUILDERS[kind]
            except KeyError:
                raise ValueError(
                    f"unsupported channel type {kind!r} for channel {name!r}"
                ) from None
            evidence[name] = builder(name, spec)
        return evidence

    @staticmethod
    def _compute_total_embedding_dimension(evidence: dict[str, Any]) -> int:
        return sum(ev.output_dim for ev in evidence.values())

    @staticmethod
    def _validate_registered_dog_id(value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("dog_id must be a registered UUIDv5 identity")
        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise ValueError("dog_id must be a registered UUIDv5 identity") from exc
        if parsed.version != 5 or str(parsed) != value:
            raise ValueError("dog_id must be a canonical UUIDv5 identity")
        return str(parsed)

    def enroll(self, image: Image.Image, dog_id: str,
               breed: str | None = None,
               metadata: dict | None = None,
               idempotency_key: str | None = None) -> int:
        registered_dog_id = self._validate_registered_dog_id(dog_id)
        canonical_breed = _validate_breed(breed, "breed")
        canonical_metadata = _validate_metadata(metadata)
        canonical_idempotency_key = _validate_idempotency_key(idempotency_key)
        return self._writer.enroll(
            image,
            registered_dog_id,
            canonical_breed,
            canonical_metadata,
            canonical_idempotency_key,
        )

    def search(self, image: Image.Image, top_k: int = 5,
               breed_filter: list[str] | None = None) -> list[Match]:
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= _MAXIMUM_TOP_K
        ):
            raise ValueError(f"top_k must be an integer from 1 to {_MAXIMUM_TOP_K}")
        canonical_filters = _validate_breed_filters(breed_filter)
        raw = self._retrieval.search(image, top_k, canonical_filters)
        return [
            Match(
                r.registered_dog_id,
                r.similarity,
                r.evidence,
                r.metadata,
                evidence_availability=r.evidence_availability,
                scorer_hash=r.scorer_hash,
                exact=r.exact,
            )
            for r in raw
        ]

    def explain(self, image: Image.Image, dog_id: str) -> dict[str, Any]:
        registered_dog_id = self._validate_registered_dog_id(dog_id)
        return self._retrieval.explain(image, registered_dog_id)

    @property
    def size(self) -> int:
        return self._gallery.size

    def save(self) -> None:
        self._gallery.save()

    def close(self) -> None:
        try:
            self.save()
        finally:
            self._gallery.close()


def _require_exact_channel_schema(
    name: str,
    spec: dict[str, Any],
    allowed_keys: tuple[frozenset[str], ...],
    schema_name: str,
) -> None:
    if frozenset(spec) not in allowed_keys:
        raise ValueError(f"channel {name!r} must use the exact {schema_name}")


def _require_channel_text_fields(
    name: str,
    spec: dict[str, Any],
    fields: tuple[str, ...],
) -> None:
    for field_name in fields:
        if not isinstance(spec.get(field_name), str) or not spec[field_name]:
            raise ValueError(f"channel {name!r} requires {field_name}")


def _require_channel_device(name: str, device: object) -> str:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"channel {name!r} device must be 'cpu' or 'cuda'")
    return str(device)


def _build_miewid_channel(name: str, spec: dict[str, Any]) -> Any:
    from identification.export.backbones.miewid import (
        MiewIDArtifactManifest,
        MiewIDReIDExtractor,
    )

    required = frozenset(
        {"type", "model_path", "manifest_path", "parity_receipt_path"}
    )
    _require_exact_channel_schema(
        name,
        spec,
        (required, required | {"device"}),
        "MiewID bundle schema",
    )
    _require_channel_text_fields(
        name,
        spec,
        ("model_path", "manifest_path", "parity_receipt_path"),
    )
    device = _require_channel_device(name, spec.get("device", "cpu"))
    manifest = MiewIDArtifactManifest.from_dict(
        _read_strict_json_object(
            Path(spec["manifest_path"]),
            label=f"channel {name!r} MiewID manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        )
    )
    return MiewIDReIDExtractor(
        Path(spec["model_path"]),
        manifest,
        Path(spec["parity_receipt_path"]),
        use_cuda=device == "cuda",
    )


def _build_landmark_channel(name: str, spec: dict[str, Any]) -> Any:
    from shared.contracts.artifact_manifest import (
        LandmarkGraphManifest,
        LandmarkKeypointManifest,
    )
    from identification.export.landmark import LandmarkEvidencer

    required = frozenset(
        {
            "type",
            "keypoint_model_path",
            "keypoint_manifest_path",
            "graph_model_path",
            "graph_manifest_path",
            "device",
        }
    )
    _require_exact_channel_schema(
        name,
        spec,
        (required,),
        "landmark_onnx channel schema",
    )
    _require_channel_text_fields(
        name,
        spec,
        (
            "keypoint_model_path",
            "keypoint_manifest_path",
            "graph_model_path",
            "graph_manifest_path",
        ),
    )
    device = _require_channel_device(name, spec["device"])
    keypoint_manifest = LandmarkKeypointManifest.from_dict(
        _read_strict_json_object(
            Path(spec["keypoint_manifest_path"]),
            label=f"channel {name!r} keypoint manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        )
    )
    graph_manifest = LandmarkGraphManifest.from_dict(
        _read_strict_json_object(
            Path(spec["graph_manifest_path"]),
            label=f"channel {name!r} graph manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        )
    )
    return LandmarkEvidencer(
        Path(spec["keypoint_model_path"]),
        keypoint_manifest,
        Path(spec["graph_model_path"]),
        graph_manifest,
        use_cuda=device == "cuda",
    )


def _build_nose_channel(name: str, spec: dict[str, Any]) -> Any:
    from shared.contracts.artifact_manifest import (
        NoseDetectorManifest,
        NoseEmbeddingManifest,
        NoseMaskManifest,
    )
    from identification.export.nose.extractor import NosePrintExtractor, NoseRoiPolicy

    required = frozenset(
        {
            "type",
            "detector_model_path",
            "detector_manifest_path",
            "embedding_model_path",
            "embedding_manifest_path",
            "roi_policy",
            "device",
        }
    )
    mask_fields = frozenset({"mask_model_path", "mask_manifest_path"})
    _require_exact_channel_schema(
        name,
        spec,
        (required, required | mask_fields),
        "composite nose_print_onnx bundle schema",
    )
    _require_channel_text_fields(
        name,
        spec,
        (
            "detector_model_path",
            "detector_manifest_path",
            "embedding_model_path",
            "embedding_manifest_path",
        ),
    )
    device = _require_channel_device(name, spec["device"])
    roi_payload = spec["roi_policy"]
    roi_fields = {
        "min_box_width",
        "min_box_height",
        "min_resolution_width",
        "min_resolution_height",
    }
    if not isinstance(roi_payload, dict) or set(roi_payload) != roi_fields:
        raise ValueError(f"channel {name!r} roi_policy must use the exact schema")
    roi_policy = NoseRoiPolicy(**roi_payload)
    detector_manifest = NoseDetectorManifest.from_dict(
        _read_strict_json_object(
            Path(spec["detector_manifest_path"]),
            label=f"channel {name!r} detector manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        )
    )
    embedding_manifest = NoseEmbeddingManifest.from_dict(
        _read_strict_json_object(
            Path(spec["embedding_manifest_path"]),
            label=f"channel {name!r} embedding manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        )
    )
    mask_manifest = (
        NoseMaskManifest.from_dict(
            _read_strict_json_object(
                Path(spec["mask_manifest_path"]),
                label=f"channel {name!r} mask manifest",
                maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
            )
        )
        if mask_fields <= set(spec)
        else None
    )
    return NosePrintExtractor(
        Path(spec["detector_model_path"]),
        detector_manifest,
        Path(spec["embedding_model_path"]),
        embedding_manifest,
        roi_policy,
        mask_path=Path(spec["mask_model_path"]) if mask_manifest is not None else None,
        mask_manifest=mask_manifest,
        use_cuda=device == "cuda",
    )


def _build_dinov2_channel(name: str, spec: dict[str, Any]) -> Any:
    from identification.export.appearance import ReceiptBoundDinov2Small

    required = frozenset(
        {
            "type",
            "model_dir",
            "weight_intake_bundle",
            "preprocessor_intake_bundle",
            "device",
        }
    )
    _require_exact_channel_schema(
        name,
        spec,
        (required,),
        "dinov2_local channel schema",
    )
    _require_channel_text_fields(
        name,
        spec,
        ("model_dir", "weight_intake_bundle", "preprocessor_intake_bundle"),
    )
    device = spec["device"]
    if not isinstance(device, str):
        raise ValueError(f"channel {name!r} device must be 'cpu' or 'cuda'")
    device = _require_channel_device(name, device)
    return ReceiptBoundDinov2Small(
        model_directory=Path(spec["model_dir"]),
        weight_intake_bundle=Path(spec["weight_intake_bundle"]),
        preprocessor_intake_bundle=Path(spec["preprocessor_intake_bundle"]),
        device=device,
    )


def _build_onnx_channel(name: str, spec: dict[str, Any]) -> Any:
    from shared.contracts.model_contract import (
        ConvNeXtModelManifest,
        DogFaceNetModelManifest,
        PetReIDModelManifest,
    )
    from identification.export.backbones.extractors import (
        ConvNeXtExtractor,
        DogFaceNetExtractor,
        PetReIDExtractor,
    )

    required = frozenset({"type", "model_path", "manifest_path"})
    _require_exact_channel_schema(
        name,
        spec,
        (required, required | {"device"}),
        "ONNX channel schema",
    )
    _require_channel_text_fields(name, spec, ("model_path", "manifest_path"))
    device = _require_channel_device(name, spec.get("device", "cpu"))
    kind = spec["type"]
    manifest_type, extractor_type = {
        "dogfacenet_onnx": (DogFaceNetModelManifest, DogFaceNetExtractor),
        "convnext_onnx": (ConvNeXtModelManifest, ConvNeXtExtractor),
        "petreid_nose_onnx": (PetReIDModelManifest, PetReIDExtractor),
    }[kind]
    manifest = manifest_type.from_dict(
        _read_strict_json_object(
            Path(spec["manifest_path"]),
            label=f"channel {name!r} model manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        )
    )
    return extractor_type(
        Path(spec["model_path"]), manifest, use_cuda=device == "cuda"
    )


_CHANNEL_BUILDERS = {
    "miewid": _build_miewid_channel,
    "miewid_reid": _build_miewid_channel,
    "wildlife_reid": _build_miewid_channel,
    "landmark_onnx": _build_landmark_channel,
    "nose_print_onnx": _build_nose_channel,
    "dinov2_local": _build_dinov2_channel,
    "dogfacenet_onnx": _build_onnx_channel,
    "convnext_onnx": _build_onnx_channel,
    "petreid_nose_onnx": _build_onnx_channel,
}


def _read_strict_json_object(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - the supported release target is Linux
        raise RuntimeError(
            "secure IdentityEngine configuration reads require O_NOFOLLOW support"
        )
    try:
        descriptor = os.open(path, flags | no_follow)
    except OSError as exc:
        raise ValueError(
            f"unable to read {label} file {str(path)!r} without following links"
        ) from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > maximum_bytes:
            raise ValueError(f"{label} must be a bounded regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(maximum_bytes + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _parse_strict_json_object(
        payload,
        label=label,
        maximum_bytes=maximum_bytes,
    )


def _parse_strict_json_object(
    payload: bytes,
    *,
    label: str,
    maximum_bytes: int,
) -> dict[str, Any]:
    if not payload or len(payload) > maximum_bytes:
        raise ValueError(f"{label} must contain bounded JSON bytes")
    try:
        text = payload.decode("utf-8", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be strict JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(  # noqa: TRY004 - public input-validation contract
            f"{label} must be a JSON object"
        )
    return parsed


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not accepted: {value}")


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is not accepted: {value}")
    return parsed


def _canonical_json_object(
    value: dict[str, Any],
    label: str,
    *,
    maximum_bytes: int,
) -> dict[str, Any]:
    _validate_json_value(value, label)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{label} exceeds its JSON size limit")
    return json.loads(encoded.decode("utf-8"))


def _validate_json_value(root: object, label: str) -> None:
    stack: list[tuple[object, int]] = [(root, 1)]
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if depth > 32 or nodes > 10_000:
            raise ValueError(f"{label} exceeds JSON structural limits")
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError(f"{label} JSON object keys must be strings")
            stack.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value)
        elif isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError(f"{label} must contain only finite JSON values")
        elif value is None or isinstance(value, (str, bool, int)):
            continue
        else:
            raise ValueError(f"{label} must contain only JSON value types")


def _validate_breed(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > _MAXIMUM_BREED_BYTES
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{label} must be bounded canonical text")
    return value


def _validate_breed_filters(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    if not isinstance(values, list) or len(values) > _MAXIMUM_BREED_FILTERS:
        raise ValueError("breed_filter must be a bounded JSON array")
    result: list[str] = []
    for index, value in enumerate(values):
        canonical = _validate_breed(value, f"breed_filter[{index}]")
        if canonical is None:
            raise ValueError("breed_filter entries must be strings")
        result.append(canonical)
    if len(result) != len(set(result)):
        raise ValueError("breed_filter entries must be unique")
    return result


def _validate_metadata(metadata: dict | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError(  # noqa: TRY004 - public input-validation contract
            "metadata must be a JSON object"
        )
    if set(metadata) & _RESERVED_METADATA_KEYS:
        raise ValueError(
            "metadata keys template_id, content_sha256, idempotency_key, and "
            "template_schema are reserved"
        )
    return _canonical_json_object(
        metadata,
        "metadata",
        maximum_bytes=_MAXIMUM_METADATA_BYTES,
    )


def _validate_idempotency_key(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("idempotency_key must be a non-empty string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("idempotency_key must be valid UTF-8 text") from exc
    if len(encoded) > _MAXIMUM_IDEMPOTENCY_KEY_BYTES:
        raise ValueError("idempotency_key exceeds its UTF-8 byte limit")
    return value
