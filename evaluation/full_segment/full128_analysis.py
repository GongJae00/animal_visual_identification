"""Bounded private representation traces and public-safe trace summaries."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from foundation.provenance import content_sha256
from embedding.methods.full_segment.models.successor_models import (
    Dinov2OccupancyProbe128,
    PatchRepresentationDecomposition,
    SpatialScorer128,
)

TRACE_SCHEMA = "cvi.full128_representation_trace.v1"
PUBLIC_TRACE_SCHEMA = "cvi.full128_representation_trace_public.v3"
EXECUTED_TRACE_SCHEMA = "cvi.full128_representation_trace.v2"
EXECUTED_PUBLIC_TRACE_SCHEMA = PUBLIC_TRACE_SCHEMA
PUBLIC_ANALYSIS_SCHEMA = "cvi.full128_representation_analysis_public.v2"
EMBEDDING_DIMENSION = 128
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SPATIAL_MAP_NAMES = {
    "mask_occupancy",
    "query_saliency",
    "key_saliency",
    "pair_similarity",
}
_MAX_LAYERS = 256
_MAX_GRID_CELLS = 65_536
_EXECUTED_MAP_NAMES = {
    "query_mask_occupancy",
    "key_mask_occupancy",
    "query_pooling_weight",
    "key_pooling_weight",
    "query_spatial_scorer_logit",
    "key_spatial_scorer_logit",
    "query_pair_contribution",
    "key_pair_contribution",
    "pair_patch_contribution",
}
_EXECUTED_MAP_SEMANTICS = {
    "query_mask_occupancy": "FRACTIONAL_FOREGROUND_AREA_PER_PATCH",
    "key_mask_occupancy": "FRACTIONAL_FOREGROUND_AREA_PER_PATCH",
    "query_pooling_weight": "EXACT_NORMALIZED_QUERY_PATCH_POOLING_WEIGHT",
    "key_pooling_weight": "EXACT_NORMALIZED_KEY_PATCH_POOLING_WEIGHT",
    "query_spatial_scorer_logit": "EXECUTED_B5_QUERY_PATCH_SCORER_LOGIT",
    "key_spatial_scorer_logit": "EXECUTED_B5_KEY_PATCH_SCORER_LOGIT",
    "query_pair_contribution": (
        "AFFINE_PROJECTION_QUERY_PATCH_CONTRIBUTION_TO_PAIR_COSINE"
    ),
    "key_pair_contribution": "AFFINE_PROJECTION_KEY_PATCH_CONTRIBUTION_TO_PAIR_COSINE",
    "pair_patch_contribution": (
        "AFFINE_PROJECTION_QUERY_BY_KEY_PATCH_COSINE_CONTRIBUTION"
    ),
}
_PUBLIC_UNAVAILABLE_REASONS = {
    "transformer_attention_query_key": (
        "THE_PRODUCTION_SUCCESSOR_CONSUMES_FINAL_DINOV2_PATCH_TOKENS;"
        "ATTENTION_QUERY_KEY_TENSORS_WERE_NOT_RETURNED_BY_MODEL_EXECUTION"
    ),
    "pair_patch_correspondence": (
        "NO_CORRESPONDENCE_MODULE_EXISTS;PAIR_PATCH_CONTRIBUTIONS_ARE_"
        "AFFINE_EMBEDDING_DECOMPOSITIONS_NOT_PIXEL_CORRESPONDENCES"
    ),
    "spatial_scorer_logits": "B3_HAS_NO_SPATIAL_SCORER",
}


class RepresentationTraceError(ValueError):
    """Raised when representation evidence is unbound, malformed, or unsafe."""


def build_representation_trace_manifest(
    *,
    successor_id: str,
    sample_token: str,
    model_binding_sha256: str,
    model_input_transform: Mapping[str, Any],
    layers: Sequence[Mapping[str, Any]],
    patch_geometry: Mapping[str, Any],
    mask_occupancy: Sequence[Sequence[float]],
    embedding: np.ndarray,
    embedding_cache_descriptor_sha256: str,
    pair: Mapping[str, Any] | None = None,
    spatial_maps: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> dict[str, Any]:
    """Build one private trace containing only named, bounded representations."""

    _nonempty(successor_id, "successor_id")
    _sha(sample_token, "sample_token")
    _sha(model_binding_sha256, "model_binding_sha256")
    _sha(
        embedding_cache_descriptor_sha256,
        "embedding_cache_descriptor_sha256",
    )
    transform = _validate_transform(dict(model_input_transform))
    layer_rows = _validate_layers(layers)
    geometry = _validate_patch_geometry(dict(patch_geometry))
    occupancy = _spatial_map(mask_occupancy, geometry, "mask_occupancy")
    vector = _embedding(embedding)
    embedding_binding = {
        "dimension": EMBEDDING_DIMENSION,
        "dtype": "float32",
        "normalization": "L2",
        "vector_sha256": _array_sha256(vector),
        "cache_descriptor_sha256": embedding_cache_descriptor_sha256,
    }
    maps: dict[str, dict[str, Any]] = {"mask_occupancy": _private_map_record(occupancy)}
    for name, raw_map in sorted((spatial_maps or {}).items()):
        if name not in _SPATIAL_MAP_NAMES or name == "mask_occupancy":
            raise RepresentationTraceError(
                f"unsupported or duplicate representation spatial map: {name!r}"
            )
        maps[name] = _private_map_record(_spatial_map(raw_map, geometry, name))
    pair_record = None if pair is None else _validate_pair(dict(pair))
    payload = {
        "schema_version": TRACE_SCHEMA,
        "visibility": "PRIVATE",
        "successor_id": successor_id,
        "sample_token": sample_token,
        "model_binding_sha256": model_binding_sha256,
        "model_input_transform": transform,
        "layers": layer_rows,
        "patch_geometry": geometry,
        "mask_occupancy": _occupancy_summary(occupancy),
        "embedding_binding": embedding_binding,
        "pair": pair_record,
        "spatial_maps": maps,
    }
    return {**payload, "trace_sha256": content_sha256(payload)}


def validate_representation_trace_manifest(value: object) -> dict[str, Any]:
    """Validate a private trace and all explicitly admitted spatial maps."""

    expected = {
        "schema_version",
        "visibility",
        "successor_id",
        "sample_token",
        "model_binding_sha256",
        "model_input_transform",
        "layers",
        "patch_geometry",
        "mask_occupancy",
        "embedding_binding",
        "pair",
        "spatial_maps",
        "trace_sha256",
    }
    _keys(value, expected, "representation trace")
    trace = dict(value)
    if trace["schema_version"] != TRACE_SCHEMA or trace["visibility"] != "PRIVATE":
        raise RepresentationTraceError(
            "representation trace schema or visibility differs"
        )
    payload = {key: item for key, item in trace.items() if key != "trace_sha256"}
    if trace["trace_sha256"] != content_sha256(payload):
        raise RepresentationTraceError("representation trace digest differs")
    _nonempty(trace["successor_id"], "successor_id")
    for field in ("sample_token", "model_binding_sha256", "trace_sha256"):
        _sha(trace[field], field)
    _validate_transform(trace["model_input_transform"])
    _validate_layers(trace["layers"])
    geometry = _validate_patch_geometry(trace["patch_geometry"])
    binding = trace["embedding_binding"]
    _keys(
        binding,
        {
            "dimension",
            "dtype",
            "normalization",
            "vector_sha256",
            "cache_descriptor_sha256",
        },
        "trace embedding binding",
    )
    if (
        binding["dimension"] != EMBEDDING_DIMENSION
        or binding["dtype"] != "float32"
        or binding["normalization"] != "L2"
    ):
        raise RepresentationTraceError("trace embedding contract differs")
    _sha(binding["vector_sha256"], "trace embedding vector")
    _sha(binding["cache_descriptor_sha256"], "trace cache descriptor")
    maps = trace["spatial_maps"]
    if not isinstance(maps, Mapping) or "mask_occupancy" not in maps:
        raise RepresentationTraceError("trace must contain mask occupancy spatial map")
    if not set(maps) <= _SPATIAL_MAP_NAMES:
        raise RepresentationTraceError("trace contains an arbitrary spatial tensor")
    decoded: dict[str, np.ndarray] = {}
    for name, record in maps.items():
        decoded[name] = _validate_private_map_record(record, geometry, name)
    if trace["mask_occupancy"] != _occupancy_summary(decoded["mask_occupancy"]):
        raise RepresentationTraceError("trace mask occupancy summary differs")
    if trace["pair"] is not None:
        _validate_pair(trace["pair"])
    return trace


def sanitize_representation_trace_manifest(value: object) -> dict[str, Any]:
    """Publish only model/artifact contracts and trace evidence availability."""

    if (
        isinstance(value, Mapping)
        and value.get("schema_version") == EXECUTED_TRACE_SCHEMA
    ):
        return _sanitize_executed_representation_trace(value)
    trace = validate_representation_trace_manifest(value)
    payload = {
        "schema_version": PUBLIC_TRACE_SCHEMA,
        "visibility": "PUBLIC_SAFE_TRACE_SUMMARY",
        "trace_kind": "DECLARED_REPRESENTATION",
        "successor_id": trace["successor_id"],
        "artifact_contracts": {
            "model_binding_sha256": trace["model_binding_sha256"],
            "embedding_cache_descriptor_sha256": trace["embedding_binding"][
                "cache_descriptor_sha256"
            ],
        },
        "model_contract": _public_model_contract(
            transform=trace["model_input_transform"],
            layers=trace["layers"],
            geometry=trace["patch_geometry"],
        ),
        "execution_evidence": "DECLARED_ONLY",
        "pair_evidence": (
            "AVAILABLE_IN_PRIVATE_TRACE"
            if trace["pair"] is not None
            else "NOT_AVAILABLE_IN_PRIVATE_TRACE"
        ),
        "available_map_contracts": [
            {
                "name": name,
                "semantic": "DECLARED_SPATIAL_MAP",
                "dtype": record["dtype"],
                "shape": record["shape"],
            }
            for name, record in sorted(trace["spatial_maps"].items())
        ],
        "unavailable_evidence": [],
        "contains_raw_tensor_values": False,
        "contains_private_identifiers": False,
    }
    return validate_public_representation_trace_manifest(
        {**payload, "public_trace_sha256": content_sha256(payload)}
    )


def build_executed_representation_trace_manifest(
    *,
    successor_id: str,
    model: Dinov2OccupancyProbe128 | SpatialScorer128,
    query_sample_token: str,
    key_sample_token: str,
    cached_query_tokens: torch.Tensor,
    cached_key_tokens: torch.Tensor,
    cached_query_occupancy: torch.Tensor,
    cached_key_occupancy: torch.Tensor,
    live_query_tokens: torch.Tensor,
    live_key_tokens: torch.Tensor,
    live_query_occupancy: torch.Tensor,
    live_key_occupancy: torch.Tensor,
    cached_query_embedding: np.ndarray,
    cached_key_embedding: np.ndarray,
    model_input_transform: Mapping[str, Any],
    artifact_bindings: Mapping[str, Any],
    query_input_binding: Mapping[str, Any],
    key_input_binding: Mapping[str, Any],
    rank: int,
    query_index: int = 0,
    key_index: int = 0,
) -> dict[str, Any]:
    """Execute one real production pair and bind it exactly to persisted caches."""

    if successor_id not in {"B3", "B5-SPATIAL"}:
        raise RepresentationTraceError("executed traces admit only B3 and B5-SPATIAL")
    expected_type = (
        Dinov2OccupancyProbe128 if successor_id == "B3" else SpatialScorer128
    )
    if not isinstance(model, expected_type):
        raise RepresentationTraceError("executed trace model type differs")
    _sha(query_sample_token, "query sample token")
    _sha(key_sample_token, "key sample token")
    transform = _validate_transform(dict(model_input_transform))
    bindings = _executed_bindings(artifact_bindings)
    query_input = _input_binding(query_input_binding, "query")
    key_input = _input_binding(key_input_binding, "key")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise RepresentationTraceError("executed pair rank must be positive")

    cached_tokens = _pair_tensors(
        cached_query_tokens, cached_key_tokens, subject="cached tokens", dimension=384
    )
    live_tokens = _pair_tensors(
        live_query_tokens, live_key_tokens, subject="live tokens", dimension=384
    )
    cached_occupancy = _pair_tensors(
        cached_query_occupancy,
        cached_key_occupancy,
        subject="cached occupancy",
        dimension=None,
    )
    live_occupancy = _pair_tensors(
        live_query_occupancy,
        live_key_occupancy,
        subject="live occupancy",
        dimension=None,
    )
    if not all(
        torch.equal(live, cached)
        for live, cached in zip(live_tokens, cached_tokens, strict=True)
    ):
        raise RepresentationTraceError(
            "live DINOv2 patch tokens do not exactly match the bound token cache"
        )
    if not all(
        torch.equal(live, cached)
        for live, cached in zip(live_occupancy, cached_occupancy, strict=True)
    ):
        raise RepresentationTraceError(
            "live mask occupancy does not exactly match the bound token cache"
        )
    query_tokens, key_tokens = live_tokens
    query_occupancy, key_occupancy = live_occupancy
    if query_tokens.shape[1] != 256 or key_tokens.shape[1] != 256:
        raise RepresentationTraceError("executed trace requires a 16x16 patch grid")
    if (
        query_occupancy.shape != query_tokens.shape[:2]
        or key_occupancy.shape != key_tokens.shape[:2]
    ):
        raise RepresentationTraceError("executed trace occupancy shape differs")
    for index, tensor, subject in (
        (query_index, query_tokens, "query"),
        (key_index, key_tokens, "key"),
    ):
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(tensor)
        ):
            raise RepresentationTraceError(f"executed {subject} batch index differs")

    model.eval()
    with torch.inference_mode():
        query_batch = model.decompose_representation(query_tokens, query_occupancy)
        key_batch = model.decompose_representation(key_tokens, key_occupancy)
    query = _slice_decomposition(query_batch, query_index)
    key = _slice_decomposition(key_batch, key_index)
    query_cache = _embedding(cached_query_embedding)
    key_cache = _embedding(cached_key_embedding)
    query_execution = query.embedding[0].float().cpu().numpy()
    key_execution = key.embedding[0].float().cpu().numpy()
    if not np.array_equal(query_execution, query_cache) or not np.array_equal(
        key_execution, key_cache
    ):
        raise RepresentationTraceError(
            "executed embedding does not exactly match the bound evaluation cache"
        )

    pair_score = float(np.dot(query_cache, key_cache))
    contribution = _pair_patch_contribution(model, query, key)
    query_contribution = contribution.sum(axis=1).reshape(16, 16)
    key_contribution = contribution.sum(axis=0).reshape(16, 16)
    contribution_sum = float(contribution.sum(dtype=np.float64))
    maps = {
        "query_mask_occupancy": _executed_map_record(
            query_occupancy[query_index].float().cpu().numpy().reshape(16, 16),
            semantic="FRACTIONAL_FOREGROUND_AREA_PER_PATCH",
        ),
        "key_mask_occupancy": _executed_map_record(
            key_occupancy[key_index].float().cpu().numpy().reshape(16, 16),
            semantic="FRACTIONAL_FOREGROUND_AREA_PER_PATCH",
        ),
        "query_pooling_weight": _executed_map_record(
            query.weights[0].float().cpu().numpy().reshape(16, 16),
            semantic="EXACT_NORMALIZED_QUERY_PATCH_POOLING_WEIGHT",
        ),
        "key_pooling_weight": _executed_map_record(
            key.weights[0].float().cpu().numpy().reshape(16, 16),
            semantic="EXACT_NORMALIZED_KEY_PATCH_POOLING_WEIGHT",
        ),
        "query_pair_contribution": _executed_map_record(
            query_contribution,
            semantic="AFFINE_PROJECTION_QUERY_PATCH_CONTRIBUTION_TO_PAIR_COSINE",
        ),
        "key_pair_contribution": _executed_map_record(
            key_contribution,
            semantic="AFFINE_PROJECTION_KEY_PATCH_CONTRIBUTION_TO_PAIR_COSINE",
        ),
        "pair_patch_contribution": _executed_map_record(
            contribution,
            semantic="AFFINE_PROJECTION_QUERY_BY_KEY_PATCH_COSINE_CONTRIBUTION",
        ),
    }
    unavailable = [
        {
            "name": "transformer_attention_query_key",
            "reason": (
                "THE_PRODUCTION_SUCCESSOR_CONSUMES_FINAL_DINOV2_PATCH_TOKENS;"
                "ATTENTION_QUERY_KEY_TENSORS_WERE_NOT_RETURNED_BY_MODEL_EXECUTION"
            ),
        },
        {
            "name": "pair_patch_correspondence",
            "reason": (
                "NO_CORRESPONDENCE_MODULE_EXISTS;PAIR_PATCH_CONTRIBUTIONS_ARE_"
                "AFFINE_EMBEDDING_DECOMPOSITIONS_NOT_PIXEL_CORRESPONDENCES"
            ),
        },
    ]
    if query.logits is None or key.logits is None:
        unavailable.append(
            {
                "name": "spatial_scorer_logits",
                "reason": "B3_HAS_NO_SPATIAL_SCORER",
            }
        )
    else:
        maps["query_spatial_scorer_logit"] = _executed_map_record(
            query.logits[0].float().cpu().numpy().reshape(16, 16),
            semantic="EXECUTED_B5_QUERY_PATCH_SCORER_LOGIT",
        )
        maps["key_spatial_scorer_logit"] = _executed_map_record(
            key.logits[0].float().cpu().numpy().reshape(16, 16),
            semantic="EXECUTED_B5_KEY_PATCH_SCORER_LOGIT",
        )

    layers = _executed_layer_shapes(query_batch)
    payload = {
        "schema_version": EXECUTED_TRACE_SCHEMA,
        "visibility": "PRIVATE",
        "successor_id": successor_id,
        "private_samples": {
            "query_sample_token": query_sample_token,
            "key_sample_token": key_sample_token,
        },
        "artifact_bindings": bindings,
        "input_bindings": {"query": query_input, "key": key_input},
        "model_input_transform": transform,
        "layers": layers,
        "patch_geometry": {
            "input_height": 224,
            "input_width": 224,
            "patch_height": 14,
            "patch_width": 14,
            "grid_height": 16,
            "grid_width": 16,
        },
        "execution_verification": {
            "device": query.embedding.device.type,
            "dtype": "float32",
            "live_tokens_exact_cache_match": True,
            "live_occupancy_exact_cache_match": True,
            "query_embedding_exact_cache_match": True,
            "key_embedding_exact_cache_match": True,
        },
        "embedding_bindings": {
            "query_vector_sha256": _array_sha256(query_cache),
            "key_vector_sha256": _array_sha256(key_cache),
            "dimension": EMBEDDING_DIMENSION,
            "dtype": "float32",
            "normalization": "L2",
        },
        "pair": {
            "score": pair_score,
            "rank": rank,
            "exact_cosine": True,
            "algorithm": "EXACT_FLOAT32_DOT_OF_L2_CACHE_VECTORS",
            "pair_patch_contribution_sum": contribution_sum,
            "pair_patch_contribution_roundoff": contribution_sum - pair_score,
        },
        "available_maps": maps,
        "unavailable_evidence": sorted(unavailable, key=lambda item: item["name"]),
    }
    return {**payload, "trace_sha256": content_sha256(payload)}


def validate_executed_representation_trace_manifest(value: object) -> dict[str, Any]:
    """Validate a private trace emitted by actual production model execution."""

    expected = {
        "schema_version",
        "visibility",
        "successor_id",
        "private_samples",
        "artifact_bindings",
        "input_bindings",
        "model_input_transform",
        "layers",
        "patch_geometry",
        "execution_verification",
        "embedding_bindings",
        "pair",
        "available_maps",
        "unavailable_evidence",
        "trace_sha256",
    }
    _keys(value, expected, "executed representation trace")
    trace = dict(value)
    payload = {key: item for key, item in trace.items() if key != "trace_sha256"}
    if (
        trace["schema_version"] != EXECUTED_TRACE_SCHEMA
        or trace["visibility"] != "PRIVATE"
        or trace["successor_id"] not in {"B3", "B5-SPATIAL"}
        or trace["trace_sha256"] != content_sha256(payload)
    ):
        raise RepresentationTraceError("executed representation trace binding differs")
    _keys(
        trace["private_samples"],
        {"query_sample_token", "key_sample_token"},
        "executed private samples",
    )
    for sample_token in trace["private_samples"].values():
        _sha(sample_token, "executed private sample token")
    _executed_bindings(trace["artifact_bindings"])
    _keys(trace["input_bindings"], {"query", "key"}, "executed input bindings")
    _input_binding(trace["input_bindings"]["query"], "query")
    _input_binding(trace["input_bindings"]["key"], "key")
    _validate_transform(trace["model_input_transform"])
    _validate_layers(trace["layers"])
    _validate_patch_geometry(trace["patch_geometry"])
    _validate_execution_verification(trace["execution_verification"])
    _validate_executed_embedding_bindings(trace["embedding_bindings"])
    _validate_executed_pair(trace["pair"])
    maps = trace["available_maps"]
    if not isinstance(maps, Mapping) or not set(maps) <= _EXECUTED_MAP_NAMES:
        raise RepresentationTraceError("executed trace contains an unsupported map")
    required = {
        "query_mask_occupancy",
        "key_mask_occupancy",
        "query_pooling_weight",
        "key_pooling_weight",
        "query_pair_contribution",
        "key_pair_contribution",
        "pair_patch_contribution",
    }
    if not required <= set(maps):
        raise RepresentationTraceError("executed trace omits required maps")
    decoded = {
        name: _validate_executed_map_record(record, name)
        for name, record in maps.items()
    }
    pair_sum = float(np.sum(decoded["pair_patch_contribution"], dtype=np.float64))
    query_sum = float(np.sum(decoded["query_pair_contribution"], dtype=np.float64))
    key_sum = float(np.sum(decoded["key_pair_contribution"], dtype=np.float64))
    if (
        pair_sum != trace["pair"]["pair_patch_contribution_sum"]
        or abs(query_sum - pair_sum) > 1e-7
        or abs(key_sum - pair_sum) > 1e-7
        or not np.isclose(
            decoded["query_pooling_weight"].sum(dtype=np.float64),
            1.0,
            atol=1e-6,
            rtol=0.0,
        )
        or not np.isclose(
            decoded["key_pooling_weight"].sum(dtype=np.float64),
            1.0,
            atol=1e-6,
            rtol=0.0,
        )
        or np.any(
            (decoded["query_mask_occupancy"] < 0.0)
            | (decoded["query_mask_occupancy"] > 1.0)
        )
        or np.any(
            (decoded["key_mask_occupancy"] < 0.0)
            | (decoded["key_mask_occupancy"] > 1.0)
        )
    ):
        raise RepresentationTraceError("executed map relationships differ")
    scorer_maps = {
        "query_spatial_scorer_logit",
        "key_spatial_scorer_logit",
    }
    if (trace["successor_id"] == "B5-SPATIAL") != scorer_maps.issubset(maps):
        raise RepresentationTraceError("executed spatial scorer evidence differs")
    unavailable = trace["unavailable_evidence"]
    if not isinstance(unavailable, list) or unavailable != sorted(
        unavailable, key=lambda item: item.get("name", "")
    ):
        raise RepresentationTraceError("unavailable evidence must be sorted")
    for record in unavailable:
        _keys(record, {"name", "reason"}, "unavailable evidence")
        _nonempty(record["name"], "unavailable evidence name")
        _nonempty(record["reason"], "unavailable evidence reason")
    unavailable_names = {record["name"] for record in unavailable}
    if (
        not {"transformer_attention_query_key", "pair_patch_correspondence"}
        <= unavailable_names
    ):
        raise RepresentationTraceError(
            "executed trace omits required unavailable evidence"
        )
    if (trace["successor_id"] == "B3") != (
        "spatial_scorer_logits" in unavailable_names
    ):
        raise RepresentationTraceError("executed spatial scorer availability differs")
    return trace


def _sanitize_executed_representation_trace(value: object) -> dict[str, Any]:
    trace = validate_executed_representation_trace_manifest(value)
    payload = {
        "schema_version": EXECUTED_PUBLIC_TRACE_SCHEMA,
        "visibility": "PUBLIC_SAFE_TRACE_SUMMARY",
        "trace_kind": "EXECUTED_REPRESENTATION",
        "successor_id": trace["successor_id"],
        "artifact_contracts": trace["artifact_bindings"],
        "model_contract": _public_model_contract(
            transform=trace["model_input_transform"],
            layers=trace["layers"],
            geometry=trace["patch_geometry"],
        ),
        "execution_evidence": "ACTUAL_EXECUTION_WITH_EXACT_PRIVATE_CACHE_BINDINGS",
        "pair_evidence": "AVAILABLE_IN_PRIVATE_TRACE",
        "available_map_contracts": [
            {
                "name": name,
                "semantic": _EXECUTED_MAP_SEMANTICS[name],
                "dtype": record["dtype"],
                "shape": record["shape"],
            }
            for name, record in sorted(trace["available_maps"].items())
        ],
        "unavailable_evidence": [
            {"name": name, "reason": _PUBLIC_UNAVAILABLE_REASONS[name]}
            for name in sorted(
                record["name"]
                for record in trace["unavailable_evidence"]
                if record["name"] in _PUBLIC_UNAVAILABLE_REASONS
            )
        ],
        "contains_raw_tensor_values": False,
        "contains_private_identifiers": False,
    }
    return validate_public_representation_trace_manifest(
        {**payload, "public_trace_sha256": content_sha256(payload)}
    )


def validate_public_representation_trace_manifest(value: object) -> dict[str, Any]:
    """Validate a public trace summary and reject sample-derived detail."""

    expected = {
        "schema_version",
        "visibility",
        "trace_kind",
        "successor_id",
        "artifact_contracts",
        "model_contract",
        "execution_evidence",
        "pair_evidence",
        "available_map_contracts",
        "unavailable_evidence",
        "contains_raw_tensor_values",
        "contains_private_identifiers",
        "public_trace_sha256",
    }
    _keys(value, expected, "public representation trace")
    trace = dict(value)
    payload = {key: item for key, item in trace.items() if key != "public_trace_sha256"}
    if (
        trace["schema_version"] != PUBLIC_TRACE_SCHEMA
        or trace["visibility"] != "PUBLIC_SAFE_TRACE_SUMMARY"
        or trace["trace_kind"]
        not in {"DECLARED_REPRESENTATION", "EXECUTED_REPRESENTATION"}
        or trace["contains_raw_tensor_values"] is not False
        or trace["contains_private_identifiers"] is not False
        or trace["public_trace_sha256"] != content_sha256(payload)
    ):
        raise RepresentationTraceError("public representation trace binding differs")
    _sha(trace["public_trace_sha256"], "public representation trace")
    _nonempty(trace["successor_id"], "public successor id")
    if trace["trace_kind"] == "DECLARED_REPRESENTATION":
        _keys(
            trace["artifact_contracts"],
            {"model_binding_sha256", "embedding_cache_descriptor_sha256"},
            "public declared artifact contracts",
        )
        for name, digest in trace["artifact_contracts"].items():
            _sha(digest, name)
        expected_execution = "DECLARED_ONLY"
    else:
        _executed_bindings(trace["artifact_contracts"])
        expected_execution = "ACTUAL_EXECUTION_WITH_EXACT_PRIVATE_CACHE_BINDINGS"
    if trace["execution_evidence"] != expected_execution or trace[
        "pair_evidence"
    ] not in {
        "AVAILABLE_IN_PRIVATE_TRACE",
        "NOT_AVAILABLE_IN_PRIVATE_TRACE",
    }:
        raise RepresentationTraceError("public trace evidence availability differs")
    if (
        trace["trace_kind"] == "EXECUTED_REPRESENTATION"
        and trace["pair_evidence"] != "AVAILABLE_IN_PRIVATE_TRACE"
    ):
        raise RepresentationTraceError(
            "executed public trace pair availability differs"
        )
    _validate_public_model_contract(trace["model_contract"])
    map_names = _validate_public_map_contracts(
        trace["available_map_contracts"], trace_kind=trace["trace_kind"]
    )
    unavailable_names = _validate_public_unavailable_evidence(
        trace["unavailable_evidence"]
    )
    if trace["trace_kind"] == "EXECUTED_REPRESENTATION":
        required_maps = {
            "query_mask_occupancy",
            "key_mask_occupancy",
            "query_pooling_weight",
            "key_pooling_weight",
            "query_pair_contribution",
            "key_pair_contribution",
            "pair_patch_contribution",
        }
        scorer_maps = {
            "query_spatial_scorer_logit",
            "key_spatial_scorer_logit",
        }
        if (
            trace["successor_id"] not in {"B3", "B5-SPATIAL"}
            or not required_maps <= map_names
            or (trace["successor_id"] == "B5-SPATIAL")
            != scorer_maps.issubset(map_names)
            or not {
                "transformer_attention_query_key",
                "pair_patch_correspondence",
            }
            <= unavailable_names
            or (trace["successor_id"] == "B3")
            != ("spatial_scorer_logits" in unavailable_names)
        ):
            raise RepresentationTraceError(
                "public executed evidence availability differs"
            )
    return trace


def build_public_representation_analysis(
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a public-safe collection without claiming population aggregation."""

    validated = [
        validate_public_representation_trace_manifest(trace) for trace in traces
    ]
    validated.sort(key=lambda item: (item["successor_id"], item["public_trace_sha256"]))
    digests = [item["public_trace_sha256"] for item in validated]
    if len(set(digests)) != len(digests):
        raise RepresentationTraceError("representation analysis repeats a public trace")
    payload = {
        "schema_version": PUBLIC_ANALYSIS_SCHEMA,
        "visibility": "PUBLIC_SAFE_TRACE_SUMMARY",
        "trace_count": len(validated),
        "traces": validated,
        "aggregation": "NONE_SELECTED_TRACE_AVAILABILITY_ONLY",
        "contains_raw_tensor_values": False,
        "contains_private_identifiers": False,
    }
    return validate_public_representation_analysis(
        {**payload, "analysis_sha256": content_sha256(payload)}
    )


def validate_public_representation_analysis(value: object) -> dict[str, Any]:
    """Validate a public-safe trace collection and its content binding."""

    expected = {
        "schema_version",
        "visibility",
        "trace_count",
        "traces",
        "aggregation",
        "contains_raw_tensor_values",
        "contains_private_identifiers",
        "analysis_sha256",
    }
    _keys(value, expected, "public representation analysis")
    analysis = dict(value)
    payload = {key: item for key, item in analysis.items() if key != "analysis_sha256"}
    if (
        analysis["schema_version"] != PUBLIC_ANALYSIS_SCHEMA
        or analysis["visibility"] != "PUBLIC_SAFE_TRACE_SUMMARY"
        or analysis["aggregation"] != "NONE_SELECTED_TRACE_AVAILABILITY_ONLY"
        or analysis["contains_raw_tensor_values"] is not False
        or analysis["contains_private_identifiers"] is not False
        or analysis["analysis_sha256"] != content_sha256(payload)
        or not isinstance(analysis["traces"], list)
        or isinstance(analysis["trace_count"], bool)
        or not isinstance(analysis["trace_count"], int)
        or analysis["trace_count"] != len(analysis["traces"])
    ):
        raise RepresentationTraceError("public representation analysis binding differs")
    traces = [
        validate_public_representation_trace_manifest(trace)
        for trace in analysis["traces"]
    ]
    expected_order = sorted(
        traces, key=lambda item: (item["successor_id"], item["public_trace_sha256"])
    )
    if traces != expected_order or len(
        {item["public_trace_sha256"] for item in traces}
    ) != len(traces):
        raise RepresentationTraceError(
            "public representation analysis trace set differs"
        )
    return analysis


def _public_model_contract(
    *,
    transform: Mapping[str, Any],
    layers: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_input_transform": transform,
        "layers": [
            {"name": layer["name"], "shape": ["BATCH", *layer["shape"][1:]]}
            for layer in layers
        ],
        "patch_geometry": geometry,
        "embedding": {
            "dimension": EMBEDDING_DIMENSION,
            "dtype": "float32",
            "normalization": "L2",
        },
    }


def _validate_public_model_contract(value: object) -> None:
    _keys(
        value,
        {"model_input_transform", "layers", "patch_geometry", "embedding"},
        "public model contract",
    )
    _validate_transform(value["model_input_transform"])
    _validate_patch_geometry(value["patch_geometry"])
    layers = value["layers"]
    if not isinstance(layers, list) or not layers or len(layers) > _MAX_LAYERS:
        raise RepresentationTraceError("public model layers differ")
    names: set[str] = set()
    for layer in layers:
        _keys(layer, {"name", "shape"}, "public model layer")
        _nonempty(layer["name"], "public model layer name")
        shape = layer["shape"]
        if (
            layer["name"] in names
            or not isinstance(shape, list)
            or not shape
            or shape[0] != "BATCH"
            or len(shape) > 8
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in shape[1:]
            )
        ):
            raise RepresentationTraceError("public model layer contract differs")
        names.add(layer["name"])
    _keys(
        value["embedding"],
        {"dimension", "dtype", "normalization"},
        "public embedding contract",
    )
    if value["embedding"] != {
        "dimension": EMBEDDING_DIMENSION,
        "dtype": "float32",
        "normalization": "L2",
    }:
        raise RepresentationTraceError("public embedding contract differs")


def _validate_public_map_contracts(value: object, *, trace_kind: str) -> set[str]:
    if not isinstance(value, list):
        raise RepresentationTraceError("public map contracts must be an array")
    names: list[str] = []
    allowed = (
        _SPATIAL_MAP_NAMES
        if trace_kind == "DECLARED_REPRESENTATION"
        else _EXECUTED_MAP_NAMES
    )
    for record in value:
        _keys(record, {"name", "semantic", "dtype", "shape"}, "public map contract")
        name = record["name"]
        _nonempty(name, "public map name")
        _nonempty(record["semantic"], "public map semantic")
        expected_semantic = _EXECUTED_MAP_SEMANTICS.get(name, "DECLARED_SPATIAL_MAP")
        shape = record["shape"]
        if (
            name not in allowed
            or record["semantic"] != expected_semantic
            or record["dtype"] != "float32"
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in shape
            )
        ):
            raise RepresentationTraceError("public map contract differs")
        names.append(name)
    if names != sorted(set(names)):
        raise RepresentationTraceError("public map contracts must be unique and sorted")
    return set(names)


def _validate_public_unavailable_evidence(value: object) -> set[str]:
    if not isinstance(value, list):
        raise RepresentationTraceError("public unavailable evidence must be an array")
    expected = []
    for record in value:
        _keys(record, {"name", "reason"}, "public unavailable evidence")
        name = record["name"]
        if name not in _PUBLIC_UNAVAILABLE_REASONS:
            raise RepresentationTraceError("public unavailable evidence name differs")
        expected.append({"name": name, "reason": _PUBLIC_UNAVAILABLE_REASONS[name]})
    if value != sorted(expected, key=lambda item: item["name"]):
        raise RepresentationTraceError("public unavailable evidence differs")
    return {record["name"] for record in value}


def _validate_transform(value: object) -> dict[str, Any]:
    expected = {
        "source_size",
        "model_input_size",
        "color_mode",
        "resize_interpolation",
        "mask_application",
        "channel_mean",
        "channel_std",
    }
    _keys(value, expected, "model input transform")
    transform = dict(value)
    for field in ("source_size", "model_input_size"):
        size = transform[field]
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in size
            )
        ):
            raise RepresentationTraceError(
                f"{field} must contain two positive integers"
            )
    for field in ("color_mode", "resize_interpolation", "mask_application"):
        _nonempty(transform[field], field)
    if transform["color_mode"] != "RGB":
        raise RepresentationTraceError("representation input color mode must be RGB")
    for field in ("channel_mean", "channel_std"):
        values = transform[field]
        if not isinstance(values, list) or len(values) != 3:
            raise RepresentationTraceError(f"{field} must contain three values")
        transform[field] = [_finite(item, field) for item in values]
    if any(item <= 0.0 for item in transform["channel_std"]):
        raise RepresentationTraceError("channel standard deviations must be positive")
    return transform


def _pair_tensors(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    subject: str,
    dimension: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = (first, second)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise RepresentationTraceError(f"{subject} must be torch tensors")
    expected_ndim = 3 if dimension is not None else 2
    if any(
        value.ndim != expected_ndim
        or not torch.isfinite(value).all()
        or (dimension is not None and value.shape[-1] != dimension)
        for value in values
    ):
        raise RepresentationTraceError(f"{subject} shape or values differ")
    if first.device != second.device:
        raise RepresentationTraceError(f"{subject} pair device differs")
    return values


_EXECUTED_BINDING_FIELDS = {
    "run_manifest_sha256",
    "candidate_run_sha256",
    "model_manifest_sha256",
    "checkpoint_manifest_sha256",
    "checkpoint_state_sha256",
    "preprocessing_manifest_sha256",
    "embedding_manifest_sha256",
    "token_cache_manifest_sha256",
    "token_cache_tokens_sha256",
    "token_cache_occupancy_sha256",
    "evaluation_cache_descriptor_sha256",
    "evaluation_pack_sha256",
    "dinov2_model_sha256",
    "dinov2_config_sha256",
    "dinov2_preprocessor_sha256",
}


def _executed_bindings(value: object) -> dict[str, str]:
    _keys(value, _EXECUTED_BINDING_FIELDS, "executed artifact bindings")
    bindings = dict(value)
    for name, digest in bindings.items():
        _sha(digest, name)
    return bindings


def _input_binding(value: object, subject: str) -> dict[str, str]:
    expected = {"rgb_sha256", "mask_sha256", "crop_record_sha256"}
    _keys(value, expected, f"{subject} input binding")
    binding = dict(value)
    for name, digest in binding.items():
        _sha(digest, f"{subject} {name}")
    return binding


def _pair_patch_contribution(
    model: Dinov2OccupancyProbe128 | SpatialScorer128,
    query: PatchRepresentationDecomposition,
    key: PatchRepresentationDecomposition,
) -> np.ndarray:
    weight = model.projection.weight.detach().double().cpu().numpy()
    bias = model.projection.bias.detach().double().cpu().numpy()

    def components(value: PatchRepresentationDecomposition) -> np.ndarray:
        tokens = value.effective_tokens[0].detach().double().cpu().numpy()
        weights = value.weights[0].detach().double().cpu().numpy()
        projected = tokens @ weight.T + bias
        weighted = projected * weights[:, None]
        norm = np.linalg.norm(weighted.sum(axis=0))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise RepresentationTraceError(
                "patch contribution projection is degenerate"
            )
        return weighted / norm

    contribution = components(query) @ components(key).T
    if contribution.shape != (256, 256) or not np.isfinite(contribution).all():
        raise RepresentationTraceError("pair patch contribution differs")
    return contribution.astype(np.float32)


def _slice_decomposition(
    value: PatchRepresentationDecomposition, index: int
) -> PatchRepresentationDecomposition:
    return PatchRepresentationDecomposition(
        effective_tokens=value.effective_tokens[index : index + 1],
        occupancy=value.occupancy[index : index + 1],
        logits=None if value.logits is None else value.logits[index : index + 1],
        weights=value.weights[index : index + 1],
        pooled=value.pooled[index : index + 1],
        projected=value.projected[index : index + 1],
        embedding=value.embedding[index : index + 1],
    )


def _executed_layer_shapes(
    value: PatchRepresentationDecomposition,
) -> list[dict[str, Any]]:
    tensors: list[tuple[str, torch.Tensor]] = [
        ("dinov2_patch_tokens", value.effective_tokens),
        ("mask_patch_occupancy", value.occupancy),
        ("normalized_patch_weights", value.weights),
        ("pooled_patch_representation", value.pooled),
        ("projection_pre_normalization", value.projected),
        ("l2_embedding", value.embedding),
    ]
    if value.logits is not None:
        tensors.insert(2, ("spatial_scorer_logits", value.logits))
    return [{"name": name, "shape": list(tensor.shape)} for name, tensor in tensors]


def _executed_map_record(array: np.ndarray, *, semantic: str) -> dict[str, Any]:
    _nonempty(semantic, "executed map semantic")
    canonical = np.asarray(array, dtype="<f4")
    if (
        canonical.ndim != 2
        or canonical.size > _MAX_GRID_CELLS
        or not np.isfinite(canonical).all()
    ):
        raise RepresentationTraceError("executed map shape or values differ")
    return {
        "semantic": semantic,
        "dtype": "float32",
        "shape": list(canonical.shape),
        "values": canonical.tolist(),
        "values_sha256": _array_sha256(canonical),
        "minimum": float(np.min(canonical)),
        "maximum": float(np.max(canonical)),
        "mean": float(np.mean(canonical, dtype=np.float64)),
        "sum": float(np.sum(canonical, dtype=np.float64)),
    }


def _validate_executed_map_record(value: object, name: str) -> np.ndarray:
    expected = {
        "semantic",
        "dtype",
        "shape",
        "values",
        "values_sha256",
        "minimum",
        "maximum",
        "mean",
        "sum",
    }
    _keys(value, expected, "executed map")
    _nonempty(value["semantic"], "executed map semantic")
    if value["dtype"] != "float32":
        raise RepresentationTraceError("executed map dtype differs")
    try:
        array = np.asarray(value["values"], dtype="<f4")
    except (TypeError, ValueError) as exc:
        raise RepresentationTraceError("executed map values must be numeric") from exc
    expected_shape = (256, 256) if name == "pair_patch_contribution" else (16, 16)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise RepresentationTraceError("executed map shape or values differ")
    summaries = {
        "shape": list(array.shape),
        "values_sha256": _array_sha256(array),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array, dtype=np.float64)),
        "sum": float(np.sum(array, dtype=np.float64)),
    }
    if any(value[field] != observed for field, observed in summaries.items()):
        raise RepresentationTraceError("executed map binding differs")
    return array


def _validate_execution_verification(value: object) -> None:
    expected = {
        "device",
        "dtype",
        "live_tokens_exact_cache_match",
        "live_occupancy_exact_cache_match",
        "query_embedding_exact_cache_match",
        "key_embedding_exact_cache_match",
    }
    _keys(value, expected, "executed verification")
    if (
        value["device"] not in {"cpu", "cuda"}
        or value["dtype"] != "float32"
        or any(value[name] is not True for name in expected - {"device", "dtype"})
    ):
        raise RepresentationTraceError("executed verification must prove exact matches")


def _validate_executed_embedding_bindings(value: object) -> None:
    expected = {
        "query_vector_sha256",
        "key_vector_sha256",
        "dimension",
        "dtype",
        "normalization",
    }
    _keys(value, expected, "executed embedding bindings")
    _sha(value["query_vector_sha256"], "query vector")
    _sha(value["key_vector_sha256"], "key vector")
    if (
        value["dimension"] != EMBEDDING_DIMENSION
        or value["dtype"] != "float32"
        or value["normalization"] != "L2"
    ):
        raise RepresentationTraceError("executed embedding contract differs")


def _validate_executed_pair(value: object) -> None:
    expected = {
        "score",
        "rank",
        "exact_cosine",
        "algorithm",
        "pair_patch_contribution_sum",
        "pair_patch_contribution_roundoff",
    }
    _keys(value, expected, "executed pair")
    score = _finite(value["score"], "executed pair score")
    contribution_sum = _finite(
        value["pair_patch_contribution_sum"], "pair contribution sum"
    )
    roundoff = _finite(value["pair_patch_contribution_roundoff"], "pair roundoff")
    if (
        not -1.000001 <= score <= 1.000001
        or isinstance(value["rank"], bool)
        or not isinstance(value["rank"], int)
        or value["rank"] <= 0
        or value["exact_cosine"] is not True
        or value["algorithm"] != "EXACT_FLOAT32_DOT_OF_L2_CACHE_VECTORS"
        or roundoff != contribution_sum - score
        or abs(roundoff) > 1e-5
    ):
        raise RepresentationTraceError("executed pair cosine evidence differs")


def _validate_layers(value: object) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or len(value) > _MAX_LAYERS
    ):
        raise RepresentationTraceError("trace layers must be a bounded non-empty array")
    layers: list[dict[str, Any]] = []
    names: set[str] = set()
    for row in value:
        _keys(row, {"name", "shape"}, "trace layer")
        name = row["name"]
        _nonempty(name, "trace layer name")
        shape = row["shape"]
        if (
            name in names
            or not isinstance(shape, list)
            or not shape
            or len(shape) > 8
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in shape
            )
        ):
            raise RepresentationTraceError("trace layer name or shape differs")
        names.add(name)
        layers.append({"name": name, "shape": list(shape)})
    return layers


def _validate_patch_geometry(value: object) -> dict[str, int]:
    expected = {
        "input_height",
        "input_width",
        "patch_height",
        "patch_width",
        "grid_height",
        "grid_width",
    }
    _keys(value, expected, "patch geometry")
    geometry = dict(value)
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in geometry.values()
    ):
        raise RepresentationTraceError(
            "patch geometry values must be positive integers"
        )
    if (
        geometry["input_height"] != geometry["patch_height"] * geometry["grid_height"]
        or geometry["input_width"] != geometry["patch_width"] * geometry["grid_width"]
        or geometry["grid_height"] * geometry["grid_width"] > _MAX_GRID_CELLS
    ):
        raise RepresentationTraceError("patch geometry grid differs from model input")
    return geometry


def _embedding(value: object) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != (EMBEDDING_DIMENSION,)
    ):
        raise RepresentationTraceError("trace embedding must be exact 128D float32")
    if not np.isfinite(value).all() or not np.isclose(
        np.linalg.norm(value), 1.0, rtol=0.0, atol=1e-5
    ):
        raise RepresentationTraceError(
            "trace embedding must be finite and L2-normalized"
        )
    return value


def _spatial_map(value: object, geometry: Mapping[str, int], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RepresentationTraceError(f"{name} spatial map must be numeric") from exc
    expected = (geometry["grid_height"], geometry["grid_width"])
    if array.shape != expected or not np.isfinite(array).all():
        raise RepresentationTraceError(f"{name} spatial map shape or values differ")
    if name == "mask_occupancy" and np.any((array < 0.0) | (array > 1.0)):
        raise RepresentationTraceError("mask occupancy values must be in [0, 1]")
    return array


def _private_map_record(array: np.ndarray) -> dict[str, Any]:
    canonical = np.asarray(array, dtype="<f4")
    return {
        "dtype": "float32",
        "shape": list(canonical.shape),
        "values": canonical.tolist(),
        "values_sha256": _array_sha256(canonical),
        "minimum": float(np.min(canonical)),
        "maximum": float(np.max(canonical)),
        "mean": float(np.mean(canonical, dtype=np.float64)),
    }


def _validate_private_map_record(
    value: object, geometry: Mapping[str, int], name: str
) -> np.ndarray:
    _keys(
        value,
        {
            "dtype",
            "shape",
            "values",
            "values_sha256",
            "minimum",
            "maximum",
            "mean",
        },
        "private spatial map",
    )
    if value["dtype"] != "float32":
        raise RepresentationTraceError("private spatial map dtype differs")
    array = _spatial_map(value["values"], geometry, name)
    if value["shape"] != list(array.shape) or value["values_sha256"] != _array_sha256(
        array
    ):
        raise RepresentationTraceError("private spatial map binding differs")
    expected = (
        float(np.min(array)),
        float(np.max(array)),
        float(np.mean(array, dtype=np.float64)),
    )
    observed = (value["minimum"], value["maximum"], value["mean"])
    if observed != expected:
        raise RepresentationTraceError("private spatial map summary differs")
    return array


def _occupancy_summary(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "foreground_fraction": float(np.mean(array, dtype=np.float64)),
        "fully_foreground_patch_count": int(np.count_nonzero(array == 1.0)),
        "partially_foreground_patch_count": int(
            np.count_nonzero((array > 0.0) & (array < 1.0))
        ),
        "empty_patch_count": int(np.count_nonzero(array == 0.0)),
        "values_sha256": _array_sha256(array),
    }


def _validate_pair(value: object) -> dict[str, Any]:
    expected = {
        "query_sample_token",
        "key_sample_token",
        "winning_template_id",
        "score",
        "rank",
        "exact_cosine",
    }
    _keys(value, expected, "representation pair trace")
    pair = dict(value)
    for field in ("query_sample_token", "key_sample_token", "winning_template_id"):
        _nonempty(pair[field], field)
    pair["score"] = _finite(pair["score"], "pair score")
    if not -1.000001 <= pair["score"] <= 1.000001:
        raise RepresentationTraceError("pair cosine score is outside [-1, 1]")
    if (
        isinstance(pair["rank"], bool)
        or not isinstance(pair["rank"], int)
        or pair["rank"] <= 0
    ):
        raise RepresentationTraceError("pair rank must be positive")
    if pair["exact_cosine"] is not True:
        raise RepresentationTraceError("representation pair score must be exact cosine")
    return pair


def _array_sha256(value: np.ndarray) -> str:
    return (
        __import__("hashlib")
        .sha256(np.asarray(value, dtype="<f4").tobytes(order="C"))
        .hexdigest()
    )


def _keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RepresentationTraceError(f"{label} fields differ")


def _sha(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RepresentationTraceError(f"{label} must be lowercase SHA-256")


def _nonempty(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RepresentationTraceError(f"{label} must be non-empty text")


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RepresentationTraceError(f"{label} must be finite")
    return float(value)


__all__ = [
    "EXECUTED_PUBLIC_TRACE_SCHEMA",
    "EXECUTED_TRACE_SCHEMA",
    "PUBLIC_ANALYSIS_SCHEMA",
    "PUBLIC_TRACE_SCHEMA",
    "TRACE_SCHEMA",
    "RepresentationTraceError",
    "build_executed_representation_trace_manifest",
    "build_public_representation_analysis",
    "build_representation_trace_manifest",
    "sanitize_representation_trace_manifest",
    "validate_executed_representation_trace_manifest",
    "validate_public_representation_analysis",
    "validate_public_representation_trace_manifest",
    "validate_representation_trace_manifest",
]
